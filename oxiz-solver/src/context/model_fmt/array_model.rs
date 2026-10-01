//! Rendering an array constant's model value and a declared function's
//! interpretation for `(get-model)` (`#P2b-34`).
//!
//! Before this module an array constant printed the *sort default* and nothing
//! else — `arr = ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00)` — even
//! when the same model answered `(get-value ((select arr j)))` with `#x05`.
//! The two statements contradict each other: read the printed array at `j` and
//! it is `#x00`.  A declared function fared worse still: `(get-model)` omitted
//! it altogether, so a model over `(f x)` could be neither printed nor checked.
//!
//! Both are rendered here from what the model actually holds:
//!
//! * an array becomes the `store` chain of its published reads laid over the
//!   constant default, which is the canonical SMT-LIB spelling of exactly the
//!   function those reads describe;
//! * a function becomes the nested `ite` over its congruence-closed
//!   application entries with the interpretation's else-value at the bottom,
//!   the spelling Z3 prints.
//!
//! Neither rendering invents anything: every entry comes from
//! `Solver::publish_array_reads` / `Context::get_func_interp_raw`, and an array
//! with no published read still prints the constant default it always did.

use super::*;
use oxiz_core::ast::TermManager;

/// Widest bit-vector index sort whose elements the array renderer will treat as
/// enumerable when deciding that a chain covers its whole domain.
///
/// Sixteen elements, the same order of magnitude as the array refinement's own
/// `ARRAY_INDEX_ENUMERATION_LIMIT` — past that a
/// model does not name every index in practice, and the coverage test would
/// walk a long entry list to learn nothing.
const COVERED_DOMAIN_WIDTH_LIMIT: u32 = 4;

/// How deep the array-model rendering follows an `Array` *range* that is
/// itself an array.
///
/// Every level strictly decreases the sort nesting, so the recursion
/// terminates on its own; the limit bounds the native stack for a sort a
/// script nested pathologically deep (`define-sort` grows one level per
/// command), and at the limit the renderer answers "nothing to say about this
/// array" rather than an unjustified value.
const ARRAY_MODEL_NESTING_LIMIT: u32 = 32;

impl Context {
    /// The `store`-chain rendering of `array`'s value, or `None` when the
    /// model says nothing at all about its congruence class (the caller then
    /// keeps the constant default).
    ///
    /// # Why the rendering is per congruence *class* (`#P2b-34`)
    ///
    /// An array is never *assigned* a value term — there is no literal for
    /// "the function `{0 ↦ 5}` extended by 0" — so the model records it
    /// indirectly: the default of an `(as const d)` member of its class, the
    /// index and value of a `store` member, and the reads the solver
    /// published.  Reading only the reads of the *syntactic* term rendered
    /// `arr = ((as const A) #b1)` as `(store ((as const A) #b0) #b1 #b1)`:
    /// the class's own constant said every entry is `#b1`, and the printed
    /// array said `arr[#b0] = #b0` — a model falsifying its own assertion.
    /// The class is what the model decided; the term is just one name for it.
    ///
    /// So the rendering is, in order:
    ///
    /// 1. **base** — the default of an `(as const d)` member if the class has
    ///    one, else the rendering of the base array of a `store` member with
    ///    that store's write laid on top, else the sort default;
    /// 2. **writes** — every `store` member's own `(index, value)`;
    /// 3. **reads** — every published read of any member of the class.
    ///
    /// Two arrays in different classes print differently because the
    /// extensionality witness of `#P2b-37` gives their classes a read at an
    /// index where they differ; two arrays in one class print identically
    /// because they are rendered from the same three sources.
    pub(super) fn array_model_value(
        &self,
        array: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
    ) -> Option<String> {
        // An array the model pins *explicitly* to an array value is the one
        // case where the class is not the better source: the value did not
        // come out of the congruence closure at all.  It is installed by
        // `solver::array_completion_certify`, whose certificate was discharged
        // over exactly this term — the completed total interpretation
        // `((as const A) d)`, or a `store` chain over it — so printing
        // anything else would publish a model the certificate did not verify.
        // An array-sorted `(get-value)` hands the installed value itself in
        // (`array_query_value` resolves the constant to its model entry), and
        // it prints as itself for the same reason.  Every other array keeps
        // the class-based rendering below, which is what `#P2b-37` needs.
        let installed = model.get(array).unwrap_or(array);
        if let Some(printed) = self.format_installed_array(installed) {
            return Some(printed);
        }
        self.array_class_value(array, sort, model, class_values, 0)
    }

    /// The printed form of an installed array value — the constant array or
    /// the `store` chain over one that `solver::array_completion_certify`
    /// certified — or `None` when `value` is not one.
    ///
    /// Spelled with the same leaf formatting every other model entry uses
    /// ([`Context::format_value`]): a `Real` default prints as `0.0` here as it
    /// does everywhere else in `(get-model)`, not as the shared printer's `0`.
    pub(in crate::context) fn format_installed_array(&self, value: TermId) -> Option<String> {
        if !crate::solver::array_completion_certify::is_array_value(value, &self.terms) {
            return None;
        }
        let mut levels: Vec<(TermId, TermId)> = Vec::new();
        let mut current = value;
        while let TermKind::Store(inner, index, stored) = self.terms.get(current)?.kind {
            levels.push((index, stored));
            current = inner;
        }
        let default = crate::solver::array_axioms::const_array_default(current, &self.terms)?;
        let sort = self.terms.get(current)?.sort;
        let mut chain = format!(
            "((as const {}) {})",
            self.format_sort_name(sort),
            self.format_value(default)
        );
        // Innermost store first, the order the term itself nests them in.
        for (index, stored) in levels.into_iter().rev() {
            chain = format!(
                "(store {chain} {} {})",
                self.format_value(index),
                self.format_value(stored)
            );
        }
        Some(chain)
    }

    /// [`Context::array_model_value`] at nesting depth `depth`; see
    /// [`ARRAY_MODEL_NESTING_LIMIT`].
    fn array_class_value(
        &self,
        array: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
    ) -> Option<String> {
        let mut visiting: Vec<TermId> = Vec::new();
        let (base, entries) =
            self.array_class_parts(array, sort, model, class_values, depth, &mut visiting)?;
        let mut chain = base;
        // Innermost write first, so the chain reads left to right in the same
        // order the entries were collected and no later write is shadowed.
        for (index_str, value_str) in entries {
            chain = format!("(store {chain} {index_str} {value_str})");
        }
        Some(chain)
    }

    /// The base value and the `(index, value)` entries of `array`'s
    /// congruence class, or `None` when the model says nothing about it.
    ///
    /// Kept structured rather than pre-formatted because one caller needs to
    /// *remove* an entry: an array that is only ever described from the
    /// outside, as the base of a `store` that the model does pin down, agrees
    /// with that store everywhere except at the store's own index.
    ///
    /// `visiting` breaks the cycle `a = store(a, i, v)` would otherwise make
    /// of that inheritance.
    fn array_class_parts(
        &self,
        array: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
        visiting: &mut Vec<TermId>,
    ) -> Option<(String, Vec<(String, String)>)> {
        if depth >= ARRAY_MODEL_NESTING_LIMIT || visiting.contains(&array) {
            return None;
        }
        visiting.push(array);
        let parts = self.array_class_parts_inner(array, sort, model, class_values, depth, visiting);
        visiting.pop();
        parts
    }

    /// The body of [`Context::array_class_parts`], with the cycle guard
    /// already applied.
    fn array_class_parts_inner(
        &self,
        array: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
        visiting: &mut Vec<TermId>,
    ) -> Option<(String, Vec<(String, String)>)> {
        let members = self.solver.euf_class_terms(array);

        // ---- base ------------------------------------------------------
        // The class's array constant, if it has one: its default is the value
        // at *every* index the writes and reads below do not override.
        let mut base: Option<String> = None;
        for &member in &members {
            let Some(default) =
                crate::solver::array_axioms::const_array_default(member, &self.terms)
            else {
                continue;
            };
            let sort_name = self.format_sort_name(sort);
            let default_str = self.entry_value(
                self.evaluated(default, model),
                sort,
                model,
                class_values,
                depth,
            )?;
            base = Some(format!("((as const {sort_name}) {default_str})"));
            break;
        }

        // ---- writes ----------------------------------------------------
        // Deterministic: the model map and the class membership are hash
        // tables, the printed chain is not allowed to depend on their order.
        let mut sorted_members = members.clone();
        sorted_members.sort_unstable_by_key(|member: &TermId| member.raw());

        let mut entries: Vec<(String, String)> = Vec::new();
        let mut seen: crate::prelude::HashSet<String> = crate::prelude::HashSet::new();
        // The chain of the array a `store` member of this class writes over,
        // filled in by the member loop and merged in after the reads.
        let mut background: Option<(String, Vec<(String, String)>)> = None;
        for &member in &sorted_members {
            let Some(data) = self.terms.get(member) else {
                continue;
            };
            let TermKind::Store(store_base, index, value) = data.kind else {
                continue;
            };
            // A store member also tells us the class's *background*: every
            // index it does not write reads through to its own base array.
            // The class's *background*: every index this store does not
            // write reads through to its own base array.  Kept aside rather
            // than folded into `base` here, and merged in below once every
            // entry this class names is known — see the merge for why.
            if base.is_none() && background.is_none() {
                background = self.array_class_parts(
                    store_base,
                    sort,
                    model,
                    class_values,
                    depth + 1,
                    visiting,
                );
            }
            self.push_entry(
                index,
                value,
                sort,
                model,
                class_values,
                depth,
                &mut seen,
                &mut entries,
            );
        }

        // ---- reads -----------------------------------------------------
        let bound = self.bound_only_variables();
        let mut reads: Vec<(TermId, TermId, TermId)> = Vec::new();
        for (&term, &value) in model.assignments() {
            let Some(data) = self.terms.get(term) else {
                continue;
            };
            if let TermKind::Select(read_array, index) = data.kind
                && members.contains(&read_array)
                && !self.mentions_bound_variable(term, &bound)
            {
                reads.push((term, index, value));
            }
        }
        // A read at an index that *is* a value (a literal, a constructor) goes
        // before a read at a symbol the model merely assigns that value: the
        // first entry per position wins, and a solver-internal symbol (an
        // extensionality witness) can carry a sort default that coincides
        // with a position another read names — `a[ext] = 5` beside `a[red] =
        // 1` with `ext` defaulted to `red` printed `red ↦ 5` (`#P2b-72`).
        reads.sort_unstable_by_key(|&(term, index, _)| (!self.is_value_index(index), term.raw()));
        for (_, index, value) in reads {
            self.push_entry(
                index,
                value,
                sort,
                model,
                class_values,
                depth,
                &mut seen,
                &mut entries,
            );
        }

        // An *array-sorted* read is never in `model.assignments()`: the model
        // assigns value terms, and no term denotes a whole array.  Its value
        // is the read itself, rendered from its own class one level down —
        // which is the only way the outer array of an array of arrays says
        // anything at all.  `(= row0 (select matrix 0))` with
        // `(= (select row0 0) 42)` printed `matrix` as the sort default and
        // `row0` as `{0 ↦ 42}` side by side, contradicting the equality
        // between them (`#P2b-34`, `bench/z3_parity/benchmarks/qf_a/array_07`).
        //
        // The term-graph scan is confined to this case and to an
        // uninterpreted range: an array whose range is a scalar or a datatype
        // has all its reads in the model, while a read of an uninterpreted
        // sort has no model value either — its value is its class's
        // `@uc_S_n` witness ([`Self::entry_value`]).  Without the scan an
        // array into `U` read at `x` and `y` printed as one constant beside
        // two different witnesses for the reads (`#P2b-89`).
        let uninterpreted_range = self.range_is_uninterpreted(sort);
        if self.range_is_array(sort) || uninterpreted_range {
            let mut nested: Vec<(TermId, TermId)> = Vec::new();
            for index in 0..(self.terms.len() as u32) {
                let term = TermId(index);
                let Some(data) = self.terms.get(term) else {
                    continue;
                };
                if let TermKind::Select(read_array, read_index) = data.kind
                    && members.contains(&read_array)
                    && !(uninterpreted_range && self.mentions_bound_variable(term, &bound))
                {
                    nested.push((read_index, term));
                }
            }
            nested.sort_unstable_by_key(|&(_, term)| term.raw());
            for (read_index, read_term) in nested {
                self.push_entry(
                    read_index,
                    read_term,
                    sort,
                    model,
                    class_values,
                    depth,
                    &mut seen,
                    &mut entries,
                );
            }
        }

        // ---- background of a `store` member ----------------------------
        // Merged here, not where it was read, because an entry of the
        // background is only printable once every index *this* class names is
        // known: an index this class writes or reads overrides the background
        // at that index, and an index it does not is inherited unchanged.
        //
        // Folding the background into `base` as a ready-made chain instead —
        // which is what this did when the array printer was written — publishes
        // both, and the printed array then carries the same index twice
        // (`#P2b-45`).  A duplicated link is not cosmetic: it is a *deeper
        // store chain* in the published model, and the model is re-read as
        // SMT-LIB by every consumer that checks it.  `(= arr (store brr #b01
        // #b11))` over `(Array (_ BitVec 2) (_ BitVec 2))` printed a seven-link
        // chain with three duplicated links, and re-running the script with
        // that model substituted for the declarations answered `unknown`,
        // where the four-link chain the very same model denotes answers `sat`
        // in 0.6 ms.  A model a consumer cannot check is not much better than
        // one that is wrong.
        //
        // Suppressing the *other* side instead — dropping this class's own
        // entry wherever the background already has that index — is wrong and
        // was measured to be: the store's own write is exactly the index where
        // it differs from its base, so two arrays that differ in one cell
        // printed as the same chain, and the debug interpretation check caught
        // two `f[array]` keys with disagreeing values in the in-tree
        // `array_ext_shapes_bounded` generator.
        if base.is_none()
            && let Some((inner_base, inner_entries)) = background
        {
            let mut merged: Vec<(String, String)> = Vec::new();
            for (index_str, value_str) in inner_entries {
                if !seen.insert(index_str.clone()) {
                    // This class names that index itself; its own entry stands.
                    continue;
                }
                merged.push((index_str, value_str));
            }
            merged.append(&mut entries);
            entries = merged;
            base = Some(inner_base);
        }

        // ---- inherited from the outside --------------------------------
        // Nothing in this class names a value, but the class may still be
        // pinned by an equality it does not contain: `(= (store brr k w)
        // ((as const A) #b01))` says nothing about `brr`'s class directly,
        // while saying that `brr` is `#b01` at every index other than `k`.
        // Rendering `brr` from the sort default there printed
        // `brr[#b00] = #b00` beside an assertion demanding `#b01`
        // (`#P2b-34`).  So: take the class of a `store` this array is the
        // base of, and drop that store's own index from it.
        if base.is_none() {
            base = self.inherited_parts(
                array,
                sort,
                model,
                class_values,
                depth,
                visiting,
                &mut seen,
                &mut entries,
            );
        }

        if base.is_none() && entries.is_empty() {
            return None;
        }
        let mut base = base.unwrap_or_else(|| self.default_value(sort));
        self.collapse_covered_domain(sort, &mut base, &mut entries);
        Some((base, entries))
    }

    /// Rewrite `(base, entries)` into the shortest chain denoting the same
    /// array, when the entries already name *every* index of a finite index
    /// sort.
    ///
    /// # Why the shortest chain and not the first one built (`#P2b-45`)
    ///
    /// A published model is not only read by a human: it is re-read as SMT-LIB
    /// by every consumer that checks it, `oxiz`'s own `(get-value)` and
    /// `cargo-formal`'s evidence replay included.  Once the index sort is small
    /// enough to enumerate (`#P2b-38` strand (b) made the array refinement read
    /// every such array at every element of its domain, so the model now names
    /// them all), the chain the renderer builds carries one link per element
    /// *on top of* a base no index can reach any more.  Four links over
    /// `(Array (_ BitVec 2) (_ BitVec 2))` where one `(as const …)` says the
    /// same thing.
    ///
    /// That is not cosmetic.  A `store` chain on the left of an equality is an
    /// array atom the solver's own honesty gate refuses to answer `sat` for
    /// (`Solver::array_atoms_need_theory`), so a model published as a chain can
    /// be *undecidable on replay* where the identical array published as a
    /// constant is decided in under a millisecond: `(= arr (store brr #b01
    /// #b11))` published `arr` as a four-link chain of a constantly-`#b11`
    /// array and re-running the script with that model substituted answered
    /// `unknown`; with `arr` published as `((as const …) #b11)` it answers
    /// `sat`.  A model a consumer cannot check is not much better than a wrong
    /// one.
    ///
    /// # The rewrite
    ///
    /// `entries` is deduplicated by index value
    /// ([`Context::push_entry`]), so `entries.len()` equal to the domain's
    /// cardinality *is* full coverage — no index string has to be constructed
    /// or compared to know it.  The base is then unreachable and may be chosen
    /// freely: take the value that occurs most often (first occurrence wins a
    /// tie, so the choice does not depend on hashing) and drop every entry
    /// carrying it.  The denotation is unchanged at every index, by
    /// construction.
    ///
    /// Bounded at `COVERED_DOMAIN_WIDTH_LIMIT` bits: past that the entries can
    /// only cover the domain if the model published thousands of reads, and the
    /// tally is not worth walking.
    fn collapse_covered_domain(
        &self,
        sort: SortId,
        base: &mut String,
        entries: &mut Vec<(String, String)>,
    ) {
        let Some(SortKind::Array { domain, .. }) = self.terms.sorts.get(sort).map(|s| &s.kind)
        else {
            return;
        };
        let Some(cardinality) = self.finite_domain_cardinality(*domain) else {
            return;
        };
        if entries.len() != cardinality {
            return;
        }
        let mut tally: Vec<(&str, usize)> = Vec::new();
        for (_, value) in entries.iter() {
            match tally.iter_mut().find(|(seen, _)| *seen == value.as_str()) {
                Some((_, count)) => *count += 1,
                None => tally.push((value.as_str(), 1)),
            }
        }
        let Some(&(majority, _)) = tally.iter().max_by_key(|&&(_, count)| count) else {
            return;
        };
        let majority = majority.to_string();
        *base = format!("((as const {}) {})", self.format_sort_name(sort), majority);
        entries.retain(|(_, value)| *value != majority);
    }

    /// The number of elements of `domain`, when it is a sort small enough to
    /// have been written out index by index — otherwise `None`.
    ///
    /// Deliberately narrower than "finite": an uninterpreted sort is finite in
    /// every model this solver builds, but its elements are named by minted
    /// `@uc_S_n` witnesses whose count is a property of the model rather than
    /// of the sort, so "the entries cover the domain" cannot be read off a
    /// cardinality for it.
    fn finite_domain_cardinality(&self, domain: SortId) -> Option<usize> {
        match self.terms.sorts.get(domain).map(|s| &s.kind) {
            Some(SortKind::Bool) => Some(2),
            Some(SortKind::BitVec(width)) => {
                let width = *width;
                if width == 0 || width > COVERED_DOMAIN_WIDTH_LIMIT {
                    return None;
                }
                Some(1usize << width)
            }
            _ => None,
        }
    }

    /// The base an array inherits from a `store` it is the base of; the
    /// store's entries other than its own index are appended to `entries`.
    ///
    /// See the "inherited from the outside" block of
    /// [`Context::array_class_parts_inner`] for why this exists.
    fn inherited_parts(
        &self,
        array: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
        visiting: &mut Vec<TermId>,
        seen: &mut crate::prelude::HashSet<String>,
        entries: &mut Vec<(String, String)>,
    ) -> Option<String> {
        for index in 0..(self.terms.len() as u32) {
            let term = TermId(index);
            let Some(data) = self.terms.get(term) else {
                continue;
            };
            let TermKind::Store(store_base, store_index, _) = data.kind else {
                continue;
            };
            if store_base != array {
                continue;
            }
            let Some((outer_base, outer_entries)) =
                self.array_class_parts(term, sort, model, class_values, depth + 1, visiting)
            else {
                continue;
            };
            // The one index the store overwrote says nothing about the base.
            // Spelled the way every entry's index is spelled, so a constructor
            // or an uninterpreted-sort index is excluded too (`#P2b-72`).
            let written = model.get(store_index).unwrap_or(store_index);
            let written_str = self
                .index_position_string(written, class_values)
                .unwrap_or_default();
            for (index_str, value_str) in outer_entries {
                if index_str == written_str || !seen.insert(index_str.clone()) {
                    continue;
                }
                entries.push((index_str, value_str));
            }
            return Some(outer_base);
        }
        None
    }

    /// Add one `(index, value)` entry to the chain being built, keyed by the
    /// *evaluated* index so two syntactically different indices the model maps
    /// to one value become one entry.
    ///
    /// # Why the chain is deduplicated by index value
    ///
    /// A `store` chain is read outermost-first, so two writes at the same
    /// index would let the outer one shadow the inner — and the shadowed entry
    /// is one the model published, i.e. the printed array would contradict its
    /// own `(get-value ((select arr i)))` answer.  That is the very defect
    /// this rendering exists to remove, so the first entry in collection order
    /// wins.  Two model-equal indices carrying *different* values would be a
    /// genuine inconsistency in the model rather than a rendering choice; it
    /// is the model gate's job to refuse such a candidate.
    ///
    /// An index or value the model leaves symbolic is dropped: an index names
    /// no position in the array, and a value would print the `?` placeholder,
    /// which is not SMT-LIB at all.
    fn push_entry(
        &self,
        index: TermId,
        value: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
        seen: &mut crate::prelude::HashSet<String>,
        entries: &mut Vec<(String, String)>,
    ) {
        let index_value = model.get(index).unwrap_or(index);
        let Some(index_str) = self.index_position_string(index_value, class_values) else {
            return;
        };
        if seen.contains(&index_str) {
            return;
        }
        let value_value = model.get(value).unwrap_or(value);
        let Some(value_str) = self.entry_value(value_value, sort, model, class_values, depth)
        else {
            return;
        };
        seen.insert(index_str.clone());
        entries.push((index_str, value_str));
    }

    /// The value this model gives `select(array, index_value)`, where
    /// `index_value` is the *evaluated* index — the read-side twin of
    /// [`Context::array_class_value`] (`#P2b-35`).
    ///
    /// # Why this exists beside the renderer
    ///
    /// `(get-model)` prints an array as the `store` chain of its class, and
    /// `(get-value ((select a #b0)))` used to answer `(select a #b0)` — an
    /// echo, because no theory ever gave that read a value and the structural
    /// evaluator has no store to reduce.  Two commands then describe two
    /// different models: the printed chain says `a[#b0] = #b1` and the query
    /// says nothing at all.
    ///
    /// So this reads the *same three sources in the same order* the renderer
    /// lays down — the class's `store` members' own writes, then every
    /// published read of any member, then the class's background (an
    /// `(as const d)` member, a `store` member's base, or a `store` this array
    /// is the base of) — and any change to one has to be made in the other.
    /// The paired test `model_one_reading.rs` pins the agreement.
    pub(super) fn array_class_read(
        &self,
        array: TermId,
        index_value: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
        visiting: &mut Vec<TermId>,
    ) -> Option<String> {
        if depth >= ARRAY_MODEL_NESTING_LIMIT || visiting.contains(&array) {
            return None;
        }
        visiting.push(array);
        let value = self.array_class_read_inner(
            array,
            index_value,
            sort,
            model,
            class_values,
            depth,
            visiting,
            false,
        );
        visiting.pop();
        value
    }

    /// [`Context::array_class_read`] restricted to the class's *background* —
    /// the value every index the class's writes and reads do not name takes.
    ///
    /// The renderer needs the same distinction: an array pinned only from the
    /// outside, as the base of a `store` the model does pin, inherits that
    /// store's *base* and not its entries, because the one index the store
    /// writes is the one it says nothing about
    /// ([`Context::inherited_parts`]).  Reading the store's class as a whole
    /// there would answer with the store's own written value, which is the
    /// one value the base is free of.
    pub(super) fn array_class_background(
        &self,
        array: TermId,
        index_value: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
        visiting: &mut Vec<TermId>,
    ) -> Option<String> {
        if depth >= ARRAY_MODEL_NESTING_LIMIT || visiting.contains(&array) {
            return None;
        }
        visiting.push(array);
        let value = self.array_class_read_inner(
            array,
            index_value,
            sort,
            model,
            class_values,
            depth,
            visiting,
            true,
        );
        visiting.pop();
        value
    }

    /// The body of [`Context::array_class_read`], with the cycle guard already
    /// applied.
    fn array_class_read_inner(
        &self,
        array: TermId,
        index_value: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
        visiting: &mut Vec<TermId>,
        background_only: bool,
    ) -> Option<String> {
        let members = self.solver.euf_class_terms(array);
        let mut sorted_members = members.clone();
        sorted_members.sort_unstable_by_key(|member: &TermId| member.raw());

        // ---- writes ----------------------------------------------------
        for &member in &sorted_members {
            if background_only {
                break;
            }
            let Some(data) = self.terms.get(member) else {
                continue;
            };
            let TermKind::Store(_, index, value) = data.kind else {
                continue;
            };
            if model.get(index).unwrap_or(index) != index_value {
                continue;
            }
            return self.entry_value(
                model.get(value).unwrap_or(value),
                sort,
                model,
                class_values,
                depth,
            );
        }

        // ---- reads -----------------------------------------------------
        let bound = self.bound_only_variables();
        let mut reads: Vec<(TermId, TermId)> = Vec::new();
        for (&term, &value) in model.assignments() {
            if background_only {
                break;
            }
            let Some(data) = self.terms.get(term) else {
                continue;
            };
            if let TermKind::Select(read_array, index) = data.kind
                && members.contains(&read_array)
                && model.get(index).unwrap_or(index) == index_value
                && !self.mentions_bound_variable(term, &bound)
            {
                reads.push((term, value));
            }
        }
        // The same order the renderer lays its entries down in.
        reads.sort_unstable_by_key(|&(term, _)| {
            let literal_index = matches!(
                self.terms.get(term).map(|t| &t.kind),
                Some(TermKind::Select(_, index)) if self.is_value_index(*index)
            );
            (!literal_index, term.raw())
        });
        if let Some(&(_, value)) = reads.first() {
            return self.entry_value(value, sort, model, class_values, depth);
        }
        // An *array-sorted* read is never in `model.assignments()` — no term
        // denotes a whole array — so the read itself is the value, rendered
        // from its own class one level down (the array-of-arrays case).
        // A read of an uninterpreted sort has no model value either: the same
        // scan, as the renderer makes it (`#P2b-89`).
        let uninterpreted_range = self.range_is_uninterpreted(sort);
        if (self.range_is_array(sort) || uninterpreted_range) && !background_only {
            let mut nested: Vec<TermId> = Vec::new();
            for index in 0..(self.terms.len() as u32) {
                let term = TermId(index);
                let Some(data) = self.terms.get(term) else {
                    continue;
                };
                if let TermKind::Select(read_array, read_index) = data.kind
                    && members.contains(&read_array)
                    && model.get(read_index).unwrap_or(read_index) == index_value
                    && !(uninterpreted_range && self.mentions_bound_variable(term, &bound))
                {
                    nested.push(term);
                }
            }
            nested.sort_unstable();
            if let Some(&read) = nested.first() {
                return self.entry_value(read, sort, model, class_values, depth);
            }
        }

        // ---- background ------------------------------------------------
        for &member in &members {
            if let Some(default) =
                crate::solver::array_axioms::const_array_default(member, &self.terms)
            {
                return self.entry_value(
                    self.evaluated(default, model),
                    sort,
                    model,
                    class_values,
                    depth,
                );
            }
        }
        for &member in &sorted_members {
            let Some(data) = self.terms.get(member) else {
                continue;
            };
            let TermKind::Store(store_base, _, _) = data.kind else {
                continue;
            };
            return self.array_class_read(
                store_base,
                index_value,
                sort,
                model,
                class_values,
                depth + 1,
                visiting,
            );
        }
        // Nothing in this class names a value, but a `store` this array is the
        // base of pins every index that store does not write.
        for index in 0..(self.terms.len() as u32) {
            let term = TermId(index);
            let Some(data) = self.terms.get(term) else {
                continue;
            };
            let TermKind::Store(store_base, store_index, _) = data.kind else {
                continue;
            };
            if store_base != array {
                continue;
            }
            // At the store's *own* index the store says nothing about this
            // array, so only its class's background applies there — exactly
            // the entry [`Context::inherited_parts`] drops when it renders
            // this array.  Anywhere else the store's class is this array's.
            let value = if model.get(store_index).unwrap_or(store_index) == index_value {
                self.array_class_background(
                    term,
                    index_value,
                    sort,
                    model,
                    class_values,
                    depth + 1,
                    visiting,
                )
            } else {
                self.array_class_read(
                    term,
                    index_value,
                    sort,
                    model,
                    class_values,
                    depth + 1,
                    visiting,
                )
            };
            if let Some(value) = value {
                return Some(value);
            }
        }
        None
    }

    /// Whether `array_sort`'s range is an uninterpreted sort.
    fn range_is_uninterpreted(&self, array_sort: SortId) -> bool {
        let Some(SortKind::Array { range, .. }) = self.terms.sorts.get(array_sort).map(|s| &s.kind)
        else {
            return false;
        };
        self.is_uninterpreted_sort(*range)
    }

    /// Whether `array_sort`'s range is itself an array sort.
    fn range_is_array(&self, array_sort: SortId) -> bool {
        let Some(SortKind::Array { range, .. }) = self.terms.sorts.get(array_sort).map(|s| &s.kind)
        else {
            return false;
        };
        self.terms
            .sorts
            .get(*range)
            .is_some_and(|s| matches!(s.kind, SortKind::Array { .. }))
    }

    /// How a `store` chain spells the *position* `index_value`, or `None` when
    /// this model cannot name that position at all.
    ///
    /// A ground value spells itself.  A term of an **uninterpreted** sort never
    /// will — that sort has no literals — but the model still names its
    /// elements: `build_class_values` mints one `@uc_S_n` witness per
    /// congruence class, and every other renderer in this module prints those
    /// same witnesses for the same classes.  Reading the position out of that
    /// map is what makes an array over an uninterpreted index sort printable.
    ///
    /// # The falsifying model this closes (`#P2b-43`)
    ///
    /// ```smt2
    /// (declare-sort U 0)
    /// (declare-const a (Array U (_ BitVec 1)))
    /// (declare-const b (Array U (_ BitVec 1)))
    /// (assert (distinct a b))
    /// ```
    ///
    /// answers `sat` — correctly: the extensionality witness of `#P2b-37` is an
    /// index of sort `U` at which the two arrays are forced to read
    /// differently, and the model holds exactly that.  But the *printer*
    /// dropped every entry whose index was not a ground value, so both arrays
    /// collapsed to `((as const (Array U (_ BitVec 1))) #b0)` and the printed
    /// model falsified the one assertion it claimed to satisfy.  Fifty of fifty
    /// scripts in that family did the same, on this tree and on the 0.3.4 base
    /// alike.
    ///
    /// A position that is neither a ground value nor a class the map names is
    /// still dropped: naming no position is better than naming the wrong one,
    /// and `?` is not SMT-LIB.
    fn index_position_string(
        &self,
        index_value: TermId,
        class_values: &super::class_values::ClassValues,
    ) -> Option<String> {
        if is_ground_value(index_value, &self.terms)
            || is_constructor_value(index_value, &self.terms)
        {
            return Some(self.format_value(index_value));
        }
        let sort = self.terms.get(index_value)?.sort;
        if !self.is_uninterpreted_sort(sort) {
            return None;
        }
        class_values
            .get(self.class_key(index_value))
            .map(ToString::to_string)
    }

    /// `term`'s value in `model`, or `term` itself when the model says nothing
    /// about it.
    ///
    /// # Why an array constant's default has to go through this (`#P2b-42`)
    ///
    /// `((as const (Array D R)) d)` takes an arbitrary *term* as its default,
    /// not a literal, and a script can perfectly well write a variable there:
    ///
    /// ```smt2
    /// (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))
    /// (declare-const v (_ BitVec 1))
    /// (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) v)))
    /// (assert (= v #b1))
    /// ```
    ///
    /// Rendering the default term as written asks [`Self::entry_value`] to
    /// print `v`, which is not a ground value, so it answered `None`, the base
    /// fell back to the *sort* default, and the model printed
    /// `a = ((as const …) #b0)` beside `v = #b1` — a model that falsifies the
    /// very assertion it claims to satisfy. The default has to be read in the
    /// model, exactly as the stored value of a `store` already is.
    fn evaluated(&self, term: TermId, model: &crate::solver::Model) -> TermId {
        model.get(term).unwrap_or(term)
    }

    /// The printed form of one array entry's value: a literal for an ordinary
    /// range sort, and the recursive class rendering when the range is itself
    /// an array.
    ///
    /// `None` when the range is an array the model says nothing about, or when
    /// the nesting limit is reached — the caller then drops the entry rather
    /// than printing a value it cannot justify.
    fn entry_value(
        &self,
        value: TermId,
        array_sort: SortId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        depth: u32,
    ) -> Option<String> {
        let range = match self.terms.sorts.get(array_sort).map(|s| &s.kind) {
            Some(SortKind::Array { range, .. }) => *range,
            _ => return Some(self.format_value(value)),
        };
        let range_is_array = self
            .terms
            .sorts
            .get(range)
            .is_some_and(|s| matches!(s.kind, SortKind::Array { .. }));
        if !range_is_array {
            if is_ground_value(value, &self.terms) || is_constructor_value(value, &self.terms) {
                return Some(self.format_value(value));
            }
            // An uninterpreted element has no literal; the model names it by
            // the `@uc_S_n` witness `build_class_values` minted for its class,
            // which every other renderer prints for that class too (`#P2b-89`).
            if self.is_uninterpreted_sort(range) {
                return class_values
                    .get(self.class_key(value))
                    .map(ToString::to_string);
            }
            // A value the model left symbolic has no printable form: the `?`
            // placeholder `format_value` falls back to is not SMT-LIB, and a
            // store chain carrying one cannot be read back at all.
            return None;
        }
        // An array-sorted entry is an array in its own right: render its class
        // the same way, one level down.
        self.array_class_value(value, range, model, class_values, depth + 1)
            .or_else(|| Some(self.default_value(range)))
    }

    /// `(define-fun f ((x!0 S0) …) Ret …)` for every declared uninterpreted
    /// function, in declaration order.
    ///
    /// The body is the nested `ite` over the interpretation's entries with the
    /// else-value at the bottom — a total function, as SMT-LIB requires a
    /// model's interpretation to be.  A function with no application in the
    /// assertions is the constant else-value, which is what
    /// [`Context::get_func_interp_raw`] reports for it.
    ///
    /// An entry whose arity disagrees with the declaration is skipped rather
    /// than rendered with the wrong number of guards; that cannot happen for a
    /// well-formed declaration, and printing a malformed `define-fun` would be
    /// worse than printing the else-value alone.
    pub(super) fn func_interp_lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        // Only a model the solver stands behind is described here.  After an
        // `unsat` (or an `unknown`) `self.solver.model()` may still hold the
        // last *candidate* — a model the refutation discarded — and rendering
        // interpretations from it printed entries no verdict supports, which
        // is also what made the canonical-map assertion in
        // [`Context::get_func_interp_raw`] fire on a script the solver had
        // just refuted.  `Context::get_model` has always had this guard; this
        // is the same guard on the same query.
        if self.last_result != Some(SolverResult::Sat) {
            return lines;
        }
        let Some(solver_model) = self.published_model() else {
            return lines;
        };
        let class_values = self.build_class_values(solver_model);
        let uninterpreted: Vec<(String, Vec<SortId>, SortId)> = self
            .declared_funs
            .iter()
            .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
            .map(|d| (d.name.clone(), d.arg_sorts.clone(), d.ret_sort))
            .collect();
        // A quantified goal's tables were certified as printed (`#P2b-74`:
        // where the congruence closure left two entries at one tuple, the
        // first one printed is the one the certificate checked), so the
        // printer's own two-entry assertion is for quantifier-free goals.
        let assert_function = !self.goal_is_quantified();
        for (name, arg_sorts, ret_sort) in uninterpreted {
            let name = name.as_str();
            let Some(((entries, else_value, arity), _)) =
                self.func_interp_reading(name, solver_model, &class_values, assert_function)
            else {
                continue;
            };
            let params: Vec<String> = arg_sorts
                .iter()
                .enumerate()
                .map(|(i, &s)| format!("(x!{i} {})", self.format_sort_name(s)))
                .collect();
            let mut body = else_value;
            // Innermost `ite` last, so the first entry is tested first.
            for (args, value) in entries.into_iter().rev() {
                if args.len() != arity {
                    continue;
                }
                // An entry that agrees with what the body already says adds a
                // guard and no information: `(ite (= x!0 #x01) #x07 #x07)` is
                // the else-value spelled twice.
                if value == body {
                    continue;
                }
                let guard = match args.len() {
                    1 => match args.first() {
                        Some(arg) => format!("(= x!0 {arg})"),
                        None => continue,
                    },
                    _ => {
                        let conjuncts: Vec<String> = args
                            .iter()
                            .enumerate()
                            .map(|(i, arg)| format!("(= x!{i} {arg})"))
                            .collect();
                        format!("(and {})", conjuncts.join(" "))
                    }
                };
                body = format!("(ite {guard} {value} {body})");
            }
            // Through the same symbol encoder the constant list uses, so a
            // function declared as `|f g|` is printed back re-parsably.
            lines.push(format!(
                "  (define-fun {} ({}) {} {body})",
                oxiz_core::smtlib::format_symbol(name),
                params.join(" "),
                self.format_sort_name(ret_sort)
            ));
        }
        lines
    }
}

impl Context {
    /// Whether an index term is itself a value (a literal or a constructor
    /// value), rather than a symbol the model assigns one.
    fn is_value_index(&self, index: TermId) -> bool {
        is_ground_value(index, &self.terms) || is_constructor_value(index, &self.terms)
    }
}

/// Whether `term` is a datatype **value**: a constructor applied to values
/// (literals or constructor values), which `format_value` prints verbatim
/// (`#P2b-72`).
///
/// A constructor-valued array index used to be dropped from the printed
/// `store` chain — neither a literal nor an uninterpreted-sort class, it named
/// "no position" — so `(assert (= (select a red) 1))` over `(Array C Int)`
/// printed `a = ((as const (Array C Int)) 0)`, a model falsifying its one
/// assertion, while `(get-value ((select a red)))` answered `1`.
pub(super) fn is_constructor_value(term: TermId, manager: &TermManager) -> bool {
    let mut stack: Vec<TermId> = vec![term];
    let mut seen = 0usize;
    while let Some(current) = stack.pop() {
        seen += 1;
        if seen > 4_096 {
            return false;
        }
        match manager.get(current).map(|t| &t.kind) {
            Some(TermKind::DtConstructor { args, .. }) => stack.extend(args.iter().copied()),
            Some(_) if current != term && is_ground_value(current, manager) => {}
            _ => return false,
        }
    }
    true
}

/// Whether `term` is a ground value literal — the shapes a model entry can
/// carry and the shapes `format_value` renders verbatim.
///
/// Used to decide whether an index can be placed in a `store` chain: an index
/// the model left symbolic names no position in the array.
///
/// A negated numeral is one: `((as const (Array Int Int)) (- 3))` spells its
/// default so, and reading it as symbolic printed the array as the constant
/// `0` beside a `(get-value)` answering `-3` (`#P2b-81`'s named mechanism,
/// re-fix pass 16).
fn is_ground_value(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|t| match &t.kind {
        TermKind::IntConst(_)
        | TermKind::RealConst(_)
        | TermKind::BitVecConst { .. }
        | TermKind::True
        | TermKind::False
        | TermKind::StringLit(_) => true,
        TermKind::Neg(inner) => manager
            .get(*inner)
            .is_some_and(|i| matches!(i.kind, TermKind::IntConst(_) | TermKind::RealConst(_))),
        _ => false,
    })
}

impl Context {
    /// The variable terms a quantifier of the assertions binds and no
    /// declaration names: `i` in `(forall ((i Int)) (= (select a i) 5))`, when
    /// no `i` of that sort is also declared (`#P2b-62`).
    ///
    /// A bound variable is interned as an ordinary `Var` term, so the body's
    /// own `(select a i)` is a term like any other — and the search can give
    /// it, and `i`, a model entry (0 and 0 on
    /// `bench/z3_parity/benchmarks/AUFLIA/array_extensionality.smt2`).  Laid
    /// down as a read of `a` at index `0`, that entry shadowed the real
    /// `(select a 0) = 10` and `(get-model)` published `a[0] = 0` beside a
    /// `(get-value ((select a 0)))` answering `10`: a model falsifying the
    /// script's own ground assertion.  A term over a bound variable says
    /// nothing about the model, so the renderer and its read-side twin skip
    /// every read that mentions one.
    pub(super) fn bound_only_variables(&self) -> BoundOnly {
        let declared: crate::prelude::FxHashSet<TermId> =
            self.declared_consts.iter().map(|decl| decl.term).collect();
        let mut names: crate::prelude::FxHashSet<(oxiz_core::interner::Spur, SortId)> =
            crate::prelude::FxHashSet::default();
        let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();
        let mut stack: Vec<TermId> = self.assertions.clone();
        let mut children: Vec<TermId> = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if let TermKind::Forall { vars, .. } | TermKind::Exists { vars, .. } = &data.kind {
                names.extend(vars.iter().copied());
            }
            children.clear();
            children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
            stack.extend(children.iter().copied());
        }
        BoundOnly { names, declared }
    }

    /// Whether `term` has a bound-only variable (see
    /// [`Context::bound_only_variables`]) as a free variable.
    pub(super) fn mentions_bound_variable(&self, term: TermId, bound: &BoundOnly) -> bool {
        !bound.names.is_empty()
            && self
                .terms
                .free_vars_including_patterns(term)
                .into_iter()
                .any(|var| {
                    !bound.declared.contains(&var)
                        && self.terms.get(var).is_some_and(|data| match data.kind {
                            TermKind::Var(name) => bound.names.contains(&(name, data.sort)),
                            _ => false,
                        })
                })
    }
}

/// The `(name, sort)` pairs a quantifier of the assertions binds, and the
/// declared constants (a declared constant sharing a bound name and sort is
/// the same term, and is kept).
pub(super) struct BoundOnly {
    names: crate::prelude::FxHashSet<(oxiz_core::interner::Spur, SortId)>,
    declared: crate::prelude::FxHashSet<TermId>,
}
