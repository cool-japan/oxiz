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
    ) -> Option<String> {
        self.array_class_value(array, sort, model, 0)
    }

    /// [`Context::array_model_value`] at nesting depth `depth`; see
    /// [`ARRAY_MODEL_NESTING_LIMIT`].
    fn array_class_value(
        &self,
        array: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        depth: u32,
    ) -> Option<String> {
        let mut visiting: Vec<TermId> = Vec::new();
        let (base, entries) = self.array_class_parts(array, sort, model, depth, &mut visiting)?;
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
        depth: u32,
        visiting: &mut Vec<TermId>,
    ) -> Option<(String, Vec<(String, String)>)> {
        if depth >= ARRAY_MODEL_NESTING_LIMIT || visiting.contains(&array) {
            return None;
        }
        visiting.push(array);
        let parts = self.array_class_parts_inner(array, sort, model, depth, visiting);
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
            let default_str = self.entry_value(default, sort, model, depth)?;
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
        for &member in &sorted_members {
            let Some(data) = self.terms.get(member) else {
                continue;
            };
            let TermKind::Store(store_base, index, value) = data.kind else {
                continue;
            };
            // A store member also tells us the class's *background*: every
            // index it does not write reads through to its own base array.
            if base.is_none() {
                base = self
                    .array_class_parts(store_base, sort, model, depth + 1, visiting)
                    .map(|(inner_base, inner_entries)| {
                        let mut chain = inner_base;
                        for (index_str, value_str) in inner_entries {
                            chain = format!("(store {chain} {index_str} {value_str})");
                        }
                        chain
                    });
            }
            self.push_entry(index, value, sort, model, depth, &mut seen, &mut entries);
        }

        // ---- reads -----------------------------------------------------
        let mut reads: Vec<(TermId, TermId, TermId)> = Vec::new();
        for (&term, &value) in model.assignments() {
            let Some(data) = self.terms.get(term) else {
                continue;
            };
            if let TermKind::Select(read_array, index) = data.kind
                && members.contains(&read_array)
            {
                reads.push((term, index, value));
            }
        }
        reads.sort_unstable_by_key(|&(term, _, _)| term.raw());
        for (_, index, value) in reads {
            self.push_entry(index, value, sort, model, depth, &mut seen, &mut entries);
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
        // The term-graph scan is confined to this case: an array whose range
        // is not itself an array has all its reads in the model.
        if self.range_is_array(sort) {
            let mut nested: Vec<(TermId, TermId)> = Vec::new();
            for index in 0..(self.terms.len() as u32) {
                let term = TermId(index);
                let Some(data) = self.terms.get(term) else {
                    continue;
                };
                if let TermKind::Select(read_array, read_index) = data.kind
                    && members.contains(&read_array)
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
                    depth,
                    &mut seen,
                    &mut entries,
                );
            }
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
            base =
                self.inherited_parts(array, sort, model, depth, visiting, &mut seen, &mut entries);
        }

        if base.is_none() && entries.is_empty() {
            return None;
        }
        Some((base.unwrap_or_else(|| self.default_value(sort)), entries))
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
                self.array_class_parts(term, sort, model, depth + 1, visiting)
            else {
                continue;
            };
            // The one index the store overwrote says nothing about the base.
            let written = model.get(store_index).unwrap_or(store_index);
            let written_str = if is_ground_value(written, &self.terms) {
                self.format_value(written)
            } else {
                String::new()
            };
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
        depth: u32,
        seen: &mut crate::prelude::HashSet<String>,
        entries: &mut Vec<(String, String)>,
    ) {
        let index_value = model.get(index).unwrap_or(index);
        if !is_ground_value(index_value, &self.terms) {
            return;
        }
        let index_str = self.format_value(index_value);
        if seen.contains(&index_str) {
            return;
        }
        let value_value = model.get(value).unwrap_or(value);
        let Some(value_str) = self.entry_value(value_value, sort, model, depth) else {
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
        depth: u32,
        visiting: &mut Vec<TermId>,
    ) -> Option<String> {
        if depth >= ARRAY_MODEL_NESTING_LIMIT || visiting.contains(&array) {
            return None;
        }
        visiting.push(array);
        let value =
            self.array_class_read_inner(array, index_value, sort, model, depth, visiting, false);
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
    fn array_class_background(
        &self,
        array: TermId,
        index_value: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        depth: u32,
        visiting: &mut Vec<TermId>,
    ) -> Option<String> {
        if depth >= ARRAY_MODEL_NESTING_LIMIT || visiting.contains(&array) {
            return None;
        }
        visiting.push(array);
        let value =
            self.array_class_read_inner(array, index_value, sort, model, depth, visiting, true);
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
            return self.entry_value(model.get(value).unwrap_or(value), sort, model, depth);
        }

        // ---- reads -----------------------------------------------------
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
            {
                reads.push((term, value));
            }
        }
        reads.sort_unstable_by_key(|&(term, _)| term.raw());
        if let Some(&(_, value)) = reads.first() {
            return self.entry_value(value, sort, model, depth);
        }
        // An *array-sorted* read is never in `model.assignments()` — no term
        // denotes a whole array — so the read itself is the value, rendered
        // from its own class one level down (the array-of-arrays case).
        if self.range_is_array(sort) && !background_only {
            let mut nested: Vec<TermId> = Vec::new();
            for index in 0..(self.terms.len() as u32) {
                let term = TermId(index);
                let Some(data) = self.terms.get(term) else {
                    continue;
                };
                if let TermKind::Select(read_array, read_index) = data.kind
                    && members.contains(&read_array)
                    && model.get(read_index).unwrap_or(read_index) == index_value
                {
                    nested.push(term);
                }
            }
            nested.sort_unstable();
            if let Some(&read) = nested.first() {
                return self.entry_value(read, sort, model, depth);
            }
        }

        // ---- background ------------------------------------------------
        for &member in &members {
            if let Some(default) =
                crate::solver::array_axioms::const_array_default(member, &self.terms)
            {
                return self.entry_value(default, sort, model, depth);
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
                self.array_class_background(term, index_value, sort, model, depth + 1, visiting)
            } else {
                self.array_class_read(term, index_value, sort, model, depth + 1, visiting)
            };
            if let Some(value) = value {
                return Some(value);
            }
        }
        None
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
            // A value the model left symbolic has no printable form: the `?`
            // placeholder `format_value` falls back to is not SMT-LIB, and a
            // store chain carrying one cannot be read back at all.
            return is_ground_value(value, &self.terms).then(|| self.format_value(value));
        }
        // An array-sorted entry is an array in its own right: render its class
        // the same way, one level down.
        self.array_class_value(value, range, model, depth + 1)
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
        let Some(solver_model) = self.solver.model() else {
            return lines;
        };
        let class_values = self.build_class_values(solver_model);
        let uninterpreted: Vec<(String, Vec<SortId>, SortId)> = self
            .declared_funs
            .iter()
            .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
            .map(|d| (d.name.clone(), d.arg_sorts.clone(), d.ret_sort))
            .collect();
        for (name, arg_sorts, ret_sort) in uninterpreted {
            let name = name.as_str();
            let Some((entries, else_value, arity)) =
                self.func_interp_from(name, solver_model, &class_values)
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
            lines.push(format!(
                "  (define-fun {name} ({}) {} {body})",
                params.join(" "),
                self.format_sort_name(ret_sort)
            ));
        }
        lines
    }
}

/// Whether `term` is a ground value literal — the shapes a model entry can
/// carry and the shapes `format_value` renders verbatim.
///
/// Used to decide whether an index can be placed in a `store` chain: an index
/// the model left symbolic names no position in the array.
fn is_ground_value(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|t| {
        matches!(
            t.kind,
            TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::True
                | TermKind::False
                | TermKind::StringLit(_)
        )
    })
}
