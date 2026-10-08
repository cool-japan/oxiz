//! The one canonical *congruence class → value* map every model printer reads
//! (`#P2b-34`).
//!
//! `(get-model)` reports a model through four renderers — the constant list,
//! the `define-fun` interpretation of each declared function, the `store`
//! chain of each array, and `(get-value …)` — and every one of them needs the
//! same answer to the same question: *what value does this model give the
//! equivalence class of this term?*  Each used to answer it for itself, and
//! the answers disagreed:
//!
//! * [`Context::get_model`]'s constant loop synthesises a `@uc_S_n` abstract
//!   witness per congruence class for an uninterpreted sort, so `(= (f a) b)`
//!   with `(distinct a b)` printed `a = @uc_U_0`, `b = @uc_U_1`;
//! * `get_func_interp_raw`'s own class walk had no such synthesis, so `b`'s
//!   class looked *valueless*, the entry for `f(a)` was dropped, and `f` fell
//!   back to the return sort's default `@uc_U_0` — printing `f = @uc_U_0`
//!   beside `b = @uc_U_1` while the assertion says `f(a) = b`.  The printed
//!   model falsified the assertions it was a model of.
//!
//! Building the map once and reading it everywhere makes that class of
//! divergence unrepresentable: there is one place a class's value is decided,
//! and every renderer quotes it.  Two entries of one interpretation with the
//! same argument tuple and different values then cannot arise either, which
//! [`Context::get_func_interp_raw`] asserts in debug builds.

use super::*;
use oxiz_core::ast::TermManager;

/// A congruence class's identity, as a single integer.
///
/// Terms the congruence closure interned key by their class representative;
/// terms it never saw — the pure-equality fast path decides some formulas
/// without running EUF at all — key by their own term id, in a disjoint
/// numbering (the high bit) so a representative id and a term id can never
/// collide.
pub(in crate::context) type ClassKey = u64;

/// The canonical class → value assignment of one `(get-model)` / `(get-value)`
/// query.
#[derive(Default)]
pub(in crate::context) struct ClassValues {
    values: crate::prelude::HashMap<ClassKey, String>,
}

impl ClassValues {
    /// The value this model gives `term`'s class, if the map holds one.
    pub(in crate::context) fn get(&self, key: ClassKey) -> Option<&str> {
        self.values.get(&key).map(String::as_str)
    }

    /// Record `value` for `key` unless the class already has one.  First write
    /// wins, so the declaration-order pass below fixes the witness numbering.
    fn insert(&mut self, key: ClassKey, value: String) {
        self.values.entry(key).or_insert(value);
    }
}

impl Context {
    /// The congruence class key of `term` — see [`ClassKey`].
    pub(super) fn class_key(&self, term: TermId) -> ClassKey {
        match self.solver.euf_class_representative(term) {
            Some(rep) => (1u64 << 32) | u64::from(rep),
            None => u64::from(term.0),
        }
    }

    /// Build the canonical class → value map for `solver_model`.
    ///
    /// The declared constants are walked in declaration order, because the
    /// `@uc_S_n` witnesses are numbered in that order and the numbering is
    /// user-visible output.  `witness_of` is the same synthesis
    /// [`Context::get_model`] performs, lifted here so that the constant list
    /// and every other renderer agree by construction.
    pub(super) fn build_class_values(&self, solver_model: &crate::solver::Model) -> ClassValues {
        let mut out = ClassValues::default();
        let mut per_sort_next: crate::prelude::HashMap<SortId, usize> =
            crate::prelude::HashMap::new();

        for decl in &self.declared_consts {
            let key = self.class_key(decl.term);
            // The exact-value side-channel first, for the same reason
            // `get_model` consults it first: it is the more precise of the two
            // sources (see that loop).
            if let Some(exact) = self.solver.nl_algebraic_value(decl.term) {
                out.insert(key, render_nl_witness_value(exact));
                continue;
            }
            if let Some(val) = solver_model.get(decl.term) {
                out.insert(key, self.format_value(val));
                continue;
            }
            if self.is_uninterpreted_sort(decl.sort) {
                if out.get(key).is_some() {
                    continue;
                }
                let next = per_sort_next.entry(decl.sort).or_insert(0);
                let index = *next;
                *next += 1;
                out.insert(
                    key,
                    format!("@uc_{}_{}", self.format_sort_name(decl.sort), index),
                );
            }
        }

        // Every *other* congruence class of an uninterpreted sort gets a
        // witness too (`#P2b-40`), numbered from the declared count upward so
        // the numbering the declaration pass above fixed does not move.
        //
        // Without this the class of an `Apply` result had no value at all:
        // `class_value_string` answered `None`, `func_interp_from` dropped the
        // entry, and the interpretation fell back to the return sort's default
        // — so `(declare-sort U 0) (declare-fun g (U) U) (assert (distinct
        // (g p) (g q)))` answered `sat` and printed `(define-fun g ((x!0 U)) U
        // @uc_U_0)` beside `p = @uc_U_0` and `q = @uc_U_1`, making
        // `(g p) = (g q) = @uc_U_0` and falsifying the one assertion the model
        // claimed to satisfy.  An uninterpreted sort has as many elements as
        // it needs, so a fresh witness per class is always a legitimate
        // reading — and it is the *only* one that keeps distinct classes
        // distinct.
        //
        // Walked in term-id order, which is the order the terms were interned:
        // deterministic, and independent of hashing.
        for index in 0..(self.terms.len() as u32) {
            let term = TermId(index);
            let Some(sort) = self.terms.get(term).map(|t| t.sort) else {
                continue;
            };
            if !self.is_uninterpreted_sort(sort) {
                continue;
            }
            let key = self.class_key(term);
            if out.get(key).is_some() {
                continue;
            }
            let next = per_sort_next.entry(sort).or_insert(0);
            let witness_index = *next;
            *next += 1;
            out.insert(
                key,
                format!("@uc_{}_{}", self.format_sort_name(sort), witness_index),
            );
        }

        // Every other term the model assigned a value to, and every literal
        // value term in the term graph: an application's argument or result
        // class is frequently one of these rather than a declared constant.
        for (&term, &value) in solver_model.assignments() {
            if is_value_term(value, &self.terms) {
                out.insert(self.class_key(term), self.format_value(value));
            }
        }
        for index in 0..(self.terms.len() as u32) {
            let term = TermId(index);
            if is_value_term(term, &self.terms) {
                out.insert(self.class_key(term), self.format_value(term));
            }
        }

        out
    }
}

/// Whether `term` is a literal value of its sort — a term that *is* its own
/// model value, so a class containing one needs no lookup.
fn is_value_term(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|t| {
        matches!(
            t.kind,
            TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_)
        )
    })
}

impl Context {
    /// The constructor value this model gives the datatype term `term`, built
    /// from the values it gives `term`'s *selector applications* (`#P2b-39`).
    ///
    /// `(get-model)` reaches here only for a datatype constant the solver's own
    /// reconstruction (`model_builder::extract_datatype_model`) could not
    /// build — most often because the formula decided no tester literal, so
    /// there was nothing to reconstruct *from*.  The fall-through used to be
    /// [`Context::default_value`], a constructor whose every field is its
    /// sort's default; and a default is exactly what the script's own
    /// assertions contradict.  A `Box` with an array field and a `(_ BitVec 2)`
    /// tag printed as `(mk ((as const …) #b0) #b00)` while, in the same run,
    /// `(get-value ((tag b)))` answered the asserted `#b01` and
    /// `(get-value ((select (arr b) #b0)))` the asserted `#b1`: one model, two
    /// readings, and the printed one falsifying the script.
    ///
    /// The fields are read out of the one canonical class → value map every
    /// other renderer quotes, so the two commands now describe one model.  A
    /// field the script never mentions has no selector application to read and
    /// keeps its sort default, which is a legitimate witness there — the
    /// formula does not constrain it.
    ///
    /// `None` when the sort is not a datatype, its declaration is not in scope,
    /// or no constructor can be chosen; the caller then falls back exactly as
    /// before.
    pub(super) fn datatype_class_value(
        &self,
        term: TermId,
        sort: SortId,
        model: &crate::solver::Model,
        class_values: &ClassValues,
    ) -> Option<String> {
        let dt_name = self.terms.sorts.datatype_name(sort)?.to_string();
        let def = self.terms.sorts.get_datatype(&dt_name)?;
        let index = self.decided_constructor_index(term, &dt_name).or_else(|| {
            crate::solver::model_builder::default_constructor_index(def, &self.terms.sorts)
        })?;
        let constructor = def.constructors.get(index)?;
        let name = self.terms.resolve_str(constructor.name).to_string();
        let selectors: Vec<(oxiz_core::interner::Spur, SortId)> =
            constructor.selectors.iter().copied().collect();
        if selectors.is_empty() {
            return Some(name);
        }
        let mut fields: Vec<String> = Vec::with_capacity(selectors.len());
        let mut informed = false;
        for (selector, field_sort) in selectors {
            let read = self
                .selector_application(term, selector)
                .and_then(|application| {
                    self.field_value(application, field_sort, model, class_values)
                });
            informed |= read.is_some();
            fields.push(read.unwrap_or_else(|| self.default_value(field_sort)));
        }
        // Not one field was read from the model: this is the sort default with
        // extra steps, and `Context::default_value` is where that is built —
        // under the expansion budget that keeps an ill-founded datatype from
        // unfolding forever.  Answering here instead prepended one constructor
        // level to that budget's output, which is how a `?`-truncated default
        // grew from sixteen levels to seventeen.
        if !informed {
            return None;
        }
        Some(format!("({} {})", name, fields.join(" ")))
    }

    /// The index of the constructor a decided tester literal picks for `term`,
    /// or `None` when the model decided none.
    fn decided_constructor_index(&self, term: TermId, dt_name: &str) -> Option<usize> {
        let model = self.solver.model()?;
        let def = self.terms.sorts.get_datatype(dt_name)?;
        for index in 0..(self.terms.len() as u32) {
            let candidate = TermId(index);
            let Some(TermKind::DtTester { constructor, arg }) =
                self.terms.get(candidate).map(|t| &t.kind)
            else {
                continue;
            };
            if *arg != term {
                continue;
            }
            if !matches!(
                model
                    .get(candidate)
                    .and_then(|v| self.terms.get(v))
                    .map(|t| &t.kind),
                Some(TermKind::True)
            ) {
                continue;
            }
            let name = *constructor;
            if let Some(position) = def.constructors.iter().position(|c| c.name == name) {
                return Some(position);
            }
        }
        None
    }

    /// The interned `(selector term)` application, when the term graph already
    /// holds one.
    ///
    /// Looked up by scanning rather than interned, because `(get-model)` takes
    /// `&self`: a model query must not grow the term graph it is describing.
    /// The scan is linear in the term count and runs once per datatype constant
    /// per field, which is a print-time cost on a model that is about to be
    /// formatted anyway.
    fn selector_application(
        &self,
        term: TermId,
        selector: oxiz_core::interner::Spur,
    ) -> Option<TermId> {
        (0..(self.terms.len() as u32)).map(TermId).find(|&candidate| {
            matches!(
                self.terms.get(candidate).map(|t| &t.kind),
                Some(TermKind::DtSelector { selector: s, arg }) if *s == selector && *arg == term
            )
        })
    }

    /// The value this model gives one constructor field, rendered the way the
    /// field's sort is rendered everywhere else: an array as the `store` chain
    /// [`Context::array_model_value`] prints, anything else as the class value.
    fn field_value(
        &self,
        application: TermId,
        field_sort: SortId,
        model: &crate::solver::Model,
        class_values: &ClassValues,
    ) -> Option<String> {
        if matches!(
            self.terms.sorts.get(field_sort).map(|s| &s.kind),
            Some(SortKind::Array { .. })
        ) {
            return self.array_model_value(application, field_sort, model, class_values);
        }
        if let Some(value) = model.get(application) {
            return Some(self.format_value(value));
        }
        class_values
            .get(self.class_key(application))
            .map(ToString::to_string)
    }
}
