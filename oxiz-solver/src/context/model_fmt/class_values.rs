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
