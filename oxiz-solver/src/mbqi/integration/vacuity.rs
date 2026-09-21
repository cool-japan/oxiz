//! Discharging a quantifier the candidate model already satisfies.
//!
//! One method of [`MBQIIntegration`](super::MBQIIntegration), split out of
//! `integration/mod.rs` to keep that file under the workspace's 2,000-line
//! refactoring policy, plus the one free-variable predicate that guards blind
//! instantiation.  Both exist for `#P2b-57`; the method's own doc comment
//! carries the soundness argument.

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::traversal::collect_free_vars_including_patterns;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;

use super::MBQIIntegration;
use crate::mbqi::QuantifiedFormula;

impl MBQIIntegration {
    /// Is `quantifier` satisfied by `model` *whatever* its bound variables
    /// range over, because the ground Boolean atoms in its body already decide
    /// it?
    ///
    /// # What this closes
    ///
    /// [`encode::quant_guard`](crate::solver::Solver) ties a conditionally
    /// placed quantifier to a fresh Boolean constant `g` by asserting the plain
    /// universal `∀x. (g → φ)` beside the assertion.  That universal is a real
    /// obligation the search must discharge before it may answer `Sat` — and
    /// when the search sets `g` to `false` it is *already* discharged, for
    /// every `x` at once, with no instantiation at all.  Without this check the
    /// obligation is handed to the counterexample generator, `φ`'s
    /// uninterpreted applications come back symbolic, `all_evaluations_ground`
    /// is cleared and the round cannot conclude `Satisfied`:
    ///
    /// ```text
    /// (declare-fun r ((_ BitVec 7)) Bool)
    /// (declare-const p Bool)
    /// (assert p)
    /// (assert (or p (forall ((x (_ BitVec 7))) (r x))))
    /// ```
    ///
    /// `(assert p)` alone satisfies the disjunction, so the quantifier plays no
    /// part in the verdict — and the script answered `unknown` (`#P2b-57`).
    ///
    /// # Why it is sound
    ///
    /// The reduction substitutes, for a Boolean **variable** of the body that
    /// none of `quantifier`'s own binders bind, the value `model` gives it.
    /// Such a variable is ground, so the substitution is valid *uniformly in
    /// the bound variables*: if the result simplifies to `true`, then `φ` is
    /// true in `model` at every point of the domain and `model ⊨ ∀x. φ`.
    ///
    /// Three restrictions make that argument airtight, and each is load-bearing:
    ///
    /// * `model` is the **partial model the solver publishes**, not the MBQI
    ///   completion.  A completion value the user never sees must not be what a
    ///   `Sat` rests on.
    /// * Only `TermKind::Var` atoms are substituted.  A compound Boolean term
    ///   carries its value through the theory solvers, and this pass must not
    ///   have to trust that the Boolean abstraction of `(= a b)` agrees with
    ///   `a` and `b`'s own entries.
    /// * A variable the model does **not** assign is left alone, so the check
    ///   cannot fire on an undetermined Boolean — which is the very thing
    ///   `Solver::model_leaves_a_boolean_undetermined` reports as `unknown`
    ///   (`#P2b-27`).
    ///
    /// Discharging is per *round* and per *model*: nothing is remembered, so a
    /// later round with a different candidate re-derives its own answer.
    pub(super) fn quantifier_vacuous_under_model(
        &self,
        quantifier: &QuantifiedFormula,
        model: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> bool {
        let bound: FxHashSet<Spur> = quantifier.bound_vars.iter().map(|(n, _)| *n).collect();
        let bool_sort = manager.sorts.bool_sort;
        let mut substitution: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut stack: Vec<TermId> = vec![quantifier.body];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some((kind, sort)) = manager.get(current).map(|t| (t.kind.clone(), t.sort)) else {
                continue;
            };
            if sort == bool_sort
                && let TermKind::Var(name) = kind
                && !bound.contains(&name)
                && let Some(&value) = model.get(&current)
                && matches!(
                    manager.get(value).map(|t| &t.kind),
                    Some(TermKind::True | TermKind::False)
                )
            {
                substitution.insert(current, value);
                continue;
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&kind));
        }

        if substitution.is_empty() {
            // Nothing the model decides occurs in the body; this is exactly the
            // case `all_quantifiers_trivially_valid` already answers, and
            // re-deriving it here would cost a substitution for no new verdict.
            return false;
        }
        let reduced = manager.substitute(quantifier.body, &substitution);
        let simplified = self.deep_simplify(reduced, manager);
        manager
            .get(simplified)
            .is_some_and(|t| matches!(t.kind, TermKind::True))
    }

    /// Does `term` still contain a free occurrence of a variable that one of
    /// the tracked quantifiers binds?
    ///
    /// The one guard on emitting a blind instantiation lemma (see
    /// `generate_blind_instantiations`).  `Instantiation` lemmas are asserted
    /// as hard clauses, and a stray bound variable in one would be read as the
    /// *declared constant* of the same name — a constraint the formula never
    /// stated.  Patterns count: [`collect_free_vars_including_patterns`] looks
    /// inside triggers for exactly that reason.
    pub(super) fn mentions_tracked_bound_var(&self, term: TermId, manager: &TermManager) -> bool {
        let bound: FxHashSet<Spur> = self
            .quantifiers
            .iter()
            .flat_map(|q| q.bound_vars.iter().map(|(name, _)| *name))
            .collect();
        if bound.is_empty() {
            return false;
        }
        collect_free_vars_including_patterns(term, manager)
            .iter()
            .any(|&free| {
                matches!(manager.get(free).map(|t| &t.kind),
                    Some(TermKind::Var(name)) if bound.contains(name))
            })
    }
}
