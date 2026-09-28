//! The SAT certification's saturation test, and the goal it needs to complete
//! an array index position's relevant set (`#P2b-60`; see
//! `mbqi::sat_certify::unnamed_region`).

use oxiz_core::ast::{TermId, TermKind, TermManager};

use super::super::model_completion::CompletedModel;
use super::super::sat_certify::{self, CertifyResult};
use super::super::{Instantiation, MBQIResult, QuantifiedFormula};
use super::{MBQIIntegration, SAT_CERTIFY_CAP, SolverCallback};

impl MBQIIntegration {
    /// Record the assertions in scope, for the next MBQI round.
    ///
    /// Called by `Solver::check_core` before each round rather than kept
    /// incrementally: the assertion stack moves with `push`/`pop`, and a copy
    /// taken from it cannot outlive the scope that asserted a term.
    pub fn note_goal(&mut self, assertions: &[TermId]) {
        self.goal_assertions = assertions.to_vec();
    }

    /// The certifier's verdict on one round's complete instance set `insts`:
    /// the fresh instances to refine with, `Satisfied`, or `None` when the
    /// goal turns out not to be certifiable after all (the caller then keeps
    /// its ordinary path, which never concludes `Satisfied` on an unsampled
    /// domain).
    ///
    /// Saturation over the relevant set is not yet a conclusion: it is
    /// re-checked with the representatives of the array-index region no term
    /// names added, and only when those instances are not fresh either is the
    /// goal `Satisfied` (`#P2b-60`).
    pub(super) fn certify_or_refine(
        &mut self,
        insts: Vec<Instantiation>,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
        callback: &mut dyn SolverCallback,
    ) -> Option<MBQIResult> {
        let fresh = self.fresh_fragment_instances(insts, manager, callback);
        if !fresh.is_empty() {
            return Some(MBQIResult::NewInstantiations(fresh));
        }
        let goal = self.goal_assertions.clone();
        match sat_certify::collect_fragment_instances(
            quantifiers,
            model,
            Some(&goal),
            manager,
            SAT_CERTIFY_CAP,
            self.current_round as u32,
        ) {
            CertifyResult::Instances(insts) => {
                let fresh = self.fresh_fragment_instances(insts, manager, callback);
                self.unnamed_only = !fresh.is_empty();
                Some(if fresh.is_empty() {
                    // Saturated: every relevant instance, and every instance
                    // at a representative of the unnamed region, was emitted
                    // in an earlier round or is a tautology, and the ground
                    // solver still found a model — so by the completeness
                    // argument for this fragment the goal is `Sat`.
                    MBQIResult::Satisfied
                } else {
                    MBQIResult::NewInstantiations(fresh)
                })
            }
            CertifyResult::NotEligible => None,
        }
    }

    /// Whether the instances this round returned are exactly the
    /// representatives of the unnamed region — the relevant set itself is
    /// saturated.  The solver then tries to conclude with a certified model
    /// completion first (`Solver::certify_sat_by_array_completion`), which
    /// needs none of them: the extra index term those instances add is what
    /// an array search over `ite`-selected bases pays for most, and a sound
    /// `sat` that does not need it should not pay it.
    pub fn only_unnamed_region_pending(&self) -> bool {
        self.unnamed_only
    }

    /// The instances of `insts` not emitted in an earlier round, each
    /// recorded, simplified, and dropped again if it is a tautology.
    fn fresh_fragment_instances(
        &mut self,
        insts: Vec<Instantiation>,
        manager: &mut TermManager,
        callback: &mut dyn SolverCallback,
    ) -> Vec<Instantiation> {
        let mut fresh = Vec::new();
        for mut inst in insts {
            if self.is_duplicate(&inst) {
                continue;
            }
            // Record against the (quantifier, binding) key *before* the
            // tautology filter below.  Saturation is detected purely by this
            // key (never by the result term), so recording every relevant
            // tuple — even those whose body collapses to `true` — is what lets
            // a later round observe "nothing fresh" and conclude `Satisfied`
            // soundly.
            self.record_instantiation(&inst);

            // Simplify so that the concrete guards of a bounded-box instance
            // collapse: e.g. `(and (>= 1 0) (<= 1 10) (= (f 1) (f 2)))` becomes
            // `(= 1 2)` and reduces to the clean disequality.  Emitting the raw
            // guarded implication instead feeds the downstream pigeonhole /
            // integer-domain clause heuristics a spurious "bounded integer
            // variable" shape (the substituted constants still parse as
            // `(>= c 0) (<= c 10)` conjuncts), which over-constrains the ground
            // problem and can flip a satisfiable goal to a spurious `unsat`.
            // This mirrors the enumerative path, which simplifies for the same
            // reason.
            inst.result = self.deep_simplify(inst.result, manager);

            // A tautology instance (body ≡ ⊤) constrains nothing.  It is
            // already recorded above (so the set can still saturate), so just
            // skip emitting it as a lemma.
            if manager
                .get(inst.result)
                .is_some_and(|t| matches!(t.kind, TermKind::True))
            {
                continue;
            }

            callback.on_instantiation(&inst);
            fresh.push(inst);
        }
        fresh
    }
}
