//! Solver settings: the logic, the search configuration and the individual
//! switches, plus the polarity analysis and datatype constructor extraction
//! that hang off them.
//!
//! These methods are extracted from `mod.rs` to keep that file under the
//! 2000-line refactoring threshold.
//!
//! # Every mutator here goes through [`Solver::settings_changed`]
//!
//! A [`Solver::check`] reads two things: the assertion stack, and the settings
//! in this module.  The assertion stack has its own invalidation hook
//! (`Solver::invalidate_results`, called from `assert` / `push` / `pop` /
//! `reset`); this module is the other half.  A setter that changed a lever the
//! solve loop honours *without* announcing it would let the cached verdict of
//! [`crate::solver::verdict_cache`] answer a question it was never asked —
//! `(check-sat)` timing out, `:timeout` being raised, and the same `unknown`
//! coming straight back out of the cache.  So the rule is mechanical: **every
//! `&mut self` method in this module ends by calling
//! [`Solver::settings_changed`]**, and the fingerprint additionally carries the
//! settings by value so a future setter that forgets is still caught.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_theories::arithmetic::ArithSolver;

use super::Solver;
use super::types::{Polarity, SolverConfig};

impl Solver {
    /// Announce that a solver *setting* has changed.
    ///
    /// Bumps [`Solver::settings_epoch`] (which the goal fingerprint carries) and
    /// drops the cached verdict, so the next [`Solver::check`] runs a real
    /// search under the new settings instead of replaying one run under the old
    /// ones.
    ///
    /// # Why this drops the verdict but not the model
    ///
    /// A model and an unsat core are statements about the *assertion stack*, and
    /// a setting change does not move it: a model that satisfied every assertion
    /// still satisfies them under a different timeout or random seed, so
    /// `(get-model)` stays answerable.  A cached verdict is different — it is a
    /// statement about what *this configuration's search* concluded, and
    /// `Unknown` in particular is a statement about resource exhaustion rather
    /// than about the goal.  Handing back an `Unknown` produced under a
    /// millisecond budget to a caller who has just raised the budget to ten
    /// minutes is simply a wrong answer to the question asked, and the same
    /// applies to answering with a verdict computed while unsat-core production
    /// was off (there is no core to hand over afterwards) or before a new random
    /// seed was supposed to perturb which model comes back.
    pub(super) fn settings_changed(&mut self) {
        self.settings_epoch = self.settings_epoch.wrapping_add(1);
        self.last_check = None;
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Replace the whole search configuration.
    pub fn set_config(&mut self, config: SolverConfig) {
        self.config = config;
        self.settings_changed();
    }

    /// Set a wall-clock timeout.
    pub fn set_timeout(&mut self, timeout: core::time::Duration) {
        self.config.timeout_ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        self.settings_changed();
    }

    /// Set the maximum number of SAT conflicts.
    pub fn set_conflict_limit(&mut self, max_conflicts: u64) {
        self.config.max_conflicts = max_conflicts;
        self.settings_changed();
    }

    /// Set the maximum number of SAT decisions.
    pub fn set_decision_limit(&mut self, max_decisions: u64) {
        self.config.max_decisions = max_decisions;
        self.settings_changed();
    }

    /// Enable or disable theory-aware branching
    pub fn set_theory_aware_branching(&mut self, enabled: bool) {
        self.theory_aware_branching = enabled;
        self.settings_changed();
    }

    /// Enable or disable unsat core production
    pub fn set_produce_unsat_cores(&mut self, produce: bool) {
        self.produce_unsat_cores = produce;
        self.settings_changed();
    }

    /// Seed the embedded SAT engine's phase-randomization PRNG.
    ///
    /// This realises the SMT-LIB `:random-seed` option: the SAT solver samples a
    /// random phase with probability `random_polarity_prob` (nonzero by default),
    /// so the seed genuinely perturbs the decision order and hence which model is
    /// returned for a satisfiable problem — while never affecting the sat/unsat
    /// verdict (soundness is seed-independent).  A seed of `0` reproduces the
    /// default out-of-the-box behaviour.
    pub fn set_random_seed(&mut self, seed: u64) {
        self.sat.set_random_seed(seed);
        self.settings_changed();
    }

    /// Set the logic
    pub fn set_logic(&mut self, logic: &str) {
        self.logic = Some(logic.to_string());
        self.settings_changed();

        // Switch ArithSolver based on logic
        // QF_NIA and QF_NRA use NLSAT solver for nonlinear arithmetic
        if logic.contains("NIA") {
            // Nonlinear integer arithmetic - use NLSAT with integer mode.
            //
            // Without the `nlsat` feature there is no NLSAT to install. The
            // linear fallback below is still set, and QF_NIA keeps the two
            // procedures that never lived in `oxiz-nlsat` — the static UNSAT
            // patterns and the witness-verified model searches — so on this
            // tree it decides the same goals either way. Anything none of them
            // reaches is caught by `check_core`'s `arith_atoms_need_theory`
            // gate and answered `unknown`.
            #[cfg(feature = "nlsat")]
            {
                self.nlsat = Some(oxiz_theories::nlsat::NlsatTheory::new(true));
            }
            self.arith = ArithSolver::lia(); // Keep LIA as fallback for linear constraints
            #[cfg(all(feature = "tracing", feature = "nlsat"))]
            tracing::info!("Using NLSAT solver for QF_NIA (nonlinear integer arithmetic)");
            #[cfg(all(feature = "tracing", not(feature = "nlsat")))]
            tracing::info!(
                "QF_NIA requested but this build has no `nlsat` feature: linear \
                 integer arithmetic, plus the static UNSAT patterns and the \
                 witness-verified model searches; anything they cannot reach \
                 answers unknown"
            );
        } else if logic.contains("NRA") {
            // Nonlinear real arithmetic - use NLSAT with real mode. This is
            // the logic that actually loses something when `nlsat` is off: the
            // model searches named above are gated on QF_NIA, so a *real*
            // nonlinear goal has nothing left to decide it and is conceded.
            #[cfg(feature = "nlsat")]
            {
                self.nlsat = Some(oxiz_theories::nlsat::NlsatTheory::new(false));
            }
            self.arith = ArithSolver::lra(); // Keep LRA as fallback for linear constraints
            #[cfg(all(feature = "tracing", feature = "nlsat"))]
            tracing::info!("Using NLSAT solver for QF_NRA (nonlinear real arithmetic)");
            #[cfg(all(feature = "tracing", not(feature = "nlsat")))]
            tracing::info!(
                "QF_NRA requested but this build has no `nlsat` feature: linear \
                 real arithmetic only, every nonlinear goal will answer unknown"
            );
        } else if logic.contains("LIA") || logic.contains("IDL") {
            // Integer arithmetic logic (QF_LIA, LIA, QF_AUFLIA, QF_IDL, etc.)
            self.arith = ArithSolver::lia();
        } else if logic.contains("LRA") || logic.contains("RDL") {
            // Real arithmetic logic (QF_LRA, LRA, QF_RDL, etc.)
            self.arith = ArithSolver::lra();
        } else if logic.contains("BV") {
            // Bitvector logic - use LIA since BV comparisons are handled
            // as bounded integer arithmetic
            self.arith = ArithSolver::lia();
        }
        // For other logics (QF_UF, etc.) keep the default LRA
    }

    /// Extract (variable, constructor) pair from an equality if one side is a variable
    /// and the other is a DtConstructor
    pub(super) fn extract_dt_var_constructor(
        &self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
    ) -> Option<(TermId, oxiz_core::interner::Spur)> {
        let lhs_term = manager.get(lhs)?;
        let rhs_term = manager.get(rhs)?;

        // lhs is var, rhs is constructor
        if matches!(lhs_term.kind, TermKind::Var(_)) {
            if let TermKind::DtConstructor { constructor, .. } = &rhs_term.kind {
                return Some((lhs, *constructor));
            }
        }
        // rhs is var, lhs is constructor
        if matches!(rhs_term.kind, TermKind::Var(_)) {
            if let TermKind::DtConstructor { constructor, .. } = &lhs_term.kind {
                return Some((rhs, *constructor));
            }
        }
        None
    }

    /// Collect polarity information for all subterms
    /// This is used for polarity-aware encoding optimization
    ///
    /// Iterative: `(term, polarity)` work items on an explicit heap stack, so
    /// nesting depth cannot overflow the native call stack.  The stored
    /// polarity of every term is a monotone join (towards `Both`), so a visit
    /// that does not change a term's stored polarity re-delivers exactly what
    /// an earlier visit already propagated to its subterms and is pruned;
    /// this both preserves the final `polarities` map exactly and bounds the
    /// work to three productive visits per term on a shared DAG.
    pub(super) fn collect_polarities(
        &mut self,
        term: TermId,
        polarity: Polarity,
        manager: &TermManager,
    ) {
        let mut stack: Vec<(TermId, Polarity)> = vec![(term, polarity)];
        while let Some((term, polarity)) = stack.pop() {
            // Update the polarity for this term
            let current = self.polarities.get(&term).copied();
            let new_polarity = match (current, polarity) {
                (Some(Polarity::Both), _) | (_, Polarity::Both) => Polarity::Both,
                (Some(Polarity::Positive), Polarity::Negative)
                | (Some(Polarity::Negative), Polarity::Positive) => Polarity::Both,
                (Some(p), _) => p,
                (None, p) => p,
            };

            // A visit that leaves the stored polarity unchanged would deliver
            // the very propagation an earlier visit with the same join result
            // already delivered, so its subterm walk is redundant.  (This
            // subsumes the previous `current == Some(Both)` early-out.)
            if current == Some(new_polarity) {
                continue;
            }
            self.polarities.insert(term, new_polarity);

            let Some(t) = manager.get(term) else {
                continue;
            };

            // Children are pushed in reverse so they pop in the original
            // left-to-right order.
            match &t.kind {
                TermKind::Not(arg) => {
                    let neg_polarity = match polarity {
                        Polarity::Positive => Polarity::Negative,
                        Polarity::Negative => Polarity::Positive,
                        Polarity::Both => Polarity::Both,
                    };
                    stack.push((*arg, neg_polarity));
                }
                TermKind::And(args) | TermKind::Or(args) => {
                    stack.extend(args.iter().rev().map(|&arg| (arg, polarity)));
                }
                TermKind::Implies(lhs, rhs) => {
                    let neg_polarity = match polarity {
                        Polarity::Positive => Polarity::Negative,
                        Polarity::Negative => Polarity::Positive,
                        Polarity::Both => Polarity::Both,
                    };
                    stack.push((*rhs, polarity));
                    stack.push((*lhs, neg_polarity));
                }
                TermKind::Ite(cond, then_br, else_br) => {
                    stack.push((*else_br, polarity));
                    stack.push((*then_br, polarity));
                    stack.push((*cond, Polarity::Both));
                }
                TermKind::Xor(lhs, rhs) | TermKind::Eq(lhs, rhs) => {
                    // For XOR and Eq, both sides appear in both polarities
                    stack.push((*rhs, Polarity::Both));
                    stack.push((*lhs, Polarity::Both));
                }
                _ => {
                    // For other terms (constants, variables, theory atoms),
                    // stop the walk
                }
            }
        }
    }
}
