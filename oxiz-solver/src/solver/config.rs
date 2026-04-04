//! Solver configuration helpers: logic selection, polarity analysis, and
//! datatype constructor extraction.
//!
//! These methods are extracted from `mod.rs` to keep that file under the
//! 2000-line refactoring threshold.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_theories::arithmetic::ArithSolver;

use super::Solver;
use super::types::Polarity;

impl Solver {
    /// Set the logic
    pub fn set_logic(&mut self, logic: &str) {
        self.logic = Some(logic.to_string());

        // Switch ArithSolver based on logic
        // QF_NIA and QF_NRA use NLSAT solver for nonlinear arithmetic
        if logic.contains("NIA") {
            // Nonlinear integer arithmetic - use NLSAT with integer mode
            #[cfg(feature = "std")]
            {
                self.nlsat = Some(oxiz_theories::nlsat::NlsatTheory::new(true));
            }
            self.arith = ArithSolver::lia(); // Keep LIA as fallback for linear constraints
            #[cfg(feature = "tracing")]
            tracing::info!("Using NLSAT solver for QF_NIA (nonlinear integer arithmetic)");
        } else if logic.contains("NRA") {
            // Nonlinear real arithmetic - use NLSAT with real mode
            #[cfg(feature = "std")]
            {
                self.nlsat = Some(oxiz_theories::nlsat::NlsatTheory::new(false));
            }
            self.arith = ArithSolver::lra(); // Keep LRA as fallback for linear constraints
            #[cfg(feature = "tracing")]
            tracing::info!("Using NLSAT solver for QF_NRA (nonlinear real arithmetic)");
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
    pub(super) fn collect_polarities(
        &mut self,
        term: TermId,
        polarity: Polarity,
        manager: &TermManager,
    ) {
        // Update the polarity for this term
        let current = self.polarities.get(&term).copied();
        let new_polarity = match (current, polarity) {
            (Some(Polarity::Both), _) | (_, Polarity::Both) => Polarity::Both,
            (Some(Polarity::Positive), Polarity::Negative)
            | (Some(Polarity::Negative), Polarity::Positive) => Polarity::Both,
            (Some(p), _) => p,
            (None, p) => p,
        };
        self.polarities.insert(term, new_polarity);

        // If we've reached Both polarity, no need to recurse further
        if current == Some(Polarity::Both) {
            return;
        }

        // Recursively collect polarities for subterms
        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Not(arg) => {
                let neg_polarity = match polarity {
                    Polarity::Positive => Polarity::Negative,
                    Polarity::Negative => Polarity::Positive,
                    Polarity::Both => Polarity::Both,
                };
                self.collect_polarities(*arg, neg_polarity, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    self.collect_polarities(arg, polarity, manager);
                }
            }
            TermKind::Implies(lhs, rhs) => {
                let neg_polarity = match polarity {
                    Polarity::Positive => Polarity::Negative,
                    Polarity::Negative => Polarity::Positive,
                    Polarity::Both => Polarity::Both,
                };
                self.collect_polarities(*lhs, neg_polarity, manager);
                self.collect_polarities(*rhs, polarity, manager);
            }
            TermKind::Ite(cond, then_br, else_br) => {
                self.collect_polarities(*cond, Polarity::Both, manager);
                self.collect_polarities(*then_br, polarity, manager);
                self.collect_polarities(*else_br, polarity, manager);
            }
            TermKind::Xor(lhs, rhs) | TermKind::Eq(lhs, rhs) => {
                // For XOR and Eq, both sides appear in both polarities
                self.collect_polarities(*lhs, Polarity::Both, manager);
                self.collect_polarities(*rhs, Polarity::Both, manager);
            }
            _ => {
                // For other terms (constants, variables, theory atoms), stop recursion
            }
        }
    }
}
