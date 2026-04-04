//! Model and unsat core building

#[allow(unused_imports)]
use crate::prelude::*;
use num_traits::ToPrimitive;
use oxiz_core::ast::{TermId, TermKind, TermManager};

use super::Solver;
use super::types::Constraint;
use super::types::{Model, UnsatCore};

impl Solver {
    pub(super) fn build_model(&mut self, manager: &mut TermManager) {
        let mut model = Model::new();
        let sat_model = self.sat.model();

        // Get boolean values from SAT model
        for (&term, &var) in &self.term_to_var {
            let val = sat_model.get(var.index()).copied();
            if let Some(v) = val {
                let bool_val = if v.is_true() {
                    manager.mk_true()
                } else if v.is_false() {
                    manager.mk_false()
                } else {
                    continue;
                };
                model.set(term, bool_val);
            }
        }

        // Extract values from equality constraints (e.g., x = 5)
        // This handles cases where a variable is equated to a constant
        for (&var, constraint) in &self.var_to_constraint {
            // Check if the equality is assigned true in the SAT model
            let is_true = sat_model
                .get(var.index())
                .copied()
                .is_some_and(|v| v.is_true());

            if !is_true {
                continue;
            }

            if let Constraint::Eq(lhs, rhs) = constraint {
                // Check if one side is a tracked variable and the other is a constant.
                // Also handle Apply terms (uninterpreted function applications) that are
                // not in arith_terms due to the restriction on Apply terms with arith args.
                let lhs_is_apply = manager
                    .get(*lhs)
                    .is_some_and(|t| matches!(t.kind, TermKind::Apply { .. }));
                let rhs_is_apply = manager
                    .get(*rhs)
                    .is_some_and(|t| matches!(t.kind, TermKind::Apply { .. }));
                let (var_term, const_term) = if self.arith_terms.contains(lhs)
                    || self.bv_terms.contains(lhs)
                    || lhs_is_apply
                {
                    (*lhs, *rhs)
                } else if self.arith_terms.contains(rhs)
                    || self.bv_terms.contains(rhs)
                    || rhs_is_apply
                {
                    (*rhs, *lhs)
                } else {
                    continue;
                };

                // Check if const_term is actually a constant
                let Some(const_term_data) = manager.get(const_term) else {
                    continue;
                };

                match &const_term_data.kind {
                    TermKind::IntConst(n) => {
                        if let Some(val) = n.to_i64() {
                            let value_term = manager.mk_int(val);
                            model.set(var_term, value_term);
                        }
                    }
                    TermKind::RealConst(r) => {
                        let value_term = manager.mk_real(*r);
                        model.set(var_term, value_term);
                    }
                    TermKind::BitVecConst { value, width } => {
                        if let Some(val) = value.to_u64() {
                            let value_term = manager.mk_bitvec(val, *width);
                            model.set(var_term, value_term);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Get arithmetic values from theory solver
        // Iterate over tracked arithmetic terms
        for &term in &self.arith_terms {
            // Don't overwrite if already set (e.g., from equality extraction above)
            if model.get(term).is_some() {
                continue;
            }

            if let Some(value) = self.arith.value(term) {
                // Determine whether the term has Int or Real sort, and create the
                // matching constant kind.  Using the term sort (rather than the
                // denominator of the rational value) is essential: a Real-sorted
                // term whose arith model value happens to be an integer ratio (e.g.
                // 2/1) must be represented as RealConst(2), not IntConst(2).  If
                // stored as IntConst, mixed-type comparisons like (f(c) <= 1.0)
                // become symbolic because eval_le requires both sides to be the
                // same constant kind, preventing counterexample detection.
                let is_int_sort = manager
                    .get(term)
                    .map(|t| t.sort == manager.sorts.int_sort)
                    .unwrap_or(true);
                let value_term = if is_int_sort {
                    // Integer-sorted term: convert to BigInt
                    manager.mk_int(*value.numer())
                } else {
                    // Real-sorted term: always use RealConst regardless of denominator
                    manager.mk_real(value)
                };
                model.set(term, value_term);
            } else {
                // If no value from ArithSolver (e.g., unconstrained variable), use default
                // Get the sort to determine if it's Int or Real
                let is_int = manager
                    .get(term)
                    .map(|t| t.sort == manager.sorts.int_sort)
                    .unwrap_or(true);

                let value_term = if is_int {
                    manager.mk_int(0i64)
                } else {
                    manager.mk_real(num_rational::Rational64::from_integer(0))
                };
                model.set(term, value_term);
            }
        }

        // Get bitvector values - check ArithSolver first (for BV comparisons),
        // then BvSolver (for BV arithmetic/bit operations)
        for &term in &self.bv_terms {
            // Don't overwrite if already set (shouldn't happen, but be safe)
            if model.get(term).is_some() {
                continue;
            }

            // Get the bitvector width from the term's sort
            let width = manager
                .get(term)
                .and_then(|t| manager.sorts.get(t.sort))
                .and_then(|s| s.bitvec_width())
                .unwrap_or(64);

            // For BV comparisons handled as bounded integer arithmetic,
            // check ArithSolver FIRST (it has the actual constraint values)
            if let Some(arith_value) = self.arith.value(term) {
                let int_value = arith_value.to_integer();
                let value_term = manager.mk_bitvec(int_value, width);
                model.set(term, value_term);
            } else if let Some(bv_value) = self.bv.get_value(term) {
                // For BV bit operations, get value from BvSolver
                let value_term = manager.mk_bitvec(bv_value, width);
                model.set(term, value_term);
            } else {
                // If no value from either solver, use default value (0)
                // This handles unconstrained BV variables
                let value_term = manager.mk_bitvec(0i64, width);
                model.set(term, value_term);
            }
        }

        self.model = Some(model);
    }

    /// Build unsat core for trivial conflicts (assertion of false)
    pub(super) fn build_unsat_core_trivial_false(&mut self) {
        if !self.produce_unsat_cores {
            self.unsat_core = None;
            return;
        }

        // Find all assertions that are trivially false
        let mut core = UnsatCore::new();

        for (i, &term) in self.assertions.iter().enumerate() {
            if term == TermId::new(1) {
                // This is a false assertion
                core.indices.push(i as u32);

                // Find the name if there is one
                if let Some(named) = self.named_assertions.iter().find(|na| na.index == i as u32)
                    && let Some(ref name) = named.name
                {
                    core.names.push(name.clone());
                }
            }
        }

        self.unsat_core = Some(core);
    }

    /// Build unsat core from SAT solver conflict analysis
    pub(super) fn build_unsat_core(&mut self) {
        if !self.produce_unsat_cores {
            self.unsat_core = None;
            return;
        }

        // Build unsat core from the named assertions
        // In assumption-based mode, we would use the failed assumptions from the SAT solver
        // For now, we use a heuristic approach based on the conflict analysis

        let mut core = UnsatCore::new();

        // If assumption_vars is populated, we can use assumption-based extraction
        if !self.assumption_vars.is_empty() {
            // Assumption-based core extraction
            // Get the failed assumptions from the SAT solver
            // Note: This requires SAT solver support for assumption tracking
            // For now, include all named assertions as a conservative approach
            for na in &self.named_assertions {
                core.indices.push(na.index);
                if let Some(ref name) = na.name {
                    core.names.push(name.clone());
                }
            }
        } else {
            // Fallback: include all named assertions
            // This provides a valid unsat core, though not necessarily minimal
            for na in &self.named_assertions {
                core.indices.push(na.index);
                if let Some(ref name) = na.name {
                    core.names.push(name.clone());
                }
            }
        }

        self.unsat_core = Some(core);
    }
}
