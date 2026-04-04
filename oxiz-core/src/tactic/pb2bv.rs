//! Pseudo-Boolean to Bit-Vector tactic.

use super::core::*;
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;

/// Pseudo-Boolean to Bit-Vector tactic
///
/// This tactic converts pseudo-boolean constraints (linear combinations of
/// booleans with integer coefficients) into bit-vector arithmetic.
///
/// # Example
///
/// `2*x + 3*y + z <= 5` where x, y, z are booleans
///
/// becomes a bit-vector constraint using:
/// - Each boolean as a 1-bit BV (or zero-extended)
/// - Integer coefficients as BV constants
/// - Addition and comparison in BV arithmetic
///
/// # Reference
///
/// Based on Z3's `pb2bv_tactic` in `src/tactic/arith/pb2bv_tactic.cpp`
#[derive(Debug)]
pub struct Pb2BvTactic<'a> {
    manager: &'a mut TermManager,
    /// Bit width for intermediate results (auto-computed or specified)
    bit_width: Option<u32>,
}

/// A term in a pseudo-boolean constraint: coefficient * boolean_var
#[derive(Debug, Clone)]
struct PbTerm {
    /// The coefficient (positive or negative)
    coefficient: i64,
    /// The boolean variable
    var: TermId,
}

/// A pseudo-boolean constraint
#[derive(Debug)]
struct PbConstraint {
    /// Linear combination of boolean variables
    terms: Vec<PbTerm>,
    /// Constant term (right-hand side)
    bound: i64,
    /// Constraint type: true for <=, false for =
    is_le: bool,
}

impl<'a> Pb2BvTactic<'a> {
    /// Create a new PB to BV tactic with auto bit-width
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            bit_width: None,
        }
    }

    /// Create with explicit bit width
    pub fn with_bit_width(manager: &'a mut TermManager, width: u32) -> Self {
        Self {
            manager,
            bit_width: Some(width),
        }
    }

    /// Apply the tactic mutably
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut new_assertions = Vec::new();
        let mut changed = false;

        for &assertion in &goal.assertions {
            if let Some(converted) = self.convert_constraint(assertion) {
                new_assertions.push(converted);
                changed = true;
            } else {
                new_assertions.push(assertion);
            }
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Try to convert a constraint to bit-vector form
    fn convert_constraint(&mut self, term: TermId) -> Option<TermId> {
        let pb = self.extract_pb_constraint(term)?;

        // Compute required bit width
        let width = self.compute_bit_width(&pb);

        // Convert to bit-vector constraint
        self.encode_pb_as_bv(&pb, width)
    }

    /// Extract a PB constraint from a term
    fn extract_pb_constraint(&self, term: TermId) -> Option<PbConstraint> {
        let t = self.manager.get(term)?;

        match &t.kind {
            TermKind::Le(lhs, rhs) => {
                // lhs <= rhs
                // Convert to: lhs - rhs <= 0, then to: lhs <= rhs
                let (terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                Some(PbConstraint {
                    terms,
                    bound: rhs_val,
                    is_le: true,
                })
            }
            TermKind::Ge(lhs, rhs) => {
                // lhs >= rhs => -lhs <= -rhs => rhs <= lhs
                // Convert to: -lhs + rhs <= 0
                let (mut terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                // Negate all coefficients
                for term in &mut terms {
                    term.coefficient = -term.coefficient;
                }

                Some(PbConstraint {
                    terms,
                    bound: -rhs_val,
                    is_le: true,
                })
            }
            TermKind::Lt(lhs, rhs) => {
                // lhs < rhs => lhs <= rhs - 1
                let (terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                Some(PbConstraint {
                    terms,
                    bound: rhs_val - 1,
                    is_le: true,
                })
            }
            TermKind::Gt(lhs, rhs) => {
                // lhs > rhs => rhs < lhs => rhs <= lhs - 1
                let (mut terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                for term in &mut terms {
                    term.coefficient = -term.coefficient;
                }

                Some(PbConstraint {
                    terms,
                    bound: -rhs_val - 1,
                    is_le: true,
                })
            }
            TermKind::Eq(lhs, rhs) => {
                let (terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                Some(PbConstraint {
                    terms,
                    bound: rhs_val,
                    is_le: false, // equality
                })
            }
            _ => None,
        }
    }

    /// Extract a linear combination of boolean variables
    /// Returns (terms, constant) where the expression is Σ(coeff * var) + constant
    fn extract_linear_bool_comb(&self, term: TermId) -> Option<(Vec<PbTerm>, i64)> {
        let t = self.manager.get(term)?;

        match &t.kind {
            TermKind::Add(args) => {
                let mut all_terms = Vec::new();
                let mut total_const = 0i64;

                for &arg in args.iter() {
                    if let Some((terms, c)) = self.extract_linear_bool_comb(arg) {
                        all_terms.extend(terms);
                        total_const += c;
                    } else {
                        return None;
                    }
                }

                Some((all_terms, total_const))
            }
            TermKind::Mul(args) if args.len() == 2 => {
                // coeff * var or var * coeff
                let first = self.manager.get(args[0])?;
                let second = self.manager.get(args[1])?;

                if let TermKind::IntConst(c) = &first.kind {
                    // c * var
                    if self.is_boolean_term(args[1]) {
                        let coeff = c.try_into().ok()?;
                        return Some((
                            vec![PbTerm {
                                coefficient: coeff,
                                var: args[1],
                            }],
                            0,
                        ));
                    }
                }

                if let TermKind::IntConst(c) = &second.kind {
                    // var * c
                    if self.is_boolean_term(args[0]) {
                        let coeff = c.try_into().ok()?;
                        return Some((
                            vec![PbTerm {
                                coefficient: coeff,
                                var: args[0],
                            }],
                            0,
                        ));
                    }
                }

                None
            }
            TermKind::IntConst(c) => {
                let val = c.try_into().ok()?;
                Some((Vec::new(), val))
            }
            TermKind::Ite(cond, then_br, else_br) => {
                // if cond then 1 else 0 (common pattern for bool-to-int)
                let then_t = self.manager.get(*then_br)?;
                let else_t = self.manager.get(*else_br)?;

                if matches!(then_t.kind, TermKind::IntConst(ref v) if *v == 1.into())
                    && matches!(else_t.kind, TermKind::IntConst(ref v) if *v == 0.into())
                {
                    // This is (ite bool 1 0), treat as bool with coefficient 1
                    Some((
                        vec![PbTerm {
                            coefficient: 1,
                            var: *cond,
                        }],
                        0,
                    ))
                } else {
                    None
                }
            }
            _ => {
                // Check if it's a standalone boolean variable
                if self.is_boolean_term(term) {
                    Some((
                        vec![PbTerm {
                            coefficient: 1,
                            var: term,
                        }],
                        0,
                    ))
                } else {
                    None
                }
            }
        }
    }

    /// Check if a term is a boolean
    fn is_boolean_term(&self, term: TermId) -> bool {
        if let Some(t) = self.manager.get(term) {
            t.sort == self.manager.sorts.bool_sort
        } else {
            false
        }
    }

    /// Extract an integer constant
    fn extract_int_const(&self, term: TermId) -> Option<i64> {
        let t = self.manager.get(term)?;
        if let TermKind::IntConst(c) = &t.kind {
            c.try_into().ok()
        } else {
            None
        }
    }

    /// Compute the required bit width for a PB constraint
    fn compute_bit_width(&self, pb: &PbConstraint) -> u32 {
        if let Some(w) = self.bit_width {
            return w;
        }

        // Compute the maximum possible sum
        let mut max_sum: i64 = 0;
        for term in &pb.terms {
            max_sum += term.coefficient.abs();
        }
        max_sum = max_sum.max(pb.bound.abs());

        // Compute bits needed (including sign bit for safety)
        let bits_needed = if max_sum == 0 {
            1
        } else {
            (64 - max_sum.leading_zeros()).max(1) + 1
        };

        bits_needed.min(64) // Cap at 64 bits
    }

    /// Encode a PB constraint as bit-vector arithmetic
    fn encode_pb_as_bv(&mut self, pb: &PbConstraint, width: u32) -> Option<TermId> {
        let _bv_sort = self.manager.sorts.bitvec(width);

        // Build the sum: Σ(coeff * bool_to_bv(var))
        let mut sum_terms: Vec<TermId> = Vec::new();

        for term in &pb.terms {
            // Convert boolean to BV: (ite var 1bv 0bv)
            let bv_one = self.manager.mk_bitvec(1u64, width);
            let bv_zero = self.manager.mk_bitvec(0u64, width);
            let var_bv = self.manager.mk_ite(term.var, bv_one, bv_zero);

            // Multiply by coefficient
            let coeff_bv = if term.coefficient >= 0 {
                self.manager.mk_bitvec(term.coefficient as u64, width)
            } else {
                // Negative coefficient: use two's complement
                let abs_coeff = self.manager.mk_bitvec((-term.coefficient) as u64, width);
                self.manager.mk_bv_neg(abs_coeff)
            };

            let prod = self.manager.mk_bv_mul(coeff_bv, var_bv);
            sum_terms.push(prod);
        }

        // Sum all terms
        let sum = if sum_terms.is_empty() {
            self.manager.mk_bitvec(0u64, width)
        } else if sum_terms.len() == 1 {
            sum_terms[0]
        } else {
            let mut acc = sum_terms[0];
            for &term in &sum_terms[1..] {
                acc = self.manager.mk_bv_add(acc, term);
            }
            acc
        };

        // Create the bound as BV
        let bound_bv = if pb.bound >= 0 {
            self.manager.mk_bitvec(pb.bound as u64, width)
        } else {
            let abs_bound = self.manager.mk_bitvec((-pb.bound) as u64, width);
            self.manager.mk_bv_neg(abs_bound)
        };

        // Create the comparison
        if pb.is_le {
            // sum <= bound (signed comparison)
            Some(self.manager.mk_bv_sle(sum, bound_bv))
        } else {
            // sum = bound
            Some(self.manager.mk_eq(sum, bound_bv))
        }
    }
}

/// Stateless wrapper for PB2BV tactic
#[derive(Debug, Default, Clone, Copy)]
pub struct StatelessPb2BvTactic;

impl Tactic for StatelessPb2BvTactic {
    fn name(&self) -> &str {
        "pb2bv"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![(*goal).clone()]))
    }

    fn description(&self) -> &str {
        "Convert pseudo-boolean constraints to bit-vector arithmetic"
    }
}
