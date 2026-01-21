//! Bit-vector blasting tactics.

use super::core::*;
use crate::ast::{TermId, TermManager};
use crate::error::Result;

/// Bit-blasting tactic - converts BV operations to propositional logic
pub struct BitBlastTactic<'a> {
    manager: &'a TermManager,
}

impl<'a> BitBlastTactic<'a> {
    /// Create a new bit-blast tactic
    pub fn new(manager: &'a TermManager) -> Self {
        Self { manager }
    }

    /// Check if a term is a BitVector term
    fn is_bv_term(&self, term_id: TermId) -> bool {
        use crate::ast::TermKind;
        if let Some(term) = self.manager.get(term_id) {
            matches!(
                term.kind,
                TermKind::BitVecConst { .. }
                    | TermKind::BvConcat(_, _)
                    | TermKind::BvExtract { .. }
                    | TermKind::BvNot(_)
                    | TermKind::BvAnd(_, _)
                    | TermKind::BvOr(_, _)
                    | TermKind::BvXor(_, _)
                    | TermKind::BvAdd(_, _)
                    | TermKind::BvSub(_, _)
                    | TermKind::BvMul(_, _)
                    | TermKind::BvUdiv(_, _)
                    | TermKind::BvSdiv(_, _)
                    | TermKind::BvUrem(_, _)
                    | TermKind::BvSrem(_, _)
                    | TermKind::BvShl(_, _)
                    | TermKind::BvLshr(_, _)
                    | TermKind::BvAshr(_, _)
                    | TermKind::BvUlt(_, _)
                    | TermKind::BvUle(_, _)
                    | TermKind::BvSlt(_, _)
                    | TermKind::BvSle(_, _)
            ) || self.is_bv_sort(term.sort)
        } else {
            false
        }
    }

    /// Check if a sort is a BitVector sort
    fn is_bv_sort(&self, sort_id: crate::sort::SortId) -> bool {
        if let Some(sort) = self.manager.sorts.get(sort_id) {
            sort.bitvec_width().is_some()
        } else {
            false
        }
    }

    /// Check if a term contains any BitVector subterms
    fn contains_bv_term(&self, term_id: TermId) -> bool {
        use crate::ast::TermKind;

        if self.is_bv_term(term_id) {
            return true;
        }

        if let Some(term) = self.manager.get(term_id) {
            match &term.kind {
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::Var(_) => self.is_bv_sort(term.sort),
                TermKind::Not(a) | TermKind::Neg(a) | TermKind::BvNot(a) => {
                    self.contains_bv_term(*a)
                }
                TermKind::BvExtract { arg, .. } => self.contains_bv_term(*arg),
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::Distinct(args) => args.iter().any(|&a| self.contains_bv_term(a)),
                TermKind::StringLit(_)
                | TermKind::StrLen(_)
                | TermKind::StrToInt(_)
                | TermKind::IntToStr(_) => false,
                TermKind::Implies(a, b)
                | TermKind::Xor(a, b)
                | TermKind::Eq(a, b)
                | TermKind::Sub(a, b)
                | TermKind::Div(a, b)
                | TermKind::Mod(a, b)
                | TermKind::Lt(a, b)
                | TermKind::Le(a, b)
                | TermKind::Gt(a, b)
                | TermKind::Ge(a, b)
                | TermKind::Select(a, b)
                | TermKind::StrConcat(a, b)
                | TermKind::StrAt(a, b)
                | TermKind::StrContains(a, b)
                | TermKind::StrPrefixOf(a, b)
                | TermKind::StrSuffixOf(a, b)
                | TermKind::StrInRe(a, b)
                | TermKind::BvConcat(a, b)
                | TermKind::BvAnd(a, b)
                | TermKind::BvOr(a, b)
                | TermKind::BvXor(a, b)
                | TermKind::BvAdd(a, b)
                | TermKind::BvSub(a, b)
                | TermKind::BvMul(a, b)
                | TermKind::BvUdiv(a, b)
                | TermKind::BvSdiv(a, b)
                | TermKind::BvUrem(a, b)
                | TermKind::BvSrem(a, b)
                | TermKind::BvShl(a, b)
                | TermKind::BvLshr(a, b)
                | TermKind::BvAshr(a, b)
                | TermKind::BvUlt(a, b)
                | TermKind::BvUle(a, b)
                | TermKind::BvSlt(a, b)
                | TermKind::BvSle(a, b) => self.contains_bv_term(*a) || self.contains_bv_term(*b),
                TermKind::Ite(c, t, e)
                | TermKind::Store(c, t, e)
                | TermKind::StrSubstr(c, t, e)
                | TermKind::StrIndexOf(c, t, e)
                | TermKind::StrReplace(c, t, e)
                | TermKind::StrReplaceAll(c, t, e) => {
                    self.contains_bv_term(*c)
                        || self.contains_bv_term(*t)
                        || self.contains_bv_term(*e)
                }
                TermKind::Apply { args, .. } => args.iter().any(|&a| self.contains_bv_term(a)),
                TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
                    self.contains_bv_term(*body)
                }
                TermKind::Let { bindings, body } => {
                    bindings.iter().any(|(_, t)| self.contains_bv_term(*t))
                        || self.contains_bv_term(*body)
                }
                // Floating-point operations don't contain BV terms
                TermKind::FpLit { .. }
                | TermKind::FpPlusInfinity { .. }
                | TermKind::FpMinusInfinity { .. }
                | TermKind::FpPlusZero { .. }
                | TermKind::FpMinusZero { .. }
                | TermKind::FpNaN { .. } => false,
                TermKind::FpAbs(a)
                | TermKind::FpNeg(a)
                | TermKind::FpSqrt(_, a)
                | TermKind::FpRoundToIntegral(_, a)
                | TermKind::FpIsNormal(a)
                | TermKind::FpIsSubnormal(a)
                | TermKind::FpIsZero(a)
                | TermKind::FpIsInfinite(a)
                | TermKind::FpIsNaN(a)
                | TermKind::FpIsNegative(a)
                | TermKind::FpIsPositive(a)
                | TermKind::FpToReal(a) => self.contains_bv_term(*a),
                TermKind::FpAdd(_, a, b)
                | TermKind::FpSub(_, a, b)
                | TermKind::FpMul(_, a, b)
                | TermKind::FpDiv(_, a, b)
                | TermKind::FpRem(a, b)
                | TermKind::FpMin(a, b)
                | TermKind::FpMax(a, b)
                | TermKind::FpLeq(a, b)
                | TermKind::FpLt(a, b)
                | TermKind::FpGeq(a, b)
                | TermKind::FpGt(a, b)
                | TermKind::FpEq(a, b) => self.contains_bv_term(*a) || self.contains_bv_term(*b),
                TermKind::FpFma(_, a, b, c) => {
                    self.contains_bv_term(*a)
                        || self.contains_bv_term(*b)
                        || self.contains_bv_term(*c)
                }
                TermKind::FpToFp { arg, .. }
                | TermKind::FpToSBV { arg, .. }
                | TermKind::FpToUBV { arg, .. }
                | TermKind::RealToFp { arg, .. }
                | TermKind::SBVToFp { arg, .. }
                | TermKind::UBVToFp { arg, .. } => self.contains_bv_term(*arg),
                // Algebraic datatypes
                TermKind::DtConstructor { args, .. } => {
                    args.iter().any(|&a| self.contains_bv_term(a))
                }
                TermKind::DtTester { arg, .. } | TermKind::DtSelector { arg, .. } => {
                    self.contains_bv_term(*arg)
                }
                // Match expressions
                TermKind::Match { scrutinee, cases } => {
                    self.contains_bv_term(*scrutinee)
                        || cases.iter().any(|c| self.contains_bv_term(c.body))
                }
            }
        } else {
            false
        }
    }

    /// Apply bit-blasting to a goal
    ///
    /// Currently, this returns a marker indicating the goal contains BV terms
    /// and should be solved by the BV theory solver. Full bit-blasting to
    /// pure Boolean logic would be implemented here for complete integration.
    pub fn apply_check(&self, goal: &Goal) -> Result<TacticResult> {
        // Check if any assertion contains BitVector terms
        let has_bv = goal.assertions.iter().any(|&a| self.contains_bv_term(a));

        if !has_bv {
            return Ok(TacticResult::NotApplicable);
        }

        // For now, we just mark that this goal needs BV solving
        // A full implementation would:
        // 1. Create Boolean variables for each bit of each BV variable
        // 2. Encode BV operations as Boolean circuits
        // 3. Return a goal with only Boolean constraints

        // Return the goal unchanged - the BV solver will handle it
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessBitBlastTactic;

impl Tactic for StatelessBitBlastTactic {
    fn name(&self) -> &str {
        "bit-blast"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Converts BitVector operations to propositional logic"
    }
}
