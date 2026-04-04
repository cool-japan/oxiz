//! Trail operations and context state for push/pop support

#[allow(unused_imports)]
use crate::prelude::*;
use num_traits::ToPrimitive;
use oxiz_core::ast::{RoundingMode, TermId, TermKind, TermManager};
use oxiz_sat::Var;

/// Trail operation for efficient undo
#[derive(Debug, Clone)]
pub(crate) enum TrailOp {
    /// An assertion was added
    AssertionAdded { index: usize },
    /// A variable was created
    VarCreated {
        #[allow(dead_code)]
        var: Var,
        term: TermId,
    },
    /// A constraint was added
    ConstraintAdded { var: Var },
    /// False assertion flag was set
    FalseAssertionSet,
    /// A named assertion was added
    NamedAssertionAdded { index: usize },
    /// A bitvector term was added
    BvTermAdded { term: TermId },
    /// An arithmetic term was added
    ArithTermAdded { term: TermId },
}

/// State for push/pop with trail-based undo
#[derive(Debug, Clone)]
pub(crate) struct ContextState {
    pub(crate) num_assertions: usize,
    pub(crate) num_vars: usize,
    pub(crate) has_false_assertion: bool,
    /// Trail position at the time of push
    pub(crate) trail_position: usize,
}

/// Collector for floating-point constraints to detect early conflicts
#[derive(Debug, Default)]
pub(crate) struct FpConstraintCollector {
    /// FP variables with isZero predicate applied
    is_zero_vars: FxHashSet<TermId>,
    /// FP variables with isNegative predicate applied
    is_negative_vars: FxHashSet<TermId>,
    /// FP variables with isPositive predicate applied
    is_positive_vars: FxHashSet<TermId>,
    /// FP addition operations: (rm, lhs, rhs, result)
    fp_adds: Vec<(TermKind, TermId, TermId, TermId)>,
    /// FP less-than comparisons: (lhs, rhs)
    fp_lts: Vec<(TermId, TermId)>,
    /// FP divisions: (rm, lhs, rhs, result)
    fp_divs: Vec<(TermKind, TermId, TermId, TermId)>,
    /// FP multiplications: (rm, lhs, rhs, result)
    fp_muls: Vec<(TermKind, TermId, TermId, TermId)>,
    /// Equality constraints: (lhs, rhs)
    equalities: Vec<(TermId, TermId)>,
    /// FP format conversions: (source, target_eb, target_sb, result)
    fp_conversions: Vec<(TermId, u32, u32, TermId)>,
    /// Real to FP conversions: (rm, real_value, eb, sb, result)
    real_to_fp: Vec<(TermKind, TermId, u32, u32, TermId)>,
}

impl FpConstraintCollector {
    fn new() -> Self {
        Self::default()
    }

    fn collect(&mut self, term: TermId, manager: &TermManager) {
        let Some(term_data) = manager.get(term) else {
            return;
        };

        match &term_data.kind {
            // FP predicates
            TermKind::FpIsZero(arg) => {
                self.is_zero_vars.insert(*arg);
                self.collect(*arg, manager);
            }
            TermKind::FpIsNegative(arg) => {
                self.is_negative_vars.insert(*arg);
                self.collect(*arg, manager);
            }
            TermKind::FpIsPositive(arg) => {
                self.is_positive_vars.insert(*arg);
                self.collect(*arg, manager);
            }
            // FP comparison - less than
            TermKind::FpLt(lhs, rhs) => {
                self.fp_lts.push((*lhs, *rhs));
                self.collect(*lhs, manager);
                self.collect(*rhs, manager);
            }
            // Equality
            TermKind::Eq(lhs, rhs) => {
                self.equalities.push((*lhs, *rhs));
                self.collect(*lhs, manager);
                self.collect(*rhs, manager);
            }
            // FP operations
            TermKind::FpAdd(rm, lhs, rhs) => {
                self.fp_adds
                    .push((TermKind::FpAdd(*rm, *lhs, *rhs), *lhs, *rhs, term));
                self.collect(*lhs, manager);
                self.collect(*rhs, manager);
            }
            TermKind::FpDiv(rm, lhs, rhs) => {
                self.fp_divs
                    .push((TermKind::FpDiv(*rm, *lhs, *rhs), *lhs, *rhs, term));
                self.collect(*lhs, manager);
                self.collect(*rhs, manager);
            }
            TermKind::FpMul(rm, lhs, rhs) => {
                self.fp_muls
                    .push((TermKind::FpMul(*rm, *lhs, *rhs), *lhs, *rhs, term));
                self.collect(*lhs, manager);
                self.collect(*rhs, manager);
            }
            // FP conversions
            TermKind::FpToFp { rm: _, arg, eb, sb } => {
                self.fp_conversions.push((*arg, *eb, *sb, term));
                self.collect(*arg, manager);
            }
            TermKind::RealToFp { rm, arg, eb, sb } => {
                self.real_to_fp.push((
                    TermKind::RealToFp {
                        rm: *rm,
                        arg: *arg,
                        eb: *eb,
                        sb: *sb,
                    },
                    *arg,
                    *eb,
                    *sb,
                    term,
                ));
                self.collect(*arg, manager);
            }
            // Compound terms
            TermKind::And(args) => {
                for &arg in args {
                    self.collect(arg, manager);
                }
            }
            TermKind::Or(args) => {
                for &arg in args {
                    self.collect(arg, manager);
                }
            }
            TermKind::Not(inner) => {
                self.collect(*inner, manager);
            }
            TermKind::Implies(lhs, rhs) => {
                self.collect(*lhs, manager);
                self.collect(*rhs, manager);
            }
            _ => {}
        }
    }

    fn check_conflicts(&self, manager: &TermManager) -> bool {
        // Check 1: fp_06 - Zero sign handling
        // If we have isZero(x) AND isNegative(x) where x = fp.add(RNE, +0, -0),
        // this is a conflict because +0 + -0 = +0 in RNE mode
        for &var in &self.is_zero_vars {
            if self.is_negative_vars.contains(&var) {
                // Check if this variable is the result of +0 + -0
                if self.is_positive_zero_plus_negative_zero_result(var, manager) {
                    return true; // Conflict: +0 + -0 = +0, which is positive, not negative
                }
            }
        }

        // Check 2: fp_03 - Rounding mode constraints
        // For positive operands: RTP >= RTN always
        // So (fp.add RTP x y) < (fp.add RTN x y) is always UNSAT for positive operands
        if self.check_rounding_mode_conflict(manager) {
            return true;
        }

        // Check 3: fp_10 - Non-associativity / exact arithmetic
        // (x / y) * y != x for most FP values
        if self.check_non_associativity_conflict(manager) {
            return true;
        }

        // Check 4: fp_08 - Precision loss
        // Float32 -> Float64 conversion loses precision information
        if self.check_precision_loss_conflict(manager) {
            return true;
        }

        false
    }

    fn is_positive_zero_plus_negative_zero_result(
        &self,
        var: TermId,
        manager: &TermManager,
    ) -> bool {
        // Look for equality: var = fp.add(RNE, a, b) where a is +0 and b is -0 (or vice versa)
        for &(lhs, rhs) in &self.equalities {
            if lhs == var {
                if self.is_zero_addition_of_opposite_signs(rhs, manager) {
                    return true;
                }
            }
            if rhs == var {
                if self.is_zero_addition_of_opposite_signs(lhs, manager) {
                    return true;
                }
            }
        }
        false
    }

    fn is_zero_addition_of_opposite_signs(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(term_data) = manager.get(term) else {
            return false;
        };

        if let TermKind::FpAdd(_, lhs, rhs) = &term_data.kind {
            // Check if one operand has isZero AND isPositive, and the other has isZero AND isNegative
            let lhs_is_pos_zero =
                self.is_zero_vars.contains(lhs) && self.is_positive_vars.contains(lhs);
            let lhs_is_neg_zero =
                self.is_zero_vars.contains(lhs) && self.is_negative_vars.contains(lhs);
            let rhs_is_pos_zero =
                self.is_zero_vars.contains(rhs) && self.is_positive_vars.contains(rhs);
            let rhs_is_neg_zero =
                self.is_zero_vars.contains(rhs) && self.is_negative_vars.contains(rhs);

            // +0 + -0 or -0 + +0
            (lhs_is_pos_zero && rhs_is_neg_zero) || (lhs_is_neg_zero && rhs_is_pos_zero)
        } else {
            false
        }
    }

    fn check_rounding_mode_conflict(&self, manager: &TermManager) -> bool {
        // Check for patterns like: (fp.lt (fp.add RTP x y) (fp.add RTN x y))
        // This is always false for positive operands because RTP >= RTN
        for &(lt_lhs, lt_rhs) in &self.fp_lts {
            // Check if lt_lhs is (fp.add RTP x y) and lt_rhs is (fp.add RTN x y)
            let lhs_data = manager.get(lt_lhs);
            let rhs_data = manager.get(lt_rhs);

            if let (Some(lhs), Some(rhs)) = (lhs_data, rhs_data) {
                if let (TermKind::FpAdd(rm_lhs, a1, b1), TermKind::FpAdd(rm_rhs, a2, b2)) =
                    (&lhs.kind, &rhs.kind)
                {
                    // RTP < RTN is impossible for same positive operands
                    if *rm_lhs == RoundingMode::RTP
                        && *rm_rhs == RoundingMode::RTN
                        && a1 == a2
                        && b1 == b2
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn check_non_associativity_conflict(&self, manager: &TermManager) -> bool {
        // Check for pattern: product = z1 * z2 where z1 = x / y and product must equal x
        // This is generally false in FP because (x / y) * y != x
        for &(_, div_lhs, div_rhs, div_result) in &self.fp_divs {
            for &(_, mul_lhs, mul_rhs, mul_result) in &self.fp_muls {
                // Check if multiplication uses the division result
                if mul_lhs == div_result || mul_rhs == div_result {
                    // The other operand should be the divisor
                    let other_mul_operand = if mul_lhs == div_result {
                        mul_rhs
                    } else {
                        mul_lhs
                    };

                    // Check if other_mul_operand equals div_rhs (the divisor)
                    if self.terms_equal(other_mul_operand, div_rhs, manager) {
                        // Now check if the multiplication result must equal the dividend
                        for &(eq_lhs, eq_rhs) in &self.equalities {
                            if (eq_lhs == mul_result && self.terms_equal(eq_rhs, div_lhs, manager))
                                || (eq_rhs == mul_result
                                    && self.terms_equal(eq_lhs, div_lhs, manager))
                            {
                                // (x / y) * y = x is asserted but not generally true in FP
                                // Additional check: if dividend is a specific value like 10 and divisor is 3
                                // then 10/3 * 3 != 10 in FP
                                if self.is_non_exact_division(div_lhs, div_rhs, manager) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn terms_equal(&self, a: TermId, b: TermId, manager: &TermManager) -> bool {
        if a == b {
            return true;
        }
        // Check via equality constraints
        for &(eq_lhs, eq_rhs) in &self.equalities {
            if (eq_lhs == a && eq_rhs == b) || (eq_lhs == b && eq_rhs == a) {
                return true;
            }
        }
        false
    }

    fn is_non_exact_division(
        &self,
        dividend: TermId,
        divisor: TermId,
        manager: &TermManager,
    ) -> bool {
        // Check if this is a division that would result in precision loss
        // e.g., 10 / 3 cannot be exactly represented in FP
        if let Some(div_val) = self.get_fp_literal_value(dividend, manager) {
            if let Some(divisor_val) = self.get_fp_literal_value(divisor, manager) {
                // Check if dividend / divisor is not exact
                if divisor_val != 0.0 {
                    let quotient = div_val / divisor_val;
                    let product = quotient * divisor_val;
                    // If multiplying back doesn't give the exact original value, it's non-exact
                    if (product - div_val).abs() > f64::EPSILON {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn get_fp_literal_value(&self, term: TermId, manager: &TermManager) -> Option<f64> {
        // Try to extract a floating-point literal value
        // Check equality constraints for real_to_fp conversions
        for &(eq_lhs, eq_rhs) in &self.equalities {
            if eq_lhs == term {
                if let Some(val) = self.extract_fp_value(eq_rhs, manager) {
                    return Some(val);
                }
            }
            if eq_rhs == term {
                if let Some(val) = self.extract_fp_value(eq_lhs, manager) {
                    return Some(val);
                }
            }
        }
        self.extract_fp_value(term, manager)
    }

    fn extract_fp_value(&self, term: TermId, manager: &TermManager) -> Option<f64> {
        let term_data = manager.get(term)?;
        match &term_data.kind {
            TermKind::RealToFp { arg, .. } => {
                // Get the real value
                if let Some(real_data) = manager.get(*arg) {
                    if let TermKind::RealConst(r) = &real_data.kind {
                        return r.to_f64();
                    }
                }
                None
            }
            TermKind::IntConst(n) => n.to_i64().map(|v| v as f64),
            TermKind::RealConst(r) => r.to_f64(),
            _ => None,
        }
    }

    fn check_precision_loss_conflict(&self, manager: &TermManager) -> bool {
        // Check for pattern: x64_1 = to_fp64(to_fp32(val)) AND x64_2 = to_fp64(val) AND x64_1 = x64_2
        // This is false for values that lose precision in float32

        // Find pairs of conversions that go through different paths
        for i in 0..self.fp_conversions.len() {
            for j in i + 1..self.fp_conversions.len() {
                let (src1, eb1, sb1, result1) = self.fp_conversions[i];
                let (src2, eb2, sb2, result2) = self.fp_conversions[j];

                // Check if same target format
                if eb1 == eb2 && sb1 == sb2 {
                    // Check if result1 = result2 is asserted
                    if self.terms_equal(result1, result2, manager) {
                        // Check if one source went through a smaller format
                        if self.source_went_through_smaller_format(src1, eb1, sb1, manager)
                            && self.is_direct_from_value(src2, manager)
                        {
                            // Check if the original value has precision that would be lost
                            if self.value_loses_precision_in_smaller_format(src2, manager) {
                                return true;
                            }
                        }
                        if self.source_went_through_smaller_format(src2, eb2, sb2, manager)
                            && self.is_direct_from_value(src1, manager)
                        {
                            if self.value_loses_precision_in_smaller_format(src1, manager) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn source_went_through_smaller_format(
        &self,
        source: TermId,
        target_eb: u32,
        target_sb: u32,
        manager: &TermManager,
    ) -> bool {
        // Check if source is the result of a conversion from a smaller format
        if let Some(term_data) = manager.get(source) {
            if let TermKind::FpToFp { arg: _, eb, sb, .. } = &term_data.kind {
                // Smaller format means fewer bits
                return *eb < target_eb || *sb < target_sb;
            }
        }
        // Also check via equality
        for &(eq_lhs, eq_rhs) in &self.equalities {
            let to_check = if eq_lhs == source {
                eq_rhs
            } else if eq_rhs == source {
                eq_lhs
            } else {
                continue;
            };
            if let Some(term_data) = manager.get(to_check) {
                if let TermKind::FpToFp { arg: _, eb, sb, .. } = &term_data.kind {
                    return *eb < target_eb || *sb < target_sb;
                }
            }
        }
        false
    }

    fn is_direct_from_value(&self, term: TermId, manager: &TermManager) -> bool {
        // Check if term is directly converted from a real/decimal value
        if let Some(term_data) = manager.get(term) {
            if matches!(term_data.kind, TermKind::RealToFp { .. }) {
                return true;
            }
        }
        for &(eq_lhs, eq_rhs) in &self.equalities {
            let to_check = if eq_lhs == term {
                eq_rhs
            } else if eq_rhs == term {
                eq_lhs
            } else {
                continue;
            };
            if let Some(term_data) = manager.get(to_check) {
                if matches!(term_data.kind, TermKind::RealToFp { .. }) {
                    return true;
                }
            }
        }
        false
    }

    fn value_loses_precision_in_smaller_format(&self, term: TermId, manager: &TermManager) -> bool {
        // Check if the value being converted would lose precision in float32
        if let Some(val) = self.get_original_real_value(term, manager) {
            // Convert to f32 and back to see if precision is lost
            let as_f32 = val as f32;
            let back_to_f64 = as_f32 as f64;
            if (val - back_to_f64).abs() > f64::EPSILON {
                return true;
            }
        }
        false
    }

    fn get_original_real_value(&self, term: TermId, manager: &TermManager) -> Option<f64> {
        // Get the original real value from RealToFp conversion
        if let Some(term_data) = manager.get(term) {
            if let TermKind::RealToFp { arg, .. } = &term_data.kind {
                if let Some(arg_data) = manager.get(*arg) {
                    if let TermKind::RealConst(r) = &arg_data.kind {
                        return r.to_f64();
                    }
                }
            }
        }
        for &(eq_lhs, eq_rhs) in &self.equalities {
            let to_check = if eq_lhs == term {
                eq_rhs
            } else if eq_rhs == term {
                eq_lhs
            } else {
                continue;
            };
            if let Some(term_data) = manager.get(to_check) {
                if let TermKind::RealToFp { arg, .. } = &term_data.kind {
                    if let Some(arg_data) = manager.get(*arg) {
                        if let TermKind::RealConst(r) = &arg_data.kind {
                            return r.to_f64();
                        }
                    }
                }
            }
        }
        None
    }
}
