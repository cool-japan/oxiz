//! Sort inference for terms
//!
//! This module provides functionality to infer the sort (type) of terms
//! based on their structure and the sorts of their subterms.

use crate::ast::{Term, TermId, TermKind, TermManager};
use crate::error::{OxizError, Result, SourceSpan};
use crate::sort::{SortId, SortKind, SortManager};

/// Infer the sort of a term based on its structure
///
/// This function examines the term's kind and the sorts of its children
/// to determine what sort the term should have.
pub fn infer_term_sort(term: &Term, manager: &TermManager) -> Result<SortId> {
    match &term.kind {
        // Constants have known sorts
        TermKind::True | TermKind::False => Ok(manager.sorts.bool_sort),
        TermKind::IntConst(_) => Ok(manager.sorts.int_sort),
        TermKind::RealConst(_) => Ok(manager.sorts.real_sort),
        TermKind::BitVecConst { .. } | TermKind::StringLit(_) => Ok(term.sort),

        // Variables already have assigned sorts
        TermKind::Var(_) => Ok(term.sort),

        // Boolean operations return Bool
        TermKind::Not(_)
        | TermKind::And(_)
        | TermKind::Or(_)
        | TermKind::Implies(_, _)
        | TermKind::Xor(_, _) => Ok(manager.sorts.bool_sort),

        // Comparisons return Bool
        TermKind::Eq(_, _)
        | TermKind::Distinct(_)
        | TermKind::Lt(_, _)
        | TermKind::Le(_, _)
        | TermKind::Gt(_, _)
        | TermKind::Ge(_, _) => Ok(manager.sorts.bool_sort),

        // String operations that return Bool
        TermKind::StrContains(_, _)
        | TermKind::StrPrefixOf(_, _)
        | TermKind::StrSuffixOf(_, _)
        | TermKind::StrInRe(_, _) => Ok(manager.sorts.bool_sort),

        // String operations that return Int
        TermKind::StrLen(_) | TermKind::StrToInt(_) | TermKind::StrIndexOf(_, _, _) => {
            Ok(manager.sorts.int_sort)
        }

        // String operations that return String
        TermKind::StrConcat(_, _)
        | TermKind::StrAt(_, _)
        | TermKind::StrSubstr(_, _, _)
        | TermKind::StrReplace(_, _, _)
        | TermKind::StrReplaceAll(_, _, _)
        | TermKind::IntToStr(_) => Ok(term.sort),

        // Arithmetic operations inherit sort from operands
        TermKind::Add(args) | TermKind::Mul(args) => {
            if args.is_empty() {
                return Ok(manager.sorts.int_sort);
            }
            infer_arithmetic_sort(args[0], manager)
        }

        TermKind::Sub(lhs, _) | TermKind::Div(lhs, _) | TermKind::Mod(lhs, _) => {
            infer_arithmetic_sort(*lhs, manager)
        }

        TermKind::Neg(arg) => infer_arithmetic_sort(*arg, manager),

        // ITE inherits sort from branches
        TermKind::Ite(_, then_branch, _) => {
            if let Some(then_term) = manager.get(*then_branch) {
                Ok(then_term.sort)
            } else {
                Err(OxizError::Internal(
                    "ITE then-branch term not found".to_string(),
                ))
            }
        }

        // Bit-vector operations
        TermKind::BvNot(arg)
        | TermKind::BvAnd(arg, _)
        | TermKind::BvOr(arg, _)
        | TermKind::BvXor(arg, _)
        | TermKind::BvAdd(arg, _)
        | TermKind::BvSub(arg, _)
        | TermKind::BvMul(arg, _)
        | TermKind::BvUdiv(arg, _)
        | TermKind::BvSdiv(arg, _)
        | TermKind::BvUrem(arg, _)
        | TermKind::BvSrem(arg, _)
        | TermKind::BvShl(arg, _)
        | TermKind::BvLshr(arg, _)
        | TermKind::BvAshr(arg, _)
        | TermKind::BvConcat(arg, _) => {
            if let Some(arg_term) = manager.get(*arg) {
                Ok(arg_term.sort)
            } else {
                Err(OxizError::Internal("BV operand term not found".to_string()))
            }
        }

        // Bit-vector extract returns stored sort
        TermKind::BvExtract { .. } => Ok(term.sort),

        // Bit-vector comparisons return Bool
        TermKind::BvUlt(_, _)
        | TermKind::BvUle(_, _)
        | TermKind::BvSlt(_, _)
        | TermKind::BvSle(_, _) => Ok(manager.sorts.bool_sort),

        // Array operations
        TermKind::Select(array, _) => {
            if let Some(array_term) = manager.get(*array)
                && let Some(sort) = manager.sorts.get(array_term.sort)
                && let SortKind::Array { range, .. } = sort.kind
            {
                return Ok(range);
            }
            Err(OxizError::Internal(
                "Cannot infer sort for array select".to_string(),
            ))
        }

        TermKind::Store(array, _, _) => {
            if let Some(array_term) = manager.get(*array) {
                Ok(array_term.sort)
            } else {
                Err(OxizError::Internal("Array term not found".to_string()))
            }
        }

        // Function applications - use stored sort
        TermKind::Apply { .. } => Ok(term.sort),

        // Quantifiers return Bool
        TermKind::Forall { .. } | TermKind::Exists { .. } => Ok(manager.sorts.bool_sort),

        // Let expressions inherit sort from body
        TermKind::Let { body, .. } => {
            if let Some(body_term) = manager.get(*body) {
                Ok(body_term.sort)
            } else {
                Err(OxizError::Internal("Let body term not found".to_string()))
            }
        }

        // Floating-point literals and special values - use stored sort
        TermKind::FpLit { .. }
        | TermKind::FpPlusInfinity { .. }
        | TermKind::FpMinusInfinity { .. }
        | TermKind::FpPlusZero { .. }
        | TermKind::FpMinusZero { .. }
        | TermKind::FpNaN { .. } => Ok(term.sort),

        // FP unary operations that preserve FP sort
        TermKind::FpAbs(arg)
        | TermKind::FpNeg(arg)
        | TermKind::FpSqrt(_, arg)
        | TermKind::FpRoundToIntegral(_, arg) => {
            if let Some(arg_term) = manager.get(*arg) {
                Ok(arg_term.sort)
            } else {
                Err(OxizError::Internal("FP operand term not found".to_string()))
            }
        }

        // FP binary operations that preserve FP sort
        TermKind::FpAdd(_, lhs, _)
        | TermKind::FpSub(_, lhs, _)
        | TermKind::FpMul(_, lhs, _)
        | TermKind::FpDiv(_, lhs, _)
        | TermKind::FpRem(lhs, _)
        | TermKind::FpMin(lhs, _)
        | TermKind::FpMax(lhs, _) => {
            if let Some(lhs_term) = manager.get(*lhs) {
                Ok(lhs_term.sort)
            } else {
                Err(OxizError::Internal("FP operand term not found".to_string()))
            }
        }

        // FP ternary operations (FMA) that preserve FP sort
        TermKind::FpFma(_, x, _, _) => {
            if let Some(x_term) = manager.get(*x) {
                Ok(x_term.sort)
            } else {
                Err(OxizError::Internal("FP operand term not found".to_string()))
            }
        }

        // FP comparisons return Bool
        TermKind::FpLeq(_, _)
        | TermKind::FpLt(_, _)
        | TermKind::FpGeq(_, _)
        | TermKind::FpGt(_, _)
        | TermKind::FpEq(_, _) => Ok(manager.sorts.bool_sort),

        // FP predicates return Bool
        TermKind::FpIsNormal(_)
        | TermKind::FpIsSubnormal(_)
        | TermKind::FpIsZero(_)
        | TermKind::FpIsInfinite(_)
        | TermKind::FpIsNaN(_)
        | TermKind::FpIsNegative(_)
        | TermKind::FpIsPositive(_) => Ok(manager.sorts.bool_sort),

        // FP conversions - use stored sort
        TermKind::FpToFp { .. }
        | TermKind::RealToFp { .. }
        | TermKind::SBVToFp { .. }
        | TermKind::UBVToFp { .. } => Ok(term.sort),

        // FP to other types
        TermKind::FpToReal(_) => Ok(manager.sorts.real_sort),
        TermKind::FpToSBV { .. } | TermKind::FpToUBV { .. } => Ok(term.sort),

        // Algebraic datatypes - use stored sort
        TermKind::DtConstructor { .. } => Ok(term.sort),
        TermKind::DtTester { .. } => Ok(manager.sorts.bool_sort),
        TermKind::DtSelector { .. } => Ok(term.sort),

        // Match expressions - use stored sort (inferred from case bodies)
        TermKind::Match { .. } => Ok(term.sort),
    }
}

/// Infer the sort of an arithmetic operation
fn infer_arithmetic_sort(arg: TermId, manager: &TermManager) -> Result<SortId> {
    if let Some(term) = manager.get(arg) {
        let sort = manager
            .sorts
            .get(term.sort)
            .ok_or_else(|| OxizError::Internal(format!("Sort {} not found", term.sort.0)))?;

        match sort.kind {
            SortKind::Int => Ok(manager.sorts.int_sort),
            SortKind::Real => Ok(manager.sorts.real_sort),
            _ => Ok(manager.sorts.int_sort), // Default to Int
        }
    } else {
        Ok(manager.sorts.int_sort) // Default to Int
    }
}

/// Check if a term's sort is compatible with an expected sort
pub fn check_sort_compatibility(
    term_sort: SortId,
    expected_sort: SortId,
    sorts: &SortManager,
    location: SourceSpan,
) -> Result<()> {
    if term_sort != expected_sort {
        let term_sort_str = format_sort(term_sort, sorts);
        let expected_sort_str = format_sort(expected_sort, sorts);

        Err(OxizError::sort_mismatch(
            location,
            expected_sort_str,
            term_sort_str,
        ))
    } else {
        Ok(())
    }
}

/// Format a sort for error messages
fn format_sort(sort_id: SortId, sorts: &SortManager) -> String {
    if let Some(sort) = sorts.get(sort_id) {
        match &sort.kind {
            SortKind::Bool => "Bool".to_string(),
            SortKind::Int => "Int".to_string(),
            SortKind::Real => "Real".to_string(),
            SortKind::String => "String".to_string(),
            SortKind::BitVec(w) => format!("(_ BitVec {})", w),
            SortKind::FloatingPoint { eb, sb } => format!("(_ FloatingPoint {} {})", eb, sb),
            SortKind::Array { domain, range } => {
                let domain_str = format_sort(*domain, sorts);
                let range_str = format_sort(*range, sorts);
                format!("(Array {} {})", domain_str, range_str)
            }
            SortKind::Uninterpreted(spur) => {
                format!("Uninterpreted({})", spur.into_inner())
            }
            SortKind::Parameter(spur) => {
                format!("Param({})", spur.into_inner())
            }
            SortKind::Parametric { name, args } => {
                let arg_strs: Vec<_> = args.iter().map(|a| format_sort(*a, sorts)).collect();
                format!("({} {})", name.into_inner(), arg_strs.join(" "))
            }
            SortKind::Datatype(spur) => {
                format!("Datatype({})", spur.into_inner())
            }
        }
    } else {
        format!("Sort({})", sort_id.0)
    }
}

/// Verify that all arguments to an operation have compatible sorts
pub fn check_homogeneous_sorts(
    args: &[TermId],
    manager: &TermManager,
    location: SourceSpan,
) -> Result<SortId> {
    if args.is_empty() {
        return Ok(manager.sorts.int_sort);
    }

    let first_term = manager
        .get(args[0])
        .ok_or_else(|| OxizError::Internal("First argument term not found".to_string()))?;
    let expected_sort = first_term.sort;

    for &arg in &args[1..] {
        if let Some(term) = manager.get(arg) {
            check_sort_compatibility(term.sort, expected_sort, &manager.sorts, location)?;
        }
    }

    Ok(expected_sort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_infer_bool_constants() {
        let manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();

        if let Some(true_term) = manager.get(t) {
            let inferred = infer_term_sort(true_term, &manager).unwrap();
            assert_eq!(inferred, manager.sorts.bool_sort);
        }

        if let Some(false_term) = manager.get(f) {
            let inferred = infer_term_sort(false_term, &manager).unwrap();
            assert_eq!(inferred, manager.sorts.bool_sort);
        }
    }

    #[test]
    fn test_infer_int_const() {
        let mut manager = TermManager::new();
        let five = manager.mk_int(5);

        if let Some(term) = manager.get(five) {
            let inferred = infer_term_sort(term, &manager).unwrap();
            assert_eq!(inferred, manager.sorts.int_sort);
        }
    }

    #[test]
    fn test_infer_arithmetic() {
        let mut manager = TermManager::new();
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let sum = manager.mk_add(vec![five, ten]);

        if let Some(term) = manager.get(sum) {
            let inferred = infer_term_sort(term, &manager).unwrap();
            assert_eq!(inferred, manager.sorts.int_sort);
        }
    }

    #[test]
    fn test_infer_comparison() {
        let mut manager = TermManager::new();
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(five, ten);

        if let Some(term) = manager.get(lt) {
            let inferred = infer_term_sort(term, &manager).unwrap();
            assert_eq!(inferred, manager.sorts.bool_sort);
        }
    }

    #[test]
    fn test_infer_boolean_ops() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();
        let and = manager.mk_and(vec![t, f]);

        if let Some(term) = manager.get(and) {
            let inferred = infer_term_sort(term, &manager).unwrap();
            assert_eq!(inferred, manager.sorts.bool_sort);
        }
    }

    #[test]
    fn test_infer_ite() {
        let mut manager = TermManager::new();
        let cond = manager.mk_true();
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let ite = manager.mk_ite(cond, five, ten);

        if let Some(term) = manager.get(ite) {
            let inferred = infer_term_sort(term, &manager).unwrap();
            assert_eq!(inferred, manager.sorts.int_sort);
        }
    }

    #[test]
    fn test_check_homogeneous_sorts() {
        let mut manager = TermManager::new();
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let location = crate::error::SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(location);

        let result = check_homogeneous_sorts(&[five, ten], &manager, span);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), manager.sorts.int_sort);
    }

    #[test]
    fn test_check_sort_compatibility_success() {
        let manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let location = crate::error::SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(location);

        let result = check_sort_compatibility(int_sort, int_sort, &manager.sorts, span);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_sort_compatibility_failure() {
        let manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let location = crate::error::SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(location);

        let result = check_sort_compatibility(int_sort, bool_sort, &manager.sorts, span);
        assert!(result.is_err());
    }
}
