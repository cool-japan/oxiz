//! Sort inference for terms
//!
//! This module provides functionality to infer the sort (type) of terms
//! based on their structure and the sorts of their subterms.

use crate::ast::{Term, TermId, TermKind, TermManager};
use crate::error::{OxizError, Result, SourceSpan};
#[allow(unused_imports)]
use crate::prelude::*;
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
        | TermKind::StrInRe(_, _)
        | TermKind::StrLt(_, _)
        | TermKind::StrLe(_, _) => Ok(manager.sorts.bool_sort),

        // String operations that return Int
        TermKind::StrLen(_)
        | TermKind::StrToInt(_)
        | TermKind::StrIndexOf(_, _, _)
        | TermKind::StrToCode(_) => Ok(manager.sorts.int_sort),

        // String operations that return String
        TermKind::StrConcat(_, _)
        | TermKind::StrAt(_, _)
        | TermKind::StrSubstr(_, _, _)
        | TermKind::StrReplace(_, _, _)
        | TermKind::StrReplaceAll(_, _, _)
        | TermKind::StrReplaceRe(_, _, _)
        | TermKind::StrReplaceReAll(_, _, _)
        | TermKind::StrFromCode(_)
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

/// One unit of work for the iterative sort formatter
enum FormatWork {
    /// Render this sort next
    Sort(SortId),
    /// Emit already-decided text (a separator or a closing paren)
    Literal(&'static str),
}

/// Format a sort for error messages
///
/// Uses an explicit work stack instead of recursion. `check_sort_compatibility`
/// is public API and the sort it formats comes from a term whose nesting depth
/// is not bounded by the parser, so a deeply nested `(Array (Array ...))`
/// would otherwise overflow the stack while building an *error message*.
fn format_sort(sort_id: SortId, sorts: &SortManager) -> String {
    let mut out = String::new();
    let mut stack = vec![FormatWork::Sort(sort_id)];

    while let Some(work) = stack.pop() {
        match work {
            FormatWork::Literal(text) => out.push_str(text),
            FormatWork::Sort(id) => {
                let Some(sort) = sorts.get(id) else {
                    out.push_str(&format!("Sort({})", id.0));
                    continue;
                };
                match &sort.kind {
                    SortKind::Bool => out.push_str("Bool"),
                    SortKind::Int => out.push_str("Int"),
                    SortKind::Real => out.push_str("Real"),
                    SortKind::String => out.push_str("String"),
                    SortKind::BitVec(w) => out.push_str(&format!("(_ BitVec {})", w)),
                    SortKind::FloatingPoint { eb, sb } => {
                        out.push_str(&format!("(_ FloatingPoint {} {})", eb, sb));
                    }
                    SortKind::RoundingMode => out.push_str("RoundingMode"),
                    SortKind::Array { domain, range } => {
                        // "(Array " <domain> " " <range> ")"
                        out.push_str("(Array ");
                        stack.push(FormatWork::Literal(")"));
                        stack.push(FormatWork::Sort(*range));
                        stack.push(FormatWork::Literal(" "));
                        stack.push(FormatWork::Sort(*domain));
                    }
                    SortKind::Uninterpreted(spur) => {
                        out.push_str(&format!("Uninterpreted({})", spur.into_inner()));
                    }
                    SortKind::Parameter(spur) => {
                        out.push_str(&format!("Param({})", spur.into_inner()));
                    }
                    SortKind::Parametric { name, args } => {
                        // "(" <name> (" " <arg>)* ")" — with a lone space for
                        // an empty argument list, exactly as the previous
                        // `args.join(" ")` formulation produced.
                        out.push_str(&format!("({}", name.into_inner()));
                        stack.push(FormatWork::Literal(")"));
                        if args.is_empty() {
                            stack.push(FormatWork::Literal(" "));
                        }
                        for arg in args.iter().rev() {
                            stack.push(FormatWork::Sort(*arg));
                            stack.push(FormatWork::Literal(" "));
                        }
                    }
                    SortKind::Datatype(spur) => {
                        out.push_str(&format!("Datatype({})", spur.into_inner()));
                    }
                }
            }
        }
    }

    out
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
            let inferred =
                infer_term_sort(true_term, &manager).expect("test operation should succeed");
            assert_eq!(inferred, manager.sorts.bool_sort);
        }

        if let Some(false_term) = manager.get(f) {
            let inferred =
                infer_term_sort(false_term, &manager).expect("test operation should succeed");
            assert_eq!(inferred, manager.sorts.bool_sort);
        }
    }

    #[test]
    fn test_infer_int_const() {
        let mut manager = TermManager::new();
        let five = manager.mk_int(5);

        if let Some(term) = manager.get(five) {
            let inferred = infer_term_sort(term, &manager).expect("test operation should succeed");
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
            let inferred = infer_term_sort(term, &manager).expect("test operation should succeed");
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
            let inferred = infer_term_sort(term, &manager).expect("test operation should succeed");
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
            let inferred = infer_term_sort(term, &manager).expect("test operation should succeed");
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
            let inferred = infer_term_sort(term, &manager).expect("test operation should succeed");
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
        assert_eq!(
            result.expect("test operation should succeed"),
            manager.sorts.int_sort
        );
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

    #[test]
    fn test_format_sort_nested_array() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let inner = manager.sorts.array(int_sort, bool_sort);
        let outer = manager.sorts.array(inner, int_sort);

        assert_eq!(
            format_sort(outer, &manager.sorts),
            "(Array (Array Int Bool) Int)"
        );
    }

    #[test]
    fn test_format_sort_deep_nesting_does_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let mut current = int_sort;
                for _ in 0..6_250 {
                    current = manager.sorts.array(int_sort, current);
                }
                // Formatting an error message must not overflow the stack.
                let rendered = format_sort(current, &manager.sorts);
                (
                    rendered.starts_with("(Array Int (Array Int "),
                    // The innermost level is `(Array Int Int)`.
                    rendered.contains("Int Int)"),
                    rendered.matches("(Array").count(),
                )
            })
            .expect("thread spawn should succeed");

        let (starts, ends, count) = handle.join().expect("deep formatting must not overflow");
        assert!(starts);
        assert!(ends);
        assert_eq!(count, 6_250);
    }
}
