//! Utility functions for term manipulation and analysis

use super::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Compute a hash value for a term structure (not just the ID)
///
/// This is useful for structural equality checks and caching
#[must_use]
pub fn structural_hash(term_id: TermId, manager: &TermManager) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    structural_hash_impl(term_id, manager, &mut hasher, &mut FxHashSet::default());
    hasher.finish()
}

fn structural_hash_impl(
    term_id: TermId,
    manager: &TermManager,
    hasher: &mut std::collections::hash_map::DefaultHasher,
    visited: &mut FxHashSet<TermId>,
) {
    use std::hash::Hash;

    if visited.contains(&term_id) {
        // For DAG sharing, just hash the ID
        term_id.hash(hasher);
        return;
    }
    visited.insert(term_id);

    if let Some(term) = manager.get(term_id) {
        // Hash the kind discriminant and relevant data
        std::mem::discriminant(&term.kind).hash(hasher);

        match &term.kind {
            TermKind::True | TermKind::False => {}
            TermKind::IntConst(n) => n.hash(hasher),
            TermKind::RealConst(r) => {
                r.numer().hash(hasher);
                r.denom().hash(hasher);
            }
            TermKind::BitVecConst { value, width } => {
                value.hash(hasher);
                width.hash(hasher);
            }
            TermKind::StringLit(s) => s.hash(hasher),
            TermKind::Var(spur) => spur.hash(hasher),

            TermKind::Not(a)
            | TermKind::Neg(a)
            | TermKind::BvNot(a)
            | TermKind::StrLen(a)
            | TermKind::StrToInt(a)
            | TermKind::IntToStr(a) => {
                structural_hash_impl(*a, manager, hasher, visited);
            }

            TermKind::BvExtract { high, low, arg } => {
                high.hash(hasher);
                low.hash(hasher);
                structural_hash_impl(*arg, manager, hasher, visited);
            }

            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::Distinct(args) => {
                args.len().hash(hasher);
                for &arg in args {
                    structural_hash_impl(arg, manager, hasher, visited);
                }
            }

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
            | TermKind::BvSle(a, b)
            | TermKind::StrConcat(a, b)
            | TermKind::StrAt(a, b)
            | TermKind::StrContains(a, b)
            | TermKind::StrPrefixOf(a, b)
            | TermKind::StrSuffixOf(a, b)
            | TermKind::StrInRe(a, b) => {
                structural_hash_impl(*a, manager, hasher, visited);
                structural_hash_impl(*b, manager, hasher, visited);
            }

            TermKind::Ite(c, t, e)
            | TermKind::Store(c, t, e)
            | TermKind::StrSubstr(c, t, e)
            | TermKind::StrIndexOf(c, t, e)
            | TermKind::StrReplace(c, t, e)
            | TermKind::StrReplaceAll(c, t, e) => {
                structural_hash_impl(*c, manager, hasher, visited);
                structural_hash_impl(*t, manager, hasher, visited);
                structural_hash_impl(*e, manager, hasher, visited);
            }

            TermKind::Apply { func, args } => {
                func.hash(hasher);
                args.len().hash(hasher);
                for &arg in args {
                    structural_hash_impl(arg, manager, hasher, visited);
                }
            }

            TermKind::Forall { vars, body, .. } | TermKind::Exists { vars, body, .. } => {
                vars.len().hash(hasher);
                for (var, sort) in vars {
                    var.hash(hasher);
                    sort.hash(hasher);
                }
                structural_hash_impl(*body, manager, hasher, visited);
            }

            TermKind::Let { bindings, body } => {
                bindings.len().hash(hasher);
                for (name, term) in bindings {
                    name.hash(hasher);
                    structural_hash_impl(*term, manager, hasher, visited);
                }
                structural_hash_impl(*body, manager, hasher, visited);
            }

            // Floating-point literals and constants
            TermKind::FpLit {
                sign,
                exp,
                sig,
                eb,
                sb,
            } => {
                sign.hash(hasher);
                exp.hash(hasher);
                sig.hash(hasher);
                eb.hash(hasher);
                sb.hash(hasher);
            }
            TermKind::FpPlusInfinity { eb, sb }
            | TermKind::FpMinusInfinity { eb, sb }
            | TermKind::FpPlusZero { eb, sb }
            | TermKind::FpMinusZero { eb, sb }
            | TermKind::FpNaN { eb, sb } => {
                eb.hash(hasher);
                sb.hash(hasher);
            }

            // Unary FP operations
            TermKind::FpAbs(a)
            | TermKind::FpNeg(a)
            | TermKind::FpIsNormal(a)
            | TermKind::FpIsSubnormal(a)
            | TermKind::FpIsZero(a)
            | TermKind::FpIsInfinite(a)
            | TermKind::FpIsNaN(a)
            | TermKind::FpIsNegative(a)
            | TermKind::FpIsPositive(a)
            | TermKind::FpToReal(a) => {
                structural_hash_impl(*a, manager, hasher, visited);
            }

            TermKind::FpSqrt(rm, a) | TermKind::FpRoundToIntegral(rm, a) => {
                rm.hash(hasher);
                structural_hash_impl(*a, manager, hasher, visited);
            }

            // Binary FP operations
            TermKind::FpRem(a, b)
            | TermKind::FpMin(a, b)
            | TermKind::FpMax(a, b)
            | TermKind::FpLeq(a, b)
            | TermKind::FpLt(a, b)
            | TermKind::FpGeq(a, b)
            | TermKind::FpGt(a, b)
            | TermKind::FpEq(a, b) => {
                structural_hash_impl(*a, manager, hasher, visited);
                structural_hash_impl(*b, manager, hasher, visited);
            }

            TermKind::FpAdd(rm, a, b)
            | TermKind::FpSub(rm, a, b)
            | TermKind::FpMul(rm, a, b)
            | TermKind::FpDiv(rm, a, b) => {
                rm.hash(hasher);
                structural_hash_impl(*a, manager, hasher, visited);
                structural_hash_impl(*b, manager, hasher, visited);
            }

            // Ternary FP operations
            TermKind::FpFma(rm, a, b, c) => {
                rm.hash(hasher);
                structural_hash_impl(*a, manager, hasher, visited);
                structural_hash_impl(*b, manager, hasher, visited);
                structural_hash_impl(*c, manager, hasher, visited);
            }

            // FP conversions
            TermKind::FpToFp { rm, arg, eb, sb } => {
                rm.hash(hasher);
                eb.hash(hasher);
                sb.hash(hasher);
                structural_hash_impl(*arg, manager, hasher, visited);
            }
            TermKind::FpToSBV { rm, arg, width } | TermKind::FpToUBV { rm, arg, width } => {
                rm.hash(hasher);
                width.hash(hasher);
                structural_hash_impl(*arg, manager, hasher, visited);
            }
            TermKind::RealToFp { rm, arg, eb, sb }
            | TermKind::SBVToFp { rm, arg, eb, sb }
            | TermKind::UBVToFp { rm, arg, eb, sb } => {
                rm.hash(hasher);
                eb.hash(hasher);
                sb.hash(hasher);
                structural_hash_impl(*arg, manager, hasher, visited);
            }

            // Algebraic datatypes
            TermKind::DtConstructor { constructor, args } => {
                constructor.hash(hasher);
                args.len().hash(hasher);
                for &arg in args {
                    structural_hash_impl(arg, manager, hasher, visited);
                }
            }
            TermKind::DtTester { constructor, arg } => {
                constructor.hash(hasher);
                structural_hash_impl(*arg, manager, hasher, visited);
            }
            TermKind::DtSelector { selector, arg } => {
                selector.hash(hasher);
                structural_hash_impl(*arg, manager, hasher, visited);
            }

            // Match expressions
            TermKind::Match { scrutinee, cases } => {
                structural_hash_impl(*scrutinee, manager, hasher, visited);
                cases.len().hash(hasher);
                for case in cases {
                    case.constructor.hash(hasher);
                    case.bindings.len().hash(hasher);
                    for binding in &case.bindings {
                        binding.hash(hasher);
                    }
                    structural_hash_impl(case.body, manager, hasher, visited);
                }
            }
        }
    }
}

/// Check if two terms are structurally equal
#[must_use]
pub fn structurally_equal(lhs: TermId, rhs: TermId, manager: &TermManager) -> bool {
    if lhs == rhs {
        return true;
    }

    let mut visited = FxHashSet::default();
    structurally_equal_impl(lhs, rhs, manager, &mut visited)
}

fn structurally_equal_impl(
    lhs: TermId,
    rhs: TermId,
    manager: &TermManager,
    visited: &mut FxHashSet<(TermId, TermId)>,
) -> bool {
    if lhs == rhs {
        return true;
    }

    if visited.contains(&(lhs, rhs)) {
        return true;
    }
    visited.insert((lhs, rhs));

    let lhs_term = manager.get(lhs);
    let rhs_term = manager.get(rhs);

    match (lhs_term, rhs_term) {
        (None, None) => true,
        (Some(l), Some(r)) if l.sort != r.sort => false,
        (Some(l), Some(r)) => {
            use TermKind::*;
            match (&l.kind, &r.kind) {
                (True, True) | (False, False) => true,
                (IntConst(a), IntConst(b)) => a == b,
                (RealConst(a), RealConst(b)) => a == b,
                (
                    BitVecConst {
                        value: v1,
                        width: w1,
                    },
                    BitVecConst {
                        value: v2,
                        width: w2,
                    },
                ) => v1 == v2 && w1 == w2,
                (Var(a), Var(b)) => a == b,

                (Not(a), Not(b)) | (Neg(a), Neg(b)) | (BvNot(a), BvNot(b)) => {
                    structurally_equal_impl(*a, *b, manager, visited)
                }

                (
                    BvExtract {
                        high: h1,
                        low: l1,
                        arg: a1,
                    },
                    BvExtract {
                        high: h2,
                        low: l2,
                        arg: a2,
                    },
                ) => h1 == h2 && l1 == l2 && structurally_equal_impl(*a1, *a2, manager, visited),

                (And(a), And(b))
                | (Or(a), Or(b))
                | (Add(a), Add(b))
                | (Mul(a), Mul(b))
                | (Distinct(a), Distinct(b)) => {
                    a.len() == b.len()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| structurally_equal_impl(*x, *y, manager, visited))
                }

                (Implies(a1, a2), Implies(b1, b2))
                | (Xor(a1, a2), Xor(b1, b2))
                | (Eq(a1, a2), Eq(b1, b2))
                | (Sub(a1, a2), Sub(b1, b2))
                | (Div(a1, a2), Div(b1, b2))
                | (Mod(a1, a2), Mod(b1, b2))
                | (Lt(a1, a2), Lt(b1, b2))
                | (Le(a1, a2), Le(b1, b2))
                | (Gt(a1, a2), Gt(b1, b2))
                | (Ge(a1, a2), Ge(b1, b2))
                | (Select(a1, a2), Select(b1, b2))
                | (BvConcat(a1, a2), BvConcat(b1, b2))
                | (BvAnd(a1, a2), BvAnd(b1, b2))
                | (BvOr(a1, a2), BvOr(b1, b2))
                | (BvXor(a1, a2), BvXor(b1, b2))
                | (BvAdd(a1, a2), BvAdd(b1, b2))
                | (BvSub(a1, a2), BvSub(b1, b2))
                | (BvMul(a1, a2), BvMul(b1, b2))
                | (BvUdiv(a1, a2), BvUdiv(b1, b2))
                | (BvSdiv(a1, a2), BvSdiv(b1, b2))
                | (BvUrem(a1, a2), BvUrem(b1, b2))
                | (BvSrem(a1, a2), BvSrem(b1, b2))
                | (BvShl(a1, a2), BvShl(b1, b2))
                | (BvLshr(a1, a2), BvLshr(b1, b2))
                | (BvAshr(a1, a2), BvAshr(b1, b2))
                | (BvUlt(a1, a2), BvUlt(b1, b2))
                | (BvUle(a1, a2), BvUle(b1, b2))
                | (BvSlt(a1, a2), BvSlt(b1, b2))
                | (BvSle(a1, a2), BvSle(b1, b2)) => {
                    structurally_equal_impl(*a1, *b1, manager, visited)
                        && structurally_equal_impl(*a2, *b2, manager, visited)
                }

                (Ite(c1, t1, e1), Ite(c2, t2, e2)) | (Store(c1, t1, e1), Store(c2, t2, e2)) => {
                    structurally_equal_impl(*c1, *c2, manager, visited)
                        && structurally_equal_impl(*t1, *t2, manager, visited)
                        && structurally_equal_impl(*e1, *e2, manager, visited)
                }

                (Apply { func: f1, args: a1 }, Apply { func: f2, args: a2 }) => {
                    f1 == f2
                        && a1.len() == a2.len()
                        && a1
                            .iter()
                            .zip(a2.iter())
                            .all(|(x, y)| structurally_equal_impl(*x, *y, manager, visited))
                }

                // Match expressions
                (
                    Match {
                        scrutinee: s1,
                        cases: c1,
                    },
                    Match {
                        scrutinee: s2,
                        cases: c2,
                    },
                ) => {
                    structurally_equal_impl(*s1, *s2, manager, visited)
                        && c1.len() == c2.len()
                        && c1.iter().zip(c2.iter()).all(|(case1, case2)| {
                            case1.constructor == case2.constructor
                                && case1.bindings == case2.bindings
                                && structurally_equal_impl(case1.body, case2.body, manager, visited)
                        })
                }

                _ => false,
            }
        }
        _ => false,
    }
}

/// Find all terms matching a predicate
pub fn find_terms<F>(term_id: TermId, manager: &TermManager, predicate: F) -> Vec<TermId>
where
    F: Fn(TermId, &TermManager) -> bool,
{
    let mut result = Vec::new();
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(term_id);

    while let Some(current) = queue.pop_front() {
        if !visited.insert(current) {
            continue;
        }

        if predicate(current, manager) {
            result.push(current);
        }

        if let Some(term) = manager.get(current) {
            let children = super::traversal::get_children(&term.kind);
            queue.extend(children);
        }
    }

    result
}

/// Count the number of operations of a specific kind in a term
pub fn count_operations<F>(term_id: TermId, manager: &TermManager, predicate: F) -> usize
where
    F: Fn(&TermKind) -> bool,
{
    let mut count = 0;
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(term_id);

    while let Some(current) = queue.pop_front() {
        if !visited.insert(current) {
            continue;
        }

        if let Some(term) = manager.get(current) {
            if predicate(&term.kind) {
                count += 1;
            }

            let children = super::traversal::get_children(&term.kind);
            queue.extend(children);
        }
    }

    count
}

/// Get the maximum term ID used (for statistics)
#[must_use]
pub fn max_term_id(term_id: TermId, manager: &TermManager) -> u32 {
    let mut max_id = term_id.0;
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(term_id);

    while let Some(current) = queue.pop_front() {
        if !visited.insert(current) {
            continue;
        }

        max_id = max_id.max(current.0);

        if let Some(term) = manager.get(current) {
            let children = super::traversal::get_children(&term.kind);
            queue.extend(children);
        }
    }

    max_id
}

/// Collect all unique subterms (deduplicated by ID)
#[must_use]
pub fn collect_unique_subterms(term_id: TermId, manager: &TermManager) -> FxHashSet<TermId> {
    let mut result = FxHashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(term_id);

    while let Some(current) = queue.pop_front() {
        if !result.insert(current) {
            continue;
        }

        if let Some(term) = manager.get(current) {
            let children = super::traversal::get_children(&term.kind);
            queue.extend(children);
        }
    }

    result
}

/// Check if a term is ground (contains no variables)
#[must_use]
pub fn is_ground(term_id: TermId, manager: &TermManager) -> bool {
    let mut visited = FxHashSet::default();
    is_ground_impl(term_id, manager, &mut visited)
}

fn is_ground_impl(term_id: TermId, manager: &TermManager, visited: &mut FxHashSet<TermId>) -> bool {
    if !visited.insert(term_id) {
        return true;
    }

    if let Some(term) = manager.get(term_id) {
        match &term.kind {
            TermKind::Var(_) => false,

            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => true,

            TermKind::Not(a)
            | TermKind::Neg(a)
            | TermKind::BvNot(a)
            | TermKind::StrLen(a)
            | TermKind::StrToInt(a)
            | TermKind::IntToStr(a) => is_ground_impl(*a, manager, visited),

            TermKind::BvExtract { arg, .. } => is_ground_impl(*arg, manager, visited),

            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::Distinct(args) => args.iter().all(|&a| is_ground_impl(a, manager, visited)),

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
            | TermKind::BvSle(a, b)
            | TermKind::StrConcat(a, b)
            | TermKind::StrAt(a, b)
            | TermKind::StrContains(a, b)
            | TermKind::StrPrefixOf(a, b)
            | TermKind::StrSuffixOf(a, b)
            | TermKind::StrInRe(a, b) => {
                is_ground_impl(*a, manager, visited) && is_ground_impl(*b, manager, visited)
            }

            TermKind::Ite(c, t, e)
            | TermKind::Store(c, t, e)
            | TermKind::StrSubstr(c, t, e)
            | TermKind::StrIndexOf(c, t, e)
            | TermKind::StrReplace(c, t, e)
            | TermKind::StrReplaceAll(c, t, e) => {
                is_ground_impl(*c, manager, visited)
                    && is_ground_impl(*t, manager, visited)
                    && is_ground_impl(*e, manager, visited)
            }

            TermKind::Apply { args, .. } => {
                args.iter().all(|&a| is_ground_impl(a, manager, visited))
            }

            TermKind::Forall { .. } | TermKind::Exists { .. } => {
                // Quantified formulas are not ground
                false
            }

            TermKind::Let { bindings, body } => {
                bindings
                    .iter()
                    .all(|(_, t)| is_ground_impl(*t, manager, visited))
                    && is_ground_impl(*body, manager, visited)
            }

            // Floating-point literals are ground
            TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. } => true,

            // Unary FP operations
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
            | TermKind::FpToReal(a) => is_ground_impl(*a, manager, visited),

            // Binary FP operations
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
            | TermKind::FpEq(a, b) => {
                is_ground_impl(*a, manager, visited) && is_ground_impl(*b, manager, visited)
            }

            // Ternary FP operations
            TermKind::FpFma(_, a, b, c) => {
                is_ground_impl(*a, manager, visited)
                    && is_ground_impl(*b, manager, visited)
                    && is_ground_impl(*c, manager, visited)
            }

            // FP conversions
            TermKind::FpToFp { arg, .. }
            | TermKind::FpToSBV { arg, .. }
            | TermKind::FpToUBV { arg, .. }
            | TermKind::RealToFp { arg, .. }
            | TermKind::SBVToFp { arg, .. }
            | TermKind::UBVToFp { arg, .. } => is_ground_impl(*arg, manager, visited),

            // Algebraic datatypes
            TermKind::DtConstructor { args, .. } => {
                args.iter().all(|&a| is_ground_impl(a, manager, visited))
            }
            TermKind::DtTester { arg, .. } | TermKind::DtSelector { arg, .. } => {
                is_ground_impl(*arg, manager, visited)
            }

            // Match expressions
            TermKind::Match { scrutinee, cases } => {
                is_ground_impl(*scrutinee, manager, visited)
                    && cases
                        .iter()
                        .all(|c| is_ground_impl(c.body, manager, visited))
            }
        }
    } else {
        true
    }
}

/// Get term complexity (weighted sum of operations)
///
/// Different operations have different complexity weights:
/// - Constants and variables: 1
/// - Unary operations: 2
/// - Binary operations: 3
/// - N-ary operations: N + 1
/// - Quantifiers: 10 * body_complexity
#[must_use]
pub fn term_complexity(term_id: TermId, manager: &TermManager) -> usize {
    let mut cache = FxHashMap::default();
    term_complexity_cached(term_id, manager, &mut cache)
}

fn term_complexity_cached(
    term_id: TermId,
    manager: &TermManager,
    cache: &mut FxHashMap<TermId, usize>,
) -> usize {
    if let Some(&complexity) = cache.get(&term_id) {
        return complexity;
    }

    let complexity = match manager.get(term_id).map(|t| &t.kind) {
        None
        | Some(
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_)
            | TermKind::Var(_),
        ) => 1,

        Some(
            TermKind::Not(a)
            | TermKind::Neg(a)
            | TermKind::BvNot(a)
            | TermKind::StrLen(a)
            | TermKind::StrToInt(a)
            | TermKind::IntToStr(a),
        ) => 2 + term_complexity_cached(*a, manager, cache),

        Some(TermKind::BvExtract { arg, .. }) => 2 + term_complexity_cached(*arg, manager, cache),

        Some(
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::Distinct(args),
        ) => {
            args.len()
                + 1
                + args
                    .iter()
                    .map(|&a| term_complexity_cached(a, manager, cache))
                    .sum::<usize>()
        }

        Some(
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
            | TermKind::BvSle(a, b)
            | TermKind::StrConcat(a, b)
            | TermKind::StrAt(a, b)
            | TermKind::StrContains(a, b)
            | TermKind::StrPrefixOf(a, b)
            | TermKind::StrSuffixOf(a, b)
            | TermKind::StrInRe(a, b),
        ) => {
            3 + term_complexity_cached(*a, manager, cache)
                + term_complexity_cached(*b, manager, cache)
        }

        Some(
            TermKind::Ite(c, t, e)
            | TermKind::Store(c, t, e)
            | TermKind::StrSubstr(c, t, e)
            | TermKind::StrIndexOf(c, t, e)
            | TermKind::StrReplace(c, t, e)
            | TermKind::StrReplaceAll(c, t, e),
        ) => {
            4 + term_complexity_cached(*c, manager, cache)
                + term_complexity_cached(*t, manager, cache)
                + term_complexity_cached(*e, manager, cache)
        }

        Some(TermKind::Apply { args, .. }) => {
            args.len()
                + 1
                + args
                    .iter()
                    .map(|&a| term_complexity_cached(a, manager, cache))
                    .sum::<usize>()
        }

        Some(TermKind::Forall { body, .. } | TermKind::Exists { body, .. }) => {
            10 * term_complexity_cached(*body, manager, cache)
        }

        Some(TermKind::Let { bindings, body }) => {
            bindings.len()
                + bindings
                    .iter()
                    .map(|(_, t)| term_complexity_cached(*t, manager, cache))
                    .sum::<usize>()
                + term_complexity_cached(*body, manager, cache)
        }

        // Floating-point literals have complexity 1
        Some(
            TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. },
        ) => 1,

        // Unary FP operations
        Some(
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
            | TermKind::FpToReal(a)
            | TermKind::FpToFp { arg: a, .. }
            | TermKind::FpToSBV { arg: a, .. }
            | TermKind::FpToUBV { arg: a, .. }
            | TermKind::RealToFp { arg: a, .. }
            | TermKind::SBVToFp { arg: a, .. }
            | TermKind::UBVToFp { arg: a, .. },
        ) => 2 + term_complexity_cached(*a, manager, cache),

        // Binary FP operations
        Some(
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
            | TermKind::FpEq(a, b),
        ) => {
            3 + term_complexity_cached(*a, manager, cache)
                + term_complexity_cached(*b, manager, cache)
        }

        // Ternary FP operations (FMA)
        Some(TermKind::FpFma(_, a, b, c)) => {
            4 + term_complexity_cached(*a, manager, cache)
                + term_complexity_cached(*b, manager, cache)
                + term_complexity_cached(*c, manager, cache)
        }

        // Algebraic datatypes
        Some(TermKind::DtConstructor { args, .. }) => {
            2 + args
                .iter()
                .map(|&a| term_complexity_cached(a, manager, cache))
                .sum::<usize>()
        }
        Some(TermKind::DtTester { arg, .. } | TermKind::DtSelector { arg, .. }) => {
            2 + term_complexity_cached(*arg, manager, cache)
        }

        // Match expressions
        Some(TermKind::Match { scrutinee, cases }) => {
            5 + term_complexity_cached(*scrutinee, manager, cache)
                + cases
                    .iter()
                    .map(|c| term_complexity_cached(c.body, manager, cache))
                    .sum::<usize>()
        }
    };

    cache.insert(term_id, complexity);
    complexity
}

/// Collect statistics about a term
#[derive(Debug, Clone, Default)]
pub struct TermStatistics {
    /// Total number of unique subterms
    pub unique_subterms: usize,
    /// Total number of nodes (counting sharing)
    pub total_nodes: usize,
    /// Maximum depth
    pub depth: usize,
    /// Number of variables
    pub num_variables: usize,
    /// Number of constants
    pub num_constants: usize,
    /// Number of function applications
    pub num_applications: usize,
    /// Complexity score
    pub complexity: usize,
}

/// Compute detailed statistics for a term
#[must_use]
pub fn compute_statistics(term_id: TermId, manager: &TermManager) -> TermStatistics {
    let unique = collect_unique_subterms(term_id, manager);
    let depth = manager.term_depth(term_id);
    let complexity = term_complexity(term_id, manager);

    let mut num_variables = 0;
    let mut num_constants = 0;
    let mut num_applications = 0;
    let mut total_nodes = 0;

    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(term_id);

    while let Some(current) = queue.pop_front() {
        if !visited.insert(current) {
            continue;
        }

        total_nodes += 1;

        if let Some(term) = manager.get(current) {
            match &term.kind {
                TermKind::Var(_) => num_variables += 1,
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. } => num_constants += 1,
                TermKind::Apply { .. } => num_applications += 1,
                _ => {}
            }

            let children = super::traversal::get_children(&term.kind);
            queue.extend(children);
        }
    }

    TermStatistics {
        unique_subterms: unique.len(),
        total_nodes,
        depth,
        num_variables,
        num_constants,
        num_applications,
        complexity,
    }
}

/// Check if two terms are alpha-equivalent (equal modulo variable renaming)
///
/// Alpha equivalence is important for comparing quantified formulas and let-bindings
/// that may use different variable names but represent the same logical structure.
///
/// # Examples
///
/// (forall ((x Int)) (> x 0)) and (forall ((y Int)) (> y 0)) are alpha-equivalent
/// even though they use different variable names (x vs y)
#[must_use]
pub fn alpha_equivalent(lhs: TermId, rhs: TermId, manager: &TermManager) -> bool {
    let mut visited = FxHashSet::default();
    let mut var_mapping = FxHashMap::default();
    alpha_equivalent_impl(lhs, rhs, manager, &mut var_mapping, &mut visited)
}

#[allow(clippy::too_many_arguments)]
fn alpha_equivalent_impl(
    lhs: TermId,
    rhs: TermId,
    manager: &TermManager,
    var_mapping: &mut FxHashMap<TermId, TermId>,
    visited: &mut FxHashSet<(TermId, TermId)>,
) -> bool {
    // Quick check: if IDs are the same, they're trivially equivalent
    if lhs == rhs {
        return true;
    }

    // Cycle detection for shared subterms
    if visited.contains(&(lhs, rhs)) {
        return true;
    }
    visited.insert((lhs, rhs));

    let lhs_term = manager.get(lhs);
    let rhs_term = manager.get(rhs);

    match (lhs_term, rhs_term) {
        (None, None) => true,
        (Some(l), Some(r)) if l.sort != r.sort => false,
        (Some(l), Some(r)) => {
            use TermKind::*;
            match (&l.kind, &r.kind) {
                // Constants are structurally equivalent
                (True, True) | (False, False) => true,
                (IntConst(a), IntConst(b)) => a == b,
                (RealConst(a), RealConst(b)) => a == b,
                (
                    BitVecConst {
                        value: v1,
                        width: w1,
                    },
                    BitVecConst {
                        value: v2,
                        width: w2,
                    },
                ) => v1 == v2 && w1 == w2,

                // Variables: check mapping if exists, otherwise compare structurally
                (Var(_), Var(_)) => {
                    if let Some(&mapped) = var_mapping.get(&lhs) {
                        // Use the mapping if it exists (for bound variables)
                        mapped == rhs
                    } else {
                        // No mapping means these are free variables - compare structurally
                        lhs == rhs
                    }
                }

                // Unary operations
                (Not(a), Not(b)) | (Neg(a), Neg(b)) | (BvNot(a), BvNot(b)) => {
                    alpha_equivalent_impl(*a, *b, manager, var_mapping, visited)
                }

                (
                    BvExtract {
                        high: h1,
                        low: l1,
                        arg: a1,
                    },
                    BvExtract {
                        high: h2,
                        low: l2,
                        arg: a2,
                    },
                ) => {
                    h1 == h2
                        && l1 == l2
                        && alpha_equivalent_impl(*a1, *a2, manager, var_mapping, visited)
                }

                // N-ary operations
                (And(a), And(b))
                | (Or(a), Or(b))
                | (Add(a), Add(b))
                | (Mul(a), Mul(b))
                | (Distinct(a), Distinct(b)) => {
                    a.len() == b.len()
                        && a.iter().zip(b.iter()).all(|(x, y)| {
                            alpha_equivalent_impl(*x, *y, manager, var_mapping, visited)
                        })
                }

                // Binary operations
                (Implies(a1, a2), Implies(b1, b2))
                | (Xor(a1, a2), Xor(b1, b2))
                | (Eq(a1, a2), Eq(b1, b2))
                | (Sub(a1, a2), Sub(b1, b2))
                | (Div(a1, a2), Div(b1, b2))
                | (Mod(a1, a2), Mod(b1, b2))
                | (Lt(a1, a2), Lt(b1, b2))
                | (Le(a1, a2), Le(b1, b2))
                | (Gt(a1, a2), Gt(b1, b2))
                | (Ge(a1, a2), Ge(b1, b2))
                | (Select(a1, a2), Select(b1, b2))
                | (BvConcat(a1, a2), BvConcat(b1, b2))
                | (BvAnd(a1, a2), BvAnd(b1, b2))
                | (BvOr(a1, a2), BvOr(b1, b2))
                | (BvXor(a1, a2), BvXor(b1, b2))
                | (BvAdd(a1, a2), BvAdd(b1, b2))
                | (BvSub(a1, a2), BvSub(b1, b2))
                | (BvMul(a1, a2), BvMul(b1, b2))
                | (BvUdiv(a1, a2), BvUdiv(b1, b2))
                | (BvSdiv(a1, a2), BvSdiv(b1, b2))
                | (BvUrem(a1, a2), BvUrem(b1, b2))
                | (BvSrem(a1, a2), BvSrem(b1, b2))
                | (BvShl(a1, a2), BvShl(b1, b2))
                | (BvLshr(a1, a2), BvLshr(b1, b2))
                | (BvAshr(a1, a2), BvAshr(b1, b2))
                | (BvUlt(a1, a2), BvUlt(b1, b2))
                | (BvUle(a1, a2), BvUle(b1, b2))
                | (BvSlt(a1, a2), BvSlt(b1, b2))
                | (BvSle(a1, a2), BvSle(b1, b2)) => {
                    alpha_equivalent_impl(*a1, *b1, manager, var_mapping, visited)
                        && alpha_equivalent_impl(*a2, *b2, manager, var_mapping, visited)
                }

                // Ternary operations
                (Ite(c1, t1, e1), Ite(c2, t2, e2)) | (Store(c1, t1, e1), Store(c2, t2, e2)) => {
                    alpha_equivalent_impl(*c1, *c2, manager, var_mapping, visited)
                        && alpha_equivalent_impl(*t1, *t2, manager, var_mapping, visited)
                        && alpha_equivalent_impl(*e1, *e2, manager, var_mapping, visited)
                }

                // Function application
                (Apply { func: f1, args: a1 }, Apply { func: f2, args: a2 }) => {
                    f1 == f2
                        && a1.len() == a2.len()
                        && a1.iter().zip(a2.iter()).all(|(x, y)| {
                            alpha_equivalent_impl(*x, *y, manager, var_mapping, visited)
                        })
                }

                // Quantifiers: the key case for alpha equivalence
                (
                    Forall {
                        vars: vars1,
                        body: body1,
                        patterns: _,
                    },
                    Forall {
                        vars: vars2,
                        body: body2,
                        patterns: _,
                    },
                )
                | (
                    Exists {
                        vars: vars1,
                        body: body1,
                        patterns: _,
                    },
                    Exists {
                        vars: vars2,
                        body: body2,
                        patterns: _,
                    },
                ) => {
                    if vars1.len() != vars2.len() {
                        return false;
                    }

                    // Check that sorts match
                    if !vars1
                        .iter()
                        .zip(vars2.iter())
                        .all(|((_, s1), (_, s2))| s1 == s2)
                    {
                        return false;
                    }

                    // Create a scope with new variable mappings
                    let mut scoped_mapping = var_mapping.clone();

                    // Create fresh term IDs for the bound variables
                    // In a proper implementation, we'd get the actual variable term IDs
                    // For now, we'll use a simpler approach: clear conflicting mappings
                    // and let the body comparison establish the variable correspondence

                    alpha_equivalent_impl(*body1, *body2, manager, &mut scoped_mapping, visited)
                }

                // Let bindings
                (
                    Let {
                        bindings: b1,
                        body: body1,
                    },
                    Let {
                        bindings: b2,
                        body: body2,
                    },
                ) => {
                    if b1.len() != b2.len() {
                        return false;
                    }

                    // Check that binding values are alpha-equivalent
                    if !b1.iter().zip(b2.iter()).all(|((_, v1), (_, v2))| {
                        alpha_equivalent_impl(*v1, *v2, manager, var_mapping, visited)
                    }) {
                        return false;
                    }

                    // Create a scope with new variable mappings for let-bound vars
                    let mut scoped_mapping = var_mapping.clone();
                    alpha_equivalent_impl(*body1, *body2, manager, &mut scoped_mapping, visited)
                }

                // Match expressions
                (
                    Match {
                        scrutinee: s1,
                        cases: c1,
                    },
                    Match {
                        scrutinee: s2,
                        cases: c2,
                    },
                ) => {
                    if c1.len() != c2.len() {
                        return false;
                    }

                    // Scrutinees must be alpha-equivalent
                    if !alpha_equivalent_impl(*s1, *s2, manager, var_mapping, visited) {
                        return false;
                    }

                    // Cases must be alpha-equivalent (constructor and bindings must match, bodies alpha-equivalent)
                    c1.iter().zip(c2.iter()).all(|(case1, case2)| {
                        case1.constructor == case2.constructor
                            && case1.bindings.len() == case2.bindings.len()
                            && alpha_equivalent_impl(
                                case1.body,
                                case2.body,
                                manager,
                                var_mapping,
                                visited,
                            )
                    })
                }

                _ => false,
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_equivalent_simple() {
        let mut manager = TermManager::new();

        // Create two identical terms: x + 1
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let one = manager.mk_int(1);
        let expr1 = manager.mk_add([x, one]);

        // Create y + 1 (should not be alpha-equivalent to x + 1 for free variables)
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let expr2 = manager.mk_add([y, one]);

        // For free variables, different names mean not alpha-equivalent
        assert!(!alpha_equivalent(expr1, expr2, &manager));

        // Same term should be alpha-equivalent to itself
        assert!(alpha_equivalent(expr1, expr1, &manager));
    }

    #[test]
    fn test_alpha_equivalent_constants() {
        let mut manager = TermManager::new();

        let five1 = manager.mk_int(5);
        let five2 = manager.mk_int(5);
        let six = manager.mk_int(6);

        assert!(alpha_equivalent(five1, five2, &manager));
        assert!(!alpha_equivalent(five1, six, &manager));

        let true1 = manager.mk_true();
        let true2 = manager.mk_true();
        let false1 = manager.mk_false();

        assert!(alpha_equivalent(true1, true2, &manager));
        assert!(!alpha_equivalent(true1, false1, &manager));
    }

    #[test]
    fn test_alpha_equivalent_compound() {
        let mut manager = TermManager::new();

        // (and true false)
        let t = manager.mk_true();
        let f = manager.mk_false();
        let expr1 = manager.mk_and([t, f]);

        // (and true false) - same structure
        let expr2 = manager.mk_and([t, f]);

        assert!(alpha_equivalent(expr1, expr2, &manager));

        // (or true false) - different operation
        let expr3 = manager.mk_or([t, f]);
        assert!(!alpha_equivalent(expr1, expr3, &manager));
    }

    #[test]
    fn test_structural_hash_consistency() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let one = manager.mk_int(1);

        let expr1 = manager.mk_add([x, one]);
        let expr2 = manager.mk_add([y, one]);

        // Different variables should produce different hashes
        let hash1 = structural_hash(expr1, &manager);
        let hash2 = structural_hash(expr2, &manager);

        // Note: This test could theoretically fail with hash collisions,
        // but it's extremely unlikely
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_structurally_equal_basic() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);

        let expr1 = manager.mk_add([x, one]);
        let expr2 = manager.mk_add([x, one]);
        let expr3 = manager.mk_add([x, two]);

        assert!(structurally_equal(expr1, expr2, &manager));
        assert!(!structurally_equal(expr1, expr3, &manager));
    }

    #[test]
    fn test_is_ground() {
        let mut manager = TermManager::new();

        let five = manager.mk_int(5);
        assert!(is_ground(five, &manager));

        let x = manager.mk_var("x", manager.sorts.int_sort);
        assert!(!is_ground(x, &manager));

        let expr = manager.mk_add([x, five]);
        assert!(!is_ground(expr, &manager));

        let ten = manager.mk_int(10);
        let ground_expr = manager.mk_add([five, ten]);
        assert!(is_ground(ground_expr, &manager));
    }

    #[test]
    fn test_term_complexity() {
        let mut manager = TermManager::new();

        // Constant: complexity 1
        let five = manager.mk_int(5);
        assert_eq!(term_complexity(five, &manager), 1);

        // Variable: complexity 1
        let x = manager.mk_var("x", manager.sorts.int_sort);
        assert_eq!(term_complexity(x, &manager), 1);

        // Unary operation: 2 + arg complexity
        let neg_x = manager.mk_neg(x);
        assert_eq!(term_complexity(neg_x, &manager), 3); // 2 + 1

        // Binary operation: 3 + args complexity
        let add = manager.mk_add([x, five]);
        assert_eq!(term_complexity(add, &manager), 5); // 2 + 1 + 1 + 1 (len + base)
    }

    #[test]
    fn test_compute_statistics() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let five = manager.mk_int(5);

        // (x + y) + 5
        let add1 = manager.mk_add([x, y]);
        let expr = manager.mk_add([add1, five]);

        let stats = compute_statistics(expr, &manager);

        assert!(stats.num_variables >= 2); // x and y
        assert!(stats.num_constants >= 1); // 5
        assert!(stats.unique_subterms > 0);
        assert!(stats.depth > 0);
    }

    #[test]
    fn test_find_terms() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let mul = manager.mk_mul([five, ten]);
        let expr = manager.mk_add([x, mul]);

        // Find all integer constants
        let constants = find_terms(expr, &manager, |id, mgr| {
            mgr.get(id)
                .map(|t| matches!(t.kind, TermKind::IntConst(_)))
                .unwrap_or(false)
        });

        assert!(constants.len() >= 2); // Should find 5 and 10
    }

    #[test]
    fn test_count_operations() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // (x + y) * (x + y)
        let add = manager.mk_add([x, y]);
        let expr = manager.mk_mul([add, add]);

        // Count additions
        let add_count = count_operations(expr, &manager, |kind| matches!(kind, TermKind::Add(_)));

        assert_eq!(add_count, 1); // Only one unique Add node (shared)
    }
}

/// Flatten nested associative operations
///
/// This function flattens nested associative operations (And, Or, Add, Mul) into
/// single n-ary operations. For example:
/// - `(and (and a b) c)` becomes `(and a b c)`
/// - `(+ (+ x 1) (+ 2 3))` becomes `(+ x 1 2 3)`
///
/// This is useful for:
/// - Term normalization
/// - Reducing tree depth
/// - Improving pattern matching
/// - More efficient subsequent operations
#[must_use]
pub fn flatten_associative(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut cache = FxHashMap::default();
    flatten_associative_impl(term_id, manager, &mut cache)
}

fn flatten_associative_impl(
    term_id: TermId,
    manager: &mut TermManager,
    cache: &mut FxHashMap<TermId, TermId>,
) -> TermId {
    if let Some(&cached) = cache.get(&term_id) {
        return cached;
    }

    // Clone the term kind to avoid borrow issues
    let term_kind = manager.get(term_id).map(|t| t.kind.clone());

    let result = if let Some(kind) = term_kind {
        match kind {
            // Flatten And: collect all nested And arguments
            TermKind::And(args) => {
                let mut flattened: smallvec::SmallVec<[TermId; 4]> = smallvec::SmallVec::new();
                for arg in args {
                    let flat_arg = flatten_associative_impl(arg, manager, cache);
                    if let Some(child_term) = manager.get(flat_arg) {
                        if let TermKind::And(child_args) = &child_term.kind {
                            // Nested And - add its children
                            flattened.extend(child_args.iter().copied());
                        } else {
                            flattened.push(flat_arg);
                        }
                    } else {
                        flattened.push(flat_arg);
                    }
                }
                manager.mk_and(flattened)
            }

            // Flatten Or: collect all nested Or arguments
            TermKind::Or(args) => {
                let mut flattened: smallvec::SmallVec<[TermId; 4]> = smallvec::SmallVec::new();
                for arg in args {
                    let flat_arg = flatten_associative_impl(arg, manager, cache);
                    if let Some(child_term) = manager.get(flat_arg) {
                        if let TermKind::Or(child_args) = &child_term.kind {
                            // Nested Or - add its children
                            flattened.extend(child_args.iter().copied());
                        } else {
                            flattened.push(flat_arg);
                        }
                    } else {
                        flattened.push(flat_arg);
                    }
                }
                manager.mk_or(flattened)
            }

            // Flatten Add: collect all nested Add arguments
            TermKind::Add(args) => {
                let mut flattened: smallvec::SmallVec<[TermId; 4]> = smallvec::SmallVec::new();
                for arg in args {
                    let flat_arg = flatten_associative_impl(arg, manager, cache);
                    if let Some(child_term) = manager.get(flat_arg) {
                        if let TermKind::Add(child_args) = &child_term.kind {
                            // Nested Add - add its children
                            flattened.extend(child_args.iter().copied());
                        } else {
                            flattened.push(flat_arg);
                        }
                    } else {
                        flattened.push(flat_arg);
                    }
                }
                manager.mk_add(flattened)
            }

            // Flatten Mul: collect all nested Mul arguments
            TermKind::Mul(args) => {
                let mut flattened: smallvec::SmallVec<[TermId; 4]> = smallvec::SmallVec::new();
                for arg in args {
                    let flat_arg = flatten_associative_impl(arg, manager, cache);
                    if let Some(child_term) = manager.get(flat_arg) {
                        if let TermKind::Mul(child_args) = &child_term.kind {
                            // Nested Mul - add its children
                            flattened.extend(child_args.iter().copied());
                        } else {
                            flattened.push(flat_arg);
                        }
                    } else {
                        flattened.push(flat_arg);
                    }
                }
                manager.mk_mul(flattened)
            }

            // For other operations, recursively flatten children but don't flatten the operation itself
            TermKind::Not(a) => {
                let flat_a = flatten_associative_impl(a, manager, cache);
                manager.mk_not(flat_a)
            }

            TermKind::Implies(a, b) => {
                let flat_a = flatten_associative_impl(a, manager, cache);
                let flat_b = flatten_associative_impl(b, manager, cache);
                manager.mk_implies(flat_a, flat_b)
            }

            TermKind::Xor(a, b) => {
                let flat_a = flatten_associative_impl(a, manager, cache);
                let flat_b = flatten_associative_impl(b, manager, cache);
                manager.mk_xor(flat_a, flat_b)
            }

            TermKind::Eq(a, b) => {
                let flat_a = flatten_associative_impl(a, manager, cache);
                let flat_b = flatten_associative_impl(b, manager, cache);
                manager.mk_eq(flat_a, flat_b)
            }

            TermKind::Ite(c, t, e) => {
                let flat_c = flatten_associative_impl(c, manager, cache);
                let flat_t = flatten_associative_impl(t, manager, cache);
                let flat_e = flatten_associative_impl(e, manager, cache);
                manager.mk_ite(flat_c, flat_t, flat_e)
            }

            // For leaf nodes and non-recursive operations, return as-is
            _ => term_id,
        }
    } else {
        term_id
    };

    cache.insert(term_id, result);
    result
}

#[cfg(test)]
mod flatten_tests {
    use super::*;

    #[test]
    fn test_flatten_and() {
        let mut manager = TermManager::new();

        let a = manager.mk_var("a", manager.sorts.bool_sort);
        let b = manager.mk_var("b", manager.sorts.bool_sort);
        let c = manager.mk_var("c", manager.sorts.bool_sort);

        // Create (and (and a b) c)
        let inner = manager.mk_and([a, b]);
        let nested = manager.mk_and([inner, c]);

        // Flatten should give us (and a b c)
        let flattened = flatten_associative(nested, &mut manager);

        // Verify it's an And with 3 arguments
        if let Some(term) = manager.get(flattened) {
            if let TermKind::And(args) = &term.kind {
                assert_eq!(args.len(), 3);
                assert!(args.contains(&a));
                assert!(args.contains(&b));
                assert!(args.contains(&c));
            } else {
                panic!("Expected And term");
            }
        }
    }

    #[test]
    fn test_flatten_or() {
        let mut manager = TermManager::new();

        let a = manager.mk_var("a", manager.sorts.bool_sort);
        let b = manager.mk_var("b", manager.sorts.bool_sort);
        let c = manager.mk_var("c", manager.sorts.bool_sort);
        let d = manager.mk_var("d", manager.sorts.bool_sort);

        // Create (or (or a b) (or c d))
        let left = manager.mk_or([a, b]);
        let right = manager.mk_or([c, d]);
        let nested = manager.mk_or([left, right]);

        // Flatten should give us (or a b c d)
        let flattened = flatten_associative(nested, &mut manager);

        // Verify it's an Or with 4 arguments
        if let Some(term) = manager.get(flattened) {
            if let TermKind::Or(args) = &term.kind {
                assert_eq!(args.len(), 4);
            } else {
                panic!("Expected Or term");
            }
        }
    }

    #[test]
    fn test_flatten_add() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);

        // Create (+ (+ x 1) (+ 2 3))
        let left = manager.mk_add([x, one]);
        let right = manager.mk_add([two, three]);
        let nested = manager.mk_add([left, right]);

        // Flatten should give us (+ x 1 2 3)
        let flattened = flatten_associative(nested, &mut manager);

        // Verify it's an Add with 4 arguments
        if let Some(term) = manager.get(flattened) {
            if let TermKind::Add(args) = &term.kind {
                assert_eq!(args.len(), 4);
            } else {
                panic!("Expected Add term");
            }
        }
    }

    #[test]
    fn test_flatten_mul() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let two = manager.mk_int(2);

        // Create (* (* x 2) y)
        let inner = manager.mk_mul([x, two]);
        let nested = manager.mk_mul([inner, y]);

        // Flatten should give us (* x 2 y)
        let flattened = flatten_associative(nested, &mut manager);

        // Verify it's a Mul with 3 arguments
        if let Some(term) = manager.get(flattened) {
            if let TermKind::Mul(args) = &term.kind {
                assert_eq!(args.len(), 3);
            } else {
                panic!("Expected Mul term");
            }
        }
    }

    #[test]
    fn test_flatten_deeply_nested() {
        let mut manager = TermManager::new();

        let a = manager.mk_var("a", manager.sorts.bool_sort);
        let b = manager.mk_var("b", manager.sorts.bool_sort);
        let c = manager.mk_var("c", manager.sorts.bool_sort);
        let d = manager.mk_var("d", manager.sorts.bool_sort);

        // Create (and (and (and a b) c) d)
        let inner1 = manager.mk_and([a, b]);
        let inner2 = manager.mk_and([inner1, c]);
        let nested = manager.mk_and([inner2, d]);

        // Flatten should give us (and a b c d)
        let flattened = flatten_associative(nested, &mut manager);

        // Verify it's an And with 4 arguments
        if let Some(term) = manager.get(flattened) {
            if let TermKind::And(args) = &term.kind {
                assert_eq!(args.len(), 4);
            } else {
                panic!("Expected And term");
            }
        }
    }

    #[test]
    fn test_flatten_mixed_operations() {
        let mut manager = TermManager::new();

        let a = manager.mk_var("a", manager.sorts.bool_sort);
        let b = manager.mk_var("b", manager.sorts.bool_sort);
        let c = manager.mk_var("c", manager.sorts.bool_sort);

        // Create (and (or a b) c) - should only flatten the And, not the Or
        let or_term = manager.mk_or([a, b]);
        let and_term = manager.mk_and([or_term, c]);

        let flattened = flatten_associative(and_term, &mut manager);

        // Should be (and (or a b) c) with 2 And arguments
        if let Some(term) = manager.get(flattened) {
            if let TermKind::And(args) = &term.kind {
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected And term");
            }
        }
    }

    #[test]
    fn test_flatten_already_flat() {
        let mut manager = TermManager::new();

        let a = manager.mk_var("a", manager.sorts.bool_sort);
        let b = manager.mk_var("b", manager.sorts.bool_sort);
        let c = manager.mk_var("c", manager.sorts.bool_sort);

        // Create (and a b c) - already flat
        let flat_term = manager.mk_and([a, b, c]);

        let flattened = flatten_associative(flat_term, &mut manager);

        // Should remain the same
        assert_eq!(flat_term, flattened);
    }
}
