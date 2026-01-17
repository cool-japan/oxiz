//! DAG traversal utilities for terms
//!
//! This module provides efficient traversal mechanisms for term DAGs,
//! including visitors, iterators, and utility functions for collecting
//! subterms, free variables, and other structural information.

use super::{TermId, TermKind, TermManager};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

/// Visitor trait for traversing term DAGs
pub trait TermVisitor {
    /// Visit a term (pre-order)
    fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
        let _ = (term_id, manager);
        VisitorAction::Continue
    }

    /// Visit a term (post-order, after visiting children)
    fn visit_post(&mut self, term_id: TermId, manager: &TermManager) {
        let _ = (term_id, manager);
    }
}

/// Action to take after visiting a term
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisitorAction {
    /// Continue traversing children
    Continue,
    /// Skip children of this term
    SkipChildren,
    /// Stop traversal completely
    Stop,
}

/// Traverse a term DAG with a visitor
pub fn traverse<V: TermVisitor>(
    term_id: TermId,
    manager: &TermManager,
    visitor: &mut V,
) -> Result<(), TraversalError> {
    let mut visited = FxHashSet::default();
    let mut stack = vec![(term_id, false)]; // (term_id, post_visit)

    while let Some((current_id, is_post)) = stack.pop() {
        if is_post {
            visitor.visit_post(current_id, manager);
            continue;
        }

        // Pre-visit
        let action = visitor.visit_pre(current_id, manager);

        match action {
            VisitorAction::Stop => return Ok(()),
            VisitorAction::SkipChildren => continue,
            VisitorAction::Continue => {}
        }

        // Avoid revisiting (DAG not tree)
        if !visited.insert(current_id) {
            continue;
        }

        // Schedule post-visit
        stack.push((current_id, true));

        // Push children for pre-visit
        if let Some(term) = manager.get(current_id) {
            let children = get_children(&term.kind);
            for &child in children.iter().rev() {
                stack.push((child, false));
            }
        }
    }

    Ok(())
}

/// Traversal error
#[derive(Debug, Clone, thiserror::Error)]
pub enum TraversalError {
    /// Term not found
    #[error("Term not found: {0:?}")]
    TermNotFound(TermId),
}

/// Get immediate children of a term
#[must_use]
pub fn get_children(kind: &TermKind) -> SmallVec<[TermId; 4]> {
    let mut children = SmallVec::new();

    match kind {
        // Nullary
        TermKind::True
        | TermKind::False
        | TermKind::IntConst(_)
        | TermKind::RealConst(_)
        | TermKind::BitVecConst { .. }
        | TermKind::StringLit(_)
        | TermKind::Var(_) => {}

        // Unary
        TermKind::Not(a)
        | TermKind::Neg(a)
        | TermKind::BvNot(a)
        | TermKind::StrLen(a)
        | TermKind::StrToInt(a)
        | TermKind::IntToStr(a) => {
            children.push(*a);
        }

        TermKind::BvExtract { arg, .. } => {
            children.push(*arg);
        }

        // Binary
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
        | TermKind::BvSle(a, b) => {
            children.push(*a);
            children.push(*b);
        }

        // Ternary
        TermKind::Ite(c, t, e)
        | TermKind::Store(c, t, e)
        | TermKind::StrSubstr(c, t, e)
        | TermKind::StrIndexOf(c, t, e)
        | TermKind::StrReplace(c, t, e)
        | TermKind::StrReplaceAll(c, t, e) => {
            children.push(*c);
            children.push(*t);
            children.push(*e);
        }

        // N-ary
        TermKind::And(args)
        | TermKind::Or(args)
        | TermKind::Add(args)
        | TermKind::Mul(args)
        | TermKind::Distinct(args) => {
            children.extend(args.iter().copied());
        }

        // Function application
        TermKind::Apply { args, .. } => {
            children.extend(args.iter().copied());
        }

        // Algebraic datatypes
        TermKind::DtConstructor { args, .. } => {
            children.extend(args.iter().copied());
        }
        TermKind::DtTester { arg, .. } | TermKind::DtSelector { arg, .. } => {
            children.push(*arg);
        }

        // Quantifiers
        TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
            children.push(*body);
        }

        // Let bindings
        TermKind::Let { bindings, body } => {
            for (_, value) in bindings {
                children.push(*value);
            }
            children.push(*body);
        }

        // Floating-point literals have no children
        TermKind::FpLit { .. }
        | TermKind::FpPlusInfinity { .. }
        | TermKind::FpMinusInfinity { .. }
        | TermKind::FpPlusZero { .. }
        | TermKind::FpMinusZero { .. }
        | TermKind::FpNaN { .. } => {}

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
        | TermKind::FpToReal(a)
        | TermKind::FpToFp { arg: a, .. }
        | TermKind::FpToSBV { arg: a, .. }
        | TermKind::FpToUBV { arg: a, .. }
        | TermKind::RealToFp { arg: a, .. }
        | TermKind::SBVToFp { arg: a, .. }
        | TermKind::UBVToFp { arg: a, .. } => {
            children.push(*a);
        }

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
            children.push(*a);
            children.push(*b);
        }

        // Ternary FP operations
        TermKind::FpFma(_, a, b, c) => {
            children.push(*a);
            children.push(*b);
            children.push(*c);
        }

        // Match expression
        TermKind::Match { scrutinee, cases } => {
            children.push(*scrutinee);
            for case in cases {
                children.push(case.body);
            }
        }
    }

    children
}

/// Collect all subterms (including the term itself) in post-order
#[must_use]
pub fn collect_subterms(term_id: TermId, manager: &TermManager) -> Vec<TermId> {
    struct Collector {
        subterms: Vec<TermId>,
        visited: FxHashSet<TermId>,
    }

    impl TermVisitor for Collector {
        fn visit_post(&mut self, term_id: TermId, _manager: &TermManager) {
            if self.visited.insert(term_id) {
                self.subterms.push(term_id);
            }
        }
    }

    let mut collector = Collector {
        subterms: Vec::new(),
        visited: FxHashSet::default(),
    };

    let _ = traverse(term_id, manager, &mut collector);
    collector.subterms
}

/// Collect all free variables in a term
#[must_use]
pub fn collect_free_vars(term_id: TermId, manager: &TermManager) -> FxHashSet<TermId> {
    struct VarCollector {
        vars: FxHashSet<TermId>,
        bound_vars: Vec<FxHashSet<lasso::Spur>>,
    }

    impl TermVisitor for VarCollector {
        fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
            if let Some(term) = manager.get(term_id) {
                match &term.kind {
                    TermKind::Var(name) => {
                        // Check if this variable is bound
                        let is_bound = self.bound_vars.iter().any(|scope| scope.contains(name));
                        if !is_bound {
                            self.vars.insert(term_id);
                        }
                    }
                    TermKind::Forall { vars, .. } | TermKind::Exists { vars, .. } => {
                        // Push new scope with bound variables
                        let mut scope = FxHashSet::default();
                        for (var_name, _) in vars {
                            scope.insert(*var_name);
                        }
                        self.bound_vars.push(scope);
                    }
                    TermKind::Let { bindings, .. } => {
                        // Push new scope with let-bound variables
                        let mut scope = FxHashSet::default();
                        for (var_name, _) in bindings {
                            scope.insert(*var_name);
                        }
                        self.bound_vars.push(scope);
                    }
                    _ => {}
                }
            }
            VisitorAction::Continue
        }

        fn visit_post(&mut self, term_id: TermId, manager: &TermManager) {
            // Pop scopes when exiting quantifiers/let bindings
            if let Some(term) = manager.get(term_id) {
                match term.kind {
                    TermKind::Forall { .. } | TermKind::Exists { .. } | TermKind::Let { .. } => {
                        self.bound_vars.pop();
                    }
                    _ => {}
                }
            }
        }
    }

    let mut collector = VarCollector {
        vars: FxHashSet::default(),
        bound_vars: Vec::new(),
    };

    let _ = traverse(term_id, manager, &mut collector);
    collector.vars
}

/// Count the number of nodes in the term DAG
#[must_use]
pub fn count_nodes(term_id: TermId, manager: &TermManager) -> usize {
    collect_subterms(term_id, manager).len()
}

/// Compute the depth (height) of a term
#[must_use]
pub fn compute_depth(term_id: TermId, manager: &TermManager) -> usize {
    struct DepthCalculator {
        depths: rustc_hash::FxHashMap<TermId, usize>,
    }

    impl TermVisitor for DepthCalculator {
        fn visit_post(&mut self, term_id: TermId, manager: &TermManager) {
            if let Some(term) = manager.get(term_id) {
                let children = get_children(&term.kind);
                let max_child_depth = children
                    .iter()
                    .filter_map(|&child| self.depths.get(&child))
                    .max()
                    .unwrap_or(&0);
                self.depths.insert(term_id, max_child_depth + 1);
            }
        }
    }

    let mut calculator = DepthCalculator {
        depths: rustc_hash::FxHashMap::default(),
    };

    let _ = traverse(term_id, manager, &mut calculator);
    calculator.depths.get(&term_id).copied().unwrap_or(0)
}

/// Check if a term contains a specific subterm
#[must_use]
pub fn contains_term(haystack: TermId, needle: TermId, manager: &TermManager) -> bool {
    struct ContainsChecker {
        needle: TermId,
        found: bool,
    }

    impl TermVisitor for ContainsChecker {
        fn visit_pre(&mut self, term_id: TermId, _manager: &TermManager) -> VisitorAction {
            if term_id == self.needle {
                self.found = true;
                VisitorAction::Stop
            } else {
                VisitorAction::Continue
            }
        }
    }

    let mut checker = ContainsChecker {
        needle,
        found: false,
    };

    let _ = traverse(haystack, manager, &mut checker);
    checker.found
}

/// Map a function over all subterms (bottom-up)
pub fn map_terms<F>(term_id: TermId, manager: &mut TermManager, mut f: F) -> TermId
where
    F: FnMut(TermId, &TermManager) -> Option<TermId>,
{
    use rustc_hash::FxHashMap;

    let mut cache: FxHashMap<TermId, TermId> = FxHashMap::default();
    let subterms = collect_subterms(term_id, manager);

    for &subterm_id in &subterms {
        if let Some(new_id) = f(subterm_id, manager) {
            cache.insert(subterm_id, new_id);
        } else if let Some(term) = manager.get(subterm_id) {
            // Rebuild the term with potentially transformed children
            let new_kind = transform_children(&term.kind, &cache);
            if new_kind != term.kind {
                let new_id = manager.intern(new_kind, term.sort);
                cache.insert(subterm_id, new_id);
            }
        }
    }

    cache.get(&term_id).copied().unwrap_or(term_id)
}

/// Transform children of a term kind using a substitution cache
fn transform_children(kind: &TermKind, cache: &rustc_hash::FxHashMap<TermId, TermId>) -> TermKind {
    let subst = |id: &TermId| cache.get(id).copied().unwrap_or(*id);

    match kind {
        // Nullary - no changes
        k @ (TermKind::True
        | TermKind::False
        | TermKind::IntConst(_)
        | TermKind::RealConst(_)
        | TermKind::BitVecConst { .. }
        | TermKind::StringLit(_)
        | TermKind::Var(_)) => k.clone(),

        // Unary
        TermKind::Not(a) => TermKind::Not(subst(a)),
        TermKind::Neg(a) => TermKind::Neg(subst(a)),
        TermKind::BvNot(a) => TermKind::BvNot(subst(a)),
        TermKind::StrLen(a) => TermKind::StrLen(subst(a)),
        TermKind::StrToInt(a) => TermKind::StrToInt(subst(a)),
        TermKind::IntToStr(a) => TermKind::IntToStr(subst(a)),

        TermKind::BvExtract { high, low, arg } => TermKind::BvExtract {
            high: *high,
            low: *low,
            arg: subst(arg),
        },

        // Binary
        TermKind::Implies(a, b) => TermKind::Implies(subst(a), subst(b)),
        TermKind::Xor(a, b) => TermKind::Xor(subst(a), subst(b)),
        TermKind::Eq(a, b) => TermKind::Eq(subst(a), subst(b)),
        TermKind::Sub(a, b) => TermKind::Sub(subst(a), subst(b)),
        TermKind::Div(a, b) => TermKind::Div(subst(a), subst(b)),
        TermKind::Mod(a, b) => TermKind::Mod(subst(a), subst(b)),
        TermKind::Lt(a, b) => TermKind::Lt(subst(a), subst(b)),
        TermKind::Le(a, b) => TermKind::Le(subst(a), subst(b)),
        TermKind::Gt(a, b) => TermKind::Gt(subst(a), subst(b)),
        TermKind::Ge(a, b) => TermKind::Ge(subst(a), subst(b)),
        TermKind::Select(a, b) => TermKind::Select(subst(a), subst(b)),
        TermKind::BvConcat(a, b) => TermKind::BvConcat(subst(a), subst(b)),
        TermKind::BvAnd(a, b) => TermKind::BvAnd(subst(a), subst(b)),
        TermKind::BvOr(a, b) => TermKind::BvOr(subst(a), subst(b)),
        TermKind::BvXor(a, b) => TermKind::BvXor(subst(a), subst(b)),
        TermKind::BvAdd(a, b) => TermKind::BvAdd(subst(a), subst(b)),
        TermKind::BvSub(a, b) => TermKind::BvSub(subst(a), subst(b)),
        TermKind::BvMul(a, b) => TermKind::BvMul(subst(a), subst(b)),
        TermKind::BvUdiv(a, b) => TermKind::BvUdiv(subst(a), subst(b)),
        TermKind::BvSdiv(a, b) => TermKind::BvSdiv(subst(a), subst(b)),
        TermKind::BvUrem(a, b) => TermKind::BvUrem(subst(a), subst(b)),
        TermKind::BvSrem(a, b) => TermKind::BvSrem(subst(a), subst(b)),
        TermKind::BvShl(a, b) => TermKind::BvShl(subst(a), subst(b)),
        TermKind::BvLshr(a, b) => TermKind::BvLshr(subst(a), subst(b)),
        TermKind::BvAshr(a, b) => TermKind::BvAshr(subst(a), subst(b)),
        TermKind::BvUlt(a, b) => TermKind::BvUlt(subst(a), subst(b)),
        TermKind::BvUle(a, b) => TermKind::BvUle(subst(a), subst(b)),
        TermKind::BvSlt(a, b) => TermKind::BvSlt(subst(a), subst(b)),
        TermKind::BvSle(a, b) => TermKind::BvSle(subst(a), subst(b)),
        TermKind::StrConcat(a, b) => TermKind::StrConcat(subst(a), subst(b)),
        TermKind::StrAt(a, b) => TermKind::StrAt(subst(a), subst(b)),
        TermKind::StrContains(a, b) => TermKind::StrContains(subst(a), subst(b)),
        TermKind::StrPrefixOf(a, b) => TermKind::StrPrefixOf(subst(a), subst(b)),
        TermKind::StrSuffixOf(a, b) => TermKind::StrSuffixOf(subst(a), subst(b)),
        TermKind::StrInRe(a, b) => TermKind::StrInRe(subst(a), subst(b)),

        // Ternary
        TermKind::Ite(c, t, e) => TermKind::Ite(subst(c), subst(t), subst(e)),
        TermKind::Store(a, i, v) => TermKind::Store(subst(a), subst(i), subst(v)),
        TermKind::StrSubstr(s, i, n) => TermKind::StrSubstr(subst(s), subst(i), subst(n)),
        TermKind::StrIndexOf(s, t, o) => TermKind::StrIndexOf(subst(s), subst(t), subst(o)),
        TermKind::StrReplace(s, p, r) => TermKind::StrReplace(subst(s), subst(p), subst(r)),
        TermKind::StrReplaceAll(s, p, r) => TermKind::StrReplaceAll(subst(s), subst(p), subst(r)),

        // N-ary
        TermKind::And(args) => TermKind::And(args.iter().map(subst).collect()),
        TermKind::Or(args) => TermKind::Or(args.iter().map(subst).collect()),
        TermKind::Add(args) => TermKind::Add(args.iter().map(subst).collect()),
        TermKind::Mul(args) => TermKind::Mul(args.iter().map(subst).collect()),
        TermKind::Distinct(args) => TermKind::Distinct(args.iter().map(subst).collect()),

        // Function application
        TermKind::Apply { func, args } => TermKind::Apply {
            func: *func,
            args: args.iter().map(subst).collect(),
        },

        // Algebraic datatypes
        TermKind::DtConstructor { constructor, args } => TermKind::DtConstructor {
            constructor: *constructor,
            args: args.iter().map(subst).collect(),
        },
        TermKind::DtTester { constructor, arg } => TermKind::DtTester {
            constructor: *constructor,
            arg: subst(arg),
        },
        TermKind::DtSelector { selector, arg } => TermKind::DtSelector {
            selector: *selector,
            arg: subst(arg),
        },

        // Quantifiers
        TermKind::Forall {
            vars,
            body,
            patterns,
        } => TermKind::Forall {
            vars: vars.clone(),
            body: subst(body),
            patterns: patterns
                .iter()
                .map(|p| p.iter().map(subst).collect())
                .collect(),
        },
        TermKind::Exists {
            vars,
            body,
            patterns,
        } => TermKind::Exists {
            vars: vars.clone(),
            body: subst(body),
            patterns: patterns
                .iter()
                .map(|p| p.iter().map(subst).collect())
                .collect(),
        },

        // Let bindings
        TermKind::Let { bindings, body } => TermKind::Let {
            bindings: bindings
                .iter()
                .map(|(name, value)| (*name, subst(value)))
                .collect(),
            body: subst(body),
        },

        // Floating-point literals - no transformation needed
        k @ (TermKind::FpLit { .. }
        | TermKind::FpPlusInfinity { .. }
        | TermKind::FpMinusInfinity { .. }
        | TermKind::FpPlusZero { .. }
        | TermKind::FpMinusZero { .. }
        | TermKind::FpNaN { .. }) => k.clone(),

        // Unary FP operations
        TermKind::FpAbs(a) => TermKind::FpAbs(subst(a)),
        TermKind::FpNeg(a) => TermKind::FpNeg(subst(a)),
        TermKind::FpSqrt(rm, a) => TermKind::FpSqrt(*rm, subst(a)),
        TermKind::FpRoundToIntegral(rm, a) => TermKind::FpRoundToIntegral(*rm, subst(a)),
        TermKind::FpIsNormal(a) => TermKind::FpIsNormal(subst(a)),
        TermKind::FpIsSubnormal(a) => TermKind::FpIsSubnormal(subst(a)),
        TermKind::FpIsZero(a) => TermKind::FpIsZero(subst(a)),
        TermKind::FpIsInfinite(a) => TermKind::FpIsInfinite(subst(a)),
        TermKind::FpIsNaN(a) => TermKind::FpIsNaN(subst(a)),
        TermKind::FpIsNegative(a) => TermKind::FpIsNegative(subst(a)),
        TermKind::FpIsPositive(a) => TermKind::FpIsPositive(subst(a)),
        TermKind::FpToReal(a) => TermKind::FpToReal(subst(a)),

        // Binary FP operations
        TermKind::FpAdd(rm, a, b) => TermKind::FpAdd(*rm, subst(a), subst(b)),
        TermKind::FpSub(rm, a, b) => TermKind::FpSub(*rm, subst(a), subst(b)),
        TermKind::FpMul(rm, a, b) => TermKind::FpMul(*rm, subst(a), subst(b)),
        TermKind::FpDiv(rm, a, b) => TermKind::FpDiv(*rm, subst(a), subst(b)),
        TermKind::FpRem(a, b) => TermKind::FpRem(subst(a), subst(b)),
        TermKind::FpMin(a, b) => TermKind::FpMin(subst(a), subst(b)),
        TermKind::FpMax(a, b) => TermKind::FpMax(subst(a), subst(b)),
        TermKind::FpLeq(a, b) => TermKind::FpLeq(subst(a), subst(b)),
        TermKind::FpLt(a, b) => TermKind::FpLt(subst(a), subst(b)),
        TermKind::FpGeq(a, b) => TermKind::FpGeq(subst(a), subst(b)),
        TermKind::FpGt(a, b) => TermKind::FpGt(subst(a), subst(b)),
        TermKind::FpEq(a, b) => TermKind::FpEq(subst(a), subst(b)),

        // Ternary FP operations
        TermKind::FpFma(rm, a, b, c) => TermKind::FpFma(*rm, subst(a), subst(b), subst(c)),

        // FP conversions
        TermKind::FpToFp { rm, arg, eb, sb } => TermKind::FpToFp {
            rm: *rm,
            arg: subst(arg),
            eb: *eb,
            sb: *sb,
        },
        TermKind::FpToSBV { rm, arg, width } => TermKind::FpToSBV {
            rm: *rm,
            arg: subst(arg),
            width: *width,
        },
        TermKind::FpToUBV { rm, arg, width } => TermKind::FpToUBV {
            rm: *rm,
            arg: subst(arg),
            width: *width,
        },
        TermKind::RealToFp { rm, arg, eb, sb } => TermKind::RealToFp {
            rm: *rm,
            arg: subst(arg),
            eb: *eb,
            sb: *sb,
        },
        TermKind::SBVToFp { rm, arg, eb, sb } => TermKind::SBVToFp {
            rm: *rm,
            arg: subst(arg),
            eb: *eb,
            sb: *sb,
        },
        TermKind::UBVToFp { rm, arg, eb, sb } => TermKind::UBVToFp {
            rm: *rm,
            arg: subst(arg),
            eb: *eb,
            sb: *sb,
        },

        // Match expression
        TermKind::Match { scrutinee, cases } => TermKind::Match {
            scrutinee: subst(scrutinee),
            cases: cases
                .iter()
                .map(|c| crate::ast::term::MatchCase {
                    constructor: c.constructor,
                    bindings: c.bindings.clone(),
                    body: subst(&c.body),
                })
                .collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_collect_subterms() {
        let mut manager = TermManager::new();

        // (+ 1 2)
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let sum = manager.mk_add([one, two]);

        let subterms = collect_subterms(sum, &manager);
        // Should contain: 1, 2, (+ 1 2)
        assert_eq!(subterms.len(), 3);
        assert!(subterms.contains(&one));
        assert!(subterms.contains(&two));
        assert!(subterms.contains(&sum));
    }

    #[test]
    fn test_collect_free_vars() {
        let mut manager = TermManager::new();

        // (+ x y)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y]);

        let free_vars = collect_free_vars(sum, &manager);
        assert_eq!(free_vars.len(), 2);
        assert!(free_vars.contains(&x));
        assert!(free_vars.contains(&y));
    }

    #[test]
    fn test_compute_depth() {
        let mut manager = TermManager::new();

        // (+ (+ 1 2) 3)
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let inner_sum = manager.mk_add([one, two]);
        let outer_sum = manager.mk_add([inner_sum, three]);

        assert_eq!(compute_depth(one, &manager), 1);
        assert_eq!(compute_depth(inner_sum, &manager), 2);
        assert_eq!(compute_depth(outer_sum, &manager), 3);
    }

    #[test]
    fn test_contains_term() {
        let mut manager = TermManager::new();

        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let sum = manager.mk_add([one, two]);

        assert!(contains_term(sum, one, &manager));
        assert!(contains_term(sum, two, &manager));
        assert!(!contains_term(sum, three, &manager));
    }

    #[test]
    fn test_count_nodes() {
        let mut manager = TermManager::new();

        // (+ 1 2 3)
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let sum = manager.mk_add([one, two, three]);

        // Should count: 1, 2, 3, (+ 1 2 3) = 4 nodes
        assert_eq!(count_nodes(sum, &manager), 4);
    }

    #[test]
    fn test_visitor_pattern() {
        struct CountVisitor {
            count: usize,
        }

        impl TermVisitor for CountVisitor {
            fn visit_pre(&mut self, _term_id: TermId, _manager: &TermManager) -> VisitorAction {
                self.count += 1;
                VisitorAction::Continue
            }
        }

        let mut manager = TermManager::new();
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let sum = manager.mk_add([one, two]);

        let mut visitor = CountVisitor { count: 0 };
        traverse(sum, &manager, &mut visitor).unwrap();

        // Should visit each node once (due to visited set)
        assert_eq!(visitor.count, 3);
    }
}
