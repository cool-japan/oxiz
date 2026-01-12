//! Probe system for goal analysis
//!
//! Probes are functions that analyze goals and return numeric values.
//! They are used to guide tactic selection based on goal properties.

use crate::ast::traversal::{
    TermVisitor, VisitorAction, collect_subterms, compute_depth, traverse,
};
use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashSet;

use super::Goal;

/// A probe analyzes a goal and returns a numeric value.
///
/// Probes are used for adaptive tactic selection, allowing tactics
/// to be chosen based on goal properties.
pub trait Probe: Send + Sync {
    /// Get the name of this probe
    fn name(&self) -> &str;

    /// Evaluate the probe on a goal
    ///
    /// Returns a numeric value that can be used for comparison
    /// or conditional branching.
    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64;

    /// Get a description of this probe
    fn description(&self) -> &str {
        ""
    }
}

/// Probe that returns the number of assertions in a goal
#[derive(Debug, Default, Clone, Copy)]
pub struct SizeProbe;

impl Probe for SizeProbe {
    fn name(&self) -> &str {
        "size"
    }

    fn evaluate(&self, goal: &Goal, _manager: &TermManager) -> f64 {
        goal.assertions.len() as f64
    }

    fn description(&self) -> &str {
        "Returns the number of assertions in the goal"
    }
}

/// Probe that returns the total number of term nodes
#[derive(Debug, Default, Clone, Copy)]
pub struct NodeCountProbe;

impl Probe for NodeCountProbe {
    fn name(&self) -> &str {
        "num-exprs"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let mut total = 0;
        let mut seen = FxHashSet::default();

        for &assertion in &goal.assertions {
            let subterms = collect_subterms(assertion, manager);
            for term in subterms {
                if seen.insert(term) {
                    total += 1;
                }
            }
        }

        total as f64
    }

    fn description(&self) -> &str {
        "Returns the total number of unique term nodes"
    }
}

/// Probe that returns the maximum depth of any assertion
#[derive(Debug, Default, Clone, Copy)]
pub struct DepthProbe;

impl Probe for DepthProbe {
    fn name(&self) -> &str {
        "depth"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        goal.assertions
            .iter()
            .map(|&a| compute_depth(a, manager))
            .max()
            .unwrap_or(0) as f64
    }

    fn description(&self) -> &str {
        "Returns the maximum depth of any assertion"
    }
}

/// Probe that checks if the goal contains quantifiers
#[derive(Debug, Default, Clone, Copy)]
pub struct HasQuantifierProbe;

impl Probe for HasQuantifierProbe {
    fn name(&self) -> &str {
        "has-quantifiers"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        if super::quantifier::goal_has_quantifiers(goal, manager) {
            1.0
        } else {
            0.0
        }
    }

    fn description(&self) -> &str {
        "Returns 1.0 if the goal contains quantifiers, 0.0 otherwise"
    }
}

/// Probe that checks if all arithmetic is linear
#[derive(Debug, Default, Clone, Copy)]
pub struct IsLinearProbe;

impl IsLinearProbe {
    fn is_linear_term(term_id: TermId, manager: &TermManager) -> bool {
        struct LinearChecker {
            is_linear: bool,
        }

        impl TermVisitor for LinearChecker {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id) {
                    match &term.kind {
                        // Multiplication is only linear if at least one operand is constant
                        TermKind::Mul(args) => {
                            let const_count = args
                                .iter()
                                .filter(|&&a| manager.get(a).map(|t| t.is_const()).unwrap_or(false))
                                .count();
                            // Need at least n-1 constants for n arguments to be linear
                            if const_count < args.len().saturating_sub(1) {
                                self.is_linear = false;
                                return VisitorAction::Stop;
                            }
                        }
                        // Division by non-constant is nonlinear
                        TermKind::Div(_, divisor) => {
                            if !manager.get(*divisor).map(|t| t.is_const()).unwrap_or(false) {
                                self.is_linear = false;
                                return VisitorAction::Stop;
                            }
                        }
                        // Mod by non-constant is nonlinear
                        TermKind::Mod(_, divisor) => {
                            if !manager.get(*divisor).map(|t| t.is_const()).unwrap_or(false) {
                                self.is_linear = false;
                                return VisitorAction::Stop;
                            }
                        }
                        _ => {}
                    }
                }
                VisitorAction::Continue
            }
        }

        let mut checker = LinearChecker { is_linear: true };
        let _ = traverse(term_id, manager, &mut checker);
        checker.is_linear
    }
}

impl Probe for IsLinearProbe {
    fn name(&self) -> &str {
        "is-linear"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let all_linear = goal
            .assertions
            .iter()
            .all(|&a| Self::is_linear_term(a, manager));
        if all_linear { 1.0 } else { 0.0 }
    }

    fn description(&self) -> &str {
        "Returns 1.0 if all arithmetic is linear, 0.0 otherwise"
    }
}

/// Probe that checks if the goal contains bit-vector operations
#[derive(Debug, Default, Clone, Copy)]
pub struct HasBitVectorProbe;

impl HasBitVectorProbe {
    fn has_bv(term_id: TermId, manager: &TermManager) -> bool {
        struct BvChecker {
            found: bool,
        }

        impl TermVisitor for BvChecker {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id) {
                    let is_bv = matches!(
                        term.kind,
                        TermKind::BitVecConst { .. }
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
                            | TermKind::BvConcat(_, _)
                            | TermKind::BvExtract { .. }
                    );
                    if is_bv {
                        self.found = true;
                        return VisitorAction::Stop;
                    }
                }
                VisitorAction::Continue
            }
        }

        let mut checker = BvChecker { found: false };
        let _ = traverse(term_id, manager, &mut checker);
        checker.found
    }
}

impl Probe for HasBitVectorProbe {
    fn name(&self) -> &str {
        "has-bitvector"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let has_bv = goal.assertions.iter().any(|&a| Self::has_bv(a, manager));
        if has_bv { 1.0 } else { 0.0 }
    }

    fn description(&self) -> &str {
        "Returns 1.0 if the goal contains bit-vector operations, 0.0 otherwise"
    }
}

/// Probe that checks if the goal contains array operations
#[derive(Debug, Default, Clone, Copy)]
pub struct HasArrayProbe;

impl HasArrayProbe {
    fn has_array(term_id: TermId, manager: &TermManager) -> bool {
        struct ArrayChecker {
            found: bool,
        }

        impl TermVisitor for ArrayChecker {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id) {
                    let is_array =
                        matches!(term.kind, TermKind::Select(_, _) | TermKind::Store(_, _, _));
                    if is_array {
                        self.found = true;
                        return VisitorAction::Stop;
                    }
                }
                VisitorAction::Continue
            }
        }

        let mut checker = ArrayChecker { found: false };
        let _ = traverse(term_id, manager, &mut checker);
        checker.found
    }
}

impl Probe for HasArrayProbe {
    fn name(&self) -> &str {
        "has-array"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let has_array = goal.assertions.iter().any(|&a| Self::has_array(a, manager));
        if has_array { 1.0 } else { 0.0 }
    }

    fn description(&self) -> &str {
        "Returns 1.0 if the goal contains array operations, 0.0 otherwise"
    }
}

/// Probe that checks if the goal contains floating-point operations
#[derive(Debug, Default, Clone, Copy)]
pub struct HasFloatingPointProbe;

impl HasFloatingPointProbe {
    fn has_fp(term_id: TermId, manager: &TermManager) -> bool {
        struct FpChecker {
            found: bool,
        }

        impl TermVisitor for FpChecker {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id) {
                    let is_fp = matches!(
                        term.kind,
                        TermKind::FpLit { .. }
                            | TermKind::FpPlusInfinity { .. }
                            | TermKind::FpMinusInfinity { .. }
                            | TermKind::FpPlusZero { .. }
                            | TermKind::FpMinusZero { .. }
                            | TermKind::FpNaN { .. }
                            | TermKind::FpAbs(_)
                            | TermKind::FpNeg(_)
                            | TermKind::FpAdd(_, _, _)
                            | TermKind::FpSub(_, _, _)
                            | TermKind::FpMul(_, _, _)
                            | TermKind::FpDiv(_, _, _)
                            | TermKind::FpRem(_, _)
                            | TermKind::FpSqrt(_, _)
                            | TermKind::FpFma(_, _, _, _)
                            | TermKind::FpRoundToIntegral(_, _)
                            | TermKind::FpMin(_, _)
                            | TermKind::FpMax(_, _)
                            | TermKind::FpLeq(_, _)
                            | TermKind::FpLt(_, _)
                            | TermKind::FpGeq(_, _)
                            | TermKind::FpGt(_, _)
                            | TermKind::FpEq(_, _)
                            | TermKind::FpIsNormal(_)
                            | TermKind::FpIsSubnormal(_)
                            | TermKind::FpIsZero(_)
                            | TermKind::FpIsInfinite(_)
                            | TermKind::FpIsNaN(_)
                            | TermKind::FpIsNegative(_)
                            | TermKind::FpIsPositive(_)
                            | TermKind::FpToReal(_)
                            | TermKind::FpToFp { .. }
                            | TermKind::FpToSBV { .. }
                            | TermKind::FpToUBV { .. }
                            | TermKind::RealToFp { .. }
                            | TermKind::SBVToFp { .. }
                            | TermKind::UBVToFp { .. }
                    );
                    if is_fp {
                        self.found = true;
                        return VisitorAction::Stop;
                    }
                }
                VisitorAction::Continue
            }
        }

        let mut checker = FpChecker { found: false };
        let _ = traverse(term_id, manager, &mut checker);
        checker.found
    }
}

impl Probe for HasFloatingPointProbe {
    fn name(&self) -> &str {
        "has-floating-point"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let has_fp = goal.assertions.iter().any(|&a| Self::has_fp(a, manager));
        if has_fp { 1.0 } else { 0.0 }
    }

    fn description(&self) -> &str {
        "Returns 1.0 if the goal contains floating-point operations, 0.0 otherwise"
    }
}

/// Probe that counts the number of distinct variables
#[derive(Debug, Default, Clone, Copy)]
pub struct NumVarsProbe;

impl Probe for NumVarsProbe {
    fn name(&self) -> &str {
        "num-vars"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let mut vars = FxHashSet::default();

        struct VarCollector<'a> {
            vars: &'a mut FxHashSet<TermId>,
        }

        impl TermVisitor for VarCollector<'_> {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id)
                    && matches!(term.kind, TermKind::Var(_))
                {
                    self.vars.insert(term_id);
                }
                VisitorAction::Continue
            }
        }

        for &assertion in &goal.assertions {
            let mut collector = VarCollector { vars: &mut vars };
            let _ = traverse(assertion, manager, &mut collector);
        }

        vars.len() as f64
    }

    fn description(&self) -> &str {
        "Returns the number of distinct variables in the goal"
    }
}

/// Probe that counts the number of boolean connectives
#[derive(Debug, Default, Clone, Copy)]
pub struct NumConnectivesProbe;

impl Probe for NumConnectivesProbe {
    fn name(&self) -> &str {
        "num-connectives"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        struct ConnectiveCounter {
            count: usize,
        }

        impl TermVisitor for ConnectiveCounter {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id) {
                    let is_connective = matches!(
                        term.kind,
                        TermKind::Not(_)
                            | TermKind::And(_)
                            | TermKind::Or(_)
                            | TermKind::Xor(_, _)
                            | TermKind::Implies(_, _)
                            | TermKind::Ite(_, _, _)
                    );
                    if is_connective {
                        self.count += 1;
                    }
                }
                VisitorAction::Continue
            }
        }

        let mut total = 0;
        for &assertion in &goal.assertions {
            let mut counter = ConnectiveCounter { count: 0 };
            let _ = traverse(assertion, manager, &mut counter);
            total += counter.count;
        }

        total as f64
    }

    fn description(&self) -> &str {
        "Returns the number of boolean connectives in the goal"
    }
}

/// Constant probe that always returns a fixed value
#[derive(Debug, Clone, Copy)]
pub struct ConstProbe {
    value: f64,
}

impl ConstProbe {
    /// Create a new constant probe
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl Probe for ConstProbe {
    fn name(&self) -> &str {
        "const"
    }

    fn evaluate(&self, _goal: &Goal, _manager: &TermManager) -> f64 {
        self.value
    }

    fn description(&self) -> &str {
        "Returns a constant value"
    }
}

/// Negation probe: returns -p
#[derive(Debug)]
pub struct NegProbe<P: Probe> {
    inner: P,
}

impl<P: Probe> NegProbe<P> {
    /// Create a new negation probe
    #[must_use]
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<P: Probe> Probe for NegProbe<P> {
    fn name(&self) -> &str {
        "neg"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        -self.inner.evaluate(goal, manager)
    }
}

/// Addition probe: returns p1 + p2
#[derive(Debug)]
pub struct AddProbe<P1: Probe, P2: Probe> {
    left: P1,
    right: P2,
}

impl<P1: Probe, P2: Probe> AddProbe<P1, P2> {
    /// Create a new addition probe
    #[must_use]
    pub fn new(left: P1, right: P2) -> Self {
        Self { left, right }
    }
}

impl<P1: Probe, P2: Probe> Probe for AddProbe<P1, P2> {
    fn name(&self) -> &str {
        "add"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        self.left.evaluate(goal, manager) + self.right.evaluate(goal, manager)
    }
}

/// Comparison probe: returns 1.0 if p1 < p2, 0.0 otherwise
#[derive(Debug)]
pub struct LtProbe<P1: Probe, P2: Probe> {
    left: P1,
    right: P2,
}

impl<P1: Probe, P2: Probe> LtProbe<P1, P2> {
    /// Create a new less-than probe
    #[must_use]
    pub fn new(left: P1, right: P2) -> Self {
        Self { left, right }
    }
}

impl<P1: Probe, P2: Probe> Probe for LtProbe<P1, P2> {
    fn name(&self) -> &str {
        "lt"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        if self.left.evaluate(goal, manager) < self.right.evaluate(goal, manager) {
            1.0
        } else {
            0.0
        }
    }
}

/// And probe: returns 1.0 if both probes return non-zero
#[derive(Debug)]
pub struct AndProbe<P1: Probe, P2: Probe> {
    left: P1,
    right: P2,
}

impl<P1: Probe, P2: Probe> AndProbe<P1, P2> {
    /// Create a new conjunction probe
    #[must_use]
    pub fn new(left: P1, right: P2) -> Self {
        Self { left, right }
    }
}

impl<P1: Probe, P2: Probe> Probe for AndProbe<P1, P2> {
    fn name(&self) -> &str {
        "and"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let l = self.left.evaluate(goal, manager);
        if l == 0.0 {
            return 0.0;
        }
        let r = self.right.evaluate(goal, manager);
        if r == 0.0 {
            return 0.0;
        }
        1.0
    }
}

/// Or probe: returns 1.0 if either probe returns non-zero
#[derive(Debug)]
pub struct OrProbe<P1: Probe, P2: Probe> {
    left: P1,
    right: P2,
}

impl<P1: Probe, P2: Probe> OrProbe<P1, P2> {
    /// Create a new disjunction probe
    #[must_use]
    pub fn new(left: P1, right: P2) -> Self {
        Self { left, right }
    }
}

impl<P1: Probe, P2: Probe> Probe for OrProbe<P1, P2> {
    fn name(&self) -> &str {
        "or"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        let l = self.left.evaluate(goal, manager);
        if l != 0.0 {
            return 1.0;
        }
        let r = self.right.evaluate(goal, manager);
        if r != 0.0 {
            return 1.0;
        }
        0.0
    }
}

/// Not probe: returns 1.0 if inner probe returns 0.0, otherwise 0.0
#[derive(Debug)]
pub struct NotProbe<P: Probe> {
    inner: P,
}

impl<P: Probe> NotProbe<P> {
    /// Create a new logical negation probe
    #[must_use]
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<P: Probe> Probe for NotProbe<P> {
    fn name(&self) -> &str {
        "not"
    }

    fn evaluate(&self, goal: &Goal, manager: &TermManager) -> f64 {
        if self.inner.evaluate(goal, manager) == 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    fn setup_manager() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_size_probe() {
        let manager = setup_manager();
        let goal = Goal::new(vec![]);

        let probe = SizeProbe;
        assert_eq!(probe.evaluate(&goal, &manager), 0.0);
    }

    #[test]
    fn test_has_quantifier_probe() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: forall x. x > 0
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_gt(x, zero);
        let forall = manager.mk_forall([("x", int_sort)], body);

        let goal_with_quant = Goal::new(vec![forall]);
        let goal_without_quant = Goal::new(vec![body]);

        let probe = HasQuantifierProbe;
        assert_eq!(probe.evaluate(&goal_with_quant, &manager), 1.0);
        assert_eq!(probe.evaluate(&goal_without_quant, &manager), 0.0);
    }

    #[test]
    fn test_is_linear_probe() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Linear: x + y
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let linear = manager.mk_add([x, y]);

        // Linear: 2 * x
        let two = manager.mk_int(2);
        let linear_mul = manager.mk_mul([two, x]);

        // Nonlinear: x * y
        let nonlinear = manager.mk_mul([x, y]);

        let probe = IsLinearProbe;

        let goal_linear = Goal::new(vec![linear, linear_mul]);
        assert_eq!(probe.evaluate(&goal_linear, &manager), 1.0);

        let goal_nonlinear = Goal::new(vec![nonlinear]);
        assert_eq!(probe.evaluate(&goal_nonlinear, &manager), 0.0);
    }

    #[test]
    fn test_and_probe() {
        let manager = setup_manager();
        let goal = Goal::new(vec![]);

        let probe = AndProbe::new(ConstProbe::new(1.0), ConstProbe::new(1.0));
        assert_eq!(probe.evaluate(&goal, &manager), 1.0);

        let probe = AndProbe::new(ConstProbe::new(1.0), ConstProbe::new(0.0));
        assert_eq!(probe.evaluate(&goal, &manager), 0.0);

        let probe = AndProbe::new(ConstProbe::new(0.0), ConstProbe::new(1.0));
        assert_eq!(probe.evaluate(&goal, &manager), 0.0);
    }
}
