//! Tactic framework for OxiZ
//!
//! Tactics are transformers that take a Goal and produce sub-Goals.
//! They form the basis of the strategy layer in the solver.

pub mod lia2card;
pub mod mbp;
pub mod nla2bv;
pub mod probe;
pub mod quantifier;

use crate::ast::normal_forms::{to_cnf, to_nnf};
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use num_integer::Integer;
use num_traits::Signed;

// Re-export probe types
pub use probe::{
    AddProbe, AndProbe, ConstProbe, DepthProbe, HasArrayProbe, HasBitVectorProbe,
    HasFloatingPointProbe, HasQuantifierProbe, IsLinearProbe, LtProbe, NegProbe, NodeCountProbe,
    NotProbe, NumConnectivesProbe, NumVarsProbe, OrProbe, Probe, SizeProbe,
};

// Re-export quantifier tactics
pub use quantifier::{
    Binding, GroundTermCollector, Pattern, PatternMatcher, QuantifierInstantiationTactic,
    SkolemizationTactic, UniversalEliminationTactic, contains_quantifier, goal_has_quantifiers,
};

// Re-export DER (Destructive Equality Resolution) types
pub use quantifier::{DerConfig, DerTactic, StatelessDerTactic};

// Re-export MBP types
pub use mbp::{MbpConfig, MbpEngine, MbpResult, MbpTactic, Model as MbpModel, ProjectorKind};

// Re-export NLA2BV types
pub use nla2bv::{Nla2BvConfig, Nla2BvTactic, StatelessNla2BvTactic};

// Re-export LIA2Card types
pub use lia2card::{
    CardinalityConstraint, CardinalityEncoding, Lia2CardConfig, Lia2CardTactic,
    StatelessLia2CardTactic,
};

/// A goal represents a formula to be solved
#[derive(Debug, Clone)]
pub struct Goal {
    /// The assertions in this goal
    pub assertions: Vec<TermId>,
    /// Model precision (for optimization)
    pub precision: Precision,
}

/// Precision level for model generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Under approximation - may miss solutions
    Under,
    /// Exact solution required
    #[default]
    Precise,
    /// Over approximation - may include spurious solutions
    Over,
}

impl Goal {
    /// Create a new goal with the given assertions
    #[must_use]
    pub fn new(assertions: Vec<TermId>) -> Self {
        Self {
            assertions,
            precision: Precision::Precise,
        }
    }

    /// Create an empty goal (trivially satisfiable)
    #[must_use]
    pub fn empty() -> Self {
        Self {
            assertions: Vec::new(),
            precision: Precision::Precise,
        }
    }

    /// Add an assertion to the goal
    pub fn add(&mut self, term: TermId) {
        self.assertions.push(term);
    }

    /// Check if the goal is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assertions.is_empty()
    }

    /// Get the number of assertions
    #[must_use]
    pub fn len(&self) -> usize {
        self.assertions.len()
    }
}

/// Result of applying a tactic
#[derive(Debug)]
pub enum TacticResult {
    /// The goal was solved (sat/unsat)
    Solved(SolveResult),
    /// The goal was transformed into sub-goals
    SubGoals(Vec<Goal>),
    /// The tactic does not apply to this goal
    NotApplicable,
    /// The tactic failed with an error
    Failed(String),
}

/// Solve result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown
    Unknown,
}

/// A tactic transforms goals into sub-goals
pub trait Tactic: Send + Sync {
    /// Get the name of this tactic
    fn name(&self) -> &str;

    /// Apply the tactic to a goal
    fn apply(&self, goal: &Goal) -> Result<TacticResult>;

    /// Get a description of the tactic
    fn description(&self) -> &str {
        ""
    }
}

/// A tactic that simplifies boolean and arithmetic expressions
#[derive(Debug)]
pub struct SimplifyTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> SimplifyTactic<'a> {
    /// Create a new simplify tactic with a term manager
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply simplification to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let simplified: Vec<TermId> = goal
            .assertions
            .iter()
            .map(|&term| self.manager.simplify(term))
            .collect();

        // Check if all assertions simplified to true
        let all_true = simplified.iter().all(|&t| t == self.manager.mk_true());
        if all_true {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        // Check if any assertion simplified to false
        let any_false = simplified.iter().any(|&t| t == self.manager.mk_false());
        if any_false {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = simplified
            .into_iter()
            .filter(|&t| t != self.manager.mk_true())
            .collect();

        // Check if anything changed
        if filtered == goal.assertions {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// A stateless simplify tactic that uses an owned manager
#[derive(Debug, Default)]
pub struct StatelessSimplifyTactic;

impl Tactic for StatelessSimplifyTactic {
    fn name(&self) -> &str {
        "simplify"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        // Real simplification requires the mutable SimplifyTactic
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Simplifies boolean and arithmetic expressions"
    }
}

/// A tactic that propagates constant values
#[derive(Debug)]
pub struct PropagateValuesTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> PropagateValuesTactic<'a> {
    /// Create a new propagate-values tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply value propagation to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        use crate::ast::TermKind;
        use rustc_hash::FxHashMap;

        // Phase 1: Collect equalities of the form (= var constant)
        let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();

        for &assertion in &goal.assertions {
            if let Some(term) = self.manager.get(assertion) {
                if let TermKind::Eq(lhs, rhs) = &term.kind {
                    let lhs_term = self.manager.get(*lhs);
                    let rhs_term = self.manager.get(*rhs);

                    match (lhs_term.map(|t| &t.kind), rhs_term.map(|t| &t.kind)) {
                        // x = constant
                        (Some(TermKind::Var(_)), Some(k)) if is_constant(k) => {
                            subst.insert(*lhs, *rhs);
                        }
                        // constant = x
                        (Some(k), Some(TermKind::Var(_))) if is_constant(k) => {
                            subst.insert(*rhs, *lhs);
                        }
                        _ => {}
                    }
                }
            }
        }

        // No substitutions found
        if subst.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 2: Apply substitutions to all assertions
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());
        let mut changed = false;

        for &assertion in &goal.assertions {
            let substituted = self.manager.substitute(assertion, &subst);
            let simplified = self.manager.simplify(substituted);

            if simplified != assertion {
                changed = true;
            }
            new_assertions.push(simplified);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        // Check for trivially true/false
        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        // Check if any assertion is false
        if new_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = new_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        // If all assertions are true, goal is SAT
        if filtered.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// Check if a term kind is a constant
fn is_constant(kind: &crate::ast::TermKind) -> bool {
    use crate::ast::TermKind;
    matches!(
        kind,
        TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
    )
}

/// Stateless version for the Tactic trait (placeholder)
#[derive(Debug, Default)]
pub struct StatelessPropagateValuesTactic;

impl Tactic for StatelessPropagateValuesTactic {
    fn name(&self) -> &str {
        "propagate-values"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Propagates constant values through the formula"
    }
}

/// Bit-blasting tactic - converts BitVector operations to propositional logic
///
/// This tactic transforms BitVector constraints into an equisatisfiable
/// set of Boolean constraints by representing each bit as a separate
/// Boolean variable.
#[derive(Debug)]
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

/// Ackermannization tactic - eliminates uninterpreted functions
///
/// This tactic implements Ackermann's reduction which replaces function
/// applications with fresh variables and adds functional consistency
/// constraints: for any two applications f(a1,...,an) and f(b1,...,bn),
/// if a1=b1 ∧ ... ∧ an=bn then f(a1,...,an) = f(b1,...,bn).
#[derive(Debug)]
pub struct AckermannizeTactic<'a> {
    manager: &'a mut TermManager,
}

/// A function application occurrence
#[derive(Debug, Clone)]
struct FuncApp {
    /// Fresh variable representing this application
    fresh_var: TermId,
    /// The arguments
    args: smallvec::SmallVec<[TermId; 4]>,
}

impl<'a> AckermannizeTactic<'a> {
    /// Create a new ackermannize tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Collect all function applications from a term
    fn collect_func_apps(
        &self,
        term_id: TermId,
        apps: &mut Vec<(lasso::Spur, smallvec::SmallVec<[TermId; 4]>, TermId)>,
        visited: &mut rustc_hash::FxHashSet<TermId>,
    ) {
        use crate::ast::TermKind;

        if visited.contains(&term_id) {
            return;
        }
        visited.insert(term_id);

        if let Some(term) = self.manager.get(term_id) {
            match &term.kind {
                TermKind::Apply { func, args } => {
                    apps.push((*func, args.clone(), term_id));
                    for &arg in args {
                        self.collect_func_apps(arg, apps, visited);
                    }
                }
                TermKind::Not(a) | TermKind::Neg(a) | TermKind::BvNot(a) => {
                    self.collect_func_apps(*a, apps, visited);
                }
                TermKind::BvExtract { arg, .. } => {
                    self.collect_func_apps(*arg, apps, visited);
                }
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::Distinct(args) => {
                    for &arg in args {
                        self.collect_func_apps(arg, apps, visited);
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
                | TermKind::BvSle(a, b) => {
                    self.collect_func_apps(*a, apps, visited);
                    self.collect_func_apps(*b, apps, visited);
                }
                TermKind::Ite(c, t, e) | TermKind::Store(c, t, e) => {
                    self.collect_func_apps(*c, apps, visited);
                    self.collect_func_apps(*t, apps, visited);
                    self.collect_func_apps(*e, apps, visited);
                }
                TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
                    self.collect_func_apps(*body, apps, visited);
                }
                TermKind::Let { bindings, body } => {
                    for (_, t) in bindings {
                        self.collect_func_apps(*t, apps, visited);
                    }
                    self.collect_func_apps(*body, apps, visited);
                }
                // Constants and variables don't contain function applications
                _ => {}
            }
        }
    }

    /// Apply ackermannization to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        use rustc_hash::{FxHashMap, FxHashSet};

        // Collect all function applications
        let mut all_apps: Vec<(lasso::Spur, smallvec::SmallVec<[TermId; 4]>, TermId)> = Vec::new();
        let mut visited = FxHashSet::default();

        for &assertion in &goal.assertions {
            self.collect_func_apps(assertion, &mut all_apps, &mut visited);
        }

        // No function applications found
        if all_apps.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Group applications by function symbol
        let mut func_groups: FxHashMap<lasso::Spur, Vec<FuncApp>> = FxHashMap::default();
        let mut term_to_var: FxHashMap<TermId, TermId> = FxHashMap::default();

        for (var_counter, (func, args, term_id)) in all_apps.into_iter().enumerate() {
            // Create a fresh variable for this application
            let Some(term) = self.manager.get(term_id) else {
                continue; // Skip if term not found
            };
            let sort = term.sort;
            let var_name = format!("!ack_{}", var_counter);
            let fresh_var = self.manager.mk_var(&var_name, sort);

            term_to_var.insert(term_id, fresh_var);

            func_groups
                .entry(func)
                .or_default()
                .push(FuncApp { fresh_var, args });
        }

        // Generate functional consistency constraints
        // For each pair of applications of the same function:
        // (a1 = b1 ∧ ... ∧ an = bn) => (f(a) = f(b))
        let mut constraints: Vec<TermId> = Vec::new();

        for apps in func_groups.values() {
            for i in 0..apps.len() {
                for j in (i + 1)..apps.len() {
                    let app_i = &apps[i];
                    let app_j = &apps[j];

                    // Only compare if they have the same arity
                    if app_i.args.len() != app_j.args.len() {
                        continue;
                    }

                    // Build: (a1 = b1) ∧ (a2 = b2) ∧ ... => (var_i = var_j)
                    let mut arg_eqs: Vec<TermId> = Vec::new();
                    for k in 0..app_i.args.len() {
                        let eq = self.manager.mk_eq(app_i.args[k], app_j.args[k]);
                        arg_eqs.push(eq);
                    }

                    let antecedent = if arg_eqs.len() == 1 {
                        arg_eqs[0]
                    } else {
                        self.manager.mk_and(arg_eqs)
                    };

                    let consequent = self.manager.mk_eq(app_i.fresh_var, app_j.fresh_var);
                    let constraint = self.manager.mk_implies(antecedent, consequent);
                    constraints.push(constraint);
                }
            }
        }

        // Substitute function applications with their fresh variables in the goal
        let mut new_assertions: Vec<TermId> = Vec::new();

        for &assertion in &goal.assertions {
            let substituted = self.manager.substitute(assertion, &term_to_var);
            new_assertions.push(substituted);
        }

        // Add the functional consistency constraints
        new_assertions.extend(constraints);

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessAckermannizeTactic;

impl Tactic for StatelessAckermannizeTactic {
    fn name(&self) -> &str {
        "ackermannize"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Eliminates uninterpreted functions by adding functional consistency constraints"
    }
}

/// Context-dependent simplification tactic
///
/// This tactic simplifies each assertion using the other assertions as context.
/// For example, given assertions [x = 5, x + y = 10], it can:
/// 1. Use "x = 5" to simplify "x + y = 10" to "5 + y = 10" -> "y = 5"
/// 2. Iteratively simplify until fixpoint
#[derive(Debug)]
pub struct CtxSolverSimplifyTactic<'a> {
    manager: &'a mut TermManager,
    /// Maximum number of iterations
    max_iterations: usize,
}

impl<'a> CtxSolverSimplifyTactic<'a> {
    /// Create a new context-solver-simplify tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            max_iterations: 10,
        }
    }

    /// Create with custom max iterations
    pub fn with_max_iterations(manager: &'a mut TermManager, max_iterations: usize) -> Self {
        Self {
            manager,
            max_iterations,
        }
    }

    /// Extract equalities from context that can be used for substitution
    fn extract_substitutions(
        &self,
        assertions: &[TermId],
        skip_index: usize,
    ) -> rustc_hash::FxHashMap<TermId, TermId> {
        use crate::ast::TermKind;
        use rustc_hash::FxHashMap;

        let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();

        for (i, &assertion) in assertions.iter().enumerate() {
            if i == skip_index {
                continue;
            }

            if let Some(term) = self.manager.get(assertion) {
                if let TermKind::Eq(lhs, rhs) = &term.kind {
                    let lhs_term = self.manager.get(*lhs);
                    let rhs_term = self.manager.get(*rhs);

                    match (lhs_term.map(|t| &t.kind), rhs_term.map(|t| &t.kind)) {
                        // x = constant
                        (Some(TermKind::Var(_)), Some(k)) if is_constant(k) => {
                            subst.insert(*lhs, *rhs);
                        }
                        // constant = x
                        (Some(k), Some(TermKind::Var(_))) if is_constant(k) => {
                            subst.insert(*rhs, *lhs);
                        }
                        // x = y (prefer lower term ID as representative)
                        (Some(TermKind::Var(_)), Some(TermKind::Var(_))) => {
                            if lhs.0 > rhs.0 {
                                subst.insert(*lhs, *rhs);
                            } else {
                                subst.insert(*rhs, *lhs);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        subst
    }

    /// Apply context-dependent simplification to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        if goal.assertions.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut current_assertions = goal.assertions.clone();
        let mut changed = false;

        // Iterate until fixpoint or max iterations
        for _ in 0..self.max_iterations {
            let mut iteration_changed = false;
            let mut new_assertions = Vec::with_capacity(current_assertions.len());

            for i in 0..current_assertions.len() {
                // Extract substitutions from other assertions
                let subst = self.extract_substitutions(&current_assertions, i);

                if subst.is_empty() {
                    new_assertions.push(current_assertions[i]);
                    continue;
                }

                // Apply substitution and simplify
                let substituted = self.manager.substitute(current_assertions[i], &subst);
                let simplified = self.manager.simplify(substituted);

                if simplified != current_assertions[i] {
                    iteration_changed = true;
                    changed = true;
                }
                new_assertions.push(simplified);
            }

            current_assertions = new_assertions;

            if !iteration_changed {
                break;
            }
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        // Check for trivially true/false
        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        // Check if any assertion is false
        if current_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = current_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        // If all assertions are true, goal is SAT
        if filtered.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessCtxSolverSimplifyTactic;

impl Tactic for StatelessCtxSolverSimplifyTactic {
    fn name(&self) -> &str {
        "ctx-solver-simplify"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Simplifies assertions using other assertions as context"
    }
}

/// Combinator: apply tactics in sequence
pub struct ThenTactic {
    tactics: Vec<Box<dyn Tactic>>,
}

impl std::fmt::Debug for ThenTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThenTactic")
            .field("tactics_count", &self.tactics.len())
            .finish()
    }
}

impl ThenTactic {
    /// Create a new sequential combinator
    pub fn new(tactics: Vec<Box<dyn Tactic>>) -> Self {
        Self { tactics }
    }
}

impl Tactic for ThenTactic {
    fn name(&self) -> &str {
        "then"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let mut current_goals = vec![goal.clone()];

        for tactic in &self.tactics {
            let mut next_goals = Vec::new();

            for g in &current_goals {
                match tactic.apply(g)? {
                    TacticResult::Solved(result) => {
                        return Ok(TacticResult::Solved(result));
                    }
                    TacticResult::SubGoals(sub) => {
                        next_goals.extend(sub);
                    }
                    TacticResult::NotApplicable => {
                        next_goals.push(g.clone());
                    }
                    TacticResult::Failed(msg) => {
                        return Ok(TacticResult::Failed(msg));
                    }
                }
            }

            current_goals = next_goals;
        }

        Ok(TacticResult::SubGoals(current_goals))
    }
}

/// Combinator: try tactics in order, use first that applies
pub struct OrElseTactic {
    tactics: Vec<Box<dyn Tactic>>,
}

impl std::fmt::Debug for OrElseTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrElseTactic")
            .field("tactics_count", &self.tactics.len())
            .finish()
    }
}

impl OrElseTactic {
    /// Create a new or-else combinator
    pub fn new(tactics: Vec<Box<dyn Tactic>>) -> Self {
        Self { tactics }
    }
}

impl Tactic for OrElseTactic {
    fn name(&self) -> &str {
        "or-else"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        for tactic in &self.tactics {
            match tactic.apply(goal)? {
                TacticResult::NotApplicable => continue,
                result => return Ok(result),
            }
        }
        Ok(TacticResult::NotApplicable)
    }
}

/// Combinator: repeat a tactic until fixpoint
pub struct RepeatTactic {
    tactic: Box<dyn Tactic>,
    max_iterations: usize,
}

impl std::fmt::Debug for RepeatTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepeatTactic")
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl RepeatTactic {
    /// Create a new repeat combinator
    pub fn new(tactic: Box<dyn Tactic>, max_iterations: usize) -> Self {
        Self {
            tactic,
            max_iterations,
        }
    }
}

impl Tactic for RepeatTactic {
    fn name(&self) -> &str {
        "repeat"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let mut current = goal.clone();

        for _ in 0..self.max_iterations {
            match self.tactic.apply(&current)? {
                TacticResult::Solved(result) => {
                    return Ok(TacticResult::Solved(result));
                }
                TacticResult::SubGoals(sub) if sub.len() == 1 => {
                    if sub[0].assertions == current.assertions {
                        // Fixpoint reached
                        break;
                    }
                    if let Some(next) = sub.into_iter().next() {
                        current = next;
                    } else {
                        break;
                    }
                }
                result => return Ok(result),
            }
        }

        Ok(TacticResult::SubGoals(vec![current]))
    }
}

/// Combinator: run tactics in parallel and return first successful result
///
/// This combinator executes multiple tactics concurrently using threads
/// and returns the result from the first tactic that completes successfully.
/// This is useful for portfolio-style solving where you want to try
/// different strategies simultaneously.
///
/// The parallel tactic will:
/// - Run all tactics concurrently
/// - Return the first `Solved` result if any tactic solves the goal
/// - Return the first `SubGoals` result if no tactic solves but one produces subgoals
/// - Return `NotApplicable` if all tactics return `NotApplicable`
/// - Return `Failed` if all tactics fail
pub struct ParallelTactic {
    tactics: Vec<std::sync::Arc<dyn Tactic>>,
}

impl std::fmt::Debug for ParallelTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelTactic")
            .field("tactics_count", &self.tactics.len())
            .finish()
    }
}

impl ParallelTactic {
    /// Create a new parallel combinator from Arc-wrapped tactics
    pub fn new(tactics: Vec<std::sync::Arc<dyn Tactic>>) -> Self {
        Self { tactics }
    }

    /// Create a new parallel combinator from boxed tactics
    pub fn from_boxes(tactics: Vec<Box<dyn Tactic>>) -> Self {
        Self {
            tactics: tactics
                .into_iter()
                .map(|t| -> std::sync::Arc<dyn Tactic> { t.into() })
                .collect(),
        }
    }
}

impl Tactic for ParallelTactic {
    fn name(&self) -> &str {
        "parallel"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        use std::sync::mpsc;
        use std::thread;

        if self.tactics.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        if self.tactics.len() == 1 {
            // No need for parallelism with a single tactic
            return self.tactics[0].apply(goal);
        }

        let (tx, rx) = mpsc::channel();

        // Spawn a thread for each tactic
        let handles: Vec<_> = self
            .tactics
            .iter()
            .enumerate()
            .map(|(idx, tactic)| {
                let goal_clone = goal.clone();
                let tx_clone = tx.clone();
                let tactic_clone = std::sync::Arc::clone(tactic);

                thread::spawn(move || {
                    let result = tactic_clone.apply(&goal_clone);
                    let _ = tx_clone.send((idx, result));
                })
            })
            .collect();

        // Drop the original sender so the receiver knows when all threads are done
        drop(tx);

        // Collect results
        let mut results = Vec::new();
        while let Ok((idx, result)) = rx.recv() {
            results.push((idx, result));
        }

        // Wait for all threads to complete
        for handle in handles {
            let _ = handle.join();
        }

        // Process results in priority order:
        // 1. First Solved result
        // 2. First SubGoals result
        // 3. NotApplicable if all are NotApplicable
        // 4. Failed otherwise

        let mut has_subgoals = None;
        let mut all_not_applicable = true;

        for (_idx, result) in results {
            match result {
                Ok(TacticResult::Solved(solve_result)) => {
                    return Ok(TacticResult::Solved(solve_result));
                }
                Ok(TacticResult::SubGoals(sub)) => {
                    if has_subgoals.is_none() {
                        has_subgoals = Some(sub);
                    }
                    all_not_applicable = false;
                }
                Ok(TacticResult::NotApplicable) => {}
                Ok(TacticResult::Failed(_)) | Err(_) => {
                    all_not_applicable = false;
                }
            }
        }

        if let Some(subgoals) = has_subgoals {
            Ok(TacticResult::SubGoals(subgoals))
        } else if all_not_applicable {
            Ok(TacticResult::NotApplicable)
        } else {
            Ok(TacticResult::Failed(
                "All parallel tactics failed".to_string(),
            ))
        }
    }

    fn description(&self) -> &str {
        "Run tactics in parallel and return first successful result"
    }
}

/// Timeout tactic - applies a tactic with a time limit
///
/// This tactic wraps another tactic and enforces a maximum execution time.
/// If the wrapped tactic doesn't complete within the timeout, it returns Failed.
pub struct TimeoutTactic {
    tactic: std::sync::Arc<dyn Tactic>,
    timeout_ms: u64,
}

impl std::fmt::Debug for TimeoutTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeoutTactic")
            .field("tactic_name", &self.tactic.name())
            .field("timeout_ms", &self.timeout_ms)
            .finish()
    }
}

impl TimeoutTactic {
    /// Create a new timeout tactic
    ///
    /// # Arguments
    /// * `tactic` - The tactic to run with a timeout
    /// * `timeout_ms` - Timeout in milliseconds
    pub fn new(tactic: std::sync::Arc<dyn Tactic>, timeout_ms: u64) -> Self {
        Self { tactic, timeout_ms }
    }

    /// Create a new timeout tactic from a boxed tactic
    pub fn from_box(tactic: Box<dyn Tactic>, timeout_ms: u64) -> Self {
        Self {
            tactic: tactic.into(),
            timeout_ms,
        }
    }
}

impl Tactic for TimeoutTactic {
    fn name(&self) -> &str {
        "timeout"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let (tx, rx) = mpsc::channel();
        let goal_clone = goal.clone();
        let tactic_clone = std::sync::Arc::clone(&self.tactic);

        // Spawn a thread to run the tactic
        let handle = thread::spawn(move || {
            let result = tactic_clone.apply(&goal_clone);
            let _ = tx.send(result);
        });

        // Wait for result with timeout
        match rx.recv_timeout(Duration::from_millis(self.timeout_ms)) {
            Ok(result) => {
                // Tactic completed within timeout
                let _ = handle.join();
                result
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Timeout exceeded
                // Note: The thread will continue running but we ignore its result
                Ok(TacticResult::Failed(format!(
                    "Tactic '{}' timed out after {}ms",
                    self.tactic.name(),
                    self.timeout_ms
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Thread panicked or dropped sender
                let _ = handle.join();
                Ok(TacticResult::Failed(format!(
                    "Tactic '{}' failed unexpectedly",
                    self.tactic.name()
                )))
            }
        }
    }

    fn description(&self) -> &str {
        "Apply a tactic with a time limit"
    }
}

/// Split tactic - performs case splitting on boolean variables
///
/// This tactic picks a boolean subterm and creates two sub-goals:
/// one with the term assumed true, and one with the term assumed false.
/// This is useful for exploring different branches of the search space.
#[derive(Debug)]
pub struct SplitTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> SplitTactic<'a> {
    /// Create a new split tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Find a good boolean term to split on
    fn find_split_candidate(&self, goal: &Goal) -> Option<TermId> {
        use crate::ast::traversal::collect_subterms;
        use rustc_hash::FxHashSet;

        let mut candidates = Vec::new();

        // Collect all boolean subterms from all assertions
        for &assertion in &goal.assertions {
            let subterms = collect_subterms(assertion, self.manager);

            for term_id in subterms {
                if let Some(term) = self.manager.get(term_id) {
                    // Only consider boolean terms that are not constants
                    if term.sort == self.manager.sorts.bool_sort {
                        match &term.kind {
                            TermKind::True | TermKind::False => {}
                            _ => {
                                candidates.push(term_id);
                            }
                        }
                    }
                }
            }
        }

        // Remove duplicates
        let unique: FxHashSet<TermId> = candidates.into_iter().collect();

        // Prefer variables over complex terms
        for &candidate in &unique {
            if let Some(term) = self.manager.get(candidate) {
                if matches!(term.kind, TermKind::Var(_)) {
                    return Some(candidate);
                }
            }
        }

        // Return any candidate if no variables found
        unique.into_iter().next()
    }

    /// Apply case splitting to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        // Find a boolean term to split on
        let Some(split_var) = self.find_split_candidate(goal) else {
            return Ok(TacticResult::NotApplicable);
        };

        // Create two sub-goals:
        // 1. goal ∧ split_var
        // 2. goal ∧ ¬split_var

        let true_goal = {
            let mut assertions = goal.assertions.clone();
            assertions.push(split_var);
            Goal {
                assertions,
                precision: goal.precision,
            }
        };

        let false_goal = {
            let not_split_var = self.manager.mk_not(split_var);
            let mut assertions = goal.assertions.clone();
            assertions.push(not_split_var);
            Goal {
                assertions,
                precision: goal.precision,
            }
        };

        Ok(TacticResult::SubGoals(vec![true_goal, false_goal]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessSplitTactic;

impl Tactic for StatelessSplitTactic {
    fn name(&self) -> &str {
        "split"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Performs case splitting on boolean subterms"
    }
}

/// Eliminate unconstrained variables tactic
///
/// This tactic identifies variables that appear in the formula but are not
/// actually constrained (i.e., they can take any value without affecting
/// satisfiability). Such variables can be eliminated to simplify the formula.
#[derive(Debug)]
pub struct EliminateUnconstrainedTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> EliminateUnconstrainedTactic<'a> {
    /// Create a new eliminate-unconstrained tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Count occurrences of each variable in the goal
    fn count_variable_occurrences(&self, goal: &Goal) -> rustc_hash::FxHashMap<TermId, usize> {
        use crate::ast::traversal::collect_subterms;
        use rustc_hash::FxHashMap;

        let mut counts: FxHashMap<TermId, usize> = FxHashMap::default();

        for &assertion in &goal.assertions {
            let subterms = collect_subterms(assertion, self.manager);
            for term_id in subterms {
                if let Some(term) = self.manager.get(term_id) {
                    if matches!(term.kind, TermKind::Var(_)) {
                        *counts.entry(term_id).or_insert(0) += 1;
                    }
                }
            }
        }

        counts
    }

    /// Check if a variable appears in an eliminable context
    fn is_eliminable(&self, var: TermId, assertion: TermId) -> bool {
        if let Some(term) = self.manager.get(assertion) {
            match &term.kind {
                TermKind::Eq(lhs, rhs) => {
                    if *lhs == var {
                        !self.contains_var(var, *rhs)
                    } else if *rhs == var {
                        !self.contains_var(var, *lhs)
                    } else {
                        false
                    }
                }
                TermKind::Or(args) => args.contains(&var),
                TermKind::And(args) => args.iter().any(|&arg| self.is_eliminable(var, arg)),
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a variable appears in a term
    fn contains_var(&self, var: TermId, term: TermId) -> bool {
        use crate::ast::traversal::contains_term;
        contains_term(term, var, self.manager)
    }

    /// Eliminate a variable from assertions
    fn eliminate_variable(&mut self, var: TermId, assertions: &[TermId]) -> Vec<TermId> {
        let mut result = Vec::new();

        for &assertion in assertions {
            if let Some(term) = self.manager.get(assertion) {
                match &term.kind {
                    TermKind::Eq(lhs, rhs) if *lhs == var || *rhs == var => {
                        continue;
                    }
                    TermKind::Or(args) if args.contains(&var) => {
                        result.push(self.manager.mk_true());
                    }
                    _ => {
                        result.push(assertion);
                    }
                }
            } else {
                result.push(assertion);
            }
        }

        result
    }

    /// Apply the eliminate-unconstrained tactic
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let counts = self.count_variable_occurrences(goal);

        let mut candidates: Vec<TermId> = counts
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(&var, _)| var)
            .collect();

        if candidates.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        candidates.retain(|&var| {
            goal.assertions
                .iter()
                .any(|&assertion| self.is_eliminable(var, assertion))
        });

        if candidates.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut new_assertions = goal.assertions.clone();
        for var in candidates {
            new_assertions = self.eliminate_variable(var, &new_assertions);
        }

        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        if new_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        let filtered: Vec<TermId> = new_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        if filtered.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        if filtered == goal.assertions {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessEliminateUnconstrainedTactic;

impl Tactic for StatelessEliminateUnconstrainedTactic {
    fn name(&self) -> &str {
        "elim-uncnstr"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Eliminates unconstrained variables from the formula"
    }
}

/// Solve-eqs tactic - Gaussian elimination for linear equations
///
/// This tactic performs variable elimination on systems of linear equations.
/// It finds equations of the form `x = expr` (where x doesn't appear in expr)
/// and substitutes them throughout the formula, effectively performing
/// Gaussian elimination.
///
/// The algorithm:
/// 1. Find equations of the form `var = expr` where `var` is not in `expr`
/// 2. Substitute `var -> expr` throughout all constraints
/// 3. Remove the solved equation
/// 4. Repeat until no more equations can be solved
///
/// This tactic is essential for simplifying linear integer/real arithmetic problems.
#[derive(Debug)]
pub struct SolveEqsTactic<'a> {
    manager: &'a mut TermManager,
    /// Maximum iterations for solving
    max_iterations: usize,
}

impl<'a> SolveEqsTactic<'a> {
    /// Create a new solve-eqs tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            max_iterations: 100,
        }
    }

    /// Create with custom max iterations
    pub fn with_max_iterations(manager: &'a mut TermManager, max_iterations: usize) -> Self {
        Self {
            manager,
            max_iterations,
        }
    }

    /// Check if a variable appears in a term
    fn var_occurs_in(&self, var: TermId, term: TermId) -> bool {
        use crate::ast::traversal::contains_term;
        contains_term(term, var, self.manager)
    }

    /// Apply Gaussian elimination to solve equations
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        if goal.assertions.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut current_assertions = goal.assertions.clone();
        let mut total_changed = false;

        for _ in 0..self.max_iterations {
            let mut iteration_changed = false;
            let mut solved_equation_index = None;
            let mut solution: Option<(TermId, TermId)> = None;

            // Find an equation we can solve
            for (i, &assertion) in current_assertions.iter().enumerate() {
                if let Some((var, expr)) = self.try_solve_equality_concrete(assertion) {
                    solved_equation_index = Some(i);
                    solution = Some((var, expr));
                    break;
                }
            }

            // If we found a solvable equation, apply the substitution
            if let (Some(idx), Some((var, expr))) = (solved_equation_index, solution) {
                iteration_changed = true;
                total_changed = true;

                // Build substitution map
                let mut subst = rustc_hash::FxHashMap::default();
                subst.insert(var, expr);

                // Apply substitution to all assertions except the solved one
                let mut new_assertions: Vec<TermId> = Vec::with_capacity(current_assertions.len());

                for (i, &assertion) in current_assertions.iter().enumerate() {
                    if i == idx {
                        // Skip the solved equation (it's now redundant)
                        continue;
                    }

                    let substituted = self.manager.substitute(assertion, &subst);
                    let simplified = self.manager.simplify(substituted);
                    new_assertions.push(simplified);
                }

                current_assertions = new_assertions;
            }

            if !iteration_changed {
                break;
            }
        }

        if !total_changed {
            return Ok(TacticResult::NotApplicable);
        }

        // Check for trivially true/false
        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        // Check if any assertion is false
        if current_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = current_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        // If all assertions are true, goal is SAT
        if filtered.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }

    /// Concrete implementation that returns actual TermIds (not pending computations)
    fn try_solve_equality_concrete(&mut self, eq_term: TermId) -> Option<(TermId, TermId)> {
        let term = self.manager.get(eq_term)?;

        if let TermKind::Eq(lhs, rhs) = term.kind {
            let lhs_kind = self.manager.get(lhs).map(|t| t.kind.clone());
            let rhs_kind = self.manager.get(rhs).map(|t| t.kind.clone());

            // Case 1: x = expr (where x doesn't appear in expr)
            if let Some(TermKind::Var(_)) = &lhs_kind {
                if !self.var_occurs_in(lhs, rhs) {
                    return Some((lhs, rhs));
                }
            }

            // Case 2: expr = x (where x doesn't appear in expr)
            if let Some(TermKind::Var(_)) = &rhs_kind {
                if !self.var_occurs_in(rhs, lhs) {
                    return Some((rhs, lhs));
                }
            }

            // Case 3: Handle linear addition: (x + a) = b => x = b - a
            if let Some(TermKind::Add(args)) = lhs_kind.clone() {
                if let Some((var, result)) = self.solve_linear_add_concrete(&args, rhs) {
                    return Some((var, result));
                }
            }

            // Case 4: Handle subtraction: (x - a) = b => x = b + a
            if let Some(TermKind::Sub(minuend, subtrahend)) = lhs_kind {
                if let Some((var, result)) =
                    self.solve_linear_sub_concrete(minuend, subtrahend, rhs)
                {
                    return Some((var, result));
                }
            }
        }

        None
    }

    /// Solve (x + a1 + a2 + ...) = b => x = b - a1 - a2 - ...
    fn solve_linear_add_concrete(
        &mut self,
        args: &smallvec::SmallVec<[TermId; 4]>,
        rhs: TermId,
    ) -> Option<(TermId, TermId)> {
        for (i, &arg) in args.iter().enumerate() {
            let arg_term = self.manager.get(arg)?;
            if let TermKind::Var(_) = &arg_term.kind {
                // Collect other arguments
                let other_args: smallvec::SmallVec<[TermId; 4]> = args
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &a)| a)
                    .collect();

                // Check if variable doesn't appear in other args or rhs
                let var_in_others = other_args.iter().any(|&a| self.var_occurs_in(arg, a));
                let var_in_rhs = self.var_occurs_in(arg, rhs);

                if !var_in_others && !var_in_rhs {
                    // Compute result: rhs - sum(other_args)
                    let result = if other_args.is_empty() {
                        rhs
                    } else if other_args.len() == 1 {
                        self.manager.mk_sub(rhs, other_args[0])
                    } else {
                        // rhs - (a1 + a2 + ...)
                        let sum = self.manager.mk_add(other_args);
                        self.manager.mk_sub(rhs, sum)
                    };

                    return Some((arg, result));
                }
            }
        }
        None
    }

    /// Solve (x - a) = b => x = b + a
    fn solve_linear_sub_concrete(
        &mut self,
        minuend: TermId,
        subtrahend: TermId,
        rhs: TermId,
    ) -> Option<(TermId, TermId)> {
        let minuend_term = self.manager.get(minuend)?;

        if let TermKind::Var(_) = &minuend_term.kind {
            if !self.var_occurs_in(minuend, subtrahend) && !self.var_occurs_in(minuend, rhs) {
                // x = rhs + subtrahend
                let result = self.manager.mk_add([rhs, subtrahend]);
                return Some((minuend, result));
            }
        }
        None
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessSolveEqsTactic;

impl Tactic for StatelessSolveEqsTactic {
    fn name(&self) -> &str {
        "solve-eqs"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Gaussian elimination for linear equations - solves x = expr and substitutes"
    }
}

// ============================================================================
// Fourier-Motzkin Elimination Tactic
// ============================================================================

/// Coefficient in a linear constraint
///
/// Represented as a pair of BigInts for exact rational arithmetic:
/// the coefficient is `numerator / denominator`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Coefficient {
    numerator: num_bigint::BigInt,
    denominator: num_bigint::BigInt,
}

impl Coefficient {
    fn zero() -> Self {
        Self {
            numerator: num_bigint::BigInt::ZERO,
            denominator: num_bigint::BigInt::from(1),
        }
    }

    fn one() -> Self {
        Self {
            numerator: num_bigint::BigInt::from(1),
            denominator: num_bigint::BigInt::from(1),
        }
    }

    fn from_int(n: impl Into<num_bigint::BigInt>) -> Self {
        Self {
            numerator: n.into(),
            denominator: num_bigint::BigInt::from(1),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator == num_bigint::BigInt::ZERO
    }

    fn is_positive(&self) -> bool {
        let sign = self.numerator.sign() * self.denominator.sign();
        sign == num_bigint::Sign::Plus && !self.is_zero()
    }

    fn is_negative(&self) -> bool {
        let sign = self.numerator.sign() * self.denominator.sign();
        sign == num_bigint::Sign::Minus
    }

    fn negate(&self) -> Self {
        Self {
            numerator: -&self.numerator,
            denominator: self.denominator.clone(),
        }
    }

    fn abs(&self) -> Self {
        Self {
            numerator: num_bigint::BigInt::from(self.numerator.magnitude().clone()),
            denominator: num_bigint::BigInt::from(self.denominator.magnitude().clone()),
        }
    }

    /// Multiply two coefficients
    fn multiply(&self, other: &Self) -> Self {
        let num = &self.numerator * &other.numerator;
        let den = &self.denominator * &other.denominator;
        Self::simplify(num, den)
    }

    /// Add two coefficients
    fn add(&self, other: &Self) -> Self {
        let num = &self.numerator * &other.denominator + &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;
        Self::simplify(num, den)
    }

    /// Simplify by dividing by GCD
    fn simplify(num: num_bigint::BigInt, den: num_bigint::BigInt) -> Self {
        if num == num_bigint::BigInt::ZERO {
            return Self::zero();
        }
        let g = Integer::gcd(&num, &den);
        let mut simplified_num = &num / &g;
        let mut simplified_den = &den / &g;
        // Ensure denominator is positive
        if simplified_den.sign() == num_bigint::Sign::Minus {
            simplified_num = -simplified_num;
            simplified_den = -simplified_den;
        }
        Self {
            numerator: simplified_num,
            denominator: simplified_den,
        }
    }
}

/// A linear constraint in the form: Σ aᵢxᵢ ≤ c (or < c if strict)
///
/// The constraint can also have boolean literals as a disjunctive prefix:
/// lit₁ ∨ lit₂ ∨ ... ∨ (Σ aᵢxᵢ ≤ c)
#[derive(Debug, Clone)]
struct LinearConstraint {
    /// Unique identifier (reserved for future use)
    #[allow(dead_code)]
    id: usize,
    /// Coefficients for each variable (indexed by variable id)
    /// Only non-zero coefficients are stored
    coefficients: rustc_hash::FxHashMap<TermId, Coefficient>,
    /// Right-hand side constant
    constant: Coefficient,
    /// Whether this is a strict inequality (<) or non-strict (≤)
    strict: bool,
    /// Boolean literals (disjunctive prefix)
    literals: smallvec::SmallVec<[TermId; 4]>,
    /// Whether this constraint has been eliminated
    dead: bool,
}

impl LinearConstraint {
    fn new(id: usize) -> Self {
        Self {
            id,
            coefficients: rustc_hash::FxHashMap::default(),
            constant: Coefficient::zero(),
            strict: false,
            literals: smallvec::SmallVec::new(),
            dead: false,
        }
    }

    fn get_coeff(&self, var: TermId) -> Coefficient {
        self.coefficients
            .get(&var)
            .cloned()
            .unwrap_or_else(Coefficient::zero)
    }

    fn set_coeff(&mut self, var: TermId, coeff: Coefficient) {
        if coeff.is_zero() {
            self.coefficients.remove(&var);
        } else {
            self.coefficients.insert(var, coeff);
        }
    }

    /// Check if this constraint has no variables (is a constant bound)
    fn is_constant(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Check if this is a trivially true constraint (0 ≤ c where c > 0)
    fn is_tautology(&self) -> bool {
        if !self.is_constant() {
            return false;
        }
        // 0 ≤ c is true if c ≥ 0
        // 0 < c is true if c > 0
        if self.strict {
            self.constant.is_positive()
        } else {
            !self.constant.is_negative()
        }
    }

    /// Check if this is a trivially false constraint (0 ≤ c where c < 0)
    fn is_contradiction(&self) -> bool {
        if !self.is_constant() {
            return false;
        }
        // 0 ≤ c is false if c < 0
        // 0 < c is false if c ≤ 0
        if self.strict {
            !self.constant.is_positive()
        } else {
            self.constant.is_negative()
        }
    }

    /// Get all variables with non-zero coefficients
    fn variables(&self) -> impl Iterator<Item = TermId> + '_ {
        self.coefficients.keys().copied()
    }

    /// Normalize coefficients by dividing by their GCD
    fn normalize(&mut self) {
        if self.coefficients.is_empty() {
            return;
        }

        // Collect all numerators (convert to common denominator first)
        let mut lcm = num_bigint::BigInt::from(1);
        for coeff in self.coefficients.values() {
            lcm = Integer::lcm(&lcm, &coeff.denominator);
        }
        lcm = Integer::lcm(&lcm, &self.constant.denominator);

        // Convert to integers using the LCM
        let mut int_coeffs: Vec<num_bigint::BigInt> = self
            .coefficients
            .values()
            .map(|c| &c.numerator * (&lcm / &c.denominator))
            .collect();
        let int_const = &self.constant.numerator * (&lcm / &self.constant.denominator);
        int_coeffs.push(int_const.clone());

        // Find GCD of all
        let mut gcd = Signed::abs(&int_coeffs[0]);
        for c in &int_coeffs[1..] {
            gcd = Integer::gcd(&gcd, &Signed::abs(c));
        }

        if gcd > num_bigint::BigInt::from(1) {
            // Divide all coefficients by GCD
            for coeff in self.coefficients.values_mut() {
                let int_val = &coeff.numerator * (&lcm / &coeff.denominator);
                let new_val = &int_val / &gcd;
                *coeff = Coefficient::from_int(new_val);
            }
            let new_const = &int_const / &gcd;
            self.constant = Coefficient::from_int(new_const);
        }
    }
}

/// Fourier-Motzkin elimination tactic for linear real arithmetic
///
/// This tactic performs quantifier elimination for linear real arithmetic
/// by systematically eliminating variables through pairwise resolution of
/// upper and lower bounds.
///
/// ## Algorithm
///
/// For each variable x to eliminate:
/// 1. Collect lower bounds: constraints where x has negative coefficient (after isolating x)
/// 2. Collect upper bounds: constraints where x has positive coefficient
/// 3. For each pair (lower, upper), resolve to eliminate x
/// 4. Remove original constraints involving x, add resolved constraints
///
/// ## Complexity
///
/// - Worst case: O(n²) new constraints per variable elimination
/// - Uses cutoff parameters to avoid exponential blowup
///
/// ## Limitations
///
/// - Currently only handles real arithmetic (not integer)
/// - May not terminate for very dense constraint systems
#[derive(Debug)]
pub struct FourierMotzkinTactic<'a> {
    manager: &'a mut TermManager,
    /// Maximum number of operations before giving up
    op_limit: usize,
    /// Current operation count
    op_count: usize,
    /// Cutoff: skip if |lowers| > cutoff1 AND |uppers| > cutoff1
    cutoff1: usize,
    /// Cutoff: skip if |lowers| × |uppers| > cutoff2
    cutoff2: usize,
    /// Next constraint ID
    next_constraint_id: usize,
}

impl<'a> FourierMotzkinTactic<'a> {
    /// Create a new Fourier-Motzkin elimination tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            op_limit: 500_000,
            op_count: 0,
            cutoff1: 8,
            cutoff2: 256,
            next_constraint_id: 0,
        }
    }

    /// Set the operation limit
    pub fn with_op_limit(mut self, limit: usize) -> Self {
        self.op_limit = limit;
        self
    }

    /// Set the cutoff parameters
    pub fn with_cutoffs(mut self, cutoff1: usize, cutoff2: usize) -> Self {
        self.cutoff1 = cutoff1;
        self.cutoff2 = cutoff2;
        self
    }

    /// Apply Fourier-Motzkin elimination to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        // Phase 1: Extract linear constraints from goal
        let mut constraints = Vec::new();
        let mut non_linear = Vec::new();

        for &assertion in &goal.assertions {
            match self.extract_constraint(assertion) {
                Some(c) => constraints.push(c),
                None => non_linear.push(assertion),
            }
        }

        if constraints.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 2: Collect all variables
        let mut all_vars: rustc_hash::FxHashSet<TermId> = rustc_hash::FxHashSet::default();
        for c in &constraints {
            for var in c.variables() {
                all_vars.insert(var);
            }
        }

        // Phase 3: Build variable bounds index
        let mut lowers: rustc_hash::FxHashMap<TermId, Vec<usize>> =
            rustc_hash::FxHashMap::default();
        let mut uppers: rustc_hash::FxHashMap<TermId, Vec<usize>> =
            rustc_hash::FxHashMap::default();

        for (idx, c) in constraints.iter().enumerate() {
            for var in c.variables() {
                let coeff = c.get_coeff(var);
                if coeff.is_negative() {
                    lowers.entry(var).or_default().push(idx);
                } else if coeff.is_positive() {
                    uppers.entry(var).or_default().push(idx);
                }
            }
        }

        // Phase 4: Eliminate variables
        let mut eliminated_any = false;

        // Sort variables by elimination cost
        let mut vars_to_eliminate: Vec<_> = all_vars.iter().copied().collect();
        vars_to_eliminate.sort_by_key(|&var| {
            let l = lowers.get(&var).map_or(0, |v| v.len());
            let u = uppers.get(&var).map_or(0, |v| v.len());
            l * u
        });

        for var in vars_to_eliminate {
            if self.op_count >= self.op_limit {
                break;
            }

            let lower_indices = lowers.get(&var).map_or(&[] as &[usize], |v| v.as_slice());
            let upper_indices = uppers.get(&var).map_or(&[] as &[usize], |v| v.as_slice());

            // Apply cutoffs
            if lower_indices.len() > self.cutoff1 && upper_indices.len() > self.cutoff1 {
                continue;
            }
            if lower_indices.len() * upper_indices.len() > self.cutoff2 {
                continue;
            }

            // Trivial elimination: no lower or upper bounds
            if lower_indices.is_empty() || upper_indices.is_empty() {
                // Mark constraints involving this variable as dead
                for &idx in lower_indices.iter().chain(upper_indices.iter()) {
                    constraints[idx].dead = true;
                }
                eliminated_any = true;
                continue;
            }

            // Perform pairwise resolution
            let mut new_constraints = Vec::new();

            for &lower_idx in lower_indices {
                for &upper_idx in upper_indices {
                    self.op_count += 1;
                    if self.op_count >= self.op_limit {
                        break;
                    }

                    if constraints[lower_idx].dead || constraints[upper_idx].dead {
                        continue;
                    }

                    if let Some(resolved) =
                        self.resolve(&constraints[lower_idx], &constraints[upper_idx], var)
                    {
                        // Check for contradiction
                        if resolved.is_contradiction() {
                            return Ok(TacticResult::Solved(SolveResult::Unsat));
                        }

                        // Skip tautologies
                        if resolved.is_tautology() {
                            continue;
                        }

                        new_constraints.push(resolved);
                    }
                }
            }

            // Mark old constraints as dead
            for &idx in lower_indices.iter().chain(upper_indices.iter()) {
                constraints[idx].dead = true;
            }

            // Add new constraints
            constraints.extend(new_constraints);
            eliminated_any = true;

            // Rebuild index for remaining variables
            lowers.clear();
            uppers.clear();
            for (idx, c) in constraints.iter().enumerate() {
                if c.dead {
                    continue;
                }
                for v in c.variables() {
                    let coeff = c.get_coeff(v);
                    if coeff.is_negative() {
                        lowers.entry(v).or_default().push(idx);
                    } else if coeff.is_positive() {
                        uppers.entry(v).or_default().push(idx);
                    }
                }
            }
        }

        if !eliminated_any {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 5: Convert remaining constraints back to assertions
        let mut new_assertions = non_linear;

        for c in &constraints {
            if c.dead {
                continue;
            }
            if c.is_tautology() {
                continue;
            }
            if let Some(term) = self.constraint_to_term(c) {
                new_assertions.push(term);
            }
        }

        // Check if all constraints eliminated to true
        if new_assertions.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Extract a linear constraint from a term
    fn extract_constraint(&mut self, term_id: TermId) -> Option<LinearConstraint> {
        let term = self.manager.get(term_id)?;

        match &term.kind {
            // a ≤ b  →  a - b ≤ 0
            TermKind::Le(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = false;
                self.extract_linear(*lhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*rhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a < b  →  a - b < 0
            TermKind::Lt(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = true;
                self.extract_linear(*lhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*rhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a ≥ b  →  b - a ≤ 0
            TermKind::Ge(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = false;
                self.extract_linear(*rhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*lhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a > b  →  b - a < 0
            TermKind::Gt(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = true;
                self.extract_linear(*rhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*lhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a = b  →  a ≤ b ∧ b ≤ a (not directly handled as a single constraint)
            // For FM, we'd need to split equalities, which we skip for simplicity
            _ => None,
        }
    }

    /// Extract linear terms recursively
    /// Returns None if the term is not linear
    fn extract_linear(
        &self,
        term_id: TermId,
        scale: &Coefficient,
        constraint: &mut LinearConstraint,
    ) -> Option<()> {
        let term = self.manager.get(term_id)?;

        match &term.kind {
            // Integer constant
            TermKind::IntConst(n) => {
                let coeff = Coefficient::from_int(n.clone()).multiply(scale);
                constraint.constant = constraint.constant.add(&coeff.negate());
                Some(())
            }

            // Rational constant
            TermKind::RealConst(r) => {
                let coeff = Coefficient {
                    numerator: num_bigint::BigInt::from(*r.numer()),
                    denominator: num_bigint::BigInt::from(*r.denom()),
                }
                .multiply(scale);
                constraint.constant = constraint.constant.add(&coeff.negate());
                Some(())
            }

            // Variable
            TermKind::Var(_) => {
                let existing = constraint.get_coeff(term_id);
                constraint.set_coeff(term_id, existing.add(scale));
                Some(())
            }

            // Addition
            TermKind::Add(args) => {
                for &arg in args {
                    self.extract_linear(arg, scale, constraint)?;
                }
                Some(())
            }

            // Subtraction
            TermKind::Sub(lhs, rhs) => {
                self.extract_linear(*lhs, scale, constraint)?;
                self.extract_linear(*rhs, &scale.negate(), constraint)?;
                Some(())
            }

            // Negation
            TermKind::Neg(arg) => self.extract_linear(*arg, &scale.negate(), constraint),

            // Multiplication by constant
            TermKind::Mul(args) => {
                // Check if all but one are constants
                let mut const_part = Coefficient::one();
                let mut var_part = None;

                for &arg in args {
                    let arg_term = self.manager.get(arg)?;
                    match &arg_term.kind {
                        TermKind::IntConst(n) => {
                            const_part = const_part.multiply(&Coefficient::from_int(n.clone()));
                        }
                        TermKind::RealConst(r) => {
                            const_part = const_part.multiply(&Coefficient {
                                numerator: num_bigint::BigInt::from(*r.numer()),
                                denominator: num_bigint::BigInt::from(*r.denom()),
                            });
                        }
                        _ => {
                            if var_part.is_some() {
                                // Multiple non-constant terms - not linear
                                return None;
                            }
                            var_part = Some(arg);
                        }
                    }
                }

                let new_scale = scale.multiply(&const_part);
                match var_part {
                    Some(v) => self.extract_linear(v, &new_scale, constraint),
                    None => {
                        // All constants
                        constraint.constant = constraint.constant.add(&new_scale.negate());
                        Some(())
                    }
                }
            }

            // Not linear
            _ => None,
        }
    }

    /// Resolve two constraints to eliminate a variable
    ///
    /// Given:
    /// - lower: ... + (-a)*x + ... ≤ c₁  (a > 0, so x ≥ (... - c₁) / a)
    /// - upper: ... + b*x + ... ≤ c₂     (b > 0, so x ≤ (c₂ - ...) / b)
    ///
    /// Produces: b*(lower_without_x) + a*(upper_without_x) ≤ b*c₁ + a*c₂
    fn resolve(
        &mut self,
        lower: &LinearConstraint,
        upper: &LinearConstraint,
        var: TermId,
    ) -> Option<LinearConstraint> {
        let coeff_l = lower.get_coeff(var); // negative
        let coeff_u = upper.get_coeff(var); // positive

        if !coeff_l.is_negative() || !coeff_u.is_positive() {
            return None;
        }

        let abs_a = coeff_l.abs();
        let b = coeff_u.clone();

        let mut result = LinearConstraint::new(self.next_constraint_id);
        self.next_constraint_id += 1;

        // Combine strictness
        result.strict = lower.strict || upper.strict;

        // Combine literals (union)
        result.literals.extend(lower.literals.iter().copied());
        for lit in &upper.literals {
            if !result.literals.contains(lit) {
                result.literals.push(*lit);
            }
        }

        // Combine coefficients: b * (lower coeffs) + a * (upper coeffs)
        // but skip the eliminated variable
        for (&v, coeff) in &lower.coefficients {
            if v == var {
                continue;
            }
            let scaled = coeff.multiply(&b);
            let existing = result.get_coeff(v);
            result.set_coeff(v, existing.add(&scaled));
        }

        for (&v, coeff) in &upper.coefficients {
            if v == var {
                continue;
            }
            let scaled = coeff.multiply(&abs_a);
            let existing = result.get_coeff(v);
            result.set_coeff(v, existing.add(&scaled));
        }

        // Combine constants: b * c₁ + a * c₂
        let scaled_c1 = lower.constant.multiply(&b);
        let scaled_c2 = upper.constant.multiply(&abs_a);
        result.constant = scaled_c1.add(&scaled_c2);

        // Normalize
        result.normalize();

        Some(result)
    }

    /// Convert a constraint back to a term
    fn constraint_to_term(&mut self, c: &LinearConstraint) -> Option<TermId> {
        if c.coefficients.is_empty() {
            // Constant constraint - already checked for tautology/contradiction
            return None;
        }

        // Build: Σ aᵢxᵢ ≤/< c
        // Which is: Σ aᵢxᵢ - c ≤/< 0
        // Rearranged: sum ≤/< -constant (since we stored as Σ - c ≤ 0)

        let mut positive_terms: Vec<TermId> = Vec::new();
        let mut negative_terms: Vec<TermId> = Vec::new();

        for (&var, coeff) in &c.coefficients {
            if coeff.is_zero() {
                continue;
            }

            // Handle coefficient
            let term = if Signed::abs(&coeff.numerator) == num_bigint::BigInt::from(1)
                && coeff.denominator == num_bigint::BigInt::from(1)
            {
                // Just the variable (possibly negated)
                if coeff.is_negative() {
                    negative_terms.push(var);
                    continue;
                } else {
                    var
                }
            } else {
                // c * var
                let abs_coeff = self.coeff_to_term(&coeff.abs());
                self.manager.mk_mul([abs_coeff, var])
            };

            if coeff.is_positive() {
                positive_terms.push(term);
            } else {
                negative_terms.push(term);
            }
        }

        // Build LHS sum
        let lhs = if positive_terms.is_empty() && negative_terms.is_empty() {
            self.manager.mk_int(0)
        } else if positive_terms.is_empty() {
            // All negative: return -(a + b + ...)
            let sum = if negative_terms.len() == 1 {
                negative_terms[0]
            } else {
                self.manager.mk_add(negative_terms)
            };
            self.manager.mk_neg(sum)
        } else if negative_terms.is_empty() {
            if positive_terms.len() == 1 {
                positive_terms[0]
            } else {
                self.manager.mk_add(positive_terms)
            }
        } else {
            // Mix: (pos_sum) - (neg_sum)
            let pos_sum = if positive_terms.len() == 1 {
                positive_terms[0]
            } else {
                self.manager.mk_add(positive_terms)
            };
            let neg_sum = if negative_terms.len() == 1 {
                negative_terms[0]
            } else {
                self.manager.mk_add(negative_terms)
            };
            self.manager.mk_sub(pos_sum, neg_sum)
        };

        // RHS is -constant (since we stored as Σ coeff*x ≤ -constant)
        let rhs = self.coeff_to_term(&c.constant.negate());

        // Build inequality
        if c.strict {
            Some(self.manager.mk_lt(lhs, rhs))
        } else {
            Some(self.manager.mk_le(lhs, rhs))
        }
    }

    /// Convert a coefficient to a term
    fn coeff_to_term(&mut self, c: &Coefficient) -> TermId {
        if c.denominator == num_bigint::BigInt::from(1) {
            // Integer
            self.manager.mk_int(c.numerator.clone())
        } else {
            // Rational - approximate as integer for now
            // A more sophisticated implementation would use Real sort
            let approx = &c.numerator / &c.denominator;
            self.manager.mk_int(approx)
        }
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessFourierMotzkinTactic;

impl Tactic for StatelessFourierMotzkinTactic {
    fn name(&self) -> &str {
        "fm"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Fourier-Motzkin variable elimination for linear arithmetic"
    }
}

// ============================================================================
// Scriptable Tactic (Rhai-based)
// ============================================================================

/// A tactic that executes user-defined Rhai scripts
///
/// This allows users to create custom simplification strategies by writing
/// Rhai scripts. The scripts have access to goal assertions and can return
/// simplified goals or solve results.
///
/// # Example Script
///
/// ```rhai
/// // Check if goal has any false assertions
/// fn apply_script(assertions) {
///     for assertion in assertions {
///         if is_false(assertion) {
///             return #{result: "unsat"};
///         }
///     }
///     #{result: "unchanged", assertions: assertions}
/// }
/// ```
#[derive(Debug)]
pub struct ScriptableTactic {
    engine: rhai::Engine,
    script: String,
    name: String,
    description: String,
}

impl ScriptableTactic {
    /// Create a new scriptable tactic with the given Rhai script
    ///
    /// # Arguments
    ///
    /// * `name` - Name for this tactic
    /// * `script` - Rhai script that defines an `apply_script` function
    /// * `description` - Description of what this tactic does
    ///
    /// # Errors
    ///
    /// Returns an error if the script fails to compile
    pub fn new(name: String, script: String, description: String) -> Result<Self> {
        let mut engine = rhai::Engine::new();

        // Register helper functions that scripts can use
        Self::register_builtins(&mut engine);

        // Validate that the script compiles
        engine.compile(&script).map_err(|e| {
            crate::error::OxizError::Internal(format!("Script compilation failed: {}", e))
        })?;

        Ok(Self {
            engine,
            script,
            name,
            description,
        })
    }

    /// Register built-in helper functions for scripts
    fn register_builtins(engine: &mut rhai::Engine) {
        // Register helper to check if an assertion ID represents false
        // In a real implementation, this would need access to the term manager
        engine.register_fn("is_false", |_id: i64| -> bool {
            // Placeholder - in real usage, we'd check against TermManager
            false
        });

        // Register helper to check if an assertion ID represents true
        engine.register_fn("is_true", |_id: i64| -> bool {
            // Placeholder - in real usage, we'd check against TermManager
            false
        });

        // Register len getter for Vec<i64>
        engine.register_get("len", |arr: &mut Vec<i64>| -> i64 { arr.len() as i64 });

        // Register indexer for Vec<i64>
        engine.register_indexer_get(|arr: &mut Vec<i64>, idx: i64| -> i64 {
            arr.get(idx as usize).copied().unwrap_or(0)
        });

        // Register indexer setter for Vec<i64>
        engine.register_indexer_set(|arr: &mut Vec<i64>, idx: i64, value: i64| {
            if (idx as usize) < arr.len() {
                arr[idx as usize] = value;
            }
        });
    }

    /// Apply the script to a goal using a term manager for context
    ///
    /// This is the stateful version that can access term information
    pub fn apply_with_manager(&self, goal: &Goal, _manager: &TermManager) -> Result<TacticResult> {
        // Convert goal assertions to Rhai array
        let assertions: Vec<i64> = goal.assertions.iter().map(|id| id.0 as i64).collect();

        // Create scope with assertions
        let mut scope = rhai::Scope::new();
        scope.push("assertions", assertions.clone());

        // Execute the script
        let result: rhai::Dynamic = self
            .engine
            .eval_with_scope(&mut scope, &self.script)
            .map_err(|e| {
                crate::error::OxizError::Internal(format!("Script execution failed: {}", e))
            })?;

        // Parse the result
        if let Some(map) = result.try_cast::<rhai::Map>() {
            if let Some(result_type) = map.get("result") {
                match result_type.to_string().as_str() {
                    "sat" => return Ok(TacticResult::Solved(SolveResult::Sat)),
                    "unsat" => return Ok(TacticResult::Solved(SolveResult::Unsat)),
                    "unknown" => return Ok(TacticResult::Solved(SolveResult::Unknown)),
                    "unchanged" => return Ok(TacticResult::NotApplicable),
                    _ => {}
                }
            }

            // Check if script returned modified assertions
            if let Some(new_assertions) = map.get("assertions") {
                if let Some(arr) = new_assertions.clone().try_cast::<rhai::Array>() {
                    let new_ids: Vec<TermId> = arr
                        .iter()
                        .filter_map(|v| v.as_int().ok().map(|i| TermId(i as u32)))
                        .collect();

                    if new_ids != goal.assertions {
                        return Ok(TacticResult::SubGoals(vec![Goal::new(new_ids)]));
                    }
                }
            }
        }

        Ok(TacticResult::NotApplicable)
    }
}

impl Tactic for ScriptableTactic {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Stateless version - limited functionality without TermManager
        // Convert goal assertions to Rhai array
        let assertions: Vec<i64> = goal.assertions.iter().map(|id| id.0 as i64).collect();

        // Create scope with assertions
        let mut scope = rhai::Scope::new();
        scope.push("assertions", assertions.clone());

        // Execute the script
        let result: rhai::Dynamic = self
            .engine
            .eval_with_scope(&mut scope, &self.script)
            .map_err(|e| {
                crate::error::OxizError::Internal(format!("Script execution failed: {}", e))
            })?;

        // Parse the result
        if let Some(map) = result.try_cast::<rhai::Map>() {
            if let Some(result_type) = map.get("result") {
                match result_type.to_string().as_str() {
                    "sat" => return Ok(TacticResult::Solved(SolveResult::Sat)),
                    "unsat" => return Ok(TacticResult::Solved(SolveResult::Unsat)),
                    "unknown" => return Ok(TacticResult::Solved(SolveResult::Unknown)),
                    "unchanged" => return Ok(TacticResult::NotApplicable),
                    _ => {}
                }
            }

            // Check if script returned modified assertions
            if let Some(new_assertions) = map.get("assertions") {
                if let Some(arr) = new_assertions.clone().try_cast::<rhai::Array>() {
                    let new_ids: Vec<TermId> = arr
                        .iter()
                        .filter_map(|v| v.as_int().ok().map(|i| TermId(i as u32)))
                        .collect();

                    if new_ids != goal.assertions {
                        return Ok(TacticResult::SubGoals(vec![Goal::new(new_ids)]));
                    }
                }
            }
        }

        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        &self.description
    }
}

/// Conditional tactic - chooses between tactics based on a probe value
///
/// This tactic evaluates a probe on the goal and selects one of two tactics
/// based on whether the probe value exceeds a threshold.
pub struct CondTactic {
    probe: std::sync::Arc<dyn Probe>,
    threshold: f64,
    if_true: std::sync::Arc<dyn Tactic>,
    if_false: std::sync::Arc<dyn Tactic>,
}

impl std::fmt::Debug for CondTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CondTactic")
            .field("probe_name", &self.probe.name())
            .field("threshold", &self.threshold)
            .field("if_true", &self.if_true.name())
            .field("if_false", &self.if_false.name())
            .finish()
    }
}

impl CondTactic {
    /// Create a new conditional tactic
    ///
    /// # Arguments
    /// * `probe` - The probe to evaluate
    /// * `threshold` - If probe value > threshold, use if_true, else use if_false
    /// * `if_true` - Tactic to use when probe > threshold
    /// * `if_false` - Tactic to use when probe <= threshold
    pub fn new(
        probe: std::sync::Arc<dyn Probe>,
        threshold: f64,
        if_true: std::sync::Arc<dyn Tactic>,
        if_false: std::sync::Arc<dyn Tactic>,
    ) -> Self {
        Self {
            probe,
            threshold,
            if_true,
            if_false,
        }
    }

    /// Create from boxed values
    pub fn from_box(
        probe: Box<dyn Probe>,
        threshold: f64,
        if_true: Box<dyn Tactic>,
        if_false: Box<dyn Tactic>,
    ) -> Self {
        Self {
            probe: probe.into(),
            threshold,
            if_true: if_true.into(),
            if_false: if_false.into(),
        }
    }
}

impl Tactic for CondTactic {
    fn name(&self) -> &str {
        "cond"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Note: We need a TermManager to evaluate probes properly
        // For now, we'll use a dummy evaluation that creates a temporary manager
        // In production, the probe should be evaluated with proper context
        let probe_value = {
            // Create a minimal manager for probe evaluation
            // This is a workaround - ideally we'd have access to the real manager
            let manager = crate::ast::TermManager::new();
            self.probe.evaluate(goal, &manager)
        };

        if probe_value > self.threshold {
            self.if_true.apply(goal)
        } else {
            self.if_false.apply(goal)
        }
    }

    fn description(&self) -> &str {
        "Conditional tactic selection based on probe value"
    }
}

/// When tactic - applies a tactic only when a probe condition is met
///
/// This is a convenience wrapper around CondTactic that returns NotApplicable
/// when the condition is not met.
pub struct WhenTactic {
    probe: std::sync::Arc<dyn Probe>,
    threshold: f64,
    tactic: std::sync::Arc<dyn Tactic>,
}

impl std::fmt::Debug for WhenTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhenTactic")
            .field("probe_name", &self.probe.name())
            .field("threshold", &self.threshold)
            .field("tactic", &self.tactic.name())
            .finish()
    }
}

impl WhenTactic {
    /// Create a new when tactic
    pub fn new(
        probe: std::sync::Arc<dyn Probe>,
        threshold: f64,
        tactic: std::sync::Arc<dyn Tactic>,
    ) -> Self {
        Self {
            probe,
            threshold,
            tactic,
        }
    }
}

impl Tactic for WhenTactic {
    fn name(&self) -> &str {
        "when"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let probe_value = {
            let manager = crate::ast::TermManager::new();
            self.probe.evaluate(goal, &manager)
        };

        if probe_value > self.threshold {
            self.tactic.apply(goal)
        } else {
            Ok(TacticResult::NotApplicable)
        }
    }

    fn description(&self) -> &str {
        "Apply tactic only when probe condition is met"
    }
}

/// FailIf tactic - fails if a probe condition is met
///
/// Useful for checking preconditions before applying tactics.
pub struct FailIfTactic {
    probe: std::sync::Arc<dyn Probe>,
    threshold: f64,
    message: String,
}

impl std::fmt::Debug for FailIfTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FailIfTactic")
            .field("probe_name", &self.probe.name())
            .field("threshold", &self.threshold)
            .finish()
    }
}

impl FailIfTactic {
    /// Create a new fail-if tactic
    pub fn new(probe: std::sync::Arc<dyn Probe>, threshold: f64, message: String) -> Self {
        Self {
            probe,
            threshold,
            message,
        }
    }
}

impl Tactic for FailIfTactic {
    fn name(&self) -> &str {
        "fail-if"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let probe_value = {
            let manager = crate::ast::TermManager::new();
            self.probe.evaluate(goal, &manager)
        };

        if probe_value > self.threshold {
            Ok(TacticResult::Failed(self.message.clone()))
        } else {
            Ok(TacticResult::NotApplicable)
        }
    }

    fn description(&self) -> &str {
        "Fail if probe condition is met"
    }
}

/// NNF tactic - converts formulas to Negation Normal Form
///
/// In NNF, negations are pushed inward so they only appear directly
/// before atoms, and only AND, OR, NOT operations remain.
#[derive(Debug)]
pub struct NnfTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> NnfTactic<'a> {
    /// Create a new NNF tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply NNF conversion to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());

        for &assertion in &goal.assertions {
            let nnf = to_nnf(assertion, self.manager);
            if nnf != assertion {
                changed = true;
            }
            new_assertions.push(nnf);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Stateless NNF tactic
#[derive(Debug, Default)]
pub struct StatelessNnfTactic;

impl Tactic for StatelessNnfTactic {
    fn name(&self) -> &str {
        "nnf"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Convert formulas to Negation Normal Form"
    }
}

/// Tseitin CNF tactic - converts formulas to Conjunctive Normal Form
///
/// Uses the Tseitin transformation which introduces auxiliary variables
/// to avoid exponential blowup in formula size.
#[derive(Debug)]
pub struct TseitinCnfTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> TseitinCnfTactic<'a> {
    /// Create a new Tseitin CNF tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply CNF conversion to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());

        for &assertion in &goal.assertions {
            let cnf = to_cnf(assertion, self.manager);
            if cnf != assertion {
                changed = true;
            }
            new_assertions.push(cnf);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Stateless CNF tactic
#[derive(Debug, Default)]
pub struct StatelessCnfTactic;

impl Tactic for StatelessCnfTactic {
    fn name(&self) -> &str {
        "tseitin-cnf"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Convert formulas to Conjunctive Normal Form using Tseitin transformation"
    }
}

// ============================================================================
// Pseudo-Boolean to Bit-Vector Tactic
// ============================================================================

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
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Convert pseudo-boolean constraints to bit-vector arithmetic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_goal() {
        let goal = Goal::empty();
        assert!(goal.is_empty());
        assert_eq!(goal.len(), 0);
    }

    #[test]
    fn test_simplify_tactic() {
        let goal = Goal::empty();
        let tactic = StatelessSimplifyTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_simplify_tactic_with_manager() {
        let mut manager = TermManager::new();

        // Create (+ 1 2) which should simplify to 3
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let sum = manager.mk_add([one, two]);

        // Create (= (+ 1 2) 3) which should simplify to true
        let three = manager.mk_int(3);
        let eq = manager.mk_eq(sum, three);

        let goal = Goal::new(vec![eq]);
        let mut tactic = SimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // The goal should be solved as SAT (all assertions simplified to true)
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_simplify_tactic_unsat() {
        let mut manager = TermManager::new();

        // Create (< 5 3) which should simplify to false
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);
        let lt = manager.mk_lt(five, three);

        let goal = Goal::new(vec![lt]);
        let mut tactic = SimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // The goal should be solved as UNSAT (an assertion simplified to false)
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_simplify_constant_folding() {
        let mut manager = TermManager::new();

        // Create (+ 1 2 3) which should simplify to 6
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let sum = manager.mk_add([one, two, three]);

        let simplified = manager.simplify(sum);
        // Check it's an integer constant with value 6
        if let Some(term) = manager.get(simplified) {
            if let crate::ast::TermKind::IntConst(n) = &term.kind {
                assert_eq!(*n, num_bigint::BigInt::from(6));
            } else {
                panic!("Expected IntConst");
            }
        } else {
            panic!("Term not found");
        }
    }

    #[test]
    fn test_simplify_mul_by_zero() {
        let mut manager = TermManager::new();

        // Create (* x 0) which should simplify to 0
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let product = manager.mk_mul([x, zero]);

        let simplified = manager.simplify(product);
        assert_eq!(simplified, zero);
    }

    #[test]
    fn test_propagate_values_basic() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (< x 10)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After propagation: (= 5 5) -> true, (< 5 10) -> true
        // Both simplify to true, so goal should be SAT
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_propagate_values_unsat() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (< x 3)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, three);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After propagation: (= 5 5) -> true, (< 5 3) -> false
        // Contains false, so goal should be UNSAT
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_propagate_values_partial() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (< y 10)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(y, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After propagation: (= 5 5) -> true (removed), (< y 10) stays
        // Should produce a subgoal with just (< y 10)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            assert_eq!(goals[0].assertions.len(), 1);
            // The remaining assertion should be (< y 10)
            assert_eq!(goals[0].assertions[0], lt);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_propagate_values_not_applicable() {
        let mut manager = TermManager::new();

        // Create goal: (< x 10) - no equalities to propagate
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // No equalities of form (= var const), so not applicable
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_propagate_values_chained() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (= y x), (< y 10)
        // Note: This tests that x=5 propagates, and whether we handle (= y x)
        // Currently, we only handle (= var const), so y won't be substituted in first pass
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq_x_5 = manager.mk_eq(x, five);
        let eq_y_x = manager.mk_eq(y, x);
        let lt_y_10 = manager.mk_lt(y, ten);

        let goal = Goal::new(vec![eq_x_5, eq_y_x, lt_y_10]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // x=5 is propagated, (= y x) becomes (= y 5)
        // We should get a reduced goal
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 2 assertions: (= y 5) and (< y 10)
            assert_eq!(goals[0].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result, got {:?}", result);
        }
    }

    #[test]
    fn test_bit_blast_not_applicable() {
        let manager = TermManager::new();

        // Create a goal with no BV terms
        let goal = Goal::empty();
        let tactic = BitBlastTactic::new(&manager);

        let result = tactic.apply_check(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_bit_blast_with_bv_constant() {
        let mut manager = TermManager::new();

        // Create a BV constant term
        let bv = manager.mk_bitvec(42u64, 8);
        let goal = Goal::new(vec![bv]);

        let tactic = BitBlastTactic::new(&manager);
        let result = tactic.apply_check(&goal).unwrap();

        // Should detect BV term and return SubGoals (not NotApplicable)
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_bit_blast_with_bv_operation() {
        let mut manager = TermManager::new();

        // Create BV variables and an operation
        let bv8_sort = manager.sorts.bitvec(8);
        let a = manager.mk_var("a", bv8_sort);
        let b = manager.mk_var("b", bv8_sort);
        let sum = manager.mk_bv_add(a, b);
        let hundred = manager.mk_bitvec(100u64, 8);
        let eq = manager.mk_eq(sum, hundred);

        let goal = Goal::new(vec![eq]);
        let tactic = BitBlastTactic::new(&manager);
        let result = tactic.apply_check(&goal).unwrap();

        // Should detect BV terms
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_bit_blast_mixed_goal() {
        let mut manager = TermManager::new();

        // Create a mixed goal with both Int and BV terms
        let int_sort = manager.sorts.int_sort;
        let bv8_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", int_sort);
        let a = manager.mk_var("a", bv8_sort);
        let ten = manager.mk_int(10);
        let int_constraint = manager.mk_lt(x, ten);
        let hundred = manager.mk_bitvec(100u64, 8);
        let bv_constraint = manager.mk_bv_ult(a, hundred);

        let goal = Goal::new(vec![int_constraint, bv_constraint]);
        let tactic = BitBlastTactic::new(&manager);
        let result = tactic.apply_check(&goal).unwrap();

        // Should detect BV terms in the goal
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_stateless_bit_blast() {
        let goal = Goal::empty();
        let tactic = StatelessBitBlastTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "bit-blast");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_ackermannize_not_applicable() {
        let mut manager = TermManager::new();

        // Create a goal with no function applications
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let constraint = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_ackermannize_single_function() {
        let mut manager = TermManager::new();

        // Create f(x) = 5
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let five = manager.mk_int(5);
        let constraint = manager.mk_eq(fx, five);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce a subgoal with the function replaced by a variable
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 1 assertion (no consistency constraints needed for single occurrence)
            assert_eq!(goals[0].assertions.len(), 1);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_ackermannize_two_applications() {
        let mut manager = TermManager::new();

        // Create f(x) = f(y)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let fy = manager.mk_apply("f", [y], int_sort);
        let constraint = manager.mk_eq(fx, fy);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce a subgoal with:
        // 1. The original constraint with f(x) and f(y) replaced by fresh vars
        // 2. A consistency constraint: (x = y) => (ack_0 = ack_1)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 2 assertions: the substituted constraint + consistency constraint
            assert_eq!(goals[0].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_ackermannize_nested_functions() {
        let mut manager = TermManager::new();

        // Create f(f(x)) = y
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let ffx = manager.mk_apply("f", [fx], int_sort);
        let constraint = manager.mk_eq(ffx, y);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce subgoals with both function applications replaced
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 2 assertions: substituted constraint + consistency for f(x) and f(f(x))
            assert_eq!(goals[0].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_stateless_ackermannize() {
        let goal = Goal::empty();
        let tactic = StatelessAckermannizeTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "ackermannize");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_ctx_solver_simplify_not_applicable() {
        let mut manager = TermManager::new();

        // Empty goal - not applicable
        let goal = Goal::empty();
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_ctx_solver_simplify_single_assertion() {
        let mut manager = TermManager::new();

        // Single assertion with no context to use
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let constraint = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // No context to use, so not applicable
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_ctx_solver_simplify_propagate() {
        let mut manager = TermManager::new();

        // x = 5, x < 10 should simplify to x = 5, 5 < 10 -> x = 5, true
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // The (< x 10) should simplify to (< 5 10) -> true and be filtered out
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Only the equality should remain (the < simplified to true)
            assert_eq!(goals[0].assertions.len(), 1);
        } else {
            panic!("Expected SubGoals result, got {:?}", result);
        }
    }

    #[test]
    fn test_ctx_solver_simplify_unsat() {
        let mut manager = TermManager::new();

        // x = 5, x < 3 should simplify to x = 5, false -> UNSAT
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, three);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_ctx_solver_simplify_multiple_equalities() {
        let mut manager = TermManager::new();

        // x = 5, y = x, y < 10
        // Should propagate x -> 5, then y -> 5, resulting in all true
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq_x = manager.mk_eq(x, five);
        let eq_y = manager.mk_eq(y, x);
        let lt = manager.mk_lt(y, ten);

        let goal = Goal::new(vec![eq_x, eq_y, lt]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should reduce the goal significantly
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // All assertions simplified to true - OK
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Some assertions may remain
                assert!(goals[0].assertions.len() <= 3);
            }
            _ => panic!("Expected SubGoals or Sat result"),
        }
    }

    #[test]
    fn test_stateless_ctx_solver_simplify() {
        let goal = Goal::empty();
        let tactic = StatelessCtxSolverSimplifyTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "ctx-solver-simplify");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_split_tactic_not_applicable() {
        let mut manager = TermManager::new();

        // Goal with only constants - nothing to split on
        let goal = Goal::new(vec![manager.mk_true()]);
        let mut tactic = SplitTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_split_tactic_simple() {
        let mut manager = TermManager::new();

        // Goal: x (where x is a boolean variable)
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let goal = Goal::new(vec![x]);

        let mut tactic = SplitTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce two sub-goals: (x, x) and (x, ¬x)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 2);

            // First goal should have (x, x) -> simplified to just x
            assert_eq!(goals[0].assertions.len(), 2);

            // Second goal should have (x, ¬x) - contradiction
            assert_eq!(goals[1].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_split_tactic_with_constraint() {
        let mut manager = TermManager::new();

        // Goal: (x ∨ y)
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let or_xy = manager.mk_or([x, y]);

        let goal = Goal::new(vec![or_xy]);
        let mut tactic = SplitTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce two sub-goals
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 2);

            // Each goal should have 2 assertions (the original plus the split condition)
            assert_eq!(goals[0].assertions.len(), 2);
            assert_eq!(goals[1].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_split_prefers_variables() {
        let mut manager = TermManager::new();

        // Goal: (x ∧ (y ∨ z))
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let z = manager.mk_var("z", manager.sorts.bool_sort);
        let or_yz = manager.mk_or([y, z]);
        let and_expr = manager.mk_and([x, or_yz]);

        let goal = Goal::new(vec![and_expr]);
        let mut tactic = SplitTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should split on a variable (x, y, or z) rather than the complex term (y ∨ z)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 2);

            // The split term should be a variable
            let split_term_true = goals[0].assertions.last().unwrap();
            let split_term_false = goals[1].assertions.last().unwrap();

            // One should be a variable, the other should be its negation
            if let Some(term_false) = manager.get(*split_term_false) {
                if let TermKind::Not(inner) = term_false.kind {
                    assert_eq!(inner, *split_term_true);
                }
            }
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_stateless_split() {
        let goal = Goal::empty();
        let tactic = StatelessSplitTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "split");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_eliminate_unconstrained_basic() {
        let mut manager = TermManager::new();
        let sort_int = manager.sorts.int_sort;

        // Create two variables: x and y
        let x = manager.mk_var("x", sort_int);
        let y = manager.mk_var("y", sort_int);

        // Create equation: x = 5 (x is eliminable since it appears once)
        let five = manager.mk_int(5);
        let eq = manager.mk_eq(x, five);

        // Create another constraint that doesn't involve x: y > 0
        let zero = manager.mk_int(0);
        let gt = manager.mk_gt(y, zero);

        let goal = Goal::new(vec![eq, gt]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // x should be eliminated, leaving only y > 0
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            assert_eq!(goals[0].assertions.len(), 1);
            assert_eq!(goals[0].assertions[0], gt);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_eliminate_unconstrained_or() {
        let mut manager = TermManager::new();
        let sort_bool = manager.sorts.bool_sort;

        // Create variables x and y
        let x = manager.mk_var("x", sort_bool);
        let y = manager.mk_var("y", sort_bool);

        // Create disjunction: x | y
        // x appears once and can be eliminated from the disjunction
        let or = manager.mk_or(vec![x, y]);

        let goal = Goal::new(vec![or]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // After eliminating x, the disjunction becomes true (unconstrained)
        if let TacticResult::Solved(SolveResult::Sat) = result {
            // Expected outcome
        } else {
            panic!("Expected Sat result, got {:?}", result);
        }
    }

    #[test]
    fn test_eliminate_unconstrained_not_applicable() {
        let mut manager = TermManager::new();
        let sort_int = manager.sorts.int_sort;

        // Create variable x that appears multiple times
        let x = manager.mk_var("x", sort_int);

        // Create constraints where x appears twice
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq1 = manager.mk_eq(x, five);
        let eq2 = manager.mk_eq(x, ten);

        let goal = Goal::new(vec![eq1, eq2]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // x appears twice, so it's not eliminable
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_eliminate_unconstrained_circular_dependency() {
        let mut manager = TermManager::new();
        let sort_int = manager.sorts.int_sort;

        // Create variables x and y
        let x = manager.mk_var("x", sort_int);
        let y = manager.mk_var("y", sort_int);

        // Create equation where x depends on itself: x = x + y
        // This shouldn't be eliminable
        let sum = manager.mk_add(vec![x, y]);
        let eq = manager.mk_eq(x, sum);

        let goal = Goal::new(vec![eq]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // x appears on both sides, so it shouldn't be eliminated
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_stateless_eliminate_unconstrained() {
        let goal = Goal::empty();
        let tactic = StatelessEliminateUnconstrainedTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "elim-uncnstr");
        assert!(!tactic.description().is_empty());
    }

    // ==================== Solve-Eqs Tactic Tests ====================

    #[test]
    fn test_solve_eqs_simple() {
        let mut manager = TermManager::new();

        // Create: x = 5, y = x + 3
        // After solving: x = 5, y = 8, then y = 8 simplifies to Sat
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);

        let eq_x = manager.mk_eq(x, five);
        let x_plus_3 = manager.mk_add([x, three]);
        let eq_y = manager.mk_eq(y, x_plus_3);

        let goal = Goal::new(vec![eq_x, eq_y]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should solve x = 5, substitute into y = x + 3 -> y = 8
        // Then solve y = 8, which results in all equations being solved (SAT)
        // or SubGoals with reduced assertions
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // Both equations got solved completely
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Should have at most one assertion left
                assert!(goals[0].assertions.len() <= 1);
            }
            _ => panic!("Expected SubGoals or Sat result, got {:?}", result),
        }
    }

    #[test]
    fn test_solve_eqs_chain() {
        let mut manager = TermManager::new();

        // Create: x = 5, y = x, z = y
        // After solving: all equalities are resolved
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let z = manager.mk_var("z", int_sort);
        let five = manager.mk_int(5);

        let eq_x = manager.mk_eq(x, five);
        let eq_y = manager.mk_eq(y, x);
        let eq_z = manager.mk_eq(z, y);

        let goal = Goal::new(vec![eq_x, eq_y, eq_z]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should solve the chain
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // All equations simplified to true
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Should have fewer assertions
                assert!(goals[0].assertions.len() < 3);
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_solve_eqs_not_applicable() {
        let mut manager = TermManager::new();

        // Create: x + y = 10 (no simple var = expr form)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let ten = manager.mk_int(10);

        let sum = manager.mk_add([x, y]);
        let eq = manager.mk_eq(sum, ten);
        let lt = manager.mk_lt(x, y); // Another constraint

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should solve (x + y) = 10 => x = 10 - y
        // Then substitute x in the second constraint
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // The goal should be simplified
            assert!(goals[0].assertions.len() <= 2);
        } else if let TacticResult::NotApplicable = result {
            // If we can't solve linear add yet, this is acceptable
        } else {
            panic!("Unexpected result: {:?}", result);
        }
    }

    #[test]
    fn test_solve_eqs_unsat() {
        let mut manager = TermManager::new();

        // Create: x = 5, x = 10 (contradiction)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let eq1 = manager.mk_eq(x, five);
        let eq2 = manager.mk_eq(x, ten);

        let goal = Goal::new(vec![eq1, eq2]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After substituting x = 5 into x = 10, we get 5 = 10 which is false
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_solve_eqs_sat() {
        let mut manager = TermManager::new();

        // Create: x = 5, x = 5 (trivially true)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let eq1 = manager.mk_eq(x, five);
        let eq2 = manager.mk_eq(x, five);

        let goal = Goal::new(vec![eq1, eq2]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Both equations should simplify to true
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_solve_eqs_with_constraints() {
        let mut manager = TermManager::new();

        // Create: x = 5, x < 10
        // After solving: 5 < 10 => true => SAT
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After substituting x = 5 into x < 10, we get 5 < 10 which is true
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_solve_eqs_cyclic() {
        let mut manager = TermManager::new();

        // Create: x = y, y = x (cyclic, should still work)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);

        let eq1 = manager.mk_eq(x, y);
        let eq2 = manager.mk_eq(y, x);

        let goal = Goal::new(vec![eq1, eq2]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should be able to solve at least one equation
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // Both simplified to true (x=y, y=y -> true, true)
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert!(goals[0].assertions.len() <= 2);
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_stateless_solve_eqs() {
        let goal = Goal::empty();
        let tactic = StatelessSolveEqsTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "solve-eqs");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_timeout_tactic_completes() {
        let goal = Goal::empty();
        let tactic = std::sync::Arc::new(StatelessSimplifyTactic);
        let timeout_tactic = TimeoutTactic::new(tactic, 1000); // 1 second timeout

        let result = timeout_tactic.apply(&goal).unwrap();
        // Should complete quickly (stateless simplify returns SubGoals)
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(timeout_tactic.name(), "timeout");
    }

    #[test]
    fn test_timeout_tactic_exceeds() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        // Create a custom slow tactic for testing
        struct SlowTactic {
            delay_ms: u64,
        }

        impl Tactic for SlowTactic {
            fn name(&self) -> &str {
                "slow"
            }

            fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
                thread::sleep(Duration::from_millis(self.delay_ms));
                Ok(TacticResult::NotApplicable)
            }

            fn description(&self) -> &str {
                "A slow tactic for testing timeouts"
            }
        }

        let goal = Goal::empty();
        let slow_tactic = Arc::new(SlowTactic { delay_ms: 500 });
        let timeout_tactic = TimeoutTactic::new(slow_tactic, 100); // 100ms timeout, but tactic takes 500ms

        let result = timeout_tactic.apply(&goal).unwrap();
        // Should timeout and return Failed
        match result {
            TacticResult::Failed(msg) => {
                assert!(msg.contains("timed out"));
            }
            _ => panic!("Expected timeout failure, got: {:?}", result),
        }
    }

    #[test]
    fn test_timeout_from_box() {
        let goal = Goal::empty();
        let tactic = Box::new(StatelessSimplifyTactic);
        let timeout_tactic = TimeoutTactic::from_box(tactic, 1000);

        let result = timeout_tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    // ========================================================================
    // ScriptableTactic tests
    // ========================================================================

    #[test]
    fn test_scriptable_tactic_basic() {
        let script = r#"
            #{result: "unchanged", assertions: assertions}
        "#
        .to_string();

        let tactic =
            ScriptableTactic::new("test-script".to_string(), script, "Test script".to_string())
                .unwrap();

        let goal = Goal::new(vec![TermId(1), TermId(2)]);
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::NotApplicable));
        assert_eq!(tactic.name(), "test-script");
        assert_eq!(tactic.description(), "Test script");
    }

    #[test]
    fn test_scriptable_tactic_return_sat() {
        let script = r#"
            #{result: "sat"}
        "#
        .to_string();

        let tactic =
            ScriptableTactic::new("sat-tactic".to_string(), script, "Returns SAT".to_string())
                .unwrap();

        let goal = Goal::empty();
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_scriptable_tactic_return_unsat() {
        let script = r#"
            #{result: "unsat"}
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "unsat-tactic".to_string(),
            script,
            "Returns UNSAT".to_string(),
        )
        .unwrap();

        let goal = Goal::empty();
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_scriptable_tactic_return_unknown() {
        let script = r#"
            #{result: "unknown"}
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "unknown-tactic".to_string(),
            script,
            "Returns Unknown".to_string(),
        )
        .unwrap();

        let goal = Goal::empty();
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::Solved(SolveResult::Unknown)));
    }

    #[test]
    fn test_scriptable_tactic_modify_assertions() {
        let script = r#"
            // Filter out assertion with ID 2
            let new_assertions = [];
            let i = 0;
            while i < assertions.len {
                let a = assertions[i];
                if a != 2 {
                    new_assertions.push(a);
                }
                i += 1;
            }
            #{result: "modified", assertions: new_assertions}
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "filter-tactic".to_string(),
            script,
            "Filters assertions".to_string(),
        )
        .unwrap();

        let goal = Goal::new(vec![TermId(1), TermId(2), TermId(3)]);
        let result = tactic.apply(&goal).unwrap();

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert_eq!(goals[0].assertions, vec![TermId(1), TermId(3)]);
            }
            _ => panic!("Expected SubGoals result"),
        }
    }

    #[test]
    fn test_scriptable_tactic_compilation_error() {
        let script = r#"
            // Invalid Rhai syntax
            this is not valid rhai code !@#$
        "#
        .to_string();

        let result = ScriptableTactic::new(
            "invalid-tactic".to_string(),
            script,
            "Invalid script".to_string(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_scriptable_tactic_access_assertions() {
        let script = r#"
            // Check assertion count
            if assertions.len == 0 {
                #{result: "sat"}
            } else {
                #{result: "unchanged", assertions: assertions}
            }
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "check-empty".to_string(),
            script,
            "Check if goal is empty".to_string(),
        )
        .unwrap();

        // Test with empty goal
        let empty_goal = Goal::empty();
        let result = tactic.apply(&empty_goal).unwrap();
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));

        // Test with non-empty goal
        let non_empty_goal = Goal::new(vec![TermId(1)]);
        let result = tactic.apply(&non_empty_goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_scriptable_tactic_complex_logic() {
        let script = r#"
            // Remove duplicate assertion IDs
            let seen = [];
            let unique = [];
            let i = 0;
            while i < assertions.len {
                let a = assertions[i];
                if !seen.contains(a) {
                    seen.push(a);
                    unique.push(a);
                }
                i += 1;
            }
            if unique.len != assertions.len {
                #{result: "modified", assertions: unique}
            } else {
                #{result: "unchanged"}
            }
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "deduplicate".to_string(),
            script,
            "Remove duplicate assertions".to_string(),
        )
        .unwrap();

        // Goal with duplicates
        let goal_with_dupes =
            Goal::new(vec![TermId(1), TermId(2), TermId(1), TermId(3), TermId(2)]);
        let result = tactic.apply(&goal_with_dupes).unwrap();

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert_eq!(goals[0].assertions, vec![TermId(1), TermId(2), TermId(3)]);
            }
            _ => panic!("Expected SubGoals result"),
        }

        // Goal without duplicates
        let goal_no_dupes = Goal::new(vec![TermId(1), TermId(2), TermId(3)]);
        let result = tactic.apply(&goal_no_dupes).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    // ========================================================================
    // Fourier-Motzkin Tactic Tests
    // ========================================================================

    #[test]
    fn test_fm_not_applicable_non_linear() {
        let mut manager = TermManager::new();

        // Create a goal with non-linear constraints (not applicable)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let eq = manager.mk_eq(x, y);

        let goal = Goal::new(vec![eq]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // Equality constraints are not directly handled by FM
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_fm_simple_inequality() {
        let mut manager = TermManager::new();

        // x < 5
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let lt = manager.mk_lt(x, five);

        let goal = Goal::new(vec![lt]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Single bound with only one variable - should still be applicable
        // since the variable has only upper bound but no lower bound
        match result {
            TacticResult::SubGoals(goals) => {
                // Variable eliminated (no lower bound)
                assert!(goals[0].assertions.is_empty() || goals[0].assertions.len() <= 1);
            }
            TacticResult::Solved(SolveResult::Sat) => {
                // All constraints eliminated
            }
            TacticResult::NotApplicable => {
                // No elimination happened (acceptable since only one constraint)
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_fm_bounded_variable() {
        let mut manager = TermManager::new();

        // x >= 0 and x <= 10 (lower and upper bounds)
        // Rewritten as: -x <= 0 and x <= 10
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let ten = manager.mk_int(10);
        let lower = manager.mk_ge(x, zero); // x >= 0
        let upper = manager.mk_le(x, ten); // x <= 10

        let goal = Goal::new(vec![lower, upper]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After FM elimination of x: 0 <= 10 (tautology)
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // All constraints resolved to tautologies
            }
            TacticResult::SubGoals(goals) => {
                // Reduced goal
                assert!(goals.len() == 1);
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_fm_unsat_bounds() {
        let mut manager = TermManager::new();

        // x >= 10 and x <= 5 (infeasible)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let five = manager.mk_int(5);
        let lower = manager.mk_ge(x, ten); // x >= 10
        let upper = manager.mk_le(x, five); // x <= 5

        let goal = Goal::new(vec![lower, upper]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After FM: 10 <= 5 (contradiction)
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_fm_two_variables() {
        let mut manager = TermManager::new();

        // x + y <= 10 and x >= 0 and y >= 0
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let ten = manager.mk_int(10);

        let sum = manager.mk_add([x, y]);
        let c1 = manager.mk_le(sum, ten); // x + y <= 10
        let c2 = manager.mk_ge(x, zero); // x >= 0
        let c3 = manager.mk_ge(y, zero); // y >= 0

        let goal = Goal::new(vec![c1, c2, c3]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // FM should eliminate some variables
        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Some constraints may remain
            }
            TacticResult::Solved(SolveResult::Sat) => {
                // All constraints eliminated (system is feasible)
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_fm_coefficient_scaling() {
        let mut manager = TermManager::new();

        // 2x <= 10 and x >= 1
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let two = manager.mk_int(2);
        let one = manager.mk_int(1);
        let ten = manager.mk_int(10);

        let two_x = manager.mk_mul([two, x]);
        let c1 = manager.mk_le(two_x, ten); // 2x <= 10, i.e., x <= 5

        let c2 = manager.mk_ge(x, one); // x >= 1

        let goal = Goal::new(vec![c1, c2]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Feasible: 1 <= x <= 5
        match result {
            TacticResult::Solved(SolveResult::Sat) | TacticResult::SubGoals(_) => {
                // Expected
            }
            TacticResult::Solved(SolveResult::Unsat) => {
                panic!("System should be SAT");
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_stateless_fm() {
        let goal = Goal::empty();
        let tactic = StatelessFourierMotzkinTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "fm");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_stateless_pb2bv() {
        let goal = Goal::empty();
        let tactic = StatelessPb2BvTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "pb2bv");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_pb2bv_not_applicable() {
        let mut manager = TermManager::new();

        // A non-PB constraint (regular boolean)
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let goal = Goal::new(vec![p]);

        let mut tactic = Pb2BvTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // Should be not applicable since it's just a boolean variable
        assert!(matches!(result, TacticResult::NotApplicable));
    }
}
