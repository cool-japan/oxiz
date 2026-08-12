//! Optimization context.
//!
//! This module provides the main interface for optimization modulo theories.
//! It integrates MaxSAT solving with SMT solving to support:
//! - Soft constraints with weights
//! - Objective function optimization (minimize/maximize)
//! - Multi-objective (Pareto) optimization
//!
//! Reference: Z3's `opt/opt_context.cpp`

mod model_eval;

use crate::maxsat::{MaxSatConfig, MaxSatError, MaxSatResult, Weight};
use crate::objective::{Objective, ObjectiveId, ObjectiveKind};
use crate::pareto::ParetoConfig;
use model_eval::model_eval_bool;
use num_bigint::BigInt;
use num_rational::{BigRational, Rational64};
use num_traits::ToPrimitive;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_solver::{Model, Solver, SolverResult};
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Compute the greatest common divisor of two non-negative `BigInt`s
/// (Euclidean algorithm).
fn bigint_gcd(a: &BigInt, b: &BigInt) -> BigInt {
    let zero = BigInt::from(0);
    let (mut a, mut b) = (a.clone(), b.clone());
    while b != zero {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

/// Compute the least common multiple of two positive `BigInt`s.
fn bigint_lcm(a: &BigInt, b: &BigInt) -> BigInt {
    let zero = BigInt::from(0);
    if *a == zero || *b == zero {
        return zero;
    }
    (a / bigint_gcd(a, b)) * b
}

/// Convert a soft-constraint [`Weight`] to an exact integer cost, scaled by
/// `scale`. `scale` must be a common multiple of every rational weight's
/// denominator (see callers) so that rational weights are represented
/// exactly rather than truncated/coerced.
fn scaled_weight_int(weight: &Weight, scale: &BigInt) -> BigInt {
    match weight {
        Weight::Int(n) => n * scale,
        Weight::Rational(r) => {
            // `scale` is a multiple of `r.denom()` by construction, so this
            // division is always exact.
            let factor = scale / r.denom();
            r.numer() * factor
        }
        // Infinite weight is treated as an effectively-hard constraint: use
        // a sentinel far larger than the sum of every other (scaled) soft
        // weight so it always dominates the optimization.
        Weight::Infinite => BigInt::from(i64::MAX / 2) * scale,
    }
}

/// Errors that can occur during optimization
#[derive(Error, Debug)]
pub enum OptError {
    /// Hard constraints are unsatisfiable
    #[error("hard constraints unsatisfiable")]
    Unsatisfiable,
    /// No solution found within limits
    #[error("unknown (resource limit)")]
    Unknown,
    /// MaxSAT error
    #[error("maxsat error: {0}")]
    MaxSat(#[from] MaxSatError),
    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result of optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptResult {
    /// Optimal solution found
    Optimal,
    /// Solution found but optimality not proven
    Satisfiable,
    /// No solution exists
    Unsatisfiable,
    /// Could not determine
    Unknown,
    /// Objective is unbounded (no finite optimum)
    Unbounded,
}

impl From<MaxSatResult> for OptResult {
    fn from(r: MaxSatResult) -> Self {
        match r {
            MaxSatResult::Optimal => OptResult::Optimal,
            MaxSatResult::Satisfiable => OptResult::Satisfiable,
            MaxSatResult::Unsatisfiable => OptResult::Unsatisfiable,
            MaxSatResult::Unknown => OptResult::Unknown,
        }
    }
}

impl std::fmt::Display for OptResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptResult::Optimal => write!(f, "optimal"),
            OptResult::Satisfiable => write!(f, "satisfiable"),
            OptResult::Unsatisfiable => write!(f, "unsatisfiable"),
            OptResult::Unknown => write!(f, "unknown"),
            OptResult::Unbounded => write!(f, "unbounded"),
        }
    }
}

/// Unique identifier for a soft constraint at the SMT level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SoftConstraintId(pub u32);

impl SoftConstraintId {
    /// Create a new soft constraint ID
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for SoftConstraintId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for SoftConstraintId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl From<SoftConstraintId> for u32 {
    fn from(id: SoftConstraintId) -> Self {
        id.0
    }
}

impl From<SoftConstraintId> for usize {
    fn from(id: SoftConstraintId) -> Self {
        id.0 as usize
    }
}

/// A soft constraint at the SMT level
#[derive(Debug, Clone)]
pub struct SoftConstraint {
    /// Unique identifier
    pub id: SoftConstraintId,
    /// The term representing the constraint
    pub term: TermId,
    /// Weight of this soft constraint
    pub weight: Weight,
    /// Group this constraint belongs to (for grouped optimization)
    pub group: Option<String>,
}

/// Configuration for optimization context
#[derive(Debug, Clone)]
pub struct OptConfig {
    /// MaxSAT configuration
    pub maxsat: MaxSatConfig,
    /// Pareto configuration
    pub pareto: ParetoConfig,
    /// Enable incremental optimization
    pub incremental: bool,
    /// Timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            maxsat: MaxSatConfig::default(),
            pareto: ParetoConfig::default(),
            incremental: true,
            timeout_ms: 0,
            verbose: false,
        }
    }
}

/// Statistics from optimization
#[derive(Debug, Clone, Default)]
pub struct OptStats {
    /// Number of SAT/SMT calls
    pub solver_calls: u32,
    /// Number of cores extracted
    pub cores_extracted: u32,
    /// Number of objective bounds updated
    pub bound_updates: u32,
    /// Time spent in solving (ms)
    pub solve_time_ms: u64,
}

/// Model value types
#[derive(Debug, Clone)]
pub enum ModelValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(BigInt),
    /// Rational value
    Rational(BigRational),
    /// Bitvector value (width, value)
    BitVec(u32, BigInt),
}

impl std::fmt::Display for ModelValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelValue::Bool(b) => write!(f, "{}", b),
            ModelValue::Int(n) => write!(f, "{}", n),
            ModelValue::Rational(r) => write!(f, "{}", r),
            ModelValue::BitVec(width, val) => {
                write!(f, "#b{:0width$b}", val, width = *width as usize)
            }
        }
    }
}

/// Convert a `TermId` value term (returned by `Model::eval`) into a `ModelValue`.
///
/// Returns `None` when the term is a non-constant (variable or compound expression
/// that could not be fully evaluated in the model).
fn term_id_to_model_value(val: TermId, tm: &TermManager) -> Option<ModelValue> {
    let t = tm.get(val)?;
    match &t.kind {
        TermKind::True => Some(ModelValue::Bool(true)),
        TermKind::False => Some(ModelValue::Bool(false)),
        TermKind::IntConst(n) => Some(ModelValue::Int(n.clone())),
        TermKind::RealConst(r) => {
            // Rational64 → BigRational
            let big_r = BigRational::new(BigInt::from(*r.numer()), BigInt::from(*r.denom()));
            Some(ModelValue::Rational(big_r))
        }
        TermKind::BitVecConst { value, width } => Some(ModelValue::BitVec(*width, value.clone())),
        _ => None,
    }
}

/// Convert a `TermId` value term into a `Weight` (for storing objective bounds).
fn term_id_to_weight(val: TermId, tm: &TermManager) -> Weight {
    let Some(t) = tm.get(val) else {
        return Weight::Infinite;
    };
    match &t.kind {
        TermKind::IntConst(n) => Weight::Int(n.clone()),
        TermKind::RealConst(r) => {
            let big_r = BigRational::new(BigInt::from(*r.numer()), BigInt::from(*r.denom()));
            Weight::Rational(big_r)
        }
        _ => Weight::Infinite,
    }
}

/// Snapshot a solver [`Model`] into the `TermId -> ModelValue` map used
/// throughout `OptContext`. Shared by every optimization path so the
/// (var-term, value-term) -> `ModelValue` conversion logic lives in one place.
fn snapshot_model(model: &Model, tm: &TermManager) -> FxHashMap<TermId, ModelValue> {
    model
        .assignments()
        .iter()
        .filter_map(|(&var_term, &val_term)| {
            let mv = term_id_to_model_value(val_term, tm)?;
            Some((var_term, mv))
        })
        .collect()
}

/// Floor of `x / 2` (correct for negative operands, unlike truncating
/// division). Used by the integer objective binary search below.
fn floor_half(x: BigInt) -> BigInt {
    let two = BigInt::from(2);
    let q = &x / &two;
    let r = x - &q * &two;
    if r.sign() == num_bigint::Sign::Minus {
        q - BigInt::from(1)
    } else {
        q
    }
}

/// Pareto dominance over already-evaluated objective values.
///
/// `p` dominates `q` iff `p` is no worse than `q` in every objective and
/// strictly better in at least one ("better" = smaller for `Minimize`, larger
/// for `Maximize`). Relies on `Weight`'s numeric (variant-independent) `Ord`
/// so Int/Rational objective values compare correctly against each other.
fn dominates_weights(p: &[Weight], q: &[Weight], kinds: &[ObjectiveKind]) -> bool {
    use std::cmp::Ordering;
    let mut strictly_better = false;
    for i in 0..kinds.len() {
        let ord = p[i].cmp(&q[i]);
        let (no_worse, better) = match kinds[i] {
            ObjectiveKind::Minimize => (ord != Ordering::Greater, ord == Ordering::Less),
            ObjectiveKind::Maximize => (ord != Ordering::Less, ord == Ordering::Greater),
        };
        if !no_worse {
            return false;
        }
        if better {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Outcome of a bounded feasibility probe for an integer objective bound.
enum IntBoundProbe {
    /// A feasible model was found; carries the attained objective value.
    Sat(BigInt, Model),
    /// The bound is infeasible.
    Unsat,
    /// The solver could not decide (incomplete/timeout/non-concrete value).
    Unknown,
}

/// Outcome of a bounded feasibility probe for a real (rational) objective bound.
enum RealBoundProbe {
    /// A feasible model was found; carries the attained objective value.
    Sat(Rational64, Model),
    /// The bound is infeasible.
    Unsat,
    /// The solver could not decide (incomplete/timeout/non-concrete value).
    Unknown,
}

/// Outcome of a single-objective search (see [`OptContext::optimize_int_objective`]
/// / [`OptContext::optimize_real_objective`]).
enum SingleObjOutcome {
    /// The search proved `value` is the exact optimum.
    Optimal {
        value: Weight,
        model: FxHashMap<TermId, ModelValue>,
    },
    /// `value` is feasible but optimality was not proven (iteration budget,
    /// deadline, or solver `Unknown` cut the search short). Must never be
    /// reported to callers as `Optimal`.
    Feasible {
        value: Weight,
        model: FxHashMap<TermId, ModelValue>,
    },
    /// The objective is unbounded in its optimization direction.
    Unbounded,
    /// The hard constraints (plus any fixed prior objectives) are unsatisfiable.
    Unsat,
    /// The solver could not decide even the initial feasibility check.
    Unknown,
}

/// Optimization context wrapping the SMT solver.
///
/// This is the main interface for optimization problems. It supports:
/// - Hard constraints (must be satisfied)
/// - Soft constraints with weights (maximize satisfaction)
/// - Objective functions (minimize/maximize expressions)
/// - Multi-objective optimization
#[derive(Debug)]
pub struct OptContext {
    /// Hard constraints (terms that must be true)
    hard_constraints: Vec<TermId>,
    /// Soft constraints
    soft_constraints: Vec<SoftConstraint>,
    /// Next soft constraint ID
    next_soft_id: u32,
    /// Objectives to optimize
    objectives: Vec<Objective>,
    /// Next objective ID
    next_obj_id: u32,
    /// Configuration
    config: OptConfig,
    /// Statistics
    stats: OptStats,
    /// Best model found
    best_model: Option<FxHashMap<TermId, ModelValue>>,
    /// Current lower bounds for objectives
    lower_bounds: FxHashMap<ObjectiveId, Weight>,
    /// Current upper bounds for objectives
    upper_bounds: FxHashMap<ObjectiveId, Weight>,
    /// Soft constraint groups
    groups: FxHashMap<String, Vec<SoftConstraintId>>,
    /// Context stack for push/pop
    context_stack: Vec<ContextSnapshot>,
    /// Term manager for building auxiliary terms during solving
    pub terms: TermManager,
    /// Counter for generating fresh selector variable names
    next_sel_id: u32,
    /// Cached Pareto front from last `optimize_pareto` call
    pareto_front: Vec<FxHashMap<TermId, ModelValue>>,
}

/// Snapshot for push/pop
#[derive(Debug, Clone)]
struct ContextSnapshot {
    num_hard: usize,
    num_soft: usize,
    num_objectives: usize,
}

impl Default for OptContext {
    fn default() -> Self {
        Self::new()
    }
}

impl OptContext {
    /// Create a new optimization context.
    pub fn new() -> Self {
        Self::with_config(OptConfig::default())
    }

    /// Create a new optimization context with configuration.
    pub fn with_config(config: OptConfig) -> Self {
        Self {
            hard_constraints: Vec::new(),
            soft_constraints: Vec::new(),
            next_soft_id: 0,
            objectives: Vec::new(),
            next_obj_id: 0,
            config,
            stats: OptStats::default(),
            best_model: None,
            lower_bounds: FxHashMap::default(),
            upper_bounds: FxHashMap::default(),
            groups: FxHashMap::default(),
            context_stack: Vec::new(),
            terms: TermManager::new(),
            next_sel_id: 0,
            pareto_front: Vec::new(),
        }
    }

    /// Add a hard constraint
    pub fn add_hard(&mut self, term: TermId) {
        self.hard_constraints.push(term);
    }

    /// Add a soft constraint with unit weight
    pub fn add_soft(&mut self, term: TermId) -> SoftConstraintId {
        self.add_soft_weighted(term, Weight::one())
    }

    /// Add a soft constraint with weight
    pub fn add_soft_weighted(&mut self, term: TermId, weight: Weight) -> SoftConstraintId {
        self.add_soft_grouped(term, weight, None)
    }

    /// Add a soft constraint with weight and group
    pub fn add_soft_grouped(
        &mut self,
        term: TermId,
        weight: Weight,
        group: Option<String>,
    ) -> SoftConstraintId {
        let id = SoftConstraintId(self.next_soft_id);
        self.next_soft_id += 1;

        let constraint = SoftConstraint {
            id,
            term,
            weight,
            group: group.clone(),
        };
        self.soft_constraints.push(constraint);

        // Add to group if specified
        if let Some(g) = group {
            self.groups.entry(g).or_default().push(id);
        }

        id
    }

    /// Add a minimization objective
    pub fn minimize(&mut self, term: TermId) -> ObjectiveId {
        self.add_objective(term, ObjectiveKind::Minimize)
    }

    /// Add a maximization objective
    pub fn maximize(&mut self, term: TermId) -> ObjectiveId {
        self.add_objective(term, ObjectiveKind::Maximize)
    }

    /// Add an objective with specified kind
    fn add_objective(&mut self, term: TermId, kind: ObjectiveKind) -> ObjectiveId {
        let id = ObjectiveId(self.next_obj_id);
        self.next_obj_id += 1;

        let objective = Objective {
            id,
            term,
            kind,
            priority: 0,
        };
        self.objectives.push(objective);

        // Initialize bounds
        self.lower_bounds.insert(id, Weight::Infinite);
        self.upper_bounds.insert(id, Weight::Infinite);

        id
    }

    /// Set the priority of an objective (lower = higher priority)
    pub fn set_priority(&mut self, id: ObjectiveId, priority: u32) {
        if let Some(obj) = self.objectives.iter_mut().find(|o| o.id == id) {
            obj.priority = priority;
        }
    }

    /// Get the number of hard constraints
    pub fn num_hard(&self) -> usize {
        self.hard_constraints.len()
    }

    /// Get the number of soft constraints
    pub fn num_soft(&self) -> usize {
        self.soft_constraints.len()
    }

    /// Get the number of objectives
    pub fn num_objectives(&self) -> usize {
        self.objectives.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &OptStats {
        &self.stats
    }

    /// Get the configuration
    pub fn config(&self) -> &OptConfig {
        &self.config
    }

    /// Get the best model
    pub fn best_model(&self) -> Option<&FxHashMap<TermId, ModelValue>> {
        self.best_model.as_ref()
    }

    /// Get the Pareto front from the last `optimize()` call (multi-objective case).
    ///
    /// Each element is a model (variable → value map) corresponding to one
    /// Pareto-optimal solution. Empty unless `optimize()` was called with
    /// multiple objectives.
    pub fn pareto_front(&self) -> &[FxHashMap<TermId, ModelValue>] {
        &self.pareto_front
    }

    /// Get the value of an objective in the best model
    pub fn objective_value(&self, id: ObjectiveId) -> Option<&Weight> {
        self.lower_bounds.get(&id)
    }

    /// Get the lower bound for an objective
    pub fn objective_lower_bound(&self, id: ObjectiveId) -> Option<&Weight> {
        self.lower_bounds.get(&id)
    }

    /// Get the upper bound for an objective
    pub fn objective_upper_bound(&self, id: ObjectiveId) -> Option<&Weight> {
        self.upper_bounds.get(&id)
    }

    /// Get all objectives
    pub fn objectives(&self) -> &[Objective] {
        &self.objectives
    }

    /// Get all soft constraints
    pub fn soft_constraints(&self) -> &[SoftConstraint] {
        &self.soft_constraints
    }

    /// Get a model value for a term
    pub fn get_model_value(&self, term: TermId) -> Option<&ModelValue> {
        self.best_model.as_ref().and_then(|m| m.get(&term))
    }

    /// Extract model as a map
    pub fn extract_model(&self) -> Option<FxHashMap<TermId, ModelValue>> {
        self.best_model.clone()
    }

    /// Push a new context level
    pub fn push(&mut self) {
        self.context_stack.push(ContextSnapshot {
            num_hard: self.hard_constraints.len(),
            num_soft: self.soft_constraints.len(),
            num_objectives: self.objectives.len(),
        });
    }

    /// Pop to previous context level
    pub fn pop(&mut self) {
        if let Some(snapshot) = self.context_stack.pop() {
            self.hard_constraints.truncate(snapshot.num_hard);
            self.soft_constraints.truncate(snapshot.num_soft);
            self.objectives.truncate(snapshot.num_objectives);
        }
    }

    /// Build a fresh [`Solver`], honoring the configured `timeout_ms`
    /// (0 = no timeout). Previously the timeout was accepted but silently
    /// dropped on every solve path.
    fn new_solver(&self) -> Solver {
        let mut solver = Solver::new();
        if self.config.timeout_ms > 0 {
            solver.set_timeout(core::time::Duration::from_millis(self.config.timeout_ms));
        }
        solver
    }

    /// Compute the wall-clock deadline for one top-level `optimize()` call from
    /// `config.timeout_ms` (0 = no deadline). `new_solver()` already bounds each
    /// *individual* `Solver::check` to `timeout_ms`, but a search that issues many
    /// solver calls (binary search, Pareto enumeration, ...) needs this additional
    /// wall-clock cap so the whole call — not just one solve inside it — actually
    /// stops within roughly `timeout_ms`.
    fn deadline(&self) -> Option<oxiz_time::Instant> {
        if self.config.timeout_ms == 0 {
            return None;
        }
        // A `checked_add` overflow (astronomically large timeout) degrades to "no
        // deadline" rather than panicking.
        oxiz_time::Instant::now()
            .checked_add(core::time::Duration::from_millis(self.config.timeout_ms))
    }

    /// True once `deadline` (if any) has passed.
    fn deadline_passed(deadline: Option<oxiz_time::Instant>) -> bool {
        deadline.is_some_and(|d| oxiz_time::Instant::now() >= d)
    }

    /// Check satisfiability (ignoring soft constraints and objectives)
    pub fn check_sat(&mut self) -> OptResult {
        let mut solver = self.new_solver();
        for &term in &self.hard_constraints {
            solver.assert(term, &mut self.terms);
        }
        self.stats.solver_calls += 1;
        match solver.check(&mut self.terms) {
            SolverResult::Sat => {
                if let Some(model) = solver.model() {
                    let snapshot = model
                        .assignments()
                        .iter()
                        .filter_map(|(&var_term, &val_term)| {
                            let mv = term_id_to_model_value(val_term, &self.terms)?;
                            Some((var_term, mv))
                        })
                        .collect();
                    self.best_model = Some(snapshot);
                }
                OptResult::Satisfiable
            }
            SolverResult::Unsat => OptResult::Unsatisfiable,
            SolverResult::Unknown => OptResult::Unknown,
        }
    }

    /// Optimize the problem
    ///
    /// This is the main optimization entry point. It will:
    /// 1. Check hard constraint satisfiability
    /// 2. Optimize soft constraints (MaxSMT)
    /// 3. Optimize objectives
    pub fn optimize(&mut self) -> Result<OptResult, OptError> {
        // If no soft constraints or objectives, just check satisfiability
        if self.soft_constraints.is_empty() && self.objectives.is_empty() {
            return Ok(self.check_sat());
        }

        // Use Pareto optimization for multiple objectives
        if self.objectives.len() > 1 {
            return self.optimize_pareto();
        }

        // For single objective or pure MaxSMT, use appropriate solver
        if !self.soft_constraints.is_empty() && self.objectives.is_empty() {
            return self.optimize_maxsmt();
        }

        // Single objective optimization
        if !self.objectives.is_empty() {
            return self.optimize_single_objective();
        }

        Ok(OptResult::Unknown)
    }

    /// Optimize using MaxSMT (soft constraints only).
    ///
    /// Uses selector-variable encoding:
    ///   For each soft constraint `t_i` with weight `w_i`:
    ///   1. Introduce fresh boolean selector `b_i`
    ///   2. Assert `b_i → t_i` (hard)
    ///   3. Model cost as integer: if `b_i` is false, pay weight `w_i`
    ///
    /// Then binary-search over the total cost budget to find the minimum
    /// cost (= maximum satisfaction) feasible assignment.
    fn optimize_maxsmt(&mut self) -> Result<OptResult, OptError> {
        if self.soft_constraints.is_empty() {
            return Ok(self.check_sat());
        }

        let int_sort = self.terms.sorts.int_sort;
        let bool_sort = self.terms.sorts.bool_sort;

        // Build selector variables and implication hard constraints.
        // Also build individual cost variables and their definitional constraints.
        let num_soft = self.soft_constraints.len();
        let mut sel_vars: Vec<TermId> = Vec::with_capacity(num_soft);
        let mut cost_vars: Vec<TermId> = Vec::with_capacity(num_soft);
        let mut selector_implications: Vec<TermId> = Vec::with_capacity(num_soft);
        let mut cost_defs: Vec<TermId> = Vec::with_capacity(num_soft * 2);

        // Determine the scale factor needed to represent every soft weight
        // (including rational ones) as an exact integer. Coercing rational
        // weights to a constant would silently change which constraints get
        // sacrificed whenever rational and integer weights are mixed.
        let mut weight_scale = BigInt::from(1);
        for sc in &self.soft_constraints {
            if let Weight::Rational(r) = &sc.weight {
                weight_scale = bigint_lcm(&weight_scale, r.denom());
            }
        }

        // Pre-compute total weight for upper bound of binary search.
        let total_weight: BigInt = self
            .soft_constraints
            .iter()
            .map(|sc| scaled_weight_int(&sc.weight, &weight_scale))
            .fold(BigInt::from(0), |acc, w| acc + w);

        for sc in &self.soft_constraints {
            let sel_name = format!("__opt_sel_{}", self.next_sel_id);
            self.next_sel_id += 1;
            let cost_name = format!("__opt_cost_{}", self.next_sel_id);
            self.next_sel_id += 1;

            let sel = self.terms.mk_var(&sel_name, bool_sort);
            let cost_var = self.terms.mk_var(&cost_name, int_sort);
            sel_vars.push(sel);
            cost_vars.push(cost_var);

            // b_i → t_i
            let implication = self.terms.mk_implies(sel, sc.term);
            selector_implications.push(implication);

            // cost_i = ite(b_i, 0, w_i)
            // Encoded as two implications: b_i → cost_i = 0; ¬b_i → cost_i = w_i
            let weight_int = scaled_weight_int(&sc.weight, &weight_scale);
            let w_term = self.terms.mk_int(weight_int);
            let zero = self.terms.mk_int(0i64);
            let not_sel = self.terms.mk_not(sel);
            let cost_eq_zero = self.terms.mk_eq(cost_var, zero);
            let cost_eq_w = self.terms.mk_eq(cost_var, w_term);
            let def_true = self.terms.mk_implies(sel, cost_eq_zero);
            let def_false = self.terms.mk_implies(not_sel, cost_eq_w);
            cost_defs.push(def_true);
            cost_defs.push(def_false);
        }

        // Build sum-of-costs expression: cost_0 + cost_1 + ... + cost_{n-1}
        let cost_sum = self.terms.mk_add(cost_vars.iter().copied());

        // Binary search: find minimum cost budget K such that the problem is SAT.
        let mut lo = BigInt::from(0i64);
        let mut hi = total_weight.clone();

        // First check feasibility with no soft constraints (hi = total cost).
        // We need to know if hard constraints are satisfiable at all.
        let feasible = {
            let mut solver = self.new_solver();
            for &h in &self.hard_constraints {
                solver.assert(h, &mut self.terms);
            }
            for &imp in &selector_implications {
                solver.assert(imp, &mut self.terms);
            }
            for &cd in &cost_defs {
                solver.assert(cd, &mut self.terms);
            }
            self.stats.solver_calls += 1;
            solver.check(&mut self.terms) == SolverResult::Sat
        };

        if !feasible {
            return Ok(OptResult::Unsatisfiable);
        }

        // Binary search for minimum cost.
        let mut best_model_snapshot: Option<FxHashMap<TermId, ModelValue>> = None;
        // Becomes true if the inner solver ever answers Unknown: in that
        // case the search did not converge and we must not report the
        // result as a proven optimum.
        let mut search_incomplete = false;

        while lo < hi {
            let mid: BigInt = (lo.clone() + hi.clone()) / 2i32;
            let bound_term = self.terms.mk_int(mid.clone());
            let cost_le_mid = self.terms.mk_le(cost_sum, bound_term);

            let mut solver = self.new_solver();
            for &h in &self.hard_constraints {
                solver.assert(h, &mut self.terms);
            }
            for &imp in &selector_implications {
                solver.assert(imp, &mut self.terms);
            }
            for &cd in &cost_defs {
                solver.assert(cd, &mut self.terms);
            }
            solver.assert(cost_le_mid, &mut self.terms);

            self.stats.solver_calls += 1;
            match solver.check(&mut self.terms) {
                SolverResult::Sat => {
                    hi = mid;
                    if let Some(model) = solver.model() {
                        best_model_snapshot = Some(
                            model
                                .assignments()
                                .iter()
                                .filter_map(|(&k, &v)| {
                                    let mv = term_id_to_model_value(v, &self.terms)?;
                                    Some((k, mv))
                                })
                                .collect(),
                        );
                    }
                }
                SolverResult::Unsat => {
                    lo = mid + BigInt::from(1i32);
                }
                SolverResult::Unknown => {
                    search_incomplete = true;
                    break;
                }
            }
        }

        // Solve once more at lo to get the actual optimal model (only
        // meaningful if the search above actually converged).
        if !search_incomplete {
            let bound_term = self.terms.mk_int(lo.clone());
            let cost_le_lo = self.terms.mk_le(cost_sum, bound_term);
            let mut solver = self.new_solver();
            for &h in &self.hard_constraints {
                solver.assert(h, &mut self.terms);
            }
            for &imp in &selector_implications {
                solver.assert(imp, &mut self.terms);
            }
            for &cd in &cost_defs {
                solver.assert(cd, &mut self.terms);
            }
            solver.assert(cost_le_lo, &mut self.terms);
            self.stats.solver_calls += 1;
            match solver.check(&mut self.terms) {
                SolverResult::Sat => {
                    if let Some(model) = solver.model() {
                        best_model_snapshot = Some(
                            model
                                .assignments()
                                .iter()
                                .filter_map(|(&k, &v)| {
                                    let mv = term_id_to_model_value(v, &self.terms)?;
                                    Some((k, mv))
                                })
                                .collect(),
                        );
                    }
                }
                SolverResult::Unsat => {
                    // The binary search's own invariant (Sat at `hi`, Unsat
                    // strictly below `lo`) guarantees `lo` is feasible; an
                    // Unsat here would mean that invariant broke down.
                    // Treat conservatively as non-convergence rather than
                    // fabricating an optimum.
                    search_incomplete = true;
                }
                SolverResult::Unknown => {
                    search_incomplete = true;
                }
            }
        }

        self.best_model = best_model_snapshot;
        if search_incomplete {
            // The search did not prove optimality. Report whatever feasible
            // model (if any) was found along the way, honestly labeled as
            // non-optimal rather than claiming a proven optimum.
            if self.best_model.is_some() {
                Ok(OptResult::Satisfiable)
            } else {
                Ok(OptResult::Unknown)
            }
        } else {
            Ok(OptResult::Optimal)
        }
    }

    /// Optimize a single objective.
    ///
    /// This drives its own binary-search (integer objectives) / strict-
    /// improvement (real objectives) loop directly against [`Self::new_solver`]
    /// rather than delegating to `oxiz_solver::Optimizer`, because the latter
    /// builds every internal probe solver with no timeout at all — it has no
    /// hook to honor `config.timeout_ms`. Every solver call here goes through
    /// `new_solver()` (so each individual `check` is itself timeout-bounded) and
    /// the loop additionally checks `Self::deadline_passed` between iterations,
    /// so the whole search — not just one solve inside it — respects
    /// `timeout_ms`. On timeout the best feasible value found so far is
    /// reported as `Satisfiable`, never a fabricated `Optimal`.
    fn optimize_single_objective(&mut self) -> Result<OptResult, OptError> {
        let obj = match self.objectives.first().cloned() {
            Some(obj) => obj,
            None => return Ok(OptResult::Unknown),
        };

        let is_int = self
            .terms
            .get(obj.term)
            .is_some_and(|t| t.sort == self.terms.sorts.int_sort);

        let deadline = self.deadline();
        let outcome = if is_int {
            self.optimize_int_objective(obj.term, obj.kind, deadline)
        } else {
            self.optimize_real_objective(obj.term, obj.kind, deadline)
        };

        match outcome {
            SingleObjOutcome::Optimal { value, model } => {
                self.lower_bounds.insert(obj.id, value.clone());
                self.upper_bounds.insert(obj.id, value);
                self.best_model = Some(model);
                Ok(OptResult::Optimal)
            }
            SingleObjOutcome::Feasible { value, model } => {
                self.lower_bounds.insert(obj.id, value.clone());
                self.upper_bounds.insert(obj.id, value);
                self.best_model = Some(model);
                Ok(OptResult::Satisfiable)
            }
            SingleObjOutcome::Unbounded => Ok(OptResult::Unbounded),
            SingleObjOutcome::Unsat => Ok(OptResult::Unsatisfiable),
            SingleObjOutcome::Unknown => Ok(OptResult::Unknown),
        }
    }

    /// Probe `obj_term <= bound` (minimize) / `obj_term >= bound` (maximize)
    /// on a fresh, timeout-bounded solver seeded with the hard constraints.
    fn probe_int_bound(
        &mut self,
        obj_term: TermId,
        bound: &BigInt,
        minimize: bool,
    ) -> IntBoundProbe {
        let bound_term = self.terms.mk_int(bound.clone());
        let constraint = if minimize {
            self.terms.mk_le(obj_term, bound_term)
        } else {
            self.terms.mk_ge(obj_term, bound_term)
        };
        let mut solver = self.new_solver();
        for &h in &self.hard_constraints {
            solver.assert(h, &mut self.terms);
        }
        solver.assert(constraint, &mut self.terms);
        self.stats.solver_calls += 1;
        match solver.check(&mut self.terms) {
            SolverResult::Sat => match solver.model() {
                Some(model) => {
                    let model = model.clone();
                    let value_term = model.eval(obj_term, &mut self.terms);
                    match self.terms.get(value_term).map(|t| t.kind.clone()) {
                        Some(TermKind::IntConst(n)) => IntBoundProbe::Sat(n, model),
                        _ => IntBoundProbe::Unknown,
                    }
                }
                None => IntBoundProbe::Unknown,
            },
            SolverResult::Unsat => IntBoundProbe::Unsat,
            SolverResult::Unknown => IntBoundProbe::Unknown,
        }
    }

    /// Probe a real objective bound (`<`/`>` when `strict`, else `<=`/`>=`,
    /// direction chosen by `minimize`) on a fresh, timeout-bounded solver.
    fn probe_real_bound(
        &mut self,
        obj_term: TermId,
        bound: Rational64,
        minimize: bool,
        strict: bool,
    ) -> RealBoundProbe {
        let bound_term = self.terms.mk_real(bound);
        let constraint = match (minimize, strict) {
            (true, false) => self.terms.mk_le(obj_term, bound_term),
            (true, true) => self.terms.mk_lt(obj_term, bound_term),
            (false, false) => self.terms.mk_ge(obj_term, bound_term),
            (false, true) => self.terms.mk_gt(obj_term, bound_term),
        };
        let mut solver = self.new_solver();
        for &h in &self.hard_constraints {
            solver.assert(h, &mut self.terms);
        }
        solver.assert(constraint, &mut self.terms);
        self.stats.solver_calls += 1;
        match solver.check(&mut self.terms) {
            SolverResult::Sat => match solver.model() {
                Some(model) => {
                    let model = model.clone();
                    let value_term = model.eval(obj_term, &mut self.terms);
                    match self.terms.get(value_term).map(|t| t.kind.clone()) {
                        Some(TermKind::RealConst(v)) => RealBoundProbe::Sat(v, model),
                        Some(TermKind::IntConst(n)) => match n.to_i64() {
                            Some(i) => RealBoundProbe::Sat(Rational64::from_integer(i), model),
                            None => RealBoundProbe::Unknown,
                        },
                        _ => RealBoundProbe::Unknown,
                    }
                }
                None => RealBoundProbe::Unknown,
            },
            SolverResult::Unsat => RealBoundProbe::Unsat,
            SolverResult::Unknown => RealBoundProbe::Unknown,
        }
    }

    /// Optimize an integer objective by bounded binary search (mirrors
    /// `oxiz_solver::Optimizer::optimize_int`), with periodic `deadline` checks
    /// so a slow search yields an honest `Feasible` (unproven) result instead of
    /// running unbounded.
    fn optimize_int_objective(
        &mut self,
        obj_term: TermId,
        kind: ObjectiveKind,
        deadline: Option<oxiz_time::Instant>,
    ) -> SingleObjOutcome {
        let minimize = matches!(kind, ObjectiveKind::Minimize);
        // 2^40 is far larger than any realistic finite optimum yet small enough
        // that the i64-based simplex never overflows while probing near it.
        let threshold = BigInt::from(1u64 << 40);
        let one = BigInt::from(1);

        // Phase 1: initial feasibility + attained value.
        let (mut best_value, mut best_model) = {
            let mut solver = self.new_solver();
            for &h in &self.hard_constraints {
                solver.assert(h, &mut self.terms);
            }
            self.stats.solver_calls += 1;
            match solver.check(&mut self.terms) {
                SolverResult::Unsat => return SingleObjOutcome::Unsat,
                SolverResult::Unknown => return SingleObjOutcome::Unknown,
                SolverResult::Sat => match solver.model() {
                    Some(model) => {
                        let model = model.clone();
                        let value_term = model.eval(obj_term, &mut self.terms);
                        match self.terms.get(value_term).map(|t| t.kind.clone()) {
                            Some(TermKind::IntConst(n)) => (n, model),
                            // Not a concrete integer in this model: return the
                            // feasible point as-is rather than driving a
                            // numeric search we cannot interpret.
                            _ => {
                                return SingleObjOutcome::Feasible {
                                    value: term_id_to_weight(value_term, &self.terms),
                                    model: snapshot_model(&model, &self.terms),
                                };
                            }
                        }
                    }
                    None => return SingleObjOutcome::Unknown,
                },
            }
        };

        if Self::deadline_passed(deadline) {
            return SingleObjOutcome::Feasible {
                value: Weight::Int(best_value),
                model: snapshot_model(&best_model, &self.terms),
            };
        }

        // Phase 2: unbounded-direction probe at the extreme representable bound.
        let extreme = if minimize {
            -threshold.clone()
        } else {
            threshold.clone()
        };
        match self.probe_int_bound(obj_term, &extreme, minimize) {
            IntBoundProbe::Unknown => {
                return SingleObjOutcome::Feasible {
                    value: Weight::Int(best_value),
                    model: snapshot_model(&best_model, &self.terms),
                };
            }
            IntBoundProbe::Sat(_, _) => return SingleObjOutcome::Unbounded,
            IntBoundProbe::Unsat => {}
        }

        // Phase 3: binary search in the now-finite bracket.
        let (mut lo, mut hi) = if minimize {
            ((-&threshold) + &one, best_value.clone())
        } else {
            (best_value.clone(), &threshold - &one)
        };

        let mut converged = false;
        let mut guard = 0u32;
        loop {
            if lo >= hi {
                converged = true;
                break;
            }
            if Self::deadline_passed(deadline) {
                break;
            }
            guard += 1;
            if guard > 4096 {
                break;
            }
            let mid = if minimize {
                floor_half(&lo + &hi)
            } else {
                floor_half(&lo + &hi + &one)
            };
            match self.probe_int_bound(obj_term, &mid, minimize) {
                IntBoundProbe::Unknown => break,
                IntBoundProbe::Sat(val, model) => {
                    best_model = model;
                    best_value = val.clone();
                    if minimize {
                        hi = val;
                    } else {
                        lo = val;
                    }
                }
                IntBoundProbe::Unsat => {
                    if minimize {
                        lo = &mid + &one;
                    } else {
                        hi = &mid - &one;
                    }
                }
            }
        }

        let value = Weight::Int(best_value);
        let model = snapshot_model(&best_model, &self.terms);
        if converged {
            SingleObjOutcome::Optimal { value, model }
        } else {
            SingleObjOutcome::Feasible { value, model }
        }
    }

    /// Optimize a real objective via unbounded detection plus strict-improvement
    /// search (mirrors `oxiz_solver::Optimizer::optimize_real`), with periodic
    /// `deadline` checks between iterations.
    fn optimize_real_objective(
        &mut self,
        obj_term: TermId,
        kind: ObjectiveKind,
        deadline: Option<oxiz_time::Instant>,
    ) -> SingleObjOutcome {
        let minimize = matches!(kind, ObjectiveKind::Minimize);

        // Phase 1: initial feasibility + attained value.
        let (mut best_value, mut best_model) = {
            let mut solver = self.new_solver();
            for &h in &self.hard_constraints {
                solver.assert(h, &mut self.terms);
            }
            self.stats.solver_calls += 1;
            match solver.check(&mut self.terms) {
                SolverResult::Unsat => return SingleObjOutcome::Unsat,
                SolverResult::Unknown => return SingleObjOutcome::Unknown,
                SolverResult::Sat => match solver.model() {
                    Some(model) => {
                        let model = model.clone();
                        let value_term = model.eval(obj_term, &mut self.terms);
                        match self.terms.get(value_term).map(|t| t.kind.clone()) {
                            Some(TermKind::RealConst(v)) => (v, model),
                            Some(TermKind::IntConst(n)) => match n.to_i64() {
                                Some(i) => (Rational64::from_integer(i), model),
                                None => return SingleObjOutcome::Unknown,
                            },
                            _ => {
                                return SingleObjOutcome::Feasible {
                                    value: term_id_to_weight(value_term, &self.terms),
                                    model: snapshot_model(&model, &self.terms),
                                };
                            }
                        }
                    }
                    None => return SingleObjOutcome::Unknown,
                },
            }
        };

        if Self::deadline_passed(deadline) {
            return SingleObjOutcome::Feasible {
                value: Weight::Rational(BigRational::new(
                    BigInt::from(*best_value.numer()),
                    BigInt::from(*best_value.denom()),
                )),
                model: snapshot_model(&best_model, &self.terms),
            };
        }

        // Phase 2: unbounded detection. 2^40 fits the i64 numerator of
        // `Rational64` with room to spare and keeps the theory solver well
        // clear of overflow.
        let huge = Rational64::from_integer(1i64 << 40);
        let extreme = if minimize { -huge } else { huge };
        match self.probe_real_bound(obj_term, extreme, minimize, false) {
            RealBoundProbe::Sat(_, _) => return SingleObjOutcome::Unbounded,
            RealBoundProbe::Unknown => {
                let value = Weight::Rational(BigRational::new(
                    BigInt::from(*best_value.numer()),
                    BigInt::from(*best_value.denom()),
                ));
                return SingleObjOutcome::Feasible {
                    value,
                    model: snapshot_model(&best_model, &self.terms),
                };
            }
            RealBoundProbe::Unsat => {}
        }

        // Phase 3: strict-improvement search for the exact attained optimum.
        let max_iterations = 1000;
        let mut converged = false;
        for _ in 0..max_iterations {
            if Self::deadline_passed(deadline) {
                break;
            }
            match self.probe_real_bound(obj_term, best_value, minimize, true) {
                RealBoundProbe::Sat(v, model) => {
                    best_value = v;
                    best_model = model;
                }
                RealBoundProbe::Unsat => {
                    // No strictly-better solution exists — best_value is optimal.
                    converged = true;
                    break;
                }
                RealBoundProbe::Unknown => break,
            }
        }

        let value = Weight::Rational(BigRational::new(
            BigInt::from(*best_value.numer()),
            BigInt::from(*best_value.denom()),
        ));
        let model = snapshot_model(&best_model, &self.terms);
        if converged {
            SingleObjOutcome::Optimal { value, model }
        } else {
            // Iteration budget or deadline exhausted without proving
            // optimality: be honest, this is a feasible point only.
            SingleObjOutcome::Feasible { value, model }
        }
    }

    /// Optimize using Pareto (multiple objectives).
    ///
    /// Guided-improvement enumeration, driven directly against
    /// [`Self::new_solver`] for the same reason as
    /// [`Self::optimize_single_objective`]: `oxiz_solver::Optimizer` offers no
    /// timeout hook. `deadline` is checked before each solver call; a search
    /// that reaches the deadline (or the `max_points` cap) before the solver
    /// proves no further improving point exists reports `Satisfiable` — the
    /// front found so far — rather than a fabricated `Optimal`.
    fn optimize_pareto(&mut self) -> Result<OptResult, OptError> {
        if self.objectives.is_empty() {
            return Ok(self.check_sat());
        }

        let deadline = self.deadline();

        // Objectives in priority order (lower `priority` value first) so the
        // configured priorities are honored rather than silently ignored in
        // favor of raw insertion order.
        let mut ordered: Vec<Objective> = self.objectives.clone();
        ordered.sort_by_key(|o| o.priority);
        let kinds: Vec<ObjectiveKind> = ordered.iter().map(|o| o.kind).collect();
        let obj_terms: Vec<TermId> = ordered.iter().map(|o| o.term).collect();

        struct ParetoPointLocal {
            values: Vec<Weight>,
            model: FxHashMap<TermId, ModelValue>,
        }

        let mut front: Vec<ParetoPointLocal> = Vec::new();
        // Blocking clauses forcing each subsequent solution to strictly
        // improve at least one objective relative to every point found so far.
        let mut blocking: Vec<TermId> = Vec::new();
        let max_points = 100; // Limit to avoid runaway search
        let mut proved_complete = false;

        for _ in 0..max_points {
            if Self::deadline_passed(deadline) {
                break;
            }

            let mut solver = self.new_solver();
            for &h in &self.hard_constraints {
                solver.assert(h, &mut self.terms);
            }
            for &b in &blocking {
                solver.assert(b, &mut self.terms);
            }
            self.stats.solver_calls += 1;
            match solver.check(&mut self.terms) {
                SolverResult::Sat => {
                    let Some(model) = solver.model() else {
                        break;
                    };
                    let model = model.clone();

                    let mut raw_values = Vec::with_capacity(obj_terms.len());
                    for &term in &obj_terms {
                        raw_values.push(model.eval(term, &mut self.terms));
                    }
                    let values: Vec<Weight> = raw_values
                        .iter()
                        .map(|&v| term_id_to_weight(v, &self.terms))
                        .collect();

                    // Remove any recorded point dominated by the new point.
                    front.retain(|pt| !dominates_weights(&values, &pt.values, &kinds));

                    // Record the new point unless (defensively) it is
                    // dominated by a surviving one.
                    let dominated = front
                        .iter()
                        .any(|pt| dominates_weights(&pt.values, &values, &kinds));
                    if !dominated {
                        front.push(ParetoPointLocal {
                            values: values.clone(),
                            model: snapshot_model(&model, &self.terms),
                        });
                    }

                    // Block all solutions weakly dominated by this point: a
                    // future solution must strictly improve at least one
                    // objective.
                    let mut improvement_disjuncts = Vec::with_capacity(obj_terms.len());
                    for (idx, &term) in obj_terms.iter().enumerate() {
                        let current_value = raw_values[idx];
                        let improvement = match kinds[idx] {
                            ObjectiveKind::Minimize => self.terms.mk_lt(term, current_value),
                            ObjectiveKind::Maximize => self.terms.mk_gt(term, current_value),
                        };
                        improvement_disjuncts.push(improvement);
                    }
                    blocking.push(self.terms.mk_or(improvement_disjuncts));
                }
                SolverResult::Unsat => {
                    proved_complete = true;
                    break;
                }
                SolverResult::Unknown => break,
            }
        }

        if front.is_empty() {
            return Ok(OptResult::Unsatisfiable);
        }

        self.pareto_front.clear();
        for point in &front {
            self.pareto_front.push(point.model.clone());
        }

        // Use the last Pareto point as the best model.
        if let Some(last) = self.pareto_front.last() {
            self.best_model = Some(last.clone());
        }

        if proved_complete {
            Ok(OptResult::Optimal)
        } else {
            // The `max_points` cap or the deadline cut enumeration short
            // before the solver proved no further improving point exists:
            // the returned front is real but not proven complete.
            Ok(OptResult::Satisfiable)
        }
    }

    /// Reset the context
    pub fn reset(&mut self) {
        self.hard_constraints.clear();
        self.soft_constraints.clear();
        self.next_soft_id = 0;
        self.objectives.clear();
        self.next_obj_id = 0;
        self.stats = OptStats::default();
        self.best_model = None;
        self.lower_bounds.clear();
        self.upper_bounds.clear();
        self.groups.clear();
        self.context_stack.clear();
        self.terms = TermManager::new();
        self.next_sel_id = 0;
        self.pareto_front.clear();
    }

    /// Get all soft constraints in a group
    pub fn get_group(&self, group: &str) -> Option<&[SoftConstraintId]> {
        self.groups.get(group).map(|v| v.as_slice())
    }

    /// Get the weight of a soft constraint
    pub fn soft_weight(&self, id: SoftConstraintId) -> Option<&Weight> {
        self.soft_constraints.get(id.0 as usize).map(|c| &c.weight)
    }

    /// Check if a soft constraint is satisfied in the best model
    pub fn is_soft_satisfied(&self, id: SoftConstraintId) -> bool {
        // Get the soft constraint
        let soft = match self.soft_constraints.get(id.0 as usize) {
            Some(s) => s,
            None => return false,
        };

        // Check if we have a model
        let model = match &self.best_model {
            Some(m) => m,
            None => return false, // No model, can't determine satisfaction
        };

        // Recursively evaluate the (possibly compound) constraint term against
        // the model. A direct lookup alone would mis-report any non-leaf soft
        // term — e.g. `(not p)` or `(<= x 3)` — as unsatisfied, over-counting
        // the cost. `None` (undeterminable) is treated conservatively as
        // unsatisfied.
        matches!(model_eval_bool(soft.term, model, &self.terms), Some(true))
    }

    /// Get the sum of weights of unsatisfied soft constraints
    pub fn cost(&self) -> Weight {
        // If no model, all soft constraints are unsatisfied
        if self.best_model.is_none() {
            return self
                .soft_constraints
                .iter()
                .fold(Weight::zero(), |acc, c| acc.add(&c.weight));
        }

        // Sum weights of unsatisfied constraints
        self.soft_constraints
            .iter()
            .filter(|c| !self.is_soft_satisfied(c.id))
            .fold(Weight::zero(), |acc, c| acc.add(&c.weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_context_new() {
        let ctx = OptContext::new();
        assert_eq!(ctx.num_hard(), 0);
        assert_eq!(ctx.num_soft(), 0);
        assert_eq!(ctx.num_objectives(), 0);
    }

    #[test]
    fn test_add_hard_constraint() {
        let mut ctx = OptContext::new();
        let term = TermId::from(1);
        ctx.add_hard(term);
        assert_eq!(ctx.num_hard(), 1);
    }

    #[test]
    fn test_add_soft_constraint() {
        let mut ctx = OptContext::new();
        let term = TermId::from(1);
        let id = ctx.add_soft(term);
        assert_eq!(id.0, 0);
        assert_eq!(ctx.num_soft(), 1);
    }

    #[test]
    fn test_add_weighted_soft() {
        let mut ctx = OptContext::new();
        let term = TermId::from(1);
        let id = ctx.add_soft_weighted(term, Weight::from(5));
        assert_eq!(ctx.soft_weight(id), Some(&Weight::from(5)));
    }

    #[test]
    fn test_add_grouped_soft() {
        let mut ctx = OptContext::new();
        let term1 = TermId::from(1);
        let term2 = TermId::from(2);

        let id1 = ctx.add_soft_grouped(term1, Weight::one(), Some("group1".to_string()));
        let id2 = ctx.add_soft_grouped(term2, Weight::one(), Some("group1".to_string()));

        let group = ctx.get_group("group1");
        assert!(group.is_some());
        assert_eq!(group.expect("test operation should succeed").len(), 2);
        assert!(group.expect("test operation should succeed").contains(&id1));
        assert!(group.expect("test operation should succeed").contains(&id2));
    }

    #[test]
    fn test_add_objectives() {
        let mut ctx = OptContext::new();
        let term1 = TermId::from(1);
        let term2 = TermId::from(2);

        let id1 = ctx.minimize(term1);
        let id2 = ctx.maximize(term2);

        assert_eq!(ctx.num_objectives(), 2);
        assert_eq!(id1.0, 0);
        assert_eq!(id2.0, 1);
    }

    #[test]
    fn test_push_pop() {
        let mut ctx = OptContext::new();
        let term1 = TermId::from(1);
        let term2 = TermId::from(2);

        ctx.add_hard(term1);
        assert_eq!(ctx.num_hard(), 1);

        ctx.push();
        ctx.add_hard(term2);
        assert_eq!(ctx.num_hard(), 2);

        ctx.pop();
        assert_eq!(ctx.num_hard(), 1);
    }

    #[test]
    fn test_reset() {
        let mut ctx = OptContext::new();
        ctx.add_hard(TermId::from(1));
        ctx.add_soft(TermId::from(2));
        ctx.minimize(TermId::from(3));

        ctx.reset();

        assert_eq!(ctx.num_hard(), 0);
        assert_eq!(ctx.num_soft(), 0);
        assert_eq!(ctx.num_objectives(), 0);
    }

    #[test]
    fn test_config() {
        let config = OptConfig {
            incremental: false,
            timeout_ms: 5000,
            ..Default::default()
        };
        let ctx = OptContext::with_config(config);
        assert!(!ctx.config.incremental);
        assert_eq!(ctx.config.timeout_ms, 5000);
    }

    #[test]
    fn test_objective_bounds() {
        let mut ctx = OptContext::new();
        let term = TermId::from(1);
        let id = ctx.minimize(term);

        // Initially bounds are infinite
        assert!(ctx.objective_lower_bound(id).is_some());
        assert!(ctx.objective_upper_bound(id).is_some());
    }

    #[test]
    fn test_get_objectives() {
        let mut ctx = OptContext::new();
        let term1 = TermId::from(1);
        let term2 = TermId::from(2);

        ctx.minimize(term1);
        ctx.maximize(term2);

        let objs = ctx.objectives();
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn test_get_soft_constraints() {
        let mut ctx = OptContext::new();
        let term1 = TermId::from(1);
        let term2 = TermId::from(2);

        ctx.add_soft(term1);
        ctx.add_soft_weighted(term2, Weight::from(5));

        let softs = ctx.soft_constraints();
        assert_eq!(softs.len(), 2);
    }

    #[test]
    fn test_extract_model() {
        let ctx = OptContext::new();
        let model = ctx.extract_model();
        assert!(model.is_none()); // No model yet
    }

    #[test]
    fn test_get_model_value() {
        let ctx = OptContext::new();
        let term = TermId::from(1);
        let value = ctx.get_model_value(term);
        assert!(value.is_none()); // No model yet
    }

    #[test]
    fn test_opt_result_display() {
        assert_eq!(OptResult::Optimal.to_string(), "optimal");
        assert_eq!(OptResult::Satisfiable.to_string(), "satisfiable");
        assert_eq!(OptResult::Unsatisfiable.to_string(), "unsatisfiable");
        assert_eq!(OptResult::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_model_value_display() {
        assert_eq!(ModelValue::Bool(true).to_string(), "true");
        assert_eq!(ModelValue::Bool(false).to_string(), "false");
        assert_eq!(ModelValue::Int(BigInt::from(42)).to_string(), "42");
        assert_eq!(
            ModelValue::Rational(BigRational::new(BigInt::from(3), BigInt::from(2))).to_string(),
            "3/2"
        );
    }

    // Tests for From implementations

    #[test]
    fn test_soft_constraint_id_from_u32() {
        let id: SoftConstraintId = SoftConstraintId::from(5u32);
        assert_eq!(id.raw(), 5);
    }

    #[test]
    fn test_soft_constraint_id_from_usize() {
        let id: SoftConstraintId = SoftConstraintId::from(10usize);
        assert_eq!(id.raw(), 10);
    }

    #[test]
    fn test_soft_constraint_id_to_u32() {
        let id = SoftConstraintId::new(7);
        let n: u32 = id.into();
        assert_eq!(n, 7);
    }

    #[test]
    fn test_soft_constraint_id_to_usize() {
        let id = SoftConstraintId::new(9);
        let n: usize = id.into();
        assert_eq!(n, 9);
    }

    #[test]
    fn test_soft_constraint_id_roundtrip() {
        let original = 42u32;
        let id: SoftConstraintId = original.into();
        let back: u32 = id.into();
        assert_eq!(original, back);
    }
}
