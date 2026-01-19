//! Optimization context.
//!
//! This module provides the main interface for optimization modulo theories.
//! It integrates MaxSAT solving with SMT solving to support:
//! - Soft constraints with weights
//! - Objective function optimization (minimize/maximize)
//! - Multi-objective (Pareto) optimization
//!
//! Reference: Z3's `opt/opt_context.cpp`

use crate::maxsat::{MaxSatConfig, MaxSatError, MaxSatResult, MaxSatSolver, Weight};
use crate::objective::{Objective, ObjectiveId, ObjectiveKind};
use crate::pareto::ParetoConfig;
use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;
use thiserror::Error;

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

    /// Get the best model
    pub fn best_model(&self) -> Option<&FxHashMap<TermId, ModelValue>> {
        self.best_model.as_ref()
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

    /// Check satisfiability (ignoring soft constraints and objectives)
    pub fn check_sat(&mut self) -> OptResult {
        // For now, return Unknown - full integration with SMT solver needed
        OptResult::Unknown
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

    /// Optimize using MaxSMT (soft constraints only)
    fn optimize_maxsmt(&mut self) -> Result<OptResult, OptError> {
        // Create MaxSAT solver with our configuration
        let _maxsat = MaxSatSolver::with_config(self.config.maxsat.clone());

        // Add soft constraints to MaxSAT solver
        // Note: This is a simplified encoding that treats SMT terms as propositional
        // A full implementation would need to integrate with the SMT solver
        // to handle theory reasoning
        for soft in &self.soft_constraints {
            // For now, we can't directly encode SMT terms to SAT literals
            // This would require a proper tseitin transformation or similar
            // So we track this as a structural placeholder
            let _ = soft;
        }

        self.stats.solver_calls += 1;

        // In a full implementation, we would:
        // 1. Encode SMT terms to SAT using the SMT solver's Boolean abstraction
        // 2. Solve the MaxSAT problem
        // 3. Extract the model and verify with SMT solver
        // 4. Iterate if theory conflicts are found

        // For now, return Unknown to indicate this needs SMT integration
        // The actual algorithms are implemented in maxsat.rs and work correctly
        // when given SAT-level inputs
        Ok(OptResult::Unknown)
    }

    /// Optimize a single objective
    fn optimize_single_objective(&mut self) -> Result<OptResult, OptError> {
        use crate::omt::{OmtConfig, OmtSolver, OmtStrategy};

        // Get the single objective
        let obj = match self.objectives.first().cloned() {
            Some(obj) => obj,
            None => return Ok(OptResult::Unknown),
        };

        // Create OMT solver with appropriate strategy
        // Use binary search for better performance on most objectives
        let omt_config = OmtConfig {
            strategy: OmtStrategy::BinarySearch,
            ..Default::default()
        };

        let _omt = OmtSolver::with_config(omt_config);

        // Add hard constraints (as arithmetic constraints)
        // Note: This requires converting TermId to arithmetic constraints
        // which would need integration with the SMT solver
        for _hard in &self.hard_constraints {
            // Would convert TermId to ArithConstraint here
        }

        // Set the objective based on kind
        // Note: This requires converting TermId to an arithmetic expression
        // which would need integration with the SMT solver
        let _objective_term = obj.term;

        self.stats.solver_calls += 1;

        // In a full implementation, we would:
        // 1. Convert hard constraints from TermId to ArithConstraint
        // 2. Convert objective TermId to arithmetic expression
        // 3. Call omt.solve() with minimize or maximize
        // 4. Extract bounds and model
        // 5. Update self.lower_bounds, self.upper_bounds, self.best_model

        // The OMT algorithms are fully implemented in omt.rs and work correctly
        // when given arithmetic-level inputs
        Ok(OptResult::Unknown)
    }

    /// Optimize using Pareto (multiple objectives)
    fn optimize_pareto(&mut self) -> Result<OptResult, OptError> {
        use crate::pareto::ParetoSolver;

        // Create Pareto solver with our configuration
        let _pareto = ParetoSolver::with_config(self.config.pareto.clone());

        // Set up objectives
        // Note: This requires converting TermId objectives to arithmetic expressions
        // which would need integration with the SMT solver
        for obj in &self.objectives {
            let _obj_term = obj.term;
            let _obj_kind = &obj.kind;
            // Would add objectives to pareto solver here
            // pareto.add_objective(...) based on obj.kind (Minimize/Maximize)
        }

        // Add constraints
        // Note: Similar to single objective, this needs TermId -> constraint conversion
        for _hard in &self.hard_constraints {
            // Would convert and add constraints
        }

        self.stats.solver_calls += 1;

        // In a full implementation, we would:
        // 1. Convert objectives from TermId to arithmetic expressions
        // 2. Convert constraints from TermId to appropriate constraint types
        // 3. Call pareto.solve() to enumerate Pareto front
        // 4. Extract all Pareto-optimal solutions
        // 5. Update bounds and models accordingly

        // The Pareto algorithms are fully implemented in pareto.rs and work correctly
        // when given arithmetic-level inputs
        Ok(OptResult::Unknown)
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

        // Check if the constraint's term is satisfied in the model
        // For now, we check if the term has a value in the model
        // A full implementation would need to evaluate the term
        match model.get(&soft.term) {
            Some(ModelValue::Bool(true)) => true,
            Some(ModelValue::Bool(false)) => false,
            _ => false, // Non-boolean or not in model - conservatively say unsatisfied
        }
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
        assert_eq!(group.unwrap().len(), 2);
        assert!(group.unwrap().contains(&id1));
        assert!(group.unwrap().contains(&id2));
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
