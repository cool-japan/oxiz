//! Weighted MaxSAT Theory Solver
//!
//! Provides a theory solver interface for weighted MaxSAT within SMT context.
//! This integrates weighted soft constraints with the CDCL(T) framework.
//!
//! # Features
//!
//! - Soft clause management with weights
//! - Cost bound propagation
//! - Conflict detection for cost bounds
//! - Model extraction with cost information
//!
//! # Example
//!
//! ```ignore
//! use oxiz_theories::wmaxsat::{WMaxSatSolver, SoftClause};
//!
//! let mut solver = WMaxSatSolver::new();
//! solver.add_soft(SoftClause::new(vec![lit1, lit2], 5));
//! solver.set_cost_bound(10);
//!
//! let result = solver.check();
//! ```

use std::collections::{HashMap, HashSet};

/// A literal identifier (variable with polarity)
pub type LitId = i32;

/// Weight type for soft clauses
pub type Weight = u64;

/// A soft clause with weight
#[derive(Debug, Clone)]
pub struct SoftClause {
    /// Literals in the clause
    pub literals: Vec<LitId>,
    /// Weight of the clause
    pub weight: Weight,
    /// Clause ID
    pub id: usize,
    /// Relaxation literal (if any)
    pub relaxation_lit: Option<LitId>,
}

impl SoftClause {
    /// Create a new soft clause
    pub fn new(literals: Vec<LitId>, weight: Weight) -> Self {
        Self {
            literals,
            weight,
            id: 0,
            relaxation_lit: None,
        }
    }

    /// Create with ID
    pub fn with_id(literals: Vec<LitId>, weight: Weight, id: usize) -> Self {
        Self {
            literals,
            weight,
            id,
            relaxation_lit: None,
        }
    }

    /// Set relaxation literal
    pub fn set_relaxation(&mut self, lit: LitId) {
        self.relaxation_lit = Some(lit);
    }

    /// Check if clause is satisfied under given assignment
    pub fn is_satisfied(&self, assignment: &HashMap<i32, bool>) -> bool {
        for &lit in &self.literals {
            let var = lit.abs();
            let pol = lit > 0;
            if let Some(&val) = assignment.get(&var)
                && val == pol
            {
                return true;
            }
        }
        // Also check relaxation literal
        if let Some(relax) = self.relaxation_lit {
            let var = relax.abs();
            let pol = relax > 0;
            if let Some(&val) = assignment.get(&var)
                && val == pol
            {
                return true;
            }
        }
        false
    }

    /// Check if clause is falsified under given assignment
    pub fn is_falsified(&self, assignment: &HashMap<i32, bool>) -> bool {
        for &lit in &self.literals {
            let var = lit.abs();
            let pol = lit > 0;
            match assignment.get(&var) {
                Some(&val) if val == pol => return false, // Literal is true
                None => return false,                     // Literal is unassigned
                _ => {}                                   // Literal is false, continue
            }
        }
        // Check relaxation literal
        if let Some(relax) = self.relaxation_lit {
            let var = relax.abs();
            let pol = relax > 0;
            match assignment.get(&var) {
                Some(&val) if val == pol => return false,
                None => return false,
                _ => {}
            }
        }
        true
    }
}

/// Result of WMaxSAT checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WMaxSatResult {
    /// Satisfiable within cost bound
    Sat,
    /// Unsatisfiable (cost exceeds bound)
    Unsat,
    /// Unknown (needs more propagation)
    Unknown,
}

/// Configuration for WMaxSAT solver
#[derive(Debug, Clone)]
pub struct WMaxSatConfig {
    /// Initial cost bound (infinity = no bound)
    pub initial_bound: Option<Weight>,
    /// Enable stratification (solve by weight strata)
    pub stratification: bool,
    /// Enable core extraction
    pub extract_cores: bool,
}

impl Default for WMaxSatConfig {
    fn default() -> Self {
        Self {
            initial_bound: None,
            stratification: true,
            extract_cores: true,
        }
    }
}

/// Statistics for WMaxSAT solving
#[derive(Debug, Clone, Default)]
pub struct WMaxSatStats {
    /// Number of soft clauses
    pub num_soft_clauses: u64,
    /// Total weight of all soft clauses
    pub total_weight: Weight,
    /// Current cost (sum of falsified soft clause weights)
    pub current_cost: Weight,
    /// Best cost found
    pub best_cost: Weight,
    /// Number of bounds updates
    pub bound_updates: u64,
    /// Number of cores found
    pub cores_found: u64,
}

/// Weighted MaxSAT Theory Solver
///
/// Integrates weighted soft constraints with SMT solving.
#[derive(Debug)]
pub struct WMaxSatSolver {
    /// Soft clauses
    soft_clauses: Vec<SoftClause>,
    /// Cost bound (maximum allowed cost)
    cost_bound: Option<Weight>,
    /// Current assignment
    assignment: HashMap<i32, bool>,
    /// Falsified clauses (by ID)
    falsified: HashSet<usize>,
    /// Configuration
    config: WMaxSatConfig,
    /// Statistics
    stats: WMaxSatStats,
    /// Decision level to clause ID mapping
    level_clauses: Vec<HashSet<usize>>,
    /// Next clause ID
    next_clause_id: usize,
    /// Next relaxation variable ID
    next_relax_var: i32,
}

impl WMaxSatSolver {
    /// Create a new WMaxSAT solver
    pub fn new() -> Self {
        Self {
            soft_clauses: Vec::new(),
            cost_bound: None,
            assignment: HashMap::new(),
            falsified: HashSet::new(),
            config: WMaxSatConfig::default(),
            stats: WMaxSatStats::default(),
            level_clauses: vec![HashSet::new()],
            next_clause_id: 0,
            next_relax_var: 1_000_000, // Start high to avoid conflicts
        }
    }

    /// Create with configuration
    pub fn with_config(config: WMaxSatConfig) -> Self {
        let cost_bound = config.initial_bound;
        Self {
            soft_clauses: Vec::new(),
            cost_bound,
            assignment: HashMap::new(),
            falsified: HashSet::new(),
            config,
            stats: WMaxSatStats::default(),
            level_clauses: vec![HashSet::new()],
            next_clause_id: 0,
            next_relax_var: 1_000_000,
        }
    }

    /// Add a soft clause
    pub fn add_soft(&mut self, mut clause: SoftClause) {
        clause.id = self.next_clause_id;
        self.next_clause_id += 1;

        self.stats.num_soft_clauses += 1;
        self.stats.total_weight += clause.weight;

        self.soft_clauses.push(clause);
    }

    /// Add a soft clause with a relaxation variable
    pub fn add_soft_with_relax(&mut self, mut clause: SoftClause) -> LitId {
        clause.id = self.next_clause_id;
        self.next_clause_id += 1;

        let relax_var = self.next_relax_var;
        self.next_relax_var += 1;
        clause.set_relaxation(relax_var);

        self.stats.num_soft_clauses += 1;
        self.stats.total_weight += clause.weight;

        self.soft_clauses.push(clause);
        relax_var
    }

    /// Set the cost bound
    pub fn set_cost_bound(&mut self, bound: Weight) {
        self.cost_bound = Some(bound);
        self.stats.bound_updates += 1;
    }

    /// Get the current cost bound
    pub fn cost_bound(&self) -> Option<Weight> {
        self.cost_bound
    }

    /// Assign a literal
    pub fn assign(&mut self, lit: LitId, level: usize) {
        let var = lit.abs();
        let pol = lit > 0;
        self.assignment.insert(var, pol);

        // Ensure level exists
        while self.level_clauses.len() <= level {
            self.level_clauses.push(HashSet::new());
        }

        // Check if this falsifies any soft clauses
        for clause in &self.soft_clauses {
            if clause.is_falsified(&self.assignment) && !self.falsified.contains(&clause.id) {
                self.falsified.insert(clause.id);
                self.level_clauses[level].insert(clause.id);
            }
        }

        self.update_cost();
    }

    /// Unassign a literal
    pub fn unassign(&mut self, lit: LitId) {
        let var = lit.abs();
        self.assignment.remove(&var);
        // Note: falsified clauses are cleared via backtrack
    }

    /// Backtrack to a decision level
    pub fn backtrack(&mut self, level: usize) {
        // Remove assignments above the level
        // In practice, the SAT solver handles this; we just update our tracking

        // Clear falsified clauses from higher levels
        while self.level_clauses.len() > level + 1 {
            if let Some(clauses) = self.level_clauses.pop() {
                for clause_id in clauses {
                    self.falsified.remove(&clause_id);
                }
            }
        }

        self.update_cost();
    }

    /// Update the current cost
    fn update_cost(&mut self) {
        let mut cost = 0;
        for clause_id in &self.falsified {
            if let Some(clause) = self.soft_clauses.iter().find(|c| c.id == *clause_id) {
                cost += clause.weight;
            }
        }
        self.stats.current_cost = cost;
    }

    /// Get the current cost
    pub fn current_cost(&self) -> Weight {
        self.stats.current_cost
    }

    /// Check if the current assignment respects the cost bound
    pub fn check(&self) -> WMaxSatResult {
        if let Some(bound) = self.cost_bound
            && self.stats.current_cost > bound
        {
            return WMaxSatResult::Unsat;
        }
        WMaxSatResult::Sat
    }

    /// Propagate cost bounds
    ///
    /// Returns literals that must be false to stay within bound
    pub fn propagate(&self) -> Vec<LitId> {
        let mut propagations = Vec::new();

        if let Some(bound) = self.cost_bound {
            let remaining = bound.saturating_sub(self.stats.current_cost);

            // Check each unassigned soft clause
            for clause in &self.soft_clauses {
                if self.falsified.contains(&clause.id) {
                    continue;
                }

                // If falsifying this clause would exceed the bound
                if clause.weight > remaining {
                    // At least one literal must be true
                    // If all but one are false, propagate the last one
                    let unassigned: Vec<_> = clause
                        .literals
                        .iter()
                        .filter(|&&lit| !self.assignment.contains_key(&lit.abs()))
                        .copied()
                        .collect();

                    if unassigned.len() == 1 {
                        propagations.push(unassigned[0]);
                    }
                }
            }
        }

        propagations
    }

    /// Get conflict clause when cost bound is exceeded
    pub fn get_conflict(&self) -> Option<Vec<LitId>> {
        if let Some(bound) = self.cost_bound
            && self.stats.current_cost > bound
        {
            // The conflict is: the conjunction of assignments that falsified clauses
            let mut conflict = Vec::new();
            for clause_id in &self.falsified {
                if let Some(clause) = self.soft_clauses.iter().find(|c| c.id == *clause_id) {
                    // Add negation of the literals that falsified this clause
                    for &lit in &clause.literals {
                        let var = lit.abs();
                        if let Some(&val) = self.assignment.get(&var) {
                            let assigned_lit = if val { var } else { -var };
                            if !conflict.contains(&-assigned_lit) {
                                conflict.push(-assigned_lit);
                            }
                        }
                    }
                }
            }
            return Some(conflict);
        }
        None
    }

    /// Extract a core (set of soft clauses that cannot all be satisfied)
    pub fn extract_core(&mut self) -> Vec<usize> {
        if !self.config.extract_cores {
            return Vec::new();
        }

        // The core is the set of falsified clauses
        self.stats.cores_found = self.stats.cores_found.saturating_add(1);
        self.falsified.iter().copied().collect()
    }

    /// Get weight strata for stratification
    pub fn get_strata(&self) -> Vec<Weight> {
        if !self.config.stratification {
            return Vec::new();
        }

        let mut weights: Vec<Weight> = self.soft_clauses.iter().map(|c| c.weight).collect();
        weights.sort_unstable();
        weights.dedup();
        weights.reverse(); // Highest weight first
        weights
    }

    /// Get clauses in a weight stratum
    pub fn clauses_in_stratum(&self, weight: Weight) -> Vec<usize> {
        self.soft_clauses
            .iter()
            .filter(|c| c.weight == weight)
            .map(|c| c.id)
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &WMaxSatStats {
        &self.stats
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.assignment.clear();
        self.falsified.clear();
        self.level_clauses.clear();
        self.level_clauses.push(HashSet::new());
        self.stats.current_cost = 0;
    }

    /// Get all soft clauses
    pub fn soft_clauses(&self) -> &[SoftClause] {
        &self.soft_clauses
    }

    /// Get number of soft clauses
    pub fn num_soft(&self) -> usize {
        self.soft_clauses.len()
    }
}

impl Default for WMaxSatSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_clause_creation() {
        let clause = SoftClause::new(vec![1, -2, 3], 5);
        assert_eq!(clause.literals.len(), 3);
        assert_eq!(clause.weight, 5);
    }

    #[test]
    fn test_soft_clause_satisfied() {
        let clause = SoftClause::new(vec![1, 2], 1);
        let mut assignment = HashMap::new();
        assignment.insert(1, true);
        assert!(clause.is_satisfied(&assignment));

        assignment.clear();
        assignment.insert(1, false);
        assignment.insert(2, false);
        assert!(!clause.is_satisfied(&assignment));
    }

    #[test]
    fn test_soft_clause_falsified() {
        let clause = SoftClause::new(vec![1, 2], 1);
        let mut assignment = HashMap::new();
        assignment.insert(1, false);
        assert!(!clause.is_falsified(&assignment)); // 2 is unassigned

        assignment.insert(2, false);
        assert!(clause.is_falsified(&assignment)); // Both false
    }

    #[test]
    fn test_wmaxsat_creation() {
        let solver = WMaxSatSolver::new();
        assert_eq!(solver.num_soft(), 0);
        assert!(solver.cost_bound().is_none());
    }

    #[test]
    fn test_wmaxsat_add_soft() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1, 2], 5));
        solver.add_soft(SoftClause::new(vec![3], 3));

        assert_eq!(solver.num_soft(), 2);
        assert_eq!(solver.stats().total_weight, 8);
    }

    #[test]
    fn test_wmaxsat_cost_bound() {
        let mut solver = WMaxSatSolver::new();
        solver.set_cost_bound(10);
        assert_eq!(solver.cost_bound(), Some(10));
    }

    #[test]
    fn test_wmaxsat_assignment() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1, 2], 5));

        solver.assign(-1, 0);
        solver.assign(-2, 0);

        assert_eq!(solver.current_cost(), 5);
    }

    #[test]
    fn test_wmaxsat_check_within_bound() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1], 5));
        solver.set_cost_bound(10);

        solver.assign(-1, 0);
        assert_eq!(solver.check(), WMaxSatResult::Sat);
    }

    #[test]
    fn test_wmaxsat_check_exceeds_bound() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1], 15));
        solver.set_cost_bound(10);

        solver.assign(-1, 0);
        assert_eq!(solver.check(), WMaxSatResult::Unsat);
    }

    #[test]
    fn test_wmaxsat_backtrack() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1], 5));
        solver.add_soft(SoftClause::new(vec![2], 3));

        solver.assign(-1, 0);
        solver.assign(-2, 1);

        assert_eq!(solver.current_cost(), 8);

        solver.backtrack(0);
        assert_eq!(solver.current_cost(), 5);
    }

    #[test]
    fn test_wmaxsat_get_conflict() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1], 15));
        solver.set_cost_bound(10);

        solver.assign(-1, 0);
        let conflict = solver.get_conflict();
        assert!(conflict.is_some());
    }

    #[test]
    fn test_wmaxsat_strata() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1], 5));
        solver.add_soft(SoftClause::new(vec![2], 3));
        solver.add_soft(SoftClause::new(vec![3], 5));
        solver.add_soft(SoftClause::new(vec![4], 1));

        let strata = solver.get_strata();
        assert_eq!(strata, vec![5, 3, 1]);
    }

    #[test]
    fn test_wmaxsat_clauses_in_stratum() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1], 5));
        solver.add_soft(SoftClause::new(vec![2], 3));
        solver.add_soft(SoftClause::new(vec![3], 5));

        let stratum = solver.clauses_in_stratum(5);
        assert_eq!(stratum.len(), 2);
    }

    #[test]
    fn test_wmaxsat_reset() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1], 5));
        solver.assign(-1, 0);

        assert_eq!(solver.current_cost(), 5);

        solver.reset();
        assert_eq!(solver.current_cost(), 0);
    }

    #[test]
    fn test_wmaxsat_with_relaxation() {
        let mut solver = WMaxSatSolver::new();
        let relax = solver.add_soft_with_relax(SoftClause::new(vec![1], 5));

        assert!(relax >= 1_000_000);
        assert!(solver.soft_clauses()[0].relaxation_lit.is_some());
    }

    #[test]
    fn test_wmaxsat_propagate() {
        let mut solver = WMaxSatSolver::new();
        solver.add_soft(SoftClause::new(vec![1, 2], 15));
        solver.set_cost_bound(10);

        // Assign 1 = false, leaving only 2 unassigned
        solver.assign(-1, 0);

        let props = solver.propagate();
        // Should propagate 2 = true to avoid exceeding bound
        assert_eq!(props.len(), 1);
        assert_eq!(props[0], 2);
    }
}
