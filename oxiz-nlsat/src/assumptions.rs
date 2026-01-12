//! Assumption-based incremental solving.
//!
//! This module provides support for incremental SMT solving with assumptions.
//! Assumptions are temporary constraints that can be added and removed efficiently
//! using a push/pop interface, enabling incremental solving scenarios.
//!
//! Reference: Modern SMT solvers' incremental interfaces

use crate::clause::{ClauseId, NULL_CLAUSE};
use crate::types::Literal;
use std::collections::HashSet;

/// A scope level for incremental solving.
#[derive(Debug, Clone, PartialEq)]
pub struct Scope {
    /// Clauses added at this scope level.
    pub clauses: Vec<ClauseId>,
    /// Assumptions active at this scope level.
    pub assumptions: Vec<Literal>,
    /// Number of boolean variables at this scope level.
    pub num_bool_vars: u32,
    /// Number of arithmetic variables at this scope level.
    pub num_arith_vars: u32,
}

impl Scope {
    /// Create a new scope.
    pub fn new(num_bool_vars: u32, num_arith_vars: u32) -> Self {
        Self {
            clauses: Vec::new(),
            assumptions: Vec::new(),
            num_bool_vars,
            num_arith_vars,
        }
    }

    /// Add a clause to this scope.
    pub fn add_clause(&mut self, clause_id: ClauseId) {
        self.clauses.push(clause_id);
    }

    /// Add an assumption to this scope.
    pub fn add_assumption(&mut self, lit: Literal) {
        self.assumptions.push(lit);
    }
}

/// Manager for assumption-based incremental solving.
#[derive(Debug, Clone)]
pub struct AssumptionManager {
    /// Stack of scopes for push/pop.
    scopes: Vec<Scope>,
    /// Current assumptions (union of all scope assumptions).
    current_assumptions: Vec<Literal>,
    /// Set of current assumptions for quick lookup.
    assumption_set: HashSet<Literal>,
}

impl AssumptionManager {
    /// Create a new assumption manager.
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            current_assumptions: Vec::new(),
            assumption_set: HashSet::new(),
        }
    }

    /// Get the current scope level (0 = base level).
    pub fn level(&self) -> usize {
        self.scopes.len()
    }

    /// Push a new scope level.
    pub fn push(&mut self, num_bool_vars: u32, num_arith_vars: u32) {
        self.scopes.push(Scope::new(num_bool_vars, num_arith_vars));
    }

    /// Pop the current scope level.
    ///
    /// Returns the popped scope, or None if at base level.
    pub fn pop(&mut self) -> Option<Scope> {
        if let Some(scope) = self.scopes.pop() {
            // Remove assumptions from this scope
            for &lit in &scope.assumptions {
                self.assumption_set.remove(&lit);
            }

            // Rebuild current assumptions
            self.current_assumptions.clear();
            for scope in &self.scopes {
                self.current_assumptions.extend(&scope.assumptions);
            }

            Some(scope)
        } else {
            None
        }
    }

    /// Add an assumption at the current scope level.
    ///
    /// If no scope is active, the assumption is added to the base level.
    pub fn add_assumption(&mut self, lit: Literal) {
        if !self.assumption_set.contains(&lit) {
            self.assumption_set.insert(lit);
            self.current_assumptions.push(lit);

            if let Some(scope) = self.scopes.last_mut() {
                scope.add_assumption(lit);
            }
        }
    }

    /// Add a clause to the current scope.
    pub fn add_clause(&mut self, clause_id: ClauseId) {
        if clause_id == NULL_CLAUSE {
            return;
        }

        if let Some(scope) = self.scopes.last_mut() {
            scope.add_clause(clause_id);
        }
    }

    /// Get all current assumptions.
    pub fn assumptions(&self) -> &[Literal] {
        &self.current_assumptions
    }

    /// Check if a literal is an assumption.
    pub fn is_assumption(&self, lit: Literal) -> bool {
        self.assumption_set.contains(&lit)
    }

    /// Clear all assumptions and scopes.
    pub fn clear(&mut self) {
        self.scopes.clear();
        self.current_assumptions.clear();
        self.assumption_set.clear();
    }

    /// Get the number of active scopes.
    pub fn num_scopes(&self) -> usize {
        self.scopes.len()
    }

    /// Get the clauses added at a specific scope level.
    pub fn clauses_at_level(&self, level: usize) -> Option<&[ClauseId]> {
        self.scopes.get(level).map(|s| s.clauses.as_slice())
    }

    /// Get all clauses that should be removed when popping.
    pub fn clauses_to_remove(&self) -> Vec<ClauseId> {
        if let Some(scope) = self.scopes.last() {
            scope.clauses.clone()
        } else {
            Vec::new()
        }
    }
}

impl Default for AssumptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BoolVar;

    fn make_literal(var: BoolVar, sign: bool) -> Literal {
        Literal::new(var, sign)
    }

    #[test]
    fn test_assumption_manager_new() {
        let mgr = AssumptionManager::new();
        assert_eq!(mgr.level(), 0);
        assert_eq!(mgr.assumptions().len(), 0);
    }

    #[test]
    fn test_push_pop() {
        let mut mgr = AssumptionManager::new();

        mgr.push(0, 0);
        assert_eq!(mgr.level(), 1);

        mgr.push(0, 0);
        assert_eq!(mgr.level(), 2);

        mgr.pop();
        assert_eq!(mgr.level(), 1);

        mgr.pop();
        assert_eq!(mgr.level(), 0);

        assert_eq!(mgr.pop(), None);
    }

    #[test]
    fn test_assumptions() {
        let mut mgr = AssumptionManager::new();

        let lit1 = make_literal(0, true);
        let lit2 = make_literal(1, false);

        mgr.push(0, 0);
        mgr.add_assumption(lit1);

        assert_eq!(mgr.assumptions().len(), 1);
        assert!(mgr.is_assumption(lit1));
        assert!(!mgr.is_assumption(lit2));

        mgr.push(0, 0);
        mgr.add_assumption(lit2);

        assert_eq!(mgr.assumptions().len(), 2);
        assert!(mgr.is_assumption(lit1));
        assert!(mgr.is_assumption(lit2));

        // Pop should remove lit2
        mgr.pop();
        assert_eq!(mgr.assumptions().len(), 1);
        assert!(mgr.is_assumption(lit1));
        assert!(!mgr.is_assumption(lit2));

        // Pop should remove lit1
        mgr.pop();
        assert_eq!(mgr.assumptions().len(), 0);
        assert!(!mgr.is_assumption(lit1));
    }

    #[test]
    fn test_duplicate_assumptions() {
        let mut mgr = AssumptionManager::new();

        let lit1 = make_literal(0, true);

        mgr.push(0, 0);
        mgr.add_assumption(lit1);
        mgr.add_assumption(lit1); // Duplicate

        assert_eq!(mgr.assumptions().len(), 1);
    }

    #[test]
    fn test_clause_tracking() {
        let mut mgr = AssumptionManager::new();

        mgr.push(0, 0);
        mgr.add_clause(1);
        mgr.add_clause(2);

        assert_eq!(mgr.clauses_at_level(0).unwrap().len(), 2);

        mgr.push(0, 0);
        mgr.add_clause(3);

        assert_eq!(mgr.clauses_at_level(1).unwrap().len(), 1);

        let to_remove = mgr.clauses_to_remove();
        assert_eq!(to_remove.len(), 1);
        assert_eq!(to_remove[0], 3);

        mgr.pop();
        assert_eq!(mgr.clauses_at_level(1), None);
    }

    #[test]
    fn test_clear() {
        let mut mgr = AssumptionManager::new();

        mgr.push(0, 0);
        mgr.add_assumption(make_literal(0, true));
        mgr.add_clause(1);

        mgr.clear();

        assert_eq!(mgr.level(), 0);
        assert_eq!(mgr.assumptions().len(), 0);
        assert_eq!(mgr.num_scopes(), 0);
    }
}
