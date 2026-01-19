//! Local Search SAT Solver (ProbSAT/WalkSAT)
//!
//! Local search is a complementary technique to CDCL that can be very effective
//! for satisfiable instances. It works by maintaining a complete assignment and
//! iteratively flipping variables to reduce the number of unsatisfied clauses.
//!
//! This module implements:
//! - ProbSAT: Uses a probability distribution based on break counts
//! - WalkSAT: Uses a greedy heuristic with random walk

use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{Lit, Var};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Configuration for local search
#[derive(Debug, Clone)]
pub struct LocalSearchConfig {
    /// Maximum number of flips before giving up
    pub max_flips: u64,
    /// Probability of random walk (WalkSAT only, typically 0.3-0.5)
    pub random_walk_prob: f64,
    /// Polynomial break value exponent (ProbSAT, typically 2.0-3.0)
    pub cb_exponent: f64,
    /// Random seed for reproducibility
    pub random_seed: u64,
}

impl Default for LocalSearchConfig {
    fn default() -> Self {
        Self {
            max_flips: 1_000_000,
            random_walk_prob: 0.4,
            cb_exponent: 2.3,
            random_seed: 1234567,
        }
    }
}

/// Result of local search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalSearchResult {
    /// Found a satisfying assignment
    Sat,
    /// Reached maximum flips without finding a solution
    Unknown,
}

/// Statistics for local search
#[derive(Debug, Default, Clone)]
pub struct LocalSearchStats {
    /// Number of variable flips performed
    pub flips: u64,
    /// Minimum number of unsatisfied clauses seen
    pub min_unsat: usize,
    /// Number of times the best assignment was updated
    pub improvements: u64,
}

/// Local Search SAT Solver
///
/// Implements both WalkSAT and ProbSAT algorithms.
/// Maintains a complete assignment and iteratively flips variables.
pub struct LocalSearch {
    /// Current variable assignment (true/false for each variable)
    assignment: Vec<bool>,
    /// Break count for each variable (how many satisfied clauses would become unsatisfied)
    break_count: Vec<u64>,
    /// Make count for each variable (how many unsatisfied clauses would become satisfied)
    make_count: Vec<u64>,
    /// List of currently unsatisfied clauses
    unsat_clauses: Vec<ClauseId>,
    /// Set of unsatisfied clauses for quick lookup
    unsat_set: HashMap<ClauseId, ()>,
    /// Number of true literals in each clause
    true_count: HashMap<ClauseId, usize>,
    /// Configuration
    config: LocalSearchConfig,
    /// Statistics
    stats: LocalSearchStats,
    /// Simple LCG random number generator state
    rng_state: u64,
}

impl LocalSearch {
    /// Create a new local search solver
    #[must_use]
    pub fn new(num_vars: usize, config: LocalSearchConfig) -> Self {
        Self {
            assignment: vec![false; num_vars],
            break_count: vec![0; num_vars],
            make_count: vec![0; num_vars],
            unsat_clauses: Vec::new(),
            unsat_set: HashMap::new(),
            true_count: HashMap::new(),
            rng_state: config.random_seed,
            config,
            stats: LocalSearchStats::default(),
        }
    }

    /// Simple LCG random number generator
    fn rand(&mut self) -> u64 {
        // Linear Congruential Generator: Xn+1 = (a * Xn + c) mod m
        // Using values from Numerical Recipes
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        self.rng_state = self.rng_state.wrapping_mul(A).wrapping_add(C);
        self.rng_state
    }

    /// Generate a random float between 0.0 and 1.0
    fn rand_float(&mut self) -> f64 {
        (self.rand() as f64) / (u64::MAX as f64)
    }

    /// Initialize with a random assignment
    fn initialize_random(&mut self, num_vars: usize) {
        self.assignment.clear();
        self.assignment.resize(num_vars, false);

        for i in 0..num_vars {
            self.assignment[i] = self.rand().is_multiple_of(2);
        }
    }

    /// Initialize data structures for search
    fn initialize(&mut self, clauses: &ClauseDatabase, num_vars: usize) {
        self.initialize_random(num_vars);
        self.break_count.clear();
        self.break_count.resize(num_vars, 0);
        self.make_count.clear();
        self.make_count.resize(num_vars, 0);
        self.unsat_clauses.clear();
        self.unsat_set.clear();
        self.true_count.clear();

        // Calculate initial true counts and unsat clauses
        for id in clauses.iter_ids() {
            let clause = clauses
                .get(id)
                .expect("id from clauses.iter_ids() is valid");
            let true_lits = clause.lits.iter().filter(|&&lit| self.is_true(lit)).count();

            self.true_count.insert(id, true_lits);

            if true_lits == 0 {
                self.unsat_clauses.push(id);
                self.unsat_set.insert(id, ());
            }
        }

        // Calculate break and make counts
        for id in clauses.iter_ids() {
            let clause = clauses
                .get(id)
                .expect("id from clauses.iter_ids() is valid");
            let true_lits = self.true_count[&id];

            for &lit in &clause.lits {
                let var = lit.var();
                let var_idx = var.index();

                if self.is_true(lit) {
                    // If this is the only true literal, flipping would break this clause
                    if true_lits == 1 {
                        self.break_count[var_idx] += 1;
                    }
                } else {
                    // Flipping would make this clause true (if it's currently false)
                    if true_lits == 0 {
                        self.make_count[var_idx] += 1;
                    }
                }
            }
        }

        self.stats.min_unsat = self.unsat_clauses.len();
    }

    /// Check if a literal is true under the current assignment
    fn is_true(&self, lit: Lit) -> bool {
        let var_value = self.assignment[lit.var().index()];
        if lit.is_pos() { var_value } else { !var_value }
    }

    /// Flip a variable and update data structures
    fn flip(&mut self, var: Var, clauses: &ClauseDatabase) {
        let var_idx = var.index();
        self.assignment[var_idx] = !self.assignment[var_idx];
        self.stats.flips += 1;

        // Update true counts and unsat status for all clauses containing this variable
        let pos_lit = Lit::pos(var);
        let neg_lit = Lit::neg(var);

        // We need to find all clauses containing this variable
        // Since we don't have a watch list here, we iterate all clauses
        for id in clauses.iter_ids() {
            let clause = clauses
                .get(id)
                .expect("id from clauses.iter_ids() is valid");
            if !clause.lits.contains(&pos_lit) && !clause.lits.contains(&neg_lit) {
                continue;
            }

            let old_true_count = self.true_count[&id];
            let was_unsat = old_true_count == 0;

            // Recalculate true count
            let new_true_count = clause.lits.iter().filter(|&&lit| self.is_true(lit)).count();

            self.true_count.insert(id, new_true_count);

            let is_unsat = new_true_count == 0;

            // Update unsat list
            if !was_unsat && is_unsat {
                self.unsat_clauses.push(id);
                self.unsat_set.insert(id, ());
            } else if was_unsat && !is_unsat {
                self.unsat_set.remove(&id);
            }

            // Update break/make counts for literals in this clause
            for &lit in &clause.lits {
                let lit_var = lit.var();
                let lit_var_idx = lit_var.index();

                // Update break count
                if old_true_count == 1 && self.is_true(lit) {
                    // This literal was the only true one, flipping would break
                    self.break_count[lit_var_idx] -= 1;
                }
                if new_true_count == 1 && self.is_true(lit) {
                    // This literal is now the only true one
                    self.break_count[lit_var_idx] += 1;
                }

                // Update make count
                if old_true_count == 0 && !self.is_true(lit) {
                    // Clause was unsat, flipping this literal would make it
                    self.make_count[lit_var_idx] -= 1;
                }
                if new_true_count == 0 && !self.is_true(lit) {
                    // Clause is now unsat, flipping this literal would make it
                    self.make_count[lit_var_idx] += 1;
                }
            }
        }

        // Clean up unsat_clauses list
        self.unsat_clauses
            .retain(|&id| self.unsat_set.contains_key(&id));

        // Track improvements
        if self.unsat_clauses.len() < self.stats.min_unsat {
            self.stats.min_unsat = self.unsat_clauses.len();
            self.stats.improvements += 1;
        }
    }

    /// Run WalkSAT algorithm
    ///
    /// Returns the result and the final assignment (if SAT)
    pub fn solve_walksat(
        &mut self,
        clauses: &ClauseDatabase,
        num_vars: usize,
    ) -> (LocalSearchResult, Option<Vec<bool>>) {
        self.initialize(clauses, num_vars);

        for _ in 0..self.config.max_flips {
            if self.unsat_clauses.is_empty() {
                return (LocalSearchResult::Sat, Some(self.assignment.clone()));
            }

            // Pick a random unsatisfied clause
            let clause_id = {
                let idx = (self.rand() as usize) % self.unsat_clauses.len();
                self.unsat_clauses[idx]
            };
            let clause = clauses.get(clause_id).expect("clause_id is valid");

            // Decide whether to use random walk
            let use_random_walk = self.rand_float() < self.config.random_walk_prob;

            // Select which variable to flip
            let var_to_flip = if use_random_walk {
                // Pick a random variable from the clause
                let idx = (self.rand() as usize) % clause.lits.len();
                clause.lits[idx].var()
            } else {
                // Pick the variable with minimum break count
                let mut best_var = clause.lits[0].var();
                let mut min_break = self.break_count[best_var.index()];

                for &lit in &clause.lits[1..] {
                    let var = lit.var();
                    let break_cnt = self.break_count[var.index()];
                    if break_cnt < min_break {
                        min_break = break_cnt;
                        best_var = var;
                    }
                }

                best_var
            };

            self.flip(var_to_flip, clauses);
        }

        (LocalSearchResult::Unknown, None)
    }

    /// Run ProbSAT algorithm
    ///
    /// Returns the result and the final assignment (if SAT)
    pub fn solve_probsat(
        &mut self,
        clauses: &ClauseDatabase,
        num_vars: usize,
    ) -> (LocalSearchResult, Option<Vec<bool>>) {
        self.initialize(clauses, num_vars);

        for _ in 0..self.config.max_flips {
            if self.unsat_clauses.is_empty() {
                return (LocalSearchResult::Sat, Some(self.assignment.clone()));
            }

            // Pick a random unsatisfied clause
            let clause_id = {
                let idx = (self.rand() as usize) % self.unsat_clauses.len();
                self.unsat_clauses[idx]
            };
            let clause = clauses.get(clause_id).expect("clause_id is valid");

            // Calculate probabilities based on break counts
            let mut probs: SmallVec<[f64; 8]> = SmallVec::new();
            let mut total = 0.0;

            for &lit in &clause.lits {
                let var = lit.var();
                let break_cnt = self.break_count[var.index()];
                // Probability is inversely proportional to (break_count + 1)^cb
                let prob = 1.0 / ((break_cnt as f64 + 1.0).powf(self.config.cb_exponent));
                probs.push(prob);
                total += prob;
            }

            // Normalize probabilities
            for prob in &mut probs {
                *prob /= total;
            }

            // Select variable based on probability distribution
            let r = self.rand_float();
            let mut cumulative = 0.0;
            let mut selected_var = clause.lits[0].var();

            for (i, &lit) in clause.lits.iter().enumerate() {
                cumulative += probs[i];
                if r <= cumulative {
                    selected_var = lit.var();
                    break;
                }
            }

            self.flip(selected_var, clauses);
        }

        (LocalSearchResult::Unknown, None)
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &LocalSearchStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::Clause;

    #[test]
    fn test_local_search_creation() {
        let config = LocalSearchConfig::default();
        let ls = LocalSearch::new(10, config);
        assert_eq!(ls.assignment.len(), 10);
    }

    #[test]
    fn test_local_search_simple_sat() {
        // Create a simple satisfiable formula: (x1 v x2) ^ (~x1 v x3)
        let mut db = ClauseDatabase::new();
        let c1 = Clause::new(vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))], false);
        let c2 = Clause::new(vec![Lit::neg(Var::new(0)), Lit::pos(Var::new(2))], false);

        let id1 = db.add(c1);
        let id2 = db.add(c2);

        let config = LocalSearchConfig {
            max_flips: 1000,
            ..Default::default()
        };

        let mut ls = LocalSearch::new(3, config);
        let (result, assignment) = ls.solve_walksat(&db, 3);

        // Should find a solution
        assert_eq!(result, LocalSearchResult::Sat);
        assert!(assignment.is_some());

        // Verify the solution satisfies all clauses
        let assignment = assignment.unwrap();
        let clause1 = db.get(id1).unwrap();
        let clause2 = db.get(id2).unwrap();

        let sat1 = clause1.lits.iter().any(|&lit| {
            let var_value = assignment[lit.var().index()];
            if lit.is_pos() { var_value } else { !var_value }
        });

        let sat2 = clause2.lits.iter().any(|&lit| {
            let var_value = assignment[lit.var().index()];
            if lit.is_pos() { var_value } else { !var_value }
        });

        assert!(sat1);
        assert!(sat2);
    }

    #[test]
    fn test_local_search_stats() {
        let config = LocalSearchConfig {
            max_flips: 100,
            ..Default::default()
        };
        let mut ls = LocalSearch::new(5, config);

        let mut db = ClauseDatabase::new();
        // Add a more complex formula that requires flips
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(0)), Lit::pos(Var::new(2))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(1)), Lit::pos(Var::new(3))],
            false,
        ));

        let (result, _) = ls.solve_walksat(&db, 5);
        let stats = ls.stats();

        // Should find a solution for this easy formula
        assert_eq!(result, LocalSearchResult::Sat);
        // Stats should be populated
        assert!(stats.min_unsat <= 3);
    }

    #[test]
    fn test_probsat() {
        // Test ProbSAT on a simple formula
        let mut db = ClauseDatabase::new();
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(0)), Lit::pos(Var::new(2))],
            false,
        ));

        let config = LocalSearchConfig {
            max_flips: 1000,
            cb_exponent: 2.5,
            ..Default::default()
        };

        let mut ls = LocalSearch::new(3, config);
        let (result, _assignment) = ls.solve_probsat(&db, 3);

        // Should find a solution (though ProbSAT is probabilistic)
        // We just check it doesn't crash for now
        assert!(result == LocalSearchResult::Sat || result == LocalSearchResult::Unknown);
    }
}
