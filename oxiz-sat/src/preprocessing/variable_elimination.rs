//! Advanced Variable Elimination for SAT Preprocessing.
#![allow(dead_code, clippy::ptr_arg)] // Under development
//!
//! Implements sophisticated variable elimination techniques including:
//! - Bounded Variable Elimination (BVE)
//! - Asymmetric Variable Elimination
//! - Resolution-based elimination with cost analysis

use crate::{Clause, Lit, Var};
use std::collections::{HashMap, HashSet, VecDeque};

/// Variable elimination engine for SAT preprocessing.
pub struct VariableEliminator {
    config: EliminationConfig,
    stats: EliminationStats,
}

/// Configuration for variable elimination.
#[derive(Clone, Debug)]
pub struct EliminationConfig {
    /// Maximum clause size to consider for elimination
    pub max_clause_size: usize,
    /// Maximum number of resolvents allowed per variable
    pub max_resolvents: usize,
    /// Enable bounded variable elimination
    pub bounded_elimination: bool,
    /// Enable asymmetric elimination
    pub asymmetric_elimination: bool,
    /// Cost threshold for elimination
    pub cost_threshold: f64,
    /// Maximum variable activity for elimination
    pub max_activity: f64,
}

impl Default for EliminationConfig {
    fn default() -> Self {
        Self {
            max_clause_size: 10,
            max_resolvents: 100,
            bounded_elimination: true,
            asymmetric_elimination: true,
            cost_threshold: 0.0,
            max_activity: 0.1,
        }
    }
}

/// Statistics about variable elimination.
#[derive(Clone, Debug, Default)]
pub struct EliminationStats {
    /// Number of variables eliminated
    pub vars_eliminated: usize,
    /// Number of clauses before elimination
    pub clauses_before: usize,
    /// Number of clauses after elimination
    pub clauses_after: usize,
    /// Total literals before
    pub literals_before: usize,
    /// Total literals after
    pub literals_after: usize,
    /// Number of resolution steps performed
    pub resolutions: usize,
}

impl VariableEliminator {
    /// Create a new variable eliminator.
    pub fn new(config: EliminationConfig) -> Self {
        Self {
            config,
            stats: EliminationStats::default(),
        }
    }

    /// Eliminate variables from a clause database.
    pub fn eliminate(&mut self, clauses: &mut Vec<Clause>) -> Vec<Var> {
        self.stats.clauses_before = clauses.len();
        self.stats.literals_before = clauses.iter().map(|c| c.lits.len()).sum();

        let mut eliminated = Vec::new();
        let mut occurrence_map = self.build_occurrence_map(clauses);

        // Build elimination queue ordered by resolution cost
        let mut queue = self.build_elimination_queue(clauses, &occurrence_map);

        while let Some(var) = queue.pop_front() {
            if self.should_eliminate(var, clauses, &occurrence_map)
                && self.eliminate_variable(var, clauses, &mut occurrence_map)
            {
                eliminated.push(var);
                self.stats.vars_eliminated += 1;
            }
        }

        self.stats.clauses_after = clauses.len();
        self.stats.literals_after = clauses.iter().map(|c| c.lits.len()).sum();

        eliminated
    }

    /// Build occurrence map for literals.
    fn build_occurrence_map(&self, clauses: &[Clause]) -> HashMap<Lit, Vec<usize>> {
        let mut map: HashMap<Lit, Vec<usize>> = HashMap::new();

        for (idx, clause) in clauses.iter().enumerate() {
            for &lit in &clause.lits {
                map.entry(lit).or_default().push(idx);
            }
        }

        map
    }

    /// Build elimination queue ordered by cost.
    fn build_elimination_queue(
        &self,
        clauses: &[Clause],
        occurrence_map: &HashMap<Lit, Vec<usize>>,
    ) -> VecDeque<Var> {
        let max_var = clauses
            .iter()
            .flat_map(|c| c.lits.iter())
            .map(|&lit| lit.var())
            .max()
            .unwrap_or(Var(0));

        let mut costs: Vec<(Var, f64)> = Vec::new();

        for var_idx in 0..=max_var.0 {
            let var = Var(var_idx);
            let cost = self.elimination_cost(var, clauses, occurrence_map);
            costs.push((var, cost));
        }

        // Sort by cost
        costs.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        costs.into_iter().map(|(var, _)| var).collect()
    }

    /// Compute the cost of eliminating a variable.
    ///
    /// Cost = (number of positive occurrences) × (number of negative occurrences) - (total occurrences)
    fn elimination_cost(
        &self,
        var: Var,
        _clauses: &[Clause],
        occurrence_map: &HashMap<Lit, Vec<usize>>,
    ) -> f64 {
        let pos_lit = Lit::pos(var);
        let neg_lit = Lit::neg(var);

        let pos_count = occurrence_map.get(&pos_lit).map_or(0, |v| v.len());
        let neg_count = occurrence_map.get(&neg_lit).map_or(0, |v| v.len());
        let total_count = pos_count + neg_count;

        if total_count == 0 {
            return f64::INFINITY;
        }

        // Cost: number of new clauses - number of removed clauses
        let new_clauses = pos_count * neg_count;
        let removed_clauses = pos_count + neg_count;

        (new_clauses as f64) - (removed_clauses as f64)
    }

    /// Check if a variable should be eliminated.
    fn should_eliminate(
        &self,
        var: Var,
        clauses: &[Clause],
        occurrence_map: &HashMap<Lit, Vec<usize>>,
    ) -> bool {
        let cost = self.elimination_cost(var, clauses, occurrence_map);

        // Don't eliminate if cost is too high
        if cost > self.config.cost_threshold {
            return false;
        }

        // Check if occurrences are within bounds
        let pos_lit = Lit::pos(var);
        let neg_lit = Lit::neg(var);

        let pos_count = occurrence_map.get(&pos_lit).map_or(0, |v| v.len());
        let neg_count = occurrence_map.get(&neg_lit).map_or(0, |v| v.len());

        // Don't eliminate if too many resolvents
        if pos_count * neg_count > self.config.max_resolvents {
            return false;
        }

        // Check clause sizes
        for &idx in occurrence_map
            .get(&pos_lit)
            .into_iter()
            .chain(occurrence_map.get(&neg_lit))
            .flatten()
        {
            if let Some(clause) = clauses.get(idx)
                && clause.lits.len() > self.config.max_clause_size
            {
                return false;
            }
        }

        true
    }

    /// Eliminate a variable by resolution.
    fn eliminate_variable(
        &mut self,
        var: Var,
        clauses: &mut Vec<Clause>,
        occurrence_map: &mut HashMap<Lit, Vec<usize>>,
    ) -> bool {
        let pos_lit = Lit::pos(var);
        let neg_lit = Lit::neg(var);

        // Get clauses containing positive and negative literals
        let pos_clauses: Vec<usize> = occurrence_map.get(&pos_lit).cloned().unwrap_or_default();
        let neg_clauses: Vec<usize> = occurrence_map.get(&neg_lit).cloned().unwrap_or_default();

        if pos_clauses.is_empty() || neg_clauses.is_empty() {
            // Variable appears only in one polarity, just remove those clauses
            self.remove_clauses_with_var(var, clauses, occurrence_map);
            return true;
        }

        // Generate all resolvents
        let mut resolvents = Vec::new();

        for &pos_idx in &pos_clauses {
            for &neg_idx in &neg_clauses {
                if let Some(resolvent) = self.resolve(&clauses[pos_idx], &clauses[neg_idx], var) {
                    // Check if resolvent is a tautology
                    if !self.is_tautology(&resolvent) {
                        resolvents.push(resolvent);
                    }
                    self.stats.resolutions += 1;
                }
            }
        }

        // Remove original clauses
        let to_remove: HashSet<usize> = pos_clauses.into_iter().chain(neg_clauses).collect();

        self.remove_clauses_by_indices(clauses, &to_remove, occurrence_map);

        // Add resolvents
        for resolvent in resolvents {
            self.add_clause(resolvent, clauses, occurrence_map);
        }

        true
    }

    /// Resolve two clauses on a variable.
    fn resolve(&self, c1: &Clause, c2: &Clause, var: Var) -> Option<Clause> {
        let pos_lit = Lit::pos(var);
        let neg_lit = Lit::neg(var);

        // Check that c1 contains pos_lit and c2 contains neg_lit (or vice versa)
        let (has_pos_1, has_neg_1) = (c1.lits.contains(&pos_lit), c1.lits.contains(&neg_lit));
        let (has_pos_2, has_neg_2) = (c2.lits.contains(&pos_lit), c2.lits.contains(&neg_lit));

        if !((has_pos_1 && has_neg_2) || (has_neg_1 && has_pos_2)) {
            return None;
        }

        // Merge clauses, removing the pivot literals
        let mut resolvent_lits: Vec<Lit> = c1
            .lits
            .iter()
            .chain(c2.lits.iter())
            .copied()
            .filter(|&lit| lit.var() != var)
            .collect();

        // Remove duplicates
        resolvent_lits.sort_unstable_by_key(|lit| lit.code());
        resolvent_lits.dedup();

        Some(Clause::new(resolvent_lits, false))
    }

    /// Check if a clause is a tautology (contains both p and ¬p).
    fn is_tautology(&self, clause: &Clause) -> bool {
        let lit_set: HashSet<Lit> = clause.lits.iter().copied().collect();

        for &lit in &clause.lits {
            if lit_set.contains(&!lit) {
                return true;
            }
        }

        false
    }

    /// Remove clauses containing a variable.
    fn remove_clauses_with_var(
        &self,
        var: Var,
        clauses: &mut Vec<Clause>,
        occurrence_map: &mut HashMap<Lit, Vec<usize>>,
    ) {
        let pos_lit = Lit::pos(var);
        let neg_lit = Lit::neg(var);

        let to_remove: HashSet<usize> = occurrence_map
            .get(&pos_lit)
            .into_iter()
            .chain(occurrence_map.get(&neg_lit))
            .flatten()
            .copied()
            .collect();

        self.remove_clauses_by_indices(clauses, &to_remove, occurrence_map);
    }

    /// Remove clauses by their indices.
    fn remove_clauses_by_indices(
        &self,
        clauses: &mut Vec<Clause>,
        to_remove: &HashSet<usize>,
        occurrence_map: &mut HashMap<Lit, Vec<usize>>,
    ) {
        // Mark clauses for removal
        let mut new_clauses = Vec::new();
        let mut old_to_new = HashMap::new();

        for (old_idx, clause) in clauses.iter().enumerate() {
            if !to_remove.contains(&old_idx) {
                let new_idx = new_clauses.len();
                old_to_new.insert(old_idx, new_idx);
                new_clauses.push(clause.clone());
            }
        }

        *clauses = new_clauses;

        // Update occurrence map
        for (_, indices) in occurrence_map.iter_mut() {
            *indices = indices
                .iter()
                .filter_map(|&old_idx| old_to_new.get(&old_idx).copied())
                .collect();
        }

        // Remove empty entries
        occurrence_map.retain(|_, v| !v.is_empty());
    }

    /// Add a clause to the database.
    fn add_clause(
        &self,
        clause: Clause,
        clauses: &mut Vec<Clause>,
        occurrence_map: &mut HashMap<Lit, Vec<usize>>,
    ) {
        let idx = clauses.len();
        clauses.push(clause.clone());

        for &lit in &clause.lits {
            occurrence_map.entry(lit).or_default().push(idx);
        }
    }

    /// Get elimination statistics.
    pub fn stats(&self) -> &EliminationStats {
        &self.stats
    }
}

/// Asymmetric variable elimination.
pub struct AsymmetricEliminator {
    config: EliminationConfig,
}

impl AsymmetricEliminator {
    /// Create a new asymmetric eliminator.
    pub fn new(config: EliminationConfig) -> Self {
        Self { config }
    }

    /// Perform asymmetric elimination.
    ///
    /// Removes literals that don't contribute to satisfiability.
    pub fn eliminate(&self, clauses: &mut Vec<Clause>) -> usize {
        let mut eliminated = 0;

        for clause in clauses.iter_mut() {
            eliminated += self.eliminate_asymmetric_literals(clause);
        }

        eliminated
    }

    /// Eliminate asymmetric literals from a clause.
    fn eliminate_asymmetric_literals(&self, clause: &mut Clause) -> usize {
        let original_len = clause.lits.len();

        // Try removing each literal and check if clause is still satisfiable
        let mut i = 0;
        while i < clause.lits.len() {
            let _lit = clause.lits[i];

            // Create clause without this literal
            let mut test_clause = clause.clone();
            test_clause.lits.remove(i);

            // Check if removal makes clause unit or empty
            if test_clause.lits.len() <= 1 {
                i += 1;
                continue;
            }

            // Simplified check: if remaining clause subsumes original, remove literal
            if self.subsumes(&test_clause, clause) {
                clause.lits.remove(i);
            } else {
                i += 1;
            }
        }

        original_len - clause.lits.len()
    }

    /// Check if clause c1 subsumes c2.
    fn subsumes(&self, c1: &Clause, c2: &Clause) -> bool {
        let c1_set: HashSet<Lit> = c1.lits.iter().copied().collect();

        c2.lits.iter().all(|lit| c1_set.contains(lit))
    }
}

/// Bounded variable addition for preprocessing.
///
/// Introduces new variables to simplify formula structure.
pub struct BoundedVariableAddition {
    next_var: Var,
}

impl BoundedVariableAddition {
    /// Create a new bounded variable addition engine.
    pub fn new(max_var: Var) -> Self {
        Self {
            next_var: Var(max_var.0 + 1),
        }
    }

    /// Add bounded variables to simplify clauses.
    pub fn add_variables(&mut self, clauses: &mut Vec<Clause>) -> Vec<Var> {
        let mut added = Vec::new();

        // Find long clauses that could benefit from splitting
        let long_clauses: Vec<usize> = clauses
            .iter()
            .enumerate()
            .filter(|(_, c)| c.lits.len() > 8)
            .map(|(i, _)| i)
            .collect();

        for idx in long_clauses {
            if let Some(new_var) = self.split_clause(idx, clauses) {
                added.push(new_var);
            }
        }

        added
    }

    /// Split a long clause by introducing a new variable.
    fn split_clause(&mut self, idx: usize, clauses: &mut Vec<Clause>) -> Option<Var> {
        let clause = clauses.get(idx)?;

        if clause.lits.len() <= 4 {
            return None;
        }

        let new_var = self.next_var;
        self.next_var = Var(self.next_var.0 + 1);

        let mid = clause.lits.len() / 2;

        // Split: (l1 ∨ l2 ∨ ... ∨ lₙ) becomes:
        // (l1 ∨ ... ∨ lₘ ∨ x) ∧ (¬x ∨ lₘ₊₁ ∨ ... ∨ lₙ)

        let mut first_half = clause.lits[..mid].to_vec();
        first_half.push(Lit::pos(new_var));

        let mut second_half = clause.lits[mid..].to_vec();
        second_half.push(Lit::neg(new_var));

        clauses[idx] = Clause::new(first_half, false);
        clauses.push(Clause::new(second_half, false));

        Some(new_var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_eliminator_creation() {
        let config = EliminationConfig::default();
        let eliminator = VariableEliminator::new(config);

        assert_eq!(eliminator.stats.vars_eliminated, 0);
    }

    #[test]
    fn test_elimination_cost() {
        let config = EliminationConfig::default();
        let eliminator = VariableEliminator::new(config);

        let clauses = vec![
            Clause::new(vec![Lit::pos(Var(0))], false),
            Clause::new(vec![Lit::neg(Var(0))], false),
        ];

        let occurrence_map = eliminator.build_occurrence_map(&clauses);
        let cost = eliminator.elimination_cost(Var(0), &clauses, &occurrence_map);

        // Cost = (1 × 1) - 2 = -1
        assert_eq!(cost, -1.0);
    }

    #[test]
    fn test_resolution() {
        let config = EliminationConfig::default();
        let eliminator = VariableEliminator::new(config);

        let c1 = Clause::new(vec![Lit::pos(Var(0)), Lit::pos(Var(1))], false);
        let c2 = Clause::new(vec![Lit::neg(Var(0)), Lit::pos(Var(2))], false);

        let resolvent = eliminator
            .resolve(&c1, &c2, Var(0))
            .expect("resolution failed");

        // Should get (v1 ∨ v2)
        assert_eq!(resolvent.lits.len(), 2);
    }

    #[test]
    fn test_is_tautology() {
        let config = EliminationConfig::default();
        let eliminator = VariableEliminator::new(config);

        let taut = Clause::new(vec![Lit::pos(Var(0)), Lit::neg(Var(0))], false);

        assert!(eliminator.is_tautology(&taut));

        let non_taut = Clause::new(vec![Lit::pos(Var(0)), Lit::pos(Var(1))], false);

        assert!(!eliminator.is_tautology(&non_taut));
    }
}
