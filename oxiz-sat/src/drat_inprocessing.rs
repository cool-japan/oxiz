//! DRAT-based Inprocessing
//!
//! Provides inprocessing (simplification during search) while maintaining
//! DRAT proof correctness. All simplification steps are logged to ensure
//! the final proof remains valid.
//!
//! Key features:
//! - Proof-producing clause elimination
//! - Proof-producing subsumption
//! - Proof-producing variable elimination
//! - Proof-producing blocked clause elimination
//! - Ensures all simplifications are verifiable via DRAT

use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{Lit, Var};
use crate::proof::DratProof;
use std::collections::{HashMap, HashSet};

/// Statistics for DRAT-based inprocessing
#[derive(Debug, Default, Clone)]
pub struct DratInprocessingStats {
    /// Number of clauses eliminated
    pub clauses_eliminated: usize,
    /// Number of variables eliminated
    pub variables_eliminated: usize,
    /// Number of subsumptions performed
    pub subsumptions: usize,
    /// Number of blocked clauses eliminated
    pub blocked_clauses_eliminated: usize,
    /// Number of proof steps logged
    pub proof_steps: usize,
    /// Number of inprocessing rounds
    pub rounds: usize,
}

impl DratInprocessingStats {
    /// Display statistics in a human-readable format
    pub fn display(&self) -> String {
        format!(
            "DRAT Inprocessing Stats:\n\
             - Rounds: {}\n\
             - Clauses eliminated: {}\n\
             - Variables eliminated: {}\n\
             - Subsumptions: {}\n\
             - Blocked clauses: {}\n\
             - Proof steps: {}",
            self.rounds,
            self.clauses_eliminated,
            self.variables_eliminated,
            self.subsumptions,
            self.blocked_clauses_eliminated,
            self.proof_steps,
        )
    }
}

/// Configuration for DRAT-based inprocessing
#[derive(Debug, Clone)]
pub struct DratInprocessingConfig {
    /// Enable subsumption elimination
    pub enable_subsumption: bool,
    /// Enable variable elimination
    pub enable_variable_elimination: bool,
    /// Enable blocked clause elimination
    pub enable_blocked_clause_elimination: bool,
    /// Maximum clause size for variable elimination
    pub max_clause_size_for_elimination: usize,
    /// Maximum resolution size (product of clause sizes)
    pub max_resolution_size: usize,
}

impl Default for DratInprocessingConfig {
    fn default() -> Self {
        Self {
            enable_subsumption: true,
            enable_variable_elimination: true,
            enable_blocked_clause_elimination: true,
            max_clause_size_for_elimination: 10,
            max_resolution_size: 100,
        }
    }
}

/// DRAT-based Inprocessor
///
/// Performs simplifications during search while maintaining proof correctness
pub struct DratInprocessor {
    /// Configuration
    config: DratInprocessingConfig,
    /// Statistics
    stats: DratInprocessingStats,
    /// Clauses that have been eliminated
    eliminated_clauses: HashSet<ClauseId>,
    /// Variables that have been eliminated
    eliminated_vars: HashSet<Var>,
}

impl DratInprocessor {
    /// Create a new DRAT inprocessor
    pub fn new(config: DratInprocessingConfig) -> Self {
        Self {
            config,
            stats: DratInprocessingStats::default(),
            eliminated_clauses: HashSet::new(),
            eliminated_vars: HashSet::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(DratInprocessingConfig::default())
    }

    /// Perform inprocessing on the clause database
    ///
    /// Returns the number of simplifications made
    pub fn inprocess(
        &mut self,
        db: &mut ClauseDatabase,
        proof: &mut DratProof,
    ) -> std::io::Result<usize> {
        self.stats.rounds += 1;
        let mut simplifications = 0;

        // Subsumption elimination
        if self.config.enable_subsumption {
            simplifications += self.eliminate_subsumed(db, proof)?;
        }

        // Blocked clause elimination
        if self.config.enable_blocked_clause_elimination {
            simplifications += self.eliminate_blocked_clauses(db, proof)?;
        }

        // Variable elimination
        if self.config.enable_variable_elimination {
            simplifications += self.eliminate_variables(db, proof)?;
        }

        Ok(simplifications)
    }

    /// Eliminate subsumed clauses with DRAT logging
    fn eliminate_subsumed(
        &mut self,
        db: &mut ClauseDatabase,
        proof: &mut DratProof,
    ) -> std::io::Result<usize> {
        let mut eliminated = 0;

        // Collect all clause IDs to check
        let clause_ids: Vec<ClauseId> = db.iter_ids().collect();

        for i in 0..clause_ids.len() {
            let cid1 = clause_ids[i];
            if self.eliminated_clauses.contains(&cid1) {
                continue;
            }

            let clause1 = match db.get(cid1) {
                Some(c) => c.lits.to_vec(),
                None => continue,
            };

            for &cid2 in clause_ids.iter().skip(i + 1) {
                if self.eliminated_clauses.contains(&cid2) {
                    continue;
                }

                let clause2 = match db.get(cid2) {
                    Some(c) => c.lits.to_vec(),
                    None => continue,
                };

                // Check if clause1 subsumes clause2
                if Self::subsumes(&clause1, &clause2) {
                    // Log deletion to proof
                    proof.delete_clause(&clause2)?;
                    self.stats.proof_steps += 1;

                    // Mark clause2 as eliminated
                    self.eliminated_clauses.insert(cid2);
                    self.stats.subsumptions += 1;
                    eliminated += 1;
                } else if Self::subsumes(&clause2, &clause1) {
                    // Log deletion to proof
                    proof.delete_clause(&clause1)?;
                    self.stats.proof_steps += 1;

                    // Mark clause1 as eliminated
                    self.eliminated_clauses.insert(cid1);
                    self.stats.subsumptions += 1;
                    eliminated += 1;
                    break; // clause1 is eliminated, move to next i
                }
            }
        }

        self.stats.clauses_eliminated += eliminated;
        Ok(eliminated)
    }

    /// Check if clause1 subsumes clause2
    fn subsumes(clause1: &[Lit], clause2: &[Lit]) -> bool {
        if clause1.len() > clause2.len() {
            return false;
        }

        // Every literal in clause1 must be in clause2
        clause1.iter().all(|lit1| clause2.contains(lit1))
    }

    /// Eliminate blocked clauses with DRAT logging
    fn eliminate_blocked_clauses(
        &mut self,
        db: &mut ClauseDatabase,
        proof: &mut DratProof,
    ) -> std::io::Result<usize> {
        let mut eliminated = 0;

        let clause_ids: Vec<ClauseId> = db.iter_ids().collect();

        for cid in clause_ids {
            if self.eliminated_clauses.contains(&cid) {
                continue;
            }

            let clause = match db.get(cid) {
                Some(c) => c.lits.to_vec(),
                None => continue,
            };

            // Check if this clause is blocked on any of its literals
            for &lit in &clause {
                if self.is_blocked(db, &clause, lit) {
                    // Log deletion to proof
                    proof.delete_clause(&clause)?;
                    self.stats.proof_steps += 1;

                    // Mark as eliminated
                    self.eliminated_clauses.insert(cid);
                    self.stats.blocked_clauses_eliminated += 1;
                    eliminated += 1;
                    break;
                }
            }
        }

        self.stats.clauses_eliminated += eliminated;
        Ok(eliminated)
    }

    /// Check if a clause is blocked on a literal
    ///
    /// A clause C is blocked on literal l if for every clause D containing ~l,
    /// the resolvent of C and D is a tautology
    fn is_blocked(&self, db: &ClauseDatabase, clause: &[Lit], lit: Lit) -> bool {
        let neg_lit = lit.negate();

        // Find all clauses containing ~lit
        for other_cid in db.iter_ids() {
            if self.eliminated_clauses.contains(&other_cid) {
                continue;
            }

            let other_clause = match db.get(other_cid) {
                Some(c) => &c.lits,
                None => continue,
            };

            // Check if other_clause contains ~lit
            if !other_clause.contains(&neg_lit) {
                continue;
            }

            // Compute resolvent and check if it's a tautology
            let resolvent = Self::resolve(clause, other_clause, lit.var());
            if !Self::is_tautology(&resolvent) {
                return false; // Found a non-tautological resolvent
            }
        }

        true // All resolvents are tautologies
    }

    /// Resolve two clauses on a variable
    fn resolve(clause1: &[Lit], clause2: &[Lit], var: Var) -> Vec<Lit> {
        let mut resolvent = Vec::new();

        // Add all literals from clause1 except those involving var
        for &lit in clause1 {
            if lit.var() != var {
                resolvent.push(lit);
            }
        }

        // Add all literals from clause2 except those involving var
        for &lit in clause2 {
            if lit.var() != var && !resolvent.contains(&lit) {
                resolvent.push(lit);
            }
        }

        resolvent
    }

    /// Check if a clause is a tautology (contains both l and ~l)
    fn is_tautology(clause: &[Lit]) -> bool {
        for &lit in clause {
            if clause.contains(&lit.negate()) {
                return true;
            }
        }
        false
    }

    /// Eliminate variables with DRAT logging
    fn eliminate_variables(
        &mut self,
        db: &mut ClauseDatabase,
        proof: &mut DratProof,
    ) -> std::io::Result<usize> {
        let mut eliminated = 0;

        // Find candidate variables for elimination
        let vars = self.find_elimination_candidates(db);

        for var in vars {
            if self.can_eliminate_variable(db, var) {
                eliminated += self.eliminate_variable(db, proof, var)?;
                self.eliminated_vars.insert(var);
                self.stats.variables_eliminated += 1;
            }
        }

        Ok(eliminated)
    }

    /// Find candidate variables for elimination
    fn find_elimination_candidates(&self, db: &ClauseDatabase) -> Vec<Var> {
        let mut var_counts: HashMap<Var, usize> = HashMap::new();

        // Count occurrences of each variable
        for cid in db.iter_ids() {
            if self.eliminated_clauses.contains(&cid) {
                continue;
            }

            let clause = match db.get(cid) {
                Some(c) => &c.lits,
                None => continue,
            };

            for &lit in clause {
                *var_counts.entry(lit.var()).or_insert(0) += 1;
            }
        }

        // Select variables that appear infrequently
        let mut candidates: Vec<_> = var_counts
            .into_iter()
            .filter(|(_, count)| *count <= 5) // Heuristic: low occurrence count
            .map(|(var, _)| var)
            .collect();

        candidates.sort_unstable_by_key(|v| v.0);
        candidates
    }

    /// Check if a variable can be eliminated
    fn can_eliminate_variable(&self, db: &ClauseDatabase, var: Var) -> bool {
        if self.eliminated_vars.contains(&var) {
            return false;
        }

        // Collect clauses containing var positively and negatively
        let (pos_clauses, neg_clauses) = self.collect_clauses_with_var(db, var);

        // Check if elimination would create too many or too large clauses
        let num_resolvents = pos_clauses.len() * neg_clauses.len();
        if num_resolvents > self.config.max_resolution_size {
            return false;
        }

        // Check clause sizes
        for pos_clause in &pos_clauses {
            if pos_clause.len() > self.config.max_clause_size_for_elimination {
                return false;
            }
        }

        for neg_clause in &neg_clauses {
            if neg_clause.len() > self.config.max_clause_size_for_elimination {
                return false;
            }
        }

        true
    }

    /// Collect clauses containing a variable
    fn collect_clauses_with_var(
        &self,
        db: &ClauseDatabase,
        var: Var,
    ) -> (Vec<Vec<Lit>>, Vec<Vec<Lit>>) {
        let mut pos_clauses = Vec::new();
        let mut neg_clauses = Vec::new();

        for cid in db.iter_ids() {
            if self.eliminated_clauses.contains(&cid) {
                continue;
            }

            let clause = match db.get(cid) {
                Some(c) => c.lits.to_vec(),
                None => continue,
            };

            let pos_lit = Lit::pos(var);
            let neg_lit = Lit::neg(var);

            if clause.contains(&pos_lit) {
                pos_clauses.push(clause);
            } else if clause.contains(&neg_lit) {
                neg_clauses.push(clause);
            }
        }

        (pos_clauses, neg_clauses)
    }

    /// Eliminate a variable by resolution
    fn eliminate_variable(
        &mut self,
        db: &mut ClauseDatabase,
        proof: &mut DratProof,
        var: Var,
    ) -> std::io::Result<usize> {
        let (pos_clauses, neg_clauses) = self.collect_clauses_with_var(db, var);

        // Generate all resolvents
        for pos_clause in &pos_clauses {
            for neg_clause in &neg_clauses {
                let resolvent = Self::resolve(pos_clause, neg_clause, var);

                // Skip tautologies
                if Self::is_tautology(&resolvent) {
                    continue;
                }

                // Add resolvent to proof
                proof.add_clause(&resolvent)?;
                self.stats.proof_steps += 1;
            }
        }

        // Delete original clauses from proof
        for clause in pos_clauses.iter().chain(neg_clauses.iter()) {
            proof.delete_clause(clause)?;
            self.stats.proof_steps += 1;
        }

        let eliminated = pos_clauses.len() + neg_clauses.len();
        self.stats.clauses_eliminated += eliminated;

        Ok(eliminated)
    }

    /// Get statistics
    pub fn stats(&self) -> &DratInprocessingStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DratInprocessingStats::default();
    }

    /// Clear eliminated clauses and variables
    pub fn clear(&mut self) {
        self.eliminated_clauses.clear();
        self.eliminated_vars.clear();
        self.stats = DratInprocessingStats::default();
    }
}

impl Default for DratInprocessor {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drat_inprocessor_creation() {
        let inprocessor = DratInprocessor::default();
        assert_eq!(inprocessor.stats().rounds, 0);
    }

    #[test]
    fn test_subsumes() {
        let v0 = Var(0);
        let v1 = Var(1);

        let clause1 = vec![Lit::pos(v0)];
        let clause2 = vec![Lit::pos(v0), Lit::pos(v1)];

        assert!(DratInprocessor::subsumes(&clause1, &clause2));
        assert!(!DratInprocessor::subsumes(&clause2, &clause1));
    }

    #[test]
    fn test_subsumes_equal() {
        let v0 = Var(0);
        let v1 = Var(1);

        let clause1 = vec![Lit::pos(v0), Lit::pos(v1)];
        let clause2 = vec![Lit::pos(v0), Lit::pos(v1)];

        assert!(DratInprocessor::subsumes(&clause1, &clause2));
        assert!(DratInprocessor::subsumes(&clause2, &clause1));
    }

    #[test]
    fn test_resolve() {
        let v0 = Var(0);
        let v1 = Var(1);

        let clause1 = vec![Lit::pos(v0), Lit::pos(v1)];
        let clause2 = vec![Lit::neg(v0)];

        let resolvent = DratInprocessor::resolve(&clause1, &clause2, v0);
        assert_eq!(resolvent.len(), 1);
        assert!(resolvent.contains(&Lit::pos(v1)));
    }

    #[test]
    fn test_is_tautology() {
        let v0 = Var(0);
        let v1 = Var(1);

        let tautology = vec![Lit::pos(v0), Lit::neg(v0), Lit::pos(v1)];
        let non_tautology = vec![Lit::pos(v0), Lit::pos(v1)];

        assert!(DratInprocessor::is_tautology(&tautology));
        assert!(!DratInprocessor::is_tautology(&non_tautology));
    }

    #[test]
    fn test_config_default() {
        let config = DratInprocessingConfig::default();
        assert!(config.enable_subsumption);
        assert!(config.enable_variable_elimination);
        assert!(config.enable_blocked_clause_elimination);
    }

    #[test]
    fn test_stats_display() {
        let stats = DratInprocessingStats {
            clauses_eliminated: 10,
            variables_eliminated: 2,
            subsumptions: 5,
            blocked_clauses_eliminated: 3,
            proof_steps: 20,
            rounds: 1,
        };

        let display = stats.display();
        assert!(display.contains("10"));
        assert!(display.contains("2"));
        assert!(display.contains("5"));
    }

    #[test]
    fn test_clear() {
        let mut inprocessor = DratInprocessor::default();
        inprocessor.stats.rounds = 5;
        inprocessor.eliminated_vars.insert(Var(0));

        inprocessor.clear();

        assert_eq!(inprocessor.stats.rounds, 0);
        assert!(inprocessor.eliminated_vars.is_empty());
    }
}
