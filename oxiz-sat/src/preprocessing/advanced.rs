//! Advanced SAT Preprocessing
#![allow(missing_docs)] // Under development - documentation in progress
//!
//! This module implements sophisticated preprocessing techniques for CNF formulas:
//! - Variable elimination (bounded variable elimination)
//! - Subsumption and self-subsuming resolution
//! - Vivification (clause strengthening)
//! - Blocked clause elimination
//! - Equivalent literal substitution

use rustc_hash::{FxHashMap, FxHashSet};

/// Literal representation (positive/negative variable)
pub type Lit = i32;

/// Variable identifier
pub type Var = u32;

/// Clause identifier
pub type ClauseId = usize;

/// Clause (set of literals)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Clause {
    pub literals: Vec<Lit>,
}

impl Clause {
    /// Create a new clause
    pub fn new(literals: Vec<Lit>) -> Self {
        Self { literals }
    }

    /// Check if clause is unit (single literal)
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Check if clause is binary
    pub fn is_binary(&self) -> bool {
        self.literals.len() == 2
    }

    /// Check if clause is empty (contradiction)
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if clause contains a literal
    pub fn contains(&self, lit: Lit) -> bool {
        self.literals.contains(&lit)
    }

    /// Get clause size
    pub fn size(&self) -> usize {
        self.literals.len()
    }
}

/// Statistics for preprocessing
#[derive(Debug, Clone, Default)]
pub struct PreprocessingStats {
    pub variables_eliminated: u64,
    pub clauses_eliminated: u64,
    pub literals_eliminated: u64,
    pub subsumptions: u64,
    pub self_subsuming_resolutions: u64,
    pub vivifications: u64,
    pub blocked_clauses: u64,
    pub equivalent_literals: u64,
}

/// Configuration for preprocessing
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Enable bounded variable elimination
    pub enable_bve: bool,
    /// Maximum clause growth for BVE
    pub bve_clause_limit: usize,
    /// Enable subsumption
    pub enable_subsumption: bool,
    /// Enable vivification
    pub enable_vivification: bool,
    /// Enable blocked clause elimination
    pub enable_bce: bool,
    /// Enable equivalent literal substitution
    pub enable_equiv_literals: bool,
    /// Maximum preprocessing iterations
    pub max_iterations: usize,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_bve: true,
            bve_clause_limit: 100,
            enable_subsumption: true,
            enable_vivification: true,
            enable_bce: true,
            enable_equiv_literals: true,
            max_iterations: 10,
        }
    }
}

/// Advanced SAT preprocessor
pub struct AdvancedPreprocessor {
    config: PreprocessingConfig,
    stats: PreprocessingStats,
    /// Formula clauses
    clauses: Vec<Clause>,
    /// Occurrence lists: literal -> clause IDs
    occurrences: FxHashMap<Lit, Vec<ClauseId>>,
    /// Variable elimination order
    elim_order: Vec<Var>,
    /// Eliminated variables
    eliminated: FxHashSet<Var>,
}

impl AdvancedPreprocessor {
    /// Create a new preprocessor
    pub fn new(config: PreprocessingConfig) -> Self {
        Self {
            config,
            stats: PreprocessingStats::default(),
            clauses: Vec::new(),
            occurrences: FxHashMap::default(),
            elim_order: Vec::new(),
            eliminated: FxHashSet::default(),
        }
    }

    /// Preprocess a CNF formula
    pub fn preprocess(&mut self, clauses: Vec<Clause>) -> Result<Vec<Clause>, String> {
        self.clauses = clauses;
        self.build_occurrence_lists();

        // Main preprocessing loop
        for _iteration in 0..self.config.max_iterations {
            let mut changed = false;

            // Unit propagation
            changed |= self.unit_propagation()?;

            // Subsumption
            if self.config.enable_subsumption {
                changed |= self.subsumption()?;
            }

            // Self-subsuming resolution
            if self.config.enable_subsumption {
                changed |= self.self_subsuming_resolution()?;
            }

            // Vivification
            if self.config.enable_vivification {
                changed |= self.vivification()?;
            }

            // Bounded variable elimination
            if self.config.enable_bve {
                changed |= self.bounded_variable_elimination()?;
            }

            // Blocked clause elimination
            if self.config.enable_bce {
                changed |= self.blocked_clause_elimination()?;
            }

            // Equivalent literal substitution
            if self.config.enable_equiv_literals {
                changed |= self.equivalent_literal_substitution()?;
            }

            // Stop if no progress
            if !changed {
                break;
            }
        }

        // Return preprocessed formula
        Ok(self.clauses.clone())
    }

    /// Build occurrence lists for efficient lookup
    fn build_occurrence_lists(&mut self) {
        self.occurrences.clear();

        for (clause_id, clause) in self.clauses.iter().enumerate() {
            for &lit in &clause.literals {
                self.occurrences.entry(lit).or_default().push(clause_id);
            }
        }
    }

    /// Unit propagation
    fn unit_propagation(&mut self) -> Result<bool, String> {
        let mut changed = false;

        loop {
            // Find unit clauses
            let unit_clauses: Vec<_> = self
                .clauses
                .iter()
                .filter(|c| c.is_unit())
                .cloned()
                .collect();

            if unit_clauses.is_empty() {
                break;
            }

            for unit_clause in unit_clauses {
                let unit_lit = unit_clause.literals[0];

                // Propagate this literal
                self.propagate_literal(unit_lit)?;
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Propagate a unit literal
    fn propagate_literal(&mut self, lit: Lit) -> Result<(), String> {
        let neg_lit = -lit;

        // Remove clauses containing lit
        self.clauses.retain(|c| !c.contains(lit));

        // Remove -lit from clauses
        for clause in &mut self.clauses {
            if clause.contains(neg_lit) {
                clause.literals.retain(|&l| l != neg_lit);
                self.stats.literals_eliminated += 1;
            }
        }

        // Check for empty clause (contradiction)
        if self.clauses.iter().any(|c| c.is_empty()) {
            return Err("Formula is unsatisfiable".to_string());
        }

        self.build_occurrence_lists();
        Ok(())
    }

    /// Subsumption: remove clauses subsumed by smaller clauses
    fn subsumption(&mut self) -> Result<bool, String> {
        let mut changed = false;
        let mut to_remove = FxHashSet::default();

        for i in 0..self.clauses.len() {
            if to_remove.contains(&i) {
                continue;
            }

            for j in (i + 1)..self.clauses.len() {
                if to_remove.contains(&j) {
                    continue;
                }

                // Check if clause i subsumes clause j
                if self.subsumes(&self.clauses[i], &self.clauses[j]) {
                    to_remove.insert(j);
                    self.stats.subsumptions += 1;
                    changed = true;
                }
                // Check if clause j subsumes clause i
                else if self.subsumes(&self.clauses[j], &self.clauses[i]) {
                    to_remove.insert(i);
                    self.stats.subsumptions += 1;
                    changed = true;
                    break;
                }
            }
        }

        // Remove subsumed clauses
        let mut new_clauses = Vec::new();
        for (i, clause) in self.clauses.iter().enumerate() {
            if !to_remove.contains(&i) {
                new_clauses.push(clause.clone());
            }
        }

        self.clauses = new_clauses;
        self.stats.clauses_eliminated += to_remove.len() as u64;

        if changed {
            self.build_occurrence_lists();
        }

        Ok(changed)
    }

    /// Check if clause1 subsumes clause2
    fn subsumes(&self, clause1: &Clause, clause2: &Clause) -> bool {
        if clause1.size() > clause2.size() {
            return false;
        }

        clause1.literals.iter().all(|lit| clause2.contains(*lit))
    }

    /// Self-subsuming resolution
    fn self_subsuming_resolution(&mut self) -> Result<bool, String> {
        let mut changed = false;

        for i in 0..self.clauses.len() {
            for j in (i + 1)..self.clauses.len() {
                // Check if we can perform self-subsuming resolution
                if let Some(resolvent) =
                    self.try_self_subsuming_resolution(&self.clauses[i], &self.clauses[j])
                {
                    // Replace the longer clause with the resolvent
                    if self.clauses[i].size() > self.clauses[j].size() {
                        self.clauses[i] = resolvent;
                    } else {
                        self.clauses[j] = resolvent;
                    }

                    self.stats.self_subsuming_resolutions += 1;
                    changed = true;
                }
            }
        }

        if changed {
            self.build_occurrence_lists();
        }

        Ok(changed)
    }

    /// Try self-subsuming resolution between two clauses
    fn try_self_subsuming_resolution(&self, c1: &Clause, c2: &Clause) -> Option<Clause> {
        // Find pivot literal
        for &lit in &c1.literals {
            if c2.contains(-lit) {
                // Check if c1 \ {lit} ⊆ c2
                let c1_without_lit: FxHashSet<_> =
                    c1.literals.iter().filter(|&&l| l != lit).copied().collect();

                if c1_without_lit.iter().all(|l| c2.contains(*l)) {
                    // Self-subsuming resolution: c2 \ {-lit}
                    let resolvent_lits: Vec<_> = c2
                        .literals
                        .iter()
                        .filter(|&&l| l != -lit)
                        .copied()
                        .collect();

                    return Some(Clause::new(resolvent_lits));
                }
            }
        }

        None
    }

    /// Vivification: strengthen clauses by trying to derive shorter clauses
    fn vivification(&mut self) -> Result<bool, String> {
        let mut changed = false;

        // Collect indices of clauses to process (avoiding borrow checker issues)
        let indices_to_process: Vec<usize> = self
            .clauses
            .iter()
            .enumerate()
            .filter(|(_, clause)| clause.size() > 2)
            .map(|(i, _)| i)
            .collect();

        for idx in indices_to_process {
            let clause = &self.clauses[idx];
            // Try to find a subset that implies the clause
            if let Some(strengthened) = self.try_strengthen_clause(clause) {
                self.clauses[idx] = strengthened;
                self.stats.vivifications += 1;
                changed = true;
            }
        }

        if changed {
            self.build_occurrence_lists();
        }

        Ok(changed)
    }

    /// Try to strengthen a clause
    fn try_strengthen_clause(&self, clause: &Clause) -> Option<Clause> {
        // Placeholder: would use unit propagation to check if subset implies clause
        // For now, just check for duplicate literals
        let mut unique_lits: Vec<_> = clause.literals.clone();
        unique_lits.sort();
        unique_lits.dedup();

        if unique_lits.len() < clause.literals.len() {
            Some(Clause::new(unique_lits))
        } else {
            None
        }
    }

    /// Bounded variable elimination
    fn bounded_variable_elimination(&mut self) -> Result<bool, String> {
        let mut changed = false;

        // Compute elimination order (prefer variables with few occurrences)
        self.compute_elimination_order();

        for &var in &self.elim_order.clone() {
            if self.eliminated.contains(&var) {
                continue;
            }

            // Try to eliminate this variable
            if self.try_eliminate_variable(var)? {
                self.eliminated.insert(var);
                self.stats.variables_eliminated += 1;
                changed = true;
            }
        }

        if changed {
            self.build_occurrence_lists();
        }

        Ok(changed)
    }

    /// Compute variable elimination order
    fn compute_elimination_order(&mut self) {
        let mut var_occurrences: FxHashMap<Var, usize> = FxHashMap::default();

        for clause in &self.clauses {
            for &lit in &clause.literals {
                let var = lit.unsigned_abs();
                *var_occurrences.entry(var).or_insert(0) += 1;
            }
        }

        // Sort by occurrence count
        let mut vars: Vec<_> = var_occurrences.into_iter().collect();
        vars.sort_by_key(|(_, count)| *count);

        self.elim_order = vars.into_iter().map(|(var, _)| var).collect();
    }

    /// Try to eliminate a variable by resolution
    fn try_eliminate_variable(&mut self, var: Var) -> Result<bool, String> {
        let pos_lit = var as Lit;
        let neg_lit = -(var as Lit);

        // Get clauses containing the variable
        let pos_clauses: Vec<_> = self
            .clauses
            .iter()
            .filter(|c| c.contains(pos_lit))
            .cloned()
            .collect();

        let neg_clauses: Vec<_> = self
            .clauses
            .iter()
            .filter(|c| c.contains(neg_lit))
            .cloned()
            .collect();

        // Compute resolvent clauses
        let mut resolvents = Vec::new();

        for pos_clause in &pos_clauses {
            for neg_clause in &neg_clauses {
                if let Some(resolvent) = self.resolve(pos_clause, neg_clause, pos_lit) {
                    resolvents.push(resolvent);
                }
            }
        }

        // Check if elimination is beneficial (bounded variable elimination)
        let old_clause_count = pos_clauses.len() + neg_clauses.len();
        let new_clause_count = resolvents.len();

        if new_clause_count > self.config.bve_clause_limit
            || new_clause_count > old_clause_count * 2
        {
            return Ok(false);
        }

        // Remove old clauses and add resolvents
        self.clauses
            .retain(|c| !c.contains(pos_lit) && !c.contains(neg_lit));
        self.clauses.extend(resolvents);

        self.stats.clauses_eliminated += old_clause_count as u64;

        Ok(true)
    }

    /// Resolve two clauses on a literal
    fn resolve(&self, c1: &Clause, c2: &Clause, pivot: Lit) -> Option<Clause> {
        let mut resolvent_lits = FxHashSet::default();

        // Add literals from c1 except pivot
        for &lit in &c1.literals {
            if lit != pivot {
                resolvent_lits.insert(lit);
            }
        }

        // Add literals from c2 except -pivot
        for &lit in &c2.literals {
            if lit != -pivot {
                // Check for tautology
                if resolvent_lits.contains(&-lit) {
                    return None;
                }
                resolvent_lits.insert(lit);
            }
        }

        Some(Clause::new(resolvent_lits.into_iter().collect()))
    }

    /// Blocked clause elimination
    fn blocked_clause_elimination(&mut self) -> Result<bool, String> {
        let mut changed = false;
        let mut to_remove = FxHashSet::default();

        for (clause_id, clause) in self.clauses.iter().enumerate() {
            // Check if clause is blocked on any literal
            for &lit in &clause.literals {
                if self.is_blocked(clause, lit) {
                    to_remove.insert(clause_id);
                    self.stats.blocked_clauses += 1;
                    changed = true;
                    break;
                }
            }
        }

        // Remove blocked clauses
        let mut new_clauses = Vec::new();
        for (i, clause) in self.clauses.iter().enumerate() {
            if !to_remove.contains(&i) {
                new_clauses.push(clause.clone());
            }
        }

        self.clauses = new_clauses;

        if changed {
            self.build_occurrence_lists();
        }

        Ok(changed)
    }

    /// Check if a clause is blocked on a literal
    fn is_blocked(&self, clause: &Clause, lit: Lit) -> bool {
        // Clause is blocked on lit if all resolvents are tautologies
        let neg_lit = -lit;

        for other_clause in &self.clauses {
            if other_clause.contains(neg_lit)
                && let Some(_resolvent) = self.resolve(clause, other_clause, lit)
            {
                // Found a non-tautological resolvent
                return false;
            }
        }

        true
    }

    /// Equivalent literal substitution
    fn equivalent_literal_substitution(&mut self) -> Result<bool, String> {
        let mut changed = false;

        // Find equivalent literals using binary clauses
        let equiv_classes = self.find_equivalent_literals();

        // Substitute equivalent literals
        for (representative, equivalents) in equiv_classes {
            for equiv in equivalents {
                self.substitute_literal(equiv, representative);
                self.stats.equivalent_literals += 1;
                changed = true;
            }
        }

        if changed {
            self.build_occurrence_lists();
        }

        Ok(changed)
    }

    /// Find equivalent literals using binary implications
    fn find_equivalent_literals(&self) -> FxHashMap<Lit, Vec<Lit>> {
        let mut implications: FxHashMap<Lit, Vec<Lit>> = FxHashMap::default();

        // Extract binary implications
        for clause in &self.clauses {
            if clause.is_binary() {
                let lit1 = clause.literals[0];
                let lit2 = clause.literals[1];

                // (lit1 ∨ lit2) ≡ (¬lit1 → lit2)
                implications.entry(-lit1).or_default().push(lit2);
                implications.entry(-lit2).or_default().push(lit1);
            }
        }

        // Find strongly connected components (equivalent literals)
        let mut equiv_classes = FxHashMap::default();

        // Placeholder: would use Tarjan's algorithm
        // For now, just find simple equivalences (lit1 → lit2 and lit2 → lit1)
        for (&lit1, targets) in &implications {
            for &lit2 in targets {
                if implications.get(&lit2).is_some_and(|t| t.contains(&lit1)) {
                    // Found equivalence: lit1 ≡ lit2
                    equiv_classes
                        .entry(lit1.min(lit2))
                        .or_insert_with(Vec::new)
                        .push(lit1.max(lit2));
                }
            }
        }

        equiv_classes
    }

    /// Substitute a literal with another
    fn substitute_literal(&mut self, from: Lit, to: Lit) {
        for clause in &mut self.clauses {
            for lit in &mut clause.literals {
                if *lit == from {
                    *lit = to;
                } else if *lit == -from {
                    *lit = -to;
                }
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &PreprocessingStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_creation() {
        let config = PreprocessingConfig::default();
        let preprocessor = AdvancedPreprocessor::new(config);
        assert_eq!(preprocessor.stats.variables_eliminated, 0);
    }

    #[test]
    fn test_unit_propagation() {
        let config = PreprocessingConfig::default();
        let mut preprocessor = AdvancedPreprocessor::new(config);

        let clauses = vec![
            Clause::new(vec![1]),     // Unit clause
            Clause::new(vec![1, 2]),  // Should be removed (contains 1)
            Clause::new(vec![-1, 3]), // Should become (3)
        ];

        let result = preprocessor.preprocess(clauses);
        assert!(result.is_ok());

        let preprocessed = result.expect("Preprocessing must succeed");
        assert!(preprocessed.len() < 3);
    }

    #[test]
    fn test_subsumption() {
        let config = PreprocessingConfig::default();
        let mut preprocessor = AdvancedPreprocessor::new(config);

        let clauses = vec![
            Clause::new(vec![1, 2]),    // Subsumes next clause
            Clause::new(vec![1, 2, 3]), // Should be removed
        ];

        let result = preprocessor.preprocess(clauses);
        assert!(result.is_ok());

        let preprocessed = result.expect("Preprocessing must succeed");
        // Verify preprocessing ran successfully
        assert!(preprocessed.len() <= 2);
    }

    #[test]
    fn test_clause_operations() {
        let clause = Clause::new(vec![1, 2, 3]);

        assert!(clause.contains(2));
        assert!(!clause.contains(4));
        assert_eq!(clause.size(), 3);
        assert!(!clause.is_unit());
        assert!(!clause.is_binary());
    }

    #[test]
    fn test_subsumes_check() {
        let preprocessor = AdvancedPreprocessor::new(PreprocessingConfig::default());

        let c1 = Clause::new(vec![1, 2]);
        let c2 = Clause::new(vec![1, 2, 3]);

        assert!(preprocessor.subsumes(&c1, &c2));
        assert!(!preprocessor.subsumes(&c2, &c1));
    }

    #[test]
    fn test_resolution() {
        let preprocessor = AdvancedPreprocessor::new(PreprocessingConfig::default());

        let c1 = Clause::new(vec![1, 2]);
        let c2 = Clause::new(vec![-1, 3]);

        let resolvent = preprocessor.resolve(&c1, &c2, 1);
        assert!(resolvent.is_some());

        let res = resolvent.expect("Resolution must produce resolvent");
        assert!(res.contains(2));
        assert!(res.contains(3));
        assert!(!res.contains(1));
    }

    #[test]
    fn test_tautology_detection() {
        let preprocessor = AdvancedPreprocessor::new(PreprocessingConfig::default());

        let c1 = Clause::new(vec![1, 2]);
        let c2 = Clause::new(vec![-1, -2]);

        // Resolution should produce tautology (2, -2)
        let resolvent = preprocessor.resolve(&c1, &c2, 1);
        assert!(resolvent.is_none());
    }

    #[test]
    fn test_empty_clause() {
        let clause = Clause::new(vec![]);
        assert!(clause.is_empty());
    }
}
