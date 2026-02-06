//! Theory Conflict Resolution for Combination.
//!
//! This module handles conflicts that arise from theory combination:
//! - Multi-theory conflict analysis
//! - Minimal explanation generation
//! - Conflict minimization
//! - Theory blame assignment
//! - Conflict-driven clause learning (CDCL) for theory combination
//!
//! ## Theory Conflicts
//!
//! In theory combination, conflicts can arise from:
//! - A single theory detecting inconsistency
//! - Incompatible propagations from multiple theories
//! - Violation of shared term constraints
//!
//! ## Conflict Analysis
//!
//! When a conflict occurs, we perform analysis to:
//! - Identify the root cause
//! - Generate a minimal explanation
//! - Learn conflict clauses to prevent similar conflicts
//! - Determine the backtrack level
//!
//! ## References
//!
//! - Silva & Sakallah (1996): "GRASP: A Search Algorithm for Propositional Satisfiability"
//! - Nieuwenhuis, Oliveras, Tinelli (2006): "Solving SAT and SAT Modulo Theories"
//! - Z3's `smt/theory_combination.cpp`, `smt/smt_conflict.cpp`

use rustc_hash::{FxHashMap, FxHashSet};

/// Term identifier.
pub type TermId = u32;

/// Theory identifier.
pub type TheoryId = u32;

/// Decision level.
pub type DecisionLevel = u32;

/// Literal (term with polarity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal {
    /// Term.
    pub term: TermId,
    /// Polarity.
    pub polarity: bool,
}

impl Literal {
    /// Create positive literal.
    pub fn positive(term: TermId) -> Self {
        Self {
            term,
            polarity: true,
        }
    }

    /// Create negative literal.
    pub fn negative(term: TermId) -> Self {
        Self {
            term,
            polarity: false,
        }
    }

    /// Negate literal.
    pub fn negate(self) -> Self {
        Self {
            term: self.term,
            polarity: !self.polarity,
        }
    }
}

/// Equality between terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Equality {
    /// Left-hand side.
    pub lhs: TermId,
    /// Right-hand side.
    pub rhs: TermId,
}

impl Equality {
    /// Create new equality.
    pub fn new(lhs: TermId, rhs: TermId) -> Self {
        if lhs <= rhs {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }
}

/// Explanation for a propagation or conflict.
#[derive(Debug, Clone)]
pub enum Explanation {
    /// Given as input.
    Given,

    /// Theory propagation.
    TheoryPropagation {
        /// Source theory.
        theory: TheoryId,
        /// Antecedent literals.
        antecedents: Vec<Literal>,
    },

    /// Equality propagation.
    EqualityPropagation {
        /// Equalities used.
        equalities: Vec<Equality>,
        /// Supporting literals.
        support: Vec<Literal>,
    },

    /// Transitivity.
    Transitivity {
        /// Chain of equalities.
        chain: Vec<Equality>,
    },

    /// Congruence.
    Congruence {
        /// Function applications.
        function: TermId,
        /// Argument equalities.
        arg_equalities: Vec<Equality>,
    },
}

/// Theory conflict.
#[derive(Debug, Clone)]
pub struct TheoryConflict {
    /// Theory that detected the conflict.
    pub theory: TheoryId,

    /// Conflicting literals.
    pub literals: Vec<Literal>,

    /// Explanation for the conflict.
    pub explanation: Explanation,

    /// Decision level where conflict occurred.
    pub level: DecisionLevel,
}

/// Conflict clause learned from analysis.
#[derive(Debug, Clone)]
pub struct ConflictClause {
    /// Literals in the clause.
    pub literals: Vec<Literal>,

    /// UIP (unique implication point) literal.
    pub uip: Option<Literal>,

    /// Backtrack level.
    pub backtrack_level: DecisionLevel,

    /// Theories involved.
    pub theories: FxHashSet<TheoryId>,

    /// Activity score (for clause deletion).
    pub activity: f64,
}

/// Conflict analysis result.
#[derive(Debug, Clone)]
pub struct ConflictAnalysis {
    /// Learned clause.
    pub clause: ConflictClause,

    /// Minimal explanation.
    pub explanation: Explanation,

    /// Theories responsible.
    pub blamed_theories: FxHashSet<TheoryId>,
}

/// Configuration for conflict resolution.
#[derive(Debug, Clone)]
pub struct ConflictResolutionConfig {
    /// Enable conflict minimization.
    pub enable_minimization: bool,

    /// Enable UIP-based learning.
    pub enable_uip: bool,

    /// Minimization algorithm.
    pub minimization_algorithm: MinimizationAlgorithm,

    /// Maximum resolution steps.
    pub max_resolution_steps: usize,

    /// Enable theory blame tracking.
    pub track_theory_blame: bool,

    /// Enable conflict clause learning.
    pub enable_learning: bool,
}

impl Default for ConflictResolutionConfig {
    fn default() -> Self {
        Self {
            enable_minimization: true,
            enable_uip: true,
            minimization_algorithm: MinimizationAlgorithm::Recursive,
            max_resolution_steps: 1000,
            track_theory_blame: true,
            enable_learning: true,
        }
    }
}

/// Minimization algorithm for conflict clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizationAlgorithm {
    /// No minimization.
    None,
    /// Simple minimization (remove redundant literals).
    Simple,
    /// Recursive minimization.
    Recursive,
    /// Binary resolution minimization.
    BinaryResolution,
}

/// Statistics for conflict resolution.
#[derive(Debug, Clone, Default)]
pub struct ConflictResolutionStats {
    /// Total conflicts analyzed.
    pub conflicts_analyzed: u64,
    /// Clauses learned.
    pub clauses_learned: u64,
    /// Literals minimized away.
    pub literals_minimized: u64,
    /// UIP conflicts.
    pub uip_conflicts: u64,
    /// Resolution steps performed.
    pub resolution_steps: u64,
    /// Theory blames assigned.
    pub theory_blames: u64,
}

/// Conflict resolution engine.
pub struct ConflictResolver {
    /// Configuration.
    config: ConflictResolutionConfig,

    /// Statistics.
    stats: ConflictResolutionStats,

    /// Assignment trail.
    trail: Vec<(Literal, DecisionLevel, Explanation)>,

    /// Literal to trail position.
    literal_position: FxHashMap<Literal, usize>,

    /// Decision level boundaries in trail.
    level_boundaries: FxHashMap<DecisionLevel, usize>,

    /// Current decision level.
    current_level: DecisionLevel,

    /// Learned clauses.
    learned_clauses: Vec<ConflictClause>,

    /// Theory blame counters.
    theory_blame: FxHashMap<TheoryId, u64>,
}

impl ConflictResolver {
    /// Create new conflict resolver.
    pub fn new() -> Self {
        Self::with_config(ConflictResolutionConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: ConflictResolutionConfig) -> Self {
        let mut level_boundaries = FxHashMap::default();
        // Initialize level 0 boundary at position 0
        level_boundaries.insert(0, 0);

        Self {
            config,
            stats: ConflictResolutionStats::default(),
            trail: Vec::new(),
            literal_position: FxHashMap::default(),
            level_boundaries,
            current_level: 0,
            learned_clauses: Vec::new(),
            theory_blame: FxHashMap::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ConflictResolutionStats {
        &self.stats
    }

    /// Add assignment to trail.
    pub fn add_assignment(
        &mut self,
        literal: Literal,
        level: DecisionLevel,
        explanation: Explanation,
    ) {
        let position = self.trail.len();
        self.trail.push((literal, level, explanation));
        self.literal_position.insert(literal, position);

        self.level_boundaries.entry(level).or_insert(position);
    }

    /// Push decision level.
    pub fn push_decision_level(&mut self) {
        self.current_level += 1;
    }

    /// Backtrack to decision level.
    pub fn backtrack(&mut self, level: DecisionLevel) -> Result<(), String> {
        if level > self.current_level {
            return Err("Cannot backtrack to future level".to_string());
        }

        // Find position to backtrack to
        let backtrack_pos = self
            .level_boundaries
            .get(&level)
            .copied()
            .unwrap_or(self.trail.len());

        // Remove assignments above this level
        self.trail.truncate(backtrack_pos);

        // Rebuild literal position map
        self.literal_position.clear();
        for (i, &(literal, _, _)) in self.trail.iter().enumerate() {
            self.literal_position.insert(literal, i);
        }

        // Remove level boundaries above this level
        self.level_boundaries.retain(|&l, _| l <= level);

        self.current_level = level;
        Ok(())
    }

    /// Analyze a theory conflict.
    pub fn analyze_conflict(
        &mut self,
        conflict: TheoryConflict,
    ) -> Result<ConflictAnalysis, String> {
        self.stats.conflicts_analyzed += 1;

        if self.config.track_theory_blame {
            *self.theory_blame.entry(conflict.theory).or_insert(0) += 1;
            self.stats.theory_blames += 1;
        }

        // Extract conflict literals
        let mut conflict_literals = conflict.literals.clone();

        // Perform resolution to find UIP if enabled
        if self.config.enable_uip {
            conflict_literals = self.find_uip(&conflict_literals, conflict.level)?;
            self.stats.uip_conflicts += 1;
        }

        // Minimize conflict clause
        if self.config.enable_minimization {
            let before_size = conflict_literals.len();
            conflict_literals = self.minimize_conflict(&conflict_literals)?;
            let after_size = conflict_literals.len();
            self.stats.literals_minimized += (before_size - after_size) as u64;
        }

        // Determine backtrack level
        let backtrack_level = self.compute_backtrack_level(&conflict_literals, conflict.level)?;

        // Build learned clause
        let clause = ConflictClause {
            literals: conflict_literals.clone(),
            uip: self.find_uip_literal(&conflict_literals),
            backtrack_level,
            theories: {
                let mut theories = FxHashSet::default();
                theories.insert(conflict.theory);
                theories
            },
            activity: 1.0,
        };

        // Learn clause if enabled
        if self.config.enable_learning {
            self.learned_clauses.push(clause.clone());
            self.stats.clauses_learned += 1;
        }

        Ok(ConflictAnalysis {
            clause,
            explanation: conflict.explanation,
            blamed_theories: {
                let mut theories = FxHashSet::default();
                theories.insert(conflict.theory);
                theories
            },
        })
    }

    /// Find UIP (Unique Implication Point) using resolution.
    fn find_uip(
        &mut self,
        literals: &[Literal],
        level: DecisionLevel,
    ) -> Result<Vec<Literal>, String> {
        let mut current_clause: FxHashSet<Literal> = literals.iter().copied().collect();
        let mut seen = FxHashSet::default();
        let mut counter = 0;

        // Count literals at current level
        for &lit in &current_clause {
            if self.get_decision_level(lit) == Some(level) {
                counter += 1;
            }
        }

        // Resolution loop
        for _ in 0..self.config.max_resolution_steps {
            self.stats.resolution_steps += 1;

            if counter <= 1 {
                break; // Found UIP
            }

            // Find next literal to resolve
            let resolve_lit = self.find_resolution_literal(&current_clause, level, &seen)?;
            seen.insert(resolve_lit);

            // Get reason for this literal
            let reason = self.get_reason(resolve_lit)?;

            // Perform resolution
            current_clause.remove(&resolve_lit);
            counter -= 1;

            for &lit in &reason {
                if !current_clause.contains(&lit) {
                    current_clause.insert(lit);
                    if self.get_decision_level(lit) == Some(level) {
                        counter += 1;
                    }
                }
            }
        }

        Ok(current_clause.into_iter().collect())
    }

    /// Find literal for resolution.
    fn find_resolution_literal(
        &self,
        clause: &FxHashSet<Literal>,
        level: DecisionLevel,
        seen: &FxHashSet<Literal>,
    ) -> Result<Literal, String> {
        // Find last assigned literal at current level that hasn't been seen
        for &(literal, lit_level, _) in self.trail.iter().rev() {
            if lit_level == level && clause.contains(&literal) && !seen.contains(&literal) {
                return Ok(literal);
            }
        }

        Err("No resolution literal found".to_string())
    }

    /// Get decision level for a literal.
    fn get_decision_level(&self, literal: Literal) -> Option<DecisionLevel> {
        self.literal_position
            .get(&literal)
            .and_then(|&pos| self.trail.get(pos))
            .map(|(_, level, _)| *level)
    }

    /// Get reason (explanation) for a literal.
    fn get_reason(&self, literal: Literal) -> Result<Vec<Literal>, String> {
        let position = self
            .literal_position
            .get(&literal)
            .ok_or("Literal not in trail")?;

        let (_, _, explanation) = &self.trail[*position];

        match explanation {
            Explanation::TheoryPropagation { antecedents, .. } => Ok(antecedents.clone()),
            Explanation::EqualityPropagation { support, .. } => Ok(support.clone()),
            _ => Ok(Vec::new()),
        }
    }

    /// Minimize conflict clause.
    fn minimize_conflict(&self, literals: &[Literal]) -> Result<Vec<Literal>, String> {
        match self.config.minimization_algorithm {
            MinimizationAlgorithm::None => Ok(literals.to_vec()),
            MinimizationAlgorithm::Simple => self.minimize_simple(literals),
            MinimizationAlgorithm::Recursive => self.minimize_recursive(literals),
            MinimizationAlgorithm::BinaryResolution => self.minimize_binary_resolution(literals),
        }
    }

    /// Simple minimization (remove obviously redundant literals).
    fn minimize_simple(&self, literals: &[Literal]) -> Result<Vec<Literal>, String> {
        // Remove duplicates and keep only necessary literals
        let mut minimal = Vec::new();
        let mut seen = FxHashSet::default();

        for &lit in literals {
            if !seen.contains(&lit) {
                seen.insert(lit);
                minimal.push(lit);
            }
        }

        Ok(minimal)
    }

    /// Recursive minimization.
    fn minimize_recursive(&self, literals: &[Literal]) -> Result<Vec<Literal>, String> {
        let mut minimal = Vec::new();
        let mut redundant = FxHashSet::default();

        for &lit in literals {
            if self.is_redundant(lit, literals, &mut redundant)? {
                continue;
            }
            minimal.push(lit);
        }

        Ok(minimal)
    }

    /// Check if a literal is redundant.
    fn is_redundant(
        &self,
        literal: Literal,
        clause: &[Literal],
        redundant: &mut FxHashSet<Literal>,
    ) -> Result<bool, String> {
        if redundant.contains(&literal) {
            return Ok(true);
        }

        let reason = self.get_reason(literal).ok().unwrap_or_default();

        for &reason_lit in &reason {
            if !clause.contains(&reason_lit)
                && !redundant.contains(&reason_lit)
                && !self.is_redundant(reason_lit, clause, redundant)?
            {
                return Ok(false);
            }
        }

        redundant.insert(literal);
        Ok(true)
    }

    /// Binary resolution minimization.
    fn minimize_binary_resolution(&self, literals: &[Literal]) -> Result<Vec<Literal>, String> {
        // Simplified: same as simple for now
        self.minimize_simple(literals)
    }

    /// Compute backtrack level.
    fn compute_backtrack_level(
        &self,
        literals: &[Literal],
        _conflict_level: DecisionLevel,
    ) -> Result<DecisionLevel, String> {
        // Find second-highest decision level in the clause
        let mut levels: Vec<DecisionLevel> = literals
            .iter()
            .filter_map(|&lit| self.get_decision_level(lit))
            .collect();

        levels.sort_unstable();
        levels.dedup();

        if levels.len() >= 2 {
            Ok(levels[levels.len() - 2])
        } else if !levels.is_empty() {
            Ok(levels[0].saturating_sub(1))
        } else {
            Ok(0)
        }
    }

    /// Find UIP literal in clause.
    fn find_uip_literal(&self, literals: &[Literal]) -> Option<Literal> {
        // Find literal at highest decision level
        literals
            .iter()
            .max_by_key(|&&lit| self.get_decision_level(lit).unwrap_or(0))
            .copied()
    }

    /// Get learned clauses.
    pub fn learned_clauses(&self) -> &[ConflictClause] {
        &self.learned_clauses
    }

    /// Get theory blame statistics.
    pub fn theory_blame(&self) -> &FxHashMap<TheoryId, u64> {
        &self.theory_blame
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.trail.clear();
        self.literal_position.clear();
        self.level_boundaries.clear();
        self.current_level = 0;
        self.learned_clauses.clear();
        self.theory_blame.clear();
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ConflictResolutionStats::default();
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Explanation generator for theory conflicts.
pub struct ExplanationGenerator {
    /// Explanation cache.
    cache: FxHashMap<Literal, Explanation>,
}

impl ExplanationGenerator {
    /// Create new explanation generator.
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    /// Add explanation for a literal.
    pub fn add_explanation(&mut self, literal: Literal, explanation: Explanation) {
        self.cache.insert(literal, explanation);
    }

    /// Get explanation for a literal.
    pub fn get_explanation(&self, literal: Literal) -> Option<&Explanation> {
        self.cache.get(&literal)
    }

    /// Build explanation chain.
    pub fn build_chain(&self, literals: &[Literal]) -> Explanation {
        let mut antecedents = Vec::new();

        for &lit in literals {
            if let Some(explanation) = self.cache.get(&lit)
                && let Explanation::TheoryPropagation {
                    antecedents: ants, ..
                } = explanation
            {
                antecedents.extend_from_slice(ants);
            }
        }

        Explanation::TheoryPropagation {
            theory: 0,
            antecedents,
        }
    }

    /// Clear cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for ExplanationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-theory conflict analyzer.
pub struct MultiTheoryConflictAnalyzer {
    /// Individual theory resolvers.
    resolvers: FxHashMap<TheoryId, ConflictResolver>,

    /// Combined conflict statistics.
    combined_stats: ConflictResolutionStats,
}

impl MultiTheoryConflictAnalyzer {
    /// Create new multi-theory analyzer.
    pub fn new() -> Self {
        Self {
            resolvers: FxHashMap::default(),
            combined_stats: ConflictResolutionStats::default(),
        }
    }

    /// Register theory.
    pub fn register_theory(&mut self, theory: TheoryId, config: ConflictResolutionConfig) {
        self.resolvers
            .insert(theory, ConflictResolver::with_config(config));
    }

    /// Analyze conflict from a theory.
    pub fn analyze(&mut self, conflict: TheoryConflict) -> Result<ConflictAnalysis, String> {
        let resolver = self
            .resolvers
            .get_mut(&conflict.theory)
            .ok_or("Theory not registered")?;

        let analysis = resolver.analyze_conflict(conflict)?;

        // Update combined stats
        self.combined_stats.conflicts_analyzed += 1;

        Ok(analysis)
    }

    /// Get combined statistics.
    pub fn combined_stats(&self) -> &ConflictResolutionStats {
        &self.combined_stats
    }

    /// Get resolver for a theory.
    pub fn get_resolver(&self, theory: TheoryId) -> Option<&ConflictResolver> {
        self.resolvers.get(&theory)
    }

    /// Clear all resolvers.
    pub fn clear(&mut self) {
        for resolver in self.resolvers.values_mut() {
            resolver.clear();
        }
        self.combined_stats = ConflictResolutionStats::default();
    }
}

impl Default for MultiTheoryConflictAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let lit = Literal::positive(1);
        assert_eq!(lit.term, 1);
        assert!(lit.polarity);
    }

    #[test]
    fn test_literal_negation() {
        let lit = Literal::positive(1);
        let neg = lit.negate();
        assert!(!neg.polarity);
    }

    #[test]
    fn test_resolver_creation() {
        let resolver = ConflictResolver::new();
        assert_eq!(resolver.stats().conflicts_analyzed, 0);
    }

    #[test]
    fn test_add_assignment() {
        let mut resolver = ConflictResolver::new();
        let lit = Literal::positive(1);

        resolver.add_assignment(lit, 0, Explanation::Given);
        assert_eq!(resolver.trail.len(), 1);
    }

    #[test]
    fn test_decision_level() {
        let mut resolver = ConflictResolver::new();

        resolver.push_decision_level();
        assert_eq!(resolver.current_level, 1);
    }

    #[test]
    fn test_backtrack() {
        let mut resolver = ConflictResolver::new();

        resolver.push_decision_level();
        resolver.add_assignment(Literal::positive(1), 1, Explanation::Given);

        resolver.backtrack(0).expect("Backtrack failed");
        assert_eq!(resolver.trail.len(), 0);
    }

    #[test]
    fn test_conflict_analysis() {
        let mut resolver = ConflictResolver::new();

        let conflict = TheoryConflict {
            theory: 0,
            literals: vec![Literal::positive(1), Literal::negative(2)],
            explanation: Explanation::Given,
            level: 0,
        };

        let analysis = resolver.analyze_conflict(conflict);
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_explanation_generator() {
        let mut generator = ExplanationGenerator::new();
        let lit = Literal::positive(1);

        generator.add_explanation(lit, Explanation::Given);
        assert!(generator.get_explanation(lit).is_some());
    }

    #[test]
    fn test_multi_theory_analyzer() {
        let mut analyzer = MultiTheoryConflictAnalyzer::new();
        analyzer.register_theory(0, ConflictResolutionConfig::default());

        let conflict = TheoryConflict {
            theory: 0,
            literals: vec![Literal::positive(1)],
            explanation: Explanation::Given,
            level: 0,
        };

        let result = analyzer.analyze(conflict);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simple_minimization() {
        let resolver = ConflictResolver::new();

        let literals = vec![
            Literal::positive(1),
            Literal::positive(2),
            Literal::positive(1), // Duplicate
        ];

        let minimized = resolver
            .minimize_simple(&literals)
            .expect("Minimization failed");
        assert_eq!(minimized.len(), 2);
    }

    #[test]
    fn test_backtrack_level_computation() {
        let mut resolver = ConflictResolver::new();

        resolver.add_assignment(Literal::positive(1), 0, Explanation::Given);
        resolver.push_decision_level();
        resolver.add_assignment(Literal::positive(2), 1, Explanation::Given);
        resolver.push_decision_level();
        resolver.add_assignment(Literal::positive(3), 2, Explanation::Given);

        let literals = vec![
            Literal::positive(1),
            Literal::positive(2),
            Literal::positive(3),
        ];

        let level = resolver.compute_backtrack_level(&literals, 2);
        assert!(level.is_ok());
    }
}
