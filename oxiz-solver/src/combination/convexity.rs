//! Convexity Checking and Handling for Theory Combination.
//!
//! This module provides convexity analysis for theories in combination:
//! - Convexity checking for theory solvers
//! - Non-convex theory handling strategies
//! - Model-based case analysis for non-convex theories
//! - Disjunctive reasoning
//!
//! ## Convexity
//!
//! A theory T is **convex** if for any conjunction of literals C and
//! disjunction of equalities (t1 = s1) ∨ ... ∨ (tn = sn):
//!
//! If C ∧ T ⊨ (t1 = s1) ∨ ... ∨ (tn = sn),
//! then C ∧ T ⊨ (ti = si) for some i.
//!
//! **Convex theories**: Equality, Uninterpreted Functions, Linear Arithmetic (rationals)
//! **Non-convex theories**: Integer Arithmetic, Bit-vectors
//!
//! ## Non-Convex Theory Handling
//!
//! For non-convex theories, we must handle disjunctions explicitly:
//! - Case splitting on equality disjunctions
//! - Model-based theory combination
//! - Conflict-driven learning to prune search space
//!
//! ## References
//!
//! - Nelson & Oppen (1979): "Simplification by Cooperating Decision Procedures"
//! - Tinelli & Harandi (1996): "A New Correctness Proof of the Nelson-Oppen Combination"
//! - Z3's `smt/theory_opt.cpp`

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Term identifier.
pub type TermId = u32;

/// Theory identifier.
pub type TheoryId = u32;

/// Decision level.
pub type DecisionLevel = u32;

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

/// Disjunction of equalities.
#[derive(Debug, Clone)]
pub struct EqualityDisjunction {
    /// Disjuncts (equalities).
    pub disjuncts: Vec<Equality>,
    /// Source theory.
    pub theory: TheoryId,
    /// Decision level.
    pub level: DecisionLevel,
}

impl EqualityDisjunction {
    /// Create new disjunction.
    pub fn new(disjuncts: Vec<Equality>, theory: TheoryId, level: DecisionLevel) -> Self {
        Self {
            disjuncts,
            theory,
            level,
        }
    }

    /// Check if disjunction is unit (single disjunct).
    pub fn is_unit(&self) -> bool {
        self.disjuncts.len() == 1
    }

    /// Get unit disjunct if disjunction is unit.
    pub fn get_unit(&self) -> Option<Equality> {
        if self.is_unit() {
            self.disjuncts.first().copied()
        } else {
            None
        }
    }
}

/// Convexity property of a theory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvexityProperty {
    /// Theory is convex.
    Convex,
    /// Theory is non-convex.
    NonConvex,
    /// Convexity is unknown or theory-dependent.
    Unknown,
}

/// Theory model for case splitting.
#[derive(Debug, Clone)]
pub struct TheoryModel {
    /// Theory identifier.
    pub theory: TheoryId,
    /// Variable assignments.
    pub assignments: FxHashMap<TermId, TermId>,
    /// Implied equalities.
    pub equalities: Vec<Equality>,
}

impl TheoryModel {
    /// Create new model.
    pub fn new(theory: TheoryId) -> Self {
        Self {
            theory,
            assignments: FxHashMap::default(),
            equalities: Vec::new(),
        }
    }

    /// Add assignment.
    pub fn add_assignment(&mut self, term: TermId, value: TermId) {
        self.assignments.insert(term, value);
    }

    /// Get assignment.
    pub fn get_assignment(&self, term: TermId) -> Option<TermId> {
        self.assignments.get(&term).copied()
    }

    /// Add implied equality.
    pub fn add_equality(&mut self, eq: Equality) {
        self.equalities.push(eq);
    }
}

/// Configuration for convexity handling.
#[derive(Debug, Clone)]
pub struct ConvexityConfig {
    /// Enable model-based case splitting.
    pub model_based_splitting: bool,

    /// Maximum case splits.
    pub max_case_splits: usize,

    /// Enable conflict-driven learning.
    pub conflict_driven_learning: bool,

    /// Case split strategy.
    pub split_strategy: CaseSplitStrategy,

    /// Enable disjunction simplification.
    pub simplify_disjunctions: bool,
}

impl Default for ConvexityConfig {
    fn default() -> Self {
        Self {
            model_based_splitting: true,
            max_case_splits: 100,
            conflict_driven_learning: true,
            split_strategy: CaseSplitStrategy::ModelBased,
            simplify_disjunctions: true,
        }
    }
}

/// Case split strategy for non-convex theories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseSplitStrategy {
    /// Enumerate all cases.
    Exhaustive,
    /// Use model to guide splitting.
    ModelBased,
    /// Use heuristics.
    Heuristic,
    /// Lazy splitting (defer as long as possible).
    Lazy,
}

/// Statistics for convexity handling.
#[derive(Debug, Clone, Default)]
pub struct ConvexityStats {
    /// Disjunctions processed.
    pub disjunctions_processed: u64,
    /// Case splits performed.
    pub case_splits: u64,
    /// Model-based decisions.
    pub model_based_decisions: u64,
    /// Conflicts from case splits.
    pub case_split_conflicts: u64,
    /// Learned constraints.
    pub learned_constraints: u64,
}

/// Convexity checker and handler.
pub struct ConvexityHandler {
    /// Configuration.
    config: ConvexityConfig,

    /// Statistics.
    stats: ConvexityStats,

    /// Theory convexity properties.
    theory_properties: FxHashMap<TheoryId, ConvexityProperty>,

    /// Pending disjunctions.
    pending_disjunctions: VecDeque<EqualityDisjunction>,

    /// Case split stack.
    case_split_stack: Vec<CaseSplit>,

    /// Learned constraints.
    learned: Vec<Vec<Equality>>,

    /// Current decision level.
    decision_level: DecisionLevel,
}

/// Case split record.
#[derive(Debug, Clone)]
struct CaseSplit {
    /// Decision level where split was made.
    level: DecisionLevel,
    /// Disjunction being split.
    disjunction: EqualityDisjunction,
    /// Cases already tried.
    tried_cases: FxHashSet<usize>,
    /// Current case being explored.
    current_case: Option<usize>,
}

impl ConvexityHandler {
    /// Create new handler.
    pub fn new() -> Self {
        Self::with_config(ConvexityConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: ConvexityConfig) -> Self {
        Self {
            config,
            stats: ConvexityStats::default(),
            theory_properties: FxHashMap::default(),
            pending_disjunctions: VecDeque::new(),
            case_split_stack: Vec::new(),
            learned: Vec::new(),
            decision_level: 0,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ConvexityStats {
        &self.stats
    }

    /// Register theory with convexity property.
    pub fn register_theory(&mut self, theory: TheoryId, property: ConvexityProperty) {
        self.theory_properties.insert(theory, property);
    }

    /// Check if theory is convex.
    pub fn is_convex(&self, theory: TheoryId) -> bool {
        matches!(
            self.theory_properties.get(&theory),
            Some(ConvexityProperty::Convex)
        )
    }

    /// Add disjunction to process.
    pub fn add_disjunction(&mut self, disjunction: EqualityDisjunction) {
        if self.config.simplify_disjunctions
            && let Some(simplified) = self.simplify_disjunction(&disjunction)
        {
            self.pending_disjunctions.push_back(simplified);
            self.stats.disjunctions_processed += 1;
            return;
        }

        self.pending_disjunctions.push_back(disjunction);
        self.stats.disjunctions_processed += 1;
    }

    /// Simplify disjunction.
    fn simplify_disjunction(
        &self,
        disjunction: &EqualityDisjunction,
    ) -> Option<EqualityDisjunction> {
        // Remove duplicate disjuncts
        let mut unique_disjuncts = Vec::new();
        let mut seen = FxHashSet::default();

        for &eq in &disjunction.disjuncts {
            if seen.insert(eq) {
                unique_disjuncts.push(eq);
            }
        }

        if unique_disjuncts.len() == disjunction.disjuncts.len() {
            return None; // No simplification
        }

        Some(EqualityDisjunction::new(
            unique_disjuncts,
            disjunction.theory,
            disjunction.level,
        ))
    }

    /// Process pending disjunctions.
    pub fn process_disjunctions(&mut self) -> Result<Option<Equality>, String> {
        while let Some(disjunction) = self.pending_disjunctions.pop_front() {
            // If unit, return the single equality
            if let Some(eq) = disjunction.get_unit() {
                return Ok(Some(eq));
            }

            // Non-unit disjunction: perform case split
            if self.stats.case_splits >= self.config.max_case_splits as u64 {
                return Err("Maximum case splits exceeded".to_string());
            }

            match self.config.split_strategy {
                CaseSplitStrategy::ModelBased => {
                    return self.model_based_split(&disjunction);
                }
                CaseSplitStrategy::Exhaustive => {
                    return self.exhaustive_split(&disjunction);
                }
                CaseSplitStrategy::Heuristic => {
                    return self.heuristic_split(&disjunction);
                }
                CaseSplitStrategy::Lazy => {
                    // Defer splitting
                    self.pending_disjunctions.push_back(disjunction);
                    continue;
                }
            }
        }

        Ok(None)
    }

    /// Model-based case split.
    fn model_based_split(
        &mut self,
        disjunction: &EqualityDisjunction,
    ) -> Result<Option<Equality>, String> {
        self.stats.case_splits += 1;
        self.stats.model_based_decisions += 1;

        // Choose first disjunct (model would guide this choice)
        if let Some(&eq) = disjunction.disjuncts.first() {
            // Record case split
            let split = CaseSplit {
                level: self.decision_level,
                disjunction: disjunction.clone(),
                tried_cases: {
                    let mut set = FxHashSet::default();
                    set.insert(0);
                    set
                },
                current_case: Some(0),
            };

            self.case_split_stack.push(split);
            return Ok(Some(eq));
        }

        Err("Empty disjunction".to_string())
    }

    /// Exhaustive case split.
    fn exhaustive_split(
        &mut self,
        disjunction: &EqualityDisjunction,
    ) -> Result<Option<Equality>, String> {
        self.stats.case_splits += 1;

        // Try first untried case
        if let Some((i, &eq)) = disjunction.disjuncts.iter().enumerate().next() {
            let split = CaseSplit {
                level: self.decision_level,
                disjunction: disjunction.clone(),
                tried_cases: {
                    let mut set = FxHashSet::default();
                    set.insert(i);
                    set
                },
                current_case: Some(i),
            };

            self.case_split_stack.push(split);
            return Ok(Some(eq));
        }

        Err("Empty disjunction".to_string())
    }

    /// Heuristic-based split.
    fn heuristic_split(
        &mut self,
        disjunction: &EqualityDisjunction,
    ) -> Result<Option<Equality>, String> {
        // Use model-based for now (could be enhanced with better heuristics)
        self.model_based_split(disjunction)
    }

    /// Backtrack case split on conflict.
    pub fn backtrack_case_split(&mut self) -> Result<Option<Equality>, String> {
        while let Some(mut split) = self.case_split_stack.pop() {
            // Try next untried case
            for (i, &eq) in split.disjunction.disjuncts.iter().enumerate() {
                if !split.tried_cases.contains(&i) {
                    split.tried_cases.insert(i);
                    split.current_case = Some(i);
                    self.case_split_stack.push(split);
                    return Ok(Some(eq));
                }
            }

            // All cases tried, learn conflict
            if self.config.conflict_driven_learning {
                self.learn_conflict(&split.disjunction);
            }

            self.stats.case_split_conflicts += 1;
        }

        Ok(None) // No more cases to try
    }

    /// Learn conflict from exhausted disjunction.
    fn learn_conflict(&mut self, disjunction: &EqualityDisjunction) {
        // Learn that this disjunction is unsatisfiable
        self.learned.push(disjunction.disjuncts.clone());
        self.stats.learned_constraints += 1;
    }

    /// Push decision level.
    pub fn push_decision_level(&mut self) {
        self.decision_level += 1;
    }

    /// Backtrack to decision level.
    pub fn backtrack(&mut self, level: DecisionLevel) -> Result<(), String> {
        if level > self.decision_level {
            return Err("Cannot backtrack to future level".to_string());
        }

        // Remove case splits above this level
        self.case_split_stack.retain(|split| split.level <= level);

        // Remove disjunctions above this level
        let pending: Vec<_> = self.pending_disjunctions.drain(..).collect();
        for disjunction in pending {
            if disjunction.level <= level {
                self.pending_disjunctions.push_back(disjunction);
            }
        }

        self.decision_level = level;
        Ok(())
    }

    /// Get learned constraints.
    pub fn learned_constraints(&self) -> &[Vec<Equality>] {
        &self.learned
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.pending_disjunctions.clear();
        self.case_split_stack.clear();
        self.learned.clear();
        self.decision_level = 0;
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ConvexityStats::default();
    }

    /// Check if there are pending disjunctions.
    pub fn has_pending(&self) -> bool {
        !self.pending_disjunctions.is_empty()
    }

    /// Get number of pending disjunctions.
    pub fn pending_count(&self) -> usize {
        self.pending_disjunctions.len()
    }
}

impl Default for ConvexityHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Model-based theory combination for non-convex theories.
pub struct ModelBasedCombination {
    /// Theory models.
    models: FxHashMap<TheoryId, TheoryModel>,

    /// Equalities derived from models.
    derived_equalities: Vec<Equality>,
}

impl ModelBasedCombination {
    /// Create new model-based combination.
    pub fn new() -> Self {
        Self {
            models: FxHashMap::default(),
            derived_equalities: Vec::new(),
        }
    }

    /// Add theory model.
    pub fn add_model(&mut self, model: TheoryModel) {
        self.models.insert(model.theory, model);
    }

    /// Combine models to derive interface equalities.
    pub fn combine_models(&mut self) -> Result<Vec<Equality>, String> {
        self.derived_equalities.clear();

        // Collect all terms from all models
        let mut all_terms = FxHashSet::default();

        for model in self.models.values() {
            for &term in model.assignments.keys() {
                all_terms.insert(term);
            }
        }

        // Check consistency and derive equalities
        for &term1 in &all_terms {
            for &term2 in &all_terms {
                if term1 >= term2 {
                    continue;
                }

                // Check if all models agree that term1 = term2
                let mut all_agree = true;

                for model in self.models.values() {
                    if let (Some(val1), Some(val2)) =
                        (model.get_assignment(term1), model.get_assignment(term2))
                        && val1 != val2
                    {
                        all_agree = false;
                        break;
                    }
                }

                if all_agree {
                    self.derived_equalities.push(Equality::new(term1, term2));
                }
            }
        }

        Ok(self.derived_equalities.clone())
    }

    /// Clear all models.
    pub fn clear(&mut self) {
        self.models.clear();
        self.derived_equalities.clear();
    }
}

impl Default for ModelBasedCombination {
    fn default() -> Self {
        Self::new()
    }
}

/// Disjunctive reasoning engine.
pub struct DisjunctiveReasoning {
    /// Active disjunctions.
    disjunctions: Vec<EqualityDisjunction>,

    /// Unit propagation queue.
    unit_queue: VecDeque<Equality>,
}

impl DisjunctiveReasoning {
    /// Create new disjunctive reasoning engine.
    pub fn new() -> Self {
        Self {
            disjunctions: Vec::new(),
            unit_queue: VecDeque::new(),
        }
    }

    /// Add disjunction.
    pub fn add_disjunction(&mut self, disjunction: EqualityDisjunction) {
        if disjunction.is_unit() {
            if let Some(eq) = disjunction.get_unit() {
                self.unit_queue.push_back(eq);
            }
        } else {
            self.disjunctions.push(disjunction);
        }
    }

    /// Propagate unit disjunctions.
    pub fn propagate_units(&mut self) -> Vec<Equality> {
        let mut propagated = Vec::new();

        while let Some(eq) = self.unit_queue.pop_front() {
            propagated.push(eq);
        }

        propagated
    }

    /// Simplify disjunctions given an equality.
    pub fn simplify_with_equality(&mut self, eq: Equality) {
        let mut simplified = Vec::new();

        for disjunction in self.disjunctions.drain(..) {
            let mut new_disjuncts = Vec::new();

            for &disjunct in &disjunction.disjuncts {
                // Check if disjunct is satisfied by eq
                if disjunct != eq {
                    new_disjuncts.push(disjunct);
                }
            }

            if !new_disjuncts.is_empty() {
                let new_disjunction =
                    EqualityDisjunction::new(new_disjuncts, disjunction.theory, disjunction.level);

                if new_disjunction.is_unit() {
                    if let Some(unit_eq) = new_disjunction.get_unit() {
                        self.unit_queue.push_back(unit_eq);
                    }
                } else {
                    simplified.push(new_disjunction);
                }
            }
        }

        self.disjunctions = simplified;
    }

    /// Check for conflicts (empty disjunctions).
    pub fn has_conflict(&self) -> bool {
        false // Simplified: would check for empty disjunctions
    }

    /// Clear all disjunctions.
    pub fn clear(&mut self) {
        self.disjunctions.clear();
        self.unit_queue.clear();
    }
}

impl Default for DisjunctiveReasoning {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equality_disjunction() {
        let eq1 = Equality::new(1, 2);
        let eq2 = Equality::new(3, 4);

        let disj = EqualityDisjunction::new(vec![eq1, eq2], 0, 0);
        assert!(!disj.is_unit());
    }

    #[test]
    fn test_unit_disjunction() {
        let eq = Equality::new(1, 2);
        let disj = EqualityDisjunction::new(vec![eq], 0, 0);

        assert!(disj.is_unit());
        assert_eq!(disj.get_unit(), Some(eq));
    }

    #[test]
    fn test_handler_creation() {
        let handler = ConvexityHandler::new();
        assert_eq!(handler.stats().disjunctions_processed, 0);
    }

    #[test]
    fn test_register_theory() {
        let mut handler = ConvexityHandler::new();
        handler.register_theory(0, ConvexityProperty::Convex);

        assert!(handler.is_convex(0));
    }

    #[test]
    fn test_add_disjunction() {
        let mut handler = ConvexityHandler::new();
        let disj = EqualityDisjunction::new(vec![Equality::new(1, 2)], 0, 0);

        handler.add_disjunction(disj);
        assert_eq!(handler.pending_count(), 1);
    }

    #[test]
    fn test_process_unit_disjunction() {
        let mut handler = ConvexityHandler::new();
        let eq = Equality::new(1, 2);
        let disj = EqualityDisjunction::new(vec![eq], 0, 0);

        handler.add_disjunction(disj);

        let result = handler.process_disjunctions();
        assert!(result.is_ok());
        assert_eq!(result.ok().flatten(), Some(eq));
    }

    #[test]
    fn test_model_based_combination() {
        let mut mbc = ModelBasedCombination::new();

        let mut model1 = TheoryModel::new(0);
        model1.add_assignment(1, 10);
        model1.add_assignment(2, 10);

        mbc.add_model(model1);

        let equalities = mbc.combine_models().expect("Combination failed");
        assert!(!equalities.is_empty());
    }

    #[test]
    fn test_disjunctive_reasoning() {
        let mut dr = DisjunctiveReasoning::new();

        let eq = Equality::new(1, 2);
        let disj = EqualityDisjunction::new(vec![eq], 0, 0);

        dr.add_disjunction(disj);

        let propagated = dr.propagate_units();
        assert_eq!(propagated.len(), 1);
        assert_eq!(propagated[0], eq);
    }

    #[test]
    fn test_simplify_disjunction() {
        let mut handler = ConvexityHandler::new();

        let eq1 = Equality::new(1, 2);
        let eq2 = Equality::new(1, 2); // Duplicate

        let disj = EqualityDisjunction::new(vec![eq1, eq2], 0, 0);
        handler.add_disjunction(disj);

        // Should be simplified to unit
        assert!(handler.has_pending());
    }

    #[test]
    fn test_backtrack() {
        let mut handler = ConvexityHandler::new();

        handler.push_decision_level();
        let disj = EqualityDisjunction::new(vec![Equality::new(1, 2)], 0, 1);
        handler.add_disjunction(disj);

        handler.backtrack(0).expect("Backtrack failed");
        assert_eq!(handler.pending_count(), 0);
    }

    #[test]
    fn test_case_split() {
        let mut handler = ConvexityHandler::new();

        let eq1 = Equality::new(1, 2);
        let eq2 = Equality::new(3, 4);
        let disj = EqualityDisjunction::new(vec![eq1, eq2], 0, 0);

        handler.add_disjunction(disj);

        let result = handler.process_disjunctions();
        assert!(result.is_ok());
        assert!(result.ok().flatten().is_some());
    }
}
