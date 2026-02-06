//! Trigger generation and management for E-matching
//!
//! This module provides automatic trigger (pattern) generation for quantified
//! formulas. Triggers are used to instantiate quantifiers based on matching ground
//! terms in the E-graph.
//!
//! # Trigger Selection
//!
//! Good triggers should:
//! - Cover all bound variables
//! - Be as specific as possible (reduce spurious matches)
//! - Avoid variable-only patterns
//! - Minimize cost (prefer shallow terms)
//!
//! # Algorithm
//!
//! Based on Z3's trigger generation in src/sat/smt/q_mam.cpp

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::{OxizError, Result};
use crate::sort::SortId;
use lasso::Spur;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::fmt;

/// A trigger for quantifier instantiation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trigger {
    /// The pattern terms
    pub patterns: SmallVec<[TermId; 2]>,
    /// Quality assessment
    pub quality: TriggerQuality,
    /// Estimated matching cost
    pub cost: u32,
    /// Variables covered by this trigger
    pub covered_vars: FxHashSet<Spur>,
}

/// Quality assessment of a trigger
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TriggerQuality {
    /// Excellent trigger: single pattern, covers all vars, low cost
    Excellent = 4,
    /// Good trigger: covers all vars, reasonable cost
    Good = 3,
    /// Fair trigger: may need multiple patterns or moderate cost
    Fair = 2,
    /// Poor trigger: high cost or many patterns
    Poor = 1,
    /// Unusable trigger
    Unusable = 0,
}

/// Configuration for trigger generation
#[derive(Debug, Clone)]
pub struct TriggerConfig {
    /// Maximum number of patterns per trigger
    pub max_patterns: usize,
    /// Whether to allow variable-only patterns
    pub allow_var_only: bool,
    /// Whether to allow ground patterns
    pub allow_ground: bool,
    /// Maximum pattern cost
    pub max_cost: u32,
    /// Whether to prefer single-pattern triggers
    pub prefer_single_pattern: bool,
    /// Maximum depth of patterns
    pub max_depth: usize,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            max_patterns: 3,
            allow_var_only: false,
            allow_ground: false,
            max_cost: 1000,
            prefer_single_pattern: true,
            max_depth: 10,
        }
    }
}

/// Statistics about trigger generation
#[derive(Debug, Clone, Default)]
pub struct TriggerStats {
    /// Number of triggers generated
    pub triggers_generated: usize,
    /// Number of excellent triggers
    pub excellent_triggers: usize,
    /// Number of good triggers
    pub good_triggers: usize,
    /// Number of fair triggers
    pub fair_triggers: usize,
    /// Number of poor triggers
    pub poor_triggers: usize,
    /// Number of unusable triggers
    pub unusable_triggers: usize,
}

/// Trigger generator
#[derive(Debug)]
pub struct TriggerGenerator {
    /// Configuration
    config: TriggerConfig,
    /// Statistics
    stats: TriggerStats,
}

impl TriggerGenerator {
    /// Create a new trigger generator
    pub fn new(config: TriggerConfig) -> Self {
        Self {
            config,
            stats: TriggerStats::default(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(TriggerConfig::default())
    }

    /// Generate triggers for a quantified formula
    ///
    /// Extracts patterns from the quantifier if present, otherwise infers triggers
    /// from the body.
    pub fn generate_triggers(
        &mut self,
        quant_id: TermId,
        manager: &TermManager,
    ) -> Result<Vec<Trigger>> {
        let Some(quant) = manager.get(quant_id) else {
            return Err(OxizError::EmatchError(format!(
                "Quantifier {:?} not found",
                quant_id
            )));
        };

        let (vars, body, patterns) = match &quant.kind {
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => (vars, *body, patterns),
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => (vars, *body, patterns),
            _ => {
                return Err(OxizError::EmatchError(
                    "Term is not a quantifier".to_string(),
                ));
            }
        };

        // If explicit patterns are provided, use them
        if !patterns.is_empty() {
            return self.triggers_from_explicit_patterns(vars, patterns, manager);
        }

        // Otherwise, infer triggers from the body
        self.infer_triggers(vars, body, manager)
    }

    /// Create triggers from explicit pattern annotations
    fn triggers_from_explicit_patterns(
        &mut self,
        vars: &[(Spur, SortId)],
        patterns: &[SmallVec<[TermId; 2]>],
        manager: &TermManager,
    ) -> Result<Vec<Trigger>> {
        let mut triggers = Vec::new();

        for pattern_set in patterns {
            let mut covered_vars = FxHashSet::default();
            let mut total_cost = 0;

            // Collect variables and costs from all patterns in this set
            for &pattern in pattern_set.iter() {
                self.collect_vars(pattern, vars, &mut covered_vars, manager)?;
                total_cost += self.estimate_cost(pattern, manager)?;
            }

            // Assess quality
            let quality =
                self.assess_quality(pattern_set.len(), total_cost, &covered_vars, vars.len());

            let trigger = Trigger {
                patterns: pattern_set.clone(),
                quality,
                cost: total_cost,
                covered_vars,
            };

            self.update_stats(quality);
            triggers.push(trigger);
        }

        Ok(triggers)
    }

    /// Infer triggers automatically from the quantifier body
    fn infer_triggers(
        &mut self,
        vars: &[(Spur, SortId)],
        body: TermId,
        manager: &TermManager,
    ) -> Result<Vec<Trigger>> {
        // Collect candidate patterns from the body
        let candidates = self.collect_candidates(body, vars, manager)?;

        if candidates.is_empty() {
            return Err(OxizError::EmatchError(
                "No trigger candidates found".to_string(),
            ));
        }

        // Select best triggers
        let triggers = self.select_triggers(&candidates, vars, manager)?;

        if triggers.is_empty() {
            return Err(OxizError::EmatchError(
                "No suitable triggers found".to_string(),
            ));
        }

        Ok(triggers)
    }

    /// Collect candidate patterns from a term
    fn collect_candidates(
        &self,
        term: TermId,
        vars: &[(Spur, SortId)],
        manager: &TermManager,
    ) -> Result<Vec<TermId>> {
        let mut candidates = Vec::new();
        let var_names: FxHashSet<Spur> = vars.iter().map(|(n, _)| *n).collect();

        self.collect_candidates_recursive(term, &var_names, &mut candidates, manager)?;

        Ok(candidates)
    }

    /// Recursive helper for collecting candidates
    fn collect_candidates_recursive(
        &self,
        term: TermId,
        var_names: &FxHashSet<Spur>,
        candidates: &mut Vec<TermId>,
        manager: &TermManager,
    ) -> Result<()> {
        let Some(t) = manager.get(term) else {
            return Ok(());
        };

        // Check if this term contains any bound variables
        if !self.contains_bound_var_quick(term, var_names, manager)? {
            return Ok(()); // Skip ground terms
        }

        match &t.kind {
            TermKind::Apply { args, .. } => {
                // Function application is a good candidate
                if self.is_good_candidate(term, var_names, manager)? {
                    candidates.push(term);
                }

                // Recurse into arguments
                for &arg in args.iter() {
                    self.collect_candidates_recursive(arg, var_names, candidates, manager)?;
                }
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs) => {
                // Arithmetic/equality predicates can be candidates
                if self.is_good_candidate(term, var_names, manager)? {
                    candidates.push(term);
                }

                self.collect_candidates_recursive(*lhs, var_names, candidates, manager)?;
                self.collect_candidates_recursive(*rhs, var_names, candidates, manager)?;
            }
            TermKind::Select(arr, _) => {
                // Array select is a good candidate
                if self.is_good_candidate(term, var_names, manager)? {
                    candidates.push(term);
                }

                self.collect_candidates_recursive(*arr, var_names, candidates, manager)?;
            }
            TermKind::And(args) | TermKind::Or(args) => {
                // Recurse into boolean connectives
                for &arg in args.iter() {
                    self.collect_candidates_recursive(arg, var_names, candidates, manager)?;
                }
            }
            TermKind::Not(inner) => {
                self.collect_candidates_recursive(*inner, var_names, candidates, manager)?;
            }
            TermKind::Implies(lhs, rhs) => {
                self.collect_candidates_recursive(*lhs, var_names, candidates, manager)?;
                self.collect_candidates_recursive(*rhs, var_names, candidates, manager)?;
            }
            TermKind::Ite(c, t_br, e_br) => {
                self.collect_candidates_recursive(*c, var_names, candidates, manager)?;
                self.collect_candidates_recursive(*t_br, var_names, candidates, manager)?;
                self.collect_candidates_recursive(*e_br, var_names, candidates, manager)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Check if a term is a good trigger candidate
    fn is_good_candidate(
        &self,
        term: TermId,
        var_names: &FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<bool> {
        let Some(t) = manager.get(term) else {
            return Ok(false);
        };

        // Must contain at least one bound variable
        if !self.contains_bound_var_quick(term, var_names, manager)? {
            return Ok(false);
        }

        // Check depth constraint
        let depth = self.compute_depth(term, manager)?;
        if depth > self.config.max_depth {
            return Ok(false);
        }

        match &t.kind {
            // Variable-only patterns are not good candidates (unless allowed)
            TermKind::Var(name) if var_names.contains(name) => Ok(self.config.allow_var_only),

            // Function applications are good candidates
            TermKind::Apply { .. } => Ok(true),

            // Predicates can be good candidates
            TermKind::Eq(_, _)
            | TermKind::Lt(_, _)
            | TermKind::Le(_, _)
            | TermKind::Gt(_, _)
            | TermKind::Ge(_, _) => Ok(true),

            // Array operations are good candidates
            TermKind::Select(_, _) | TermKind::Store(_, _, _) => Ok(true),

            // Ground terms are not good candidates (unless allowed)
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => Ok(self.config.allow_ground),

            _ => Ok(false),
        }
    }

    /// Quick check if term contains bound variables
    fn contains_bound_var_quick(
        &self,
        term: TermId,
        var_names: &FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<bool> {
        let Some(t) = manager.get(term) else {
            return Ok(false);
        };

        match &t.kind {
            TermKind::Var(name) => Ok(var_names.contains(name)),
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    if self.contains_bound_var_quick(arg, var_names, manager)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => Ok(self
                .contains_bound_var_quick(*lhs, var_names, manager)?
                || self.contains_bound_var_quick(*rhs, var_names, manager)?),
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                for &arg in args.iter() {
                    if self.contains_bound_var_quick(arg, var_names, manager)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    /// Select best triggers from candidates
    fn select_triggers(
        &mut self,
        candidates: &[TermId],
        vars: &[(Spur, SortId)],
        manager: &TermManager,
    ) -> Result<Vec<Trigger>> {
        let mut triggers = Vec::new();

        // Strategy 1: Try to find a single-pattern trigger that covers all variables
        if self.config.prefer_single_pattern {
            for &candidate in candidates {
                let mut covered = FxHashSet::default();
                self.collect_vars(candidate, vars, &mut covered, manager)?;

                if covered.len() == vars.len() {
                    let cost = self.estimate_cost(candidate, manager)?;
                    if cost <= self.config.max_cost {
                        let quality = self.assess_quality(1, cost, &covered, vars.len());

                        let trigger = Trigger {
                            patterns: smallvec::smallvec![candidate],
                            quality,
                            cost,
                            covered_vars: covered,
                        };

                        self.update_stats(quality);
                        triggers.push(trigger);
                    }
                }
            }
        }

        // Strategy 2: Use multi-pattern triggers if needed
        if triggers.is_empty() || !self.config.prefer_single_pattern {
            let multi = self.select_multi_pattern_triggers(candidates, vars, manager)?;
            triggers.extend(multi);
        }

        // Sort by quality (best first)
        triggers.sort_by(|a, b| b.quality.cmp(&a.quality).then(a.cost.cmp(&b.cost)));

        Ok(triggers)
    }

    /// Select multi-pattern triggers
    fn select_multi_pattern_triggers(
        &mut self,
        candidates: &[TermId],
        vars: &[(Spur, SortId)],
        manager: &TermManager,
    ) -> Result<Vec<Trigger>> {
        let mut triggers = Vec::new();

        // Try all combinations up to max_patterns
        for size in 2..=self.config.max_patterns.min(candidates.len()) {
            // Use a simple greedy approach: find sets that cover all variables
            let combinations = self.greedy_cover(candidates, vars, size, manager)?;

            for pattern_set in combinations {
                let mut covered = FxHashSet::default();
                let mut total_cost = 0;

                for &p in &pattern_set {
                    self.collect_vars(p, vars, &mut covered, manager)?;
                    total_cost += self.estimate_cost(p, manager)?;
                }

                if covered.len() == vars.len() && total_cost <= self.config.max_cost {
                    let quality =
                        self.assess_quality(pattern_set.len(), total_cost, &covered, vars.len());

                    let trigger = Trigger {
                        patterns: pattern_set,
                        quality,
                        cost: total_cost,
                        covered_vars: covered,
                    };

                    self.update_stats(quality);
                    triggers.push(trigger);
                }
            }
        }

        Ok(triggers)
    }

    /// Greedy algorithm to find pattern sets that cover all variables
    fn greedy_cover(
        &self,
        candidates: &[TermId],
        vars: &[(Spur, SortId)],
        max_size: usize,
        manager: &TermManager,
    ) -> Result<Vec<SmallVec<[TermId; 2]>>> {
        let all_vars: FxHashSet<Spur> = vars.iter().map(|(n, _)| *n).collect();
        let mut results = Vec::new();

        // Compute variable coverage for each candidate
        let mut candidate_vars: Vec<(TermId, FxHashSet<Spur>)> = Vec::new();
        for &candidate in candidates {
            let mut covered = FxHashSet::default();
            self.collect_vars(candidate, vars, &mut covered, manager)?;
            candidate_vars.push((candidate, covered));
        }

        // Sort by coverage (descending)
        candidate_vars.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        // Greedy selection
        let mut current_set = SmallVec::new();
        let mut current_coverage = FxHashSet::default();

        for (candidate, covered) in candidate_vars {
            if current_set.len() >= max_size {
                break;
            }

            // Add if it contributes new variables
            if !covered.is_subset(&current_coverage) {
                current_set.push(candidate);
                current_coverage.extend(covered);

                // Check if we've covered all variables
                if current_coverage == all_vars {
                    results.push(current_set.clone());
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Collect variables from a pattern
    fn collect_vars(
        &self,
        pattern: TermId,
        vars: &[(Spur, SortId)],
        covered: &mut FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<()> {
        let var_names: FxHashSet<Spur> = vars.iter().map(|(n, _)| *n).collect();
        self.collect_vars_recursive(pattern, &var_names, covered, manager)
    }

    /// Recursive helper for collecting variables
    fn collect_vars_recursive(
        &self,
        term: TermId,
        var_names: &FxHashSet<Spur>,
        covered: &mut FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<()> {
        let Some(t) = manager.get(term) else {
            return Ok(());
        };

        match &t.kind {
            TermKind::Var(name) if var_names.contains(name) => {
                covered.insert(*name);
            }
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    self.collect_vars_recursive(arg, var_names, covered, manager)?;
                }
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => {
                self.collect_vars_recursive(*lhs, var_names, covered, manager)?;
                self.collect_vars_recursive(*rhs, var_names, covered, manager)?;
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                for &arg in args.iter() {
                    self.collect_vars_recursive(arg, var_names, covered, manager)?;
                }
            }
            TermKind::Not(inner) | TermKind::Neg(inner) => {
                self.collect_vars_recursive(*inner, var_names, covered, manager)?;
            }
            TermKind::Select(arr, idx) => {
                self.collect_vars_recursive(*arr, var_names, covered, manager)?;
                self.collect_vars_recursive(*idx, var_names, covered, manager)?;
            }
            TermKind::Store(arr, idx, val) => {
                self.collect_vars_recursive(*arr, var_names, covered, manager)?;
                self.collect_vars_recursive(*idx, var_names, covered, manager)?;
                self.collect_vars_recursive(*val, var_names, covered, manager)?;
            }
            TermKind::Ite(c, t_br, e_br) => {
                self.collect_vars_recursive(*c, var_names, covered, manager)?;
                self.collect_vars_recursive(*t_br, var_names, covered, manager)?;
                self.collect_vars_recursive(*e_br, var_names, covered, manager)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Estimate matching cost of a pattern
    fn estimate_cost(&self, pattern: TermId, manager: &TermManager) -> Result<u32> {
        let Some(t) = manager.get(pattern) else {
            return Ok(100);
        };

        let base_cost = match &t.kind {
            TermKind::Var(_) => 10,
            TermKind::Apply { args, .. } => {
                let mut cost = 5;
                for &arg in args.iter() {
                    cost += self.estimate_cost(arg, manager)?;
                }
                cost
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs) => {
                3 + self.estimate_cost(*lhs, manager)? + self.estimate_cost(*rhs, manager)?
            }
            TermKind::Select(arr, idx) => {
                4 + self.estimate_cost(*arr, manager)? + self.estimate_cost(*idx, manager)?
            }
            _ => 1,
        };

        Ok(base_cost)
    }

    /// Compute depth of a term
    fn compute_depth(&self, term: TermId, manager: &TermManager) -> Result<usize> {
        let Some(t) = manager.get(term) else {
            return Ok(0);
        };

        let child_depth = match &t.kind {
            TermKind::Var(_)
            | TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => 0,
            TermKind::Apply { args, .. } => {
                let mut max = 0;
                for &arg in args.iter() {
                    let d = self.compute_depth(arg, manager)?;
                    if d > max {
                        max = d;
                    }
                }
                max
            }
            TermKind::Eq(lhs, rhs) | TermKind::Lt(lhs, rhs) | TermKind::Select(lhs, rhs) => {
                let d1 = self.compute_depth(*lhs, manager)?;
                let d2 = self.compute_depth(*rhs, manager)?;
                d1.max(d2)
            }
            _ => 0,
        };

        Ok(child_depth + 1)
    }

    /// Assess trigger quality
    fn assess_quality(
        &self,
        num_patterns: usize,
        cost: u32,
        covered: &FxHashSet<Spur>,
        total_vars: usize,
    ) -> TriggerQuality {
        // Must cover all variables
        if covered.len() < total_vars {
            return TriggerQuality::Unusable;
        }

        // Single pattern, low cost = excellent
        if num_patterns == 1 && cost <= 50 {
            return TriggerQuality::Excellent;
        }

        // Single pattern, moderate cost = good
        if num_patterns == 1 && cost <= 200 {
            return TriggerQuality::Good;
        }

        // Multi-pattern, reasonable cost = fair
        if num_patterns <= 2 && cost <= 500 {
            return TriggerQuality::Fair;
        }

        // Otherwise = poor
        TriggerQuality::Poor
    }

    /// Update statistics
    fn update_stats(&mut self, quality: TriggerQuality) {
        self.stats.triggers_generated += 1;
        match quality {
            TriggerQuality::Excellent => self.stats.excellent_triggers += 1,
            TriggerQuality::Good => self.stats.good_triggers += 1,
            TriggerQuality::Fair => self.stats.fair_triggers += 1,
            TriggerQuality::Poor => self.stats.poor_triggers += 1,
            TriggerQuality::Unusable => self.stats.unusable_triggers += 1,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &TriggerStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = TriggerStats::default();
    }
}

/// Represents trigger selection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerSelection {
    /// Use only the best trigger
    BestOnly,
    /// Use all good triggers
    AllGood,
    /// Use all triggers
    All,
}

impl fmt::Display for TriggerQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TriggerQuality::Excellent => write!(f, "Excellent"),
            TriggerQuality::Good => write!(f, "Good"),
            TriggerQuality::Fair => write!(f, "Fair"),
            TriggerQuality::Poor => write!(f, "Poor"),
            TriggerQuality::Unusable => write!(f, "Unusable"),
        }
    }
}

impl fmt::Display for Trigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trigger({} patterns, quality={}, cost={}, covers {} vars)",
            self.patterns.len(),
            self.quality,
            self.cost,
            self.covered_vars.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;
    use lasso::Key;

    fn setup() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_trigger_config_default() {
        let config = TriggerConfig::default();
        assert_eq!(config.max_patterns, 3);
        assert!(!config.allow_var_only);
        assert!(!config.allow_ground);
        assert_eq!(config.max_cost, 1000);
        assert!(config.prefer_single_pattern);
    }

    #[test]
    fn test_trigger_generator_creation() {
        let generator = TriggerGenerator::new_default();
        assert_eq!(generator.stats.triggers_generated, 0);
    }

    #[test]
    fn test_trigger_quality_ordering() {
        assert!(TriggerQuality::Excellent > TriggerQuality::Good);
        assert!(TriggerQuality::Good > TriggerQuality::Fair);
        assert!(TriggerQuality::Fair > TriggerQuality::Poor);
        assert!(TriggerQuality::Poor > TriggerQuality::Unusable);
    }

    #[test]
    fn test_assess_quality() {
        let generator = TriggerGenerator::new_default();
        let all_vars: FxHashSet<Spur> =
            [Spur::try_from_usize(0).unwrap()].iter().copied().collect();

        // Excellent: single pattern, low cost, covers all
        let q1 = generator.assess_quality(1, 30, &all_vars, 1);
        assert_eq!(q1, TriggerQuality::Excellent);

        // Good: single pattern, moderate cost
        let q2 = generator.assess_quality(1, 150, &all_vars, 1);
        assert_eq!(q2, TriggerQuality::Good);

        // Unusable: doesn't cover all variables
        let empty_vars: FxHashSet<Spur> = FxHashSet::default();
        let q3 = generator.assess_quality(1, 30, &empty_vars, 1);
        assert_eq!(q3, TriggerQuality::Unusable);
    }

    #[test]
    fn test_generate_triggers_with_explicit_patterns() {
        let mut manager = setup();
        let mut generator = TriggerGenerator::new_default();

        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        // Create: forall x. P(f(x)) with explicit pattern f(x)
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let p_fx = manager.mk_apply("P", [f_x], bool_sort);

        let _x_name = manager.intern_str("x");
        let var_strs = vec![("x", int_sort)];
        let patterns: Vec<SmallVec<[TermId; 2]>> = vec![smallvec::smallvec![f_x]];

        let forall = manager.mk_forall_with_patterns(var_strs, p_fx, patterns);

        let triggers = generator.generate_triggers(forall, &manager).unwrap();

        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].patterns.len(), 1);
        assert_eq!(triggers[0].patterns[0], f_x);
    }

    #[test]
    fn test_collect_candidates() {
        let mut manager = setup();
        let generator = TriggerGenerator::new_default();

        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        // Create term: P(f(x)) ∧ Q(g(x))
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let g_x = manager.mk_apply("g", [x], int_sort);
        let p_fx = manager.mk_apply("P", [f_x], bool_sort);
        let q_gx = manager.mk_apply("Q", [g_x], bool_sort);
        let body = manager.mk_and([p_fx, q_gx]);

        let x_name = manager.intern_str("x");
        let vars = vec![(x_name, int_sort)];

        let candidates = generator.collect_candidates(body, &vars, &manager).unwrap();

        // Should find P(f(x)), Q(g(x)), f(x), g(x) as candidates
        assert!(!candidates.is_empty());
        assert!(candidates.contains(&f_x));
        assert!(candidates.contains(&g_x));
    }

    #[test]
    fn test_estimate_cost() {
        let mut manager = setup();
        let generator = TriggerGenerator::new_default();

        let int_sort = manager.sorts.int_sort;

        // Variable should have moderate cost
        let x = manager.mk_var("x", int_sort);
        let cost_x = generator.estimate_cost(x, &manager).unwrap();

        // Function application should have higher cost
        let f_x = manager.mk_apply("f", [x], int_sort);
        let cost_fx = generator.estimate_cost(f_x, &manager).unwrap();

        assert!(cost_fx > cost_x);
    }

    #[test]
    fn test_compute_depth() {
        let mut manager = setup();
        let generator = TriggerGenerator::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        // x: depth 1
        let depth1 = generator.compute_depth(x, &manager).unwrap();
        assert_eq!(depth1, 1);

        // f(x): depth 2
        let f_x = manager.mk_apply("f", [x], int_sort);
        let depth2 = generator.compute_depth(f_x, &manager).unwrap();
        assert_eq!(depth2, 2);

        // g(f(x)): depth 3
        let g_fx = manager.mk_apply("g", [f_x], int_sort);
        let depth3 = generator.compute_depth(g_fx, &manager).unwrap();
        assert_eq!(depth3, 3);
    }
}
