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
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use core::fmt;
use smallvec::SmallVec;

/// Sub-terms a trigger walk descends into.
///
/// This is the exhaustive [`crate::ast::traversal::get_children`] with one
/// deliberate exception: nested binders are opaque. A `forall`/`exists`/
/// `let`/`match` below the quantifier being analyzed may rebind the same
/// name, so a `Var` found under it is *not* an occurrence of the outer bound
/// variable and terms below it are not valid trigger material for the outer
/// quantifier.
///
/// Every other kind is descended into. The previous per-kind lists ended in
/// a silent catch-all, so array, string, bit-vector, floating-point,
/// datatype, `ite`, `not` and `implies` sub-terms were invisible: a bound
/// variable occurring only under one of them was reported as absent, and the
/// enclosing term was dropped as "ground" — losing the trigger entirely.
fn trigger_children(kind: &TermKind) -> SmallVec<[TermId; 4]> {
    match kind {
        TermKind::Forall { .. }
        | TermKind::Exists { .. }
        | TermKind::Let { .. }
        | TermKind::Match { .. } => SmallVec::new(),
        other => crate::ast::traversal::get_children(other),
    }
}

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

    /// Iterative helper for collecting candidates.
    ///
    /// Pre-order walk with an explicit stack and a visited set (children are
    /// pushed in reverse so they are processed left-to-right, preserving the
    /// candidate order of the recursive form). Terms below a nested binder
    /// are not descended into — see [`trigger_children`].
    fn collect_candidates_recursive(
        &self,
        term: TermId,
        var_names: &FxHashSet<Spur>,
        candidates: &mut Vec<TermId>,
        manager: &TermManager,
    ) -> Result<()> {
        let mut stack = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(t) = manager.get(current) else {
                continue;
            };

            // Skip ground terms: nothing below them can mention a bound
            // variable either.
            if !self.contains_bound_var_quick(current, var_names, manager)? {
                continue;
            }

            // Only an *uninterpreted* head is proposable as a pattern; every
            // other kind is merely traversed through.
            //
            // A trigger is matched against the e-graph, so its head symbol
            // decides how many terms it can fire on.  An interpreted head —
            // `=`, `<`, `<=`, `>`, `>=`, and arithmetic generally — is shared
            // by every atom of the whole problem, so a pattern like the guard
            // `x <= y` of `∀x y. x ≤ y ⇒ f(x) ≤ f(y)` matches *every*
            // inequality in the e-graph, including the ones its own instances
            // introduce.  That is a matching loop: each round instantiates the
            // axiom over every arithmetic atom present and creates more, and
            // the instance set grows without bound.  Z3 rejects such patterns
            // for the same reason (`pattern_inference`: a pattern's head must
            // not be an interpreted symbol); `select` is admitted because it
            // is the array theory's *application* form, keyed by the array
            // term, not a global relation.
            //
            // This restriction used to be masked rather than enforced: the
            // walk that reaches candidates bailed out at `Implies` and
            // `Select` (see `contains_bound_var_quick`), so a comparison under
            // a `⇒` guard — the usual place one appears — was never reached.
            // Now that the walk is exhaustive, the rule has to be stated.
            let proposable = matches!(&t.kind, TermKind::Apply { .. } | TermKind::Select(_, _));
            if proposable && self.is_good_candidate(current, var_names, manager)? {
                candidates.push(current);
            }

            let children = trigger_children(&t.kind);
            stack.extend(children.iter().rev().copied());
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

            // Predicates satisfy the *shape* requirements of a pattern, but an
            // interpreted head makes them match every atom in the e-graph.
            // `collect_candidates_recursive` therefore never offers one — see
            // the head restriction there for why.
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

    /// Quick check if term contains bound variables.
    ///
    /// Iterative with a visited set. This predicate is evaluated at *every*
    /// node of [`Self::collect_candidates_recursive`], so on a hash-consed
    /// DAG the old unmemoized recursion re-expanded shared sub-terms as a
    /// tree — a body built as `x1 = f(x0, x0); x2 = f(x1, x1); …` turned
    /// trigger generation into an exponential hang at assert time. It also
    /// recursed once per level of nesting with no bound.
    fn contains_bound_var_quick(
        &self,
        term: TermId,
        var_names: &FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<bool> {
        let mut stack = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(t) = manager.get(current) else {
                continue;
            };
            if let TermKind::Var(name) = &t.kind {
                if var_names.contains(name) {
                    return Ok(true);
                }
                continue;
            }
            stack.extend(trigger_children(&t.kind));
        }

        Ok(false)
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
        candidate_vars.sort_by_key(|item| std::cmp::Reverse(item.1.len()));

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

    /// Iterative helper for collecting the bound variables a pattern covers.
    ///
    /// Explicit stack plus a visited set, descending through every kind (see
    /// [`trigger_children`]). Under-reporting coverage here made a perfectly
    /// good trigger look like it left variables uncovered, which downgrades
    /// it to [`TriggerQuality::Unusable`] and discards it.
    fn collect_vars_recursive(
        &self,
        term: TermId,
        var_names: &FxHashSet<Spur>,
        covered: &mut FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<()> {
        let mut stack = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(t) = manager.get(current) else {
                continue;
            };
            if let TermKind::Var(name) = &t.kind {
                if var_names.contains(name) {
                    covered.insert(*name);
                }
                continue;
            }
            stack.extend(trigger_children(&t.kind));
        }

        Ok(())
    }

    /// Estimate matching cost of a pattern.
    ///
    /// Iterative bottom-up fold with a memo, so a shared sub-term is
    /// evaluated once instead of once per path (the recursive form was
    /// exponential on a DAG). The cost of each *occurrence* still
    /// contributes, so the value is exactly what the recursion produced;
    /// the accumulation saturates instead of overflowing.
    fn estimate_cost(&self, pattern: TermId, manager: &TermManager) -> Result<u32> {
        let mut memo: FxHashMap<TermId, u32> = FxHashMap::default();
        for current in Self::postorder(pattern, manager) {
            let Some(t) = manager.get(current) else {
                memo.insert(current, 100);
                continue;
            };
            let child_cost = |id: &TermId, memo: &FxHashMap<TermId, u32>| -> u32 {
                memo.get(id).copied().unwrap_or(1)
            };
            let cost = match &t.kind {
                TermKind::Var(_) => 10,
                TermKind::Apply { args, .. } => args
                    .iter()
                    .fold(5u32, |acc, a| acc.saturating_add(child_cost(a, &memo))),
                TermKind::Eq(lhs, rhs)
                | TermKind::Lt(lhs, rhs)
                | TermKind::Le(lhs, rhs)
                | TermKind::Gt(lhs, rhs)
                | TermKind::Ge(lhs, rhs) => 3u32
                    .saturating_add(child_cost(lhs, &memo))
                    .saturating_add(child_cost(rhs, &memo)),
                TermKind::Select(arr, idx) => 4u32
                    .saturating_add(child_cost(arr, &memo))
                    .saturating_add(child_cost(idx, &memo)),
                // Any other kind is charged as an opaque unit, exactly as
                // before: this is a cost heuristic, not an analysis result.
                _ => 1,
            };
            memo.insert(current, cost);
        }

        Ok(memo.get(&pattern).copied().unwrap_or(100))
    }

    /// Compute depth of a term.
    ///
    /// Iterative bottom-up fold with a memo. Depth is now measured through
    /// *every* operand rather than only through function applications,
    /// equalities, `<` and `select`: under-reporting the depth of a term let
    /// arbitrarily deep patterns pass the `max_depth` filter that exists
    /// precisely to reject them.
    fn compute_depth(&self, term: TermId, manager: &TermManager) -> Result<usize> {
        let mut memo: FxHashMap<TermId, usize> = FxHashMap::default();
        for current in Self::postorder(term, manager) {
            let Some(t) = manager.get(current) else {
                memo.insert(current, 0);
                continue;
            };
            let child_depth = trigger_children(&t.kind)
                .iter()
                .map(|c| memo.get(c).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            memo.insert(current, child_depth + 1);
        }

        Ok(memo.get(&term).copied().unwrap_or(0))
    }

    /// Children-before-parents listing of the sub-terms of `root`, computed
    /// with an explicit stack. Shared sub-terms appear exactly once.
    fn postorder(root: TermId, manager: &TermManager) -> Vec<TermId> {
        let mut order = Vec::new();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack = vec![(root, false)];

        while let Some((current, expanded)) = stack.pop() {
            if expanded {
                order.push(current);
                continue;
            }
            if !visited.insert(current) {
                continue;
            }
            stack.push((current, true));
            if let Some(t) = manager.get(current) {
                for child in trigger_children(&t.kind) {
                    if !visited.contains(&child) {
                        stack.push((child, false));
                    }
                }
            }
        }

        order
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
    use crate::interner::Key;

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
            [Spur::try_from_usize(0).expect("test operation should succeed")]
                .iter()
                .copied()
                .collect();

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

        let triggers = generator
            .generate_triggers(forall, &manager)
            .expect("test operation should succeed");

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

        let candidates = generator
            .collect_candidates(body, &vars, &manager)
            .expect("test operation should succeed");

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
        let cost_x = generator
            .estimate_cost(x, &manager)
            .expect("test operation should succeed");

        // Function application should have higher cost
        let f_x = manager.mk_apply("f", [x], int_sort);
        let cost_fx = generator
            .estimate_cost(f_x, &manager)
            .expect("test operation should succeed");

        assert!(cost_fx > cost_x);
    }

    #[test]
    fn test_compute_depth() {
        let mut manager = setup();
        let generator = TriggerGenerator::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        // x: depth 1
        let depth1 = generator
            .compute_depth(x, &manager)
            .expect("test operation should succeed");
        assert_eq!(depth1, 1);

        // f(x): depth 2
        let f_x = manager.mk_apply("f", [x], int_sort);
        let depth2 = generator
            .compute_depth(f_x, &manager)
            .expect("test operation should succeed");
        assert_eq!(depth2, 2);

        // g(f(x)): depth 3
        let g_fx = manager.mk_apply("g", [f_x], int_sort);
        let depth3 = generator
            .compute_depth(g_fx, &manager)
            .expect("test operation should succeed");
        assert_eq!(depth3, 3);
    }
}

#[cfg(test)]
mod deep_walk_tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_bound_var_seen_under_array_select() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let arr = manager.mk_var("a", array_sort);
        let x = manager.mk_var("x", int_sort);
        let sel = manager.mk_select(arr, x);

        let x_name = manager.intern_str("x");
        let var_names: FxHashSet<Spur> = [x_name].into_iter().collect();
        let generator = TriggerGenerator::new_default();

        assert!(
            generator
                .contains_bound_var_quick(sel, &var_names, &manager)
                .expect("walk should succeed"),
            "bound variable below `select` was reported as absent"
        );

        let mut covered = FxHashSet::default();
        generator
            .collect_vars_recursive(sel, &var_names, &mut covered, &manager)
            .expect("walk should succeed");
        assert!(covered.contains(&x_name));
    }

    /// A pattern's head must be uninterpreted.  For `∀x y. x ≤ y ⇒ f(x) ≤ f(y)`
    /// the only admissible candidates are `f(x)` and `f(y)`: the guard `x ≤ y`
    /// and the consequent `f(x) ≤ f(y)` are headed by `≤`, an interpreted
    /// relation shared by every arithmetic atom in the problem, so using either
    /// as a trigger makes the axiom fire on every inequality in the e-graph —
    /// including the ones its own instances add.  That matching loop turned the
    /// UFLIA / UFLRA / AUFLIA `sat`-certification benchmarks from a
    /// millisecond `sat` into an unbounded instantiation run.
    #[test]
    fn test_interpreted_heads_are_not_trigger_candidates() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let fy = manager.mk_apply("f", [y], int_sort);
        let guard = manager.mk_le(x, y);
        let consequent = manager.mk_le(fx, fy);
        let body = manager.mk_implies(guard, consequent);

        let x_name = manager.intern_str("x");
        let y_name = manager.intern_str("y");
        let vars = [(x_name, int_sort), (y_name, int_sort)];

        let generator = TriggerGenerator::new_default();
        let candidates = generator
            .collect_candidates(body, &vars, &manager)
            .expect("candidate collection should succeed");

        assert!(
            candidates.contains(&fx) && candidates.contains(&fy),
            "the uninterpreted applications must still be offered: {candidates:?}"
        );
        assert!(
            !candidates.contains(&guard) && !candidates.contains(&consequent),
            "a comparison must never be offered as a trigger head: {candidates:?}"
        );
    }

    #[test]
    fn test_contains_bound_var_shared_dag_is_fast() {
        // 55 doubling levels: exponential without a visited set.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let y = manager.mk_var("y", int_sort);
        let mut level = manager.mk_apply("f", [y, y], int_sort);
        for _ in 0..55 {
            level = manager.mk_apply("f", [level, level], int_sort);
        }

        let x_name = manager.intern_str("x");
        let var_names: FxHashSet<Spur> = [x_name].into_iter().collect();
        let generator = TriggerGenerator::new_default();
        assert!(
            !generator
                .contains_bound_var_quick(level, &var_names, &manager)
                .expect("walk should succeed")
        );
    }

    #[test]
    fn test_trigger_walks_deep_nesting_do_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let x = manager.mk_var("x", int_sort);
                let mut term = x;
                for _ in 0..7_500 {
                    term = manager.mk_apply("f", [term], int_sort);
                }

                let x_name = manager.intern_str("x");
                let var_names: FxHashSet<Spur> = [x_name].into_iter().collect();
                let generator = TriggerGenerator::new_default();

                let contains = generator
                    .contains_bound_var_quick(term, &var_names, &manager)
                    .expect("walk should succeed");
                let depth = generator
                    .compute_depth(term, &manager)
                    .expect("walk should succeed");
                let cost = generator
                    .estimate_cost(term, &manager)
                    .expect("walk should succeed");
                (contains, depth, cost)
            })
            .expect("thread spawn should succeed");

        let (contains, depth, cost) = handle.join().expect("deep walks must not overflow");
        assert!(contains);
        assert_eq!(depth, 7_501);
        assert!(cost > 0);
    }
}
