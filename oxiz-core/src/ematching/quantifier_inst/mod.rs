//! Quantifier instantiation engine for E-matching
//!
//! This module provides the core quantifier instantiation engine that matches
//! trigger patterns against ground terms in the term pool and produces
//! instantiation lemmas.
//!
//! Pattern/candidate matching itself lives in the private `matcher`
//! submodule, which replaced a native lockstep recursion with an explicit
//! backtracking goal machine.

mod matcher;

use crate::ast::{TermId, TermKind, TermManager};
use crate::ematching::substitution::Substitution;
use crate::ematching::trigger::Trigger;
use crate::error::Result;
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;

use matcher::match_term;

/// Configuration for the E-matching engine
#[derive(Debug, Clone)]
pub struct EmatchConfig {
    /// Maximum number of instantiations per matching round
    pub max_instances_per_round: usize,
    /// Maximum total number of instantiations allowed
    pub max_total_instances: usize,
    /// Whether to use modification time heuristics
    pub use_mod_time: bool,
    /// Whether to use relevancy filtering
    pub use_relevancy: bool,
}

impl Default for EmatchConfig {
    fn default() -> Self {
        Self {
            max_instances_per_round: 1000,
            max_total_instances: 100000,
            use_mod_time: true,
            use_relevancy: true,
        }
    }
}

/// Statistics for E-matching
#[derive(Debug, Clone, Default)]
pub struct EmatchStats {
    /// Number of matching rounds executed
    pub rounds: usize,
    /// Total number of instantiations created
    pub total_instantiations: usize,
    /// Number of instantiations in the last round
    pub last_round_instantiations: usize,
    /// Number of candidate terms considered
    pub candidates_considered: usize,
    /// Number of match attempts
    pub match_attempts: usize,
    /// Number of successful matches
    pub successful_matches: usize,
    /// Number of deduplicated (skipped) matches
    pub deduplicated: usize,
}

/// E-matching engine
#[derive(Debug)]
pub struct EmatchEngine {
    config: EmatchConfig,
    quantifiers: Vec<QuantifierInfo>,
    cache: InstantiationCache,
    stats: EmatchStats,
}

/// Information about a quantifier
#[derive(Debug, Clone)]
pub struct QuantifierInfo {
    /// The term ID of the quantifier
    pub quant_id: TermId,
    /// Triggers for this quantifier
    pub triggers: Vec<Trigger>,
}

/// Cache for instantiations to prevent redundant work
#[derive(Debug, Default)]
pub struct InstantiationCache {
    /// Per-quantifier deduplication: quant_id -> set of substitution keys
    cache: FxHashMap<TermId, FxHashSet<Vec<(Spur, TermId)>>>,
}

impl InstantiationCache {
    /// Check if a substitution was already generated for a quantifier.
    /// Returns true if this is a new (unseen) substitution.
    fn insert_if_new(&mut self, quant_id: TermId, subst: &Substitution) -> bool {
        let mut key: Vec<(Spur, TermId)> = subst.iter().map(|(&k, &v)| (k, v)).collect();
        key.sort_by_key(|(k, _)| k.into_inner());
        let seen = self.cache.entry(quant_id).or_default();
        seen.insert(key)
    }

    fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Context for instantiation
#[derive(Debug)]
pub struct InstantiationContext {
    /// Current matching round number
    pub round: usize,
}

/// Result of instantiation
#[derive(Debug, Clone)]
pub struct InstantiationResult {
    /// The instantiated terms created
    pub instances: Vec<TermId>,
}

impl EmatchEngine {
    /// Create a new E-matching engine with the given configuration
    pub fn new(config: EmatchConfig) -> Self {
        Self {
            config,
            quantifiers: Vec::new(),
            cache: InstantiationCache::default(),
            stats: EmatchStats::default(),
        }
    }

    /// Register a quantifier with its triggers
    pub fn register_quantifier(
        &mut self,
        quant_id: TermId,
        triggers: Vec<Trigger>,
        _manager: &mut TermManager,
    ) -> Result<()> {
        self.quantifiers.push(QuantifierInfo { quant_id, triggers });
        Ok(())
    }

    /// Number of quantifiers currently registered.
    ///
    /// Registration is append-only, so this doubles as a restore point for an
    /// incremental solver: record it before entering an assertion scope and
    /// hand it back to [`EmatchEngine::truncate_quantifiers`] when the scope is
    /// popped.
    #[must_use]
    pub fn num_quantifiers(&self) -> usize {
        self.quantifiers.len()
    }

    /// Drop every quantifier registered after the `len`-th one.
    ///
    /// The instantiation dedup cache is cleared as well: its entries record
    /// which lemmas have already been emitted, and those lemmas are retracted
    /// together with the scope that produced them.  Keeping the cache would
    /// suppress a lemma that a later scope still needs (an unsound
    /// under-approximation); clearing it only costs re-derivation.
    pub fn truncate_quantifiers(&mut self, len: usize) {
        if len >= self.quantifiers.len() {
            return;
        }
        self.quantifiers.truncate(len);
        self.cache.clear();
    }

    /// Perform one round of E-matching and return instantiated terms.
    ///
    /// For each registered quantifier, iterates over its trigger patterns and
    /// attempts to match them against ground terms in the TermManager. Successful
    /// matches produce substitutions which are applied to the quantifier body to
    /// create instantiation lemmas.
    pub fn match_round(&mut self, manager: &mut TermManager) -> Result<Vec<TermId>> {
        self.stats.rounds += 1;
        let mut results = Vec::new();
        let max_this_round = self.config.max_instances_per_round;
        let max_total = self.config.max_total_instances;

        // Collect all term IDs upfront to avoid borrow conflicts
        let num_terms = manager.len();
        let all_term_ids: Vec<TermId> = (0..num_terms).map(|i| TermId::new(i as u32)).collect();

        // Clone quantifier info to avoid borrow conflict with self
        let quantifiers: Vec<QuantifierInfo> = self.quantifiers.clone();

        for quant_info in &quantifiers {
            if results.len() >= max_this_round || self.stats.total_instantiations >= max_total {
                break;
            }

            // Extract quantifier body and bound variables
            let (vars, body) = {
                let Some(quant_term) = manager.get(quant_info.quant_id) else {
                    continue;
                };
                match &quant_term.kind {
                    TermKind::Forall { vars, body, .. } | TermKind::Exists { vars, body, .. } => {
                        (vars.clone(), *body)
                    }
                    _ => continue,
                }
            };

            for trigger in &quant_info.triggers {
                if results.len() >= max_this_round || self.stats.total_instantiations >= max_total {
                    break;
                }

                // For single-pattern triggers, match directly.
                // For multi-pattern triggers, all patterns must match simultaneously.
                let matches = if trigger.patterns.len() == 1 {
                    self.match_single_pattern(trigger.patterns[0], &vars, &all_term_ids, manager)
                } else {
                    self.match_multi_pattern(&trigger.patterns, &vars, &all_term_ids, manager)
                };

                for subst in matches {
                    if results.len() >= max_this_round
                        || self.stats.total_instantiations >= max_total
                    {
                        break;
                    }

                    self.stats.successful_matches += 1;

                    // Deduplication
                    if !self.cache.insert_if_new(quant_info.quant_id, &subst) {
                        self.stats.deduplicated += 1;
                        continue;
                    }

                    // Apply substitution to the quantifier body
                    let instance = subst.apply(body, manager)?;
                    results.push(instance);
                    self.stats.total_instantiations += 1;
                }
            }
        }

        self.stats.last_round_instantiations = results.len();
        Ok(results)
    }

    /// Match a single pattern term against all ground terms in the term pool.
    fn match_single_pattern(
        &mut self,
        pattern: TermId,
        bound_vars: &[(Spur, crate::sort::SortId)],
        all_term_ids: &[TermId],
        manager: &TermManager,
    ) -> Vec<Substitution> {
        let bound_var_names: FxHashSet<Spur> = bound_vars.iter().map(|(name, _)| *name).collect();

        let mut results = Vec::new();

        let Some(pattern_term) = manager.get(pattern) else {
            return results;
        };
        let pattern_top = TopLevelShape::from_term_kind(&pattern_term.kind);

        for &candidate_id in all_term_ids {
            let Some(candidate_term) = manager.get(candidate_id) else {
                continue;
            };
            let candidate_top = TopLevelShape::from_term_kind(&candidate_term.kind);
            if !pattern_top.compatible_with(&candidate_top) {
                continue;
            }

            self.stats.candidates_considered += 1;
            self.stats.match_attempts += 1;

            let mut subst = Substitution::new();
            if match_term(pattern, candidate_id, &bound_var_names, &mut subst, manager)
                && bound_vars.iter().all(|(name, _)| subst.contains(name))
            {
                results.push(subst);
            }
        }

        results
    }

    /// Match multiple patterns simultaneously with a consistent substitution.
    fn match_multi_pattern(
        &mut self,
        patterns: &[TermId],
        bound_vars: &[(Spur, crate::sort::SortId)],
        all_term_ids: &[TermId],
        manager: &TermManager,
    ) -> Vec<Substitution> {
        if patterns.is_empty() {
            return Vec::new();
        }

        let first_matches =
            self.match_single_pattern(patterns[0], bound_vars, all_term_ids, manager);

        if patterns.len() == 1 {
            return first_matches;
        }

        let bound_var_names: FxHashSet<Spur> = bound_vars.iter().map(|(name, _)| *name).collect();

        let mut current_results = first_matches;

        for &pat in &patterns[1..] {
            let mut next_results = Vec::new();

            for existing_subst in &current_results {
                let Some(pattern_term) = manager.get(pat) else {
                    continue;
                };
                let pattern_top = TopLevelShape::from_term_kind(&pattern_term.kind);

                for &candidate_id in all_term_ids {
                    let Some(candidate_term) = manager.get(candidate_id) else {
                        continue;
                    };
                    let candidate_top = TopLevelShape::from_term_kind(&candidate_term.kind);
                    if !pattern_top.compatible_with(&candidate_top) {
                        continue;
                    }

                    let mut extended_subst = existing_subst.clone();
                    if match_term(
                        pat,
                        candidate_id,
                        &bound_var_names,
                        &mut extended_subst,
                        manager,
                    ) {
                        next_results.push(extended_subst);
                    }
                }
            }

            current_results = next_results;
        }

        current_results
            .into_iter()
            .filter(|subst| bound_vars.iter().all(|(name, _)| subst.contains(name)))
            .collect()
    }

    /// Get statistics for this engine
    pub fn stats(&self) -> &EmatchStats {
        &self.stats
    }

    /// Reset the engine to its initial state
    pub fn reset(&mut self) {
        self.quantifiers.clear();
        self.cache.clear();
        self.stats = EmatchStats::default();
    }
}

/// Top-level shape of a term, used for fast pre-filtering of candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TopLevelShape {
    Apply { func: Spur, arity: usize },
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    Add(usize),
    Mul(usize),
    And(usize),
    Or(usize),
    Not,
    Neg,
    Implies,
    Ite,
    Select,
    Store,
    Sub,
    Div,
    Variable,
    Other,
}

impl TopLevelShape {
    fn from_term_kind(kind: &TermKind) -> Self {
        match kind {
            TermKind::Var(_) => Self::Variable,
            TermKind::Apply { func, args } => Self::Apply {
                func: *func,
                arity: args.len(),
            },
            TermKind::Eq(_, _) => Self::Eq,
            TermKind::Lt(_, _) => Self::Lt,
            TermKind::Le(_, _) => Self::Le,
            TermKind::Gt(_, _) => Self::Gt,
            TermKind::Ge(_, _) => Self::Ge,
            TermKind::Add(args) => Self::Add(args.len()),
            TermKind::Mul(args) => Self::Mul(args.len()),
            TermKind::And(args) => Self::And(args.len()),
            TermKind::Or(args) => Self::Or(args.len()),
            TermKind::Not(_) => Self::Not,
            TermKind::Neg(_) => Self::Neg,
            TermKind::Implies(_, _) => Self::Implies,
            TermKind::Ite(_, _, _) => Self::Ite,
            TermKind::Select(_, _) => Self::Select,
            TermKind::Store(_, _, _) => Self::Store,
            TermKind::Sub(_, _) => Self::Sub,
            TermKind::Div(_, _) => Self::Div,
            _ => Self::Other,
        }
    }

    /// A Variable pattern is compatible with anything.
    fn compatible_with(&self, candidate: &Self) -> bool {
        if matches!(self, Self::Variable) {
            return true;
        }
        self == candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_config_default() {
        let config = EmatchConfig::default();
        assert_eq!(config.max_instances_per_round, 1000);
    }

    #[test]
    fn test_match_round_no_quantifiers() {
        let mut engine = EmatchEngine::new(EmatchConfig::default());
        let mut manager = TermManager::new();
        let result = engine.match_round(&mut manager);
        assert!(result.is_ok());
        assert!(result.is_ok_and(|v| v.is_empty()));
    }

    #[test]
    fn test_match_round_simple_pattern() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        // Create ground terms: f(a) and f(b)
        let a = manager.mk_apply("a", std::iter::empty::<TermId>(), int_sort);
        let b = manager.mk_apply("b", std::iter::empty::<TermId>(), int_sort);
        let f_a = manager.mk_apply("f", [a], int_sort);
        let _f_b = manager.mk_apply("f", [b], int_sort);

        // Create pattern: f(x) where x is bound
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let p_fx = manager.mk_apply("P", [f_x], bool_sort);

        let x_name = manager.intern_str("x");
        let covered_vars: FxHashSet<Spur> = [x_name].into_iter().collect();
        let trigger = Trigger {
            patterns: smallvec::smallvec![f_x],
            quality: crate::ematching::trigger::TriggerQuality::Excellent,
            cost: 15,
            covered_vars,
        };

        let patterns: Vec<smallvec::SmallVec<[TermId; 2]>> = vec![smallvec::smallvec![f_x]];
        let forall = manager.mk_forall_with_patterns(vec![("x", int_sort)], p_fx, patterns);

        let mut engine = EmatchEngine::new(EmatchConfig::default());
        engine
            .register_quantifier(forall, vec![trigger], &mut manager)
            .expect("register should succeed");

        let result = engine
            .match_round(&mut manager)
            .expect("match_round should succeed");

        // Should find matches for f(a), f(b), and f(x) itself (pattern term is also
        // in the TermManager, so x binds to the var term x as well). In a full solver,
        // pattern terms would be excluded from the ground term pool; here we accept
        // all syntactic matches.
        assert!(
            result.len() >= 2,
            "expected at least 2 matches, got {}",
            result.len()
        );
        let _ = f_a; // used above
    }

    #[test]
    fn test_deduplication() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let a = manager.mk_apply("a", std::iter::empty::<TermId>(), int_sort);
        let _f_a = manager.mk_apply("f", [a], int_sort);

        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let p_fx = manager.mk_apply("P", [f_x], bool_sort);

        let x_name = manager.intern_str("x");
        let covered_vars: FxHashSet<Spur> = [x_name].into_iter().collect();
        let trigger = Trigger {
            patterns: smallvec::smallvec![f_x],
            quality: crate::ematching::trigger::TriggerQuality::Excellent,
            cost: 15,
            covered_vars,
        };

        let patterns: Vec<smallvec::SmallVec<[TermId; 2]>> = vec![smallvec::smallvec![f_x]];
        let forall = manager.mk_forall_with_patterns(vec![("x", int_sort)], p_fx, patterns);

        let mut engine = EmatchEngine::new(EmatchConfig::default());
        engine
            .register_quantifier(forall, vec![trigger], &mut manager)
            .expect("register should succeed");

        let result1 = engine.match_round(&mut manager).expect("round 1");
        let count1 = result1.len();
        assert!(count1 > 0);

        // Second round should produce 0 new results due to dedup
        let result2 = engine.match_round(&mut manager).expect("round 2");
        assert_eq!(
            result2.len(),
            0,
            "second round should produce no new instances due to dedup"
        );
        assert!(engine.stats().deduplicated > 0);
    }

    #[test]
    fn test_instantiation_cache() {
        let mut manager = TermManager::new();
        let mut cache = InstantiationCache::default();

        let x_name = manager.intern_str("x");
        let five = manager.mk_int(5);
        let quant_id = TermId::new(100);

        let mut subst = Substitution::new();
        subst.insert(x_name, five);

        assert!(cache.insert_if_new(quant_id, &subst));
        assert!(!cache.insert_if_new(quant_id, &subst));
    }

    #[test]
    fn test_match_term_variable_binding() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        let bound_vars: FxHashSet<Spur> = [x_name].into_iter().collect();

        let mut subst = Substitution::new();
        assert!(match_term(x, five, &bound_vars, &mut subst, &manager));
        assert_eq!(subst.get(&x_name), Some(five));
    }

    #[test]
    fn test_match_term_consistent_binding() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Pattern f(x, x) -- x must bind to the same thing both times
        let x = manager.mk_var("x", int_sort);
        let f_xx = manager.mk_apply("f", [x, x], int_sort);

        let a = manager.mk_apply("a", std::iter::empty::<TermId>(), int_sort);
        let b = manager.mk_apply("b", std::iter::empty::<TermId>(), int_sort);

        // f(a, a) should match
        let f_aa = manager.mk_apply("f", [a, a], int_sort);
        // f(a, b) should not match (x can't be both a and b)
        let f_ab = manager.mk_apply("f", [a, b], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars: FxHashSet<Spur> = [x_name].into_iter().collect();

        let mut subst1 = Substitution::new();
        assert!(match_term(f_xx, f_aa, &bound_vars, &mut subst1, &manager));

        let mut subst2 = Substitution::new();
        assert!(!match_term(f_xx, f_ab, &bound_vars, &mut subst2, &manager));
    }

    #[test]
    fn test_nested_pattern_matching() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Pattern: f(g(x))
        let x = manager.mk_var("x", int_sort);
        let g_x = manager.mk_apply("g", [x], int_sort);
        let f_gx = manager.mk_apply("f", [g_x], int_sort);

        // Ground: f(g(5))
        let five = manager.mk_int(5);
        let g_5 = manager.mk_apply("g", [five], int_sort);
        let f_g5 = manager.mk_apply("f", [g_5], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars: FxHashSet<Spur> = [x_name].into_iter().collect();

        let mut subst = Substitution::new();
        assert!(match_term(f_gx, f_g5, &bound_vars, &mut subst, &manager));
        assert_eq!(subst.get(&x_name), Some(five));
    }

    #[test]
    fn test_top_level_shape_filtering() {
        let s1 = TopLevelShape::Eq;
        let s2 = TopLevelShape::Lt;
        let s3 = TopLevelShape::Eq;
        let sv = TopLevelShape::Variable;

        assert!(s1.compatible_with(&s3));
        assert!(!s1.compatible_with(&s2));
        assert!(sv.compatible_with(&s1)); // variable matches anything
    }

    // ===== Regression pins for the explicit-machine rewrite =====

    /// A pattern and a candidate far deeper than any native stack could
    /// carry: the retired lockstep recursion spent one frame per level and
    /// aborted the process here.
    #[test]
    fn test_match_term_deeply_nested() {
        let worker = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;

                let depth = 6_250;

                // Pattern: g(g(...g(x)...))
                let x = manager.mk_var("x", int_sort);
                let mut pattern = x;
                for _ in 0..depth {
                    pattern = manager.mk_apply("g", [pattern], int_sort);
                }

                // Candidate: g(g(...g(5)...))
                let five = manager.mk_int(5);
                let mut candidate = five;
                for _ in 0..depth {
                    candidate = manager.mk_apply("g", [candidate], int_sort);
                }

                let x_name = manager.intern_str("x");
                let bound_vars: FxHashSet<Spur> = [x_name].into_iter().collect();

                let mut subst = Substitution::new();
                assert!(match_term(
                    pattern,
                    candidate,
                    &bound_vars,
                    &mut subst,
                    &manager
                ));
                assert_eq!(subst.get(&x_name), Some(five));
            })
            .expect("spawning the deep-match worker should succeed");

        worker.join().expect("deep matching must not overflow");
    }

    /// A failed match must hand the caller's substitution back untouched.
    /// The retired recursion threaded one `&mut Substitution` through its
    /// `&&` chains and left the failed branch's partial bindings behind.
    #[test]
    fn test_match_term_failure_leaves_no_bindings() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let a = manager.mk_apply("a", std::iter::empty::<TermId>(), int_sort);
        let b = manager.mk_apply("b", std::iter::empty::<TermId>(), int_sort);
        let c = manager.mk_apply("c", std::iter::empty::<TermId>(), int_sort);

        // Pattern f(x, a) vs candidate f(b, c): x binds to b, then a vs c
        // refutes the match.
        let pattern = manager.mk_apply("f", [x, a], int_sort);
        let candidate = manager.mk_apply("f", [b, c], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars: FxHashSet<Spur> = [x_name].into_iter().collect();

        let mut subst = Substitution::new();
        assert!(!match_term(
            pattern,
            candidate,
            &bound_vars,
            &mut subst,
            &manager
        ));
        assert!(
            subst.is_empty(),
            "a failed match must not leak the bindings it made: {subst}"
        );
    }

    /// Bindings the caller supplied before the match (e.g. carried over from
    /// an earlier pattern of a multi-pattern trigger) must survive a failure
    /// unchanged -- the undo trail only removes what this match added.
    #[test]
    fn test_match_term_failure_preserves_caller_bindings() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let a = manager.mk_apply("a", std::iter::empty::<TermId>(), int_sort);
        let b = manager.mk_apply("b", std::iter::empty::<TermId>(), int_sort);
        let c = manager.mk_apply("c", std::iter::empty::<TermId>(), int_sort);

        let pattern = manager.mk_apply("f", [x, a], int_sort);
        let candidate = manager.mk_apply("f", [b, c], int_sort);

        let x_name = manager.intern_str("x");
        let y_name = manager.intern_str("y");
        let bound_vars: FxHashSet<Spur> = [x_name, y_name].into_iter().collect();

        let mut subst = Substitution::new();
        subst.insert(y_name, a);

        assert!(!match_term(
            pattern,
            candidate,
            &bound_vars,
            &mut subst,
            &manager
        ));
        assert_eq!(subst.get(&y_name), Some(a));
        assert_eq!(subst.len(), 1);
    }

    /// Backtracking across siblings: the equality's written orientation
    /// succeeds locally but conflicts with a goal *outside* the equality
    /// node, and the machine must rewind into the swapped orientation.
    ///
    /// Pattern `f((= x y), x)` against candidate `f((= p q), q)`:
    /// orientation 1 binds `x := p, y := q`, and then the sibling goal
    /// `x` vs `q` fails; the alternative binds `x := q, y := p`, under which
    /// the sibling succeeds. The retired recursion tried only orientation 1
    /// and reported no match.
    #[test]
    fn test_match_term_backtracks_across_siblings() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        // Created in this order so that `mk_eq`'s TermId-order
        // canonicalisation keeps the operands as written.
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let p = manager.mk_apply("p", std::iter::empty::<TermId>(), int_sort);
        let q = manager.mk_apply("q", std::iter::empty::<TermId>(), int_sort);

        let eq_xy = manager.mk_eq(x, y);
        let eq_pq = manager.mk_eq(p, q);
        let pattern = manager.mk_apply("f", [eq_xy, x], bool_sort);
        let candidate = manager.mk_apply("f", [eq_pq, q], bool_sort);

        let x_name = manager.intern_str("x");
        let y_name = manager.intern_str("y");
        let bound_vars: FxHashSet<Spur> = [x_name, y_name].into_iter().collect();

        let mut subst = Substitution::new();
        assert!(
            match_term(pattern, candidate, &bound_vars, &mut subst, &manager),
            "the swapped equality orientation must be reachable by backtracking"
        );
        assert_eq!(subst.get(&x_name), Some(q));
        assert_eq!(subst.get(&y_name), Some(p));
    }

    /// Equality is symmetric, so a trigger written one way round matches a
    /// ground equality written the other way round.
    #[test]
    fn test_match_term_eq_orientation() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let a = manager.mk_apply("a", std::iter::empty::<TermId>(), int_sort);
        let b = manager.mk_apply("b", std::iter::empty::<TermId>(), int_sort);

        let pattern = manager.mk_eq(x, a);
        let candidate = manager.mk_eq(a, b);

        let x_name = manager.intern_str("x");
        let bound_vars: FxHashSet<Spur> = [x_name].into_iter().collect();

        let mut subst = Substitution::new();
        assert!(match_term(
            pattern,
            candidate,
            &bound_vars,
            &mut subst,
            &manager
        ));
        assert_eq!(subst.get(&x_name), Some(b));
    }

    /// A shared subterm appearing under two different parents is matched
    /// independently in each position, with the binding made in the first
    /// constraining the second.
    #[test]
    fn test_match_term_shared_subpattern_stays_consistent() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let g_x = manager.mk_apply("g", [x], int_sort);
        let pattern = manager.mk_apply("f", [g_x, g_x], int_sort);

        let a = manager.mk_apply("a", std::iter::empty::<TermId>(), int_sort);
        let b = manager.mk_apply("b", std::iter::empty::<TermId>(), int_sort);
        let g_a = manager.mk_apply("g", [a], int_sort);
        let g_b = manager.mk_apply("g", [b], int_sort);

        let matching = manager.mk_apply("f", [g_a, g_a], int_sort);
        let conflicting = manager.mk_apply("f", [g_a, g_b], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars: FxHashSet<Spur> = [x_name].into_iter().collect();

        let mut subst = Substitution::new();
        assert!(match_term(
            pattern,
            matching,
            &bound_vars,
            &mut subst,
            &manager
        ));
        assert_eq!(subst.get(&x_name), Some(a));

        let mut clash = Substitution::new();
        assert!(!match_term(
            pattern,
            conflicting,
            &bound_vars,
            &mut clash,
            &manager
        ));
        assert!(clash.is_empty());
    }
}
