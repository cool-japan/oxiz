//! Lazy term evaluation and demand-driven simplification.
//!
//! Provides lazy evaluation strategies for terms, only computing
//! simplifications when they are actually needed (e.g., in conflicts
//! or propagation). This avoids wasted work on terms that are never
//! queried.

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;

/// Evaluation state for a lazily evaluated term
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalState {
    /// Not yet evaluated
    Pending,
    /// Currently being evaluated (cycle detection)
    InProgress,
    /// Evaluation complete
    Complete,
}

/// A lazy evaluator that only simplifies terms when they are actually needed.
///
/// This implements demand-driven evaluation: terms are only simplified
/// when their value is explicitly requested, and results are cached
/// to avoid redundant computation.
#[derive(Debug)]
pub struct LazyEvaluator {
    /// Cache of simplified term results
    cache: FxHashMap<TermId, TermId>,
    /// Evaluation state for cycle detection
    state: FxHashMap<TermId, EvalState>,
    /// Set of terms that are known to be relevant (used in conflicts/propagation)
    relevant_terms: FxHashSet<TermId>,
    /// Statistics: total evaluations performed
    eval_count: u64,
    /// Statistics: cache hits
    cache_hits: u64,
    /// Statistics: terms skipped (not relevant)
    skipped_count: u64,
}

impl LazyEvaluator {
    /// Create a new lazy evaluator
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            state: FxHashMap::default(),
            relevant_terms: FxHashSet::default(),
            eval_count: 0,
            cache_hits: 0,
            skipped_count: 0,
        }
    }

    /// Mark a term as relevant (should be evaluated when requested)
    pub fn mark_relevant(&mut self, term: TermId) {
        self.relevant_terms.insert(term);
    }

    /// Mark multiple terms as relevant
    pub fn mark_relevant_batch(&mut self, terms: &[TermId]) {
        for &term in terms {
            self.relevant_terms.insert(term);
        }
    }

    /// Check if a term is marked as relevant
    #[must_use]
    pub fn is_relevant(&self, term: TermId) -> bool {
        self.relevant_terms.contains(&term)
    }

    /// Lazily evaluate/simplify a term.
    ///
    /// If the term is not marked as relevant, returns the term unchanged
    /// (demand-driven optimization). If the term has been evaluated before,
    /// returns the cached result.
    pub fn eval(&mut self, term: TermId, manager: &mut TermManager) -> TermId {
        // Check cache first
        if let Some(&cached) = self.cache.get(&term) {
            self.cache_hits += 1;
            return cached;
        }

        // If not relevant, skip evaluation
        if !self.relevant_terms.contains(&term) {
            self.skipped_count += 1;
            return term;
        }

        // Check for cycles
        if self.state.get(&term) == Some(&EvalState::InProgress) {
            // Cycle detected, return term as-is
            return term;
        }

        self.state.insert(term, EvalState::InProgress);
        self.eval_count += 1;

        let result = self.simplify_term(term, manager);

        self.state.insert(term, EvalState::Complete);
        self.cache.insert(term, result);

        result
    }

    /// Force evaluation of a term regardless of relevance
    pub fn eval_forced(&mut self, term: TermId, manager: &mut TermManager) -> TermId {
        if let Some(&cached) = self.cache.get(&term) {
            self.cache_hits += 1;
            return cached;
        }

        if self.state.get(&term) == Some(&EvalState::InProgress) {
            return term;
        }

        self.state.insert(term, EvalState::InProgress);
        self.eval_count += 1;

        let result = self.simplify_term(term, manager);

        self.state.insert(term, EvalState::Complete);
        self.cache.insert(term, result);

        result
    }

    /// Simplify a single term
    fn simplify_term(&mut self, term: TermId, manager: &mut TermManager) -> TermId {
        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        match t.kind {
            // Constants are already simplified
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::Var(_) => term,

            // Boolean simplifications
            TermKind::Not(arg) => {
                let simplified_arg = self.eval(arg, manager);
                if let Some(at) = manager.get(simplified_arg) {
                    match at.kind {
                        TermKind::True => return manager.mk_false(),
                        TermKind::False => return manager.mk_true(),
                        TermKind::Not(inner) => return inner,
                        _ => {}
                    }
                }
                manager.mk_not(simplified_arg)
            }

            TermKind::And(ref args) => {
                let mut simplified = Vec::with_capacity(args.len());
                for &arg in args {
                    let s = self.eval(arg, manager);
                    if let Some(at) = manager.get(s) {
                        match at.kind {
                            TermKind::False => return manager.mk_false(),
                            TermKind::True => continue,
                            _ => simplified.push(s),
                        }
                    } else {
                        simplified.push(s);
                    }
                }
                match simplified.len() {
                    0 => manager.mk_true(),
                    1 => simplified[0],
                    _ => manager.mk_and(simplified),
                }
            }

            TermKind::Or(ref args) => {
                let mut simplified = Vec::with_capacity(args.len());
                for &arg in args {
                    let s = self.eval(arg, manager);
                    if let Some(at) = manager.get(s) {
                        match at.kind {
                            TermKind::True => return manager.mk_true(),
                            TermKind::False => continue,
                            _ => simplified.push(s),
                        }
                    } else {
                        simplified.push(s);
                    }
                }
                match simplified.len() {
                    0 => manager.mk_false(),
                    1 => simplified[0],
                    _ => manager.mk_or(simplified),
                }
            }

            TermKind::Eq(lhs, rhs) => {
                let sl = self.eval(lhs, manager);
                let sr = self.eval(rhs, manager);
                if sl == sr {
                    return manager.mk_true();
                }
                manager.mk_eq(sl, sr)
            }

            TermKind::Ite(cond, then_br, else_br) => {
                let sc = self.eval(cond, manager);
                if let Some(ct) = manager.get(sc) {
                    match ct.kind {
                        TermKind::True => return self.eval(then_br, manager),
                        TermKind::False => return self.eval(else_br, manager),
                        _ => {}
                    }
                }
                let st = self.eval(then_br, manager);
                let se = self.eval(else_br, manager);
                manager.mk_ite(sc, st, se)
            }

            // Default: return as-is
            _ => term,
        }
    }

    /// Invalidate the entire cache
    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.state.clear();
    }

    /// Invalidate a specific term
    pub fn invalidate_term(&mut self, term: TermId) {
        self.cache.remove(&term);
        self.state.remove(&term);
    }

    /// Clear relevance markers
    pub fn clear_relevant(&mut self) {
        self.relevant_terms.clear();
    }

    /// Get statistics: (evaluations, cache_hits, skipped)
    #[must_use]
    pub fn statistics(&self) -> (u64, u64, u64) {
        (self.eval_count, self.cache_hits, self.skipped_count)
    }

    /// Get the number of cached results
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get the number of relevant terms
    #[must_use]
    pub fn relevant_count(&self) -> usize {
        self.relevant_terms.len()
    }
}

impl Default for LazyEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_evaluator_new() {
        let evaluator = LazyEvaluator::new();
        assert_eq!(evaluator.cache_size(), 0);
        assert_eq!(evaluator.relevant_count(), 0);
        assert_eq!(evaluator.statistics(), (0, 0, 0));
    }

    #[test]
    fn test_lazy_evaluator_mark_relevant() {
        let mut evaluator = LazyEvaluator::new();
        let t1 = TermId::new(10);
        let t2 = TermId::new(20);

        evaluator.mark_relevant(t1);
        assert!(evaluator.is_relevant(t1));
        assert!(!evaluator.is_relevant(t2));

        evaluator.mark_relevant_batch(&[t1, t2]);
        assert!(evaluator.is_relevant(t2));
        assert_eq!(evaluator.relevant_count(), 2);
    }

    #[test]
    fn test_lazy_evaluator_skip_irrelevant() {
        let mut evaluator = LazyEvaluator::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        // p is not marked relevant, so eval should skip
        let result = evaluator.eval(p, &mut manager);
        assert_eq!(result, p);
        assert_eq!(evaluator.statistics().2, 1); // 1 skipped
    }

    #[test]
    fn test_lazy_evaluator_simplify_constant() {
        let mut evaluator = LazyEvaluator::new();
        let mut manager = TermManager::new();

        let t = manager.mk_true();
        evaluator.mark_relevant(t);

        let result = evaluator.eval(t, &mut manager);
        assert_eq!(result, t);
        assert_eq!(evaluator.statistics().0, 1); // 1 eval
    }

    #[test]
    fn test_lazy_evaluator_simplify_not() {
        let mut evaluator = LazyEvaluator::new();
        let mut manager = TermManager::new();

        let t = manager.mk_true();
        let not_t = manager.mk_not(t);
        evaluator.mark_relevant(not_t);
        evaluator.mark_relevant(t);

        let result = evaluator.eval(not_t, &mut manager);
        let f = manager.mk_false();
        assert_eq!(result, f);
    }

    #[test]
    fn test_lazy_evaluator_cache_hit() {
        let mut evaluator = LazyEvaluator::new();
        let mut manager = TermManager::new();

        let t = manager.mk_true();
        evaluator.mark_relevant(t);

        let _ = evaluator.eval(t, &mut manager);
        let _ = evaluator.eval(t, &mut manager);

        let (evals, hits, _) = evaluator.statistics();
        assert_eq!(evals, 1); // Only evaluated once
        assert_eq!(hits, 1); // One cache hit on second call
    }

    #[test]
    fn test_lazy_evaluator_invalidate() {
        let mut evaluator = LazyEvaluator::new();
        let mut manager = TermManager::new();

        let t = manager.mk_true();
        evaluator.mark_relevant(t);
        let _ = evaluator.eval(t, &mut manager);
        assert_eq!(evaluator.cache_size(), 1);

        evaluator.invalidate();
        assert_eq!(evaluator.cache_size(), 0);
    }

    #[test]
    fn test_lazy_evaluator_simplify_and() {
        let mut evaluator = LazyEvaluator::new();
        let mut manager = TermManager::new();

        let t = manager.mk_true();
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let and = manager.mk_and(vec![t, p]);

        evaluator.mark_relevant(and);
        evaluator.mark_relevant(t);
        evaluator.mark_relevant(p);

        let result = evaluator.eval(and, &mut manager);
        // (and true p) should simplify to p
        assert_eq!(result, p);
    }

    #[test]
    fn test_lazy_evaluator_simplify_or_with_true() {
        let mut evaluator = LazyEvaluator::new();
        let mut manager = TermManager::new();

        let t = manager.mk_true();
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let or = manager.mk_or(vec![t, p]);

        evaluator.mark_relevant(or);
        evaluator.mark_relevant(t);
        evaluator.mark_relevant(p);

        let result = evaluator.eval(or, &mut manager);
        // (or true p) should simplify to true
        assert_eq!(result, t);
    }

    #[test]
    fn test_lazy_evaluator_clear_relevant() {
        let mut evaluator = LazyEvaluator::new();
        let t1 = TermId::new(10);
        evaluator.mark_relevant(t1);
        assert_eq!(evaluator.relevant_count(), 1);

        evaluator.clear_relevant();
        assert_eq!(evaluator.relevant_count(), 0);
    }
}
