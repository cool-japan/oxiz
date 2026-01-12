//! Lookahead decision heuristics for SAT solving.
//!
//! Lookahead is a technique that evaluates the quality of potential decision
//! variables by performing test assignments and measuring the amount of
//! propagation they cause. Variables that cause more propagation are often
//! better choices for decisions.
//!
//! Reference: "March SAT solver" and related lookahead SAT solvers

use crate::types::{BoolVar, Literal};
use std::collections::HashMap;

/// Statistics for lookahead operations.
#[derive(Debug, Clone, Default)]
pub struct LookaheadStats {
    /// Number of lookahead operations performed.
    pub lookaheads: u64,
    /// Number of failed literals detected.
    pub failed_literals: u64,
    /// Total propagations during lookahead.
    pub propagations: u64,
    /// Number of autarkies found.
    pub autarkies: u64,
}

/// Configuration for lookahead.
#[derive(Debug, Clone)]
pub struct LookaheadConfig {
    /// Enable lookahead.
    pub enabled: bool,
    /// Maximum depth for lookahead propagation.
    pub max_depth: usize,
    /// Enable failed literal detection.
    pub detect_failed_literals: bool,
    /// Enable double lookahead (look at both polarities).
    pub double_lookahead: bool,
    /// Maximum number of variables to evaluate per decision.
    pub max_vars_per_decision: usize,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 10,
            detect_failed_literals: true,
            double_lookahead: true,
            max_vars_per_decision: 20,
        }
    }
}

/// Result of a lookahead operation.
#[derive(Debug, Clone)]
pub struct LookaheadResult {
    /// The literal that was tested.
    pub literal: Literal,
    /// Number of propagations caused.
    pub propagations: usize,
    /// Whether this literal is a failed literal (causes conflict).
    pub failed: bool,
    /// Score based on propagation strength.
    pub score: f64,
}

impl LookaheadResult {
    /// Create a new lookahead result.
    pub fn new(literal: Literal, propagations: usize, failed: bool) -> Self {
        let score = if failed {
            f64::INFINITY
        } else {
            propagations as f64
        };

        Self {
            literal,
            propagations,
            failed,
            score,
        }
    }
}

/// Lookahead decision engine.
pub struct LookaheadEngine {
    config: LookaheadConfig,
    stats: LookaheadStats,
    /// Scores for each variable (based on lookahead).
    var_scores: HashMap<BoolVar, f64>,
}

impl LookaheadEngine {
    /// Create a new lookahead engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(LookaheadConfig::default())
    }

    /// Create a new lookahead engine with the given configuration.
    pub fn with_config(config: LookaheadConfig) -> Self {
        Self {
            config,
            stats: LookaheadStats::default(),
            var_scores: HashMap::new(),
        }
    }

    /// Get the statistics.
    pub fn stats(&self) -> &LookaheadStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &LookaheadConfig {
        &self.config
    }

    /// Get the score for a variable.
    pub fn get_score(&self, var: BoolVar) -> f64 {
        self.var_scores.get(&var).copied().unwrap_or(0.0)
    }

    /// Perform lookahead on a literal.
    ///
    /// Returns the result of the lookahead, including propagation count
    /// and whether it's a failed literal.
    pub fn lookahead<F>(&mut self, lit: Literal, mut propagate: F) -> LookaheadResult
    where
        F: FnMut(Literal) -> Result<usize, ()>,
    {
        self.stats.lookaheads += 1;

        // Try propagating this literal
        match propagate(lit) {
            Ok(propagations) => {
                self.stats.propagations += propagations as u64;
                LookaheadResult::new(lit, propagations, false)
            }
            Err(()) => {
                // Failed literal - causes immediate conflict
                self.stats.failed_literals += 1;
                LookaheadResult::new(lit, 0, true)
            }
        }
    }

    /// Perform double lookahead on a variable (both polarities).
    ///
    /// Returns results for both positive and negative literals.
    pub fn double_lookahead<F>(
        &mut self,
        var: BoolVar,
        propagate: F,
    ) -> (LookaheadResult, LookaheadResult)
    where
        F: FnMut(Literal) -> Result<usize, ()> + Copy,
    {
        let pos_lit = Literal::new(var, false); // false = not negated = positive
        let neg_lit = Literal::new(var, true); // true = negated = negative

        let pos_result = self.lookahead(pos_lit, propagate);
        let neg_result = self.lookahead(neg_lit, propagate);

        (pos_result, neg_result)
    }

    /// Select the best decision variable using lookahead.
    ///
    /// Evaluates a set of candidate variables and returns the one with
    /// the highest lookahead score.
    pub fn select_decision<F>(
        &mut self,
        candidates: &[BoolVar],
        propagate: F,
    ) -> Option<(BoolVar, bool)>
    where
        F: FnMut(Literal) -> Result<usize, ()> + Copy,
    {
        if !self.config.enabled {
            return None;
        }

        if candidates.is_empty() {
            return None;
        }

        let mut best_var = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_polarity = true;

        let eval_count = candidates.len().min(self.config.max_vars_per_decision);

        for &var in &candidates[..eval_count] {
            if self.config.double_lookahead {
                let (pos_result, neg_result) = self.double_lookahead(var, propagate);

                // Handle failed literals
                if pos_result.failed && !neg_result.failed {
                    // Positive is failed, must use negative
                    return Some((var, false));
                } else if neg_result.failed && !pos_result.failed {
                    // Negative is failed, must use positive
                    return Some((var, true));
                } else if pos_result.failed && neg_result.failed {
                    // Both failed - conflict (should not happen in normal operation)
                    continue;
                }

                // Calculate score based on propagation strength
                // Use difference heuristic: choose polarity with more propagations
                let score = (pos_result.propagations + neg_result.propagations) as f64;
                let polarity = pos_result.propagations >= neg_result.propagations;

                if score > best_score {
                    best_score = score;
                    best_var = Some(var);
                    best_polarity = polarity;
                }

                // Update variable score
                self.var_scores.insert(var, score);
            } else {
                // Single lookahead (positive polarity only)
                let pos_lit = Literal::new(var, false); // false = not negated = positive
                let result = self.lookahead(pos_lit, propagate);

                if result.failed {
                    // Must use negative polarity
                    return Some((var, false));
                }

                if result.score > best_score {
                    best_score = result.score;
                    best_var = Some(var);
                    best_polarity = true;
                }

                self.var_scores.insert(var, result.score);
            }
        }

        best_var.map(|var| (var, best_polarity))
    }

    /// Detect failed literals in a set of literals.
    ///
    /// A failed literal is one that causes immediate conflict when assigned.
    /// Returns a list of failed literals that can be assigned to their
    /// opposite polarity.
    pub fn detect_failed_literals<F>(&mut self, literals: &[Literal], propagate: F) -> Vec<Literal>
    where
        F: FnMut(Literal) -> Result<usize, ()> + Copy,
    {
        if !self.config.detect_failed_literals {
            return Vec::new();
        }

        let mut failed = Vec::new();

        for &lit in literals {
            let result = self.lookahead(lit, propagate);

            if result.failed {
                failed.push(lit);
            }
        }

        failed
    }

    /// Clear variable scores.
    pub fn clear_scores(&mut self) {
        self.var_scores.clear();
    }
}

impl Default for LookaheadEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookahead_engine_new() {
        let engine = LookaheadEngine::new();
        assert!(engine.config().enabled);
        assert_eq!(engine.stats().lookaheads, 0);
    }

    #[test]
    fn test_lookahead_result() {
        let lit = Literal::new(0, true);
        let result = LookaheadResult::new(lit, 5, false);

        assert_eq!(result.literal, lit);
        assert_eq!(result.propagations, 5);
        assert!(!result.failed);
        assert_eq!(result.score, 5.0);
    }

    #[test]
    fn test_lookahead_result_failed() {
        let lit = Literal::new(0, true);
        let result = LookaheadResult::new(lit, 0, true);

        assert!(result.failed);
        assert_eq!(result.score, f64::INFINITY);
    }

    #[test]
    fn test_lookahead_success() {
        let mut engine = LookaheadEngine::new();

        let lit = Literal::new(0, true);
        let propagate = |_: Literal| Ok(3);

        let result = engine.lookahead(lit, propagate);

        assert!(!result.failed);
        assert_eq!(result.propagations, 3);
        assert_eq!(engine.stats().lookaheads, 1);
    }

    #[test]
    fn test_lookahead_failed() {
        let mut engine = LookaheadEngine::new();

        let lit = Literal::new(0, true);
        let propagate = |_: Literal| Err(());

        let result = engine.lookahead(lit, propagate);

        assert!(result.failed);
        assert_eq!(engine.stats().failed_literals, 1);
    }

    #[test]
    fn test_double_lookahead() {
        let mut engine = LookaheadEngine::new();

        let var = 0;
        let propagate = |lit: Literal| {
            if !lit.is_negated() { Ok(5) } else { Ok(3) }
        };

        let (pos_result, neg_result) = engine.double_lookahead(var, propagate);

        assert_eq!(pos_result.propagations, 5);
        assert_eq!(neg_result.propagations, 3);
    }

    #[test]
    fn test_select_decision_disabled() {
        let config = LookaheadConfig {
            enabled: false,
            ..Default::default()
        };
        let mut engine = LookaheadEngine::with_config(config);

        let candidates = vec![0, 1, 2];
        let propagate = |_: Literal| Ok(1);

        let result = engine.select_decision(&candidates, propagate);

        assert_eq!(result, None);
    }

    #[test]
    fn test_get_score() {
        let mut engine = LookaheadEngine::new();

        engine.var_scores.insert(0, 10.0);

        assert_eq!(engine.get_score(0), 10.0);
        assert_eq!(engine.get_score(1), 0.0);
    }

    #[test]
    fn test_clear_scores() {
        let mut engine = LookaheadEngine::new();

        engine.var_scores.insert(0, 10.0);
        engine.var_scores.insert(1, 20.0);

        engine.clear_scores();

        assert_eq!(engine.var_scores.len(), 0);
    }

    #[test]
    fn test_detect_failed_literals() {
        let mut engine = LookaheadEngine::new();

        let lit1 = Literal::new(0, true);
        let lit2 = Literal::new(1, true);
        let lit3 = Literal::new(2, true);

        let literals = vec![lit1, lit2, lit3];

        let propagate = |lit: Literal| {
            if lit.var() == 1 {
                Err(()) // lit2 is failed
            } else {
                Ok(1)
            }
        };

        let failed = engine.detect_failed_literals(&literals, propagate);

        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0], lit2);
    }
}
