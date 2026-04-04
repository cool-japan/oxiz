//! Scoring and weighting components for SLS:
//! ClauseWeightManager, VarActivity, CcanrEnhancer, DdfwManager, ClauseImportance.

use super::types::{ClauseId, Lit, Var};
use std::collections::HashSet;

// ============================================================================
// Clause Weighting Schemes
// ============================================================================

/// Clause weighting scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightingScheme {
    /// No weighting
    #[default]
    None,
    /// Additive weighting
    Additive,
    /// Multiplicative weighting
    Multiplicative,
    /// SAPS (Scaling and Probabilistic Smoothing)
    Saps,
    /// PAWS (Pure Additive Weighting Scheme)
    Paws,
}

/// Clause weight manager
#[derive(Debug)]
pub struct ClauseWeightManager {
    /// Weights for each clause
    pub(super) weights: Vec<f64>,
    /// Weighting scheme
    scheme: WeightingScheme,
    /// Additive increment
    add_inc: f64,
    /// Multiplicative factor
    mult_factor: f64,
    /// Smooth probability
    #[allow(dead_code)]
    smooth_prob: f64,
}

impl ClauseWeightManager {
    /// Create a new weight manager
    pub fn new(scheme: WeightingScheme) -> Self {
        Self {
            weights: Vec::new(),
            scheme,
            add_inc: 1.0,
            mult_factor: 1.1,
            smooth_prob: 0.01,
        }
    }

    /// Initialize weights for n clauses
    pub fn initialize(&mut self, n: usize) {
        self.weights = vec![1.0; n];
    }

    /// Update weights for unsatisfied clauses
    pub fn update(&mut self, unsat: &HashSet<ClauseId>) {
        match self.scheme {
            WeightingScheme::None => {}
            WeightingScheme::Additive => {
                for &cid in unsat {
                    let idx = cid.0 as usize;
                    if idx < self.weights.len() {
                        self.weights[idx] += self.add_inc;
                    }
                }
            }
            WeightingScheme::Multiplicative => {
                for &cid in unsat {
                    let idx = cid.0 as usize;
                    if idx < self.weights.len() {
                        self.weights[idx] *= self.mult_factor;
                    }
                }
            }
            WeightingScheme::Saps | WeightingScheme::Paws => {
                // SAPS: Scale and smooth
                for &cid in unsat {
                    let idx = cid.0 as usize;
                    if idx < self.weights.len() {
                        self.weights[idx] *= self.mult_factor;
                    }
                }
            }
        }
    }

    /// Smooth weights (decrease all weights slightly)
    pub fn smooth(&mut self) {
        let avg: f64 = self.weights.iter().sum::<f64>() / self.weights.len() as f64;
        for w in &mut self.weights {
            *w = (*w + avg) / 2.0;
        }
    }

    /// Get weight for a clause
    pub fn get_weight(&self, clause_id: ClauseId) -> f64 {
        self.weights
            .get(clause_id.0 as usize)
            .copied()
            .unwrap_or(1.0)
    }

    /// Reset all weights
    pub fn reset(&mut self) {
        for w in &mut self.weights {
            *w = 1.0;
        }
    }
}

// ============================================================================
// Variable Selection Heuristics
// ============================================================================

/// Variable selection heuristic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VarSelectHeuristic {
    /// Minimum break (standard WalkSAT)
    #[default]
    MinBreak,
    /// Maximum make
    MaxMake,
    /// Maximum net gain (make - break)
    MaxGain,
    /// Age-based (oldest variable in clause)
    Age,
    /// Score-based (VSIDS-like)
    Score,
}

/// Variable activity tracker
#[derive(Debug)]
pub struct VarActivity {
    /// Activity scores
    activities: Vec<f64>,
    /// Decay factor
    decay: f64,
    /// Bump amount
    pub(super) bump: f64,
}

impl VarActivity {
    /// Create a new activity tracker
    pub fn new() -> Self {
        Self {
            activities: Vec::new(),
            decay: 0.95,
            bump: 1.0,
        }
    }

    /// Ensure capacity for n variables
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.activities.len() < n {
            self.activities.resize(n, 0.0);
        }
    }

    /// Bump activity for a variable
    pub fn bump(&mut self, var: Var) {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);
        self.activities[idx] += self.bump;

        // Rescale if too large
        if self.activities[idx] > 1e100 {
            for a in &mut self.activities {
                *a *= 1e-100;
            }
            self.bump *= 1e-100;
        }
    }

    /// Decay all activities
    pub fn decay(&mut self) {
        self.bump /= self.decay;
    }

    /// Get activity for a variable
    pub fn get(&self, var: Var) -> f64 {
        self.activities.get(var as usize).copied().unwrap_or(0.0)
    }

    /// Reset all activities
    pub fn reset(&mut self) {
        for a in &mut self.activities {
            *a = 0.0;
        }
        self.bump = 1.0;
    }
}

impl Default for VarActivity {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CCAnr (Conflict-driven Clause Learning with Novelty and Restart)
// ============================================================================

/// CCAnr configuration
#[derive(Debug, Clone)]
pub struct CcanrConfig {
    /// Configuration score increase
    pub score_inc: f64,
    /// Average weight threshold
    pub avg_weight_threshold: f64,
    /// Clause weight limit
    pub weight_limit: f64,
    /// Enable configuration checking
    pub config_checking: bool,
}

impl Default for CcanrConfig {
    fn default() -> Self {
        Self {
            score_inc: 1.0,
            avg_weight_threshold: 3.0,
            weight_limit: 100.0,
            config_checking: true,
        }
    }
}

/// CCAnr solver enhancements
#[derive(Debug)]
pub struct CcanrEnhancer {
    /// Configuration
    config: CcanrConfig,
    /// Variable scores
    scores: Vec<f64>,
    /// Configuration checking bits
    config_bits: Vec<bool>,
    /// Total weight
    total_weight: f64,
}

impl CcanrEnhancer {
    /// Create a new CCAnr enhancer
    pub fn new(config: CcanrConfig) -> Self {
        Self {
            config,
            scores: Vec::new(),
            config_bits: Vec::new(),
            total_weight: 0.0,
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.scores = vec![0.0; n];
        self.config_bits = vec![false; n];
        self.total_weight = 0.0;
    }

    /// Update scores for unsatisfied clause
    pub fn update_scores(&mut self, clause_lits: &[Lit]) {
        for &lit in clause_lits {
            let var = lit.unsigned_abs() as usize;
            if var < self.scores.len() {
                self.scores[var] += self.config.score_inc;
            }
        }
    }

    /// Set configuration bit for variable
    pub fn set_config(&mut self, var: Var, value: bool) {
        let idx = var as usize;
        if idx < self.config_bits.len() {
            self.config_bits[idx] = value;
        }
    }

    /// Check configuration bit
    pub fn check_config(&self, var: Var) -> bool {
        let idx = var as usize;
        if idx < self.config_bits.len() {
            self.config_bits[idx]
        } else {
            false
        }
    }

    /// Get score for variable
    pub fn score(&self, var: Var) -> f64 {
        self.scores.get(var as usize).copied().unwrap_or(0.0)
    }

    /// Decay all scores
    pub fn decay_scores(&mut self, factor: f64) {
        for s in &mut self.scores {
            *s *= factor;
        }
    }

    /// Should smooth weights?
    pub fn should_smooth(&self, num_clauses: usize) -> bool {
        if num_clauses == 0 {
            return false;
        }
        let avg = self.total_weight / num_clauses as f64;
        avg > self.config.avg_weight_threshold
    }

    /// Add to total weight
    pub fn add_weight(&mut self, w: f64) {
        self.total_weight += w;
    }

    /// Reset
    pub fn reset(&mut self) {
        for s in &mut self.scores {
            *s = 0.0;
        }
        for b in &mut self.config_bits {
            *b = false;
        }
        self.total_weight = 0.0;
    }
}

// ============================================================================
// DDFW (Divide and Distribute Fixed Weights)
// ============================================================================

/// DDFW configuration
#[derive(Debug, Clone)]
pub struct DdfwConfig {
    /// Initial weight for all clauses
    pub init_weight: f64,
    /// Weight transfer amount
    pub transfer_amount: f64,
    /// Distribution frequency (every N flips)
    pub distribute_freq: u32,
}

impl Default for DdfwConfig {
    fn default() -> Self {
        Self {
            init_weight: 1.0,
            transfer_amount: 1.0,
            distribute_freq: 100,
        }
    }
}

/// DDFW weight manager
#[derive(Debug)]
pub struct DdfwManager {
    /// Configuration
    config: DdfwConfig,
    /// Clause weights
    weights: Vec<f64>,
    /// Flip counter for distribution
    flip_counter: u32,
    /// Total weight (should be constant)
    total_weight: f64,
}

impl DdfwManager {
    /// Create a new DDFW manager
    pub fn new(config: DdfwConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            flip_counter: 0,
            total_weight: 0.0,
        }
    }

    /// Initialize for n clauses
    pub fn initialize(&mut self, n: usize) {
        self.weights = vec![self.config.init_weight; n];
        self.total_weight = self.config.init_weight * n as f64;
        self.flip_counter = 0;
    }

    /// Get weight for clause
    pub fn weight(&self, clause_id: usize) -> f64 {
        self.weights
            .get(clause_id)
            .copied()
            .unwrap_or(self.config.init_weight)
    }

    /// Notify of flip
    pub fn notify_flip(&mut self) {
        self.flip_counter += 1;
    }

    /// Should distribute weights?
    pub fn should_distribute(&self) -> bool {
        self.flip_counter >= self.config.distribute_freq
    }

    /// Distribute weights from satisfied to unsatisfied clauses
    pub fn distribute(&mut self, satisfied: &[usize], unsatisfied: &[usize]) {
        if satisfied.is_empty() || unsatisfied.is_empty() {
            self.flip_counter = 0;
            return;
        }

        // Calculate transfer amount from each satisfied clause
        let transfer_per_sat = self.config.transfer_amount / satisfied.len() as f64;
        let gain_per_unsat = (transfer_per_sat * satisfied.len() as f64) / unsatisfied.len() as f64;

        // Transfer from satisfied
        for &idx in satisfied {
            if idx < self.weights.len() {
                let transfer = self.weights[idx].min(transfer_per_sat);
                self.weights[idx] -= transfer;
            }
        }

        // Add to unsatisfied
        for &idx in unsatisfied {
            if idx < self.weights.len() {
                self.weights[idx] += gain_per_unsat;
            }
        }

        self.flip_counter = 0;
    }

    /// Reset
    pub fn reset(&mut self) {
        let n = self.weights.len();
        for w in &mut self.weights {
            *w = self.config.init_weight;
        }
        self.total_weight = self.config.init_weight * n as f64;
        self.flip_counter = 0;
    }
}

// ============================================================================
// Clause Importance Tracking
// ============================================================================

/// Tracks importance of clauses based on conflict frequency
#[derive(Debug)]
pub struct ClauseImportance {
    /// Hit count (how often clause was unsatisfied)
    hit_counts: Vec<u32>,
    /// Critical count (how often clause was the only one unsatisfied)
    critical_counts: Vec<u32>,
    /// Decay factor
    decay: f64,
    /// Decay interval (flips)
    decay_interval: u32,
    /// Current flip
    current_flip: u32,
}

impl ClauseImportance {
    /// Create new importance tracker
    pub fn new() -> Self {
        Self {
            hit_counts: Vec::new(),
            critical_counts: Vec::new(),
            decay: 0.99,
            decay_interval: 1000,
            current_flip: 0,
        }
    }

    /// Initialize for n clauses
    pub fn initialize(&mut self, n: usize) {
        self.hit_counts = vec![0; n];
        self.critical_counts = vec![0; n];
        self.current_flip = 0;
    }

    /// Record hit on clause
    pub fn record_hit(&mut self, clause_id: usize) {
        if clause_id < self.hit_counts.len() {
            self.hit_counts[clause_id] = self.hit_counts[clause_id].saturating_add(1);
        }
    }

    /// Record critical clause (only unsatisfied)
    pub fn record_critical(&mut self, clause_id: usize) {
        if clause_id < self.critical_counts.len() {
            self.critical_counts[clause_id] = self.critical_counts[clause_id].saturating_add(1);
        }
    }

    /// Notify flip
    pub fn notify_flip(&mut self) {
        self.current_flip += 1;
        if self.current_flip >= self.decay_interval {
            self.decay_all();
            self.current_flip = 0;
        }
    }

    /// Decay all counts
    fn decay_all(&mut self) {
        for h in &mut self.hit_counts {
            *h = (*h as f64 * self.decay) as u32;
        }
        for c in &mut self.critical_counts {
            *c = (*c as f64 * self.decay) as u32;
        }
    }

    /// Get importance score (combines hit and critical counts)
    pub fn importance(&self, clause_id: usize) -> f64 {
        let hit = self.hit_counts.get(clause_id).copied().unwrap_or(0) as f64;
        let critical = self.critical_counts.get(clause_id).copied().unwrap_or(0) as f64;
        hit + 2.0 * critical // Critical counts worth more
    }

    /// Get most important clauses
    pub fn most_important(&self, n: usize) -> Vec<usize> {
        let mut indices: Vec<_> = (0..self.hit_counts.len())
            .map(|i| (i, self.importance(i)))
            .collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        indices.into_iter().take(n).map(|(i, _)| i).collect()
    }

    /// Reset
    pub fn reset(&mut self) {
        for h in &mut self.hit_counts {
            *h = 0;
        }
        for c in &mut self.critical_counts {
            *c = 0;
        }
        self.current_flip = 0;
    }
}

impl Default for ClauseImportance {
    fn default() -> Self {
        Self::new()
    }
}
