//! Random walk search components:
//! FocusedWalk, NoveltySelector, SparrowSelector, BmsSelector.

use super::types::{Lit, Var};

// ============================================================================
// Focused Random Walk
// ============================================================================

/// Focused random walk parameters
#[derive(Debug, Clone)]
pub struct FocusedWalkConfig {
    /// Probability of focused move
    pub focus_prob: f64,
    /// Number of variables to consider
    pub focus_size: usize,
    /// Use break score
    pub use_break: bool,
    /// Use make score
    pub use_make: bool,
}

impl Default for FocusedWalkConfig {
    fn default() -> Self {
        Self {
            focus_prob: 0.8,
            focus_size: 5,
            use_break: true,
            use_make: true,
        }
    }
}

/// Focused random walk solver
#[derive(Debug)]
pub struct FocusedWalk {
    /// Configuration
    config: FocusedWalkConfig,
    /// Focus set (candidate variables)
    focus_set: Vec<Var>,
}

impl FocusedWalk {
    /// Create a new focused walk
    pub fn new(config: FocusedWalkConfig) -> Self {
        Self {
            config,
            focus_set: Vec::new(),
        }
    }

    /// Update focus set from unsatisfied clause
    pub fn update_focus(&mut self, clause_lits: &[Lit], break_counts: &[u32]) {
        self.focus_set.clear();

        // Sort by break count
        let mut vars: Vec<_> = clause_lits
            .iter()
            .map(|&lit| {
                let var = lit.unsigned_abs();
                let break_val = break_counts.get(var as usize).copied().unwrap_or(0);
                (var, break_val)
            })
            .collect();

        vars.sort_by_key(|&(_, b)| b);

        // Take top focus_size variables
        for (var, _) in vars.into_iter().take(self.config.focus_size) {
            self.focus_set.push(var);
        }
    }

    /// Get the focus set
    pub fn focus_set(&self) -> &[Var] {
        &self.focus_set
    }
}

// ============================================================================
// Novelty and rNovelty Heuristics
// ============================================================================

/// Novelty parameter for variable selection
#[derive(Debug, Clone)]
pub struct NoveltyConfig {
    /// Probability of selecting second-best variable (p)
    pub novelty_prob: f64,
    /// Enable novelty+ (extra random walk)
    pub novelty_plus: bool,
    /// Random walk probability for novelty+
    pub wp: f64,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            novelty_prob: 0.5,
            novelty_plus: true,
            wp: 0.01,
        }
    }
}

/// Novelty variable selector
#[derive(Debug)]
pub struct NoveltySelector {
    /// Configuration
    config: NoveltyConfig,
    /// Last flipped variable
    last_flipped: Option<Var>,
    /// Flip age for each variable
    flip_age: Vec<u64>,
    /// Current time (flip count)
    current_time: u64,
}

impl NoveltySelector {
    /// Create a new novelty selector
    pub fn new(config: NoveltyConfig) -> Self {
        Self {
            config,
            last_flipped: None,
            flip_age: Vec::new(),
            current_time: 0,
        }
    }

    /// Ensure capacity for n variables
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.flip_age.len() < n {
            self.flip_age.resize(n, 0);
        }
    }

    /// Notify that a variable was flipped
    pub fn notify_flip(&mut self, var: Var) {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);
        self.flip_age[idx] = self.current_time;
        self.last_flipped = Some(var);
        self.current_time += 1;
    }

    /// Get age of a variable (flips since last flip)
    pub fn age(&self, var: Var) -> u64 {
        let idx = var as usize;
        if idx < self.flip_age.len() {
            self.current_time.saturating_sub(self.flip_age[idx])
        } else {
            self.current_time
        }
    }

    /// Select variable from candidates using novelty heuristic
    pub fn select(
        &self,
        candidates: &[(Var, i32)], // (variable, break count)
        rng: &mut u64,
    ) -> Option<Var> {
        if candidates.is_empty() {
            return None;
        }

        if candidates.len() == 1 {
            return Some(candidates[0].0);
        }

        // Sort by break count (best = lowest break)
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by_key(|&(_, b)| b);

        let best = sorted[0].0;
        let second_best = sorted[1].0;

        // If best is the same as last flipped, consider second best
        if Some(best) == self.last_flipped {
            // With probability novelty_prob, pick second best
            let r = (*rng as f64) / (u64::MAX as f64);
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;

            if r < self.config.novelty_prob {
                return Some(second_best);
            }
        }

        // Novelty+: with small probability, pick random
        if self.config.novelty_plus {
            let r = (*rng as f64) / (u64::MAX as f64);
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;

            if r < self.config.wp {
                let idx = (*rng as usize) % candidates.len();
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                return Some(candidates[idx].0);
            }
        }

        Some(best)
    }

    /// Reset the selector
    pub fn reset(&mut self) {
        self.last_flipped = None;
        self.current_time = 0;
        for age in &mut self.flip_age {
            *age = 0;
        }
    }
}

// ============================================================================
// Sparrow Algorithm
// ============================================================================

/// Sparrow algorithm configuration
#[derive(Debug, Clone)]
pub struct SparrowConfig {
    /// Base probability scaling factor c_b
    pub cb: f64,
    /// Make probability scaling factor c_m
    pub cm: f64,
    /// Smoothing probability
    pub sp: f64,
    /// Age factor
    pub age_factor: f64,
}

impl Default for SparrowConfig {
    fn default() -> Self {
        Self {
            cb: 2.06,
            cm: 0.0,
            sp: 0.8,
            age_factor: 4.0,
        }
    }
}

/// Sparrow variable selector
#[derive(Debug)]
pub struct SparrowSelector {
    /// Configuration
    config: SparrowConfig,
    /// Variable ages (flips since last flip)
    ages: Vec<u64>,
    /// Current flip count
    current_flip: u64,
    /// Probabilities workspace
    probs: Vec<f64>,
}

impl SparrowSelector {
    /// Create a new Sparrow selector
    pub fn new(config: SparrowConfig) -> Self {
        Self {
            config,
            ages: Vec::new(),
            current_flip: 0,
            probs: Vec::new(),
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.ages = vec![0; n];
        self.current_flip = 0;
    }

    /// Notify flip
    pub fn notify_flip(&mut self, var: Var) {
        let idx = var as usize;
        if idx < self.ages.len() {
            self.ages[idx] = self.current_flip;
        }
        self.current_flip += 1;
    }

    /// Get age of variable
    pub fn age(&self, var: Var) -> u64 {
        let idx = var as usize;
        if idx < self.ages.len() {
            self.current_flip.saturating_sub(self.ages[idx])
        } else {
            self.current_flip
        }
    }

    /// Select variable from clause literals
    pub fn select(
        &mut self,
        clause_lits: &[Lit],
        break_counts: &[u32],
        make_counts: &[u32],
        rng: &mut u64,
    ) -> Option<Var> {
        if clause_lits.is_empty() {
            return None;
        }

        self.probs.clear();
        let mut total = 0.0;

        for &lit in clause_lits {
            let var = lit.unsigned_abs();
            let break_val = break_counts.get(var as usize).copied().unwrap_or(0) as f64;
            let make_val = make_counts.get(var as usize).copied().unwrap_or(0) as f64;
            let age_val = self.age(var) as f64;

            // Sparrow probability function
            let prob = self.config.sp.powf(break_val)
                * (1.0 + make_val).powf(self.config.cm)
                * (1.0 + age_val).powf(self.config.age_factor);

            self.probs.push(prob);
            total += prob;
        }

        if total <= 0.0 {
            return Some(clause_lits[0].unsigned_abs());
        }

        // Roulette wheel selection
        let mut r = (*rng as f64 / u64::MAX as f64) * total;
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;

        for (i, &prob) in self.probs.iter().enumerate() {
            r -= prob;
            if r <= 0.0 {
                return Some(clause_lits[i].unsigned_abs());
            }
        }

        Some(clause_lits.last()?.unsigned_abs())
    }

    /// Reset
    pub fn reset(&mut self) {
        for age in &mut self.ages {
            *age = 0;
        }
        self.current_flip = 0;
    }
}

// ============================================================================
// Break-Make Score (BMS) Selector
// ============================================================================

/// BMS (Break-Make Score) variable selection
#[derive(Debug, Clone)]
pub struct BmsConfig {
    /// Break weight
    pub break_weight: f64,
    /// Make weight
    pub make_weight: f64,
    /// Age weight
    pub age_weight: f64,
    /// Polynomial exponent for break
    pub break_exp: f64,
    /// Polynomial exponent for make
    pub make_exp: f64,
}

impl Default for BmsConfig {
    fn default() -> Self {
        Self {
            break_weight: 1.0,
            make_weight: 0.5,
            age_weight: 0.1,
            break_exp: 2.0,
            make_exp: 1.0,
        }
    }
}

/// BMS variable selector
#[derive(Debug)]
pub struct BmsSelector {
    /// Configuration
    config: BmsConfig,
    /// Variable ages
    ages: Vec<u64>,
    /// Current flip
    current_flip: u64,
    /// Score cache
    score_cache: Vec<f64>,
}

impl BmsSelector {
    /// Create a new BMS selector
    pub fn new(config: BmsConfig) -> Self {
        Self {
            config,
            ages: Vec::new(),
            current_flip: 0,
            score_cache: Vec::new(),
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.ages = vec![0; n];
        self.score_cache = vec![0.0; n];
        self.current_flip = 0;
    }

    /// Notify flip
    pub fn notify_flip(&mut self, var: Var) {
        let idx = var as usize;
        if idx < self.ages.len() {
            self.ages[idx] = self.current_flip;
        }
        self.current_flip += 1;
    }

    /// Compute BMS score for a variable
    pub fn compute_score(&self, var: Var, break_count: u32, make_count: u32) -> f64 {
        let idx = var as usize;
        let age = if idx < self.ages.len() {
            self.current_flip.saturating_sub(self.ages[idx]) as f64
        } else {
            self.current_flip as f64
        };

        // BMS formula: -break_weight * break^exp + make_weight * make^exp + age_weight * age
        -self.config.break_weight * (break_count as f64).powf(self.config.break_exp)
            + self.config.make_weight * (make_count as f64).powf(self.config.make_exp)
            + self.config.age_weight * age
    }

    /// Select best variable by BMS score
    pub fn select(
        &self,
        candidates: &[Var],
        break_counts: &[u32],
        make_counts: &[u32],
    ) -> Option<Var> {
        candidates
            .iter()
            .map(|&v| {
                let b = break_counts.get(v as usize).copied().unwrap_or(0);
                let m = make_counts.get(v as usize).copied().unwrap_or(0);
                (v, self.compute_score(v, b, m))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(v, _)| v)
    }

    /// Reset
    pub fn reset(&mut self) {
        for age in &mut self.ages {
            *age = 0;
        }
        self.current_flip = 0;
    }
}
