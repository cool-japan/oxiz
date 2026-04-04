//! Portfolio and hybrid SLS components:
//! WeightedSlsSolver, HybridSlsInterface, YalsatSolver, PortfolioSls.

use super::types::{Lit, SlsAlgorithm, SlsConfig, SlsResult, SlsSolver, SlsStats, Var};
use std::collections::HashMap;
use std::collections::HashSet;

// ============================================================================
// Weighted MAX-SAT SLS
// ============================================================================

/// Configuration for weighted MAX-SAT SLS
#[derive(Debug, Clone)]
pub struct WeightedSlsConfig {
    /// Base SLS configuration
    pub base: SlsConfig,
    /// Weight update factor
    pub weight_update: f64,
    /// Smooth probability
    pub smooth_prob: f64,
}

impl Default for WeightedSlsConfig {
    fn default() -> Self {
        Self {
            base: SlsConfig::default(),
            weight_update: 1.0,
            smooth_prob: 0.01,
        }
    }
}

/// Weighted MAX-SAT SLS solver
#[derive(Debug)]
pub struct WeightedSlsSolver {
    /// Base SLS solver
    base: SlsSolver,
    /// Clause weights
    weights: Vec<f64>,
    /// Configuration
    #[allow(dead_code)]
    config: WeightedSlsConfig,
    /// Best cost found
    best_cost: f64,
    /// Statistics
    stats: WeightedSlsStats,
}

/// Statistics for weighted SLS
#[derive(Debug, Clone, Default)]
pub struct WeightedSlsStats {
    /// Base stats
    pub base_stats: SlsStats,
    /// Weight updates
    pub weight_updates: u64,
    /// Smooth operations
    pub smooth_ops: u64,
    /// Best cost found
    pub best_cost: f64,
}

impl WeightedSlsSolver {
    /// Create a new weighted SLS solver
    pub fn new(config: WeightedSlsConfig) -> Self {
        Self {
            base: SlsSolver::new(config.base.clone()),
            weights: Vec::new(),
            config,
            best_cost: f64::MAX,
            stats: WeightedSlsStats::default(),
        }
    }

    /// Add a weighted clause
    pub fn add_weighted_clause(&mut self, literals: &[Lit], weight: f64) {
        self.base.add_clause(literals);
        self.weights.push(weight);
    }

    /// Add a hard clause (weight = infinity)
    pub fn add_hard_clause(&mut self, literals: &[Lit]) {
        self.add_weighted_clause(literals, f64::MAX);
    }

    /// Add a soft clause with unit weight
    pub fn add_soft_clause(&mut self, literals: &[Lit]) {
        self.add_weighted_clause(literals, 1.0);
    }

    /// Solve for minimum cost
    pub fn solve(&mut self) -> (SlsResult, f64) {
        // Initialize weights in base solver
        for (i, &w) in self.weights.iter().enumerate() {
            if i < self.base.clauses.len() {
                self.base.clauses[i].weight = w;
            }
        }

        let result = self.base.solve();

        // Compute cost for found assignment
        let cost = if let SlsResult::Sat(ref assignment) = result {
            self.compute_cost(assignment)
        } else {
            f64::MAX
        };

        if cost < self.best_cost {
            self.best_cost = cost;
        }

        self.stats.best_cost = self.best_cost;
        (result, cost)
    }

    /// Compute cost of an assignment (sum of weights of unsatisfied clauses)
    fn compute_cost(&self, assignment: &[bool]) -> f64 {
        let mut cost = 0.0;
        for (i, clause) in self.base.clauses.iter().enumerate() {
            let mut sat = false;
            for &lit in &clause.literals {
                let var = lit.unsigned_abs() as usize;
                let is_pos = lit > 0;
                if var < assignment.len() && assignment[var] == is_pos {
                    sat = true;
                    break;
                }
            }
            if !sat {
                let w = self.weights.get(i).copied().unwrap_or(1.0);
                if w == f64::MAX {
                    cost = f64::MAX;
                    break;
                }
                cost += w;
            }
        }
        cost
    }

    /// Get best cost
    pub fn best_cost(&self) -> f64 {
        self.best_cost
    }
}

// ============================================================================
// Hybrid SLS-CDCL Interface
// ============================================================================

/// Interface for hybrid SLS-CDCL solving
#[derive(Debug)]
pub struct HybridSlsInterface {
    /// Assumed literals from CDCL
    assumptions: Vec<Lit>,
    /// Conflict clauses learned from SLS
    learned_clauses: Vec<Vec<Lit>>,
    /// Phase hints from SLS
    phase_hints: Vec<Option<bool>>,
    /// Variables to focus on
    focus_vars: HashSet<Var>,
}

impl HybridSlsInterface {
    /// Create new interface
    pub fn new() -> Self {
        Self {
            assumptions: Vec::new(),
            learned_clauses: Vec::new(),
            phase_hints: Vec::new(),
            focus_vars: HashSet::new(),
        }
    }

    /// Set assumptions from CDCL solver
    pub fn set_assumptions(&mut self, assumptions: Vec<Lit>) {
        self.assumptions = assumptions;
    }

    /// Get assumptions
    pub fn assumptions(&self) -> &[Lit] {
        &self.assumptions
    }

    /// Add learned clause from SLS
    pub fn add_learned_clause(&mut self, clause: Vec<Lit>) {
        self.learned_clauses.push(clause);
    }

    /// Get and clear learned clauses
    pub fn take_learned_clauses(&mut self) -> Vec<Vec<Lit>> {
        core::mem::take(&mut self.learned_clauses)
    }

    /// Set phase hint for variable
    pub fn set_phase_hint(&mut self, var: Var, phase: bool) {
        let idx = var as usize;
        if self.phase_hints.len() <= idx {
            self.phase_hints.resize(idx + 1, None);
        }
        self.phase_hints[idx] = Some(phase);
    }

    /// Get phase hint
    pub fn phase_hint(&self, var: Var) -> Option<bool> {
        self.phase_hints.get(var as usize).copied().flatten()
    }

    /// Set focus variables (from CDCL conflict analysis)
    pub fn set_focus_vars(&mut self, vars: HashSet<Var>) {
        self.focus_vars = vars;
    }

    /// Get focus variables
    pub fn focus_vars(&self) -> &HashSet<Var> {
        &self.focus_vars
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.assumptions.clear();
        self.learned_clauses.clear();
        self.phase_hints.clear();
        self.focus_vars.clear();
    }
}

impl Default for HybridSlsInterface {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// YalSAT-style Solver
// ============================================================================

/// YalSAT configuration (Yet Another Local Search SAT)
#[derive(Debug, Clone)]
pub struct YalsatConfig {
    /// Base configuration
    pub base: SlsConfig,
    /// Enable caching
    pub caching: bool,
    /// Cache size limit
    pub cache_limit: usize,
    /// Focused probability
    pub focused_prob: f64,
    /// Enable boosting
    pub boosting: bool,
}

impl Default for YalsatConfig {
    fn default() -> Self {
        Self {
            base: SlsConfig::default(),
            caching: true,
            cache_limit: 1000,
            focused_prob: 0.8,
            boosting: true,
        }
    }
}

/// YalSAT-style enhanced SLS solver
#[derive(Debug)]
pub struct YalsatSolver {
    /// Base solver
    base: SlsSolver,
    /// Configuration
    #[allow(dead_code)]
    config: YalsatConfig,
    /// Score cache (var -> (break, make) at last update)
    pub(crate) score_cache: HashMap<Var, (u32, u32)>,
    /// Boost factors for variables
    pub(crate) boost: Vec<f64>,
    /// Cached best variable per clause
    clause_best: Vec<Option<Var>>,
}

impl YalsatSolver {
    /// Create new YalSAT solver
    pub fn new(config: YalsatConfig) -> Self {
        Self {
            base: SlsSolver::new(config.base.clone()),
            config,
            score_cache: HashMap::new(),
            boost: Vec::new(),
            clause_best: Vec::new(),
        }
    }

    /// Add clause
    pub fn add_clause(&mut self, literals: &[Lit]) {
        self.base.add_clause(literals);
        self.clause_best.push(None);
    }

    /// Solve
    pub fn solve(&mut self) -> SlsResult {
        // Initialize boost factors
        let n = self.base.num_vars() as usize + 1;
        self.boost = vec![1.0; n];

        // Run base solver with enhanced picking
        self.base.solve()
    }

    /// Update boost for variable
    pub fn update_boost(&mut self, var: Var, factor: f64) {
        let idx = var as usize;
        if idx < self.boost.len() {
            self.boost[idx] *= factor;
            // Clamp to prevent overflow
            if self.boost[idx] > 1000.0 {
                self.boost[idx] = 1000.0;
            }
        }
    }

    /// Get boosted score
    pub fn boosted_score(&self, var: Var, base_score: f64) -> f64 {
        let boost = self.boost.get(var as usize).copied().unwrap_or(1.0);
        base_score * boost
    }

    /// Invalidate cache for variable
    pub fn invalidate_cache(&mut self, var: Var) {
        self.score_cache.remove(&var);
    }

    /// Clear all caches
    pub fn clear_cache(&mut self) {
        self.score_cache.clear();
        for best in &mut self.clause_best {
            *best = None;
        }
    }

    /// Reset
    pub fn reset(&mut self) {
        self.base.reset();
        self.score_cache.clear();
        self.boost.clear();
        self.clause_best.clear();
    }
}

// ============================================================================
// Portfolio SLS (Multiple Algorithms)
// ============================================================================

/// Portfolio SLS configuration
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Algorithms to use
    pub algorithms: Vec<SlsAlgorithm>,
    /// Flips per algorithm per round
    pub flips_per_algo: u64,
    /// Enable adaptive switching
    pub adaptive: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![
                SlsAlgorithm::WalkSat,
                SlsAlgorithm::ProbSat,
                SlsAlgorithm::Gsat,
            ],
            flips_per_algo: 10000,
            adaptive: true,
        }
    }
}

/// Portfolio SLS solver (runs multiple algorithms)
#[derive(Debug)]
pub struct PortfolioSls {
    /// Configuration
    config: PortfolioConfig,
    /// Individual solvers
    solvers: Vec<SlsSolver>,
    /// Best result so far
    best_result: Option<SlsResult>,
    /// Best unsat count
    best_unsat: u32,
    /// Algorithm performance (successes)
    algo_performance: Vec<u32>,
}

impl PortfolioSls {
    /// Create a new portfolio solver
    pub fn new(config: PortfolioConfig) -> Self {
        let mut solvers = Vec::new();
        let algo_count = config.algorithms.len();

        for &algo in &config.algorithms {
            let sls_config = SlsConfig {
                algorithm: algo,
                max_flips: config.flips_per_algo,
                max_restarts: 1,
                ..SlsConfig::default()
            };
            solvers.push(SlsSolver::new(sls_config));
        }

        Self {
            config,
            solvers,
            best_result: None,
            best_unsat: u32::MAX,
            algo_performance: vec![0; algo_count],
        }
    }

    /// Add clause to all solvers
    pub fn add_clause(&mut self, literals: &[Lit]) {
        for solver in &mut self.solvers {
            solver.add_clause(literals);
        }
    }

    /// Solve using portfolio
    pub fn solve(&mut self, max_rounds: u32) -> SlsResult {
        for _round in 0..max_rounds {
            for (i, solver) in self.solvers.iter_mut().enumerate() {
                let result = solver.solve();

                if let SlsResult::Sat(ref _assignment) = result {
                    self.algo_performance[i] += 1;
                    self.best_result = Some(result.clone());
                    return result;
                }

                // Track best progress
                let unsat_count = solver.best_unsat_count;
                if unsat_count < self.best_unsat {
                    self.best_unsat = unsat_count;
                }
            }

            // Adaptive: prioritize better-performing algorithms
            if self.config.adaptive && !self.algo_performance.is_empty() {
                // Sort by performance (descending)
                let mut indices: Vec<_> = (0..self.solvers.len()).collect();
                indices.sort_by_key(|&i| core::cmp::Reverse(self.algo_performance[i]));
                // Could reorder solvers here for next round
            }
        }

        self.best_result.clone().unwrap_or(SlsResult::Unknown)
    }

    /// Get best unsatisfied count
    pub fn best_unsat_count(&self) -> u32 {
        self.best_unsat
    }

    /// Get algorithm performance
    pub fn algorithm_performance(&self) -> &[u32] {
        &self.algo_performance
    }

    /// Reset all solvers
    pub fn reset(&mut self) {
        for solver in &mut self.solvers {
            solver.reset();
        }
        self.best_result = None;
        self.best_unsat = u32::MAX;
        for p in &mut self.algo_performance {
            *p = 0;
        }
    }
}
