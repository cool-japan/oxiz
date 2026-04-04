//! Repair/state components for SLS:
//! PhaseSaver, BackboneDetector, DiversificationManager,
//! ClauseSimplifier, SolutionVerifier, SolutionLearner.

use super::types::{Lit, Var};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Phase Saving and Polarity Heuristics
// ============================================================================

/// Phase saving mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhaseMode {
    /// Always start with false
    False,
    /// Always start with true
    True,
    /// Random phase
    Random,
    /// Save previous phase
    #[default]
    Save,
    /// Use polarity from unit propagation
    Unit,
}

/// Phase saver for SLS
#[derive(Debug)]
pub struct PhaseSaver {
    /// Saved phases
    phases: Vec<Option<bool>>,
    /// Mode
    mode: PhaseMode,
    /// Random seed
    rng_state: u64,
}

impl PhaseSaver {
    /// Create a new phase saver
    pub fn new(mode: PhaseMode) -> Self {
        Self {
            phases: Vec::new(),
            mode,
            rng_state: 42,
        }
    }

    /// Ensure capacity for n variables
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.phases.len() < n {
            self.phases.resize(n, None);
        }
    }

    /// Get phase for a variable
    pub fn get_phase(&mut self, var: Var) -> bool {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);

        match self.mode {
            PhaseMode::False => false,
            PhaseMode::True => true,
            PhaseMode::Random => {
                let x = &mut self.rng_state;
                *x ^= *x << 13;
                *x ^= *x >> 7;
                *x ^= *x << 17;
                *x & 1 == 0
            }
            PhaseMode::Save | PhaseMode::Unit => self.phases[idx].unwrap_or(false),
        }
    }

    /// Save phase for a variable
    pub fn save_phase(&mut self, var: Var, phase: bool) {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);
        self.phases[idx] = Some(phase);
    }

    /// Reset all phases
    pub fn reset(&mut self) {
        for phase in &mut self.phases {
            *phase = None;
        }
    }
}

// ============================================================================
// Backbone Detection
// ============================================================================

/// Backbone analysis (variables that have the same value in all solutions)
#[derive(Debug)]
pub struct BackboneDetector {
    /// Known backbone literals (positive = must be true, negative = must be false)
    backbone: Vec<Lit>,
    /// Candidate backbone literals
    candidates: HashSet<Lit>,
    /// Solutions seen
    solutions_seen: u32,
    /// Minimum solutions before declaring backbone
    min_solutions: u32,
}

impl BackboneDetector {
    /// Create a new backbone detector
    pub fn new(min_solutions: u32) -> Self {
        Self {
            backbone: Vec::new(),
            candidates: HashSet::new(),
            solutions_seen: 0,
            min_solutions,
        }
    }

    /// Initialize candidates from first solution
    pub fn initialize(&mut self, solution: &[bool]) {
        self.candidates.clear();
        for (i, &val) in solution.iter().enumerate().skip(1) {
            let lit = if val { i as i32 } else { -(i as i32) };
            self.candidates.insert(lit);
        }
        self.solutions_seen = 1;
    }

    /// Update with new solution
    pub fn update(&mut self, solution: &[bool]) {
        if self.candidates.is_empty() {
            self.initialize(solution);
            return;
        }

        // Remove candidates that don't match this solution
        self.candidates.retain(|&lit| {
            let var = lit.unsigned_abs() as usize;
            let expected = lit > 0;
            var < solution.len() && solution[var] == expected
        });

        self.solutions_seen += 1;

        // If we've seen enough solutions, commit candidates to backbone
        if self.solutions_seen >= self.min_solutions {
            self.backbone.extend(self.candidates.iter());
            self.candidates.clear();
        }
    }

    /// Get the detected backbone
    pub fn backbone(&self) -> &[Lit] {
        &self.backbone
    }

    /// Check if variable is in backbone
    pub fn is_backbone(&self, var: Var) -> Option<bool> {
        let pos = var as i32;
        let neg = -(var as i32);

        if self.backbone.contains(&pos) || self.candidates.contains(&pos) {
            Some(true)
        } else if self.backbone.contains(&neg) || self.candidates.contains(&neg) {
            Some(false)
        } else {
            None
        }
    }

    /// Number of backbone variables detected
    pub fn backbone_size(&self) -> usize {
        self.backbone.len() + self.candidates.len()
    }

    /// Reset
    pub fn reset(&mut self) {
        self.backbone.clear();
        self.candidates.clear();
        self.solutions_seen = 0;
    }
}

// ============================================================================
// Diversification Methods
// ============================================================================

/// Diversification strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiversificationStrategy {
    /// No diversification
    None,
    /// Random walk
    RandomWalk,
    /// Stagnation-based
    #[default]
    Stagnation,
    /// Configuration changing
    ConfigChange,
}

/// Diversification manager
#[derive(Debug)]
pub struct DiversificationManager {
    /// Strategy
    strategy: DiversificationStrategy,
    /// Stagnation counter
    stagnation_count: u32,
    /// Stagnation threshold
    stagnation_threshold: u32,
    /// Last best score
    last_best: u32,
    /// Configuration change rate
    config_change_rate: f64,
}

impl DiversificationManager {
    /// Create a new diversification manager
    pub fn new(strategy: DiversificationStrategy) -> Self {
        Self {
            strategy,
            stagnation_count: 0,
            stagnation_threshold: 100,
            last_best: u32::MAX,
            config_change_rate: 0.1,
        }
    }

    /// Update with current state
    pub fn update(&mut self, current_unsat: u32) {
        if current_unsat >= self.last_best {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
            self.last_best = current_unsat;
        }
    }

    /// Check if should diversify
    pub fn should_diversify(&self) -> bool {
        match self.strategy {
            DiversificationStrategy::None => false,
            DiversificationStrategy::RandomWalk => true, // Always consider
            DiversificationStrategy::Stagnation => {
                self.stagnation_count >= self.stagnation_threshold
            }
            DiversificationStrategy::ConfigChange => {
                self.stagnation_count >= self.stagnation_threshold / 2
            }
        }
    }

    /// Get diversification probability
    pub fn diversify_prob(&self) -> f64 {
        match self.strategy {
            DiversificationStrategy::None => 0.0,
            DiversificationStrategy::RandomWalk => 0.01,
            DiversificationStrategy::Stagnation => {
                if self.stagnation_count >= self.stagnation_threshold {
                    0.5
                } else {
                    0.0
                }
            }
            DiversificationStrategy::ConfigChange => self.config_change_rate,
        }
    }

    /// Reset
    pub fn reset(&mut self) {
        self.stagnation_count = 0;
        self.last_best = u32::MAX;
    }
}

// ============================================================================
// Clause Subsumption and Simplification
// ============================================================================

/// Clause simplifier for SLS
#[derive(Debug)]
pub struct ClauseSimplifier {
    /// Occurrence lists: literal -> clause indices
    pub(super) occurrences: HashMap<Lit, Vec<usize>>,
    /// Clause sizes
    pub(super) clause_sizes: Vec<usize>,
    /// Deleted clauses
    deleted: HashSet<usize>,
}

impl ClauseSimplifier {
    /// Create a new clause simplifier
    pub fn new() -> Self {
        Self {
            occurrences: HashMap::new(),
            clause_sizes: Vec::new(),
            deleted: HashSet::new(),
        }
    }

    /// Build occurrence lists
    pub fn build(&mut self, clauses: &[Vec<Lit>]) {
        self.occurrences.clear();
        self.clause_sizes.clear();
        self.deleted.clear();

        for (i, clause) in clauses.iter().enumerate() {
            self.clause_sizes.push(clause.len());
            for &lit in clause {
                self.occurrences.entry(lit).or_default().push(i);
            }
        }
    }

    /// Check if clause a subsumes clause b
    pub fn subsumes(&self, clause_a: &[Lit], clause_b: &[Lit]) -> bool {
        if clause_a.len() > clause_b.len() {
            return false;
        }

        let b_set: HashSet<Lit> = clause_b.iter().copied().collect();
        clause_a.iter().all(|lit| b_set.contains(lit))
    }

    /// Simplify by subsumption
    pub fn simplify_subsumption(&mut self, clauses: &[Vec<Lit>]) -> Vec<usize> {
        let mut to_delete = Vec::new();

        for (i, clause_a) in clauses.iter().enumerate() {
            if self.deleted.contains(&i) {
                continue;
            }

            // Check if this clause subsumes any other
            if let Some(&lit) = clause_a.first()
                && let Some(candidates) = self.occurrences.get(&lit)
            {
                for &j in candidates {
                    if i != j && !self.deleted.contains(&j) && self.subsumes(clause_a, &clauses[j])
                    {
                        to_delete.push(j);
                        self.deleted.insert(j);
                    }
                }
            }
        }

        to_delete
    }

    /// Find unit clauses
    pub fn find_units(&self, clauses: &[Vec<Lit>]) -> Vec<Lit> {
        clauses
            .iter()
            .enumerate()
            .filter(|(i, clause)| clause.len() == 1 && !self.deleted.contains(i))
            .map(|(_, clause)| clause[0])
            .collect()
    }

    /// Propagate units
    pub fn propagate_unit(&mut self, unit: Lit, clauses: &mut [Vec<Lit>]) {
        let neg_unit = -unit;

        // Remove clauses containing the unit literal
        if let Some(indices) = self.occurrences.get(&unit).cloned() {
            for i in indices {
                self.deleted.insert(i);
            }
        }

        // Remove negation from clauses
        if let Some(indices) = self.occurrences.get(&neg_unit).cloned() {
            for i in indices {
                if !self.deleted.contains(&i) {
                    clauses[i].retain(|&lit| lit != neg_unit);
                }
            }
        }
    }

    /// Check if formula is satisfiable (trivial check)
    pub fn is_trivially_unsat(&self, clauses: &[Vec<Lit>]) -> bool {
        clauses
            .iter()
            .enumerate()
            .any(|(i, clause)| clause.is_empty() && !self.deleted.contains(&i))
    }

    /// Get deleted clause count
    pub fn deleted_count(&self) -> usize {
        self.deleted.len()
    }
}

impl Default for ClauseSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Solution Verification
// ============================================================================

/// Solution verifier for SLS
#[derive(Debug)]
pub struct SolutionVerifier {
    /// Cached clause data
    clauses: Vec<Vec<Lit>>,
}

impl SolutionVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
        }
    }

    /// Set clauses to verify against
    pub fn set_clauses(&mut self, clauses: Vec<Vec<Lit>>) {
        self.clauses = clauses;
    }

    /// Verify a solution
    pub fn verify(&self, assignment: &[bool]) -> VerificationResult {
        let mut satisfied = 0;
        let mut unsatisfied = Vec::new();

        for (i, clause) in self.clauses.iter().enumerate() {
            let mut clause_sat = false;
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                let expected = lit > 0;
                if var < assignment.len() && assignment[var] == expected {
                    clause_sat = true;
                    break;
                }
            }
            if clause_sat {
                satisfied += 1;
            } else {
                unsatisfied.push(i);
            }
        }

        VerificationResult {
            is_valid: unsatisfied.is_empty(),
            satisfied_count: satisfied,
            unsatisfied_indices: unsatisfied,
        }
    }

    /// Quick check (returns true if all clauses satisfied)
    pub fn is_valid(&self, assignment: &[bool]) -> bool {
        for clause in &self.clauses {
            let mut clause_sat = false;
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                let expected = lit > 0;
                if var < assignment.len() && assignment[var] == expected {
                    clause_sat = true;
                    break;
                }
            }
            if !clause_sat {
                return false;
            }
        }
        true
    }
}

impl Default for SolutionVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of solution verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Is the solution valid (all clauses satisfied)?
    pub is_valid: bool,
    /// Number of satisfied clauses
    pub satisfied_count: usize,
    /// Indices of unsatisfied clauses
    pub unsatisfied_indices: Vec<usize>,
}

// ============================================================================
// Learning from Solutions
// ============================================================================

/// Learns variable polarities from found solutions
#[derive(Debug)]
pub struct SolutionLearner {
    /// Polarity counts: (true_count, false_count) for each variable
    polarity_counts: Vec<(u32, u32)>,
    /// Solutions collected
    solutions_count: u32,
    /// Maximum solutions to track
    max_solutions: u32,
}

impl SolutionLearner {
    /// Create new learner
    pub fn new(max_solutions: u32) -> Self {
        Self {
            polarity_counts: Vec::new(),
            solutions_count: 0,
            max_solutions,
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.polarity_counts = vec![(0, 0); n];
        self.solutions_count = 0;
    }

    /// Record a solution
    pub fn record_solution(&mut self, assignment: &[bool]) {
        if self.solutions_count >= self.max_solutions {
            return;
        }

        for (i, &val) in assignment.iter().enumerate() {
            if i < self.polarity_counts.len() {
                if val {
                    self.polarity_counts[i].0 += 1;
                } else {
                    self.polarity_counts[i].1 += 1;
                }
            }
        }
        self.solutions_count += 1;
    }

    /// Get preferred polarity for variable
    pub fn preferred_polarity(&self, var: Var) -> Option<bool> {
        let idx = var as usize;
        if idx < self.polarity_counts.len() {
            let (true_count, false_count) = self.polarity_counts[idx];
            if true_count > false_count {
                Some(true)
            } else if false_count > true_count {
                Some(false)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get polarity confidence (0.0 to 1.0)
    pub fn polarity_confidence(&self, var: Var) -> f64 {
        let idx = var as usize;
        if idx < self.polarity_counts.len() {
            let (true_count, false_count) = self.polarity_counts[idx];
            let total = true_count + false_count;
            if total == 0 {
                return 0.5;
            }
            let max_count = true_count.max(false_count);
            max_count as f64 / total as f64
        } else {
            0.5
        }
    }

    /// Get high-confidence variables
    pub fn high_confidence_vars(&self, threshold: f64) -> Vec<(Var, bool)> {
        self.polarity_counts
            .iter()
            .enumerate()
            .filter_map(|(i, &(t, f))| {
                let total = t + f;
                if total == 0 {
                    return None;
                }
                let confidence = (t.max(f) as f64) / (total as f64);
                if confidence >= threshold {
                    Some((i as Var, t > f))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Reset
    pub fn reset(&mut self) {
        for (t, f) in &mut self.polarity_counts {
            *t = 0;
            *f = 0;
        }
        self.solutions_count = 0;
    }
}
