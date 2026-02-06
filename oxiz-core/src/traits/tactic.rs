//! Composable Tactic Trait.
#![allow(missing_docs, clippy::type_complexity)] // Under development - documentation in progress
//!
//! Tactics are transformations applied to goals (formulas) to simplify them,
//! decompose them, or prepare them for specific solvers.

use crate::{TermId, TermManager};
use rustc_hash::FxHashSet;

/// Result of applying a tactic.
#[derive(Debug, Clone)]
pub enum TacticResult {
    /// Tactic succeeded and produced new subgoals.
    Success {
        /// New subgoals to solve.
        subgoals: Vec<Goal>,
        /// Proof transformation (optional).
        proof_hint: Option<String>,
    },
    /// Tactic proved the goal (no subgoals remain).
    Proved,
    /// Tactic showed the goal is unsatisfiable.
    Unsat,
    /// Tactic failed or is not applicable.
    Failed(String),
    /// Tactic cannot determine the outcome.
    Unknown,
}

impl TacticResult {
    /// Check if the tactic succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, TacticResult::Success { .. })
    }

    /// Check if the tactic proved the goal.
    pub fn is_proved(&self) -> bool {
        matches!(self, TacticResult::Proved)
    }

    /// Check if the tactic showed unsatisfiability.
    pub fn is_unsat(&self) -> bool {
        matches!(self, TacticResult::Unsat)
    }

    /// Check if the tactic failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, TacticResult::Failed(_))
    }

    /// Get subgoals if successful.
    pub fn subgoals(&self) -> Option<&[Goal]> {
        match self {
            TacticResult::Success { subgoals, .. } => Some(subgoals),
            _ => None,
        }
    }
}

/// A goal to be solved.
#[derive(Debug, Clone)]
pub struct Goal {
    /// The formula representing this goal.
    pub formula: TermId,
    /// Assumptions (context).
    pub assumptions: Vec<TermId>,
    /// Metadata attached to this goal.
    pub metadata: GoalMetadata,
}

impl Goal {
    /// Create a new goal from a formula.
    pub fn new(formula: TermId) -> Self {
        Self {
            formula,
            assumptions: Vec::new(),
            metadata: GoalMetadata::default(),
        }
    }

    /// Create a goal with assumptions.
    pub fn with_assumptions(formula: TermId, assumptions: Vec<TermId>) -> Self {
        Self {
            formula,
            assumptions,
            metadata: GoalMetadata::default(),
        }
    }

    /// Add an assumption to this goal.
    pub fn add_assumption(&mut self, assumption: TermId) {
        self.assumptions.push(assumption);
    }

    /// Check if goal is trivially true.
    pub fn is_trivially_true(&self, _tm: &TermManager) -> bool {
        // Simplified: would check if formula is 'true' constant
        false
    }

    /// Check if goal is trivially false.
    pub fn is_trivially_false(&self, _tm: &TermManager) -> bool {
        // Simplified: would check if formula is 'false' constant
        false
    }

    /// Get all free variables in the goal.
    pub fn free_vars(&self, _tm: &TermManager) -> FxHashSet<TermId> {
        // Simplified: would traverse formula and collect variables
        FxHashSet::default()
    }
}

/// Metadata associated with a goal.
#[derive(Debug, Clone, Default)]
pub struct GoalMetadata {
    /// Depth of this goal in the tactic tree.
    pub depth: usize,
    /// Number of times tactics have been applied to this goal.
    pub tactic_applications: usize,
    /// Estimated difficulty (arbitrary units).
    pub difficulty: u64,
    /// Tags for categorizing goals.
    pub tags: Vec<String>,
}

/// Statistics for tactic application.
#[derive(Debug, Clone, Default)]
pub struct TacticStats {
    /// Number of times this tactic was applied.
    pub applications: u64,
    /// Number of successful applications.
    pub successes: u64,
    /// Number of failures.
    pub failures: u64,
    /// Total time spent (microseconds).
    pub total_time_us: u64,
    /// Number of subgoals produced.
    pub subgoals_produced: u64,
}

impl TacticStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute success rate.
    pub fn success_rate(&self) -> f64 {
        if self.applications == 0 {
            0.0
        } else {
            self.successes as f64 / self.applications as f64
        }
    }

    /// Average time per application (microseconds).
    pub fn avg_time_us(&self) -> f64 {
        if self.applications == 0 {
            0.0
        } else {
            self.total_time_us as f64 / self.applications as f64
        }
    }
}

/// Configuration for tactic execution.
#[derive(Debug, Clone)]
pub struct TacticConfig {
    /// Maximum depth for recursive tactic application.
    pub max_depth: usize,
    /// Timeout for tactic execution (microseconds).
    pub timeout_us: Option<u64>,
    /// Maximum number of subgoals to produce.
    pub max_subgoals: usize,
    /// Enable proof generation.
    pub generate_proofs: bool,
}

impl Default for TacticConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            timeout_us: None,
            max_subgoals: 1000,
            generate_proofs: false,
        }
    }
}

/// Core trait for tactics.
///
/// A tactic transforms goals into (possibly empty) sets of subgoals.
pub trait Tactic: Send + Sync {
    /// Get the name of this tactic.
    fn name(&self) -> &str;

    /// Apply the tactic to a goal.
    fn apply(&mut self, goal: &Goal, tm: &mut TermManager) -> TacticResult;

    /// Check if this tactic is applicable to a goal.
    fn is_applicable(&self, _goal: &Goal, _tm: &TermManager) -> bool {
        // Default: always applicable
        true
    }

    /// Reset the tactic state.
    fn reset(&mut self) {
        // Default: no-op
    }

    /// Get statistics for this tactic.
    fn stats(&self) -> TacticStats {
        TacticStats::default()
    }

    /// Estimate the cost of applying this tactic.
    fn estimated_cost(&self, _goal: &Goal, _tm: &TermManager) -> u64 {
        100 // Default cost
    }
}

/// Extension trait for iterative tactics.
///
/// Iterative tactics can be applied multiple times until a fixpoint is reached.
pub trait IterativeTactic: Tactic {
    /// Apply the tactic until no more changes occur.
    fn apply_until_fixpoint(&mut self, goal: &Goal, tm: &mut TermManager) -> TacticResult {
        let mut current_goal = goal.clone();
        let max_iterations = 10;

        for _iter in 0..max_iterations {
            let result = self.apply(&current_goal, tm);

            match result {
                TacticResult::Success { ref subgoals, .. } => {
                    if subgoals.len() == 1 {
                        // Single subgoal: continue iterating
                        current_goal = subgoals[0].clone();
                    } else {
                        // Multiple subgoals: stop iterating
                        return result;
                    }
                }
                _ => return result,
            }
        }

        TacticResult::Success {
            subgoals: vec![current_goal],
            proof_hint: None,
        }
    }
}

/// Combinator for composing tactics.
pub enum TacticCombinator {
    /// Apply tactics in sequence (AndThen).
    Sequential(Vec<Box<dyn Tactic>>),
    /// Try tactics in order until one succeeds (OrElse).
    Fallback(Vec<Box<dyn Tactic>>),
    /// Apply tactic repeatedly until fixpoint (Repeat).
    Repeat {
        tactic: Box<dyn Tactic>,
        max_iterations: usize,
    },
    /// Apply tactic if condition holds (IfThen).
    Conditional {
        condition: Box<dyn Fn(&Goal, &TermManager) -> bool + Send + Sync>,
        tactic: Box<dyn Tactic>,
    },
    /// Apply tactics in parallel and merge results (Par).
    Parallel(Vec<Box<dyn Tactic>>),
}

impl TacticCombinator {
    /// Create a sequential combinator.
    pub fn seq(tactics: Vec<Box<dyn Tactic>>) -> Self {
        TacticCombinator::Sequential(tactics)
    }

    /// Create a fallback combinator.
    pub fn fallback(tactics: Vec<Box<dyn Tactic>>) -> Self {
        TacticCombinator::Fallback(tactics)
    }

    /// Create a repeat combinator.
    pub fn repeat(tactic: Box<dyn Tactic>, max_iterations: usize) -> Self {
        TacticCombinator::Repeat {
            tactic,
            max_iterations,
        }
    }
}

impl Tactic for TacticCombinator {
    fn name(&self) -> &str {
        match self {
            TacticCombinator::Sequential(_) => "sequential",
            TacticCombinator::Fallback(_) => "fallback",
            TacticCombinator::Repeat { .. } => "repeat",
            TacticCombinator::Conditional { .. } => "conditional",
            TacticCombinator::Parallel(_) => "parallel",
        }
    }

    fn apply(&mut self, goal: &Goal, tm: &mut TermManager) -> TacticResult {
        match self {
            TacticCombinator::Sequential(tactics) => {
                let mut current_goals = vec![goal.clone()];

                for tactic in tactics {
                    let mut next_goals = Vec::new();

                    for g in current_goals {
                        let result = tactic.apply(&g, tm);

                        match result {
                            TacticResult::Success { subgoals, .. } => {
                                next_goals.extend(subgoals);
                            }
                            TacticResult::Proved => {
                                // This goal is proved, don't add to next_goals
                            }
                            TacticResult::Unsat => {
                                return TacticResult::Unsat;
                            }
                            TacticResult::Failed(msg) => {
                                return TacticResult::Failed(msg);
                            }
                            TacticResult::Unknown => {
                                next_goals.push(g);
                            }
                        }
                    }

                    current_goals = next_goals;
                }

                if current_goals.is_empty() {
                    TacticResult::Proved
                } else {
                    TacticResult::Success {
                        subgoals: current_goals,
                        proof_hint: None,
                    }
                }
            }

            TacticCombinator::Fallback(tactics) => {
                for tactic in tactics {
                    let result = tactic.apply(goal, tm);

                    if !result.is_failed() {
                        return result;
                    }
                }

                TacticResult::Failed("All tactics in fallback failed".to_string())
            }

            TacticCombinator::Repeat {
                tactic,
                max_iterations,
            } => {
                let mut current_goal = goal.clone();

                for _iter in 0..*max_iterations {
                    let result = tactic.apply(&current_goal, tm);

                    match result {
                        TacticResult::Success { ref subgoals, .. } => {
                            if subgoals.len() == 1 {
                                current_goal = subgoals[0].clone();
                            } else {
                                return result;
                            }
                        }
                        _ => return result,
                    }
                }

                TacticResult::Success {
                    subgoals: vec![current_goal],
                    proof_hint: None,
                }
            }

            TacticCombinator::Conditional { condition, tactic } => {
                if condition(goal, tm) {
                    tactic.apply(goal, tm)
                } else {
                    TacticResult::Success {
                        subgoals: vec![goal.clone()],
                        proof_hint: None,
                    }
                }
            }

            TacticCombinator::Parallel(_) => {
                // Simplified: would run tactics in parallel
                TacticResult::Unknown
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock tactic for testing
    struct MockTactic {
        name: String,
        should_succeed: bool,
    }

    impl MockTactic {
        fn new(name: &str, should_succeed: bool) -> Self {
            Self {
                name: name.to_string(),
                should_succeed,
            }
        }
    }

    impl Tactic for MockTactic {
        fn name(&self) -> &str {
            &self.name
        }

        fn apply(&mut self, goal: &Goal, _tm: &mut TermManager) -> TacticResult {
            if self.should_succeed {
                TacticResult::Success {
                    subgoals: vec![goal.clone()],
                    proof_hint: None,
                }
            } else {
                TacticResult::Failed("mock failure".to_string())
            }
        }
    }

    #[test]
    fn test_goal_creation() {
        let goal = Goal::new(TermId::new(0));
        assert_eq!(goal.formula, TermId::new(0));
        assert!(goal.assumptions.is_empty());
    }

    #[test]
    fn test_goal_with_assumptions() {
        let goal = Goal::with_assumptions(TermId::new(0), vec![TermId::new(1), TermId::new(2)]);
        assert_eq!(goal.assumptions.len(), 2);
    }

    #[test]
    fn test_tactic_result() {
        let proved = TacticResult::Proved;
        assert!(proved.is_proved());
        assert!(!proved.is_success());

        let unsat = TacticResult::Unsat;
        assert!(unsat.is_unsat());

        let failed = TacticResult::Failed("error".to_string());
        assert!(failed.is_failed());
    }

    #[test]
    fn test_tactic_stats() {
        let mut stats = TacticStats::new();
        assert_eq!(stats.applications, 0);

        stats.applications = 100;
        stats.successes = 75;
        assert_eq!(stats.success_rate(), 0.75);
    }

    #[test]
    fn test_sequential_combinator() {
        let tactics: Vec<Box<dyn Tactic>> = vec![
            Box::new(MockTactic::new("t1", true)),
            Box::new(MockTactic::new("t2", true)),
        ];

        let combinator = TacticCombinator::seq(tactics);
        assert_eq!(combinator.name(), "sequential");
    }

    #[test]
    fn test_fallback_combinator() {
        let tactics: Vec<Box<dyn Tactic>> = vec![
            Box::new(MockTactic::new("t1", false)),
            Box::new(MockTactic::new("t2", true)),
        ];

        let combinator = TacticCombinator::fallback(tactics);
        assert_eq!(combinator.name(), "fallback");
    }
}
