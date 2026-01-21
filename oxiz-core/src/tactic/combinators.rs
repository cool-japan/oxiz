//! Tactic combinators for composing tactics.

use super::core::*;
use crate::error::Result;

/// Sequential combinator - applies tactics in sequence
pub struct ThenTactic {
    tactics: Vec<Box<dyn Tactic>>,
}

impl std::fmt::Debug for ThenTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThenTactic")
            .field("tactics_count", &self.tactics.len())
            .finish()
    }
}

impl ThenTactic {
    /// Create a new sequential combinator
    pub fn new(tactics: Vec<Box<dyn Tactic>>) -> Self {
        Self { tactics }
    }
}

impl Tactic for ThenTactic {
    fn name(&self) -> &str {
        "then"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let mut current_goals = vec![goal.clone()];

        for tactic in &self.tactics {
            let mut next_goals = Vec::new();

            for g in &current_goals {
                match tactic.apply(g)? {
                    TacticResult::Solved(result) => {
                        return Ok(TacticResult::Solved(result));
                    }
                    TacticResult::SubGoals(sub) => {
                        next_goals.extend(sub);
                    }
                    TacticResult::NotApplicable => {
                        next_goals.push(g.clone());
                    }
                    TacticResult::Failed(msg) => {
                        return Ok(TacticResult::Failed(msg));
                    }
                }
            }

            current_goals = next_goals;
        }

        Ok(TacticResult::SubGoals(current_goals))
    }
}

/// Combinator: try tactics in order, use first that applies
pub struct OrElseTactic {
    tactics: Vec<Box<dyn Tactic>>,
}

impl std::fmt::Debug for OrElseTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrElseTactic")
            .field("tactics_count", &self.tactics.len())
            .finish()
    }
}

impl OrElseTactic {
    /// Create a new or-else combinator
    pub fn new(tactics: Vec<Box<dyn Tactic>>) -> Self {
        Self { tactics }
    }
}

impl Tactic for OrElseTactic {
    fn name(&self) -> &str {
        "or-else"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        for tactic in &self.tactics {
            match tactic.apply(goal)? {
                TacticResult::NotApplicable => continue,
                result => return Ok(result),
            }
        }
        Ok(TacticResult::NotApplicable)
    }
}
/// Repeat combinator - applies a tactic repeatedly until fixpoint
pub struct RepeatTactic {
    tactic: Box<dyn Tactic>,
    max_iterations: usize,
}

impl std::fmt::Debug for RepeatTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepeatTactic")
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl RepeatTactic {
    /// Create a new repeat combinator
    pub fn new(tactic: Box<dyn Tactic>, max_iterations: usize) -> Self {
        Self {
            tactic,
            max_iterations,
        }
    }
}

impl Tactic for RepeatTactic {
    fn name(&self) -> &str {
        "repeat"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let mut current = goal.clone();

        for _ in 0..self.max_iterations {
            match self.tactic.apply(&current)? {
                TacticResult::Solved(result) => {
                    return Ok(TacticResult::Solved(result));
                }
                TacticResult::SubGoals(sub) if sub.len() == 1 => {
                    if sub[0].assertions == current.assertions {
                        // Fixpoint reached
                        break;
                    }
                    if let Some(next) = sub.into_iter().next() {
                        current = next;
                    } else {
                        break;
                    }
                }
                result => return Ok(result),
            }
        }

        Ok(TacticResult::SubGoals(vec![current]))
    }
}

/// Parallel combinator - runs multiple tactics concurrently
///
/// This combinator executes multiple tactics concurrently using threads
/// and returns the result from the first tactic that completes successfully.
///
/// The parallel tactic will:
///
/// - Run all tactics concurrently
/// - Return the first `Solved` result if any tactic solves the goal
/// - Return the first `SubGoals` result if no tactic solves but one produces subgoals
/// - Return `Failed` if all tactics fail
pub struct ParallelTactic {
    tactics: Vec<std::sync::Arc<dyn Tactic>>,
}

impl std::fmt::Debug for ParallelTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelTactic")
            .field("tactics_count", &self.tactics.len())
            .finish()
    }
}

impl ParallelTactic {
    /// Create a new parallel combinator from Arc-wrapped tactics
    pub fn new(tactics: Vec<std::sync::Arc<dyn Tactic>>) -> Self {
        Self { tactics }
    }

    /// Create a new parallel combinator from boxed tactics
    pub fn from_boxes(tactics: Vec<Box<dyn Tactic>>) -> Self {
        Self {
            tactics: tactics
                .into_iter()
                .map(|t| -> std::sync::Arc<dyn Tactic> { t.into() })
                .collect(),
        }
    }
}

impl Tactic for ParallelTactic {
    fn name(&self) -> &str {
        "parallel"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        use std::sync::mpsc;
        use std::thread;

        if self.tactics.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        if self.tactics.len() == 1 {
            // No need for parallelism with a single tactic
            return self.tactics[0].apply(goal);
        }

        let (tx, rx) = mpsc::channel();

        // Spawn a thread for each tactic
        let handles: Vec<_> = self
            .tactics
            .iter()
            .enumerate()
            .map(|(idx, tactic)| {
                let goal_clone = goal.clone();
                let tx_clone = tx.clone();
                let tactic_clone = std::sync::Arc::clone(tactic);

                thread::spawn(move || {
                    let result = tactic_clone.apply(&goal_clone);
                    let _ = tx_clone.send((idx, result));
                })
            })
            .collect();

        // Drop the original sender so the receiver knows when all threads are done
        drop(tx);

        // Collect results
        let mut results = Vec::new();
        while let Ok((idx, result)) = rx.recv() {
            results.push((idx, result));
        }

        // Wait for all threads to complete
        for handle in handles {
            let _ = handle.join();
        }

        // Process results in priority order:
        // 1. First Solved result
        // 2. First SubGoals result
        // 3. NotApplicable if all are NotApplicable
        // 4. Failed otherwise

        let mut has_subgoals = None;
        let mut all_not_applicable = true;

        for (_idx, result) in results {
            match result {
                Ok(TacticResult::Solved(solve_result)) => {
                    return Ok(TacticResult::Solved(solve_result));
                }
                Ok(TacticResult::SubGoals(sub)) => {
                    if has_subgoals.is_none() {
                        has_subgoals = Some(sub);
                    }
                    all_not_applicable = false;
                }
                Ok(TacticResult::NotApplicable) => {}
                Ok(TacticResult::Failed(_)) | Err(_) => {
                    all_not_applicable = false;
                }
            }
        }

        if let Some(subgoals) = has_subgoals {
            Ok(TacticResult::SubGoals(subgoals))
        } else if all_not_applicable {
            Ok(TacticResult::NotApplicable)
        } else {
            Ok(TacticResult::Failed(
                "All parallel tactics failed".to_string(),
            ))
        }
    }

    fn description(&self) -> &str {
        "Run tactics in parallel and return first successful result"
    }
}

/// Timeout tactic - applies a tactic with a time limit
/// This tactic wraps another tactic and enforces a maximum execution time.
/// If the wrapped tactic doesn't complete within the timeout, it returns Failed.
/// Timeout combinator - limits execution time of a tactic
pub struct TimeoutTactic {
    tactic: std::sync::Arc<dyn Tactic>,
    timeout_ms: u64,
}

impl std::fmt::Debug for TimeoutTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeoutTactic")
            .field("tactic_name", &self.tactic.name())
            .field("timeout_ms", &self.timeout_ms)
            .finish()
    }
}

impl TimeoutTactic {
    /// Create a new timeout tactic
    ///
    /// # Arguments
    /// * `tactic` - The tactic to run with a timeout
    /// * `timeout_ms` - Timeout in milliseconds
    pub fn new(tactic: std::sync::Arc<dyn Tactic>, timeout_ms: u64) -> Self {
        Self { tactic, timeout_ms }
    }

    /// Create a new timeout tactic from a boxed tactic
    pub fn from_box(tactic: Box<dyn Tactic>, timeout_ms: u64) -> Self {
        Self {
            tactic: tactic.into(),
            timeout_ms,
        }
    }
}

impl Tactic for TimeoutTactic {
    fn name(&self) -> &str {
        "timeout"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let (tx, rx) = mpsc::channel();
        let goal_clone = goal.clone();
        let tactic_clone = std::sync::Arc::clone(&self.tactic);

        // Spawn a thread to run the tactic
        let handle = thread::spawn(move || {
            let result = tactic_clone.apply(&goal_clone);
            let _ = tx.send(result);
        });

        // Wait for result with timeout
        match rx.recv_timeout(Duration::from_millis(self.timeout_ms)) {
            Ok(result) => {
                // Tactic completed within timeout
                let _ = handle.join();
                result
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Timeout exceeded
                // Note: The thread will continue running but we ignore its result
                Ok(TacticResult::Failed(format!(
                    "Tactic '{}' timed out after {}ms",
                    self.tactic.name(),
                    self.timeout_ms
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Thread panicked or dropped sender
                let _ = handle.join();
                Ok(TacticResult::Failed(format!(
                    "Tactic '{}' failed unexpectedly",
                    self.tactic.name()
                )))
            }
        }
    }

    fn description(&self) -> &str {
        "Apply a tactic with a time limit"
    }
}
