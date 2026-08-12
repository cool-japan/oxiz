//! Tactic combinators for composing tactics.

use super::core::*;
use crate::error::Result;
#[allow(unused_imports)]
use crate::prelude::*;
use core::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Grace period (in milliseconds) [`TimeoutTactic`] waits, after its
/// deadline elapses, for a cooperative worker to notice
/// [`cancellation_requested`] and finish before handing the still-running
/// `JoinHandle` off to a background reaper. Capped by `timeout_ms` itself
/// so a very short timeout does not turn into a long extra wait.
const TIMEOUT_GRACE_PERIOD_MS: u64 = 500;

thread_local! {
    /// Cancellation flag for the [`Tactic::apply`] call currently
    /// executing on this thread, installed by an enclosing
    /// [`TimeoutTactic`] worker before it invokes the wrapped tactic.
    /// `None` when no `TimeoutTactic` is on the call stack for this
    /// thread.
    static TACTIC_CANCEL_FLAG: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Returns `true` if the [`Tactic`] currently executing on this thread has
/// been asked to stop by an enclosing [`TimeoutTactic`] whose deadline has
/// elapsed.
///
/// Long-running or looping tactics should poll this periodically (e.g.
/// once per iteration of a fixpoint loop) and bail out promptly —
/// returning `Ok(TacticResult::Failed(..))` — when it becomes `true`, so
/// that `TimeoutTactic` can reclaim the worker thread within its grace
/// period instead of leaving it to run to completion in the background.
/// Tactics that never poll this still terminate correctly (the wrapping
/// thread is always eventually joined, see `TimeoutTactic::apply`), just
/// not promptly.
pub fn cancellation_requested() -> bool {
    TACTIC_CANCEL_FLAG.with(|flag| {
        flag.borrow()
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    })
}

/// Sequential combinator - applies tactics in sequence
pub struct ThenTactic {
    tactics: Vec<Box<dyn Tactic>>,
}

impl core::fmt::Debug for ThenTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
        // `current_goals` accumulates the (conjunctive) set of proof
        // obligations still outstanding: as with every other tactic in this
        // module that produces `SubGoals`, discharging the original goal
        // requires discharging *all* of them, not just the first one a
        // sub-tactic happens to solve.
        let mut current_goals = vec![goal.clone()];

        for tactic in &self.tactics {
            let mut next_goals = Vec::new();

            for g in &current_goals {
                match tactic.apply(g)? {
                    // One conjunct being unsatisfiable makes the whole
                    // conjunctive goal set unsatisfiable — sound to
                    // short-circuit.
                    TacticResult::Solved(SolveResult::Unsat) => {
                        return Ok(TacticResult::Solved(SolveResult::Unsat));
                    }
                    // Sat/Unknown for *this one* subgoal says nothing about
                    // any *other* still-pending subgoal from an earlier
                    // split — it must not be returned as the verdict for
                    // the whole goal set. Simply drop this now-discharged
                    // subgoal and keep processing the rest.
                    TacticResult::Solved(SolveResult::Sat | SolveResult::Unknown) => {}
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

        if current_goals.is_empty() {
            // Every subgoal across every tactic in the sequence was fully
            // discharged (and none came back Unsat), so the original goal
            // is proved.
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(current_goals))
    }
}

/// Combinator: try tactics in order, use first that applies
pub struct OrElseTactic {
    tactics: Vec<Box<dyn Tactic>>,
}

impl core::fmt::Debug for OrElseTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

impl core::fmt::Debug for RepeatTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

impl core::fmt::Debug for ParallelTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

impl core::fmt::Debug for TimeoutTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_for_worker = Arc::clone(&cancel);

        // Spawn a thread to run the tactic. The worker installs the
        // shared cancellation flag into its own thread-local slot so that
        // a cooperative tactic calling `cancellation_requested()` (or
        // anything it calls) observes cancellation as soon as this
        // `TimeoutTactic` gives up waiting.
        let handle = thread::spawn(move || {
            TACTIC_CANCEL_FLAG.with(|flag| *flag.borrow_mut() = Some(cancel_for_worker));
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
                // Timeout exceeded: ask the worker to stop cooperatively
                // and give it a bounded grace period to notice and exit.
                // Unlike a bare `drop(handle)`, the `JoinHandle` is never
                // silently discarded: within the grace period it is
                // joined directly; past the grace period it is handed to
                // a dedicated reaper thread that blocks until the worker
                // eventually finishes and joins it then. Either way the
                // worker thread is always reclaimed — for tactics that
                // never call `cancellation_requested()`, just not
                // promptly.
                cancel.store(true, Ordering::Relaxed);

                let grace = Duration::from_millis(self.timeout_ms.min(TIMEOUT_GRACE_PERIOD_MS));
                match rx.recv_timeout(grace) {
                    Ok(_) => {
                        let _ = handle.join();
                    }
                    Err(_) => {
                        let _ = thread::Builder::new()
                            .name("oxiz-timeout-tactic-reaper".to_string())
                            .spawn(move || {
                                let _ = handle.join();
                            });
                    }
                }

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

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    /// Splits any goal into two (identical) subgoals, once.
    #[derive(Debug, Default)]
    struct SplitInTwo;

    impl Tactic for SplitInTwo {
        fn name(&self) -> &str {
            "split-in-two"
        }
        fn apply(&self, goal: &Goal) -> Result<TacticResult> {
            Ok(TacticResult::SubGoals(vec![goal.clone(), goal.clone()]))
        }
    }

    /// Reports the *first* goal it ever sees as `Solved(Sat)`; every
    /// subsequent goal (even if structurally identical) is left pending
    /// (`NotApplicable`), simulating "this specific subgoal instance is
    /// done, but sibling subgoals from the same split are not".
    #[derive(Debug, Default)]
    struct ResolveFirstOnly {
        calls: AtomicUsize,
    }

    impl Tactic for ResolveFirstOnly {
        fn name(&self) -> &str {
            "resolve-first-only"
        }
        fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(TacticResult::Solved(SolveResult::Sat))
            } else {
                Ok(TacticResult::NotApplicable)
            }
        }
    }

    #[test]
    fn then_tactic_does_not_report_whole_set_solved_from_one_subgoal() {
        // Regression test: `ThenTactic` used to `return` as soon as *any*
        // subgoal came back `Solved(..)`, discarding every other pending
        // subgoal from an earlier split and reporting the whole (still
        // partially unresolved) goal set as solved.
        let goal = Goal::new(vec![]);
        let then = ThenTactic::new(vec![
            Box::new(SplitInTwo),
            Box::new(ResolveFirstOnly::default()),
        ]);

        let result = then.apply(&goal).expect("tactic should not error");
        match result {
            TacticResult::SubGoals(remaining) => {
                assert_eq!(
                    remaining.len(),
                    1,
                    "exactly one of the two split subgoals was resolved; \
                     the other must still be reported as outstanding"
                );
            }
            other => panic!(
                "expected one remaining subgoal to be reported, got {other:?} \
                 (a lone Solved(Sat) would silently ignore the still-pending \
                 sibling subgoal)"
            ),
        }
    }

    #[test]
    fn then_tactic_reports_sat_once_every_subgoal_is_resolved() {
        #[derive(Debug, Default)]
        struct ResolveEverything;
        impl Tactic for ResolveEverything {
            fn name(&self) -> &str {
                "resolve-everything"
            }
            fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
                Ok(TacticResult::Solved(SolveResult::Sat))
            }
        }

        let goal = Goal::new(vec![]);
        let then = ThenTactic::new(vec![Box::new(SplitInTwo), Box::new(ResolveEverything)]);

        let result = then.apply(&goal).expect("tactic should not error");
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn then_tactic_short_circuits_on_unsat_subgoal() {
        #[derive(Debug, Default)]
        struct FirstIsUnsat {
            calls: AtomicUsize,
        }
        impl Tactic for FirstIsUnsat {
            fn name(&self) -> &str {
                "first-is-unsat"
            }
            fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(TacticResult::Solved(SolveResult::Unsat))
                } else {
                    Ok(TacticResult::Solved(SolveResult::Sat))
                }
            }
        }

        let goal = Goal::new(vec![]);
        let then = ThenTactic::new(vec![
            Box::new(SplitInTwo),
            Box::new(FirstIsUnsat::default()),
        ]);

        let result = then.apply(&goal).expect("tactic should not error");
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    // Regression tests for: "TimeoutTactic leaks its worker thread on
    // timeout" — the worker must be reclaimed (cooperatively, promptly, or
    // eventually via the background reaper), never dropped-and-forgotten.

    #[test]
    fn timeout_tactic_reclaims_a_cooperative_worker_on_timeout() {
        use std::time::Duration;

        /// Loops checking `cancellation_requested()`, incrementing
        /// `iterations` each pass, until cancelled or a large iteration
        /// cap (a safety net against a misbehaving test never observing
        /// cancellation).
        #[derive(Debug)]
        struct CooperativeLoop {
            iterations: std::sync::Arc<AtomicUsize>,
        }

        impl Tactic for CooperativeLoop {
            fn name(&self) -> &str {
                "cooperative-loop"
            }
            fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
                for _ in 0..10_000 {
                    if cancellation_requested() {
                        return Ok(TacticResult::Failed("cancelled".to_string()));
                    }
                    self.iterations.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(2));
                }
                Ok(TacticResult::NotApplicable)
            }
        }

        let iterations = std::sync::Arc::new(AtomicUsize::new(0));
        let tactic = CooperativeLoop {
            iterations: std::sync::Arc::clone(&iterations),
        };
        let timeout = TimeoutTactic::new(std::sync::Arc::new(tactic), 100);

        let result = timeout
            .apply(&Goal::new(vec![]))
            .expect("apply should not error");
        assert!(matches!(result, TacticResult::Failed(_)));

        let count_at_return = iterations.load(Ordering::SeqCst);
        // Give the (already-cancelled) worker a further beat to confirm it
        // actually stopped looping rather than continuing in the
        // background after `apply` returned.
        std::thread::sleep(Duration::from_millis(200));
        let count_after_pause = iterations.load(Ordering::SeqCst);

        assert_eq!(
            count_at_return, count_after_pause,
            "the worker must have stopped incrementing once cancelled, not \
             kept running detached in the background"
        );
    }

    #[test]
    fn timeout_tactic_returns_promptly_for_a_non_cooperative_worker() {
        use oxiz_time::{Duration, Instant};

        /// Never checks `cancellation_requested()`; simulates a tactic
        /// that cannot be cooperatively cancelled.
        #[derive(Debug, Default)]
        struct SlowNonCooperative;

        impl Tactic for SlowNonCooperative {
            fn name(&self) -> &str {
                "slow-non-cooperative"
            }
            fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
                std::thread::sleep(Duration::from_secs(5));
                Ok(TacticResult::NotApplicable)
            }
        }

        let timeout = TimeoutTactic::from_box(Box::new(SlowNonCooperative), 20);
        let start = Instant::now();
        let result = timeout
            .apply(&Goal::new(vec![]))
            .expect("apply should not error");
        let elapsed = start.elapsed();

        assert!(matches!(result, TacticResult::Failed(_)));
        // Must return well before the 5s worker naturally finishes: bound
        // generously at 2s to absorb scheduling jitter while still proving
        // `apply` did not block on the non-cooperative worker.
        assert!(
            elapsed < Duration::from_secs(2),
            "TimeoutTactic::apply must return promptly instead of blocking \
             on a non-cooperative worker, elapsed = {elapsed:?}"
        );
    }
}
