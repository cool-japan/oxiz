//! Parallel PDR for solving CHC systems across real worker threads.
//!
//! ## What this is: a parallel portfolio with private term arenas
//!
//! [`DistributedCoordinator::solve`] runs a **genuine `std::thread` portfolio**:
//! it spawns N worker threads, each running an independent, sound, sequential
//! [`Spacer`] engine on its own private copy of the CHC system, configured with
//! a *different* strategy (resource limits / feature toggles). The workers race;
//! the first to reach a definite `Safe`/`Unsafe` verdict wins, and the losers
//! are cancelled through a shared [`AtomicBool`] token that each engine checks
//! in its main loop. If every worker only reaches `Unknown` (or an overall
//! [`DistributedConfig::timeout`] elapses) the coordinator returns `Unknown`.
//! Because every worker runs the same sound engine, whichever worker wins the
//! race the verdict matches what the sequential engine would produce.
//!
//! ## Why private arenas (and why lemmas are NOT shared)
//!
//! [`Spacer`] borrows `&mut TermManager` for the whole solve and term interning
//! is not thread-safe, while [`ChcSystem`] holds atomic id counters and is not
//! `Clone`. So each worker gets its **own** [`TermManager`] + [`ChcSystem`],
//! rebuilt from the original by [`crate::translate::translate_system`] (a
//! fail-closed re-intern of the linear-arithmetic/boolean fragment). A
//! consequence is that **learned lemmas are NOT shared across workers**: a
//! `TermId` from one worker's arena is meaningless in another's, so there is no
//! shared frame/lemma pool. This is a parallel portfolio of independent
//! sequential engines, **not** a shared-frame distributed PDR. When the system
//! uses a term fragment `translate_system` cannot faithfully copy (bit-vectors,
//! arrays, strings, ...), the coordinator transparently falls back to the sound
//! in-process sequential engine.
//!
//! The message/queue/shared-frame types below ([`SharedState`],
//! [`WorkerMessage`], [`WorkItem`]) are retained as the scaffolding for a
//! future shared-frame engine with cross-arena lemma translation; they are not
//! on the portfolio's solve path.
//!
//! Reference: Z3's parallel/portfolio solving; distributed PDR from literature.

use crate::chc::{ChcSystem, PredId};
use crate::frames::{FrameManager, LemmaId};
use crate::pdr::{Spacer, SpacerConfig, SpacerError, SpacerResult, SpacerStats};
use crate::pob::{Pob, PobId};
use crate::portfolio::Strategy;
use oxiz_core::{TermId, TermManager};
use oxiz_time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use thiserror::Error;

/// Errors that can occur in distributed solving
#[derive(Error, Debug)]
pub enum DistributedError {
    /// Worker error
    #[error("worker {0} error: {1}")]
    WorkerError(usize, String),
    /// Communication error
    #[error("communication error: {0}")]
    Communication(String),
    /// Coordination error
    #[error("coordination error: {0}")]
    Coordination(String),
    /// Spacer error from underlying solver
    #[error("spacer error: {0}")]
    Spacer(#[from] SpacerError),
    /// Timeout
    #[error("timeout after {0:?}")]
    Timeout(Duration),
}

/// Configuration for distributed solving
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Base configuration for each worker
    pub worker_config: SpacerConfig,
    /// Synchronization interval (ms)
    pub sync_interval_ms: u64,
    /// Timeout for distributed solving
    pub timeout: Option<Duration>,
    /// Enable work stealing between workers
    pub enable_work_stealing: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            worker_config: SpacerConfig::default(),
            sync_interval_ms: 100,
            timeout: None,
            enable_work_stealing: true,
        }
    }
}

/// Message types for worker communication
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    /// Work item (POB) to process
    Work(WorkItem),
    /// Lemma learned by a worker
    LemmaLearned {
        worker_id: usize,
        pred: PredId,
        lemma: TermId,
        level: u32,
    },
    /// Frame created
    FrameCreated { level: u32 },
    /// Result from processing a POB
    WorkResult {
        worker_id: usize,
        pob_id: PobId,
        blocked: bool,
        lemma: Option<LemmaId>,
    },
    /// Counterexample found
    Counterexample { worker_id: usize },
    /// Invariant found (fixpoint detected)
    Invariant { worker_id: usize, level: u32 },
    /// Worker requesting work (for work stealing)
    RequestWork { worker_id: usize },
    /// Shutdown signal
    Shutdown,
}

/// Work item for distributed processing
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// POB identifier
    pub pob_id: PobId,
    /// The POB to process
    pub pob: Pob,
    /// Priority (higher = more urgent)
    pub priority: i32,
}

/// Shared state between workers
pub struct SharedState {
    /// Frame manager (synchronized across workers)
    frames: Mutex<FrameManager>,
    /// Work queue
    work_queue: Mutex<VecDeque<WorkItem>>,
    /// Result
    result: Mutex<Option<SpacerResult>>,
    /// Combined statistics
    stats: Mutex<DistributedStats>,
    /// Message channels
    messages: Mutex<VecDeque<WorkerMessage>>,
}

impl SharedState {
    /// Create new shared state
    pub fn new() -> Self {
        Self {
            frames: Mutex::new(FrameManager::new()),
            work_queue: Mutex::new(VecDeque::new()),
            result: Mutex::new(None),
            stats: Mutex::new(DistributedStats::default()),
            messages: Mutex::new(VecDeque::new()),
        }
    }

    /// Add work item to queue
    pub fn enqueue_work(&self, item: WorkItem) {
        let mut queue = self.work_queue.lock().expect("lock should not be poisoned");
        // Insert based on priority (higher priority first)
        let pos = queue
            .iter()
            .position(|w| w.priority < item.priority)
            .unwrap_or(queue.len());
        queue.insert(pos, item);
    }

    /// Dequeue work item
    pub fn dequeue_work(&self) -> Option<WorkItem> {
        self.work_queue
            .lock()
            .expect("lock should not be poisoned")
            .pop_front()
    }

    /// Get number of pending work items
    pub fn work_queue_size(&self) -> usize {
        self.work_queue
            .lock()
            .expect("lock should not be poisoned")
            .len()
    }

    /// Send message to workers
    pub fn send_message(&self, msg: WorkerMessage) {
        self.messages
            .lock()
            .expect("lock should not be poisoned")
            .push_back(msg);
    }

    /// Receive message
    pub fn receive_message(&self) -> Option<WorkerMessage> {
        self.messages
            .lock()
            .expect("lock should not be poisoned")
            .pop_front()
    }

    /// Set result
    pub fn set_result(&self, result: SpacerResult) {
        *self.result.lock().expect("lock should not be poisoned") = Some(result);
    }

    /// Get result
    pub fn get_result(&self) -> Option<SpacerResult> {
        self.result
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Add lemma to frames
    pub fn add_lemma(&self, pred: PredId, formula: TermId, level: u32) -> LemmaId {
        self.frames
            .lock()
            .expect("lock should not be poisoned")
            .add_lemma(pred, formula, level)
    }

    /// Get frame manager (locked)
    pub fn with_frames<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut FrameManager) -> R,
    {
        let mut frames = self.frames.lock().expect("lock should not be poisoned");
        f(&mut frames)
    }

    /// Update statistics
    pub fn update_stats<F>(&self, f: F)
    where
        F: FnOnce(&mut DistributedStats),
    {
        let mut stats = self.stats.lock().expect("lock should not be poisoned");
        f(&mut stats);
    }

    /// Get statistics
    pub fn get_stats(&self) -> DistributedStats {
        self.stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for distributed solving
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Per-worker statistics
    pub worker_stats: HashMap<usize, SpacerStats>,
    /// Total work items processed
    pub total_work_items: u64,
    /// Total lemmas learned
    pub total_lemmas: u64,
    /// Work stealing events
    pub work_stealing_events: u64,
    /// Synchronization events
    pub sync_events: u64,
    /// Communication overhead (messages sent)
    pub messages_sent: u64,
}

impl DistributedStats {
    /// Create new distributed statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Aggregate statistics from all workers
    pub fn aggregate(&self) -> SpacerStats {
        let mut total = SpacerStats::default();
        for stats in self.worker_stats.values() {
            total.num_frames = total.num_frames.max(stats.num_frames);
            total.num_lemmas = total.num_lemmas.saturating_add(stats.num_lemmas);
            total.num_inductive = total.num_inductive.saturating_add(stats.num_inductive);
            total.num_pobs = total.num_pobs.saturating_add(stats.num_pobs);
            total.num_blocked = total.num_blocked.saturating_add(stats.num_blocked);
            total.num_smt_queries = total.num_smt_queries.saturating_add(stats.num_smt_queries);
            total.num_propagations = total
                .num_propagations
                .saturating_add(stats.num_propagations);
            total.num_subsumed = total.num_subsumed.saturating_add(stats.num_subsumed);
            total.num_mic_attempts = total
                .num_mic_attempts
                .saturating_add(stats.num_mic_attempts);
            total.num_ctg_strengthenings = total
                .num_ctg_strengthenings
                .saturating_add(stats.num_ctg_strengthenings);
        }
        total
    }
}

/// Worker thread for distributed solving
pub struct Worker {
    /// Worker ID
    id: usize,
    /// Shared state
    shared: Arc<SharedState>,
    /// Local statistics
    stats: SpacerStats,
}

impl Worker {
    /// Create a new worker
    pub fn new(id: usize, shared: Arc<SharedState>) -> Self {
        Self {
            id,
            shared,
            stats: SpacerStats::default(),
        }
    }

    /// Run the worker.
    ///
    /// Standalone single-worker helper (the [`DistributedCoordinator`]
    /// portfolio spawns its own threads and does not use this): the worker runs
    /// the sound, sequential [`Spacer`] engine on the shared system and
    /// publishes the **real** result plus its solver statistics. Genuine
    /// multi-worker POB distribution over the shared queue is future work.
    pub fn run(
        &mut self,
        terms: &mut TermManager,
        system: &ChcSystem,
        config: &SpacerConfig,
    ) -> Result<(), DistributedError> {
        // If a peer already found the answer, honor the shutdown / result and
        // do not redundantly re-solve.
        if self.shared.get_result().is_some() {
            return Ok(());
        }
        if let Some(WorkerMessage::Shutdown) = self.shared.receive_message() {
            return Ok(());
        }

        let mut spacer = Spacer::with_config(terms, system, config.clone());
        let result = spacer.solve()?;
        self.stats = spacer.stats().clone();

        // Publish the real result and this worker's statistics.
        self.shared.set_result(result);
        let worker_id = self.id;
        let stats = self.stats.clone();
        let num_pobs = u64::from(self.stats.num_pobs);
        self.shared.update_stats(move |s| {
            s.worker_stats.insert(worker_id, stats);
            s.total_work_items = s.total_work_items.saturating_add(num_pobs);
        });

        Ok(())
    }
}

/// Coordinator for the parallel PDR portfolio.
pub struct DistributedCoordinator<'a> {
    /// Term manager (source arena; workers get independent copies).
    terms: &'a mut TermManager,
    /// CHC system to solve.
    system: &'a ChcSystem,
    /// Configuration.
    config: DistributedConfig,
    /// Shared state (result/stats sink; scaffolding for a future shared-frame
    /// engine).
    shared: Arc<SharedState>,
    /// Number of worker threads actually spawned by the last [`Self::solve`].
    spawned_threads: usize,
}

impl<'a> DistributedCoordinator<'a> {
    /// Create a new distributed coordinator
    pub fn new(
        terms: &'a mut TermManager,
        system: &'a ChcSystem,
        config: DistributedConfig,
    ) -> Self {
        Self {
            terms,
            system,
            config,
            shared: Arc::new(SharedState::new()),
            spawned_threads: 0,
        }
    }

    /// Number of worker threads spawned by the most recent [`Self::solve`]
    /// call. Zero when the sequential fallback path was taken (unsupported
    /// term fragment or a single worker).
    #[must_use]
    pub fn spawned_threads(&self) -> usize {
        self.spawned_threads
    }

    /// Solve the CHC system with a real `std::thread` parallel portfolio.
    ///
    /// Each of the `num_workers` threads runs an independent, sound, sequential
    /// [`Spacer`] on its own private arena (see the module docs) with a distinct
    /// strategy. The first definite `Safe`/`Unsafe` verdict wins and cancels the
    /// rest; all-`Unknown` (or an elapsed [`DistributedConfig::timeout`]) yields
    /// `Unknown`. If the system cannot be faithfully copied into independent
    /// arenas, this transparently runs the sound single-arena sequential engine.
    /// Either way the verdict is honest and matches the sequential engine.
    pub fn solve(&mut self) -> Result<SpacerResult, DistributedError> {
        self.spawned_threads = 0;
        let num_workers = self.config.num_workers.max(1);

        // Build one independent replica per worker up front (single-threaded,
        // so interning stays safe). If any replica cannot be built the fragment
        // is unsupported for parallel copying — fall back to the sound
        // sequential engine rather than risk an unfaithful copy.
        let mut replicas: Vec<(TermManager, ChcSystem)> = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            match crate::translate::translate_system(self.terms, self.system) {
                Some(pair) => replicas.push(pair),
                None => return self.solve_sequential(),
            }
        }
        if num_workers == 1 {
            // No parallelism to gain; the sequential engine is simpler and
            // avoids a needless thread.
            return self.solve_sequential();
        }

        let strategies = worker_strategies(num_workers, &self.config.worker_config);
        let cancel = Arc::new(AtomicBool::new(false));
        let spawned = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel::<(usize, Result<SpacerResult, SpacerError>, SpacerStats)>();

        let mut handles = Vec::with_capacity(num_workers);
        for (i, (mut worker_terms, worker_system)) in replicas.into_iter().enumerate() {
            let cancel_worker = Arc::clone(&cancel);
            let spawned_worker = Arc::clone(&spawned);
            let tx_worker = tx.clone();
            let config = strategies[i].clone();
            let handle = thread::spawn(move || {
                spawned_worker.fetch_add(1, Ordering::SeqCst);
                let (result, stats) = {
                    let mut spacer = Spacer::with_config(&mut worker_terms, &worker_system, config)
                        .with_cancel(Arc::clone(&cancel_worker));
                    let result = spacer.solve();
                    (result, spacer.stats().clone())
                };
                // The receiver may have hung up after a winner was found; that
                // is expected, so ignore the send error.
                let _ = tx_worker.send((i, result, stats));
            });
            handles.push(handle);
        }
        // Drop the coordinator's own sender so `rx` disconnects once every
        // worker has finished.
        drop(tx);

        let (verdict, first_err) = collect_verdict(&rx, num_workers, self.config.timeout, &cancel);

        // Cancel any stragglers and join every worker so the call is a clean
        // barrier (no detached threads outliving the solve).
        cancel.store(true, Ordering::SeqCst);
        let mut worker_stats: Vec<(usize, SpacerStats)> = Vec::new();
        // Drain any results the workers produced after we stopped listening so
        // their stats are still recorded.
        while let Ok((id, _res, stats)) = rx.recv() {
            worker_stats.push((id, stats));
        }
        for handle in handles {
            let _ = handle.join();
        }

        self.spawned_threads = spawned.load(Ordering::SeqCst);
        let spawned_count = self.spawned_threads as u64;
        self.shared.update_stats(move |s| {
            s.total_work_items = s.total_work_items.saturating_add(spawned_count);
            for (id, stats) in worker_stats {
                s.worker_stats.insert(id, stats);
            }
        });

        match verdict {
            Some(result) => {
                self.shared.set_result(result.clone());
                Ok(result)
            }
            None => {
                // No worker reached a definite verdict. Surface a worker error
                // only if every worker errored (matching the sequential engine,
                // e.g. `NoQuery`); otherwise this is an honest `Unknown`.
                if let Some(err) = first_err {
                    return Err(DistributedError::Spacer(err));
                }
                self.shared.set_result(SpacerResult::Unknown);
                Ok(SpacerResult::Unknown)
            }
        }
    }

    /// Sound single-arena fallback: run the sequential [`Spacer`] on the shared
    /// term manager and return its exact verdict.
    fn solve_sequential(&mut self) -> Result<SpacerResult, DistributedError> {
        let config = self.config.worker_config.clone();
        let result = {
            let mut spacer = Spacer::with_config(self.terms, self.system, config);
            spacer.solve()?
        };
        self.shared.set_result(result.clone());
        self.shared.update_stats(|stats| {
            stats.total_work_items = stats.total_work_items.saturating_add(1);
        });
        Ok(result)
    }
}

/// Diversified per-worker configurations for the portfolio.
///
/// Worker 0 mirrors the caller's `base` configuration so the portfolio's
/// verdict is at least as strong as the sequential engine's; the remaining
/// workers cycle through the standard portfolio strategies for diversity.
fn worker_strategies(n: usize, base: &SpacerConfig) -> Vec<SpacerConfig> {
    let variants = [
        Strategy::conservative().config,
        Strategy::balanced().config,
        Strategy::bmc_like().config,
        Strategy::aggressive().config,
    ];
    let mut out = Vec::with_capacity(n);
    out.push(base.clone());
    for i in 1..n {
        out.push(variants[(i - 1) % variants.len()].clone());
    }
    out
}

/// Collect worker verdicts from `rx`, returning as soon as a definite
/// `Safe`/`Unsafe` arrives, when all `n` workers have reported, or when the
/// optional `timeout` elapses.
///
/// Returns `(Some(verdict), _)` on a definite result, or `(None, first_err)`
/// where `first_err` is the first worker error seen (used only when no worker
/// reached a verdict and none returned `Unknown`).
fn collect_verdict(
    rx: &mpsc::Receiver<(usize, Result<SpacerResult, SpacerError>, SpacerStats)>,
    n: usize,
    timeout: Option<Duration>,
    cancel: &Arc<AtomicBool>,
) -> (Option<SpacerResult>, Option<SpacerError>) {
    // `Instant` here is `oxiz_time::Instant` (= `std::time::Instant` off
    // wasm32-unknown-unknown). This is the one place in the workspace where the
    // frozen wasm clock would degrade more than a heuristic: with time stuck at
    // zero, `d.saturating_duration_since(now)` re-arms the *full* timeout on
    // every iteration, so the aggregate wait could reach `n * timeout` instead
    // of `timeout`. It still terminates -- the loop is bounded by `remaining`
    // and by `recv_timeout`'s own (real, OS-side) clock. Unreachable in
    // practice: this module needs OS threads and `std::sync::mpsc`, neither of
    // which exists on wasm32-unknown-unknown.
    let deadline = timeout.map(|t| Instant::now() + t);
    let mut remaining = n;
    let mut first_err: Option<SpacerError> = None;
    let mut saw_unknown = false;

    while remaining > 0 {
        let recv = match deadline {
            Some(d) => {
                let now = Instant::now();
                if now >= d {
                    break;
                }
                rx.recv_timeout(d.saturating_duration_since(now))
            }
            None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
        };

        match recv {
            Ok((_id, result, _stats)) => {
                remaining -= 1;
                match result {
                    Ok(SpacerResult::Safe) => {
                        cancel.store(true, Ordering::SeqCst);
                        return (Some(SpacerResult::Safe), None);
                    }
                    Ok(SpacerResult::Unsafe) => {
                        cancel.store(true, Ordering::SeqCst);
                        return (Some(SpacerResult::Unsafe), None);
                    }
                    Ok(SpacerResult::Unknown) => {
                        // A definite "cannot decide"; not fatal, keep waiting
                        // for a peer that may still decide.
                        saw_unknown = true;
                    }
                    Err(e) => {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    // Surface a worker error only when *every* reporting worker errored (no
    // worker reached `Unknown`), matching what the sequential engine would do
    // for deterministic errors like `NoQuery`.
    let err = if saw_unknown { None } else { first_err };
    (None, err)
}

/// Dummy num_cpus implementation (simplified)
mod num_cpus {
    pub fn get() -> usize {
        // Default to 4 workers if we can't detect
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_state_work_queue() {
        let state = SharedState::new();

        // Enqueue work items with different priorities
        state.enqueue_work(WorkItem {
            pob_id: PobId(0),
            pob: Pob::new(PobId(0), PredId(0), TermId(0), 0, 0),
            priority: 10,
        });

        state.enqueue_work(WorkItem {
            pob_id: PobId(1),
            pob: Pob::new(PobId(1), PredId(0), TermId(1), 0, 0),
            priority: 20, // Higher priority
        });

        // Should dequeue higher priority first
        let work = state.dequeue_work().expect("test operation should succeed");
        assert_eq!(work.pob_id, PobId(1));
        assert_eq!(work.priority, 20);
    }

    #[test]
    fn test_distributed_stats_aggregate() {
        let mut stats = DistributedStats::new();

        stats.worker_stats.insert(
            0,
            SpacerStats {
                num_frames: 5,
                num_lemmas: 10,
                num_pobs: 20,
                ..Default::default()
            },
        );

        stats.worker_stats.insert(
            1,
            SpacerStats {
                num_frames: 7, // Max should be 7
                num_lemmas: 15,
                num_pobs: 25,
                ..Default::default()
            },
        );

        let aggregated = stats.aggregate();
        assert_eq!(aggregated.num_frames, 7); // Max of 5 and 7
        assert_eq!(aggregated.num_lemmas, 25); // Sum: 10 + 15
        assert_eq!(aggregated.num_pobs, 45); // Sum: 20 + 25
    }
}
