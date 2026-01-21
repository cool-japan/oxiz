//! Distributed PDR for solving large CHC systems across multiple workers.
//!
//! This module provides infrastructure for distributed solving of Constrained Horn Clauses,
//! allowing multiple workers to collaborate on solving a single CHC system.
//!
//! ## Architecture
//!
//! - **Coordinator**: Manages work distribution and result aggregation
//! - **Workers**: Process proof obligations and learn lemmas independently
//! - **Shared State**: Frame lemmas are synchronized across workers
//! - **Communication**: Message passing for work items and learned lemmas
//!
//! Reference: Distributed PDR algorithms from literature

use crate::chc::{ChcSystem, PredId};
use crate::frames::{FrameManager, LemmaId};
use crate::pdr::{SpacerConfig, SpacerError, SpacerResult, SpacerStats};
use crate::pob::{Pob, PobId};
use oxiz_core::{TermId, TermManager};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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

    /// Run the worker loop
    pub fn run(
        &mut self,
        _terms: &mut TermManager,
        _system: &ChcSystem,
        _config: &SpacerConfig,
    ) -> Result<(), DistributedError> {
        loop {
            // Check for shutdown signal
            if let Some(WorkerMessage::Shutdown) = self.shared.receive_message() {
                break;
            }

            // Check if result already found
            if self.shared.get_result().is_some() {
                break;
            }

            // Try to get work
            if let Some(work_item) = self.shared.dequeue_work() {
                // Process work item
                self.process_work_item(work_item)?;
            } else {
                // No work available - request work stealing or wait
                self.shared
                    .send_message(WorkerMessage::RequestWork { worker_id: self.id });
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        // Update shared statistics
        self.shared.update_stats(|stats| {
            stats.worker_stats.insert(self.id, self.stats.clone());
        });

        Ok(())
    }

    /// Process a work item
    fn process_work_item(&mut self, work_item: WorkItem) -> Result<(), DistributedError> {
        // Process a POB (Proof Obligation)
        // 1. Try to block the POB using SMT solver
        // 2. If blocked, learn lemma and add to frames
        // 3. If not blocked, create child POBs
        // 4. Send results/lemmas via messages

        self.stats.num_pobs += 1;

        // Basic POB processing logic:
        // In a full implementation, we would:
        // - Set up SMT query for the POB
        // - Check if the state is reachable
        // - If unreachable, extract a lemma (blocking clause)
        // - If reachable, generate predecessor POBs

        // For now, implement a simple heuristic:
        // - Assume half of POBs can be blocked
        // - Generate a trivial lemma for blocked POBs
        let blocked = work_item.pob_id.0.is_multiple_of(2);

        // Generate a lemma ID if blocked
        // In reality, this would be extracted from UNSAT core and added to frames
        let lemma = if blocked {
            // In a real implementation, we would:
            // 1. Add the lemma to the frame manager
            // 2. Get its LemmaId
            // For now, just return None as we don't have a real lemma
            None
        } else {
            None
        };

        // Send the result back to the coordinator
        self.shared.send_message(WorkerMessage::WorkResult {
            worker_id: self.id,
            pob_id: work_item.pob_id,
            blocked,
            lemma,
        });

        if blocked {
            self.stats.num_blocked += 1;
        }

        Ok(())
    }
}

/// Coordinator for distributed solving
#[allow(dead_code)]
pub struct DistributedCoordinator<'a> {
    /// Term manager
    terms: &'a mut TermManager,
    /// CHC system
    system: &'a ChcSystem,
    /// Configuration
    config: DistributedConfig,
    /// Shared state
    shared: Arc<SharedState>,
    /// Start time
    start_time: Instant,
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
            start_time: Instant::now(),
        }
    }

    /// Solve the CHC system using distributed workers
    pub fn solve(&mut self) -> Result<SpacerResult, DistributedError> {
        // Distributed solving with worker threads
        // 1. Initialize frames and initial POBs
        // 2. Spawn worker threads
        // 3. Monitor progress and synchronize state
        // 4. Detect termination (invariant found or counterexample)
        // 5. Aggregate results

        use std::sync::Arc;
        use std::thread;

        // Step 1: Initialize work queue with initial POBs
        // In a real implementation, we would create POBs from the query
        use crate::pob::Pob;

        let initial_pob_data = Pob::new(
            PobId(0),
            PredId(0),
            self.terms.mk_true(), // Placeholder post-condition
            0,                    // level
            0,                    // depth
        );

        let initial_work = WorkItem {
            pob_id: PobId(0),
            pob: initial_pob_data,
            priority: 0,
        };

        self.shared.enqueue_work(initial_work);

        // Step 2: Spawn worker threads
        let mut handles = Vec::new();
        for worker_id in 0..self.config.num_workers {
            let shared = Arc::clone(&self.shared);
            let _config = self.config.worker_config.clone();
            let handle = thread::spawn(move || {
                tracing::debug!("Worker {} started", worker_id);

                // Worker loop: process work items until termination
                loop {
                    // Check for shutdown message
                    if let Some(WorkerMessage::Shutdown) = shared.receive_message() {
                        tracing::debug!("Worker {} received shutdown signal", worker_id);
                        break;
                    }

                    // Check for termination
                    if shared.get_result().is_some() {
                        tracing::debug!("Worker {} terminating (result found)", worker_id);
                        break;
                    }

                    // Try to dequeue work
                    let work_item = match shared.dequeue_work() {
                        Some(item) => item,
                        None => {
                            // No work available
                            if shared.work_queue_size() == 0 {
                                // Queue is truly empty, check if we're done
                                // In a full implementation, check for global termination
                                tracing::debug!("Worker {} idle", worker_id);
                                std::thread::sleep(Duration::from_millis(10));
                                continue;
                            }
                            continue;
                        }
                    };

                    tracing::trace!("Worker {} processing POB {:?}", worker_id, work_item.pob_id);

                    // Process the work item (POB)
                    // In a full implementation, this would:
                    // 1. Check if POB is blocked by existing lemmas
                    // 2. If not, generate a predecessor POB
                    // 3. Learn and generalize lemmas
                    // 4. Propagate lemmas forward
                    // 5. Detect fixpoints (invariants) or counterexamples

                    // For now, simulate some work
                    std::thread::sleep(Duration::from_micros(100));

                    // Report result (simulated: mark as blocked)
                    shared.send_message(WorkerMessage::WorkResult {
                        worker_id,
                        pob_id: work_item.pob_id,
                        blocked: true,
                        lemma: None,
                    });

                    // Update statistics
                    shared.update_stats(|stats| {
                        stats.total_work_items += 1;
                        stats.messages_sent += 1;
                    });
                }

                tracing::debug!("Worker {} finished", worker_id);
            });
            handles.push(handle);
        }

        // Step 3: Monitor progress and process worker messages
        let monitor_start = Instant::now();
        let sync_interval = Duration::from_millis(self.config.sync_interval_ms);
        let mut last_sync = Instant::now();

        loop {
            // Check for timeout
            if let Some(timeout) = self.config.timeout
                && monitor_start.elapsed() >= timeout
            {
                tracing::warn!("Distributed solving timed out");
                self.shared.set_result(SpacerResult::Unknown);
                break;
            }

            // Process messages from workers
            let mut messages_processed = 0;
            while let Some(msg) = self.shared.receive_message() {
                match msg {
                    WorkerMessage::WorkResult {
                        worker_id,
                        pob_id,
                        blocked,
                        lemma,
                    } => {
                        tracing::trace!(
                            "Worker {} reported result for POB {:?}: blocked={}",
                            worker_id,
                            pob_id,
                            blocked
                        );
                        if let Some(_lemma_id) = lemma {
                            // Lemma was learned
                            self.shared.update_stats(|stats| {
                                stats.total_lemmas += 1;
                            });
                        }
                    }
                    WorkerMessage::LemmaLearned {
                        pred, lemma, level, ..
                    } => {
                        // Synchronize lemma across all workers
                        self.shared.add_lemma(pred, lemma, level);
                        self.shared.update_stats(|stats| {
                            stats.total_lemmas += 1;
                            stats.sync_events += 1;
                        });
                    }
                    WorkerMessage::Counterexample { worker_id } => {
                        tracing::info!("Worker {} found counterexample", worker_id);
                        self.shared.set_result(SpacerResult::Unsafe);
                        break;
                    }
                    WorkerMessage::Invariant { worker_id, level } => {
                        tracing::info!("Worker {} found invariant at level {}", worker_id, level);
                        self.shared.set_result(SpacerResult::Safe);
                        break;
                    }
                    _ => {}
                }
                messages_processed += 1;
            }

            // Check if result was found
            if self.shared.get_result().is_some() {
                break;
            }

            // Check if all workers are idle (no work in queue)
            if self.shared.work_queue_size() == 0 && messages_processed == 0 {
                // Heuristic: if no work and no messages for a while, assume completion
                if last_sync.elapsed() > Duration::from_millis(500) {
                    tracing::debug!("No work remaining, assuming Unknown result");
                    self.shared.set_result(SpacerResult::Unknown);
                    break;
                }
            } else {
                last_sync = Instant::now();
            }

            // Sleep briefly before next check
            std::thread::sleep(std::cmp::min(sync_interval, Duration::from_millis(50)));
        }

        // Signal workers to shut down
        for _ in 0..self.config.num_workers {
            self.shared.send_message(WorkerMessage::Shutdown);
        }

        // Wait for workers to finish
        for handle in handles {
            let _ = handle.join();
        }

        // Step 4: Return the result
        self.shared
            .get_result()
            .ok_or_else(|| SpacerError::Internal("no result found".to_string()).into())
    }

    /// Check if timeout exceeded
    #[allow(dead_code)]
    fn is_timeout(&self) -> bool {
        if let Some(timeout) = self.config.timeout {
            self.start_time.elapsed() >= timeout
        } else {
            false
        }
    }
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
        let work = state.dequeue_work().unwrap();
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
