//! Parallel frame solving for Spacer.
//!
//! This module provides infrastructure for parallel processing of frames
//! and proof obligations, improving performance on multi-core systems.
//!
//! Reference: Z3's parallel PDR and portfolio approaches

use crate::frames::{FrameManager, LemmaId};
use crate::pob::{PobId, PobManager};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, trace};

/// Errors that can occur in parallel solving
#[derive(Error, Debug)]
pub enum ParallelError {
    /// Thread pool error
    #[error("thread pool error: {0}")]
    ThreadPool(String),
    /// Synchronization error
    #[error("synchronization error: {0}")]
    Sync(String),
    /// Worker timeout
    #[error("worker timeout")]
    Timeout,
}

/// Configuration for parallel solving
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads (0 = auto-detect)
    pub num_workers: usize,
    /// Enable parallel frame propagation
    pub parallel_propagation: bool,
    /// Enable parallel POB blocking
    pub parallel_blocking: bool,
    /// Maximum queue size per worker
    pub max_queue_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: 0, // Auto-detect
            parallel_propagation: true,
            parallel_blocking: true,
            max_queue_size: 1000,
        }
    }
}

/// Work item for parallel processing
#[derive(Debug, Clone)]
pub enum WorkItem {
    /// Propagate a lemma to higher frames
    PropagateLemma {
        /// Lemma to propagate
        lemma_id: LemmaId,
        /// Starting frame level
        from_level: u32,
    },
    /// Block a proof obligation
    BlockPob {
        /// POB to block
        pob_id: PobId,
    },
    /// Check subsumption between lemmas
    CheckSubsumption {
        /// First lemma
        lemma_a: LemmaId,
        /// Second lemma
        lemma_b: LemmaId,
    },
}

/// Result of processing a work item
#[derive(Debug, Clone)]
pub enum WorkResult {
    /// Lemma successfully propagated
    LemmaPropagated {
        /// Lemma ID
        lemma_id: LemmaId,
        /// New frame level
        new_level: u32,
    },
    /// POB successfully blocked
    PobBlocked {
        /// POB ID
        pob_id: PobId,
        /// Blocking lemma
        lemma_id: LemmaId,
    },
    /// Subsumption check result
    Subsumed {
        /// The subsumed lemma (can be removed)
        subsumed: LemmaId,
        /// The subsuming lemma
        subsuming: LemmaId,
    },
    /// Work item failed
    Failed {
        /// Error message
        error: String,
    },
}

/// Parallel work queue
pub struct WorkQueue {
    /// Work items to process
    items: Arc<Mutex<Vec<WorkItem>>>,
    /// Results from workers
    results: Arc<Mutex<Vec<WorkResult>>>,
    /// Configuration
    config: ParallelConfig,
}

impl WorkQueue {
    /// Create a new work queue
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            items: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    /// Add a work item to the queue
    pub fn enqueue(&self, item: WorkItem) -> Result<(), ParallelError> {
        let mut items = self
            .items
            .lock()
            .map_err(|e| ParallelError::Sync(e.to_string()))?;

        if items.len() >= self.config.max_queue_size {
            return Err(ParallelError::Sync("queue full".to_string()));
        }

        items.push(item);
        Ok(())
    }

    /// Dequeue a work item (returns None if queue is empty)
    pub fn dequeue(&self) -> Result<Option<WorkItem>, ParallelError> {
        let mut items = self
            .items
            .lock()
            .map_err(|e| ParallelError::Sync(e.to_string()))?;

        Ok(items.pop())
    }

    /// Add a result
    pub fn add_result(&self, result: WorkResult) -> Result<(), ParallelError> {
        let mut results = self
            .results
            .lock()
            .map_err(|e| ParallelError::Sync(e.to_string()))?;

        results.push(result);
        Ok(())
    }

    /// Get all results and clear the result queue
    pub fn drain_results(&self) -> Result<Vec<WorkResult>, ParallelError> {
        let mut results = self
            .results
            .lock()
            .map_err(|e| ParallelError::Sync(e.to_string()))?;

        Ok(std::mem::take(&mut *results))
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.items
            .lock()
            .map(|items| items.is_empty())
            .unwrap_or(true)
    }

    /// Get queue size
    pub fn size(&self) -> usize {
        self.items.lock().map(|items| items.len()).unwrap_or(0)
    }
}

/// Parallel frame solver
pub struct ParallelFrameSolver {
    /// Configuration
    config: ParallelConfig,
    /// Work queue
    queue: WorkQueue,
    /// Number of active workers
    active_workers: Arc<Mutex<usize>>,
}

impl ParallelFrameSolver {
    /// Create a new parallel frame solver
    pub fn new(config: ParallelConfig) -> Self {
        let queue = WorkQueue::new(config.clone());
        Self {
            config,
            queue,
            active_workers: Arc::new(Mutex::new(0)),
        }
    }

    /// Get the number of worker threads to use
    fn num_workers(&self) -> usize {
        if self.config.num_workers > 0 {
            self.config.num_workers
        } else {
            // Auto-detect: use number of CPUs
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        }
    }

    /// Spawn worker threads
    pub fn spawn_workers<F>(
        &self,
        worker_fn: F,
    ) -> Result<Vec<thread::JoinHandle<()>>, ParallelError>
    where
        F: Fn(WorkItem) -> WorkResult + Send + Sync + 'static + Clone,
    {
        let num_workers = self.num_workers();
        let mut handles = Vec::new();

        debug!("Spawning {} worker threads", num_workers);

        for worker_id in 0..num_workers {
            let queue = self.queue.items.clone();
            let results = self.queue.results.clone();
            let active_workers = self.active_workers.clone();
            let worker_fn = worker_fn.clone();

            let handle = thread::spawn(move || {
                trace!("Worker {} started", worker_id);

                // Increment active worker count
                if let Ok(mut count) = active_workers.lock() {
                    *count += 1;
                }

                loop {
                    // Try to get work item
                    let work_item = {
                        let mut items = match queue.lock() {
                            Ok(items) => items,
                            Err(e) => {
                                debug!("Worker {} failed to lock queue: {}", worker_id, e);
                                break;
                            }
                        };
                        items.pop()
                    };

                    match work_item {
                        Some(item) => {
                            trace!("Worker {} processing item: {:?}", worker_id, item);
                            let result = worker_fn(item);

                            // Add result
                            if let Ok(mut res) = results.lock() {
                                res.push(result);
                            }
                        }
                        None => {
                            // No work available, sleep briefly
                            thread::sleep(Duration::from_millis(10));

                            // Check if we should exit
                            // In a real implementation, we'd have a shutdown signal
                            break;
                        }
                    }
                }

                // Decrement active worker count
                if let Ok(mut count) = active_workers.lock() {
                    *count -= 1;
                }

                trace!("Worker {} finished", worker_id);
            });

            handles.push(handle);
        }

        Ok(handles)
    }

    /// Process work items in parallel
    pub fn process_parallel<F>(
        &self,
        items: Vec<WorkItem>,
        worker_fn: F,
    ) -> Result<Vec<WorkResult>, ParallelError>
    where
        F: Fn(WorkItem) -> WorkResult + Send + Sync + 'static + Clone,
    {
        // Add items to queue
        for item in items {
            self.queue.enqueue(item)?;
        }

        // Spawn workers
        let handles = self.spawn_workers(worker_fn)?;

        // Wait for workers to complete
        for handle in handles {
            handle
                .join()
                .map_err(|_| ParallelError::ThreadPool("worker thread panicked".to_string()))?;
        }

        // Collect results
        self.queue.drain_results()
    }

    /// Get the work queue
    pub fn queue(&self) -> &WorkQueue {
        &self.queue
    }

    /// Get active worker count
    pub fn active_workers(&self) -> usize {
        self.active_workers.lock().map(|c| *c).unwrap_or(0)
    }
}

/// Parallel lemma propagation helper
pub struct ParallelPropagator {
    /// Parallel solver
    solver: ParallelFrameSolver,
}

impl ParallelPropagator {
    /// Create a new parallel propagator
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            solver: ParallelFrameSolver::new(config),
        }
    }

    /// Propagate lemmas in parallel
    pub fn propagate_lemmas(
        &self,
        _frames: &FrameManager,
        _pobs: &PobManager,
        lemmas: Vec<(LemmaId, u32)>,
    ) -> Result<Vec<WorkResult>, ParallelError> {
        let items: Vec<WorkItem> = lemmas
            .into_iter()
            .map(|(lemma_id, from_level)| WorkItem::PropagateLemma {
                lemma_id,
                from_level,
            })
            .collect();

        let worker_fn = move |item: WorkItem| -> WorkResult {
            match item {
                WorkItem::PropagateLemma {
                    lemma_id,
                    from_level,
                } => {
                    // Placeholder: actual implementation would check if lemma
                    // is inductive at higher levels
                    // For now, just return success
                    WorkResult::LemmaPropagated {
                        lemma_id,
                        new_level: from_level + 1,
                    }
                }
                _ => WorkResult::Failed {
                    error: "unexpected work item type".to_string(),
                },
            }
        };

        self.solver.process_parallel(items, worker_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert_eq!(config.num_workers, 0); // Auto-detect
        assert!(config.parallel_propagation);
    }

    #[test]
    fn test_work_queue() {
        let config = ParallelConfig::default();
        let queue = WorkQueue::new(config);

        let item = WorkItem::PropagateLemma {
            lemma_id: LemmaId(0),
            from_level: 1,
        };

        assert!(queue.enqueue(item).is_ok());
        assert_eq!(queue.size(), 1);
        assert!(!queue.is_empty());

        let dequeued = queue.dequeue().unwrap();
        assert!(dequeued.is_some());
        assert!(queue.is_empty());
    }

    #[test]
    fn test_work_result() {
        let result = WorkResult::LemmaPropagated {
            lemma_id: LemmaId(0),
            new_level: 2,
        };

        match result {
            WorkResult::LemmaPropagated {
                lemma_id,
                new_level,
            } => {
                assert_eq!(lemma_id, LemmaId(0));
                assert_eq!(new_level, 2);
            }
            _ => panic!("unexpected result type"),
        }
    }

    #[test]
    fn test_parallel_solver_creation() {
        let config = ParallelConfig::default();
        let solver = ParallelFrameSolver::new(config);
        assert_eq!(solver.active_workers(), 0);
    }

    #[test]
    fn test_parallel_propagator() {
        let config = ParallelConfig {
            num_workers: 2,
            parallel_propagation: true,
            parallel_blocking: true,
            max_queue_size: 100,
        };

        let propagator = ParallelPropagator::new(config);
        let frames = FrameManager::new();
        let pobs = PobManager::new();

        let lemmas = vec![(LemmaId(0), 1), (LemmaId(1), 2)];
        let results = propagator.propagate_lemmas(&frames, &pobs, lemmas);

        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 2);
    }
}
