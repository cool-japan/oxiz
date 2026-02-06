//! Web Worker Support for Background Solving
#![allow(
    clippy::should_implement_trait,
    clippy::explicit_counter_loop,
    dead_code
)] // WASM constraints
//!
//! This module provides utilities for running OxiZ solver operations
//! in Web Workers, enabling background computation without blocking
//! the main UI thread.
//!
//! # Architecture
//!
//! - **Worker Pool**: Manage multiple worker threads
//! - **Message Passing**: Structured communication protocol
//! - **Task Queue**: Queue solver tasks for processing
//! - **Result Streaming**: Stream results back to main thread
//!
//! # Example (Main Thread)
//!
//! ```javascript
//! const pool = new WorkerPool(4); // 4 workers
//!
//! const task = {
//!     type: 'solve',
//!     logic: 'QF_LIA',
//!     assertions: ['(> x 0)', '(< x 10)']
//! };
//!
//! const result = await pool.execute(task);
//! console.log(result.status); // "sat" or "unsat"
//! ```
//!
//! # Example (Worker Thread)
//!
//! ```javascript
//! import { WorkerHandler } from 'oxiz-wasm/worker';
//!
//! const handler = new WorkerHandler();
//! handler.start();
//! ```

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

/// Message types for worker communication
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerMessageType {
    /// Initialize worker
    Init,
    /// Execute a task
    Execute,
    /// Cancel a task
    Cancel,
    /// Shutdown worker
    Shutdown,
    /// Task result
    Result,
    /// Task error
    Error,
    /// Progress update
    Progress,
    /// Heartbeat
    Heartbeat,
}

impl WorkerMessageType {
    /// Get message type name
    pub fn name(&self) -> &'static str {
        match self {
            WorkerMessageType::Init => "init",
            WorkerMessageType::Execute => "execute",
            WorkerMessageType::Cancel => "cancel",
            WorkerMessageType::Shutdown => "shutdown",
            WorkerMessageType::Result => "result",
            WorkerMessageType::Error => "error",
            WorkerMessageType::Progress => "progress",
            WorkerMessageType::Heartbeat => "heartbeat",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "init" => Some(WorkerMessageType::Init),
            "execute" => Some(WorkerMessageType::Execute),
            "cancel" => Some(WorkerMessageType::Cancel),
            "shutdown" => Some(WorkerMessageType::Shutdown),
            "result" => Some(WorkerMessageType::Result),
            "error" => Some(WorkerMessageType::Error),
            "progress" => Some(WorkerMessageType::Progress),
            "heartbeat" => Some(WorkerMessageType::Heartbeat),
            _ => None,
        }
    }
}

/// Worker task
#[wasm_bindgen]
pub struct WorkerTask {
    /// Task ID
    id: String,
    /// Task type
    task_type: String,
    /// Task data (JSON)
    data: JsValue,
    /// Priority (higher = more important)
    priority: u32,
    /// Timeout in milliseconds
    timeout_ms: Option<u32>,
}

#[wasm_bindgen]
impl WorkerTask {
    /// Create a new worker task
    #[wasm_bindgen(constructor)]
    pub fn new(id: String, task_type: String, data: JsValue) -> Self {
        Self {
            id,
            task_type,
            data,
            priority: 0,
            timeout_ms: None,
        }
    }

    /// Set priority
    #[wasm_bindgen(js_name = withPriority)]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set timeout
    #[wasm_bindgen(js_name = withTimeout)]
    pub fn with_timeout(mut self, timeout_ms: u32) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Get task ID
    #[wasm_bindgen(js_name = getId)]
    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    /// Get task type
    #[wasm_bindgen(js_name = getType)]
    pub fn get_type(&self) -> String {
        self.task_type.clone()
    }

    /// Get task data
    #[wasm_bindgen(js_name = getData)]
    pub fn get_data(&self) -> JsValue {
        self.data.clone()
    }

    /// Export as JS object
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"id".into(), &self.id.clone().into());
        let _ = js_sys::Reflect::set(&obj, &"type".into(), &self.task_type.clone().into());
        let _ = js_sys::Reflect::set(&obj, &"data".into(), &self.data);
        let _ = js_sys::Reflect::set(&obj, &"priority".into(), &self.priority.into());
        if let Some(timeout) = self.timeout_ms {
            let _ = js_sys::Reflect::set(&obj, &"timeout_ms".into(), &timeout.into());
        }
        obj.into()
    }
}

/// Worker status
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Worker is idle
    Idle,
    /// Worker is busy
    Busy,
    /// Worker is starting
    Starting,
    /// Worker is stopping
    Stopping,
    /// Worker has failed
    Failed,
}

impl WorkerStatus {
    /// Get status name
    pub fn name(&self) -> &'static str {
        match self {
            WorkerStatus::Idle => "idle",
            WorkerStatus::Busy => "busy",
            WorkerStatus::Starting => "starting",
            WorkerStatus::Stopping => "stopping",
            WorkerStatus::Failed => "failed",
        }
    }
}

/// Worker info
struct WorkerInfo {
    id: usize,
    status: WorkerStatus,
    current_task: Option<String>,
    tasks_completed: usize,
    tasks_failed: usize,
    total_time_ms: f64,
    created_at: f64,
}

/// Worker pool for managing multiple workers
#[wasm_bindgen]
pub struct WorkerPool {
    /// Worker info
    workers: Rc<RefCell<HashMap<usize, WorkerInfo>>>,
    /// Task queue
    queue: Rc<RefCell<VecDeque<WorkerTask>>>,
    /// Next worker ID
    next_id: Rc<RefCell<usize>>,
    /// Number of workers
    worker_count: usize,
    /// Enable verbose logging
    verbose: bool,
}

#[wasm_bindgen]
impl WorkerPool {
    /// Create a new worker pool
    #[wasm_bindgen(constructor)]
    pub fn new(worker_count: usize) -> Self {
        Self {
            workers: Rc::new(RefCell::new(HashMap::new())),
            queue: Rc::new(RefCell::new(VecDeque::new())),
            next_id: Rc::new(RefCell::new(0)),
            worker_count,
            verbose: false,
        }
    }

    /// Initialize workers
    #[wasm_bindgen(js_name = init)]
    pub fn init(&self) {
        let current_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        for _ in 0..self.worker_count {
            let id = self.get_next_id();
            self.workers.borrow_mut().insert(
                id,
                WorkerInfo {
                    id,
                    status: WorkerStatus::Starting,
                    current_task: None,
                    tasks_completed: 0,
                    tasks_failed: 0,
                    total_time_ms: 0.0,
                    created_at: current_time,
                },
            );
        }

        if self.verbose {
            web_sys::console::log_1(
                &format!("Initialized worker pool with {} workers", self.worker_count).into(),
            );
        }
    }

    /// Submit a task to the pool
    #[wasm_bindgen(js_name = submit)]
    pub fn submit(&self, task: WorkerTask) {
        self.queue.borrow_mut().push_back(task);

        if self.verbose {
            web_sys::console::log_1(
                &format!("Task queued, queue length: {}", self.queue_length()).into(),
            );
        }
    }

    /// Get queue length
    #[wasm_bindgen(js_name = queueLength)]
    pub fn queue_length(&self) -> usize {
        self.queue.borrow().len()
    }

    /// Get idle worker count
    #[wasm_bindgen(js_name = idleCount)]
    pub fn idle_count(&self) -> usize {
        self.workers
            .borrow()
            .values()
            .filter(|w| w.status == WorkerStatus::Idle)
            .count()
    }

    /// Get busy worker count
    #[wasm_bindgen(js_name = busyCount)]
    pub fn busy_count(&self) -> usize {
        self.workers
            .borrow()
            .values()
            .filter(|w| w.status == WorkerStatus::Busy)
            .count()
    }

    /// Get worker statistics
    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> JsValue {
        let stats = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&stats, &"worker_count".into(), &self.worker_count.into());
        let _ = js_sys::Reflect::set(&stats, &"idle_count".into(), &self.idle_count().into());
        let _ = js_sys::Reflect::set(&stats, &"busy_count".into(), &self.busy_count().into());
        let _ = js_sys::Reflect::set(&stats, &"queue_length".into(), &self.queue_length().into());

        let workers = self.workers.borrow();
        let total_completed: usize = workers.values().map(|w| w.tasks_completed).sum();
        let total_failed: usize = workers.values().map(|w| w.tasks_failed).sum();
        let total_time: f64 = workers.values().map(|w| w.total_time_ms).sum();

        let _ = js_sys::Reflect::set(&stats, &"total_completed".into(), &total_completed.into());
        let _ = js_sys::Reflect::set(&stats, &"total_failed".into(), &total_failed.into());
        let _ = js_sys::Reflect::set(&stats, &"total_time_ms".into(), &total_time.into());

        let avg_time = if total_completed > 0 {
            total_time / total_completed as f64
        } else {
            0.0
        };
        let _ = js_sys::Reflect::set(&stats, &"avg_time_ms".into(), &avg_time.into());

        stats.into()
    }

    /// Shutdown all workers
    #[wasm_bindgen(js_name = shutdown)]
    pub fn shutdown(&self) {
        for worker in self.workers.borrow_mut().values_mut() {
            worker.status = WorkerStatus::Stopping;
        }

        if self.verbose {
            web_sys::console::log_1(&"Worker pool shutting down".into());
        }
    }

    /// Enable verbose logging
    #[wasm_bindgen(js_name = setVerbose)]
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn get_next_id(&self) -> usize {
        let mut next_id = self.next_id.borrow_mut();
        let id = *next_id;
        *next_id += 1;
        id
    }
}

/// Worker handler for use inside worker threads
#[wasm_bindgen]
pub struct WorkerHandler {
    /// Solver context
    ctx: oxiz_solver::Context,
    /// Current task ID
    current_task: Option<String>,
    /// Statistics
    tasks_processed: usize,
}

#[wasm_bindgen]
impl WorkerHandler {
    /// Create a new worker handler
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            ctx: oxiz_solver::Context::new(),
            current_task: None,
            tasks_processed: 0,
        }
    }

    /// Handle a task
    #[wasm_bindgen(js_name = handleTask)]
    pub fn handle_task(&mut self, task: WorkerTask) -> JsValue {
        self.current_task = Some(task.get_id());

        let start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        // Process task based on type
        let result = match task.get_type().as_str() {
            "solve" => self.handle_solve(task.get_data()),
            "check-sat" => self.handle_check_sat(task.get_data()),
            "get-model" => self.handle_get_model(),
            _ => {
                let error = js_sys::Object::new();
                let _ = js_sys::Reflect::set(&error, &"error".into(), &"Unknown task type".into());
                error.into()
            }
        };

        let end_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(start_time);

        self.tasks_processed += 1;
        self.current_task = None;

        // Add timing info
        let _ = js_sys::Reflect::set(&result, &"time_ms".into(), &(end_time - start_time).into());
        let _ = js_sys::Reflect::set(&result, &"task_id".into(), &task.get_id().into());

        result
    }

    /// Get tasks processed count
    #[wasm_bindgen(js_name = tasksProcessed)]
    pub fn tasks_processed(&self) -> usize {
        self.tasks_processed
    }

    // Helper methods

    fn handle_solve(&mut self, data: JsValue) -> JsValue {
        let result = js_sys::Object::new();

        // Extract logic and assertions from data
        if let Ok(logic) = js_sys::Reflect::get(&data, &"logic".into())
            && let Some(logic_str) = logic.as_string()
        {
            self.ctx.set_logic(&logic_str);
        }

        // Execute assertions
        if let Ok(assertions) = js_sys::Reflect::get(&data, &"assertions".into())
            && let Some(arr) = assertions.dyn_ref::<js_sys::Array>()
        {
            for assertion in arr.iter() {
                if let Some(formula) = assertion.as_string() {
                    let script = format!("(assert {})", formula);
                    let _ = self.ctx.execute_script(&script);
                }
            }
        }

        // Check satisfiability
        let sat_result = self.ctx.check_sat();
        let status = match sat_result {
            oxiz_solver::SolverResult::Sat => "sat",
            oxiz_solver::SolverResult::Unsat => "unsat",
            oxiz_solver::SolverResult::Unknown => "unknown",
        };

        let _ = js_sys::Reflect::set(&result, &"status".into(), &status.into());

        result.into()
    }

    fn handle_check_sat(&mut self, _data: JsValue) -> JsValue {
        let result = js_sys::Object::new();
        let sat_result = self.ctx.check_sat();
        let status = match sat_result {
            oxiz_solver::SolverResult::Sat => "sat",
            oxiz_solver::SolverResult::Unsat => "unsat",
            oxiz_solver::SolverResult::Unknown => "unknown",
        };
        let _ = js_sys::Reflect::set(&result, &"status".into(), &status.into());
        result.into()
    }

    fn handle_get_model(&self) -> JsValue {
        let result = js_sys::Object::new();
        let model = self.ctx.get_model();
        let model_str = match model {
            Some(entries) => entries
                .iter()
                .map(|(name, sort, value)| format!("({} {} {})", name, sort, value))
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        };
        let _ = js_sys::Reflect::set(&result, &"model".into(), &model_str.into());
        result.into()
    }
}

impl Default for WorkerHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_message_type() {
        assert_eq!(WorkerMessageType::Init.name(), "init");
        assert_eq!(
            WorkerMessageType::from_str("execute"),
            Some(WorkerMessageType::Execute)
        );
        assert_eq!(WorkerMessageType::from_str("invalid"), None);
    }

    #[test]
    fn test_worker_task() {
        let task = WorkerTask::new("task1".to_string(), "solve".to_string(), JsValue::NULL);

        assert_eq!(task.get_id(), "task1");
        assert_eq!(task.get_type(), "solve");
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_worker_pool() {
        let pool = WorkerPool::new(4);
        pool.init();

        assert_eq!(pool.worker_count, 4);
        assert_eq!(pool.queue_length(), 0);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_worker_pool_submit() {
        let pool = WorkerPool::new(2);
        pool.init();

        let task = WorkerTask::new("task1".to_string(), "solve".to_string(), JsValue::NULL);

        pool.submit(task);
        assert_eq!(pool.queue_length(), 1);
    }

    #[test]
    fn test_worker_handler() {
        let handler = WorkerHandler::new();
        assert_eq!(handler.tasks_processed(), 0);
    }
}
