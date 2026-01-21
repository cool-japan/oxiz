//! Distributed SMT solving support
//!
//! This module enables distributed SMT solving across multiple machines using
//! a coordinator-worker architecture with cube-and-conquer parallelization.
//!
//! # Architecture
//!
//! - **Coordinator**: Listens for worker connections, generates cubes from the problem,
//!   distributes work units to workers, and aggregates results.
//! - **Worker**: Connects to coordinator, receives cubes (partial assignments) to solve,
//!   and reports results back.
//!
//! # Protocol
//!
//! Communication uses TCP sockets with JSON-encoded messages.
//! Workers send periodic heartbeats to indicate health.

use oxiz_solver::{Context, SolverResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// A literal in the SAT/SMT sense (variable with polarity)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    /// Variable index
    pub var: u32,
    /// Whether the literal is negated
    pub negated: bool,
}

#[allow(dead_code)]
impl Literal {
    /// Create a positive literal
    #[must_use]
    pub fn pos(var: u32) -> Self {
        Self {
            var,
            negated: false,
        }
    }

    /// Create a negative literal
    #[must_use]
    pub fn neg(var: u32) -> Self {
        Self { var, negated: true }
    }

    /// Negate this literal
    #[must_use]
    pub fn negate(self) -> Self {
        Self {
            var: self.var,
            negated: !self.negated,
        }
    }
}

/// Result status from solving a cube
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CubeSolverResult {
    /// The cube is satisfiable
    Sat,
    /// The cube is unsatisfiable (can be pruned)
    Unsat,
    /// Unknown (timeout or resource limit)
    Unknown,
}

impl From<SolverResult> for CubeSolverResult {
    fn from(result: SolverResult) -> Self {
        match result {
            SolverResult::Sat => CubeSolverResult::Sat,
            SolverResult::Unsat => CubeSolverResult::Unsat,
            SolverResult::Unknown => CubeSolverResult::Unknown,
        }
    }
}

/// Messages exchanged between coordinator and workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Work unit sent from coordinator to worker
    WorkUnit {
        /// Unique identifier for this cube
        cube_id: u64,
        /// The cube (partial assignment) to solve under
        assumptions: Vec<Literal>,
    },
    /// Result sent from worker to coordinator
    Result {
        /// The cube ID this result is for
        cube_id: u64,
        /// The solving result
        status: CubeSolverResult,
        /// Optional model variables (if SAT)
        #[serde(default)]
        model: Option<Vec<(String, String)>>,
    },
    /// Heartbeat message for health checking
    Heartbeat,
    /// Shutdown signal from coordinator
    Shutdown,
    /// Worker registration with coordinator
    Register {
        /// Worker identifier
        worker_id: String,
    },
    /// Acknowledgment from coordinator
    Ack {
        /// Message being acknowledged
        message: String,
    },
    /// Request for more work
    RequestWork,
    /// No more work available
    NoMoreWork,
    /// Problem script to solve
    Problem {
        /// The SMT-LIB2 script
        script: String,
        /// Optional logic setting
        logic: Option<String>,
    },
}

/// Configuration for distributed solving
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Address to bind/connect to (HOST:PORT)
    pub address: String,
    /// Number of cubes to generate
    pub num_cubes: usize,
    /// Heartbeat interval in seconds
    pub heartbeat_interval: u64,
    /// Worker timeout in seconds (how long before considering a worker dead)
    pub worker_timeout: u64,
    /// Per-cube solving timeout in seconds
    pub cube_timeout: u64,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:9876".to_string(),
            num_cubes: 64,
            heartbeat_interval: 10,
            worker_timeout: 60,
            cube_timeout: 300,
        }
    }
}

/// Worker state tracked by coordinator
#[derive(Debug)]
struct WorkerState {
    /// Worker identifier
    #[allow(dead_code)]
    worker_id: String,
    /// Currently assigned cube ID (if any)
    current_cube: Option<u64>,
    /// Last heartbeat time
    last_heartbeat: Instant,
    /// Stream for sending messages
    stream: TcpStream,
}

/// Cube state for tracking progress
#[derive(Debug, Clone)]
struct CubeState {
    /// The cube assumptions
    assumptions: Vec<Literal>,
    /// Whether this cube has been assigned
    assigned: bool,
    /// Whether this cube has been completed
    completed: bool,
    /// The result (if completed)
    result: Option<CubeSolverResult>,
}

/// Result from distributed solving
#[derive(Debug)]
pub struct DistributedResult {
    /// Overall result
    pub result: CubeSolverResult,
    /// Number of cubes processed
    pub cubes_processed: usize,
    /// Total time in milliseconds
    pub time_ms: u128,
    /// Number of workers used
    pub workers_used: usize,
}

/// Generate cubes using simple binary splitting
///
/// This is a basic cube generation strategy that creates 2^depth cubes
/// by splitting on the first `depth` boolean variables.
fn generate_cubes(num_cubes: usize, num_vars: u32) -> Vec<Vec<Literal>> {
    let depth = (num_cubes as f64).log2().ceil() as u32;
    let actual_num_cubes = 1usize << depth;

    let vars_to_use = depth.min(num_vars);

    let mut cubes = Vec::with_capacity(actual_num_cubes);

    for i in 0..actual_num_cubes {
        let mut cube = Vec::with_capacity(vars_to_use as usize);
        for bit in 0..vars_to_use {
            let var = bit;
            let negated = (i >> bit) & 1 == 0;
            cube.push(Literal { var, negated });
        }
        cubes.push(cube);
    }

    cubes
}

/// Run as coordinator node
///
/// The coordinator:
/// 1. Listens for worker connections
/// 2. Generates cubes from the problem
/// 3. Distributes cubes to workers
/// 4. Collects and aggregates results
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
pub fn run_coordinator(
    script: &str,
    config: &DistributedConfig,
) -> Result<DistributedResult, String> {
    let start_time = Instant::now();

    // Bind to address
    let listener = TcpListener::bind(&config.address)
        .map_err(|e| format!("Failed to bind to {}: {}", config.address, e))?;

    listener
        .set_nonblocking(true)
        .map_err(|e| format!("Failed to set non-blocking: {}", e))?;

    println!("Coordinator listening on {}", config.address);

    // Generate cubes (using a simple heuristic for number of variables)
    let num_vars = estimate_num_vars(script);
    let cubes = generate_cubes(config.num_cubes, num_vars);

    let cube_states: Arc<Mutex<HashMap<u64, CubeState>>> = Arc::new(Mutex::new(
        cubes
            .iter()
            .enumerate()
            .map(|(i, assumptions)| {
                (
                    i as u64,
                    CubeState {
                        assumptions: assumptions.clone(),
                        assigned: false,
                        completed: false,
                        result: None,
                    },
                )
            })
            .collect(),
    ));

    let workers: Arc<Mutex<HashMap<String, WorkerState>>> = Arc::new(Mutex::new(HashMap::new()));
    let script = Arc::new(script.to_string());
    let found_sat = Arc::new(AtomicBool::new(false));
    let all_done = Arc::new(AtomicBool::new(false));
    let workers_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Channel for results
    let (result_tx, result_rx): (
        Sender<(u64, CubeSolverResult)>,
        Receiver<(u64, CubeSolverResult)>,
    ) = channel();

    // Spawn result aggregator thread
    let cube_states_clone = Arc::clone(&cube_states);
    let found_sat_clone = Arc::clone(&found_sat);
    let all_done_clone = Arc::clone(&all_done);
    let _aggregator = thread::spawn(move || {
        for (cube_id, status) in result_rx {
            let mut states = cube_states_clone.lock().unwrap();
            if let Some(state) = states.get_mut(&cube_id) {
                state.completed = true;
                state.result = Some(status);

                if status == CubeSolverResult::Sat {
                    found_sat_clone.store(true, Ordering::SeqCst);
                    all_done_clone.store(true, Ordering::SeqCst);
                    break;
                }
            }

            // Check if all cubes are done
            let all_completed = states.values().all(|s| s.completed);
            if all_completed {
                all_done_clone.store(true, Ordering::SeqCst);
                break;
            }
        }
    });

    // Main coordinator loop
    let timeout = Duration::from_secs(config.worker_timeout);

    while !all_done.load(Ordering::SeqCst) {
        // Accept new connections
        if let Ok((stream, addr)) = listener.accept() {
            println!("Worker connected from {}", addr);

            let worker_id = format!("worker-{}", addr);
            let workers_clone = Arc::clone(&workers);
            let cube_states_clone = Arc::clone(&cube_states);
            let script_clone = Arc::clone(&script);
            let result_tx_clone = result_tx.clone();
            let found_sat_clone = Arc::clone(&found_sat);
            let all_done_clone = Arc::clone(&all_done);
            let workers_count_clone = Arc::clone(&workers_count);

            workers_count_clone.fetch_add(1, Ordering::SeqCst);

            // Spawn handler thread for this worker
            thread::spawn(move || {
                if let Err(e) = handle_worker(
                    stream,
                    &worker_id,
                    workers_clone,
                    cube_states_clone,
                    &script_clone,
                    result_tx_clone,
                    found_sat_clone,
                    all_done_clone,
                ) {
                    eprintln!("Worker {} error: {}", worker_id, e);
                }
                workers_count_clone.fetch_sub(1, Ordering::SeqCst);
            });
        }

        // Check for dead workers
        {
            let mut workers_guard = workers.lock().unwrap();
            let now = Instant::now();
            let dead_workers: Vec<String> = workers_guard
                .iter()
                .filter(|(_, state)| now.duration_since(state.last_heartbeat) > timeout)
                .map(|(id, _)| id.clone())
                .collect();

            for worker_id in dead_workers {
                eprintln!("Worker {} timed out", worker_id);
                if let Some(state) = workers_guard.remove(&worker_id) {
                    // Re-queue the cube if it was assigned
                    if let Some(cube_id) = state.current_cube {
                        let mut cube_states_guard = cube_states.lock().unwrap();
                        if let Some(cube_state) = cube_states_guard.get_mut(&cube_id) {
                            if !cube_state.completed {
                                cube_state.assigned = false;
                            }
                        }
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(100));
    }

    // Send shutdown to all workers
    {
        let workers_guard = workers.lock().unwrap();
        for (_, state) in workers_guard.iter() {
            let _ = send_message(&state.stream, &Message::Shutdown);
        }
    }

    // Determine final result
    let final_result = if found_sat.load(Ordering::SeqCst) {
        CubeSolverResult::Sat
    } else {
        let states = cube_states.lock().unwrap();
        let all_unsat = states
            .values()
            .all(|s| s.result == Some(CubeSolverResult::Unsat));
        if all_unsat {
            CubeSolverResult::Unsat
        } else {
            CubeSolverResult::Unknown
        }
    };

    let cubes_processed = {
        let states = cube_states.lock().unwrap();
        states.values().filter(|s| s.completed).count()
    };

    Ok(DistributedResult {
        result: final_result,
        cubes_processed,
        time_ms: start_time.elapsed().as_millis(),
        workers_used: workers_count.load(Ordering::SeqCst),
    })
}

/// Handle a connected worker
#[allow(clippy::too_many_arguments)]
fn handle_worker(
    stream: TcpStream,
    worker_id: &str,
    workers: Arc<Mutex<HashMap<String, WorkerState>>>,
    cube_states: Arc<Mutex<HashMap<u64, CubeState>>>,
    script: &str,
    result_tx: Sender<(u64, CubeSolverResult)>,
    found_sat: Arc<AtomicBool>,
    all_done: Arc<AtomicBool>,
) -> Result<(), String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("Failed to set read timeout: {}", e))?;

    // Clone stream for reading
    let read_stream = stream
        .try_clone()
        .map_err(|e| format!("Failed to clone stream: {}", e))?;
    let mut reader = BufReader::new(read_stream);

    // Register worker
    {
        let mut workers_guard = workers.lock().unwrap();
        workers_guard.insert(
            worker_id.to_string(),
            WorkerState {
                worker_id: worker_id.to_string(),
                current_cube: None,
                last_heartbeat: Instant::now(),
                stream: stream
                    .try_clone()
                    .map_err(|e| format!("Clone error: {}", e))?,
            },
        );
    }

    // Send problem to worker
    send_message(
        &stream,
        &Message::Problem {
            script: script.to_string(),
            logic: None,
        },
    )?;

    // Main loop: handle messages from worker
    loop {
        if found_sat.load(Ordering::SeqCst) || all_done.load(Ordering::SeqCst) {
            send_message(&stream, &Message::Shutdown)?;
            break;
        }

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // Connection closed
                break;
            }
            Ok(_) => {
                let msg: Message = serde_json::from_str(line.trim())
                    .map_err(|e| format!("Failed to parse message: {}", e))?;

                match msg {
                    Message::Heartbeat => {
                        let mut workers_guard = workers.lock().unwrap();
                        if let Some(state) = workers_guard.get_mut(worker_id) {
                            state.last_heartbeat = Instant::now();
                        }
                    }
                    Message::Result {
                        cube_id, status, ..
                    } => {
                        // Update worker state
                        {
                            let mut workers_guard = workers.lock().unwrap();
                            if let Some(state) = workers_guard.get_mut(worker_id) {
                                state.current_cube = None;
                            }
                        }

                        // Send result to aggregator
                        let _ = result_tx.send((cube_id, status));
                    }
                    Message::RequestWork => {
                        // Find an unassigned cube
                        let cube = {
                            let mut cube_states_guard = cube_states.lock().unwrap();
                            cube_states_guard
                                .iter_mut()
                                .find(|(_, state)| !state.assigned && !state.completed)
                                .map(|(id, state)| {
                                    state.assigned = true;
                                    (*id, state.assumptions.clone())
                                })
                        };

                        match cube {
                            Some((cube_id, assumptions)) => {
                                // Update worker state
                                {
                                    let mut workers_guard = workers.lock().unwrap();
                                    if let Some(state) = workers_guard.get_mut(worker_id) {
                                        state.current_cube = Some(cube_id);
                                    }
                                }

                                send_message(
                                    &stream,
                                    &Message::WorkUnit {
                                        cube_id,
                                        assumptions,
                                    },
                                )?;
                            }
                            None => {
                                send_message(&stream, &Message::NoMoreWork)?;
                            }
                        }
                    }
                    Message::Register { .. } => {
                        send_message(
                            &stream,
                            &Message::Ack {
                                message: "registered".to_string(),
                            },
                        )?;
                    }
                    _ => {}
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Timeout, continue loop
                thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                return Err(format!("Read error: {}", e));
            }
        }
    }

    // Remove worker
    {
        let mut workers_guard = workers.lock().unwrap();
        workers_guard.remove(worker_id);
    }

    Ok(())
}

/// Run as worker node
///
/// The worker:
/// 1. Connects to the coordinator
/// 2. Receives the problem and cubes to solve
/// 3. Solves each cube and reports results
/// 4. Sends periodic heartbeats
pub fn run_worker(config: &DistributedConfig) -> Result<(), String> {
    let stream = TcpStream::connect(&config.address)
        .map_err(|e| format!("Failed to connect to {}: {}", config.address, e))?;

    stream
        .set_read_timeout(Some(Duration::from_secs(config.heartbeat_interval * 2)))
        .map_err(|e| format!("Failed to set read timeout: {}", e))?;

    println!("Connected to coordinator at {}", config.address);

    let read_stream = stream
        .try_clone()
        .map_err(|e| format!("Failed to clone stream: {}", e))?;
    let mut reader = BufReader::new(read_stream);

    let stream = Arc::new(Mutex::new(stream));
    let running = Arc::new(AtomicBool::new(true));

    // Register with coordinator
    {
        let worker_id = format!("worker-{}", std::process::id());
        send_message_locked(&stream, &Message::Register { worker_id })?;
    }

    // Spawn heartbeat thread
    let stream_clone = Arc::clone(&stream);
    let running_clone = Arc::clone(&running);
    let heartbeat_interval = config.heartbeat_interval;
    let _heartbeat_handle = thread::spawn(move || {
        while running_clone.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_secs(heartbeat_interval));
            if let Err(e) = send_message_locked(&stream_clone, &Message::Heartbeat) {
                eprintln!("Failed to send heartbeat: {}", e);
                break;
            }
        }
    });

    let mut ctx: Option<Context> = None;
    let mut script: Option<String> = None;

    // Main worker loop
    loop {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                println!("Connection closed by coordinator");
                break;
            }
            Ok(_) => {
                let msg: Message = serde_json::from_str(line.trim())
                    .map_err(|e| format!("Failed to parse message: {}", e))?;

                match msg {
                    Message::Problem { script: s, logic } => {
                        println!("Received problem from coordinator");
                        script = Some(s.clone());

                        // Initialize context
                        let mut new_ctx = Context::new();
                        if let Some(ref l) = logic {
                            new_ctx.set_logic(l);
                        }
                        ctx = Some(new_ctx);

                        // Request first work unit
                        send_message_locked(&stream, &Message::RequestWork)?;
                    }
                    Message::WorkUnit {
                        cube_id,
                        assumptions,
                    } => {
                        println!(
                            "Received cube {} with {} assumptions",
                            cube_id,
                            assumptions.len()
                        );

                        let status = if let (Some(s), Some(c)) = (&script, &mut ctx) {
                            solve_cube(c, s, &assumptions, config.cube_timeout)
                        } else {
                            CubeSolverResult::Unknown
                        };

                        println!("Cube {} result: {:?}", cube_id, status);

                        send_message_locked(
                            &stream,
                            &Message::Result {
                                cube_id,
                                status,
                                model: None,
                            },
                        )?;

                        // Request more work
                        send_message_locked(&stream, &Message::RequestWork)?;
                    }
                    Message::NoMoreWork => {
                        println!("No more work available, waiting...");
                        thread::sleep(Duration::from_secs(1));
                        send_message_locked(&stream, &Message::RequestWork)?;
                    }
                    Message::Shutdown => {
                        println!("Received shutdown signal");
                        running.store(false, Ordering::SeqCst);
                        break;
                    }
                    Message::Ack { message } => {
                        println!("Received ack: {}", message);
                    }
                    _ => {}
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                return Err(format!("Read error: {}", e));
            }
        }
    }

    running.store(false, Ordering::SeqCst);
    Ok(())
}

/// Solve a cube (partial assignment) under the given assumptions
fn solve_cube(
    ctx: &mut Context,
    script: &str,
    assumptions: &[Literal],
    _timeout: u64,
) -> CubeSolverResult {
    // Reset context for fresh solving
    ctx.reset();

    // Execute the base script to set up the problem
    if ctx.execute_script(script).is_err() {
        return CubeSolverResult::Unknown;
    }

    // Add assumptions as assertions
    // In a real implementation, we would use check-sat-assuming or
    // properly map literals to actual term variables
    for lit in assumptions {
        let var_name = format!("__cube_var_{}", lit.var);
        let bool_sort = ctx.terms.sorts.bool_sort;
        let var_term = ctx.declare_const(&var_name, bool_sort);

        let assumption = if lit.negated {
            ctx.terms.mk_not(var_term)
        } else {
            var_term
        };
        ctx.assert(assumption);
    }

    // Solve
    ctx.check_sat().into()
}

/// Send a message over a TCP stream
fn send_message(stream: &TcpStream, msg: &Message) -> Result<(), String> {
    let json =
        serde_json::to_string(msg).map_err(|e| format!("Failed to serialize message: {}", e))?;

    let mut stream_clone = stream
        .try_clone()
        .map_err(|e| format!("Failed to clone stream: {}", e))?;

    stream_clone
        .write_all(format!("{}\n", json).as_bytes())
        .map_err(|e| format!("Failed to send message: {}", e))?;

    stream_clone
        .flush()
        .map_err(|e| format!("Failed to flush stream: {}", e))?;

    Ok(())
}

/// Send a message over a locked TCP stream
fn send_message_locked(stream: &Arc<Mutex<TcpStream>>, msg: &Message) -> Result<(), String> {
    let json =
        serde_json::to_string(msg).map_err(|e| format!("Failed to serialize message: {}", e))?;

    let mut guard = stream
        .lock()
        .map_err(|e| format!("Failed to lock stream: {}", e))?;

    guard
        .write_all(format!("{}\n", json).as_bytes())
        .map_err(|e| format!("Failed to send message: {}", e))?;

    guard
        .flush()
        .map_err(|e| format!("Failed to flush stream: {}", e))?;

    Ok(())
}

/// Estimate the number of boolean variables in the problem
fn estimate_num_vars(script: &str) -> u32 {
    // Simple heuristic: count declare-const statements with Bool type
    let mut count = 0u32;
    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.contains("declare-const") && trimmed.contains("Bool") {
            count += 1;
        }
        if trimmed.contains("declare-fun") && trimmed.contains("Bool") {
            count += 1;
        }
    }
    // Return at least 1 to avoid division by zero
    count.max(1)
}

/// Format a distributed result for display
pub fn format_distributed_result(result: &DistributedResult) -> String {
    let status = match result.result {
        CubeSolverResult::Sat => "sat",
        CubeSolverResult::Unsat => "unsat",
        CubeSolverResult::Unknown => "unknown",
    };

    format!(
        "{}\n; Distributed solving: {} cubes processed in {}ms using {} workers",
        status, result.cubes_processed, result.time_ms, result.workers_used
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let pos = Literal::pos(5);
        assert_eq!(pos.var, 5);
        assert!(!pos.negated);

        let neg = Literal::neg(3);
        assert_eq!(neg.var, 3);
        assert!(neg.negated);

        let negated = pos.negate();
        assert_eq!(negated.var, 5);
        assert!(negated.negated);
    }

    #[test]
    fn test_cube_generation() {
        let cubes = generate_cubes(4, 10);
        assert_eq!(cubes.len(), 4);

        // Each cube should have 2 literals (log2(4) = 2)
        for cube in &cubes {
            assert_eq!(cube.len(), 2);
        }

        let cubes = generate_cubes(8, 10);
        assert_eq!(cubes.len(), 8);

        for cube in &cubes {
            assert_eq!(cube.len(), 3);
        }
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::WorkUnit {
            cube_id: 42,
            assumptions: vec![Literal::pos(1), Literal::neg(2)],
        };

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();

        if let Message::WorkUnit {
            cube_id,
            assumptions,
        } = parsed
        {
            assert_eq!(cube_id, 42);
            assert_eq!(assumptions.len(), 2);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_estimate_num_vars() {
        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (declare-const q Bool)
            (declare-const r Bool)
            (assert (or p q r))
        "#;

        let count = estimate_num_vars(script);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.address, "127.0.0.1:9876");
        assert_eq!(config.num_cubes, 64);
        assert_eq!(config.heartbeat_interval, 10);
    }
}
