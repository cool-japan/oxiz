//! # Resource Limits Example
//!
//! This example demonstrates resource management and limits.
//! It covers:
//! - Time limits (timeouts)
//! - Memory limits
//! - Iteration/depth limits
//! - Resource tracking
//! - Graceful handling of limit violations
//!
//! ## Resource Management
//! Resource limits prevent unbounded computation, enforce fair resource
//! allocation, and provide predictable behavior in production systems.
//!
//! ## Complexity
//! - Time tracking: O(1) overhead per check
//! - Memory tracking: O(1) overhead
//! - Limit checks: O(1)
//!
//! ## See Also
//! - [`ResourceManager`](oxiz_core::resource::ResourceManager)
//! - [`ResourceLimits`](oxiz_core::config::ResourceLimits)

use oxiz_core::ast::TermManager;
use oxiz_core::config::{Config, ResourceLimits};
use oxiz_core::resource::{LimitStatus, ResourceManager};
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    println!("=== OxiZ Core: Resource Limits ===\n");

    // ===== Example 1: Basic Time Limit =====
    println!("--- Example 1: Time Limits (Timeouts) ---");

    let mut resources = ResourceManager::new();
    resources.set_time_limit(Duration::from_millis(500));

    println!("Set time limit: 500 ms");

    // Simulate work
    let start = Instant::now();
    loop {
        // Do some work
        let _ = (0..10000).sum::<i32>();

        // Check limits periodically
        if let Err(status) = resources.check_limits() {
            println!("Time limit exceeded after {:?}", start.elapsed());
            println!("Status: {:?}", status);
            break;
        }

        if start.elapsed() > Duration::from_secs(1) {
            println!("Unexpected: timeout not triggered");
            break;
        }
    }

    // ===== Example 2: Memory Limits =====
    println!("\n--- Example 2: Memory Limits ---");

    let mut mem_resources = ResourceManager::new();
    mem_resources.set_memory_limit(10 * 1024 * 1024); // 10 MB

    println!("Set memory limit: 10 MB");

    // Track memory usage
    let mut allocations = Vec::new();
    let chunk_size = 1024 * 1024; // 1 MB chunks

    for i in 0..15 {
        allocations.push(vec![0u8; chunk_size]);
        mem_resources.update_memory_usage(allocations.len() * chunk_size);

        println!("Allocated {} MB", i + 1);

        match mem_resources.check_limits() {
            Ok(_) => continue,
            Err(LimitStatus::MemoryLimit) => {
                println!("Memory limit exceeded!");
                println!("  Allocated: {} MB", allocations.len());
                println!("  Limit: 10 MB");
                break;
            }
            Err(e) => {
                println!("Unexpected status: {:?}", e);
                break;
            }
        }
    }

    // ===== Example 3: Iteration Limits =====
    println!("\n--- Example 3: Iteration Limits ---");

    let mut iter_resources = ResourceManager::new();
    iter_resources.set_iteration_limit(100);

    println!("Set iteration limit: 100");

    let mut count = 0;
    loop {
        count += 1;
        iter_resources.increment_iterations();

        if count % 25 == 0 {
            println!("Iteration: {}", count);
        }

        if let Err(_) = iter_resources.check_limits() {
            println!("Iteration limit exceeded at iteration {}", count);
            break;
        }

        if count > 1000 {
            println!("Unexpected: iteration limit not triggered");
            break;
        }
    }

    // ===== Example 4: Combined Limits =====
    println!("\n--- Example 4: Combined Limits (Time + Memory + Iterations) ---");

    let mut combined = ResourceManager::new();
    combined.set_time_limit(Duration::from_secs(2));
    combined.set_memory_limit(50 * 1024 * 1024); // 50 MB
    combined.set_iteration_limit(1000);

    println!("Combined limits:");
    println!("  Time: 2 seconds");
    println!("  Memory: 50 MB");
    println!("  Iterations: 1000");

    let start = Instant::now();
    let mut data = Vec::new();

    for i in 0..10000 {
        // Simulate work
        data.push(vec![0u8; 10000]); // 10 KB per iteration
        combined.increment_iterations();
        combined.update_memory_usage(data.len() * 10000);

        if i % 100 == 0 {
            match combined.check_limits() {
                Ok(_) => continue,
                Err(status) => {
                    println!("\nLimit exceeded:");
                    println!("  Status: {:?}", status);
                    println!("  Time elapsed: {:?}", start.elapsed());
                    println!("  Iterations: {}", i);
                    println!("  Memory: {} MB", data.len() * 10000 / (1024 * 1024));
                    break;
                }
            }
        }
    }

    // ===== Example 5: Resource Configuration =====
    println!("\n--- Example 5: Resource Configuration ---");

    let config = Config {
        resource_limits: ResourceLimits {
            max_time_ms: Some(5000),     // 5 seconds
            max_memory_mb: Some(100),    // 100 MB
            max_iterations: Some(10000), // 10k iterations
            max_depth: Some(100),        // Max recursion depth
        },
        ..Default::default()
    };

    println!("Configuration-based limits:");
    println!("  Time: {} ms", config.resource_limits.max_time_ms.unwrap());
    println!(
        "  Memory: {} MB",
        config.resource_limits.max_memory_mb.unwrap()
    );
    println!(
        "  Iterations: {}",
        config.resource_limits.max_iterations.unwrap()
    );
    println!("  Depth: {}", config.resource_limits.max_depth.unwrap());

    let mut config_resources = ResourceManager::from_config(&config);
    println!("\nResourceManager created from config");

    // ===== Example 6: Depth Limits (Recursion) =====
    println!("\n--- Example 6: Depth Limits (Recursion) ---");

    fn recursive_function(
        depth: usize,
        resources: &mut ResourceManager,
    ) -> Result<(), LimitStatus> {
        resources.enter_scope(); // Track recursion depth

        if let Err(status) = resources.check_limits() {
            resources.exit_scope();
            return Err(status);
        }

        if depth > 0 {
            recursive_function(depth - 1, resources)?;
        }

        resources.exit_scope();
        Ok(())
    }

    let mut depth_resources = ResourceManager::new();
    depth_resources.set_depth_limit(50);

    println!("Set depth limit: 50");

    match recursive_function(100, &mut depth_resources) {
        Ok(_) => println!("Unexpected success"),
        Err(status) => {
            println!("Depth limit exceeded:");
            println!("  Status: {:?}", status);
            println!("  Current depth: {}", depth_resources.current_depth());
        }
    }

    // ===== Example 7: Resource Reset =====
    println!("\n--- Example 7: Resource Reset ---");

    let mut resettable = ResourceManager::new();
    resettable.set_iteration_limit(10);

    for i in 0..5 {
        resettable.increment_iterations();
    }
    println!("Iterations: {}", resettable.current_iterations());

    resettable.reset();
    println!("After reset: {}", resettable.current_iterations());

    // ===== Example 8: Incremental Solving with Limits =====
    println!("\n--- Example 8: Incremental Solving with Limits ---");

    let mut tm = TermManager::new();
    let mut solver_resources = ResourceManager::new();
    solver_resources.set_time_limit(Duration::from_millis(100));

    println!("Simulating incremental solving with 100ms per query:");

    for query in 1..=5 {
        solver_resources.reset(); // Reset for each query
        let start = Instant::now();

        // Simulate solver work
        thread::sleep(Duration::from_millis(50)); // Simulate 50ms work

        match solver_resources.check_limits() {
            Ok(_) => {
                println!("  Query {}: SAT ({:?})", query, start.elapsed());
            }
            Err(_) => {
                println!("  Query {}: TIMEOUT", query);
            }
        }
    }

    // ===== Example 9: Resource Monitoring =====
    println!("\n--- Example 9: Resource Monitoring ---");

    let mut monitored = ResourceManager::new();
    monitored.set_time_limit(Duration::from_secs(10));
    monitored.set_memory_limit(200 * 1024 * 1024);

    println!("Resource monitoring:");
    println!("  Time used: {:?}", monitored.time_used());
    println!("  Time remaining: {:?}", monitored.time_remaining());
    println!("  Memory used: {} bytes", monitored.memory_used());
    println!("  Memory available: {} bytes", monitored.memory_available());
    println!("  Usage: {:.1}%", monitored.memory_usage_percent());

    // ===== Example 10: Graceful Degradation =====
    println!("\n--- Example 10: Graceful Degradation ---");

    fn solve_with_fallback(resources: &mut ResourceManager, formula: &str) -> String {
        // Try complete solver
        resources.set_time_limit(Duration::from_millis(100));

        match try_complete_solver(resources) {
            Ok(result) => result,
            Err(LimitStatus::Timeout) => {
                println!("  Complete solver timed out, trying incomplete solver");
                try_incomplete_solver()
            }
            Err(_) => "unknown".to_string(),
        }
    }

    fn try_complete_solver(resources: &mut ResourceManager) -> Result<String, LimitStatus> {
        thread::sleep(Duration::from_millis(150)); // Simulate slow solver
        resources.check_limits()?;
        Ok("sat".to_string())
    }

    fn try_incomplete_solver() -> String {
        // Fast but incomplete
        "sat (incomplete)".to_string()
    }

    let mut fallback_resources = ResourceManager::new();
    let result = solve_with_fallback(&mut fallback_resources, "(and p q)");
    println!("Graceful degradation result: {}", result);

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Time limits prevent unbounded execution");
    println!("  2. Memory limits prevent OOM errors");
    println!("  3. Iteration limits prevent infinite loops");
    println!("  4. Depth limits prevent stack overflow");
    println!("  5. Combined limits provide robust resource control");
    println!("  6. Resource monitoring enables adaptive strategies");
    println!("  7. Graceful degradation maintains availability");
}
