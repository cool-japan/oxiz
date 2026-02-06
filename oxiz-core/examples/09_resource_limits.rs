//! # Resource Limits Example
//!
//! This example demonstrates resource management and limits.
//! It covers:
//! - Time limits (timeouts)
//! - Memory limits
//! - Decision/conflict limits
//! - Resource tracking with Statistics
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
//! - [`Statistics`](oxiz_core::statistics::Statistics)

use oxiz_core::config::ResourceLimits;
use oxiz_core::resource::{LimitStatus, ResourceManager};
use oxiz_core::statistics::Statistics;
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    println!("=== OxiZ Core: Resource Limits ===\n");

    // ===== Example 1: Basic Time Limit =====
    println!("--- Example 1: Time Limits (Timeouts) ---");

    let limits = ResourceLimits {
        time_limit: Some(Duration::from_millis(500)),
        decision_limit: None,
        conflict_limit: None,
        memory_limit: None,
    };

    let mut resources = ResourceManager::new(limits);
    resources.start(); // Start the timer

    println!("Set time limit: 500 ms");

    let stats = Statistics::new();
    let start = Instant::now();

    loop {
        // Do some work
        let _: i32 = (0..10000).sum();

        // Check limits periodically
        let status = resources.check_limits(&stats);
        if status != LimitStatus::Ok {
            println!("Time limit exceeded after {:?}", start.elapsed());
            println!("Status: {:?}", status);
            break;
        }

        if start.elapsed() > Duration::from_secs(1) {
            println!("Unexpected: timeout not triggered");
            break;
        }
    }

    // ===== Example 2: Decision Limits =====
    println!("\n--- Example 2: Decision Limits ---");

    let limits = ResourceLimits {
        time_limit: None,
        decision_limit: Some(100),
        conflict_limit: None,
        memory_limit: None,
    };

    let resources = ResourceManager::new(limits);
    let mut stats = Statistics::new();

    println!("Set decision limit: 100");

    for i in 0..150 {
        stats.inc_decisions();

        if i % 25 == 0 {
            println!("Decision: {}", i + 1);
        }

        let status = resources.check_limits(&stats);
        if status != LimitStatus::Ok {
            println!("Decision limit exceeded at decision {}", i + 1);
            println!("Status: {:?}", status);
            break;
        }
    }

    // ===== Example 3: Conflict Limits =====
    println!("\n--- Example 3: Conflict Limits ---");

    let limits = ResourceLimits {
        time_limit: None,
        decision_limit: None,
        conflict_limit: Some(50),
        memory_limit: None,
    };

    let resources = ResourceManager::new(limits);
    let mut stats = Statistics::new();

    println!("Set conflict limit: 50");

    for i in 0..100 {
        stats.inc_conflicts();

        if i % 10 == 0 {
            println!("Conflict: {}", i + 1);
        }

        let status = resources.check_limits(&stats);
        if status != LimitStatus::Ok {
            println!("Conflict limit exceeded at conflict {}", i + 1);
            println!("Status: {:?}", status);
            break;
        }
    }

    // ===== Example 4: Memory Limits =====
    println!("\n--- Example 4: Memory Limits ---");

    let limits = ResourceLimits {
        time_limit: None,
        decision_limit: None,
        conflict_limit: None,
        memory_limit: Some(10 * 1024 * 1024), // 10 MB
    };

    let resources = ResourceManager::new(limits);
    let mut stats = Statistics::new();

    println!("Set memory limit: 10 MB");

    // Simulate memory usage
    for i in 1..=15 {
        let usage = i * 1024 * 1024; // i MB
        stats.set_memory_used(usage as u64);

        println!("Memory used: {} MB", i);

        let status = resources.check_limits(&stats);
        if status != LimitStatus::Ok {
            println!("Memory limit exceeded!");
            println!("  Limit: 10 MB");
            println!("  Status: {:?}", status);
            break;
        }
    }

    // ===== Example 5: Combined Limits =====
    println!("\n--- Example 5: Combined Limits ---");

    let limits = ResourceLimits {
        time_limit: Some(Duration::from_secs(2)),
        decision_limit: Some(1000),
        conflict_limit: Some(500),
        memory_limit: Some(50 * 1024 * 1024), // 50 MB
    };

    let mut resources = ResourceManager::new(limits);
    resources.start();

    println!("Combined limits:");
    println!("  Time: 2 seconds");
    println!("  Decisions: 1000");
    println!("  Conflicts: 500");
    println!("  Memory: 50 MB");

    let mut stats = Statistics::new();

    for i in 0..2000 {
        stats.inc_decisions();

        // Simulate some conflicts
        if i % 5 == 0 {
            stats.inc_conflicts();
        }

        // Simulate memory growth
        stats.set_memory_used((i as u64) * 10000);

        if i % 100 == 0 {
            let status = resources.check_limits(&stats);
            if status != LimitStatus::Ok {
                println!("\nLimit exceeded:");
                println!("  Status: {:?}", status);
                println!("  Decisions: {}", stats.decisions);
                println!("  Conflicts: {}", stats.conflicts);
                println!("  Memory: {} bytes", stats.memory_used);
                break;
            }
        }
    }

    // ===== Example 6: Resource Limits from Config =====
    println!("\n--- Example 6: ResourceLimits Structure ---");

    let config_limits = ResourceLimits {
        time_limit: Some(Duration::from_secs(60)),
        decision_limit: Some(10000),
        conflict_limit: Some(5000),
        memory_limit: Some(100 * 1024 * 1024),
    };

    println!("ResourceLimits configuration:");
    println!(
        "  Time: {:?}",
        config_limits
            .time_limit
            .map_or("unlimited".to_string(), |d| format!("{:?}", d))
    );
    println!(
        "  Decisions: {}",
        config_limits
            .decision_limit
            .map_or("unlimited".to_string(), |d| d.to_string())
    );
    println!(
        "  Conflicts: {}",
        config_limits
            .conflict_limit
            .map_or("unlimited".to_string(), |c| c.to_string())
    );
    println!(
        "  Memory: {}",
        config_limits
            .memory_limit
            .map_or("unlimited".to_string(), |m| format!("{} bytes", m))
    );

    // ===== Example 7: Resource Reset =====
    println!("\n--- Example 7: Resource Reset ---");

    let limits = ResourceLimits {
        time_limit: Some(Duration::from_secs(10)),
        decision_limit: None,
        conflict_limit: None,
        memory_limit: None,
    };

    let mut resources = ResourceManager::new(limits);
    resources.start();

    // Wait a bit
    thread::sleep(Duration::from_millis(50));

    println!("Elapsed before reset: {:?}", resources.elapsed());

    resources.reset();
    println!("After reset, elapsed: {:?}", resources.elapsed());

    // ===== Example 8: Remaining Budget =====
    println!("\n--- Example 8: Remaining Budget Tracking ---");

    let limits = ResourceLimits {
        time_limit: Some(Duration::from_secs(10)),
        decision_limit: Some(100),
        conflict_limit: Some(50),
        memory_limit: None,
    };

    let mut resources = ResourceManager::new(limits);
    resources.start();

    let mut stats = Statistics::new();
    stats.decisions = 30;
    stats.conflicts = 20;

    println!("Current usage:");
    println!("  Decisions: {}", stats.decisions);
    println!("  Conflicts: {}", stats.conflicts);

    println!("\nRemaining budget:");
    println!(
        "  Time remaining: {:?}",
        resources
            .remaining_time()
            .map_or("N/A".to_string(), |d| format!("{:?}", d))
    );
    println!(
        "  Decisions remaining: {}",
        resources
            .remaining_decisions(&stats)
            .map_or("unlimited".to_string(), |d| d.to_string())
    );
    println!(
        "  Conflicts remaining: {}",
        resources
            .remaining_conflicts(&stats)
            .map_or("unlimited".to_string(), |c| c.to_string())
    );

    // ===== Example 9: Incremental Solving with Limits =====
    println!("\n--- Example 9: Incremental Solving with Limits ---");

    println!("Simulating incremental solving with 100ms per query:");

    for query in 1..=5 {
        let limits = ResourceLimits {
            time_limit: Some(Duration::from_millis(100)),
            decision_limit: None,
            conflict_limit: None,
            memory_limit: None,
        };

        let mut resources = ResourceManager::new(limits);
        resources.start();

        let start = Instant::now();

        // Simulate solver work
        thread::sleep(Duration::from_millis(50));

        let stats = Statistics::new();
        let status = resources.check_limits(&stats);

        match status {
            LimitStatus::Ok => {
                println!("  Query {}: SAT ({:?})", query, start.elapsed());
            }
            _ => {
                println!("  Query {}: {:?}", query, status);
            }
        }
    }

    // ===== Example 10: Limit Status Types =====
    println!("\n--- Example 10: LimitStatus Variants ---");

    println!("LimitStatus variants:");
    println!("  {:?} - All limits OK", LimitStatus::Ok);
    println!("  {:?} - Time limit exceeded", LimitStatus::TimeExceeded);
    println!(
        "  {:?} - Decision limit exceeded",
        LimitStatus::DecisionExceeded
    );
    println!(
        "  {:?} - Conflict limit exceeded",
        LimitStatus::ConflictExceeded
    );
    println!(
        "  {:?} - Memory limit exceeded",
        LimitStatus::MemoryExceeded
    );

    // ===== Example 11: Statistics Tracking =====
    println!("\n--- Example 11: Statistics Tracking ---");

    let mut stats = Statistics::new();

    // Simulate solver activity
    for _ in 0..100 {
        stats.inc_decisions();
        stats.inc_propagations();
        if stats.decisions.is_multiple_of(10) {
            stats.inc_conflicts();
            stats.inc_restarts();
        }
    }

    stats.add_solve_time(Duration::from_millis(500));
    stats.set_memory_used(1024 * 1024); // 1 MB

    println!("Statistics after simulation:");
    println!("  Decisions: {}", stats.decisions);
    println!("  Propagations: {}", stats.propagations);
    println!("  Conflicts: {}", stats.conflicts);
    println!("  Restarts: {}", stats.restarts);
    println!("  Solve time: {:?}", stats.solve_time);
    println!(
        "  Memory used: {}",
        Statistics::format_memory(stats.memory_used)
    );

    println!("\nPerformance metrics:");
    println!("  Decisions/sec: {:.0}", stats.decisions_per_second());
    println!("  Conflicts/sec: {:.0}", stats.conflicts_per_second());

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Time limits prevent unbounded execution");
    println!("  2. Decision/conflict limits control search effort");
    println!("  3. Memory limits prevent OOM errors");
    println!("  4. Combined limits provide robust resource control");
    println!("  5. Statistics track solver behavior for tuning");
    println!("  6. Remaining budget helps with adaptive strategies");
}
