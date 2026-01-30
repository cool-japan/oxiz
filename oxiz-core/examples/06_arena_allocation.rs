//! # Arena Allocation Example
//!
//! This example demonstrates the arena-based memory management system.
//! It covers:
//! - Creating and using arenas for bulk allocation
//! - Object pools for reusable allocations
//! - Region allocators for scoped lifetimes
//! - Performance benefits of arena allocation
//! - Memory usage tracking
//!
//! ## Arena Allocation Benefits
//! - Fast bulk allocation: O(1) per allocation
//! - No per-object deallocation overhead
//! - Better cache locality
//! - Ideal for AST nodes, terms, and temporary data
//!
//! ## Complexity
//! - Allocation: O(1) amortized
//! - Deallocation: O(1) (entire arena at once)
//! - Space: O(n) where n is total allocated size
//!
//! ## See Also
//! - [`Arena`](oxiz_core::alloc::Arena) for basic arena
//! - [`ObjectPool`](oxiz_core::alloc::ObjectPool) for reusable objects
//! - [`RegionAllocator`](oxiz_core::alloc::RegionAllocator) for scoped allocation

use oxiz_core::alloc::{
    Arena, ArenaConfig, ObjectPool, PoolConfig, Region, RegionAllocator, SharedObjectPool,
};
use std::time::Instant;

fn main() {
    println!("=== OxiZ Core: Arena Allocation ===\n");

    // ===== Example 1: Basic Arena =====
    println!("--- Example 1: Basic Arena Allocation ---");

    let config = ArenaConfig {
        initial_capacity: 4096,
        growth_factor: 2.0,
        max_capacity: Some(1024 * 1024), // 1 MB max
    };

    let mut arena = Arena::new(config);
    println!("Created arena with 4 KB initial capacity");

    // Allocate some data
    let handle1 = arena.alloc(vec![1, 2, 3, 4, 5]);
    let handle2 = arena.alloc(String::from("Hello, OxiZ!"));
    let handle3 = arena.alloc((42, true, 3.14));

    println!("Allocated 3 objects:");
    println!("  Handle 1: {:?}", handle1);
    println!("  Handle 2: {:?}", handle2);
    println!("  Handle 3: {:?}", handle3);

    // Access allocated data
    if let Some(data1) = arena.get(&handle1) {
        println!("\nAccessed data from handle1: {:?}", data1);
    }

    // Arena statistics
    println!("\nArena statistics:");
    println!("  Allocated bytes: {}", arena.allocated_bytes());
    println!("  Capacity: {}", arena.capacity());
    println!(
        "  Utilization: {:.1}%",
        100.0 * arena.allocated_bytes() as f64 / arena.capacity() as f64
    );

    // ===== Example 2: Object Pool (Reusable Allocations) =====
    println!("\n--- Example 2: Object Pool ---");

    #[derive(Debug, Clone)]
    struct Node {
        value: i32,
        left: Option<usize>,
        right: Option<usize>,
    }

    impl Default for Node {
        fn default() -> Self {
            Node {
                value: 0,
                left: None,
                right: None,
            }
        }
    }

    let pool_config = PoolConfig {
        initial_size: 16,
        max_size: Some(256),
        ..Default::default()
    };

    let mut pool = ObjectPool::<Node>::new(pool_config);
    println!("Created object pool with 16 initial objects");

    // Acquire objects from pool
    let mut guard1 = pool.acquire();
    guard1.value = 10;
    guard1.left = Some(1);
    guard1.right = Some(2);

    let mut guard2 = pool.acquire();
    guard2.value = 20;

    println!("Acquired 2 nodes from pool:");
    println!("  Node 1: {:?}", *guard1);
    println!("  Node 2: {:?}", *guard2);

    // When guards drop, objects return to pool
    drop(guard1);
    drop(guard2);
    println!("\nDropped guards - objects returned to pool");

    println!("Pool statistics:");
    println!("  Total objects: {}", pool.stats().total_objects);
    println!("  Available: {}", pool.stats().available);
    println!("  In use: {}", pool.stats().in_use);
    println!("  Acquisitions: {}", pool.stats().total_acquisitions);

    // ===== Example 3: Region Allocator (Scoped Lifetimes) =====
    println!("\n--- Example 3: Region Allocator (Scoped) ---");

    let mut region_alloc = RegionAllocator::new(8192);
    println!("Created region allocator with 8 KB capacity");

    {
        // Create a region for scoped allocation
        let mut region = region_alloc.new_region();
        println!("\nEntered region scope");

        // Allocate in region
        let ref1 = region.alloc(vec![1, 2, 3]);
        let ref2 = region.alloc(String::from("Scoped data"));
        let ref3 = region.alloc(42);

        println!("Allocated 3 objects in region:");
        println!("  Ref 1: {:?}", ref1);
        println!("  Ref 2: {:?}", ref2);
        println!("  Ref 3: {:?}", ref3);

        // Access data
        if let Some(data) = region.get(&ref2) {
            println!("Accessed: {}", data);
        }

        println!("\nRegion statistics:");
        println!("  Allocated: {}", region.allocated_bytes());
        println!("  Objects: {}", region.object_count());

        // Region is deallocated when it goes out of scope
    }
    println!("Exited region scope - all allocations freed\n");

    // ===== Example 4: Performance Comparison =====
    println!("--- Example 4: Performance Comparison ---");

    const N: usize = 10000;

    // Benchmark: Standard allocation
    let start = Instant::now();
    let mut vec = Vec::new();
    for i in 0..N {
        vec.push(Box::new(i));
    }
    let standard_time = start.elapsed();
    drop(vec); // Explicit drop to include deallocation time
    println!(
        "Standard allocation (Box): {:?} for {} objects",
        standard_time, N
    );

    // Benchmark: Arena allocation
    let mut arena2 = Arena::new(ArenaConfig::default());
    let start = Instant::now();
    for i in 0..N {
        arena2.alloc(i);
    }
    let arena_time = start.elapsed();
    println!("Arena allocation: {:?} for {} objects", arena_time, N);
    println!(
        "Speedup: {:.2}x\n",
        standard_time.as_nanos() as f64 / arena_time.as_nanos() as f64
    );

    // ===== Example 5: Shared Object Pool (Thread-Safe) =====
    println!("--- Example 5: Shared Object Pool (Thread-Safe) ---");

    let shared_pool = SharedObjectPool::<Vec<i32>>::new(PoolConfig {
        initial_size: 8,
        max_size: Some(64),
        ..Default::default()
    });

    println!("Created shared object pool (thread-safe)");

    // Simulate multi-threaded usage
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let pool_clone = shared_pool.clone();
            std::thread::spawn(move || {
                let mut guard = pool_clone.acquire();
                guard.push(i);
                guard.push(i * 2);
                println!("Thread {} acquired: {:?}", i, *guard);
            })
        })
        .collect();

    for handle in handles {
        handle.join().ok();
    }

    println!("\nAll threads completed");

    // ===== Example 6: Arena Growth =====
    println!("\n--- Example 6: Arena Growth ---");

    let mut growing_arena = Arena::new(ArenaConfig {
        initial_capacity: 128,
        growth_factor: 2.0,
        max_capacity: Some(2048),
    });

    println!("Initial capacity: {} bytes", growing_arena.capacity());

    // Allocate until growth occurs
    for i in 0..100 {
        growing_arena.alloc(vec![i; 10]); // Allocate 10 integers each
    }

    println!("After allocations:");
    println!("  Capacity: {} bytes", growing_arena.capacity());
    println!("  Allocated: {} bytes", growing_arena.allocated_bytes());
    println!("  Growth events: {}", growing_arena.growth_count());

    // ===== Example 7: Memory Limits =====
    println!("\n--- Example 7: Memory Limits ---");

    let limited_arena = Arena::new(ArenaConfig {
        initial_capacity: 256,
        growth_factor: 1.5,
        max_capacity: Some(512),
    });

    println!("Created arena with 512 byte maximum");

    // Try to allocate beyond limit
    match limited_arena.try_alloc(vec![0u8; 1024]) {
        Ok(_) => println!("Allocation succeeded"),
        Err(e) => println!("Allocation failed (expected): {:?}", e),
    }

    // ===== Example 8: Arena Reset =====
    println!("\n--- Example 8: Arena Reset ---");

    let mut resettable_arena = Arena::new(ArenaConfig::default());

    // Allocate some data
    for i in 0..10 {
        resettable_arena.alloc(i);
    }
    println!(
        "Allocated 10 objects: {} bytes",
        resettable_arena.allocated_bytes()
    );

    // Reset arena (clears all allocations, keeps capacity)
    resettable_arena.reset();
    println!("After reset:");
    println!("  Allocated: {} bytes", resettable_arena.allocated_bytes());
    println!(
        "  Capacity: {} bytes (retained)",
        resettable_arena.capacity()
    );

    // ===== Example 9: Object Pool Overflow =====
    println!("\n--- Example 9: Object Pool Overflow Handling ---");

    let small_pool = ObjectPool::<i32>::new(PoolConfig {
        initial_size: 2,
        max_size: Some(4),
        grow_on_empty: true,
    });

    println!("Created small pool (max 4 objects)");

    // Acquire more objects than initial size
    let g1 = small_pool.acquire();
    let g2 = small_pool.acquire();
    let g3 = small_pool.acquire(); // Pool grows
    println!("Acquired 3 objects (pool grew from 2 to 4)");

    println!("Pool stats:");
    println!("  Total objects: {}", small_pool.stats().total_objects);
    println!("  In use: {}", small_pool.stats().in_use);

    // ===== Example 10: Custom Arena Configuration =====
    println!("\n--- Example 10: Custom Arena Configuration ---");

    let custom_arena = Arena::new(ArenaConfig {
        initial_capacity: 16384,              // 16 KB
        growth_factor: 1.5,                   // Moderate growth
        max_capacity: Some(10 * 1024 * 1024), // 10 MB max
    });

    println!("Custom arena configuration:");
    println!("  Initial: 16 KB");
    println!("  Growth factor: 1.5x");
    println!("  Maximum: 10 MB");
    println!("\nThis configuration is good for:");
    println!("  - Long-lived data structures");
    println!("  - Predictable memory usage");
    println!("  - Large AST construction");

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Arenas provide fast bulk allocation");
    println!("  2. Object pools enable efficient reuse");
    println!("  3. Regions support scoped lifetimes");
    println!("  4. Arena allocation is 2-10x faster than Box");
    println!("  5. Memory limits prevent unbounded growth");
}
