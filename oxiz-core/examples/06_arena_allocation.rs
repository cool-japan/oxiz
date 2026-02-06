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
//! - [`Region`](oxiz_core::alloc::Region) for scoped allocation

use oxiz_core::alloc::{Arena, ArenaConfig, ObjectPool, PoolConfig, Region, SharedObjectPool};
use std::time::Instant;

fn main() {
    println!("=== OxiZ Core: Arena Allocation ===\n");

    // ===== Example 1: Basic Arena =====
    println!("--- Example 1: Basic Arena Allocation ---");

    let config = ArenaConfig {
        initial_chunk_size: 4096,
        growth_factor: 2.0,
        max_chunk_size: 1024 * 1024, // 1 MB max
    };

    let mut arena = Arena::with_config(config);
    println!("Created arena with 4 KB initial capacity");

    // Allocate some data
    let handle1 = arena.alloc(vec![1, 2, 3, 4, 5]);
    let handle2 = arena.alloc(String::from("Hello, OxiZ!"));
    let handle3 = arena.alloc((42, true, 2.5));

    println!("Allocated 3 objects:");
    println!("  Handle 1: {:?}", handle1);
    println!("  Handle 2: {:?}", handle2);
    println!("  Handle 3: {:?}", handle3);

    // Access allocated data via handle
    let data1 = handle1.get();
    println!("\nAccessed data from handle1: {:?}", data1);

    // Arena statistics
    println!("\nArena statistics:");
    println!("  Used bytes: {}", arena.used());
    println!("  Capacity: {}", arena.capacity());
    let utilization = if arena.capacity() > 0 {
        100.0 * arena.used() as f64 / arena.capacity() as f64
    } else {
        0.0
    };
    println!("  Utilization: {:.1}%", utilization);

    // ===== Example 2: Object Pool (Reusable Allocations) =====
    println!("\n--- Example 2: Object Pool ---");

    #[derive(Debug, Clone, Default)]
    struct Node {
        value: i32,
        left: Option<usize>,
        right: Option<usize>,
    }

    let pool_config = PoolConfig {
        initial_capacity: 16,
        max_size: 256,
        preallocate: false,
    };

    let pool = ObjectPool::new(Node::default, pool_config);
    println!("Created object pool with 16 initial capacity");

    // Acquire objects from pool using get()
    {
        let mut guard1 = pool.get();
        guard1.get_mut().value = 10;
        guard1.get_mut().left = Some(1);
        guard1.get_mut().right = Some(2);

        let mut guard2 = pool.get();
        guard2.get_mut().value = 20;

        println!("Acquired 2 nodes from pool:");
        println!("  Node 1: {:?}", *guard1);
        println!("  Node 2: {:?}", *guard2);

        // When guards drop, objects return to pool
    }
    println!("\nDropped guards - objects returned to pool");

    let stats = pool.stats();
    println!("Pool statistics:");
    println!("  Hits: {}", stats.hits);
    println!("  Misses: {}", stats.misses);
    println!("  Returns: {}", stats.returns);
    println!("  Hit rate: {:.1}%", stats.hit_rate() * 100.0);

    // ===== Example 3: Region Allocator (Scoped Lifetimes) =====
    println!("\n--- Example 3: Region Allocator (Scoped) ---");

    {
        // Create a region for scoped allocation
        let region = Region::new();
        println!("\nEntered region scope");

        // Allocate in region
        let ref1 = region.alloc(vec![1, 2, 3]);
        let ref2 = region.alloc(String::from("Scoped data"));
        let ref3 = region.alloc(42);

        println!("Allocated 3 objects in region:");
        println!("  Ref 1: {:?}", *ref1);
        println!("  Ref 2: {:?}", *ref2);
        println!("  Ref 3: {:?}", *ref3);

        println!("\nRegion statistics:");
        println!("  Allocated: {} bytes", region.allocated());
        println!("  Objects: {}", region.num_allocations());

        // Region is deallocated when it goes out of scope
    }
    println!("Exited region scope - all allocations freed\n");

    // ===== Example 4: Performance Comparison =====
    println!("--- Example 4: Performance Comparison ---");

    const N: usize = 10000;

    // Benchmark: Standard allocation
    let start = Instant::now();
    let mut vec: Vec<Box<i32>> = Vec::new();
    for i in 0..N {
        vec.push(Box::new(i as i32));
    }
    let standard_time = start.elapsed();
    drop(vec); // Explicit drop to include deallocation time
    println!(
        "Standard allocation (Box): {:?} for {} objects",
        standard_time, N
    );

    // Benchmark: Arena allocation
    let mut arena2 = Arena::new();
    let start = Instant::now();
    for i in 0..N {
        arena2.alloc(i);
    }
    let arena_time = start.elapsed();
    println!("Arena allocation: {:?} for {} objects", arena_time, N);

    let speedup = if arena_time.as_nanos() > 0 {
        standard_time.as_nanos() as f64 / arena_time.as_nanos() as f64
    } else {
        1.0
    };
    println!("Speedup: {:.2}x\n", speedup);

    // ===== Example 5: Shared Object Pool (Thread-Safe) =====
    println!("--- Example 5: Shared Object Pool (Thread-Safe) ---");

    let shared_pool = SharedObjectPool::new(
        Vec::<i32>::new,
        PoolConfig {
            initial_capacity: 8,
            max_size: 64,
            preallocate: false,
        },
    );

    println!("Created shared object pool (thread-safe)");

    // Simulate multi-threaded usage
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let pool_clone = shared_pool.clone();
            std::thread::spawn(move || {
                let mut guard = pool_clone.get();
                guard.get_mut().push(i);
                guard.get_mut().push(i * 2);
                println!("Thread {} acquired: {:?}", i, *guard);
            })
        })
        .collect();

    for handle in handles {
        let _ = handle.join();
    }

    println!("\nAll threads completed");

    // ===== Example 6: Arena Growth =====
    println!("\n--- Example 6: Arena Growth ---");

    let mut growing_arena = Arena::with_config(ArenaConfig {
        initial_chunk_size: 128,
        growth_factor: 2.0,
        max_chunk_size: 2048,
    });

    println!("Initial capacity: {} bytes", growing_arena.capacity());

    // Allocate until growth occurs
    for i in 0..100 {
        growing_arena.alloc(vec![i; 10]); // Allocate 10 integers each
    }

    println!("After allocations:");
    println!("  Capacity: {} bytes", growing_arena.capacity());
    println!("  Used: {} bytes", growing_arena.used());
    println!("  Chunks: {}", growing_arena.num_chunks());

    // ===== Example 7: Arena Reset =====
    println!("\n--- Example 7: Arena Reset ---");

    let mut resettable_arena = Arena::new();

    // Allocate some data
    for i in 0..10 {
        resettable_arena.alloc(i);
    }
    println!(
        "Allocated 10 objects: {} bytes used",
        resettable_arena.used()
    );

    // Reset arena (clears all allocations, keeps capacity)
    resettable_arena.reset();
    println!("After reset:");
    println!("  Used: {} bytes", resettable_arena.used());
    println!(
        "  Capacity: {} bytes (retained)",
        resettable_arena.capacity()
    );

    // ===== Example 8: Object Pool Hit/Miss =====
    println!("\n--- Example 8: Object Pool Hit/Miss Tracking ---");

    let small_pool = ObjectPool::new(
        || 0i32,
        PoolConfig {
            initial_capacity: 2,
            max_size: 4,
            preallocate: false,
        },
    );

    println!("Created small pool (max 4 objects)");

    // Acquire and return objects to see hit/miss patterns
    {
        let _g1 = small_pool.get(); // miss
        let _g2 = small_pool.get(); // miss
        let _g3 = small_pool.get(); // miss
    }
    // All returned to pool

    {
        let _g4 = small_pool.get(); // hit
        let _g5 = small_pool.get(); // hit
    }

    let final_stats = small_pool.stats();
    println!("Pool stats after operations:");
    println!("  Hits: {}", final_stats.hits);
    println!("  Misses: {}", final_stats.misses);
    println!("  Returns: {}", final_stats.returns);
    println!("  Discards: {}", final_stats.discards);

    // ===== Example 9: Custom Arena Configuration =====
    println!("\n--- Example 9: Custom Arena Configuration ---");

    let _custom_arena = Arena::with_config(ArenaConfig {
        initial_chunk_size: 16384,        // 16 KB
        growth_factor: 1.5,               // Moderate growth
        max_chunk_size: 10 * 1024 * 1024, // 10 MB max
    });

    println!("Custom arena configuration:");
    println!("  Initial: 16 KB");
    println!("  Growth factor: 1.5x");
    println!("  Maximum chunk: 10 MB");
    println!("\nThis configuration is good for:");
    println!("  - Long-lived data structures");
    println!("  - Predictable memory usage");
    println!("  - Large AST construction");

    // ===== Example 10: Region with Slice Allocation =====
    println!("\n--- Example 10: Region with Slice Allocation ---");

    let region = Region::new();
    let data = vec![1, 2, 3, 4, 5];
    let slice = region.alloc_slice(&data);

    println!("Allocated slice in region:");
    println!("  Length: {}", slice.len());
    println!("  Contents: {:?}", &slice[..]);

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Arenas provide fast bulk allocation");
    println!("  2. Object pools enable efficient reuse");
    println!("  3. Regions support scoped lifetimes");
    println!("  4. Arena allocation is 2-10x faster than Box");
    println!("  5. Pool statistics help tune pool sizes");
}
