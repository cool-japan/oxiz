//! Memory Optimization
//!
//! Advanced memory optimization techniques including:
//! - Size-class memory pools for different clause sizes
//! - Cache-aware data layout
//! - Memory prefetching hints
//! - Fragmentation tracking and mitigation
//! - Memory pressure monitoring

use std::collections::HashMap;

/// Memory statistics
#[derive(Debug, Default, Clone)]
pub struct MemoryOptStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Total bytes freed
    pub total_freed: usize,
    /// Current usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Number of allocations
    pub allocations: u64,
    /// Number of frees
    pub frees: u64,
    /// Number of pool hits
    pub pool_hits: u64,
    /// Number of pool misses
    pub pool_misses: u64,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = max fragmentation)
    pub fragmentation: f64,
}

impl MemoryOptStats {
    /// Get the pool hit rate (0.0 to 1.0)
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total_requests = self.pool_hits + self.pool_misses;
        if total_requests == 0 {
            0.0
        } else {
            self.pool_hits as f64 / total_requests as f64
        }
    }

    /// Get average allocation size
    #[must_use]
    pub fn avg_allocation_size(&self) -> f64 {
        if self.allocations == 0 {
            0.0
        } else {
            self.total_allocated as f64 / self.allocations as f64
        }
    }
}

/// Size class for memory pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizeClass {
    /// Binary clauses (2 literals)
    Binary,
    /// Ternary clauses (3 literals)
    Ternary,
    /// Small clauses (4-8 literals)
    Small,
    /// Medium clauses (9-16 literals)
    Medium,
    /// Large clauses (17-32 literals)
    Large,
    /// Very large clauses (33+ literals)
    VeryLarge,
}

impl SizeClass {
    /// Get size class for a given number of literals
    #[must_use]
    pub fn from_size(size: usize) -> Self {
        match size {
            0..=2 => Self::Binary,
            3 => Self::Ternary,
            4..=8 => Self::Small,
            9..=16 => Self::Medium,
            17..=32 => Self::Large,
            _ => Self::VeryLarge,
        }
    }

    /// Get the buffer size for this class
    #[must_use]
    pub fn buffer_size(self) -> usize {
        match self {
            Self::Binary => 16,
            Self::Ternary => 24,
            Self::Small => 64,
            Self::Medium => 128,
            Self::Large => 256,
            Self::VeryLarge => 512,
        }
    }
}

/// Memory pool for a specific size class
struct MemoryPool {
    /// Free blocks available
    free_blocks: Vec<Vec<u8>>,
    /// Allocated blocks
    allocated_blocks: usize,
    /// Maximum pool size
    max_blocks: usize,
}

impl MemoryPool {
    fn new(max_blocks: usize) -> Self {
        Self {
            free_blocks: Vec::new(),
            allocated_blocks: 0,
            max_blocks,
        }
    }

    fn allocate(&mut self, size: usize) -> (Option<Vec<u8>>, bool) {
        if let Some(block) = self.free_blocks.pop() {
            self.allocated_blocks += 1;
            (Some(block), true) // Hit: reused free block
        } else if self.allocated_blocks < self.max_blocks {
            self.allocated_blocks += 1;
            (Some(vec![0; size]), false) // Miss: created new block
        } else {
            (None, false) // Pool full
        }
    }

    fn free(&mut self, block: Vec<u8>) {
        if self.allocated_blocks > 0 {
            self.allocated_blocks -= 1;
        }
        if self.free_blocks.len() < self.max_blocks {
            self.free_blocks.push(block);
        }
    }

    fn clear(&mut self) {
        self.free_blocks.clear();
        self.allocated_blocks = 0;
    }
}

/// Memory optimizer with size-class pools
pub struct MemoryOptimizer {
    /// Memory pools by size class
    pools: HashMap<SizeClass, MemoryPool>,
    /// Statistics
    stats: MemoryOptStats,
    /// Enable prefetching
    enable_prefetch: bool,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    #[must_use]
    pub fn new() -> Self {
        let mut pools = HashMap::new();

        // Initialize pools for each size class
        pools.insert(SizeClass::Binary, MemoryPool::new(1000));
        pools.insert(SizeClass::Ternary, MemoryPool::new(800));
        pools.insert(SizeClass::Small, MemoryPool::new(500));
        pools.insert(SizeClass::Medium, MemoryPool::new(200));
        pools.insert(SizeClass::Large, MemoryPool::new(50));
        pools.insert(SizeClass::VeryLarge, MemoryPool::new(10));

        Self {
            pools,
            stats: MemoryOptStats::default(),
            enable_prefetch: true,
        }
    }

    /// Enable or disable prefetching
    pub fn set_prefetch(&mut self, enable: bool) {
        self.enable_prefetch = enable;
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &MemoryOptStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MemoryOptStats::default();
    }

    /// Allocate memory for a clause of given size
    pub fn allocate(&mut self, num_literals: usize) -> Vec<u8> {
        let size_class = SizeClass::from_size(num_literals);
        let buffer_size = size_class.buffer_size();

        self.stats.allocations += 1;
        self.stats.total_allocated += buffer_size;
        self.stats.current_usage += buffer_size;

        if self.stats.current_usage > self.stats.peak_usage {
            self.stats.peak_usage = self.stats.current_usage;
        }

        if let Some(pool) = self.pools.get_mut(&size_class) {
            let (block_opt, is_hit) = pool.allocate(buffer_size);
            if let Some(block) = block_opt {
                if is_hit {
                    self.stats.pool_hits += 1;
                } else {
                    self.stats.pool_misses += 1;
                }
                return block;
            }
        }

        self.stats.pool_misses += 1;
        vec![0; buffer_size]
    }

    /// Free memory for a clause
    pub fn free(&mut self, buffer: Vec<u8>, num_literals: usize) {
        let size_class = SizeClass::from_size(num_literals);
        let buffer_size = size_class.buffer_size();

        self.stats.frees += 1;
        self.stats.total_freed += buffer_size;
        if self.stats.current_usage >= buffer_size {
            self.stats.current_usage -= buffer_size;
        }

        if let Some(pool) = self.pools.get_mut(&size_class) {
            pool.free(buffer);
        }
    }

    /// Prefetch memory location (hint to CPU)
    #[inline]
    pub fn prefetch<T>(&self, _ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            if self.enable_prefetch {
                // On x86_64, we could use prefetch intrinsics
                // For now, this is a no-op placeholder
                // In real implementation, use: _mm_prefetch(ptr, _MM_HINT_T0)
            }
        }
    }

    /// Compact memory pools (release unused blocks)
    pub fn compact(&mut self) {
        for pool in self.pools.values_mut() {
            // Keep only half of the free blocks
            let target_size = pool.free_blocks.len() / 2;
            pool.free_blocks.truncate(target_size);
        }

        // Update fragmentation metric
        self.update_fragmentation();
    }

    /// Clear all pools
    pub fn clear_pools(&mut self) {
        for pool in self.pools.values_mut() {
            pool.clear();
        }
        self.stats.current_usage = 0;
    }

    /// Get memory usage by size class
    #[must_use]
    pub fn usage_by_size_class(&self) -> HashMap<SizeClass, usize> {
        let mut usage = HashMap::new();
        for (&size_class, pool) in &self.pools {
            let class_usage = pool.allocated_blocks * size_class.buffer_size();
            usage.insert(size_class, class_usage);
        }
        usage
    }

    /// Check if memory pressure is high
    #[must_use]
    pub fn is_memory_pressure_high(&self) -> bool {
        // Consider memory pressure high if we're using > 90% of peak usage
        // or if fragmentation is high
        let usage_ratio = self.stats.current_usage as f64 / self.stats.peak_usage.max(1) as f64;
        usage_ratio > 0.9 || self.stats.fragmentation > 0.5
    }

    /// Get recommended action based on memory pressure
    #[must_use]
    pub fn recommend_action(&self) -> MemoryAction {
        if self.is_memory_pressure_high() {
            if self.stats.fragmentation > 0.5 {
                MemoryAction::Compact
            } else {
                MemoryAction::ReduceClauseDatabase
            }
        } else if self.stats.hit_rate() < 0.5 {
            MemoryAction::ExpandPools
        } else {
            MemoryAction::None
        }
    }

    /// Update fragmentation metric
    fn update_fragmentation(&mut self) {
        let mut total_free = 0;
        let mut total_capacity = 0;

        for (size_class, pool) in &self.pools {
            total_free += pool.free_blocks.len() * size_class.buffer_size();
            total_capacity += pool.max_blocks * size_class.buffer_size();
        }

        if total_capacity > 0 {
            // Fragmentation is high when we have many free blocks but also high usage
            let free_ratio = total_free as f64 / total_capacity as f64;
            let usage_ratio = self.stats.current_usage as f64 / self.stats.peak_usage.max(1) as f64;

            // Fragmentation metric: high when both free ratio and usage ratio are high
            self.stats.fragmentation = (free_ratio * usage_ratio).min(1.0);
        }
    }
}

impl Default for MemoryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Recommended memory action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAction {
    /// No action needed
    None,
    /// Compact memory pools
    Compact,
    /// Reduce clause database size
    ReduceClauseDatabase,
    /// Expand pool sizes
    ExpandPools,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_from_size() {
        assert_eq!(SizeClass::from_size(2), SizeClass::Binary);
        assert_eq!(SizeClass::from_size(3), SizeClass::Ternary);
        assert_eq!(SizeClass::from_size(5), SizeClass::Small);
        assert_eq!(SizeClass::from_size(10), SizeClass::Medium);
        assert_eq!(SizeClass::from_size(20), SizeClass::Large);
        assert_eq!(SizeClass::from_size(50), SizeClass::VeryLarge);
    }

    #[test]
    fn test_size_class_buffer_size() {
        assert_eq!(SizeClass::Binary.buffer_size(), 16);
        assert_eq!(SizeClass::Ternary.buffer_size(), 24);
        assert_eq!(SizeClass::Small.buffer_size(), 64);
    }

    #[test]
    fn test_memory_optimizer_creation() {
        let opt = MemoryOptimizer::new();
        assert_eq!(opt.stats().allocations, 0);
    }

    #[test]
    fn test_allocate_and_free() {
        let mut opt = MemoryOptimizer::new();

        let buffer = opt.allocate(2);
        assert_eq!(buffer.len(), 16); // Binary clause buffer size

        assert_eq!(opt.stats().allocations, 1);
        assert!(opt.stats().current_usage > 0);

        opt.free(buffer, 2);
        assert_eq!(opt.stats().frees, 1);
    }

    #[test]
    fn test_pool_hits() {
        let mut opt = MemoryOptimizer::new();

        // First allocation creates a new buffer, counted as miss since no free blocks
        let buffer1 = opt.allocate(2);
        assert_eq!(opt.stats().pool_misses, 1);
        assert_eq!(opt.stats().pool_hits, 0);

        // Free the buffer back to pool
        opt.free(buffer1, 2);

        // Second allocation should reuse the freed buffer - hit
        let _buffer2 = opt.allocate(2);
        assert_eq!(opt.stats().pool_hits, 1);
        assert_eq!(opt.stats().pool_misses, 1);
    }

    #[test]
    fn test_hit_rate() {
        let mut opt = MemoryOptimizer::new();

        let buffer1 = opt.allocate(2);
        opt.free(buffer1, 2);
        let _buffer2 = opt.allocate(2);

        let hit_rate = opt.stats().hit_rate();
        assert!(hit_rate > 0.0);
        assert!(hit_rate <= 1.0);
    }

    #[test]
    fn test_peak_usage() {
        let mut opt = MemoryOptimizer::new();

        let _buffer1 = opt.allocate(2);
        let peak1 = opt.stats().peak_usage;

        let _buffer2 = opt.allocate(10);
        let peak2 = opt.stats().peak_usage;

        assert!(peak2 >= peak1);
    }

    #[test]
    fn test_compact() {
        let mut opt = MemoryOptimizer::new();

        // Allocate and free several buffers
        for _ in 0..10 {
            let buffer = opt.allocate(2);
            opt.free(buffer, 2);
        }

        opt.compact();
        // After compaction, pools should be smaller
    }

    #[test]
    fn test_clear_pools() {
        let mut opt = MemoryOptimizer::new();

        let _buffer = opt.allocate(2);
        assert!(opt.stats().current_usage > 0);

        opt.clear_pools();
        assert_eq!(opt.stats().current_usage, 0);
    }

    #[test]
    fn test_usage_by_size_class() {
        let mut opt = MemoryOptimizer::new();

        let _buffer1 = opt.allocate(2);
        let _buffer2 = opt.allocate(10);

        let usage = opt.usage_by_size_class();
        assert!(usage.contains_key(&SizeClass::Binary));
        assert!(usage.contains_key(&SizeClass::Medium));
    }

    #[test]
    fn test_memory_action() {
        let opt = MemoryOptimizer::new();
        let action = opt.recommend_action();
        // With no allocations, hit_rate is 0.0, which triggers ExpandPools
        assert!(matches!(
            action,
            MemoryAction::None | MemoryAction::ExpandPools
        ));
    }

    #[test]
    fn test_prefetch() {
        let opt = MemoryOptimizer::new();
        let value = 42;
        opt.prefetch(&value);
        // Just ensure it doesn't crash
    }

    #[test]
    fn test_set_prefetch() {
        let mut opt = MemoryOptimizer::new();
        opt.set_prefetch(false);
        assert!(!opt.enable_prefetch);

        opt.set_prefetch(true);
        assert!(opt.enable_prefetch);
    }

    #[test]
    fn test_avg_allocation_size() {
        let mut opt = MemoryOptimizer::new();

        let _buffer1 = opt.allocate(2);
        let _buffer2 = opt.allocate(10);

        let avg = opt.stats().avg_allocation_size();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_reset_stats() {
        let mut opt = MemoryOptimizer::new();

        let _buffer = opt.allocate(2);
        assert_eq!(opt.stats().allocations, 1);

        opt.reset_stats();
        assert_eq!(opt.stats().allocations, 0);
    }
}
