//! Statistics tracking for solver performance metrics
//!
//! This module provides structures for collecting and reporting
//! various solver statistics like decisions, propagations, conflicts, etc.

use std::time::{Duration, Instant};

/// Statistics for solver performance tracking
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Number of decisions made
    pub decisions: u64,
    /// Number of propagations performed
    pub propagations: u64,
    /// Number of conflicts encountered
    pub conflicts: u64,
    /// Number of restarts performed
    pub restarts: u64,
    /// Number of learned clauses
    pub learned_clauses: u64,
    /// Number of deleted clauses
    pub deleted_clauses: u64,
    /// Total solving time
    pub solve_time: Duration,
    /// Time spent in preprocessing
    pub preprocess_time: Duration,
    /// Time spent in simplification
    pub simplify_time: Duration,
    /// Number of memory allocations (in bytes)
    pub memory_used: u64,
    /// Peak memory usage (in bytes)
    pub peak_memory: u64,
}

impl Statistics {
    /// Create a new statistics tracker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics to zero
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Increment the decision counter
    pub fn inc_decisions(&mut self) {
        self.decisions += 1;
    }

    /// Increment the propagation counter
    pub fn inc_propagations(&mut self) {
        self.propagations += 1;
    }

    /// Increment propagations by a given amount
    pub fn add_propagations(&mut self, count: u64) {
        self.propagations += count;
    }

    /// Increment the conflict counter
    pub fn inc_conflicts(&mut self) {
        self.conflicts += 1;
    }

    /// Increment the restart counter
    pub fn inc_restarts(&mut self) {
        self.restarts += 1;
    }

    /// Increment the learned clause counter
    pub fn inc_learned_clauses(&mut self) {
        self.learned_clauses += 1;
    }

    /// Add to learned clauses count
    pub fn add_learned_clauses(&mut self, count: u64) {
        self.learned_clauses += count;
    }

    /// Increment the deleted clause counter
    pub fn inc_deleted_clauses(&mut self) {
        self.deleted_clauses += 1;
    }

    /// Add to deleted clauses count
    pub fn add_deleted_clauses(&mut self, count: u64) {
        self.deleted_clauses += count;
    }

    /// Add time to solve time
    pub fn add_solve_time(&mut self, duration: Duration) {
        self.solve_time += duration;
    }

    /// Add time to preprocess time
    pub fn add_preprocess_time(&mut self, duration: Duration) {
        self.preprocess_time += duration;
    }

    /// Add time to simplify time
    pub fn add_simplify_time(&mut self, duration: Duration) {
        self.simplify_time += duration;
    }

    /// Update memory usage
    pub fn set_memory_used(&mut self, bytes: u64) {
        self.memory_used = bytes;
        if bytes > self.peak_memory {
            self.peak_memory = bytes;
        }
    }

    /// Get conflicts per second
    #[must_use]
    pub fn conflicts_per_second(&self) -> f64 {
        let secs = self.solve_time.as_secs_f64();
        if secs > 0.0 {
            self.conflicts as f64 / secs
        } else {
            0.0
        }
    }

    /// Get decisions per second
    #[must_use]
    pub fn decisions_per_second(&self) -> f64 {
        let secs = self.solve_time.as_secs_f64();
        if secs > 0.0 {
            self.decisions as f64 / secs
        } else {
            0.0
        }
    }

    /// Get propagations per second
    #[must_use]
    pub fn propagations_per_second(&self) -> f64 {
        let secs = self.solve_time.as_secs_f64();
        if secs > 0.0 {
            self.propagations as f64 / secs
        } else {
            0.0
        }
    }

    /// Format memory size in human-readable form
    #[must_use]
    pub fn format_memory(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Print statistics in a human-readable format
    pub fn print(&self) {
        println!("Statistics:");
        println!("  Decisions:       {}", self.decisions);
        println!("  Propagations:    {}", self.propagations);
        println!("  Conflicts:       {}", self.conflicts);
        println!("  Restarts:        {}", self.restarts);
        println!("  Learned clauses: {}", self.learned_clauses);
        println!("  Deleted clauses: {}", self.deleted_clauses);
        println!("  Solve time:      {:.3}s", self.solve_time.as_secs_f64());
        println!(
            "  Preprocess time: {:.3}s",
            self.preprocess_time.as_secs_f64()
        );
        println!(
            "  Simplify time:   {:.3}s",
            self.simplify_time.as_secs_f64()
        );
        println!(
            "  Memory used:     {}",
            Self::format_memory(self.memory_used)
        );
        println!(
            "  Peak memory:     {}",
            Self::format_memory(self.peak_memory)
        );
        println!("\nPerformance:");
        println!("  Decisions/sec:    {:.0}", self.decisions_per_second());
        println!("  Propagations/sec: {:.0}", self.propagations_per_second());
        println!("  Conflicts/sec:    {:.0}", self.conflicts_per_second());
    }
}

impl std::fmt::Display for Statistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Statistics:")?;
        writeln!(f, "  Decisions:       {}", self.decisions)?;
        writeln!(f, "  Propagations:    {}", self.propagations)?;
        writeln!(f, "  Conflicts:       {}", self.conflicts)?;
        writeln!(f, "  Restarts:        {}", self.restarts)?;
        writeln!(f, "  Learned clauses: {}", self.learned_clauses)?;
        writeln!(f, "  Deleted clauses: {}", self.deleted_clauses)?;
        writeln!(
            f,
            "  Solve time:      {:.3}s",
            self.solve_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  Preprocess time: {:.3}s",
            self.preprocess_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  Simplify time:   {:.3}s",
            self.simplify_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  Memory used:     {}",
            Self::format_memory(self.memory_used)
        )?;
        writeln!(
            f,
            "  Peak memory:     {}",
            Self::format_memory(self.peak_memory)
        )?;
        Ok(())
    }
}

/// A timer for measuring code execution time
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get the elapsed time since the timer started
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return the elapsed time
    #[must_use]
    pub fn stop(self) -> Duration {
        self.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_empty_statistics() {
        let stats = Statistics::new();
        assert_eq!(stats.decisions, 0);
        assert_eq!(stats.propagations, 0);
        assert_eq!(stats.conflicts, 0);
        assert_eq!(stats.restarts, 0);
        assert_eq!(stats.learned_clauses, 0);
        assert_eq!(stats.deleted_clauses, 0);
    }

    #[test]
    fn test_increment_counters() {
        let mut stats = Statistics::new();

        stats.inc_decisions();
        assert_eq!(stats.decisions, 1);

        stats.inc_propagations();
        stats.inc_propagations();
        assert_eq!(stats.propagations, 2);

        stats.inc_conflicts();
        assert_eq!(stats.conflicts, 1);

        stats.inc_restarts();
        assert_eq!(stats.restarts, 1);

        stats.inc_learned_clauses();
        assert_eq!(stats.learned_clauses, 1);

        stats.inc_deleted_clauses();
        assert_eq!(stats.deleted_clauses, 1);
    }

    #[test]
    fn test_add_counters() {
        let mut stats = Statistics::new();

        stats.add_propagations(100);
        assert_eq!(stats.propagations, 100);

        stats.add_learned_clauses(50);
        assert_eq!(stats.learned_clauses, 50);

        stats.add_deleted_clauses(25);
        assert_eq!(stats.deleted_clauses, 25);
    }

    #[test]
    fn test_time_tracking() {
        let mut stats = Statistics::new();

        stats.add_solve_time(Duration::from_secs(1));
        stats.add_preprocess_time(Duration::from_millis(500));
        stats.add_simplify_time(Duration::from_millis(250));

        assert_eq!(stats.solve_time.as_secs(), 1);
        assert_eq!(stats.preprocess_time.as_millis(), 500);
        assert_eq!(stats.simplify_time.as_millis(), 250);
    }

    #[test]
    fn test_memory_tracking() {
        let mut stats = Statistics::new();

        stats.set_memory_used(1024);
        assert_eq!(stats.memory_used, 1024);
        assert_eq!(stats.peak_memory, 1024);

        stats.set_memory_used(2048);
        assert_eq!(stats.memory_used, 2048);
        assert_eq!(stats.peak_memory, 2048);

        // Peak should remain at 2048 even if current usage decreases
        stats.set_memory_used(512);
        assert_eq!(stats.memory_used, 512);
        assert_eq!(stats.peak_memory, 2048);
    }

    #[test]
    fn test_performance_metrics() {
        let mut stats = Statistics::new();

        stats.decisions = 1000;
        stats.propagations = 10000;
        stats.conflicts = 100;
        stats.solve_time = Duration::from_secs(1);

        assert_eq!(stats.decisions_per_second(), 1000.0);
        assert_eq!(stats.propagations_per_second(), 10000.0);
        assert_eq!(stats.conflicts_per_second(), 100.0);
    }

    #[test]
    fn test_format_memory() {
        assert_eq!(Statistics::format_memory(512), "512 B");
        assert_eq!(Statistics::format_memory(1024), "1.00 KB");
        assert_eq!(Statistics::format_memory(1024 * 1024), "1.00 MB");
        assert_eq!(Statistics::format_memory(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(Statistics::format_memory(1536 * 1024), "1.50 MB");
    }

    #[test]
    fn test_reset() {
        let mut stats = Statistics::new();

        stats.inc_decisions();
        stats.inc_propagations();
        stats.inc_conflicts();
        stats.add_solve_time(Duration::from_secs(1));

        stats.reset();

        assert_eq!(stats.decisions, 0);
        assert_eq!(stats.propagations, 0);
        assert_eq!(stats.conflicts, 0);
        assert_eq!(stats.solve_time, Duration::ZERO);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed();

        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_timer_stop() {
        let timer = Timer::start();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_display() {
        let mut stats = Statistics::new();
        stats.decisions = 100;
        stats.propagations = 1000;

        let display = format!("{}", stats);
        assert!(display.contains("Decisions:"));
        assert!(display.contains("100"));
        assert!(display.contains("Propagations:"));
        assert!(display.contains("1000"));
    }
}
