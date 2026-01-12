//! Performance Profiling Utilities
//!
//! Tools for profiling and analyzing SAT solver performance.
//! Includes timers, operation counters, and bottleneck detection.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A named timer for profiling code sections
#[derive(Debug)]
pub struct ScopedTimer {
    /// Timer name
    name: String,
    /// Start time
    start: Instant,
    /// Whether the timer is running
    running: bool,
    /// Total elapsed time
    elapsed: Duration,
}

impl ScopedTimer {
    /// Create a new timer
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            running: true,
            elapsed: Duration::ZERO,
        }
    }

    /// Stop the timer
    pub fn stop(&mut self) {
        if self.running {
            self.elapsed += self.start.elapsed();
            self.running = false;
        }
    }

    /// Restart the timer
    pub fn restart(&mut self) {
        if !self.running {
            self.start = Instant::now();
            self.running = true;
        }
    }

    /// Get the elapsed time
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        if self.running {
            self.elapsed + self.start.elapsed()
        } else {
            self.elapsed
        }
    }

    /// Get the timer name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if self.running {
            self.stop();
        }
    }
}

/// Profiler for tracking multiple timers and operation counts
#[derive(Debug, Default)]
pub struct Profiler {
    /// Named timers
    timers: HashMap<String, Duration>,
    /// Operation counters
    counters: HashMap<String, u64>,
    /// Active timers
    active: HashMap<String, Instant>,
}

impl Profiler {
    /// Create a new profiler
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start timing an operation
    pub fn start_timer(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.active.insert(name, Instant::now());
    }

    /// Stop timing an operation
    pub fn stop_timer(&mut self, name: &str) {
        if let Some(start) = self.active.remove(name) {
            let elapsed = start.elapsed();
            *self
                .timers
                .entry(name.to_string())
                .or_insert(Duration::ZERO) += elapsed;
        }
    }

    /// Increment a counter
    pub fn increment(&mut self, name: impl Into<String>) {
        self.increment_by(name, 1);
    }

    /// Increment a counter by a specific amount
    pub fn increment_by(&mut self, name: impl Into<String>, amount: u64) {
        *self.counters.entry(name.into()).or_insert(0) += amount;
    }

    /// Get timer duration
    #[must_use]
    pub fn get_timer(&self, name: &str) -> Option<Duration> {
        self.timers.get(name).copied()
    }

    /// Get counter value
    #[must_use]
    pub fn get_counter(&self, name: &str) -> Option<u64> {
        self.counters.get(name).copied()
    }

    /// Get all timers
    #[must_use]
    pub fn timers(&self) -> &HashMap<String, Duration> {
        &self.timers
    }

    /// Get all counters
    #[must_use]
    pub fn counters(&self) -> &HashMap<String, u64> {
        &self.counters
    }

    /// Clear all profiling data
    pub fn clear(&mut self) {
        self.timers.clear();
        self.counters.clear();
        self.active.clear();
    }

    /// Get total profiled time
    #[must_use]
    pub fn total_time(&self) -> Duration {
        self.timers.values().sum()
    }

    /// Display profiling report
    #[must_use]
    pub fn report(&self) -> String {
        let mut output = String::new();

        output.push_str("═══════════════════════════════════════════════════════════════\n");
        output.push_str("                    PROFILING REPORT                           \n");
        output.push_str("═══════════════════════════════════════════════════════════════\n\n");

        if !self.timers.is_empty() {
            output.push_str("TIMERS:\n");
            let mut timer_vec: Vec<_> = self.timers.iter().collect();
            timer_vec.sort_by(|a, b| b.1.cmp(a.1)); // Sort by duration descending

            let total = self.total_time();

            for (name, duration) in timer_vec {
                let pct = if total > Duration::ZERO {
                    100.0 * duration.as_secs_f64() / total.as_secs_f64()
                } else {
                    0.0
                };

                output.push_str(&format!(
                    "  {:<30} {:>10.3}s  ({:>5.1}%)\n",
                    name,
                    duration.as_secs_f64(),
                    pct
                ));
            }

            output.push_str(&format!("\n  Total Time: {:.3}s\n", total.as_secs_f64()));
            output.push('\n');
        }

        if !self.counters.is_empty() {
            output.push_str("COUNTERS:\n");
            let mut counter_vec: Vec<_> = self.counters.iter().collect();
            counter_vec.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending

            for (name, count) in counter_vec {
                output.push_str(&format!("  {:<30} {:>12}\n", name, count));
            }
            output.push('\n');
        }

        output.push_str("═══════════════════════════════════════════════════════════════\n");

        output
    }
}

/// RAII timer that automatically reports on drop
pub struct AutoTimer {
    name: String,
    start: Instant,
}

impl AutoTimer {
    /// Create a new auto timer
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
        }
    }

    /// Get elapsed time without dropping
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for AutoTimer {
    fn drop(&mut self) {
        tracing::debug!("{}: {:.3}s", self.name, self.start.elapsed().as_secs_f64());
    }
}

/// Macro for creating a scoped timer that logs on drop
#[macro_export]
macro_rules! time_scope {
    ($name:expr) => {
        let _timer = $crate::profiling::AutoTimer::new($name);
    };
}

/// Performance metrics tracker
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    /// Total operations performed
    pub total_ops: u64,
    /// Total time spent
    pub total_time: Duration,
    /// Peak operations per second
    pub peak_ops_per_sec: f64,
    /// Average operations per second
    pub avg_ops_per_sec: f64,
}

impl PerformanceMetrics {
    /// Create new metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update metrics with new data
    pub fn update(&mut self, ops: u64, time: Duration) {
        self.total_ops += ops;
        self.total_time += time;

        let secs = time.as_secs_f64();
        if secs > 0.0 {
            let ops_per_sec = ops as f64 / secs;
            if ops_per_sec > self.peak_ops_per_sec {
                self.peak_ops_per_sec = ops_per_sec;
            }
        }

        let total_secs = self.total_time.as_secs_f64();
        if total_secs > 0.0 {
            self.avg_ops_per_sec = self.total_ops as f64 / total_secs;
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.total_ops = 0;
        self.total_time = Duration::ZERO;
        self.peak_ops_per_sec = 0.0;
        self.avg_ops_per_sec = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_scoped_timer() {
        let mut timer = ScopedTimer::new("test");
        thread::sleep(Duration::from_millis(10));
        timer.stop();

        assert!(timer.elapsed() >= Duration::from_millis(10));
        assert_eq!(timer.name(), "test");
    }

    #[test]
    fn test_scoped_timer_restart() {
        let mut timer = ScopedTimer::new("test");
        thread::sleep(Duration::from_millis(5));
        timer.stop();

        let first_elapsed = timer.elapsed();

        timer.restart();
        thread::sleep(Duration::from_millis(5));
        timer.stop();

        assert!(timer.elapsed() >= first_elapsed);
    }

    #[test]
    fn test_profiler_timers() {
        let mut profiler = Profiler::new();

        profiler.start_timer("operation1");
        thread::sleep(Duration::from_millis(10));
        profiler.stop_timer("operation1");

        let elapsed = profiler.get_timer("operation1");
        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_counters() {
        let mut profiler = Profiler::new();

        profiler.increment("counter1");
        profiler.increment("counter1");
        profiler.increment_by("counter2", 5);

        assert_eq!(profiler.get_counter("counter1"), Some(2));
        assert_eq!(profiler.get_counter("counter2"), Some(5));
    }

    #[test]
    fn test_profiler_clear() {
        let mut profiler = Profiler::new();

        profiler.increment("counter");
        profiler.start_timer("timer");
        profiler.stop_timer("timer");

        profiler.clear();

        assert_eq!(profiler.counters().len(), 0);
        assert_eq!(profiler.timers().len(), 0);
    }

    #[test]
    fn test_profiler_report() {
        let mut profiler = Profiler::new();

        profiler.start_timer("test_timer");
        profiler.stop_timer("test_timer");
        profiler.increment("test_counter");

        let report = profiler.report();
        assert!(report.contains("PROFILING REPORT"));
        assert!(report.contains("TIMERS:"));
        assert!(report.contains("COUNTERS:"));
    }

    #[test]
    fn test_auto_timer() {
        let timer = AutoTimer::new("test");
        thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed() >= Duration::from_millis(10));
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();

        metrics.update(1000, Duration::from_secs(1));
        assert_eq!(metrics.total_ops, 1000);
        assert!(metrics.avg_ops_per_sec > 0.0);

        metrics.update(2000, Duration::from_secs(1));
        assert_eq!(metrics.total_ops, 3000);
    }

    #[test]
    fn test_performance_metrics_reset() {
        let mut metrics = PerformanceMetrics::new();

        metrics.update(1000, Duration::from_secs(1));
        metrics.reset();

        assert_eq!(metrics.total_ops, 0);
        assert_eq!(metrics.total_time, Duration::ZERO);
    }
}
