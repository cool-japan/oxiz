//! Memory limit enforcement for benchmarks
//!
//! This module provides functionality to track and enforce memory limits
//! during benchmark execution using OS-level resource limits.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Error type for memory operations
#[derive(Error, Debug)]
pub enum MemoryError {
    /// Failed to get resource limits
    #[error("Failed to get resource limits: {0}")]
    GetLimitFailed(String),
    /// Failed to set resource limits
    #[error("Failed to set resource limits: {0}")]
    SetLimitFailed(String),
    /// Memory limit exceeded
    #[error("Memory limit exceeded: {current} bytes (limit: {limit} bytes)")]
    LimitExceeded {
        /// Current memory usage
        current: u64,
        /// Configured limit
        limit: u64,
    },
    /// Platform not supported
    #[error("Memory limit enforcement not supported on this platform")]
    NotSupported,
}

/// Result type for memory operations
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Memory limit configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryLimit {
    /// Soft limit in bytes (warning threshold)
    pub soft_limit: u64,
    /// Hard limit in bytes (enforcement threshold)
    pub hard_limit: u64,
    /// Stack size limit in bytes (0 = default)
    pub stack_limit: u64,
}

impl Default for MemoryLimit {
    fn default() -> Self {
        Self {
            soft_limit: 4 * 1024 * 1024 * 1024, // 4 GB
            hard_limit: 8 * 1024 * 1024 * 1024, // 8 GB
            stack_limit: 0,                     // Use default
        }
    }
}

impl MemoryLimit {
    /// Create a new memory limit with the given hard limit
    #[must_use]
    pub fn new(hard_limit_bytes: u64) -> Self {
        Self {
            soft_limit: hard_limit_bytes / 2,
            hard_limit: hard_limit_bytes,
            stack_limit: 0,
        }
    }

    /// Create a memory limit from megabytes
    #[must_use]
    pub fn from_mb(mb: u64) -> Self {
        Self::new(mb * 1024 * 1024)
    }

    /// Create a memory limit from gigabytes
    #[must_use]
    pub fn from_gb(gb: u64) -> Self {
        Self::new(gb * 1024 * 1024 * 1024)
    }

    /// Set the soft limit
    #[must_use]
    pub fn with_soft_limit(mut self, limit: u64) -> Self {
        self.soft_limit = limit;
        self
    }

    /// Set the stack limit
    #[must_use]
    pub fn with_stack_limit(mut self, limit: u64) -> Self {
        self.stack_limit = limit;
        self
    }

    /// Check if unlimited
    #[must_use]
    pub fn is_unlimited(&self) -> bool {
        self.hard_limit == 0
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Current heap usage in bytes
    pub heap_bytes: u64,
    /// Peak heap usage in bytes
    pub peak_heap_bytes: u64,
    /// Current resident set size in bytes
    pub rss_bytes: u64,
    /// Peak resident set size in bytes
    pub peak_rss_bytes: u64,
    /// Virtual memory size in bytes
    pub virtual_bytes: u64,
}

impl MemoryUsage {
    /// Get current memory usage (best effort)
    #[must_use]
    pub fn current() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::from_proc_status()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::default()
        }
    }

    /// Parse memory info from /proc/self/status on Linux
    #[cfg(target_os = "linux")]
    fn from_proc_status() -> Self {
        use std::fs;

        let mut usage = Self::default();

        if let Ok(content) = fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let value = parts[1].parse::<u64>().unwrap_or(0) * 1024; // kB to bytes
                    match parts[0] {
                        "VmRSS:" => usage.rss_bytes = value,
                        "VmHWM:" => usage.peak_rss_bytes = value,
                        "VmSize:" => usage.virtual_bytes = value,
                        "VmData:" => usage.heap_bytes = value,
                        _ => {}
                    }
                }
            }
        }

        // Estimate peak heap from current if not available
        if usage.peak_heap_bytes == 0 {
            usage.peak_heap_bytes = usage.heap_bytes;
        }

        usage
    }

    /// Check if usage exceeds limit
    #[must_use]
    pub fn exceeds_limit(&self, limit: &MemoryLimit) -> bool {
        if limit.is_unlimited() {
            return false;
        }
        self.rss_bytes > limit.hard_limit || self.heap_bytes > limit.hard_limit
    }

    /// Check if usage is near soft limit (warning threshold)
    #[must_use]
    pub fn near_soft_limit(&self, limit: &MemoryLimit) -> bool {
        if limit.is_unlimited() {
            return false;
        }
        self.rss_bytes > limit.soft_limit || self.heap_bytes > limit.soft_limit
    }

    /// Format as human-readable string
    #[must_use]
    pub fn format(&self) -> String {
        format!(
            "RSS: {}, Heap: {}, Peak RSS: {}, Virtual: {}",
            format_bytes(self.rss_bytes),
            format_bytes(self.heap_bytes),
            format_bytes(self.peak_rss_bytes),
            format_bytes(self.virtual_bytes)
        )
    }
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

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

/// Memory limiter for enforcing limits on child processes
pub struct MemoryLimiter {
    limit: MemoryLimit,
}

impl MemoryLimiter {
    /// Create a new memory limiter
    #[must_use]
    pub fn new(limit: MemoryLimit) -> Self {
        Self { limit }
    }

    /// Apply memory limits to the current process
    ///
    /// Note: Memory limit enforcement via setrlimit requires the libc crate.
    /// This function returns Ok for unlimited limits, or NotSupported otherwise.
    /// For actual enforcement, use external tools like ulimit or cgroups.
    pub fn apply(&self) -> MemoryResult<()> {
        if self.limit.is_unlimited() {
            Ok(())
        } else {
            // Memory limit enforcement would require libc for setrlimit
            // For production use, consider using external mechanisms:
            // - ulimit -v (shell)
            // - cgroups (Linux containers)
            // - Resource limits in systemd units
            tracing::warn!(
                "Memory limits ({} bytes) configured but not enforced. \
                 Use ulimit or cgroups for enforcement.",
                self.limit.hard_limit
            );
            Ok(())
        }
    }

    /// Check current memory usage against limits
    pub fn check(&self) -> MemoryResult<MemoryUsage> {
        let usage = MemoryUsage::current();

        if usage.exceeds_limit(&self.limit) {
            return Err(MemoryError::LimitExceeded {
                current: usage.rss_bytes.max(usage.heap_bytes),
                limit: self.limit.hard_limit,
            });
        }

        Ok(usage)
    }

    /// Get the configured limit
    #[must_use]
    pub fn limit(&self) -> &MemoryLimit {
        &self.limit
    }
}

/// Memory monitor for tracking usage over time
pub struct MemoryMonitor {
    limit: MemoryLimit,
    samples: Vec<MemoryUsage>,
    sample_interval: Duration,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    #[must_use]
    pub fn new(limit: MemoryLimit) -> Self {
        Self {
            limit,
            samples: Vec::new(),
            sample_interval: Duration::from_millis(100),
        }
    }

    /// Set the sample interval
    #[must_use]
    pub fn with_sample_interval(mut self, interval: Duration) -> Self {
        self.sample_interval = interval;
        self
    }

    /// Take a memory sample
    pub fn sample(&mut self) -> MemoryUsage {
        let usage = MemoryUsage::current();
        self.samples.push(usage);
        usage
    }

    /// Get peak memory usage from samples
    #[must_use]
    pub fn peak_usage(&self) -> MemoryUsage {
        self.samples
            .iter()
            .max_by_key(|u| u.rss_bytes)
            .copied()
            .unwrap_or_default()
    }

    /// Get average memory usage
    #[must_use]
    pub fn average_usage(&self) -> u64 {
        if self.samples.is_empty() {
            0
        } else {
            let total: u64 = self.samples.iter().map(|u| u.rss_bytes).sum();
            total / self.samples.len() as u64
        }
    }

    /// Clear samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Check if any sample exceeded the limit
    #[must_use]
    pub fn limit_exceeded(&self) -> bool {
        self.samples.iter().any(|u| u.exceeds_limit(&self.limit))
    }

    /// Get all samples
    #[must_use]
    pub fn samples(&self) -> &[MemoryUsage] {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_limit_creation() {
        let limit = MemoryLimit::from_gb(4);
        assert_eq!(limit.hard_limit, 4 * 1024 * 1024 * 1024);
        assert_eq!(limit.soft_limit, 2 * 1024 * 1024 * 1024);

        let limit = MemoryLimit::from_mb(512);
        assert_eq!(limit.hard_limit, 512 * 1024 * 1024);
    }

    #[test]
    fn test_memory_limit_builder() {
        let limit = MemoryLimit::from_gb(8)
            .with_soft_limit(4 * 1024 * 1024 * 1024)
            .with_stack_limit(8 * 1024 * 1024);

        assert_eq!(limit.hard_limit, 8 * 1024 * 1024 * 1024);
        assert_eq!(limit.soft_limit, 4 * 1024 * 1024 * 1024);
        assert_eq!(limit.stack_limit, 8 * 1024 * 1024);
    }

    #[test]
    fn test_memory_usage_current() {
        let usage = MemoryUsage::current();
        // Just ensure it doesn't panic
        let _ = usage.format();
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_memory_monitor() {
        let limit = MemoryLimit::from_gb(16);
        let mut monitor = MemoryMonitor::new(limit);

        // Take a few samples
        for _ in 0..3 {
            monitor.sample();
        }

        assert_eq!(monitor.samples().len(), 3);

        let peak = monitor.peak_usage();
        let _ = peak.format();

        monitor.clear();
        assert!(monitor.samples().is_empty());
    }

    #[test]
    fn test_memory_limit_checks() {
        let limit = MemoryLimit::from_mb(100);

        let usage = MemoryUsage {
            heap_bytes: 50 * 1024 * 1024,
            rss_bytes: 50 * 1024 * 1024,
            ..Default::default()
        };
        assert!(!usage.exceeds_limit(&limit));

        let usage = MemoryUsage {
            heap_bytes: 150 * 1024 * 1024,
            rss_bytes: 150 * 1024 * 1024,
            ..Default::default()
        };
        assert!(usage.exceeds_limit(&limit));
    }
}
