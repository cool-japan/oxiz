//! Resource limits and monitoring for the SMT solver.
//!
//! This module provides comprehensive resource limit support including:
//! - Wall-clock timeout
//! - Conflict budget for the SAT solver
//! - Decision budget
//! - Restart budget
//! - Memory limits
//! - Theory check budget
//!
//! # Example
//!
//! ```
//! use oxiz_solver::resource_limits::{ResourceLimits, ResourceMonitor};
//! use core::time::Duration;
//!
//! let limits = ResourceLimits::new()
//!     .with_timeout(Duration::from_secs(30))
//!     .with_max_conflicts(10_000)
//!     .with_max_decisions(100_000);
//!
//! let monitor = ResourceMonitor::new(limits);
//! ```

#[allow(unused_imports)]
use crate::prelude::*;

/// Which resource limit was exhausted
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceExhausted {
    /// Wall-clock timeout was reached
    Timeout,
    /// Maximum number of SAT conflicts was reached
    MaxConflicts,
    /// Maximum memory usage was reached
    MaxMemory,
    /// Maximum number of decisions was reached
    MaxDecisions,
    /// Maximum number of restarts was reached
    MaxRestarts,
    /// Maximum number of theory checks was reached
    MaxTheoryChecks,
}

impl core::fmt::Display for ResourceExhausted {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ResourceExhausted::Timeout => write!(f, "timeout"),
            ResourceExhausted::MaxConflicts => write!(f, "max conflicts reached"),
            ResourceExhausted::MaxMemory => write!(f, "max memory reached"),
            ResourceExhausted::MaxDecisions => write!(f, "max decisions reached"),
            ResourceExhausted::MaxRestarts => write!(f, "max restarts reached"),
            ResourceExhausted::MaxTheoryChecks => write!(f, "max theory checks reached"),
        }
    }
}

/// Configuration for resource limits on solver execution.
///
/// All limits are optional. When a limit is `None`, that resource is unlimited.
#[derive(Debug, Clone, Default)]
pub struct ResourceLimits {
    /// Wall-clock timeout duration
    pub timeout: Option<core::time::Duration>,
    /// Maximum number of SAT conflicts before giving up
    pub max_conflicts: Option<u64>,
    /// Maximum memory in megabytes
    pub max_memory_mb: Option<u64>,
    /// Maximum number of decisions before giving up
    pub max_decisions: Option<u64>,
    /// Maximum number of restarts before giving up
    pub max_restarts: Option<u64>,
    /// Maximum number of theory consistency checks
    pub max_theory_checks: Option<u64>,
}

impl ResourceLimits {
    /// Create a new `ResourceLimits` with no limits set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a wall-clock timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: core::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the maximum number of SAT conflicts.
    #[must_use]
    pub fn with_max_conflicts(mut self, max: u64) -> Self {
        self.max_conflicts = Some(max);
        self
    }

    /// Set the maximum memory in megabytes.
    #[must_use]
    pub fn with_max_memory_mb(mut self, max: u64) -> Self {
        self.max_memory_mb = Some(max);
        self
    }

    /// Set the maximum number of decisions.
    #[must_use]
    pub fn with_max_decisions(mut self, max: u64) -> Self {
        self.max_decisions = Some(max);
        self
    }

    /// Set the maximum number of restarts.
    #[must_use]
    pub fn with_max_restarts(mut self, max: u64) -> Self {
        self.max_restarts = Some(max);
        self
    }

    /// Set the maximum number of theory checks.
    #[must_use]
    pub fn with_max_theory_checks(mut self, max: u64) -> Self {
        self.max_theory_checks = Some(max);
        self
    }

    /// Check if any limits are set.
    #[must_use]
    pub fn has_any_limit(&self) -> bool {
        self.timeout.is_some()
            || self.max_conflicts.is_some()
            || self.max_memory_mb.is_some()
            || self.max_decisions.is_some()
            || self.max_restarts.is_some()
            || self.max_theory_checks.is_some()
    }
}

/// Tracks resource usage and checks whether limits have been exceeded.
///
/// The monitor is created from a [`ResourceLimits`] configuration and then
/// updated as the solver progresses. Call [`check`](ResourceMonitor::check) to
/// see if any limit has been hit.
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// The configured limits
    limits: ResourceLimits,
    /// Number of conflicts observed so far
    pub conflicts: u64,
    /// Number of decisions observed so far
    pub decisions: u64,
    /// Number of restarts observed so far
    pub restarts: u64,
    /// Number of theory checks observed so far
    pub theory_checks: u64,
    /// Start time (only meaningful with std)
    #[cfg(feature = "std")]
    start_time: Option<std::time::Instant>,
}

impl ResourceMonitor {
    /// Create a new monitor with the given limits and reset all counters.
    #[must_use]
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            #[cfg(feature = "std")]
            start_time: if limits.timeout.is_some() {
                Some(std::time::Instant::now())
            } else {
                None
            },
            limits,
            conflicts: 0,
            decisions: 0,
            restarts: 0,
            theory_checks: 0,
        }
    }

    /// Reset counters and restart the timer.
    pub fn reset(&mut self) {
        self.conflicts = 0;
        self.decisions = 0;
        self.restarts = 0;
        self.theory_checks = 0;
        #[cfg(feature = "std")]
        {
            self.start_time = if self.limits.timeout.is_some() {
                Some(std::time::Instant::now())
            } else {
                None
            };
        }
    }

    /// Record one conflict.
    pub fn record_conflict(&mut self) {
        self.conflicts += 1;
    }

    /// Record one decision.
    pub fn record_decision(&mut self) {
        self.decisions += 1;
    }

    /// Record one restart.
    pub fn record_restart(&mut self) {
        self.restarts += 1;
    }

    /// Record one theory check.
    pub fn record_theory_check(&mut self) {
        self.theory_checks += 1;
    }

    /// Check whether any resource limit has been exceeded.
    ///
    /// Returns `Some(reason)` if a limit was hit, or `None` if all limits are
    /// still satisfied.
    #[must_use]
    pub fn check(&self) -> Option<ResourceExhausted> {
        // Check timeout (std only)
        #[cfg(feature = "std")]
        if let (Some(timeout), Some(start)) = (self.limits.timeout, self.start_time) {
            if start.elapsed() >= timeout {
                return Some(ResourceExhausted::Timeout);
            }
        }

        // Check conflicts
        if let Some(max) = self.limits.max_conflicts {
            if self.conflicts >= max {
                return Some(ResourceExhausted::MaxConflicts);
            }
        }

        // Check decisions
        if let Some(max) = self.limits.max_decisions {
            if self.decisions >= max {
                return Some(ResourceExhausted::MaxDecisions);
            }
        }

        // Check restarts
        if let Some(max) = self.limits.max_restarts {
            if self.restarts >= max {
                return Some(ResourceExhausted::MaxRestarts);
            }
        }

        // Check theory checks
        if let Some(max) = self.limits.max_theory_checks {
            if self.theory_checks >= max {
                return Some(ResourceExhausted::MaxTheoryChecks);
            }
        }

        // Check memory (std only, best-effort)
        #[cfg(feature = "std")]
        if let Some(max_mb) = self.limits.max_memory_mb {
            // Use a simple heuristic: check process memory via /proc on Linux
            // or fallback to allocated heap estimation
            if let Some(current_mb) = Self::estimate_memory_mb() {
                if current_mb >= max_mb {
                    return Some(ResourceExhausted::MaxMemory);
                }
            }
        }

        None
    }

    /// Check limits and return a `Result`.
    ///
    /// Convenience wrapper: returns `Ok(())` when no limit is hit, or
    /// `Err(ResourceExhausted)` when one is.
    pub fn check_result(&self) -> core::result::Result<(), ResourceExhausted> {
        match self.check() {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    /// Get the configured limits.
    #[must_use]
    pub fn limits(&self) -> &ResourceLimits {
        &self.limits
    }

    /// Get elapsed time (std only).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn elapsed(&self) -> Option<core::time::Duration> {
        self.start_time.map(|s| s.elapsed())
    }

    /// Estimate current process memory usage in megabytes (best-effort).
    #[cfg(feature = "std")]
    fn estimate_memory_mb() -> Option<u64> {
        // Try reading /proc/self/statm on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/self/statm") {
                // statm fields: size resident shared text lib data dt (in pages)
                let mut parts = contents.split_whitespace();
                if let Some(resident_str) = parts.nth(1) {
                    if let Ok(pages) = resident_str.parse::<u64>() {
                        let page_size = 4096u64; // typical page size
                        return Some(pages * page_size / (1024 * 1024));
                    }
                }
            }
            None
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, use mach task_info to get resident memory
            // Fallback: rough estimate based on allocator stats is not portable,
            // so return None to skip the memory limit check.
            None
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::time::Duration;

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::new();
        assert!(limits.timeout.is_none());
        assert!(limits.max_conflicts.is_none());
        assert!(limits.max_memory_mb.is_none());
        assert!(limits.max_decisions.is_none());
        assert!(limits.max_restarts.is_none());
        assert!(limits.max_theory_checks.is_none());
        assert!(!limits.has_any_limit());
    }

    #[test]
    fn test_resource_limits_builder() {
        let limits = ResourceLimits::new()
            .with_timeout(Duration::from_secs(30))
            .with_max_conflicts(10_000)
            .with_max_decisions(100_000)
            .with_max_restarts(500)
            .with_max_theory_checks(50_000)
            .with_max_memory_mb(1024);

        assert_eq!(limits.timeout, Some(Duration::from_secs(30)));
        assert_eq!(limits.max_conflicts, Some(10_000));
        assert_eq!(limits.max_decisions, Some(100_000));
        assert_eq!(limits.max_restarts, Some(500));
        assert_eq!(limits.max_theory_checks, Some(50_000));
        assert_eq!(limits.max_memory_mb, Some(1024));
        assert!(limits.has_any_limit());
    }

    #[test]
    fn test_monitor_no_limits() {
        let monitor = ResourceMonitor::new(ResourceLimits::new());
        assert!(monitor.check().is_none());
    }

    #[test]
    fn test_monitor_conflict_limit() {
        let limits = ResourceLimits::new().with_max_conflicts(5);
        let mut monitor = ResourceMonitor::new(limits);

        for _ in 0..4 {
            monitor.record_conflict();
            assert!(monitor.check().is_none());
        }
        monitor.record_conflict();
        assert_eq!(monitor.check(), Some(ResourceExhausted::MaxConflicts));
    }

    #[test]
    fn test_monitor_decision_limit() {
        let limits = ResourceLimits::new().with_max_decisions(3);
        let mut monitor = ResourceMonitor::new(limits);

        monitor.record_decision();
        monitor.record_decision();
        assert!(monitor.check().is_none());
        monitor.record_decision();
        assert_eq!(monitor.check(), Some(ResourceExhausted::MaxDecisions));
    }

    #[test]
    fn test_monitor_restart_limit() {
        let limits = ResourceLimits::new().with_max_restarts(2);
        let mut monitor = ResourceMonitor::new(limits);

        monitor.record_restart();
        assert!(monitor.check().is_none());
        monitor.record_restart();
        assert_eq!(monitor.check(), Some(ResourceExhausted::MaxRestarts));
    }

    #[test]
    fn test_monitor_theory_check_limit() {
        let limits = ResourceLimits::new().with_max_theory_checks(10);
        let mut monitor = ResourceMonitor::new(limits);

        for _ in 0..9 {
            monitor.record_theory_check();
        }
        assert!(monitor.check().is_none());
        monitor.record_theory_check();
        assert_eq!(monitor.check(), Some(ResourceExhausted::MaxTheoryChecks));
    }

    #[test]
    fn test_monitor_reset() {
        let limits = ResourceLimits::new().with_max_conflicts(5);
        let mut monitor = ResourceMonitor::new(limits);

        for _ in 0..5 {
            monitor.record_conflict();
        }
        assert!(monitor.check().is_some());

        monitor.reset();
        assert!(monitor.check().is_none());
        assert_eq!(monitor.conflicts, 0);
    }

    #[test]
    fn test_monitor_check_result() {
        let limits = ResourceLimits::new().with_max_conflicts(1);
        let mut monitor = ResourceMonitor::new(limits);

        assert!(monitor.check_result().is_ok());
        monitor.record_conflict();
        assert!(monitor.check_result().is_err());
    }

    #[test]
    fn test_resource_exhausted_display() {
        assert_eq!(ResourceExhausted::Timeout.to_string(), "timeout");
        assert_eq!(
            ResourceExhausted::MaxConflicts.to_string(),
            "max conflicts reached"
        );
        assert_eq!(
            ResourceExhausted::MaxMemory.to_string(),
            "max memory reached"
        );
        assert_eq!(
            ResourceExhausted::MaxDecisions.to_string(),
            "max decisions reached"
        );
        assert_eq!(
            ResourceExhausted::MaxRestarts.to_string(),
            "max restarts reached"
        );
        assert_eq!(
            ResourceExhausted::MaxTheoryChecks.to_string(),
            "max theory checks reached"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_monitor_timeout_not_hit_immediately() {
        let limits = ResourceLimits::new().with_timeout(Duration::from_secs(60));
        let monitor = ResourceMonitor::new(limits);
        // A 60-second timeout should definitely not be hit immediately
        assert!(monitor.check().is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_monitor_elapsed() {
        let limits = ResourceLimits::new().with_timeout(Duration::from_secs(60));
        let monitor = ResourceMonitor::new(limits);
        let elapsed = monitor.elapsed();
        assert!(elapsed.is_some());
        // Should be very small since we just created it
        assert!(elapsed.is_some_and(|e| e < Duration::from_secs(1)));
    }
}
