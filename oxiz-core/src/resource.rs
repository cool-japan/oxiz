//! Resource limit enforcement and management
//!
//! This module provides utilities for enforcing resource limits
//! (time, memory, decisions, conflicts) during solver execution.

use crate::config::ResourceLimits;
use crate::statistics::Statistics;
use std::time::{Duration, Instant};

/// Resource manager for enforcing limits
#[derive(Debug)]
pub struct ResourceManager {
    /// The resource limits to enforce
    limits: ResourceLimits,
    /// Start time of the current operation
    start_time: Option<Instant>,
}

/// Result of checking resource limits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitStatus {
    /// All limits are within bounds
    Ok,
    /// Time limit exceeded
    TimeExceeded,
    /// Decision limit exceeded
    DecisionExceeded,
    /// Conflict limit exceeded
    ConflictExceeded,
    /// Memory limit exceeded
    MemoryExceeded,
}

impl ResourceManager {
    /// Create a new resource manager with the given limits
    #[must_use]
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            start_time: None,
        }
    }

    /// Start tracking resource usage
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Check if any resource limits have been exceeded
    #[must_use]
    pub fn check_limits(&self, stats: &Statistics) -> LimitStatus {
        // Check time limit
        if let Some(time_limit) = self.limits.time_limit
            && let Some(start) = self.start_time
            && start.elapsed() > time_limit
        {
            return LimitStatus::TimeExceeded;
        }

        // Check decision limit
        if let Some(decision_limit) = self.limits.decision_limit
            && stats.decisions >= decision_limit
        {
            return LimitStatus::DecisionExceeded;
        }

        // Check conflict limit
        if let Some(conflict_limit) = self.limits.conflict_limit
            && stats.conflicts >= conflict_limit
        {
            return LimitStatus::ConflictExceeded;
        }

        // Check memory limit
        if let Some(memory_limit) = self.limits.memory_limit
            && stats.memory_used >= memory_limit
        {
            return LimitStatus::MemoryExceeded;
        }

        LimitStatus::Ok
    }

    /// Get the elapsed time since start
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Get the remaining time budget
    #[must_use]
    pub fn remaining_time(&self) -> Option<Duration> {
        match (self.limits.time_limit, self.start_time) {
            (Some(limit), Some(start)) => {
                let elapsed = start.elapsed();
                if elapsed < limit {
                    Some(limit - elapsed)
                } else {
                    Some(Duration::ZERO)
                }
            }
            _ => None,
        }
    }

    /// Get the remaining decision budget
    #[must_use]
    pub fn remaining_decisions(&self, stats: &Statistics) -> Option<u64> {
        self.limits
            .decision_limit
            .map(|limit| limit.saturating_sub(stats.decisions))
    }

    /// Get the remaining conflict budget
    #[must_use]
    pub fn remaining_conflicts(&self, stats: &Statistics) -> Option<u64> {
        self.limits
            .conflict_limit
            .map(|limit| limit.saturating_sub(stats.conflicts))
    }

    /// Reset the resource manager
    pub fn reset(&mut self) {
        self.start_time = None;
    }

    /// Update the limits
    pub fn set_limits(&mut self, limits: ResourceLimits) {
        self.limits = limits;
    }

    /// Get the current limits
    #[must_use]
    pub fn limits(&self) -> &ResourceLimits {
        &self.limits
    }
}

impl std::fmt::Display for LimitStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LimitStatus::Ok => write!(f, "OK"),
            LimitStatus::TimeExceeded => write!(f, "Time limit exceeded"),
            LimitStatus::DecisionExceeded => write!(f, "Decision limit exceeded"),
            LimitStatus::ConflictExceeded => write!(f, "Conflict limit exceeded"),
            LimitStatus::MemoryExceeded => write!(f, "Memory limit exceeded"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_no_limits() {
        let limits = ResourceLimits::default();
        let manager = ResourceManager::new(limits);
        let stats = Statistics::new();

        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);
    }

    #[test]
    fn test_time_limit() {
        let limits = ResourceLimits {
            time_limit: Some(Duration::from_millis(50)),
            decision_limit: None,
            conflict_limit: None,
            memory_limit: None,
        };

        let mut manager = ResourceManager::new(limits);
        manager.start();

        let stats = Statistics::new();
        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        thread::sleep(Duration::from_millis(60));
        assert_eq!(manager.check_limits(&stats), LimitStatus::TimeExceeded);
    }

    #[test]
    fn test_decision_limit() {
        let limits = ResourceLimits {
            time_limit: None,
            decision_limit: Some(100),
            conflict_limit: None,
            memory_limit: None,
        };

        let manager = ResourceManager::new(limits);
        let mut stats = Statistics::new();

        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        stats.decisions = 50;
        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        stats.decisions = 100;
        assert_eq!(manager.check_limits(&stats), LimitStatus::DecisionExceeded);

        stats.decisions = 150;
        assert_eq!(manager.check_limits(&stats), LimitStatus::DecisionExceeded);
    }

    #[test]
    fn test_conflict_limit() {
        let limits = ResourceLimits {
            time_limit: None,
            decision_limit: None,
            conflict_limit: Some(50),
            memory_limit: None,
        };

        let manager = ResourceManager::new(limits);
        let mut stats = Statistics::new();

        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        stats.conflicts = 49;
        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        stats.conflicts = 50;
        assert_eq!(manager.check_limits(&stats), LimitStatus::ConflictExceeded);
    }

    #[test]
    fn test_memory_limit() {
        let limits = ResourceLimits {
            time_limit: None,
            decision_limit: None,
            conflict_limit: None,
            memory_limit: Some(1024 * 1024), // 1 MB
        };

        let manager = ResourceManager::new(limits);
        let mut stats = Statistics::new();

        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        stats.set_memory_used(512 * 1024);
        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        stats.set_memory_used(1024 * 1024);
        assert_eq!(manager.check_limits(&stats), LimitStatus::MemoryExceeded);
    }

    #[test]
    fn test_elapsed() {
        let mut manager = ResourceManager::new(ResourceLimits::default());

        assert!(manager.elapsed().is_none());

        manager.start();
        thread::sleep(Duration::from_millis(10));

        let elapsed = manager.elapsed().unwrap();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_remaining_time() {
        let limits = ResourceLimits {
            time_limit: Some(Duration::from_secs(10)),
            decision_limit: None,
            conflict_limit: None,
            memory_limit: None,
        };

        let mut manager = ResourceManager::new(limits);

        assert!(manager.remaining_time().is_none());

        manager.start();
        thread::sleep(Duration::from_millis(10));

        let remaining = manager.remaining_time().unwrap();
        assert!(remaining < Duration::from_secs(10));
        assert!(remaining > Duration::from_secs(9));
    }

    #[test]
    fn test_remaining_decisions() {
        let limits = ResourceLimits {
            time_limit: None,
            decision_limit: Some(100),
            conflict_limit: None,
            memory_limit: None,
        };

        let manager = ResourceManager::new(limits);
        let mut stats = Statistics::new();

        assert_eq!(manager.remaining_decisions(&stats), Some(100));

        stats.decisions = 30;
        assert_eq!(manager.remaining_decisions(&stats), Some(70));

        stats.decisions = 100;
        assert_eq!(manager.remaining_decisions(&stats), Some(0));

        stats.decisions = 150;
        assert_eq!(manager.remaining_decisions(&stats), Some(0));
    }

    #[test]
    fn test_remaining_conflicts() {
        let limits = ResourceLimits {
            time_limit: None,
            decision_limit: None,
            conflict_limit: Some(50),
            memory_limit: None,
        };

        let manager = ResourceManager::new(limits);
        let mut stats = Statistics::new();

        assert_eq!(manager.remaining_conflicts(&stats), Some(50));

        stats.conflicts = 20;
        assert_eq!(manager.remaining_conflicts(&stats), Some(30));

        stats.conflicts = 50;
        assert_eq!(manager.remaining_conflicts(&stats), Some(0));
    }

    #[test]
    fn test_reset() {
        let mut manager = ResourceManager::new(ResourceLimits::default());

        manager.start();
        assert!(manager.elapsed().is_some());

        manager.reset();
        assert!(manager.elapsed().is_none());
    }

    #[test]
    fn test_set_limits() {
        let mut manager = ResourceManager::new(ResourceLimits::default());

        let new_limits = ResourceLimits {
            time_limit: Some(Duration::from_secs(60)),
            decision_limit: Some(1000),
            conflict_limit: Some(500),
            memory_limit: Some(1024 * 1024 * 100),
        };

        manager.set_limits(new_limits.clone());

        assert_eq!(manager.limits().time_limit, new_limits.time_limit);
        assert_eq!(manager.limits().decision_limit, new_limits.decision_limit);
    }

    #[test]
    fn test_limit_status_display() {
        assert_eq!(LimitStatus::Ok.to_string(), "OK");
        assert_eq!(LimitStatus::TimeExceeded.to_string(), "Time limit exceeded");
        assert_eq!(
            LimitStatus::DecisionExceeded.to_string(),
            "Decision limit exceeded"
        );
        assert_eq!(
            LimitStatus::ConflictExceeded.to_string(),
            "Conflict limit exceeded"
        );
        assert_eq!(
            LimitStatus::MemoryExceeded.to_string(),
            "Memory limit exceeded"
        );
    }

    #[test]
    fn test_multiple_limits() {
        let limits = ResourceLimits {
            time_limit: Some(Duration::from_millis(100)),
            decision_limit: Some(100),
            conflict_limit: Some(50),
            memory_limit: Some(1024 * 1024),
        };

        let mut manager = ResourceManager::new(limits);
        manager.start();
        let mut stats = Statistics::new();

        // All OK initially
        assert_eq!(manager.check_limits(&stats), LimitStatus::Ok);

        // Hit conflict limit first
        stats.conflicts = 50;
        assert_eq!(manager.check_limits(&stats), LimitStatus::ConflictExceeded);
    }
}
