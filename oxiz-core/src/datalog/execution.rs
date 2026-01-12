//! Execution context and statistics for Datalog evaluation
//!
//! Provides runtime context, statistics tracking, and execution monitoring.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::relation::RelationId;
use super::rule::RuleId;

/// Execution context for Datalog evaluation
#[derive(Debug)]
pub struct ExecutionContext {
    /// Context ID
    id: u64,
    /// Start time
    start_time: Instant,
    /// Timeout duration
    timeout: Option<Duration>,
    /// Cancellation flag
    cancelled: AtomicBool,
    /// Statistics
    stats: RwLock<ExecutionStats>,
    /// Per-rule statistics
    rule_stats: RwLock<HashMap<RuleId, RuleStats>>,
    /// Per-relation statistics
    relation_stats: RwLock<HashMap<RelationId, RelationStats>>,
    /// Trace events
    trace: RwLock<Vec<TraceEvent>>,
    /// Trace enabled
    trace_enabled: bool,
}

/// Overall execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total iterations
    pub iterations: u64,
    /// Total tuples examined
    pub tuples_examined: u64,
    /// Total tuples derived
    pub tuples_derived: u64,
    /// Total rules fired
    pub rules_fired: u64,
    /// Total joins performed
    pub joins_performed: u64,
    /// Total index lookups
    pub index_lookups: u64,
    /// Total propagations
    pub propagations: u64,
    /// Memory used (bytes)
    pub memory_used: u64,
    /// Peak memory (bytes)
    pub peak_memory: u64,
    /// Time spent in rule evaluation (microseconds)
    pub rule_eval_time_us: u64,
    /// Time spent in joins (microseconds)
    pub join_time_us: u64,
    /// Time spent in propagation (microseconds)
    pub propagation_time_us: u64,
}

/// Per-rule statistics
#[derive(Debug, Clone, Default)]
pub struct RuleStats {
    /// Times rule was fired
    pub fires: u64,
    /// Tuples derived by this rule
    pub tuples_derived: u64,
    /// Time spent evaluating (microseconds)
    pub eval_time_us: u64,
    /// Average bindings per fire
    pub avg_bindings: f64,
}

/// Per-relation statistics
#[derive(Debug, Clone, Default)]
pub struct RelationStats {
    /// Tuples at start
    pub initial_size: u64,
    /// Tuples at end
    pub final_size: u64,
    /// Tuples inserted
    pub insertions: u64,
    /// Duplicate insertions (rejected)
    pub duplicates: u64,
    /// Scans performed
    pub scans: u64,
    /// Index lookups
    pub index_lookups: u64,
}

/// Trace event for debugging/profiling
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Event timestamp (microseconds from start)
    pub timestamp_us: u64,
    /// Event kind
    pub kind: TraceEventKind,
    /// Associated data
    pub data: TraceData,
}

/// Kind of trace event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEventKind {
    /// Rule evaluation started
    RuleStart,
    /// Rule evaluation ended
    RuleEnd,
    /// Tuple derived
    TupleDerived,
    /// Join started
    JoinStart,
    /// Join ended
    JoinEnd,
    /// Iteration started
    IterationStart,
    /// Iteration ended
    IterationEnd,
    /// Fixed point reached
    FixedPoint,
    /// Timeout
    Timeout,
    /// Cancelled
    Cancelled,
}

/// Data associated with trace event
#[derive(Debug, Clone)]
pub enum TraceData {
    /// No data
    None,
    /// Rule ID
    Rule(RuleId),
    /// Relation ID
    Relation(RelationId),
    /// Count
    Count(u64),
    /// Duration in microseconds
    Duration(u64),
    /// String message
    Message(String),
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            start_time: Instant::now(),
            timeout: None,
            cancelled: AtomicBool::new(false),
            stats: RwLock::new(ExecutionStats::default()),
            rule_stats: RwLock::new(HashMap::new()),
            relation_stats: RwLock::new(HashMap::new()),
            trace: RwLock::new(Vec::new()),
            trace_enabled: false,
        }
    }

    /// Create with timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        let mut ctx = Self::new();
        ctx.timeout = Some(timeout);
        ctx
    }

    /// Create with tracing enabled
    pub fn with_tracing() -> Self {
        let mut ctx = Self::new();
        ctx.trace_enabled = true;
        ctx
    }

    /// Get context ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if timed out
    pub fn is_timed_out(&self) -> bool {
        self.timeout.map(|t| self.elapsed() >= t).unwrap_or(false)
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Cancel execution
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        if self.trace_enabled {
            self.trace_event(TraceEventKind::Cancelled, TraceData::None);
        }
    }

    /// Check if should stop (timeout or cancelled)
    pub fn should_stop(&self) -> bool {
        self.is_timed_out() || self.is_cancelled()
    }

    /// Increment iteration count
    pub fn inc_iterations(&self) {
        self.stats.write().iterations += 1;
    }

    /// Increment tuples examined
    pub fn inc_tuples_examined(&self, count: u64) {
        self.stats.write().tuples_examined += count;
    }

    /// Increment tuples derived
    pub fn inc_tuples_derived(&self, count: u64) {
        self.stats.write().tuples_derived += count;
    }

    /// Increment rules fired
    pub fn inc_rules_fired(&self) {
        self.stats.write().rules_fired += 1;
    }

    /// Increment joins performed
    pub fn inc_joins(&self) {
        self.stats.write().joins_performed += 1;
    }

    /// Increment index lookups
    pub fn inc_index_lookups(&self) {
        self.stats.write().index_lookups += 1;
    }

    /// Record rule statistics
    pub fn record_rule_stats(
        &self,
        rule_id: RuleId,
        tuples_derived: u64,
        eval_time_us: u64,
        bindings: u64,
    ) {
        let mut rule_stats = self.rule_stats.write();
        let stats = rule_stats.entry(rule_id).or_default();
        stats.fires += 1;
        stats.tuples_derived += tuples_derived;
        stats.eval_time_us += eval_time_us;
        // Update average bindings
        let total_fires = stats.fires as f64;
        stats.avg_bindings =
            ((stats.avg_bindings * (total_fires - 1.0)) + bindings as f64) / total_fires;
    }

    /// Record relation statistics
    pub fn record_relation_insert(&self, rel_id: RelationId, inserted: bool) {
        let mut rel_stats = self.relation_stats.write();
        let stats = rel_stats.entry(rel_id).or_default();
        if inserted {
            stats.insertions += 1;
        } else {
            stats.duplicates += 1;
        }
    }

    /// Record relation scan
    pub fn record_relation_scan(&self, rel_id: RelationId) {
        let mut rel_stats = self.relation_stats.write();
        let stats = rel_stats.entry(rel_id).or_default();
        stats.scans += 1;
    }

    /// Record index lookup
    pub fn record_index_lookup(&self, rel_id: RelationId) {
        let mut rel_stats = self.relation_stats.write();
        let stats = rel_stats.entry(rel_id).or_default();
        stats.index_lookups += 1;
    }

    /// Add trace event
    pub fn trace_event(&self, kind: TraceEventKind, data: TraceData) {
        if !self.trace_enabled {
            return;
        }

        let timestamp_us = self.elapsed().as_micros() as u64;
        let event = TraceEvent {
            timestamp_us,
            kind,
            data,
        };
        self.trace.write().push(event);
    }

    /// Get overall statistics
    pub fn stats(&self) -> ExecutionStats {
        self.stats.read().clone()
    }

    /// Get rule statistics
    pub fn rule_stats(&self) -> HashMap<RuleId, RuleStats> {
        self.rule_stats.read().clone()
    }

    /// Get relation statistics
    pub fn relation_stats(&self) -> HashMap<RelationId, RelationStats> {
        self.relation_stats.read().clone()
    }

    /// Get trace events
    pub fn trace_events(&self) -> Vec<TraceEvent> {
        self.trace.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = ExecutionStats::default();
        self.rule_stats.write().clear();
        self.relation_stats.write().clear();
        self.trace.write().clear();
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution monitor for real-time statistics
#[derive(Debug)]
pub struct ExecutionMonitor {
    /// Context being monitored
    context: Arc<ExecutionContext>,
    /// Snapshot interval
    snapshot_interval: Duration,
    /// Snapshots
    snapshots: RwLock<Vec<StatsSnapshot>>,
    /// Running flag
    running: AtomicBool,
}

/// Statistics snapshot
#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    /// Timestamp (microseconds from start)
    pub timestamp_us: u64,
    /// Statistics at this point
    pub stats: ExecutionStats,
}

impl ExecutionMonitor {
    /// Create a new monitor
    pub fn new(context: Arc<ExecutionContext>) -> Self {
        Self {
            context,
            snapshot_interval: Duration::from_millis(100),
            snapshots: RwLock::new(Vec::new()),
            running: AtomicBool::new(false),
        }
    }

    /// Set snapshot interval
    pub fn set_interval(&mut self, interval: Duration) {
        self.snapshot_interval = interval;
    }

    /// Take a snapshot
    pub fn snapshot(&self) {
        let stats = self.context.stats();
        let timestamp_us = self.context.elapsed().as_micros() as u64;
        let snapshot = StatsSnapshot {
            timestamp_us,
            stats,
        };
        self.snapshots.write().push(snapshot);
    }

    /// Get all snapshots
    pub fn snapshots(&self) -> Vec<StatsSnapshot> {
        self.snapshots.read().clone()
    }

    /// Get latest snapshot
    pub fn latest(&self) -> Option<StatsSnapshot> {
        self.snapshots.read().last().cloned()
    }

    /// Get throughput (tuples derived per second)
    pub fn throughput(&self) -> f64 {
        let stats = self.context.stats();
        let elapsed_secs = self.context.elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            stats.tuples_derived as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    /// Clear snapshots
    pub fn clear(&self) {
        self.snapshots.write().clear();
    }
}

/// Execution timer for measuring durations
#[derive(Debug)]
pub struct ExecutionTimer {
    /// Start time
    start: Instant,
    /// Accumulated time (for pausable timers)
    accumulated: Duration,
    /// Paused flag
    paused: bool,
}

impl ExecutionTimer {
    /// Create and start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            accumulated: Duration::ZERO,
            paused: false,
        }
    }

    /// Stop and get elapsed time
    pub fn stop(&mut self) -> Duration {
        if !self.paused {
            self.accumulated += self.start.elapsed();
            self.paused = true;
        }
        self.accumulated
    }

    /// Pause timer
    pub fn pause(&mut self) {
        if !self.paused {
            self.accumulated += self.start.elapsed();
            self.paused = true;
        }
    }

    /// Resume timer
    pub fn resume(&mut self) {
        if self.paused {
            self.start = Instant::now();
            self.paused = false;
        }
    }

    /// Get elapsed time (without stopping)
    pub fn elapsed(&self) -> Duration {
        if self.paused {
            self.accumulated
        } else {
            self.accumulated + self.start.elapsed()
        }
    }

    /// Reset timer
    pub fn reset(&mut self) {
        self.start = Instant::now();
        self.accumulated = Duration::ZERO;
        self.paused = false;
    }
}

/// Scoped timer that records to context on drop
pub struct ScopedTimer<'a> {
    context: &'a ExecutionContext,
    rule_id: Option<RuleId>,
    start: Instant,
}

impl<'a> ScopedTimer<'a> {
    /// Create a timer for rule evaluation
    pub fn for_rule(context: &'a ExecutionContext, rule_id: RuleId) -> Self {
        if context.trace_enabled {
            context.trace_event(TraceEventKind::RuleStart, TraceData::Rule(rule_id));
        }
        Self {
            context,
            rule_id: Some(rule_id),
            start: Instant::now(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for ScopedTimer<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if let Some(_rule_id) = self.rule_id
            && self.context.trace_enabled
        {
            self.context.trace_event(
                TraceEventKind::RuleEnd,
                TraceData::Duration(elapsed.as_micros() as u64),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_execution_context() {
        let ctx = ExecutionContext::new();
        assert!(!ctx.is_cancelled());
        assert!(!ctx.is_timed_out());
    }

    #[test]
    fn test_statistics() {
        let ctx = ExecutionContext::new();

        ctx.inc_iterations();
        ctx.inc_tuples_derived(10);
        ctx.inc_rules_fired();

        let stats = ctx.stats();
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.tuples_derived, 10);
        assert_eq!(stats.rules_fired, 1);
    }

    #[test]
    fn test_cancellation() {
        let ctx = ExecutionContext::new();
        assert!(!ctx.should_stop());

        ctx.cancel();
        assert!(ctx.should_stop());
        assert!(ctx.is_cancelled());
    }

    #[test]
    fn test_timeout() {
        let ctx = ExecutionContext::with_timeout(Duration::from_millis(10));
        assert!(!ctx.is_timed_out());

        thread::sleep(Duration::from_millis(20));
        assert!(ctx.is_timed_out());
        assert!(ctx.should_stop());
    }

    #[test]
    fn test_rule_stats() {
        let ctx = ExecutionContext::new();
        let rule_id = RuleId::new(1);

        ctx.record_rule_stats(rule_id, 5, 100, 10);
        ctx.record_rule_stats(rule_id, 3, 50, 6);

        let stats = ctx.rule_stats();
        let rule = stats.get(&rule_id).unwrap();
        assert_eq!(rule.fires, 2);
        assert_eq!(rule.tuples_derived, 8);
        assert_eq!(rule.eval_time_us, 150);
    }

    #[test]
    fn test_execution_timer() {
        let mut timer = ExecutionTimer::start();
        thread::sleep(Duration::from_millis(10));

        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(10));

        // Stopping again shouldn't change time
        let elapsed2 = timer.stop();
        assert_eq!(elapsed, elapsed2);
    }

    #[test]
    fn test_pausable_timer() {
        let mut timer = ExecutionTimer::start();
        thread::sleep(Duration::from_millis(10));

        timer.pause();
        let elapsed1 = timer.elapsed();

        thread::sleep(Duration::from_millis(10));
        let elapsed2 = timer.elapsed();

        // Elapsed shouldn't increase while paused
        assert_eq!(elapsed1, elapsed2);

        timer.resume();
        thread::sleep(Duration::from_millis(10));
        let elapsed3 = timer.elapsed();

        // Now it should be larger
        assert!(elapsed3 > elapsed2);
    }

    #[test]
    fn test_tracing() {
        let ctx = ExecutionContext::with_tracing();

        ctx.trace_event(TraceEventKind::IterationStart, TraceData::Count(1));
        ctx.trace_event(TraceEventKind::TupleDerived, TraceData::Count(5));
        ctx.trace_event(TraceEventKind::IterationEnd, TraceData::Count(1));

        let events = ctx.trace_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].kind, TraceEventKind::IterationStart);
    }

    #[test]
    fn test_monitor() {
        let ctx = Arc::new(ExecutionContext::new());
        let monitor = ExecutionMonitor::new(ctx.clone());

        ctx.inc_tuples_derived(100);
        monitor.snapshot();

        ctx.inc_tuples_derived(200);
        monitor.snapshot();

        let snapshots = monitor.snapshots();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].stats.tuples_derived, 100);
        assert_eq!(snapshots[1].stats.tuples_derived, 300);
    }
}
