//! Solver Trace Generation.
//!
//! Records solver events (decisions, propagations, conflicts, theory checks,
//! restarts, backtracks) and provides human-readable and JSON trace output.
//!
//! ## Usage
//!
//! ```ignore
//! let config = TraceConfig::default();
//! let mut tracer = SolverTracer::new(config);
//! tracer.record(TraceEvent::Decision { var: 1, value: true, level: 0 });
//! let text = tracer.format_trace();
//! let json = tracer.write_trace_json();
//! ```
//!
//! ## References
//!
//! - Z3's `util/trace.cpp`

#[allow(unused_imports)]
use crate::prelude::*;

/// A solver trace event.
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// A decision was made.
    Decision {
        /// Variable index.
        var: u32,
        /// Value assigned.
        value: bool,
        /// Decision level.
        level: u32,
    },
    /// A propagation occurred.
    Propagation {
        /// Literal that was propagated (positive = true, negative = false).
        literal: i64,
        /// Index of the reason clause.
        reason_clause: u32,
    },
    /// A conflict was detected.
    Conflict {
        /// Index of the conflicting clause.
        conflicting_clause: u32,
        /// Index of the learned clause (if learning occurred).
        learned_clause: Option<u32>,
        /// Number of literals in the learned clause.
        learned_size: Option<usize>,
    },
    /// A theory check was performed.
    TheoryCheck {
        /// Theory name (e.g., "EUF", "LRA", "BV").
        theory: String,
        /// Result: "consistent", "conflict", "propagation".
        result: String,
        /// Time in microseconds.
        time_us: u64,
    },
    /// A restart occurred.
    Restart {
        /// Reason for the restart (e.g., "glucose", "geometric").
        reason: String,
        /// New strategy after restart (if changed).
        new_strategy: Option<String>,
    },
    /// A backtrack occurred.
    Backtrack {
        /// Level we backtracked from.
        from_level: u32,
        /// Level we backtracked to.
        to_level: u32,
    },
    /// Clause learned.
    ClauseLearned {
        /// Clause index.
        clause_id: u32,
        /// Number of literals.
        num_literals: usize,
        /// Glue/LBD value.
        glue: u32,
    },
    /// Assertion added by user.
    AssertionAdded {
        /// Assertion index.
        index: u32,
        /// Brief description.
        description: String,
    },
}

impl TraceEvent {
    /// Get the event type name.
    pub fn event_type(&self) -> &str {
        match self {
            TraceEvent::Decision { .. } => "decision",
            TraceEvent::Propagation { .. } => "propagation",
            TraceEvent::Conflict { .. } => "conflict",
            TraceEvent::TheoryCheck { .. } => "theory_check",
            TraceEvent::Restart { .. } => "restart",
            TraceEvent::Backtrack { .. } => "backtrack",
            TraceEvent::ClauseLearned { .. } => "clause_learned",
            TraceEvent::AssertionAdded { .. } => "assertion_added",
        }
    }

    /// Format a single event as a human-readable string.
    pub fn format(&self) -> String {
        match self {
            TraceEvent::Decision { var, value, level } => {
                format!("DECIDE  v{} = {} @ level {}", var, value, level)
            }
            TraceEvent::Propagation {
                literal,
                reason_clause,
            } => {
                let var = literal.unsigned_abs();
                let pol = if *literal > 0 { "T" } else { "F" };
                format!("PROP    x{} = {} (reason: clause #{})", var, pol, reason_clause)
            }
            TraceEvent::Conflict {
                conflicting_clause,
                learned_clause,
                learned_size,
            } => {
                let learned_str = match (learned_clause, learned_size) {
                    (Some(lc), Some(ls)) => format!(", learned clause #{} (size {})", lc, ls),
                    (Some(lc), None) => format!(", learned clause #{}", lc),
                    _ => String::new(),
                };
                format!(
                    "CONFLICT clause #{}{}",
                    conflicting_clause, learned_str
                )
            }
            TraceEvent::TheoryCheck {
                theory,
                result,
                time_us,
            } => {
                format!("THEORY  {} -> {} ({}us)", theory, result, time_us)
            }
            TraceEvent::Restart {
                reason,
                new_strategy,
            } => {
                let strat = new_strategy
                    .as_deref()
                    .unwrap_or("unchanged");
                format!("RESTART reason={}, strategy={}", reason, strat)
            }
            TraceEvent::Backtrack {
                from_level,
                to_level,
            } => {
                format!("BACKTRACK level {} -> {}", from_level, to_level)
            }
            TraceEvent::ClauseLearned {
                clause_id,
                num_literals,
                glue,
            } => {
                format!(
                    "LEARNED clause #{} (lits={}, glue={})",
                    clause_id, num_literals, glue
                )
            }
            TraceEvent::AssertionAdded { index, description } => {
                format!("ASSERT  [{}] {}", index, description)
            }
        }
    }

    /// Format a single event as a JSON object string (no trailing comma).
    pub fn to_json(&self) -> String {
        match self {
            TraceEvent::Decision { var, value, level } => {
                format!(
                    r#"{{"type":"decision","var":{},"value":{},"level":{}}}"#,
                    var, value, level
                )
            }
            TraceEvent::Propagation {
                literal,
                reason_clause,
            } => {
                format!(
                    r#"{{"type":"propagation","literal":{},"reason_clause":{}}}"#,
                    literal, reason_clause
                )
            }
            TraceEvent::Conflict {
                conflicting_clause,
                learned_clause,
                learned_size,
            } => {
                let lc = learned_clause.map_or("null".to_string(), |v| format!("{}", v));
                let ls = learned_size.map_or("null".to_string(), |v| format!("{}", v));
                format!(
                    r#"{{"type":"conflict","conflicting_clause":{},"learned_clause":{},"learned_size":{}}}"#,
                    conflicting_clause, lc, ls
                )
            }
            TraceEvent::TheoryCheck {
                theory,
                result,
                time_us,
            } => {
                format!(
                    r#"{{"type":"theory_check","theory":"{}","result":"{}","time_us":{}}}"#,
                    theory, result, time_us
                )
            }
            TraceEvent::Restart {
                reason,
                new_strategy,
            } => {
                let strat = new_strategy
                    .as_deref()
                    .map_or("null".to_string(), |s| format!("\"{}\"", s));
                format!(
                    r#"{{"type":"restart","reason":"{}","new_strategy":{}}}"#,
                    reason, strat
                )
            }
            TraceEvent::Backtrack {
                from_level,
                to_level,
            } => {
                format!(
                    r#"{{"type":"backtrack","from_level":{},"to_level":{}}}"#,
                    from_level, to_level
                )
            }
            TraceEvent::ClauseLearned {
                clause_id,
                num_literals,
                glue,
            } => {
                format!(
                    r#"{{"type":"clause_learned","clause_id":{},"num_literals":{},"glue":{}}}"#,
                    clause_id, num_literals, glue
                )
            }
            TraceEvent::AssertionAdded { index, description } => {
                // Escape quotes in description for JSON safety.
                let escaped = description.replace('\\', "\\\\").replace('"', "\\\"");
                format!(
                    r#"{{"type":"assertion_added","index":{},"description":"{}"}}"#,
                    index, escaped
                )
            }
        }
    }
}

/// Which event types to record.
#[derive(Debug, Clone)]
pub struct TraceFilter {
    /// Record decisions.
    pub decisions: bool,
    /// Record propagations.
    pub propagations: bool,
    /// Record conflicts.
    pub conflicts: bool,
    /// Record theory checks.
    pub theory_checks: bool,
    /// Record restarts.
    pub restarts: bool,
    /// Record backtracks.
    pub backtracks: bool,
    /// Record learned clauses.
    pub clause_learned: bool,
    /// Record assertions.
    pub assertions: bool,
}

impl Default for TraceFilter {
    fn default() -> Self {
        Self {
            decisions: true,
            propagations: true,
            conflicts: true,
            theory_checks: true,
            restarts: true,
            backtracks: true,
            clause_learned: true,
            assertions: true,
        }
    }
}

impl TraceFilter {
    /// Filter that records only high-level events (decisions, conflicts, restarts).
    pub fn high_level() -> Self {
        Self {
            decisions: true,
            propagations: false,
            conflicts: true,
            theory_checks: false,
            restarts: true,
            backtracks: true,
            clause_learned: false,
            assertions: true,
        }
    }

    /// Filter that records only conflict-related events.
    pub fn conflicts_only() -> Self {
        Self {
            decisions: false,
            propagations: false,
            conflicts: true,
            theory_checks: false,
            restarts: false,
            backtracks: true,
            clause_learned: true,
            assertions: false,
        }
    }

    /// Check if an event should be recorded.
    pub fn should_record(&self, event: &TraceEvent) -> bool {
        match event {
            TraceEvent::Decision { .. } => self.decisions,
            TraceEvent::Propagation { .. } => self.propagations,
            TraceEvent::Conflict { .. } => self.conflicts,
            TraceEvent::TheoryCheck { .. } => self.theory_checks,
            TraceEvent::Restart { .. } => self.restarts,
            TraceEvent::Backtrack { .. } => self.backtracks,
            TraceEvent::ClauseLearned { .. } => self.clause_learned,
            TraceEvent::AssertionAdded { .. } => self.assertions,
        }
    }
}

/// Configuration for trace recording.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Filter for which events to record.
    pub filter: TraceFilter,
    /// Maximum number of events to store (0 = unlimited).
    pub max_events: usize,
    /// Whether to include timestamps (event index as surrogate).
    pub include_index: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            filter: TraceFilter::default(),
            max_events: 100_000,
            include_index: true,
        }
    }
}

/// Solver tracer that records events.
#[derive(Debug)]
pub struct SolverTracer {
    /// Configuration.
    config: TraceConfig,
    /// Recorded events.
    events: Vec<TraceEvent>,
    /// Total events seen (including filtered/dropped).
    total_seen: u64,
    /// Whether recording is enabled.
    enabled: bool,
}

impl SolverTracer {
    /// Create a new tracer with the given configuration.
    pub fn new(config: TraceConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            total_seen: 0,
            enabled: true,
        }
    }

    /// Create a tracer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(TraceConfig::default())
    }

    /// Enable or disable recording.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if recording is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an event (subject to filter and max_events).
    pub fn record(&mut self, event: TraceEvent) {
        self.total_seen += 1;

        if !self.enabled {
            return;
        }

        if !self.config.filter.should_record(&event) {
            return;
        }

        if self.config.max_events > 0 && self.events.len() >= self.config.max_events {
            return;
        }

        self.events.push(event);
    }

    /// Get all recorded events.
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Get the number of recorded events.
    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    /// Get the total number of events seen (including filtered).
    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    /// Clear all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
        self.total_seen = 0;
    }

    /// Format all recorded events as a human-readable trace.
    pub fn format_trace(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Solver Trace ({} events, {} total seen) ===\n",
            self.events.len(),
            self.total_seen
        ));

        for (i, event) in self.events.iter().enumerate() {
            if self.config.include_index {
                out.push_str(&format!("[{:>6}] {}\n", i, event.format()));
            } else {
                out.push_str(&event.format());
                out.push('\n');
            }
        }

        out
    }

    /// Write all recorded events as JSON.
    pub fn write_trace_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"total_events_seen\": {},\n", self.total_seen));
        out.push_str(&format!("  \"recorded_events\": {},\n", self.events.len()));
        out.push_str("  \"events\": [\n");

        for (i, event) in self.events.iter().enumerate() {
            out.push_str("    ");
            out.push_str(&event.to_json());
            if i + 1 < self.events.len() {
                out.push(',');
            }
            out.push('\n');
        }

        out.push_str("  ]\n");
        out.push_str("}\n");
        out
    }

    /// Get events of a specific type.
    pub fn events_of_type(&self, event_type: &str) -> Vec<&TraceEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type() == event_type)
            .collect()
    }

    /// Count events of a specific type.
    pub fn count_events_of_type(&self, event_type: &str) -> usize {
        self.events
            .iter()
            .filter(|e| e.event_type() == event_type)
            .count()
    }

    /// Get the configuration.
    pub fn config(&self) -> &TraceConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_record_decision() {
        let mut tracer = SolverTracer::with_defaults();
        tracer.record(TraceEvent::Decision {
            var: 1,
            value: true,
            level: 0,
        });
        assert_eq!(tracer.num_events(), 1);
        assert_eq!(tracer.total_seen(), 1);
    }

    #[test]
    fn test_tracer_filter() {
        let config = TraceConfig {
            filter: TraceFilter::conflicts_only(),
            max_events: 1000,
            include_index: true,
        };
        let mut tracer = SolverTracer::new(config);

        tracer.record(TraceEvent::Decision {
            var: 1,
            value: true,
            level: 0,
        });
        tracer.record(TraceEvent::Conflict {
            conflicting_clause: 5,
            learned_clause: Some(10),
            learned_size: Some(3),
        });
        tracer.record(TraceEvent::Propagation {
            literal: 2,
            reason_clause: 3,
        });

        // Only conflict should be recorded
        assert_eq!(tracer.num_events(), 1);
        assert_eq!(tracer.total_seen(), 3);
        assert_eq!(tracer.events()[0].event_type(), "conflict");
    }

    #[test]
    fn test_tracer_max_events() {
        let config = TraceConfig {
            filter: TraceFilter::default(),
            max_events: 2,
            include_index: false,
        };
        let mut tracer = SolverTracer::new(config);

        for i in 0..5 {
            tracer.record(TraceEvent::Decision {
                var: i,
                value: true,
                level: i,
            });
        }

        assert_eq!(tracer.num_events(), 2);
        assert_eq!(tracer.total_seen(), 5);
    }

    #[test]
    fn test_tracer_disabled() {
        let mut tracer = SolverTracer::with_defaults();
        tracer.set_enabled(false);
        tracer.record(TraceEvent::Decision {
            var: 1,
            value: true,
            level: 0,
        });
        assert_eq!(tracer.num_events(), 0);
        assert_eq!(tracer.total_seen(), 1);
    }

    #[test]
    fn test_format_trace_text() {
        let mut tracer = SolverTracer::with_defaults();
        tracer.record(TraceEvent::Decision {
            var: 1,
            value: true,
            level: 0,
        });
        tracer.record(TraceEvent::Propagation {
            literal: -2,
            reason_clause: 3,
        });
        tracer.record(TraceEvent::Conflict {
            conflicting_clause: 5,
            learned_clause: Some(10),
            learned_size: Some(2),
        });
        tracer.record(TraceEvent::TheoryCheck {
            theory: "EUF".to_string(),
            result: "consistent".to_string(),
            time_us: 42,
        });
        tracer.record(TraceEvent::Restart {
            reason: "glucose".to_string(),
            new_strategy: None,
        });
        tracer.record(TraceEvent::Backtrack {
            from_level: 3,
            to_level: 1,
        });

        let text = tracer.format_trace();
        assert!(text.contains("DECIDE  v1 = true @ level 0"));
        assert!(text.contains("PROP    x2 = F (reason: clause #3)"));
        assert!(text.contains("CONFLICT clause #5"));
        assert!(text.contains("THEORY  EUF -> consistent (42us)"));
        assert!(text.contains("RESTART reason=glucose"));
        assert!(text.contains("BACKTRACK level 3 -> 1"));
    }

    #[test]
    fn test_write_trace_json() {
        let mut tracer = SolverTracer::with_defaults();
        tracer.record(TraceEvent::Decision {
            var: 1,
            value: true,
            level: 0,
        });
        tracer.record(TraceEvent::Conflict {
            conflicting_clause: 5,
            learned_clause: None,
            learned_size: None,
        });

        let json = tracer.write_trace_json();
        assert!(json.contains("\"total_events_seen\": 2"));
        assert!(json.contains("\"recorded_events\": 2"));
        assert!(json.contains("\"type\":\"decision\""));
        assert!(json.contains("\"type\":\"conflict\""));
        assert!(json.contains("\"learned_clause\":null"));
    }

    #[test]
    fn test_events_of_type() {
        let mut tracer = SolverTracer::with_defaults();
        tracer.record(TraceEvent::Decision {
            var: 1,
            value: true,
            level: 0,
        });
        tracer.record(TraceEvent::Decision {
            var: 2,
            value: false,
            level: 1,
        });
        tracer.record(TraceEvent::Conflict {
            conflicting_clause: 5,
            learned_clause: None,
            learned_size: None,
        });

        let decisions = tracer.events_of_type("decision");
        assert_eq!(decisions.len(), 2);

        let conflicts = tracer.count_events_of_type("conflict");
        assert_eq!(conflicts, 1);
    }

    #[test]
    fn test_tracer_clear() {
        let mut tracer = SolverTracer::with_defaults();
        tracer.record(TraceEvent::Decision {
            var: 1,
            value: true,
            level: 0,
        });
        assert_eq!(tracer.num_events(), 1);
        tracer.clear();
        assert_eq!(tracer.num_events(), 0);
        assert_eq!(tracer.total_seen(), 0);
    }

    #[test]
    fn test_event_json_escaping() {
        let event = TraceEvent::AssertionAdded {
            index: 0,
            description: "x > 0 AND y = \"hello\"".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains(r#"\"hello\""#));
    }

    #[test]
    fn test_clause_learned_event() {
        let mut tracer = SolverTracer::with_defaults();
        tracer.record(TraceEvent::ClauseLearned {
            clause_id: 42,
            num_literals: 5,
            glue: 3,
        });
        let text = tracer.format_trace();
        assert!(text.contains("LEARNED clause #42 (lits=5, glue=3)"));

        let json = tracer.write_trace_json();
        assert!(json.contains("\"clause_id\":42"));
    }

    #[test]
    fn test_high_level_filter() {
        let config = TraceConfig {
            filter: TraceFilter::high_level(),
            max_events: 1000,
            include_index: true,
        };
        let mut tracer = SolverTracer::new(config);

        tracer.record(TraceEvent::Decision { var: 1, value: true, level: 0 });
        tracer.record(TraceEvent::Propagation { literal: 2, reason_clause: 1 });
        tracer.record(TraceEvent::Conflict {
            conflicting_clause: 3,
            learned_clause: None,
            learned_size: None,
        });
        tracer.record(TraceEvent::TheoryCheck {
            theory: "LRA".to_string(),
            result: "ok".to_string(),
            time_us: 10,
        });

        // Decision, Conflict should be recorded; Propagation, TheoryCheck filtered
        assert_eq!(tracer.num_events(), 2);
    }
}
