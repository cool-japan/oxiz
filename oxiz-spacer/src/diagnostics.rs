//! Diagnostics, statistics, and tracing for Spacer.
//!
//! This module provides comprehensive diagnostics for debugging and analyzing
//! the PDR/IC3 algorithm execution.
//!
//! Reference: Z3's statistics and verbose output

use crate::chc::PredId;
use crate::frames::LemmaId;
use crate::pdr::SpacerStats;
use crate::pob::PobId;
use oxiz_core::TermId;
use std::time::{Duration, Instant};

/// Tracing level for diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TraceLevel {
    /// No tracing
    Off = 0,
    /// Minimal tracing (major events only)
    Minimal = 1,
    /// Normal tracing (key decisions)
    Normal = 2,
    /// Verbose tracing (all operations)
    Verbose = 3,
    /// Debug tracing (everything including internal state)
    Debug = 4,
}

/// Event types for tracing
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// Solver initialization
    Init,
    /// New frame created
    FrameCreated { level: u32 },
    /// POB created
    PobCreated {
        pob_id: PobId,
        pred: PredId,
        level: u32,
    },
    /// POB blocked
    PobBlocked { pob_id: PobId, lemma_id: LemmaId },
    /// Lemma learned
    LemmaLearned {
        pred: PredId,
        lemma: TermId,
        level: u32,
    },
    /// Lemma propagated
    LemmaPropagated {
        lemma_id: LemmaId,
        from_level: u32,
        to_level: u32,
    },
    /// Fixpoint detected
    FixpointDetected { level: u32 },
    /// Counterexample found
    CounterexampleFound,
    /// SMT query issued
    SmtQuery { result: bool, duration: Duration },
    /// Generalization applied
    Generalization {
        before_size: usize,
        after_size: usize,
    },
    /// Under-approximation hit (GPDR)
    UnderApproxHit { pred: PredId, state: TermId },
}

/// Trace buffer for collecting events
#[derive(Debug)]
pub struct TraceBuffer {
    /// Events collected
    events: Vec<(Instant, TraceEvent)>,
    /// Start time
    start_time: Instant,
    /// Current trace level
    level: TraceLevel,
}

impl TraceBuffer {
    /// Create a new trace buffer
    pub fn new(level: TraceLevel) -> Self {
        Self {
            events: Vec::new(),
            start_time: Instant::now(),
            level,
        }
    }

    /// Record an event
    pub fn record(&mut self, event: TraceEvent) {
        if self.level != TraceLevel::Off {
            self.events.push((Instant::now(), event));
        }
    }

    /// Get all events
    pub fn events(&self) -> &[(Instant, TraceEvent)] {
        &self.events
    }

    /// Get elapsed time since start
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
        self.start_time = Instant::now();
    }

    /// Get the trace level
    pub fn level(&self) -> TraceLevel {
        self.level
    }

    /// Set the trace level
    pub fn set_level(&mut self, level: TraceLevel) {
        self.level = level;
    }

    /// Format trace to string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str(&format!(
            "=== Trace ({} events, {:.3}s) ===\n",
            self.events.len(),
            self.elapsed().as_secs_f64()
        ));

        for (time, event) in &self.events {
            let elapsed = time.duration_since(self.start_time).as_secs_f64();
            result.push_str(&format!("[{:>8.3}s] {:?}\n", elapsed, event));
        }

        result
    }
}

impl Default for TraceBuffer {
    fn default() -> Self {
        Self::new(TraceLevel::Off)
    }
}

/// Invariant formatter for output
pub struct InvariantFormatter;

impl InvariantFormatter {
    /// Format invariants as SMT-LIB2
    pub fn format_smtlib(invariants: &[(PredId, Vec<TermId>)], _pred_names: &[String]) -> String {
        let mut result = String::new();
        result.push_str("; Inductive invariants\n");

        for (pred_id, lemmas) in invariants {
            result.push_str(&format!("; Predicate {}\n", pred_id.raw()));
            for (i, _lemma) in lemmas.iter().enumerate() {
                result.push_str(&format!(";   Lemma {}: <formula>\n", i));
                // Full implementation would format the actual term
            }
        }

        result
    }

    /// Format invariants as readable text
    pub fn format_text(invariants: &[(PredId, Vec<TermId>)], pred_names: &[String]) -> String {
        let mut result = String::new();
        result.push_str("Inductive Invariants:\n");
        result.push_str("====================\n\n");

        for (pred_id, lemmas) in invariants {
            let pred_name = pred_names
                .get(pred_id.raw() as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unknown>");

            result.push_str(&format!(
                "Predicate: {} (ID: {})\n",
                pred_name,
                pred_id.raw()
            ));
            result.push_str(&format!("  {} lemma(s):\n", lemmas.len()));

            for (i, lemma) in lemmas.iter().enumerate() {
                result.push_str(&format!("    [{}] Term ID: {}\n", i + 1, lemma.raw()));
            }
            result.push('\n');
        }

        result
    }

    /// Format invariants as JSON
    pub fn format_json(invariants: &[(PredId, Vec<TermId>)]) -> String {
        let mut result = String::new();
        result.push_str("{\n  \"invariants\": [\n");

        for (idx, (pred_id, lemmas)) in invariants.iter().enumerate() {
            if idx > 0 {
                result.push_str(",\n");
            }
            result.push_str(&format!(
                "    {{\n      \"predicate\": {},\n",
                pred_id.raw()
            ));
            result.push_str("      \"lemmas\": [");

            for (i, lemma) in lemmas.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push_str(&lemma.raw().to_string());
            }

            result.push_str("]\n    }");
        }

        result.push_str("\n  ]\n}\n");
        result
    }
}

/// Statistics reporter
pub struct StatsReporter;

impl StatsReporter {
    /// Format statistics as text
    pub fn format_text(stats: &SpacerStats) -> String {
        let mut result = String::new();
        result.push_str("Spacer Statistics:\n");
        result.push_str("==================\n\n");

        result.push_str("Core Metrics:\n");
        result.push_str(&format!("  Frames:           {}\n", stats.num_frames));
        result.push_str(&format!("  Lemmas:           {}\n", stats.num_lemmas));
        result.push_str(&format!("  Inductive:        {}\n", stats.num_inductive));
        result.push_str(&format!("  POBs:             {}\n", stats.num_pobs));
        result.push_str(&format!("  Blocked:          {}\n", stats.num_blocked));
        result.push_str(&format!("  Subsumed:         {}\n", stats.num_subsumed));
        result.push_str(&format!("  Propagations:     {}\n", stats.num_propagations));

        result.push_str("\nSMT Queries:\n");
        result.push_str(&format!("  Total:            {}\n", stats.num_smt_queries));

        result.push_str("\nGeneralization:\n");
        result.push_str(&format!("  MIC attempts:     {}\n", stats.num_mic_attempts));
        result.push_str(&format!(
            "  CTG strengthen:   {}\n",
            stats.num_ctg_strengthenings
        ));

        result.push_str("\nLazy Annotation:\n");
        result.push_str(&format!(
            "  Models deferred:  {}\n",
            stats.num_lazy_models_deferred
        ));
        result.push_str(&format!(
            "  Gen. deferred:    {}\n",
            stats.num_lazy_generalizations_deferred
        ));

        result.push_str("\nGPDR (Under-approximation):\n");
        result.push_str(&format!(
            "  States tracked:   {}\n",
            stats.num_under_approx_states
        ));
        result.push_str(&format!(
            "  Cache hits:       {}\n",
            stats.num_under_approx_hits
        ));
        result.push_str(&format!(
            "  Queries avoided:  {}\n",
            stats.num_under_approx_avoided_queries
        ));

        // Calculate efficiency metrics
        if stats.num_pobs > 0 {
            let block_rate = (stats.num_blocked as f64 / stats.num_pobs as f64) * 100.0;
            result.push_str("\nEfficiency:\n");
            result.push_str(&format!("  Block rate:       {:.1}%\n", block_rate));
        }

        if stats.num_lemmas > 0 {
            let inductive_rate = (stats.num_inductive as f64 / stats.num_lemmas as f64) * 100.0;
            result.push_str(&format!("  Inductive rate:   {:.1}%\n", inductive_rate));
        }

        result
    }

    /// Format statistics as JSON
    pub fn format_json(stats: &SpacerStats) -> String {
        format!(
            r#"{{
  "frames": {},
  "lemmas": {},
  "inductive": {},
  "pobs": {},
  "blocked": {},
  "subsumed": {},
  "propagations": {},
  "smt_queries": {},
  "mic_attempts": {},
  "ctg_strengthenings": {},
  "lazy_models_deferred": {},
  "lazy_generalizations_deferred": {},
  "under_approx_states": {},
  "under_approx_hits": {},
  "under_approx_avoided_queries": {}
}}"#,
            stats.num_frames,
            stats.num_lemmas,
            stats.num_inductive,
            stats.num_pobs,
            stats.num_blocked,
            stats.num_subsumed,
            stats.num_propagations,
            stats.num_smt_queries,
            stats.num_mic_attempts,
            stats.num_ctg_strengthenings,
            stats.num_lazy_models_deferred,
            stats.num_lazy_generalizations_deferred,
            stats.num_under_approx_states,
            stats.num_under_approx_hits,
            stats.num_under_approx_avoided_queries,
        )
    }

    /// Format statistics as CSV
    pub fn format_csv(stats: &SpacerStats) -> String {
        let header = "frames,lemmas,inductive,pobs,blocked,subsumed,propagations,smt_queries,mic_attempts,ctg_strengthenings,lazy_models_deferred,lazy_generalizations_deferred,under_approx_states,under_approx_hits,under_approx_avoided_queries\n";
        let data = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            stats.num_frames,
            stats.num_lemmas,
            stats.num_inductive,
            stats.num_pobs,
            stats.num_blocked,
            stats.num_subsumed,
            stats.num_propagations,
            stats.num_smt_queries,
            stats.num_mic_attempts,
            stats.num_ctg_strengthenings,
            stats.num_lazy_models_deferred,
            stats.num_lazy_generalizations_deferred,
            stats.num_under_approx_states,
            stats.num_under_approx_hits,
            stats.num_under_approx_avoided_queries,
        );
        format!("{}{}", header, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_buffer_creation() {
        let trace = TraceBuffer::new(TraceLevel::Normal);
        assert_eq!(trace.level(), TraceLevel::Normal);
        assert_eq!(trace.events().len(), 0);
    }

    #[test]
    fn test_trace_event_recording() {
        let mut trace = TraceBuffer::new(TraceLevel::Verbose);
        trace.record(TraceEvent::Init);
        trace.record(TraceEvent::FrameCreated { level: 1 });

        assert_eq!(trace.events().len(), 2);
    }

    #[test]
    fn test_trace_off() {
        let mut trace = TraceBuffer::new(TraceLevel::Off);
        trace.record(TraceEvent::Init);

        assert_eq!(trace.events().len(), 0); // No events when off
    }

    #[test]
    fn test_stats_format_text() {
        let stats = SpacerStats {
            num_frames: 10,
            num_lemmas: 20,
            num_pobs: 30,
            ..Default::default()
        };

        let text = StatsReporter::format_text(&stats);
        assert!(text.contains("Frames:"));
        assert!(text.contains("10"));
    }

    #[test]
    fn test_stats_format_json() {
        let stats = SpacerStats {
            num_frames: 5,
            num_lemmas: 15,
            ..Default::default()
        };

        let json = StatsReporter::format_json(&stats);
        assert!(json.contains("\"frames\": 5"));
        assert!(json.contains("\"lemmas\": 15"));
    }

    #[test]
    fn test_invariant_format_text() {
        let pred_id = PredId::new(0);
        let lemmas = vec![TermId::new(1), TermId::new(2)];
        let invariants = vec![(pred_id, lemmas)];
        let pred_names = vec!["Inv".to_string()];

        let text = InvariantFormatter::format_text(&invariants, &pred_names);
        assert!(text.contains("Inv"));
        assert!(text.contains("2 lemma"));
    }

    #[test]
    fn test_invariant_format_json() {
        let pred_id = PredId::new(0);
        let lemmas = vec![TermId::new(10), TermId::new(20)];
        let invariants = vec![(pred_id, lemmas)];

        let json = InvariantFormatter::format_json(&invariants);
        assert!(json.contains("\"predicate\": 0"));
        assert!(json.contains("10"));
        assert!(json.contains("20"));
    }
}
