//! Error types for theory solvers
//!
//! This module defines a comprehensive error type hierarchy for
//! all theory solvers, providing better error handling and diagnostics.

use oxiz_core::ast::TermId;
use std::fmt;

/// Result type for theory solver operations
pub type TheoryResult<T> = Result<T, TheoryError>;

/// Errors that can occur during theory solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TheoryError {
    /// Resource limit exceeded (pivots, depth, time, etc.)
    ResourceLimitExceeded(ResourceLimit),

    /// Numerical issue (overflow, underflow, precision loss)
    NumericalIssue(String),

    /// Unsupported operation or constraint
    UnsupportedOperation(String),

    /// Invalid constraint or input
    InvalidConstraint(String),

    /// Conflict detected (with explanation)
    Conflict(ConflictInfo),

    /// Timeout exceeded
    Timeout,

    /// Internal solver error (should not normally occur)
    Internal(String),
}

/// Resource limit types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceLimit {
    /// Maximum simplex pivots exceeded
    MaxPivots,

    /// Maximum branch-and-bound depth exceeded
    MaxDepth,

    /// Maximum cutting planes exceeded
    MaxCuts,

    /// Maximum lemma cache size exceeded
    MaxLemmas,

    /// Memory limit exceeded
    Memory,

    /// Generic iteration limit
    MaxIterations,
}

/// Information about a conflict
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictInfo {
    /// Theory that detected the conflict
    pub theory: String,

    /// Description of the conflict
    pub description: String,

    /// Terms involved in the conflict
    pub terms: Vec<TermId>,

    /// Conflict clause (if available)
    pub clause: Vec<TermId>,
}

impl ConflictInfo {
    /// Create a new conflict info
    #[must_use]
    pub fn new(theory: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            theory: theory.into(),
            description: description.into(),
            terms: Vec::new(),
            clause: Vec::new(),
        }
    }

    /// Add a term to the conflict
    pub fn add_term(&mut self, term: TermId) {
        self.terms.push(term);
    }

    /// Set the conflict clause
    pub fn set_clause(&mut self, clause: Vec<TermId>) {
        self.clause = clause;
    }
}

impl fmt::Display for TheoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceLimitExceeded(limit) => {
                write!(f, "Resource limit exceeded: {limit:?}")
            }
            Self::NumericalIssue(msg) => write!(f, "Numerical issue: {msg}"),
            Self::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {msg}"),
            Self::InvalidConstraint(msg) => write!(f, "Invalid constraint: {msg}"),
            Self::Conflict(info) => {
                write!(
                    f,
                    "Conflict in {}: {} (involving {} terms)",
                    info.theory,
                    info.description,
                    info.terms.len()
                )
            }
            Self::Timeout => write!(f, "Solver timeout"),
            Self::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for TheoryError {}

/// Statistics for solver performance
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Number of constraints asserted
    pub constraints_asserted: usize,

    /// Number of theory checks performed
    pub theory_checks: usize,

    /// Number of conflicts detected
    pub conflicts: usize,

    /// Number of propagations
    pub propagations: usize,

    /// Number of lemmas learned
    pub lemmas_learned: usize,

    /// Total pivots performed (for simplex)
    pub total_pivots: usize,

    /// Total cuts generated (for LIA)
    pub total_cuts: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,
}

impl SolverStats {
    /// Create new empty statistics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics to zero
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Cache hit rate (0.0 to 1.0)
    #[must_use]
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Average propagations per conflict
    #[must_use]
    pub fn avg_propagations_per_conflict(&self) -> f64 {
        if self.conflicts == 0 {
            0.0
        } else {
            self.propagations as f64 / self.conflicts as f64
        }
    }
}

impl fmt::Display for SolverStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Solver Statistics:")?;
        writeln!(f, "  Constraints: {}", self.constraints_asserted)?;
        writeln!(f, "  Theory checks: {}", self.theory_checks)?;
        writeln!(f, "  Conflicts: {}", self.conflicts)?;
        writeln!(f, "  Propagations: {}", self.propagations)?;
        writeln!(f, "  Lemmas learned: {}", self.lemmas_learned)?;
        writeln!(f, "  Total pivots: {}", self.total_pivots)?;
        writeln!(f, "  Total cuts: {}", self.total_cuts)?;
        writeln!(f, "  Cache hit rate: {:.2}%", self.cache_hit_rate() * 100.0)?;
        writeln!(
            f,
            "  Avg propagations/conflict: {:.2}",
            self.avg_propagations_per_conflict()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TheoryError::ResourceLimitExceeded(ResourceLimit::MaxPivots);
        assert!(err.to_string().contains("Resource limit exceeded"));

        let err = TheoryError::NumericalIssue("overflow".to_string());
        assert!(err.to_string().contains("Numerical issue"));

        let err = TheoryError::Timeout;
        assert_eq!(err.to_string(), "Solver timeout");
    }

    #[test]
    fn test_conflict_info() {
        let mut conflict = ConflictInfo::new("EUF", "incompatible equalities");
        conflict.add_term(TermId::new(1));
        conflict.add_term(TermId::new(2));
        conflict.set_clause(vec![TermId::new(3), TermId::new(4)]);

        assert_eq!(conflict.theory, "EUF");
        assert_eq!(conflict.terms.len(), 2);
        assert_eq!(conflict.clause.len(), 2);
    }

    #[test]
    fn test_solver_stats() {
        let mut stats = SolverStats::new();
        assert_eq!(stats.constraints_asserted, 0);
        assert_eq!(stats.cache_hit_rate(), 0.0);

        stats.cache_hits = 80;
        stats.cache_misses = 20;
        assert!((stats.cache_hit_rate() - 0.8).abs() < 0.001);

        stats.propagations = 100;
        stats.conflicts = 10;
        assert!((stats.avg_propagations_per_conflict() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = SolverStats::new();
        stats.conflicts = 10;
        stats.propagations = 100;
        stats.reset();
        assert_eq!(stats.conflicts, 0);
        assert_eq!(stats.propagations, 0);
    }

    #[test]
    fn test_stats_display() {
        let mut stats = SolverStats::new();
        stats.constraints_asserted = 10;
        stats.conflicts = 5;
        let display = format!("{stats}");
        assert!(display.contains("Solver Statistics"));
        assert!(display.contains("Constraints: 10"));
    }
}
