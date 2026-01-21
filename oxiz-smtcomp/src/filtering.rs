//! Benchmark filtering by expected status and other criteria
//!
//! This module provides functionality to filter benchmarks based on
//! expected status, logic, difficulty, and other properties.

use crate::benchmark::{BenchmarkStatus, SingleResult};
use crate::loader::{BenchmarkMeta, ExpectedStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Filter criteria for benchmarks
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilterCriteria {
    /// Filter by expected status
    pub expected_status: Option<ExpectedStatusFilter>,
    /// Filter by logic(s)
    pub logics: Option<HashSet<String>>,
    /// Exclude logics
    pub exclude_logics: Option<HashSet<String>>,
    /// Filter by category
    pub categories: Option<HashSet<String>>,
    /// Minimum file size in bytes
    pub min_file_size: Option<u64>,
    /// Maximum file size in bytes
    pub max_file_size: Option<u64>,
    /// Path pattern (glob-like)
    pub path_pattern: Option<String>,
    /// Exclude path pattern
    pub exclude_path_pattern: Option<String>,
}

/// Filter options for expected status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedStatusFilter {
    /// Only SAT benchmarks
    Sat,
    /// Only UNSAT benchmarks
    Unsat,
    /// Only unknown benchmarks
    Unknown,
    /// SAT or UNSAT (not unknown)
    Decided,
    /// Has any expected status
    Any,
}

impl FilterCriteria {
    /// Create empty filter (matches all)
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by expected status
    #[must_use]
    pub fn with_expected_status(mut self, status: ExpectedStatusFilter) -> Self {
        self.expected_status = Some(status);
        self
    }

    /// Filter by logic
    #[must_use]
    pub fn with_logic(mut self, logic: impl Into<String>) -> Self {
        self.logics
            .get_or_insert_with(HashSet::new)
            .insert(logic.into());
        self
    }

    /// Filter by multiple logics
    #[must_use]
    pub fn with_logics(mut self, logics: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let set = self.logics.get_or_insert_with(HashSet::new);
        for logic in logics {
            set.insert(logic.into());
        }
        self
    }

    /// Exclude a logic
    #[must_use]
    pub fn exclude_logic(mut self, logic: impl Into<String>) -> Self {
        self.exclude_logics
            .get_or_insert_with(HashSet::new)
            .insert(logic.into());
        self
    }

    /// Filter by category
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.categories
            .get_or_insert_with(HashSet::new)
            .insert(category.into());
        self
    }

    /// Set file size range
    #[must_use]
    pub fn with_file_size_range(mut self, min: Option<u64>, max: Option<u64>) -> Self {
        self.min_file_size = min;
        self.max_file_size = max;
        self
    }

    /// Set path pattern
    #[must_use]
    pub fn with_path_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.path_pattern = Some(pattern.into());
        self
    }

    /// Set exclude path pattern
    #[must_use]
    pub fn exclude_path_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.exclude_path_pattern = Some(pattern.into());
        self
    }

    /// Check if a benchmark matches the filter
    #[must_use]
    pub fn matches(&self, meta: &BenchmarkMeta) -> bool {
        // Check expected status
        if let Some(ref status_filter) = self.expected_status {
            let status_matches = matches!(
                (status_filter, meta.expected_status),
                (ExpectedStatusFilter::Sat, Some(ExpectedStatus::Sat))
                    | (ExpectedStatusFilter::Unsat, Some(ExpectedStatus::Unsat))
                    | (ExpectedStatusFilter::Unknown, Some(ExpectedStatus::Unknown))
                    | (ExpectedStatusFilter::Unknown, None)
                    | (ExpectedStatusFilter::Decided, Some(ExpectedStatus::Sat))
                    | (ExpectedStatusFilter::Decided, Some(ExpectedStatus::Unsat))
                    | (ExpectedStatusFilter::Any, Some(_))
            );
            if !status_matches {
                return false;
            }
        }

        // Check logic inclusion
        if let Some(ref logics) = self.logics {
            match &meta.logic {
                Some(logic) if logics.contains(logic) => {}
                _ => return false,
            }
        }

        // Check logic exclusion
        if let Some(ref exclude) = self.exclude_logics
            && let Some(ref logic) = meta.logic
            && exclude.contains(logic)
        {
            return false;
        }

        // Check category
        if let Some(ref categories) = self.categories {
            match &meta.category {
                Some(cat) if categories.contains(cat) => {}
                _ => return false,
            }
        }

        // Check file size
        if let Some(min) = self.min_file_size
            && meta.file_size < min
        {
            return false;
        }
        if let Some(max) = self.max_file_size
            && meta.file_size > max
        {
            return false;
        }

        // Check path pattern
        if let Some(ref pattern) = self.path_pattern {
            let path_str = meta.path.to_string_lossy();
            if !simple_glob_match(pattern, &path_str) {
                return false;
            }
        }

        // Check exclude path pattern
        if let Some(ref pattern) = self.exclude_path_pattern {
            let path_str = meta.path.to_string_lossy();
            if simple_glob_match(pattern, &path_str) {
                return false;
            }
        }

        true
    }
}

/// Simple glob-like pattern matching
fn simple_glob_match(pattern: &str, text: &str) -> bool {
    // Simple implementation: only supports * as wildcard
    if pattern.is_empty() {
        return text.is_empty();
    }

    let parts: Vec<&str> = pattern.split('*').collect();

    if parts.len() == 1 {
        // No wildcards
        return pattern == text;
    }

    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if i == 0 {
            // First part must be at the start
            if !text.starts_with(part) {
                return false;
            }
            pos = part.len();
        } else if i == parts.len() - 1 {
            // Last part must be at the end
            if !text[pos..].ends_with(part) {
                return false;
            }
        } else {
            // Middle parts must exist somewhere
            match text[pos..].find(part) {
                Some(idx) => pos = pos + idx + part.len(),
                None => return false,
            }
        }
    }

    true
}

/// Filter benchmark metadata
pub fn filter_benchmarks(
    benchmarks: &[BenchmarkMeta],
    criteria: &FilterCriteria,
) -> Vec<BenchmarkMeta> {
    benchmarks
        .iter()
        .filter(|b| criteria.matches(b))
        .cloned()
        .collect()
}

/// Result filter criteria
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResultFilterCriteria {
    /// Filter by actual status
    pub status: Option<HashSet<BenchmarkStatus>>,
    /// Filter by correctness
    pub correct: Option<bool>,
    /// Minimum time in seconds
    pub min_time_secs: Option<f64>,
    /// Maximum time in seconds
    pub max_time_secs: Option<f64>,
    /// Filter by logic
    pub logics: Option<HashSet<String>>,
    /// Only sound results
    pub sound_only: bool,
}

impl ResultFilterCriteria {
    /// Create empty filter
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by status
    #[must_use]
    pub fn with_status(mut self, status: BenchmarkStatus) -> Self {
        self.status.get_or_insert_with(HashSet::new).insert(status);
        self
    }

    /// Filter by statuses
    #[must_use]
    pub fn with_statuses(mut self, statuses: impl IntoIterator<Item = BenchmarkStatus>) -> Self {
        let set = self.status.get_or_insert_with(HashSet::new);
        for status in statuses {
            set.insert(status);
        }
        self
    }

    /// Filter by correctness
    #[must_use]
    pub fn with_correct(mut self, correct: bool) -> Self {
        self.correct = Some(correct);
        self
    }

    /// Filter by time range
    #[must_use]
    pub fn with_time_range(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.min_time_secs = min;
        self.max_time_secs = max;
        self
    }

    /// Only sound results
    #[must_use]
    pub fn sound_only(mut self) -> Self {
        self.sound_only = true;
        self
    }

    /// Check if a result matches the filter
    #[must_use]
    pub fn matches(&self, result: &SingleResult) -> bool {
        // Check status
        if let Some(ref statuses) = self.status
            && !statuses.contains(&result.status)
        {
            return false;
        }

        // Check correctness
        if let Some(expected_correct) = self.correct {
            match result.correct {
                Some(actual) if actual == expected_correct => {}
                None if !expected_correct => {} // None counts as "not correct"
                _ => return false,
            }
        }

        // Check time
        let time_secs = result.time.as_secs_f64();
        if let Some(min) = self.min_time_secs
            && time_secs < min
        {
            return false;
        }
        if let Some(max) = self.max_time_secs
            && time_secs > max
        {
            return false;
        }

        // Check logic
        if let Some(ref logics) = self.logics {
            match &result.logic {
                Some(logic) if logics.contains(logic) => {}
                _ => return false,
            }
        }

        // Check soundness
        if self.sound_only && result.correct == Some(false) {
            return false;
        }

        true
    }
}

/// Filter results
pub fn filter_results(
    results: &[SingleResult],
    criteria: &ResultFilterCriteria,
) -> Vec<SingleResult> {
    results
        .iter()
        .filter(|r| criteria.matches(r))
        .cloned()
        .collect()
}

/// Predefined filter presets
pub mod presets {
    use super::*;

    /// Only SAT benchmarks
    #[must_use]
    pub fn sat_only() -> FilterCriteria {
        FilterCriteria::new().with_expected_status(ExpectedStatusFilter::Sat)
    }

    /// Only UNSAT benchmarks
    #[must_use]
    pub fn unsat_only() -> FilterCriteria {
        FilterCriteria::new().with_expected_status(ExpectedStatusFilter::Unsat)
    }

    /// Decided (SAT or UNSAT) benchmarks
    #[must_use]
    pub fn decided_only() -> FilterCriteria {
        FilterCriteria::new().with_expected_status(ExpectedStatusFilter::Decided)
    }

    /// QF (quantifier-free) logics only
    #[must_use]
    pub fn quantifier_free() -> FilterCriteria {
        FilterCriteria::new().with_path_pattern("*QF_*")
    }

    /// Small benchmarks (< 1MB)
    #[must_use]
    pub fn small_benchmarks() -> FilterCriteria {
        FilterCriteria::new().with_file_size_range(None, Some(1024 * 1024))
    }

    /// Solved results only
    #[must_use]
    pub fn solved_only() -> ResultFilterCriteria {
        ResultFilterCriteria::new().with_statuses([BenchmarkStatus::Sat, BenchmarkStatus::Unsat])
    }

    /// Timeout results only
    #[must_use]
    pub fn timeouts_only() -> ResultFilterCriteria {
        ResultFilterCriteria::new().with_status(BenchmarkStatus::Timeout)
    }

    /// Correct results only
    #[must_use]
    pub fn correct_only() -> ResultFilterCriteria {
        ResultFilterCriteria::new().with_correct(true)
    }

    /// Fast solutions (< 1 second)
    #[must_use]
    pub fn fast_solutions() -> ResultFilterCriteria {
        ResultFilterCriteria::new()
            .with_statuses([BenchmarkStatus::Sat, BenchmarkStatus::Unsat])
            .with_time_range(None, Some(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_meta(
        path: &str,
        logic: Option<&str>,
        expected: Option<ExpectedStatus>,
        size: u64,
    ) -> BenchmarkMeta {
        BenchmarkMeta {
            path: PathBuf::from(path),
            logic: logic.map(String::from),
            expected_status: expected,
            file_size: size,
            category: None,
        }
    }

    fn make_result(status: BenchmarkStatus, time_ms: u64, logic: Option<&str>) -> SingleResult {
        let meta = make_meta("/tmp/test.smt2", logic, Some(ExpectedStatus::Sat), 100);
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    #[test]
    fn test_simple_glob_match() {
        assert!(simple_glob_match("*.smt2", "test.smt2"));
        assert!(simple_glob_match("test*", "test.smt2"));
        assert!(simple_glob_match("*test*", "my_test_file.smt2"));
        assert!(!simple_glob_match("*.smt2", "test.txt"));
        assert!(simple_glob_match("QF_*", "QF_LIA"));
    }

    #[test]
    fn test_filter_by_expected_status() {
        let sat = make_meta("/tmp/sat.smt2", None, Some(ExpectedStatus::Sat), 100);
        let unsat = make_meta("/tmp/unsat.smt2", None, Some(ExpectedStatus::Unsat), 100);
        let unknown = make_meta("/tmp/unknown.smt2", None, None, 100);

        let sat_filter = FilterCriteria::new().with_expected_status(ExpectedStatusFilter::Sat);
        assert!(sat_filter.matches(&sat));
        assert!(!sat_filter.matches(&unsat));
        assert!(!sat_filter.matches(&unknown));

        let decided_filter =
            FilterCriteria::new().with_expected_status(ExpectedStatusFilter::Decided);
        assert!(decided_filter.matches(&sat));
        assert!(decided_filter.matches(&unsat));
        assert!(!decided_filter.matches(&unknown));
    }

    #[test]
    fn test_filter_by_logic() {
        let qf_lia = make_meta("/tmp/test.smt2", Some("QF_LIA"), None, 100);
        let qf_bv = make_meta("/tmp/test.smt2", Some("QF_BV"), None, 100);
        let no_logic = make_meta("/tmp/test.smt2", None, None, 100);

        let filter = FilterCriteria::new().with_logic("QF_LIA");
        assert!(filter.matches(&qf_lia));
        assert!(!filter.matches(&qf_bv));
        assert!(!filter.matches(&no_logic));

        let multi_filter = FilterCriteria::new().with_logics(["QF_LIA", "QF_BV"]);
        assert!(multi_filter.matches(&qf_lia));
        assert!(multi_filter.matches(&qf_bv));
    }

    #[test]
    fn test_filter_by_file_size() {
        let small = make_meta("/tmp/small.smt2", None, None, 100);
        let large = make_meta("/tmp/large.smt2", None, None, 10_000_000);

        let filter = FilterCriteria::new().with_file_size_range(None, Some(1_000_000));
        assert!(filter.matches(&small));
        assert!(!filter.matches(&large));
    }

    #[test]
    fn test_result_filter_by_status() {
        let sat = make_result(BenchmarkStatus::Sat, 100, None);
        let timeout = make_result(BenchmarkStatus::Timeout, 60000, None);

        let filter = ResultFilterCriteria::new()
            .with_statuses([BenchmarkStatus::Sat, BenchmarkStatus::Unsat]);
        assert!(filter.matches(&sat));
        assert!(!filter.matches(&timeout));
    }

    #[test]
    fn test_result_filter_by_time() {
        let fast = make_result(BenchmarkStatus::Sat, 100, None);
        let slow = make_result(BenchmarkStatus::Sat, 5000, None);

        let filter = ResultFilterCriteria::new().with_time_range(None, Some(1.0));
        assert!(filter.matches(&fast));
        assert!(!filter.matches(&slow));
    }

    #[test]
    fn test_presets() {
        let sat = make_meta(
            "/tmp/sat.smt2",
            Some("QF_LIA"),
            Some(ExpectedStatus::Sat),
            100,
        );
        let unsat = make_meta(
            "/tmp/unsat.smt2",
            Some("QF_LIA"),
            Some(ExpectedStatus::Unsat),
            100,
        );

        assert!(presets::sat_only().matches(&sat));
        assert!(!presets::sat_only().matches(&unsat));

        assert!(!presets::unsat_only().matches(&sat));
        assert!(presets::unsat_only().matches(&unsat));
    }
}
