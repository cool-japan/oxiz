//! Regression pipeline integration tests
//!
//! These tests verify that the regression detection infrastructure correctly
//! identifies performance regressions, accepts within-threshold drift, and
//! catches newly timed-out benchmarks.

use oxiz_smtcomp::benchmark::{BenchmarkStatus, SingleResult};
use oxiz_smtcomp::loader::{BenchmarkMeta, ExpectedStatus};
use oxiz_smtcomp::regression::{RegressionConfig, RegressionDetector, RegressionType};
use std::path::PathBuf;
use std::time::Duration;

/// Create a `BenchmarkMeta` for testing with the given file name
fn make_meta(name: &str) -> BenchmarkMeta {
    BenchmarkMeta {
        path: PathBuf::from(format!("/tmp/{}.smt2", name)),
        logic: Some("QF_LIA".to_string()),
        expected_status: Some(ExpectedStatus::Sat),
        file_size: 256,
        category: None,
        structural_features: None,
    }
}

/// Create a `SingleResult` with specified status and solve time
fn make_result(name: &str, status: BenchmarkStatus, time_ms: u64) -> SingleResult {
    let meta = make_meta(name);
    SingleResult::new(&meta, status, Duration::from_millis(time_ms))
}

/// Test: a 20% slowdown is detected as a regression when the threshold is 10%.
///
/// The `TotalTimeRegression` finding is raised when total time increases by more
/// than `time_threshold_pct`. With baseline at 1000 ms and current at 1200 ms the
/// increase is 20%, which exceeds the 10% threshold.
#[test]
fn test_detects_injected_slowdown() {
    // Baseline: two benchmarks totalling 1000 ms
    let baseline = vec![
        make_result("alpha", BenchmarkStatus::Sat, 500),
        make_result("beta", BenchmarkStatus::Unsat, 500),
    ];

    // Current: same benchmarks but 20% slower (1200 ms total)
    let current = vec![
        make_result("alpha", BenchmarkStatus::Sat, 600),
        make_result("beta", BenchmarkStatus::Unsat, 600),
    ];

    let config = RegressionConfig::default().with_time_threshold(10.0);
    let detector = RegressionDetector::new(config);
    let analysis = detector.analyze(&baseline, &current);

    assert!(
        analysis.has_regressions,
        "Expected regression to be flagged for a 20% slowdown"
    );
    assert!(
        analysis
            .findings
            .iter()
            .any(|f| f.regression_type == RegressionType::TotalTimeRegression),
        "Expected TotalTimeRegression finding"
    );
}

/// Test: a 5% drift is accepted when the threshold is 10%.
///
/// Both benchmarks slow down by 5% (from 1000 ms to 1050 ms total).  Because
/// 5% < 10% threshold no regression should be raised.
#[test]
fn test_accepts_within_threshold_drift() {
    // Baseline: 1000 ms total
    let baseline = vec![
        make_result("gamma", BenchmarkStatus::Sat, 500),
        make_result("delta", BenchmarkStatus::Unsat, 500),
    ];

    // Current: 1050 ms total — exactly 5% increase, below 10% threshold
    let current = vec![
        make_result("gamma", BenchmarkStatus::Sat, 525),
        make_result("delta", BenchmarkStatus::Unsat, 525),
    ];

    let config = RegressionConfig::default().with_time_threshold(10.0);
    let detector = RegressionDetector::new(config);
    let analysis = detector.analyze(&baseline, &current);

    // No time regression should be flagged for 5% drift under 10% threshold
    let has_time_regression = analysis
        .findings
        .iter()
        .any(|f| f.regression_type == RegressionType::TotalTimeRegression);

    assert!(
        !has_time_regression,
        "5% drift should not be flagged as a regression with a 10% threshold"
    );
}

/// Test: a benchmark that previously solved now times out is flagged.
///
/// The `NewTimeout` regression type is raised for each benchmark that was
/// previously `Sat`/`Unsat` but is now `Timeout`.  The analysis must report
/// `has_regressions` and include at least one `NewTimeout` finding.
#[test]
fn test_catches_new_timeout() {
    // Baseline: benchmark "epsilon" solved in 200 ms
    let baseline = vec![
        make_result("epsilon", BenchmarkStatus::Sat, 200),
        make_result("zeta", BenchmarkStatus::Sat, 300),
    ];

    // Current: "epsilon" now times out; "zeta" still solves
    let epsilon_meta = make_meta("epsilon");
    let epsilon_timeout = SingleResult::new(
        &epsilon_meta,
        BenchmarkStatus::Timeout,
        Duration::from_secs(60),
    );
    let current = vec![
        epsilon_timeout,
        make_result("zeta", BenchmarkStatus::Sat, 310),
    ];

    // Allow zero new timeouts so the single timeout triggers a regression
    let config = RegressionConfig::default().with_max_new_timeouts(0);
    let detector = RegressionDetector::new(config);
    let analysis = detector.analyze(&baseline, &current);

    assert!(
        analysis.has_regressions,
        "Expected regression when a previously solved benchmark times out"
    );
    assert!(
        analysis
            .findings
            .iter()
            .any(|f| f.regression_type == RegressionType::NewTimeout),
        "Expected NewTimeout finding"
    );
}
