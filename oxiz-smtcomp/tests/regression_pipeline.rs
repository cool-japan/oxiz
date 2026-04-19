use oxiz_smtcomp::regression::{RegressionConfig, RegressionDetector, RegressionType};
use oxiz_smtcomp::{BenchmarkMeta, BenchmarkStatus, ExpectedStatus, SingleResult};
use std::path::PathBuf;
use std::time::Duration;

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

fn make_result(name: &str, status: BenchmarkStatus, time_ms: u64) -> SingleResult {
    let meta = make_meta(name);
    SingleResult::new(&meta, status, Duration::from_millis(time_ms))
}

#[test]
fn test_detects_injected_slowdown() {
    let baseline = vec![
        make_result("alpha", BenchmarkStatus::Sat, 500),
        make_result("beta", BenchmarkStatus::Unsat, 500),
    ];

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

#[test]
fn test_accepts_within_threshold_drift() {
    let baseline = vec![
        make_result("gamma", BenchmarkStatus::Sat, 500),
        make_result("delta", BenchmarkStatus::Unsat, 500),
    ];

    let current = vec![
        make_result("gamma", BenchmarkStatus::Sat, 525),
        make_result("delta", BenchmarkStatus::Unsat, 525),
    ];

    let config = RegressionConfig::default().with_time_threshold(10.0);
    let detector = RegressionDetector::new(config);
    let analysis = detector.analyze(&baseline, &current);

    let has_time_regression = analysis
        .findings
        .iter()
        .any(|f| f.regression_type == RegressionType::TotalTimeRegression);

    assert!(
        !has_time_regression,
        "5% drift should not be flagged as a regression with a 10% threshold"
    );
}

#[test]
fn test_catches_new_timeout() {
    let baseline = vec![
        make_result("epsilon", BenchmarkStatus::Sat, 200),
        make_result("zeta", BenchmarkStatus::Sat, 300),
    ];

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
