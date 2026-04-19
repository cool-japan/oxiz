//! Integration tests for interactive dashboard features
//!
//! Tests verify that the generated HTML contains the required interactive
//! elements: search input, sortable column headers, WebSocket injection,
//! detail expander JS, and basic structural well-formedness.

use oxiz_smtcomp::benchmark::{BenchmarkStatus, SingleResult};
use oxiz_smtcomp::dashboard::{DashboardConfig, DashboardGenerator};
use oxiz_smtcomp::loader::{BenchmarkMeta, ExpectedStatus};
use std::path::PathBuf;
use std::time::Duration;

/// Build a minimal `SingleResult` for tests.
fn make_result(name: &str, logic: &str, status: BenchmarkStatus, time_ms: u64) -> SingleResult {
    let meta = BenchmarkMeta {
        path: PathBuf::from(format!("/tmp/{}", name)),
        logic: Some(logic.to_string()),
        expected_status: Some(ExpectedStatus::Sat),
        file_size: 200,
        category: None,
        structural_features: None,
    };
    SingleResult::new(&meta, status, Duration::from_millis(time_ms))
}

fn sample_results() -> Vec<SingleResult> {
    vec![
        make_result("alpha.smt2", "QF_LIA", BenchmarkStatus::Sat, 123),
        make_result("beta.smt2", "QF_NRA", BenchmarkStatus::Unsat, 456),
        make_result("gamma.smt2", "QF_BV", BenchmarkStatus::Timeout, 60_000),
        make_result("delta.smt2", "LRA", BenchmarkStatus::Sat, 78),
        make_result("epsilon.smt2", "QF_LRA", BenchmarkStatus::Error, 10),
    ]
}

#[test]
fn test_dashboard_renders_search_input() {
    let config = DashboardConfig::default();
    let generator = DashboardGenerator::new(config);
    let html = generator
        .generate(&sample_results())
        .expect("dashboard generation should succeed");
    assert!(
        html.contains(r#"id="oxiz-search""#),
        "HTML should contain the search input with id='oxiz-search'"
    );
}

#[test]
fn test_dashboard_renders_sort_handlers() {
    let config = DashboardConfig::default();
    let generator = DashboardGenerator::new(config);
    let html = generator
        .generate(&sample_results())
        .expect("dashboard generation should succeed");
    assert!(
        html.contains("data-sort-key="),
        "HTML should contain at least one data-sort-key attribute on a <th>"
    );
}

#[test]
fn test_dashboard_includes_ws_when_configured() {
    let config = DashboardConfig {
        ws_url: Some("ws://localhost:8080".to_string()),
        ..Default::default()
    };
    let generator = DashboardGenerator::new(config);
    let html = generator
        .generate(&sample_results())
        .expect("dashboard generation should succeed");
    assert!(
        html.contains("new WebSocket(\"ws://localhost:8080\")"),
        "HTML should contain the configured WebSocket URL"
    );
}

#[test]
fn test_dashboard_omits_ws_when_unconfigured() {
    let config = DashboardConfig::default();
    let generator = DashboardGenerator::new(config);
    let html = generator
        .generate(&sample_results())
        .expect("dashboard generation should succeed");
    assert!(
        !html.contains("new WebSocket("),
        "HTML should NOT contain WebSocket code when ws_url is None"
    );
}

#[test]
fn test_dashboard_html_well_formed() {
    let config = DashboardConfig::default();
    let generator = DashboardGenerator::new(config);
    let html = generator
        .generate(&sample_results())
        .expect("dashboard generation should succeed");

    assert!(html.contains("<html"), "HTML must open with an <html tag");
    assert!(html.contains("</html>"), "HTML must end with </html>");

    // Expect at least 2 <script> tags:
    //   1. Chart.js CDN <script src="...">
    //   2. Inline data/stats/chart setup
    //   3. Interactive JS block
    let script_opens = html.matches("<script").count();
    assert!(
        script_opens >= 2,
        "Expected at least 2 <script> tags, got {}",
        script_opens
    );
}
