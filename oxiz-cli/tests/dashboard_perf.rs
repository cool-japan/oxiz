use oxiz_cli::dashboard::perf::render_perf_dashboard;

#[test]
fn renders_empty_history_without_panic() {
    let base = std::env::temp_dir().join(format!(
        "oxiz_dashboard_test_{}",
        std::process::id()
    ));
    let history_dir = base.join("history");
    let output_dir = base.join("output");

    std::fs::create_dir_all(&history_dir).expect("history dir should be created");
    render_perf_dashboard(&history_dir, &output_dir).expect("dashboard render should succeed");

    assert!(output_dir.join("index.html").exists());
}
