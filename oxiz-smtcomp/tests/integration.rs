//! Integration tests for oxiz-smtcomp
//!
//! These tests verify the integration of various components with
//! real or simulated benchmark data.

use oxiz_smtcomp::benchmark::{BenchmarkStatus, RunSummary, Runner, RunnerConfig, SingleResult};
use oxiz_smtcomp::filtering::{
    ExpectedStatusFilter, FilterCriteria, ResultFilterCriteria, filter_benchmarks, filter_results,
};
use oxiz_smtcomp::loader::{Benchmark, BenchmarkMeta, ExpectedStatus};
use oxiz_smtcomp::parallel::{ParallelConfig, ParallelRunner};
use oxiz_smtcomp::plotting::SvgPlot;
use oxiz_smtcomp::regression::RegressionDetector;
use oxiz_smtcomp::reporter::Reporter;
use oxiz_smtcomp::resumption::{CheckpointConfig, ResultLoader, ResultSaver};
use oxiz_smtcomp::sampling::{Sampler, SamplingConfig, SamplingStrategy};
use oxiz_smtcomp::starexec::StarExecWriter;
use oxiz_smtcomp::statistics::{SolverComparison, Statistics, cactus_data, scatter_data};
use oxiz_smtcomp::virtual_best::VirtualBestSolver;

use std::path::PathBuf;
use std::time::Duration;
use tempfile::tempdir;

/// Helper to create test benchmark metadata
fn make_meta(name: &str, logic: &str, expected: Option<ExpectedStatus>) -> BenchmarkMeta {
    BenchmarkMeta {
        path: PathBuf::from(format!("/tmp/{}", name)),
        logic: Some(logic.to_string()),
        expected_status: expected,
        file_size: 100,
        category: None,
    }
}

/// Helper to create test results
fn make_result(name: &str, status: BenchmarkStatus, time_ms: u64, logic: &str) -> SingleResult {
    let meta = make_meta(name, logic, Some(ExpectedStatus::Sat));
    SingleResult::new(&meta, status, Duration::from_millis(time_ms))
}

/// Helper to create a test benchmark with content
fn make_benchmark(name: &str, content: &str) -> Benchmark {
    Benchmark {
        meta: make_meta(name, "QF_LIA", Some(ExpectedStatus::Sat)),
        content: content.to_string(),
    }
}

#[test]
fn test_full_benchmark_workflow() {
    // Create test benchmarks
    let benchmarks = vec![
        make_benchmark(
            "simple_sat.smt2",
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        ),
        make_benchmark(
            "simple_unsat.smt2",
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (and (> x 0) (< x 0)))\n(check-sat)",
        ),
    ];

    // Run benchmarks
    let config = RunnerConfig::new(Duration::from_secs(10));
    let runner = Runner::new(config);
    let results = runner.run_all(&benchmarks);

    assert_eq!(results.len(), 2);

    // Generate summary
    let summary = RunSummary::from_results(&results);
    assert_eq!(summary.total, 2);

    // Generate report
    let reporter = Reporter::text();
    let report = reporter.to_string(&results).unwrap();
    assert!(report.contains("Total benchmarks: 2"));
}

#[test]
fn test_parallel_execution() {
    let benchmarks: Vec<_> = (0..10)
        .map(|i| {
            make_benchmark(
                &format!("bench_{}.smt2", i),
                &format!(
                    "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x {}))\n(check-sat)",
                    i
                ),
            )
        })
        .collect();

    let config = ParallelConfig::new(Duration::from_secs(5)).with_num_threads(2);
    let runner = ParallelRunner::new(config);
    let results = runner.run_all(&benchmarks);

    assert_eq!(results.len(), 10);
}

#[test]
fn test_filtering_workflow() {
    let benchmarks = vec![
        make_meta("sat1.smt2", "QF_LIA", Some(ExpectedStatus::Sat)),
        make_meta("sat2.smt2", "QF_BV", Some(ExpectedStatus::Sat)),
        make_meta("unsat1.smt2", "QF_LIA", Some(ExpectedStatus::Unsat)),
        make_meta("unknown.smt2", "QF_UF", None),
    ];

    // Filter SAT only
    let sat_filter = FilterCriteria::new().with_expected_status(ExpectedStatusFilter::Sat);
    let sat_only = filter_benchmarks(&benchmarks, &sat_filter);
    assert_eq!(sat_only.len(), 2);

    // Filter by logic
    let logic_filter = FilterCriteria::new().with_logic("QF_LIA");
    let qf_lia_only = filter_benchmarks(&benchmarks, &logic_filter);
    assert_eq!(qf_lia_only.len(), 2);

    // Combined filter
    let combined = FilterCriteria::new()
        .with_expected_status(ExpectedStatusFilter::Sat)
        .with_logic("QF_LIA");
    let combined_result = filter_benchmarks(&benchmarks, &combined);
    assert_eq!(combined_result.len(), 1);
}

#[test]
fn test_result_filtering() {
    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Unsat, 200, "QF_LIA"),
        make_result("c.smt2", BenchmarkStatus::Timeout, 60000, "QF_BV"),
        make_result("d.smt2", BenchmarkStatus::Error, 0, "QF_UF"),
    ];

    // Filter solved only
    let solved_filter =
        ResultFilterCriteria::new().with_statuses([BenchmarkStatus::Sat, BenchmarkStatus::Unsat]);
    let solved = filter_results(&results, &solved_filter);
    assert_eq!(solved.len(), 2);

    // Filter by time
    let fast_filter = ResultFilterCriteria::new().with_time_range(None, Some(0.5));
    let fast = filter_results(&results, &fast_filter);
    assert_eq!(fast.len(), 3); // All except 60s timeout
}

#[test]
fn test_statistics_calculation() {
    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
        make_result("c.smt2", BenchmarkStatus::Sat, 300, "QF_LIA"),
    ];

    let stats = Statistics::from_results(&results);

    assert_eq!(stats.count, 3);
    assert_eq!(stats.min_time, Duration::from_millis(100));
    assert_eq!(stats.max_time, Duration::from_millis(300));
}

#[test]
fn test_solver_comparison() {
    let results_a = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
        make_result("c.smt2", BenchmarkStatus::Timeout, 60000, "QF_LIA"),
    ];

    let results_b = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 150, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Timeout, 60000, "QF_LIA"),
        make_result("c.smt2", BenchmarkStatus::Sat, 500, "QF_LIA"),
    ];

    let comparison = SolverComparison::compare("A", &results_a, "B", &results_b);

    assert_eq!(comparison.both_solved, 1); // a.smt2
    assert_eq!(comparison.only_a_solved, 1); // b.smt2
    assert_eq!(comparison.only_b_solved, 1); // c.smt2
}

#[test]
fn test_virtual_best_solver() {
    let results_a = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Timeout, 60000, "QF_LIA"),
    ];

    let results_b = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 500, "QF_LIA"),
    ];

    let mut vbs = VirtualBestSolver::new();
    vbs.add_solver("Solver A", results_a);
    vbs.add_solver("Solver B", results_b);

    let vbs_results = vbs.calculate();
    assert_eq!(vbs_results.len(), 2);

    let stats = vbs.calculate_stats();
    assert_eq!(stats.solved, 2); // VBS solves both

    // Check that VBS picks best for each
    let a_result = vbs_results
        .iter()
        .find(|r| r.path.to_string_lossy().contains("a.smt2"))
        .unwrap();
    assert_eq!(a_result.best_solver, "Solver A"); // A was faster
}

#[test]
fn test_cactus_data_generation() {
    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 300, "QF_LIA"),
        make_result("c.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
        make_result("d.smt2", BenchmarkStatus::Timeout, 60000, "QF_LIA"),
    ];

    let cactus = cactus_data(&results);

    assert_eq!(cactus.len(), 3); // Only solved
    assert_eq!(cactus[0].solved, 1);
    assert_eq!(cactus[1].solved, 2);
    assert_eq!(cactus[2].solved, 3);
    // Should be sorted by time
    assert!(cactus[0].time <= cactus[1].time);
    assert!(cactus[1].time <= cactus[2].time);
}

#[test]
fn test_scatter_data_generation() {
    let results_a = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
    ];

    let results_b = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 150, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 250, "QF_LIA"),
    ];

    let scatter = scatter_data(&results_a, &results_b);
    assert_eq!(scatter.len(), 2);
}

#[test]
fn test_svg_plot_generation() {
    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
        make_result("c.smt2", BenchmarkStatus::Sat, 300, "QF_LIA"),
    ];

    let mut plot = SvgPlot::cactus("Test Plot");
    plot.add_results("Test Solver", &results);

    let svg = plot.to_svg().unwrap();
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Test Plot"));
}

#[test]
fn test_regression_detection() {
    let baseline = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
    ];

    // No regression
    let current_ok = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 90, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 180, "QF_LIA"),
    ];

    let detector = RegressionDetector::default();
    let analysis = detector.analyze(&baseline, &current_ok);
    assert!(!analysis.has_regressions);
    assert!(analysis.passes());

    // With regression (timeout)
    let current_bad = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Timeout, 60000, "QF_LIA"),
    ];

    let analysis = detector.analyze(&baseline, &current_bad);
    assert!(analysis.has_regressions);
}

#[test]
fn test_sampling() {
    let benchmarks: Vec<_> = (0..100)
        .map(|i| {
            let logic = if i % 3 == 0 {
                "QF_LIA"
            } else if i % 3 == 1 {
                "QF_BV"
            } else {
                "QF_UF"
            };
            make_meta(
                &format!("bench_{}.smt2", i),
                logic,
                Some(ExpectedStatus::Sat),
            )
        })
        .collect();

    // Random sampling
    let config = SamplingConfig::with_size(20)
        .with_strategy(SamplingStrategy::Random)
        .with_seed(42);
    let mut sampler = Sampler::new(config);
    let sample = sampler.sample(&benchmarks);
    assert_eq!(sample.len(), 20);

    // Stratified sampling
    let config = SamplingConfig::with_size(30)
        .with_strategy(SamplingStrategy::StratifiedByLogic)
        .with_seed(42);
    let mut sampler = Sampler::new(config);
    let sample = sampler.sample(&benchmarks);

    // Should have all logics represented
    let logics: std::collections::HashSet<_> =
        sample.iter().filter_map(|b| b.logic.as_ref()).collect();
    assert!(logics.len() >= 2);
}

#[test]
fn test_resumption_workflow() {
    let dir = tempdir().unwrap();

    let config = CheckpointConfig::default();
    let mut saver = ResultSaver::new(dir.path(), "test_session", 10, config).unwrap();

    // Save some results
    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Unsat, 200, "QF_LIA"),
    ];

    for result in &results {
        saver.save_result(result).unwrap();
    }
    saver.finalize().unwrap();

    // Load results back
    let loaded = ResultLoader::load(dir.path().join("test_session.jsonl")).unwrap();
    assert_eq!(loaded.len(), 2);

    // Check checkpoint
    let checkpoint =
        ResultLoader::load_checkpoint(dir.path().join("test_session.checkpoint.json")).unwrap();
    assert_eq!(checkpoint.completed, 2);
}

#[test]
fn test_starexec_format() {
    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Unsat, 200, "QF_LIA"),
    ];

    let mut writer = StarExecWriter::new("OxiZ", "default");
    let mut output = Vec::new();
    writer.write_csv(&results, &mut output).unwrap();

    let csv = String::from_utf8(output).unwrap();
    assert!(csv.contains("pair id"));
    assert!(csv.contains("sat"));
    assert!(csv.contains("unsat"));
}

#[test]
fn test_report_formats() {
    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Unsat, 200, "QF_LIA"),
    ];

    // JSON format
    let reporter = Reporter::json();
    let json = reporter.to_string(&results).unwrap();
    assert!(json.contains("\"total\""));
    assert!(json.contains("\"sat\""));

    // CSV format
    let reporter = Reporter::csv();
    let csv = reporter.to_string(&results).unwrap();
    assert!(csv.contains("benchmark,logic,status"));

    // Text format
    let reporter = Reporter::text();
    let text = reporter.to_string(&results).unwrap();
    assert!(text.contains("Total benchmarks:"));

    // SMT-COMP format
    let reporter = Reporter::smtcomp();
    let smtcomp = reporter.to_string(&results).unwrap();
    assert!(smtcomp.contains("sat"));
}

#[test]
fn test_html_report_generation() {
    use oxiz_smtcomp::html_report::{HtmlReportConfig, HtmlReportGenerator};

    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Unsat, 200, "QF_BV"),
        make_result("c.smt2", BenchmarkStatus::Timeout, 60000, "QF_UF"),
    ];

    let config = HtmlReportConfig::new("Test Report").with_solver("TestSolver", "1.0.0");
    let generator = HtmlReportGenerator::new(config);
    let html = generator.generate(&results).unwrap();

    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("Test Report"));
    assert!(html.contains("TestSolver"));
}

#[test]
fn test_dashboard_generation() {
    use oxiz_smtcomp::dashboard::{DashboardConfig, DashboardGenerator};

    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Sat, 200, "QF_LIA"),
    ];

    let config = DashboardConfig::new("Test Dashboard");
    let generator = DashboardGenerator::new(config);
    let html = generator.generate(&results).unwrap();

    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("Test Dashboard"));
    assert!(html.contains("cactusChart"));
}

#[test]
fn test_ci_integration() {
    use oxiz_smtcomp::ci_integration::{CiConfig, CiConfigGenerator, CiSystem, generate_junit_xml};

    let results = vec![
        make_result("a.smt2", BenchmarkStatus::Sat, 100, "QF_LIA"),
        make_result("b.smt2", BenchmarkStatus::Unsat, 200, "QF_LIA"),
    ];

    // Generate JUnit XML
    let xml = generate_junit_xml(&results, "SMT Tests");
    assert!(xml.contains("testsuite"));
    assert!(xml.contains("tests=\"2\""));

    // Generate GitHub Actions config
    let config = CiConfig::new(CiSystem::GitHubActions);
    let generator = CiConfigGenerator::new(config);
    let yaml = generator.generate().unwrap();
    assert!(yaml.contains("name: SMT Benchmark Tests"));

    // Generate shell script
    let config = CiConfig::new(CiSystem::Shell);
    let generator = CiConfigGenerator::new(config);
    let script = generator.generate().unwrap();
    assert!(script.contains("#!/bin/bash"));
}

#[test]
fn test_memory_limits() {
    use oxiz_smtcomp::memory::{MemoryLimit, MemoryMonitor, MemoryUsage};

    // Create limits
    let limit = MemoryLimit::from_gb(4).with_soft_limit(2 * 1024 * 1024 * 1024);

    assert_eq!(limit.hard_limit, 4 * 1024 * 1024 * 1024);
    assert!(!limit.is_unlimited());

    // Test memory usage
    let usage = MemoryUsage::current();
    let _ = usage.format(); // Just ensure it doesn't panic

    // Test monitor
    let mut monitor = MemoryMonitor::new(MemoryLimit::from_gb(16));
    monitor.sample();
    monitor.sample();
    assert_eq!(monitor.samples().len(), 2);
}

#[test]
fn test_model_verification() {
    use oxiz_smtcomp::model_verify::Model;

    // Test model creation
    let mut model = Model::new();
    model.add_bool("x", true);
    model.add_int("y", 42);
    model.add_real("z", "3.14");

    assert_eq!(model.len(), 3);

    let smtlib = model.to_smtlib();
    assert!(smtlib.contains("(model"));
    assert!(smtlib.contains("define-fun x"));
}

#[test]
fn test_end_to_end_benchmark_run() {
    // This test simulates a complete benchmark run workflow

    // 1. Create benchmarks
    let benchmarks = vec![
        make_benchmark(
            "test1.smt2",
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 5))\n(check-sat)",
        ),
        make_benchmark(
            "test2.smt2",
            "(set-logic QF_LIA)\n(declare-const y Int)\n(assert (< y 10))\n(check-sat)",
        ),
    ];

    // 2. Run with parallel runner
    let config = ParallelConfig::new(Duration::from_secs(5));
    let runner = ParallelRunner::new(config);
    let results = runner.run_all(&benchmarks);

    // 3. Calculate statistics
    let summary = RunSummary::from_results(&results);
    let stats = Statistics::from_results(&results);

    // 4. Generate cactus data
    let cactus = cactus_data(&results);

    // 5. Generate reports
    let json_report = Reporter::json().to_string(&results).unwrap();
    let text_report = Reporter::text().to_string(&results).unwrap();

    // 6. Generate visualizations
    let mut plot = SvgPlot::cactus("End-to-End Test");
    plot.add_results("OxiZ", &results);
    let svg = plot.to_svg().unwrap();

    // Verify everything worked
    assert_eq!(results.len(), 2);
    assert!(json_report.contains("total"));
    assert!(text_report.contains("benchmarks"));
    assert!(svg.contains("<svg"));
}
