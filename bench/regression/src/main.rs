//! OxiZ Performance Regression Test Runner
//!
//! This binary runs performance benchmarks and compares against a baseline,
//! detecting regressions and producing CI-friendly output.

mod benchmarks;
mod z3_compare;

use anyhow::{Context, Result};
use benchmarks::{BenchmarkCategory, BenchmarkResult, run_all_benchmarks};
use oxiz_smtcomp::loader::BenchmarkMeta;
use oxiz_smtcomp::regression::{RegressionAnalysis, RegressionConfig, RegressionDetector};
use oxiz_smtcomp::{BenchmarkStatus, SingleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;
use z3_compare::{Z3ComparisonReport, compare_with_z3};

/// Baseline performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    /// Version of the baseline
    pub version: String,
    /// Timestamp when baseline was created
    pub timestamp: String,
    /// Benchmark results
    pub benchmarks: HashMap<String, BaselineEntry>,
}

/// Single baseline entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEntry {
    /// Average time in microseconds
    pub avg_time_us: f64,
    /// Category of the benchmark
    pub category: BenchmarkCategory,
}

/// Regression report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Shared regression analysis derived from oxiz-smtcomp
    pub analysis: RegressionAnalysis,
    /// Regression threshold percentage
    pub threshold_percent: f64,
    /// Individual benchmark comparisons
    pub comparisons: Vec<BenchmarkComparison>,
}

/// Comparison of a single benchmark against baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Benchmark name
    pub name: String,
    /// Category
    pub category: BenchmarkCategory,
    /// Current time in microseconds
    pub current_us: f64,
    /// Baseline time in microseconds (if available)
    pub baseline_us: Option<f64>,
    /// Percentage change (positive = slower = regression)
    pub change_percent: Option<f64>,
    /// Whether this is a regression
    pub is_regression: bool,
    /// Status description
    pub status: ComparisonStatus,
}

/// Status of a benchmark comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonStatus {
    /// Performance regressed beyond threshold
    Regression,
    /// Performance improved beyond threshold
    Improvement,
    /// Performance is within threshold
    Unchanged,
    /// No baseline available (new benchmark)
    New,
}

impl RegressionReport {
    fn total_benchmarks(&self) -> usize {
        self.comparisons.len()
    }

    fn regressions(&self) -> usize {
        self.comparisons
            .iter()
            .filter(|comp| comp.status == ComparisonStatus::Regression)
            .count()
    }

    fn improvements(&self) -> usize {
        self.comparisons
            .iter()
            .filter(|comp| comp.status == ComparisonStatus::Improvement)
            .count()
    }

    fn unchanged(&self) -> usize {
        self.comparisons
            .iter()
            .filter(|comp| comp.status == ComparisonStatus::Unchanged)
            .count()
    }

    fn new_benchmarks(&self) -> usize {
        self.comparisons
            .iter()
            .filter(|comp| comp.status == ComparisonStatus::New)
            .count()
    }
}

impl std::fmt::Display for ComparisonStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonStatus::Regression => write!(f, "REGRESSION"),
            ComparisonStatus::Improvement => write!(f, "IMPROVEMENT"),
            ComparisonStatus::Unchanged => write!(f, "OK"),
            ComparisonStatus::New => write!(f, "NEW"),
        }
    }
}

/// Configuration for the regression runner
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to baseline file
    pub baseline_path: PathBuf,
    /// Regression threshold percentage (default: 10%)
    pub threshold_percent: f64,
    /// Whether to update the baseline
    pub update_baseline: bool,
    /// Output format
    pub output_format: OutputFormat,
    /// Whether to run only specific categories
    pub categories: Option<Vec<BenchmarkCategory>>,
    /// Whether to emit a Markdown comment file
    pub markdown: bool,
    /// Whether to compare benchmark timings with Z3
    pub compare_z3: bool,
    /// Check the geomean ratio from history snapshots and exit
    pub check_geomean: bool,
    /// Maximum allowed geomean ratio (default: 1.2)
    pub check_geomean_max: f64,
    /// Directory containing history snapshots for geomean gate
    pub history_dir: PathBuf,
    /// Whether this is a --refresh-baseline invocation (implies update_baseline)
    pub refresh_baseline: bool,
}

/// Output format for the report
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable text
    Text,
    /// JSON for automation
    Json,
    /// GitHub Actions annotations
    GithubActions,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            baseline_path: PathBuf::from("baseline.json"),
            threshold_percent: 10.0,
            update_baseline: false,
            output_format: OutputFormat::Text,
            categories: None,
            markdown: false,
            compare_z3: false,
            check_geomean: false,
            check_geomean_max: 1.2,
            history_dir: PathBuf::from("history"),
            refresh_baseline: false,
        }
    }
}

/// Load baseline from file
fn load_baseline(path: &PathBuf) -> Result<Option<Baseline>> {
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read baseline file: {}", path.display()))?;

    let baseline: Baseline =
        serde_json::from_str(&content).with_context(|| "Failed to parse baseline JSON")?;

    Ok(Some(baseline))
}

/// Save baseline to file
fn save_baseline(path: &PathBuf, results: &[BenchmarkResult]) -> Result<()> {
    let mut benchmarks = HashMap::new();

    for result in results {
        benchmarks.insert(
            result.name.clone(),
            BaselineEntry {
                avg_time_us: result.avg_time_us,
                category: result.category,
            },
        );
    }

    let baseline = Baseline {
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono_lite_timestamp(),
        benchmarks,
    };

    let json =
        serde_json::to_string_pretty(&baseline).with_context(|| "Failed to serialize baseline")?;

    fs::write(path, json)
        .with_context(|| format!("Failed to write baseline file: {}", path.display()))?;

    Ok(())
}

/// Simple timestamp without external dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", duration.as_secs())
}

fn result_meta(name: &str) -> BenchmarkMeta {
    BenchmarkMeta {
        path: PathBuf::from(name),
        logic: None,
        expected_status: None,
        file_size: 0,
        category: None,
        structural_features: None,
    }
}

fn to_single_result(name: &str, avg_time_us: f64) -> SingleResult {
    let meta = result_meta(name);
    let duration = Duration::from_secs_f64((avg_time_us / 1_000_000.0).max(0.0));
    SingleResult::new(&meta, BenchmarkStatus::Sat, duration)
}

fn build_regression_analysis(
    results: &[BenchmarkResult],
    baseline: Option<&Baseline>,
    threshold_percent: f64,
) -> RegressionAnalysis {
    let current_results: Vec<_> = results
        .iter()
        .map(|result| to_single_result(&result.name, result.avg_time_us))
        .collect();

    let baseline_results: Vec<_> = baseline
        .map(|data| {
            results
                .iter()
                .filter_map(|result| {
                    data.benchmarks
                        .get(&result.name)
                        .map(|entry| to_single_result(&result.name, entry.avg_time_us))
                })
                .collect()
        })
        .unwrap_or_default();

    let detector = RegressionDetector::new(
        RegressionConfig::default()
            .with_time_threshold(threshold_percent)
            .with_solve_threshold(threshold_percent),
    );

    detector.analyze(&baseline_results, &current_results)
}

/// Compare results against baseline
fn compare_results(
    results: &[BenchmarkResult],
    baseline: Option<&Baseline>,
    threshold_percent: f64,
) -> RegressionReport {
    let mut comparisons = Vec::new();

    for result in results {
        let baseline_entry = baseline.and_then(|b| b.benchmarks.get(&result.name));

        let (change_percent, status) = match baseline_entry {
            Some(entry) => {
                let change = ((result.avg_time_us - entry.avg_time_us) / entry.avg_time_us) * 100.0;

                let status = if change > threshold_percent {
                    ComparisonStatus::Regression
                } else if change < -threshold_percent {
                    ComparisonStatus::Improvement
                } else {
                    ComparisonStatus::Unchanged
                };

                (Some(change), status)
            }
            None => (None, ComparisonStatus::New),
        };

        match status {
            ComparisonStatus::Regression
            | ComparisonStatus::Improvement
            | ComparisonStatus::Unchanged
            | ComparisonStatus::New => {}
        }

        comparisons.push(BenchmarkComparison {
            name: result.name.clone(),
            category: result.category,
            current_us: result.avg_time_us,
            baseline_us: baseline_entry.map(|e| e.avg_time_us),
            change_percent,
            is_regression: status == ComparisonStatus::Regression,
            status,
        });
    }

    RegressionReport {
        analysis: build_regression_analysis(results, baseline, threshold_percent),
        threshold_percent,
        comparisons,
    }
}

/// Print report in text format
fn print_text_report(report: &RegressionReport) {
    println!("=== OxiZ Performance Regression Report ===\n");

    println!("Summary:");
    println!("  Total benchmarks: {}", report.total_benchmarks());
    println!(
        "  Regressions:      {} (threshold: {:.1}%)",
        report.regressions(),
        report.threshold_percent
    );
    println!("  Improvements:     {}", report.improvements());
    println!("  Unchanged:        {}", report.unchanged());
    println!("  New benchmarks:   {}", report.new_benchmarks());
    println!();

    // Group by category
    let mut by_category: HashMap<BenchmarkCategory, Vec<&BenchmarkComparison>> = HashMap::new();
    for comp in &report.comparisons {
        by_category.entry(comp.category).or_default().push(comp);
    }

    for (category, comps) in by_category {
        println!("--- {} ---", category);
        for comp in comps {
            let change_str = match comp.change_percent {
                Some(change) => format!("{:+.1}%", change),
                None => "N/A".to_string(),
            };

            let status_indicator = match comp.status {
                ComparisonStatus::Regression => "[FAIL]",
                ComparisonStatus::Improvement => "[GOOD]",
                ComparisonStatus::Unchanged => "[ OK ]",
                ComparisonStatus::New => "[NEW ]",
            };

            println!(
                "  {} {:30} {:>10.1}us  {:>8}  {}",
                status_indicator,
                comp.name,
                comp.current_us,
                change_str,
                if let Some(baseline) = comp.baseline_us {
                    format!("(baseline: {:.1}us)", baseline)
                } else {
                    String::new()
                }
            );
        }
        println!();
    }

    if report.analysis.has_regressions {
        println!("RESULT: FAILED - Performance regressions detected!");
    } else {
        println!("RESULT: PASSED - No performance regressions detected.");
    }
}

/// Print report in JSON format
fn print_json_report(report: &RegressionReport) {
    match serde_json::to_string_pretty(report) {
        Ok(json) => println!("{}", json),
        Err(error) => eprintln!("Failed to serialize report: {error}"),
    }
}

/// Print report with GitHub Actions annotations
fn print_github_actions_report(report: &RegressionReport) {
    // Print summary
    println!("## Performance Regression Report\n");
    println!("| Benchmark | Category | Current | Baseline | Change | Status |");
    println!("|-----------|----------|---------|----------|--------|--------|");

    for comp in &report.comparisons {
        let change_str = match comp.change_percent {
            Some(change) => format!("{:+.1}%", change),
            None => "N/A".to_string(),
        };

        let baseline_str = match comp.baseline_us {
            Some(b) => format!("{:.1}us", b),
            None => "N/A".to_string(),
        };

        println!(
            "| {} | {} | {:.1}us | {} | {} | {} |",
            comp.name, comp.category, comp.current_us, baseline_str, change_str, comp.status
        );
    }

    // Emit GitHub Actions annotations for regressions
    for comp in &report.comparisons {
        if comp.is_regression {
            println!(
                "::error title=Performance Regression::Benchmark '{}' regressed by {:.1}% (current: {:.1}us, baseline: {:.1}us)",
                comp.name,
                comp.change_percent.unwrap_or(0.0),
                comp.current_us,
                comp.baseline_us.unwrap_or(0.0)
            );
        }
    }

    if report.analysis.has_regressions {
        println!(
            "\n::error::Performance regressions detected! {} benchmarks regressed.",
            report.regressions()
        );
    }
}

fn format_duration_ms(micros: f64) -> String {
    format!("{:.3} ms", micros / 1000.0)
}

fn format_delta(delta: Option<f64>) -> String {
    delta.map_or_else(|| "N/A".to_string(), |value| format!("{value:+.2}%"))
}

fn write_markdown_report(report: &RegressionReport, output_path: &Path) -> Result<()> {
    let mut content = String::from("## Performance Regression Report\n\n");
    content.push_str("| Benchmark | Baseline | Current | Delta | Status |\n");
    content.push_str("|---|---|---|---|---|\n");

    for comparison in &report.comparisons {
        let baseline = comparison
            .baseline_us
            .map_or_else(|| "N/A".to_string(), format_duration_ms);
        let current = format_duration_ms(comparison.current_us);
        let delta = format_delta(comparison.change_percent);
        let status = if comparison.is_regression {
            "⚠️"
        } else {
            "✅"
        };

        content.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            comparison.name, baseline, current, delta, status
        ));
    }

    content.push_str(&format!(
        "\nGenerated by bench-regression at {}\n",
        chrono_lite_timestamp()
    ));

    fs::write(output_path, content)
        .with_context(|| format!("Failed to write Markdown report: {}", output_path.display()))?;
    Ok(())
}

fn benchmark_smt2_source(name: &str) -> Option<&'static str> {
    match name {
        "theory_lia_simple" => Some(
            "(set-logic QF_LIA)\n\
             (declare-const x Int)\n\
             (declare-const y Int)\n\
             (assert (>= x 5))\n\
             (assert (<= x 10))\n\
             (assert (= y (+ x 1)))\n\
             (check-sat)\n",
        ),
        "theory_lia_medium" => Some(
            "(set-logic QF_LIA)\n\
             (declare-const x0 Int)\n(declare-const x1 Int)\n(declare-const x2 Int)\n\
             (declare-const x3 Int)\n(declare-const x4 Int)\n\
             (assert (<= (+ x0 x1 x2 x3 x4) 100))\n\
             (assert (and (>= x0 0) (<= x0 30)))\n\
             (assert (and (>= x1 0) (<= x1 30)))\n\
             (assert (and (>= x2 0) (<= x2 30)))\n\
             (assert (and (>= x3 0) (<= x3 30)))\n\
             (assert (and (>= x4 0) (<= x4 30)))\n\
             (check-sat)\n",
        ),
        "theory_bool_arith" => Some(
            "(set-logic QF_LIA)\n\
             (declare-const x Int)\n\
             (declare-const b Bool)\n\
             (assert (=> b (> x 5)))\n\
             (assert (=> (not b) (<= x 0)))\n\
             (check-sat)\n",
        ),
        "theory_lia_unsat" => Some(
            "(set-logic QF_LIA)\n\
             (declare-const x Int)\n\
             (assert (> x 5))\n\
             (assert (< x 3))\n\
             (check-sat)\n",
        ),
        "parser_simple" => Some(
            "(declare-const x Int)\n\
             (declare-const y Int)\n\
             (assert (> x 0))\n\
             (assert (< y 10))\n\
             (assert (= (+ x y) 15))\n\
             (check-sat)\n",
        ),
        "parser_medium" => Some(
            "(set-logic QF_LIA)\n\
             (declare-const a Int)\n(declare-const b Int)\n(declare-const c Int)\n\
             (declare-const d Int)\n(declare-const e Int)\n\
             (assert (>= a 0))\n(assert (>= b 0))\n(assert (>= c 0))\n\
             (assert (>= d 0))\n(assert (>= e 0))\n\
             (assert (<= a 100))\n(assert (<= b 100))\n(assert (<= c 100))\n\
             (assert (<= d 100))\n(assert (<= e 100))\n\
             (assert (= (+ a b c) 50))\n\
             (assert (= (+ c d e) 75))\n\
             (assert (< (+ a e) 30))\n\
             (check-sat)\n",
        ),
        "parser_nested" => Some(
            "(declare-const x Int)\n\
             (assert (= x (+ 1 (+ 2 (+ 3 (+ 4 (+ 5 (+ 6 (+ 7 (+ 8 (+ 9 10)))))))))))\n\
             (check-sat)\n",
        ),
        _ => None,
    }
}

fn prepare_z3_inputs(results: &[BenchmarkResult]) -> Result<(TempDir, HashMap<String, PathBuf>)> {
    let temp_dir = TempDir::new().with_context(|| "Failed to create temporary Z3 input dir")?;
    let mut paths = HashMap::new();

    for result in results {
        if let Some(contents) = benchmark_smt2_source(&result.name) {
            let path = temp_dir.path().join(format!("{}.smt2", result.name));
            fs::write(&path, contents).with_context(|| {
                format!("Failed to write temporary SMT2 file: {}", path.display())
            })?;
            paths.insert(result.name.clone(), path);
        }
    }

    Ok((temp_dir, paths))
}

fn print_z3_report(report: &Z3ComparisonReport) {
    if !report.z3_available {
        eprintln!("Z3 comparison skipped: z3 binary not found.");
        return;
    }

    if let Some(version) = &report.z3_version {
        eprintln!("Z3 comparison using {version}");
    }

    for entry in &report.entries {
        let z3 = entry
            .z3_ms
            .map_or_else(|| "N/A".to_string(), |value| format!("{value:.3} ms"));
        let ratio = entry
            .ratio
            .map_or_else(|| "N/A".to_string(), |value| format!("{value:.3}x"));
        eprintln!(
            "[z3] {:30} oxiz={:.3} ms z3={} ratio={}",
            entry.benchmark, entry.oxiz_ms, z3, ratio
        );
    }

    if let Some(geomean) = report.geomean_ratio {
        eprintln!("  geomean ratio: {:.3}x", geomean);
    }
    if let Some(p50) = report.p50_ratio {
        eprintln!("  p50 ratio:     {:.3}x", p50);
    }
    if let Some(p95) = report.p95_ratio {
        eprintln!("  p95 ratio:     {:.3}x", p95);
    }

    eprintln!(
        "Z3 parity target (<= 1.2x): {}",
        if report.within_target {
            "met"
        } else {
            "not met"
        }
    );
}

/// Parse command line arguments
fn parse_args() -> Result<Config> {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" | "-b" => {
                i += 1;
                if i < args.len() {
                    config.baseline_path = PathBuf::from(&args[i]);
                }
            }
            "--threshold" | "-t" => {
                i += 1;
                if i < args.len() {
                    config.threshold_percent = args[i]
                        .parse()
                        .with_context(|| format!("Invalid threshold value: {}", args[i]))?;
                }
            }
            "--update" | "-u" => {
                config.update_baseline = true;
            }
            "--json" => {
                config.output_format = OutputFormat::Json;
            }
            "--github" => {
                config.output_format = OutputFormat::GithubActions;
            }
            "--markdown" => {
                config.markdown = true;
            }
            "--compare-z3" => {
                config.compare_z3 = true;
            }
            "--refresh-baseline" => {
                config.update_baseline = true;
                config.refresh_baseline = true;
            }
            "--check-geomean" => {
                config.check_geomean = true;
            }
            "--max" => {
                i += 1;
                if i < args.len() {
                    config.check_geomean_max = args[i]
                        .parse()
                        .with_context(|| format!("Invalid --max value: {}", args[i]))?;
                }
            }
            "--history-dir" => {
                i += 1;
                if i < args.len() {
                    config.history_dir = PathBuf::from(&args[i]);
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    println!("OxiZ Performance Regression Tester");
    println!();
    println!("Usage: regression [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -b, --baseline <FILE>       Path to baseline file (default: baseline.json)");
    println!("  -t, --threshold <PCT>       Regression threshold percentage (default: 10)");
    println!("  -u, --update                Update baseline with current results");
    println!(
        "      --refresh-baseline      Refresh baseline (implies --update); marks a deliberate refresh"
    );
    println!("      --json                  Output in JSON format");
    println!("      --github                Output with GitHub Actions annotations");
    println!("      --markdown              Write regression_comment.md in Markdown format");
    println!("      --compare-z3            Compare eligible benchmarks against z3");
    println!("      --check-geomean         Read latest history snapshot and check geomean ratio");
    println!(
        "      --max <RATIO>           Maximum allowed geomean ratio for --check-geomean (default: 1.2)"
    );
    println!(
        "      --history-dir <DIR>     Directory containing history snapshots (default: history)"
    );
    println!("  -h, --help                  Print help information");
}

/// Minimal structs for parsing geomean data from history snapshot files.
#[derive(Deserialize, Default)]
struct HistorySummaryField {
    #[serde(default)]
    geomean_ratio: Option<f64>,
    #[serde(default)]
    count: usize,
}

#[derive(Deserialize)]
struct HistoryFile {
    summary: HistorySummaryField,
}

/// Run the `--check-geomean` mode: find the newest history snapshot, parse geomean, and exit.
fn run_check_geomean(config: &Config) -> Result<()> {
    let history_dir = if config.history_dir.is_absolute() {
        config.history_dir.clone()
    } else {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .or_else(|_| std::env::current_dir())
            .with_context(|| "current working directory should be available")?;
        manifest_dir.join(&config.history_dir)
    };

    if !history_dir.exists() {
        eprintln!("Geomean gate: no history data, skipping (soft-gate pass)");
        std::process::exit(0);
    }

    let json_files: Vec<_> = fs::read_dir(&history_dir)
        .with_context(|| format!("Failed to read history dir: {}", history_dir.display()))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if json_files.is_empty() {
        eprintln!("Geomean gate: no history data, skipping (soft-gate pass)");
        std::process::exit(0);
    }

    // Find newest file by modified time
    let newest = json_files
        .iter()
        .filter_map(|path| {
            let modified = fs::metadata(path).ok()?.modified().ok()?;
            Some((modified, path))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path.clone());

    let newest_path = match newest {
        Some(p) => p,
        None => {
            eprintln!("Geomean gate: no history data, skipping (soft-gate pass)");
            std::process::exit(0);
        }
    };

    let content = fs::read_to_string(&newest_path)
        .with_context(|| format!("Failed to read history snapshot: {}", newest_path.display()))?;

    let history: HistoryFile = serde_json::from_str(&content).with_context(|| {
        format!(
            "Failed to parse history snapshot: {}",
            newest_path.display()
        )
    })?;

    let summary = history.summary;

    if summary.count == 0 {
        eprintln!("Geomean gate: no Z3 comparison data in snapshot, skipping (soft-gate pass)");
        std::process::exit(0);
    }

    let geomean = match summary.geomean_ratio {
        Some(g) => g,
        None => {
            eprintln!("Geomean gate: no Z3 comparison data in snapshot, skipping (soft-gate pass)");
            std::process::exit(0);
        }
    };

    let max = config.check_geomean_max;
    if geomean > max {
        eprintln!("Geomean gate FAILED: {geomean:.4} > {max:.4} (Z3 parity target)");
        std::process::exit(1);
    }

    eprintln!("Geomean gate PASSED: {geomean:.4} <= {max:.4}");
    std::process::exit(0);
}

fn main() -> Result<()> {
    let config = parse_args()?;

    // Short-circuit: --check-geomean reads history snapshots and exits
    if config.check_geomean {
        return run_check_geomean(&config);
    }

    // Determine baseline path relative to executable or current dir
    let baseline_path = if config.baseline_path.is_relative() {
        // Try to find baseline relative to the source directory
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .or_else(|_| std::env::current_dir())
            .with_context(|| "current working directory should be available")?;
        manifest_dir.join(&config.baseline_path)
    } else {
        config.baseline_path.clone()
    };

    // Load baseline if available
    let baseline = load_baseline(&baseline_path)?;

    if baseline.is_none() && !config.update_baseline {
        eprintln!(
            "Warning: No baseline file found at {}. Running benchmarks anyway.",
            baseline_path.display()
        );
    }

    // Run benchmarks
    eprintln!("Running benchmarks...");
    let results = run_all_benchmarks();
    eprintln!("Completed {} benchmarks.", results.len());

    // Compare against baseline
    let report = compare_results(&results, baseline.as_ref(), config.threshold_percent);

    if config.markdown {
        write_markdown_report(&report, Path::new("regression_comment.md"))?;
    }

    if config.compare_z3 {
        let (_temp_dir, smt2_paths) = prepare_z3_inputs(&results)?;
        let oxiz_timings = results
            .iter()
            .map(|result| (result.name.clone(), result.avg_time_us / 1000.0))
            .collect::<Vec<_>>();
        let z3_report = compare_with_z3(&oxiz_timings, &smt2_paths);
        print_z3_report(&z3_report);
    }

    // Output report
    match config.output_format {
        OutputFormat::Text => print_text_report(&report),
        OutputFormat::Json => print_json_report(&report),
        OutputFormat::GithubActions => print_github_actions_report(&report),
    }

    // Update baseline if requested
    if config.update_baseline {
        save_baseline(&baseline_path, &results)?;
        eprintln!("Baseline updated at: {}", baseline_path.display());
    }

    // Exit with appropriate code
    if report.analysis.has_regressions {
        std::process::exit(1);
    }

    Ok(())
}
