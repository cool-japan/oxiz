//! OxiZ Performance Regression Test Runner
//!
//! This binary runs performance benchmarks and compares against a baseline,
//! detecting regressions and producing CI-friendly output.

mod benchmarks;

use anyhow::{Context, Result};
use benchmarks::{run_all_benchmarks, BenchmarkCategory, BenchmarkResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

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
    /// Whether any regression was detected
    pub has_regression: bool,
    /// Regression threshold percentage
    pub threshold_percent: f64,
    /// Individual benchmark comparisons
    pub comparisons: Vec<BenchmarkComparison>,
    /// Summary statistics
    pub summary: ReportSummary,
}

/// Summary of the regression report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total benchmarks run
    pub total_benchmarks: usize,
    /// Number of regressions detected
    pub regressions: usize,
    /// Number of improvements detected
    pub improvements: usize,
    /// Number of unchanged benchmarks
    pub unchanged: usize,
    /// Number of new benchmarks (no baseline)
    pub new_benchmarks: usize,
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

    let baseline: Baseline = serde_json::from_str(&content)
        .with_context(|| "Failed to parse baseline JSON")?;

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

    let json = serde_json::to_string_pretty(&baseline)
        .with_context(|| "Failed to serialize baseline")?;

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

/// Compare results against baseline
fn compare_results(
    results: &[BenchmarkResult],
    baseline: Option<&Baseline>,
    threshold_percent: f64,
) -> RegressionReport {
    let mut comparisons = Vec::new();
    let mut has_regression = false;
    let mut regressions = 0;
    let mut improvements = 0;
    let mut unchanged = 0;
    let mut new_benchmarks = 0;

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
            ComparisonStatus::Regression => {
                has_regression = true;
                regressions += 1;
            }
            ComparisonStatus::Improvement => improvements += 1,
            ComparisonStatus::Unchanged => unchanged += 1,
            ComparisonStatus::New => new_benchmarks += 1,
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
        has_regression,
        threshold_percent,
        comparisons,
        summary: ReportSummary {
            total_benchmarks: results.len(),
            regressions,
            improvements,
            unchanged,
            new_benchmarks,
        },
    }
}

/// Print report in text format
fn print_text_report(report: &RegressionReport) {
    println!("=== OxiZ Performance Regression Report ===\n");

    println!("Summary:");
    println!("  Total benchmarks: {}", report.summary.total_benchmarks);
    println!("  Regressions:      {} (threshold: {:.1}%)", report.summary.regressions, report.threshold_percent);
    println!("  Improvements:     {}", report.summary.improvements);
    println!("  Unchanged:        {}", report.summary.unchanged);
    println!("  New benchmarks:   {}", report.summary.new_benchmarks);
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

    if report.has_regression {
        println!("RESULT: FAILED - Performance regressions detected!");
    } else {
        println!("RESULT: PASSED - No performance regressions detected.");
    }
}

/// Print report in JSON format
fn print_json_report(report: &RegressionReport) {
    let json = serde_json::to_string_pretty(report).expect("Failed to serialize report");
    println!("{}", json);
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
            comp.name,
            comp.category,
            comp.current_us,
            baseline_str,
            change_str,
            comp.status
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

    if report.has_regression {
        println!("\n::error::Performance regressions detected! {} benchmarks regressed.", report.summary.regressions);
    }
}

/// Parse command line arguments
fn parse_args() -> Config {
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
                    config.threshold_percent = args[i].parse().unwrap_or(10.0);
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
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}

fn print_help() {
    println!("OxiZ Performance Regression Tester");
    println!();
    println!("Usage: regression [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -b, --baseline <FILE>   Path to baseline file (default: baseline.json)");
    println!("  -t, --threshold <PCT>   Regression threshold percentage (default: 10)");
    println!("  -u, --update            Update baseline with current results");
    println!("      --json              Output in JSON format");
    println!("      --github            Output with GitHub Actions annotations");
    println!("  -h, --help              Print help information");
}

fn main() -> Result<()> {
    let config = parse_args();

    // Determine baseline path relative to executable or current dir
    let baseline_path = if config.baseline_path.is_relative() {
        // Try to find baseline relative to the source directory
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap());
        manifest_dir.join(&config.baseline_path)
    } else {
        config.baseline_path.clone()
    };

    // Load baseline if available
    let baseline = load_baseline(&baseline_path)?;

    if baseline.is_none() && !config.update_baseline {
        eprintln!("Warning: No baseline file found at {}. Running benchmarks anyway.", baseline_path.display());
    }

    // Run benchmarks
    eprintln!("Running benchmarks...");
    let results = run_all_benchmarks();
    eprintln!("Completed {} benchmarks.", results.len());

    // Compare against baseline
    let report = compare_results(&results, baseline.as_ref(), config.threshold_percent);

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
    if report.has_regression {
        std::process::exit(1);
    }

    Ok(())
}
