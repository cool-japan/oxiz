mod comparator;
mod oxiz_runner;
mod z3_runner;

use anyhow::{Context, Result};
use colored::Colorize;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tabled::{Table, Tabled};

use comparator::{MatchStatus, compare_results};
use oxiz_runner::run_oxiz;
use z3_runner::run_z3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SolverResult {
    Sat,
    Unsat,
    Unknown,
    Error(String),
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityResult {
    pub benchmark: String,
    pub logic: String,
    pub oxiz_result: SolverResult,
    pub z3_result: SolverResult,
    pub oxiz_time: Duration,
    pub z3_time: Duration,
    pub match_status: MatchStatus,
}

#[derive(Debug, Tabled)]
struct ResultRow {
    #[tabled(rename = "Logic")]
    logic: String,
    #[tabled(rename = "Total")]
    total: usize,
    #[tabled(rename = "Correct")]
    correct: usize,
    #[tabled(rename = "Wrong")]
    wrong: usize,
    #[tabled(rename = "Timeout")]
    timeout: usize,
    #[tabled(rename = "Error")]
    error: usize,
    #[tabled(rename = "Accuracy")]
    accuracy: String,
}

fn discover_benchmarks(base_path: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut benchmarks = Vec::new();

    for logic_dir in fs::read_dir(base_path)? {
        let logic_dir = logic_dir?;
        if !logic_dir.path().is_dir() {
            continue;
        }

        let logic_name = logic_dir.file_name().to_string_lossy().to_string();

        for entry in fs::read_dir(logic_dir.path())? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("smt2") {
                benchmarks.push((logic_name.clone(), path));
            }
        }
    }

    benchmarks.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(benchmarks)
}

fn run_benchmark(logic: &str, path: &Path) -> Result<ParityResult> {
    println!("  Running: {}", path.display());

    // Run OxiZ
    let oxiz_start = Instant::now();
    let oxiz_result = run_oxiz(path)?;
    let oxiz_time = oxiz_start.elapsed();

    // Run Z3
    let z3_start = Instant::now();
    let z3_result = run_z3(path)?;
    let z3_time = z3_start.elapsed();

    // Compare results
    let match_status = compare_results(&oxiz_result, &z3_result);

    Ok(ParityResult {
        benchmark: path.file_name().unwrap().to_string_lossy().to_string(),
        logic: logic.to_string(),
        oxiz_result,
        z3_result,
        oxiz_time,
        z3_time,
        match_status,
    })
}

fn generate_report(results: &[ParityResult]) {
    println!("\n{}", "=".repeat(80).bright_cyan());
    println!("{}", "Z3 PARITY TEST REPORT".bright_cyan().bold());
    println!("{}", "=".repeat(80).bright_cyan());

    // Group by logic
    let mut by_logic: HashMap<String, Vec<&ParityResult>> = HashMap::new();
    for result in results {
        by_logic
            .entry(result.logic.clone())
            .or_default()
            .push(result);
    }

    let mut rows = Vec::new();
    let mut total_correct = 0;
    let mut total_wrong = 0;
    let mut total_timeout = 0;
    let mut total_error = 0;
    let mut total_tests = 0;

    for (logic, logic_results) in by_logic.iter() {
        let total = logic_results.len();
        let correct = logic_results
            .iter()
            .filter(|r| matches!(r.match_status, MatchStatus::Correct))
            .count();
        let wrong = logic_results
            .iter()
            .filter(|r| matches!(r.match_status, MatchStatus::Wrong))
            .count();
        let timeout = logic_results
            .iter()
            .filter(|r| matches!(r.match_status, MatchStatus::Timeout))
            .count();
        let error = logic_results
            .iter()
            .filter(|r| matches!(r.match_status, MatchStatus::Error))
            .count();

        let accuracy = if total > 0 {
            format!("{:.1}%", (correct as f64 / total as f64) * 100.0)
        } else {
            "N/A".to_string()
        };

        total_correct += correct;
        total_wrong += wrong;
        total_timeout += timeout;
        total_error += error;
        total_tests += total;

        rows.push(ResultRow {
            logic: logic.clone(),
            total,
            correct,
            wrong,
            timeout,
            error,
            accuracy,
        });
    }

    // Add total row
    let overall_accuracy = if total_tests > 0 {
        format!(
            "{:.1}%",
            (total_correct as f64 / total_tests as f64) * 100.0
        )
    } else {
        "N/A".to_string()
    };

    rows.push(ResultRow {
        logic: "TOTAL".to_string().bold().to_string(),
        total: total_tests,
        correct: total_correct,
        wrong: total_wrong,
        timeout: total_timeout,
        error: total_error,
        accuracy: overall_accuracy,
    });

    println!("\n{}", Table::new(rows));

    // Print failures
    let failures: Vec<_> = results
        .iter()
        .filter(|r| !matches!(r.match_status, MatchStatus::Correct))
        .collect();

    if !failures.is_empty() {
        println!("\n{}", "FAILURES:".bright_red().bold());
        for failure in failures {
            println!(
                "\n  {} [{}]",
                failure.benchmark.bright_yellow(),
                failure.logic
            );
            println!(
                "    OxiZ:  {:?} ({:.3}s)",
                failure.oxiz_result,
                failure.oxiz_time.as_secs_f64()
            );
            println!(
                "    Z3:    {:?} ({:.3}s)",
                failure.z3_result,
                failure.z3_time.as_secs_f64()
            );
            println!("    Status: {:?}", failure.match_status);
        }
    }

    println!("\n{}", "=".repeat(80).bright_cyan());
}

fn main() -> Result<()> {
    println!("{}", "OxiZ vs Z3 Parity Testing Suite".bright_cyan().bold());
    println!("{}\n", "=".repeat(80).bright_cyan());

    let benchmark_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");

    if !benchmark_dir.exists() {
        anyhow::bail!("Benchmark directory not found: {}", benchmark_dir.display());
    }

    let benchmarks =
        discover_benchmarks(&benchmark_dir).context("Failed to discover benchmarks")?;

    println!("Found {} benchmarks\n", benchmarks.len());

    // Run benchmarks in parallel
    let results: Vec<ParityResult> = benchmarks
        .par_iter()
        .filter_map(|(logic, path)| match run_benchmark(logic, path) {
            Ok(result) => Some(result),
            Err(e) => {
                eprintln!("Error running {}: {}", path.display(), e);
                None
            }
        })
        .collect();

    // Save results to JSON
    let results_json = serde_json::to_string_pretty(&results)?;
    fs::write("results.json", results_json)?;
    println!("\nResults saved to results.json");

    // Generate report
    generate_report(&results);

    Ok(())
}
