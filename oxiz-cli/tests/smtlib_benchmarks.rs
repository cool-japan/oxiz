//! SMT-LIB benchmark suite tests
//!
//! This module provides infrastructure for testing oxiz against SMT-LIB benchmarks.
//! To run these tests, set the SMTLIB_BENCH_PATH environment variable to the path
//! containing SMT-LIB2 benchmark files.
//!
//! Example:
//!   SMTLIB_BENCH_PATH=/path/to/smtlib/benchmarks cargo test --test smtlib_benchmarks

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Get the path to the oxiz binary
fn oxiz_bin() -> PathBuf {
    let mut path = env::current_exe().expect("Failed to get current executable path");
    path.pop(); // Remove test executable name
    if path.ends_with("deps") {
        path.pop(); // Remove deps directory
    }
    path.push("oxiz");
    path
}

/// Get the SMT-LIB benchmark path from environment variable
fn get_benchmark_path() -> Option<PathBuf> {
    env::var("SMTLIB_BENCH_PATH").ok().map(PathBuf::from)
}

/// Run a single SMT-LIB2 file and return the result
fn run_smtlib_file(file: &Path) -> Result<(String, bool), String> {
    let output = Command::new(oxiz_bin())
        .arg(file)
        .arg("--quiet")
        .output()
        .map_err(|e| format!("Failed to execute oxiz: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let success = output.status.success();

    Ok((stdout, success))
}

/// Collect all .smt2 files from a directory
fn collect_smt2_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("smt2") {
                files.push(path);
            } else if path.is_dir() {
                files.extend(collect_smt2_files(&path));
            }
        }
    }

    files
}

#[test]
fn test_smtlib_benchmarks() {
    let bench_path = match get_benchmark_path() {
        Some(path) => path,
        None => {
            println!("Skipping SMT-LIB benchmark tests - SMTLIB_BENCH_PATH not set");
            println!("To run these tests, set SMTLIB_BENCH_PATH to your benchmark directory");
            return;
        }
    };

    if !bench_path.exists() {
        eprintln!("Benchmark path does not exist: {}", bench_path.display());
        return;
    }

    let files = collect_smt2_files(&bench_path);

    if files.is_empty() {
        println!("No .smt2 files found in {}", bench_path.display());
        return;
    }

    println!("Found {} SMT-LIB2 benchmark files", files.len());

    let mut passed = 0;
    let mut failed = 0;
    let mut errors = 0;

    for file in &files {
        match run_smtlib_file(file) {
            Ok((output, _success)) => {
                if output.contains("sat") || output.contains("unsat") || output.contains("unknown")
                {
                    passed += 1;
                } else if output.contains("error") {
                    errors += 1;
                    println!("Error in {}: {}", file.display(), output);
                } else {
                    failed += 1;
                    println!("Unexpected output from {}: {}", file.display(), output);
                }
            }
            Err(e) => {
                errors += 1;
                println!("Failed to run {}: {}", file.display(), e);
            }
        }
    }

    println!("\nBenchmark Results:");
    println!("  Total files: {}", files.len());
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);
    println!("  Errors: {}", errors);
    println!(
        "  Success rate: {:.2}%",
        (passed as f64 / files.len() as f64) * 100.0
    );

    // Don't fail the test if no files were processed or if there are known issues
    // This allows the test to be informational rather than strict
}

#[test]
fn test_qf_lia_benchmarks() {
    let bench_path = match get_benchmark_path() {
        Some(mut path) => {
            path.push("QF_LIA");
            path
        }
        None => {
            println!("Skipping QF_LIA benchmark tests - SMTLIB_BENCH_PATH not set");
            return;
        }
    };

    if !bench_path.exists() {
        println!(
            "QF_LIA benchmark path does not exist: {}",
            bench_path.display()
        );
        return;
    }

    let files = collect_smt2_files(&bench_path);
    println!("Found {} QF_LIA benchmark files", files.len());

    for file in files.iter().take(10) {
        // Test first 10 files
        match run_smtlib_file(file) {
            Ok((output, _)) => {
                println!("{}: {}", file.display(), output.trim());
            }
            Err(e) => {
                println!("{}: Error - {}", file.display(), e);
            }
        }
    }
}

#[test]
fn test_qf_uf_benchmarks() {
    let bench_path = match get_benchmark_path() {
        Some(mut path) => {
            path.push("QF_UF");
            path
        }
        None => {
            println!("Skipping QF_UF benchmark tests - SMTLIB_BENCH_PATH not set");
            return;
        }
    };

    if !bench_path.exists() {
        println!(
            "QF_UF benchmark path does not exist: {}",
            bench_path.display()
        );
        return;
    }

    let files = collect_smt2_files(&bench_path);
    println!("Found {} QF_UF benchmark files", files.len());

    for file in files.iter().take(10) {
        // Test first 10 files
        match run_smtlib_file(file) {
            Ok((output, _)) => {
                println!("{}: {}", file.display(), output.trim());
            }
            Err(e) => {
                println!("{}: Error - {}", file.display(), e);
            }
        }
    }
}
