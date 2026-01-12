//! Benchmark tests for oxiz CLI
//!
//! These tests measure performance characteristics of the CLI

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

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

/// Create a temporary SMT2 file for testing
fn create_temp_smt2(content: &str, name: &str) -> PathBuf {
    let temp_dir = env::temp_dir();
    let file_path = temp_dir.join(format!("bench_{}.smt2", name));
    fs::write(&file_path, content).expect("Failed to write temp file");
    file_path
}

#[test]
fn bench_simple_sat_problem() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content, "simple_sat");

    let start = Instant::now();
    let output = Command::new(oxiz_bin())
        .arg(temp_file.to_str().unwrap())
        .arg("--quiet")
        .output()
        .expect("Failed to execute oxiz");
    let elapsed = start.elapsed();

    fs::remove_file(temp_file).ok();

    println!("Simple SAT problem took: {:?}", elapsed);
    assert!(output.status.success() || output.status.code() == Some(1));
    assert!(
        elapsed.as_millis() < 5000,
        "Simple SAT problem took too long: {:?}",
        elapsed
    );
}

#[test]
fn bench_simple_unsat_problem() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(assert (= x 43))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content, "simple_unsat");

    let start = Instant::now();
    let output = Command::new(oxiz_bin())
        .arg(temp_file.to_str().unwrap())
        .arg("--quiet")
        .output()
        .expect("Failed to execute oxiz");
    let elapsed = start.elapsed();

    fs::remove_file(temp_file).ok();

    println!("Simple UNSAT problem took: {:?}", elapsed);
    assert!(output.status.success() || output.status.code() == Some(1));
    assert!(
        elapsed.as_millis() < 5000,
        "Simple UNSAT problem took too long: {:?}",
        elapsed
    );
}

#[test]
fn bench_boolean_logic() {
    let smt2_content = r#"
(set-logic QF_UF)
(declare-const p Bool)
(declare-const q Bool)
(declare-const r Bool)
(assert (or (and p q) (and (not p) r)))
(assert (not (and p r)))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content, "boolean_logic");

    let start = Instant::now();
    let output = Command::new(oxiz_bin())
        .arg(temp_file.to_str().unwrap())
        .arg("--quiet")
        .output()
        .expect("Failed to execute oxiz");
    let elapsed = start.elapsed();

    fs::remove_file(temp_file).ok();

    println!("Boolean logic problem took: {:?}", elapsed);
    assert!(output.status.success() || output.status.code() == Some(1));
    assert!(
        elapsed.as_millis() < 5000,
        "Boolean logic problem took too long: {:?}",
        elapsed
    );
}

#[test]
fn bench_multiple_assertions() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(assert (> x 0))
(assert (> y 0))
(assert (> z 0))
(assert (< (+ x y) 10))
(assert (< (+ y z) 10))
(assert (< (+ x z) 10))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content, "multiple_assertions");

    let start = Instant::now();
    let output = Command::new(oxiz_bin())
        .arg(temp_file.to_str().unwrap())
        .arg("--quiet")
        .output()
        .expect("Failed to execute oxiz");
    let elapsed = start.elapsed();

    fs::remove_file(temp_file).ok();

    println!("Multiple assertions problem took: {:?}", elapsed);
    assert!(output.status.success() || output.status.code() == Some(1));
    assert!(
        elapsed.as_millis() < 5000,
        "Multiple assertions problem took too long: {:?}",
        elapsed
    );
}

#[test]
fn bench_stats_output() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content, "stats");

    let start = Instant::now();
    let output = Command::new(oxiz_bin())
        .arg(temp_file.to_str().unwrap())
        .arg("--stats")
        .arg("--time")
        .output()
        .expect("Failed to execute oxiz");
    let elapsed = start.elapsed();

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Stats output problem took: {:?}", elapsed);
    println!("Stats output: {}", stdout);

    assert!(output.status.success() || output.status.code() == Some(1));
    assert!(
        stdout.contains("Statistics") || stdout.contains("Decisions"),
        "Expected statistics in output"
    );
}

#[test]
fn bench_json_output() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content, "json");

    let start = Instant::now();
    let output = Command::new(oxiz_bin())
        .arg(temp_file.to_str().unwrap())
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute oxiz");
    let elapsed = start.elapsed();

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("JSON output problem took: {:?}", elapsed);

    assert!(output.status.success() || output.status.code() == Some(1));
    assert!(stdout.contains("{") && stdout.contains("}"));
}
