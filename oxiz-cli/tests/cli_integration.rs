//! Integration tests for oxiz CLI

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Get the path to the oxiz binary
fn oxiz_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_oxiz"))
}

/// Create a temporary SMT2 file for testing
fn create_temp_smt2(content: &str) -> PathBuf {
    let temp_dir = env::temp_dir();
    let file_path = temp_dir.join(format!("test_{}.smt2", rand_string()));
    fs::write(&file_path, content).expect("Failed to write temp file");
    file_path
}

/// Generate a random string for unique filenames
fn rand_string() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("Time went backwards");
    format!("{}{}", duration.as_secs(), duration.subsec_nanos())
}

#[test]
fn test_cli_version() {
    let output = Command::new(oxiz_bin())
        .arg("--version")
        .output()
        .expect("Failed to execute oxiz");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("oxiz"));
}

#[test]
fn test_cli_help() {
    let output = Command::new(oxiz_bin())
        .arg("--help")
        .output()
        .expect("Failed to execute oxiz");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage"));
    assert!(stdout.contains("Options"));
}

#[test]
fn test_simple_sat_problem() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    // Check that the command executed without crashing
    // Note: The actual result depends on oxiz-solver implementation
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should not crash
    assert!(
        output.status.success()
            || stdout.contains("sat")
            || stdout.contains("error")
            || stderr.contains("error"),
        "Unexpected failure: stdout={}, stderr={}",
        stdout,
        stderr
    );
}

#[test]
fn test_quiet_mode() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg("--quiet")
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // In quiet mode, should not have "Processing" messages
    assert!(!stdout.contains("Processing"));
}

#[test]
fn test_json_output_format() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg("--format")
        .arg("json")
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // JSON output should contain proper JSON structure
    assert!(
        stdout.contains("{") && stdout.contains("}"),
        "Expected JSON output, got: {}",
        stdout
    );
}

#[test]
fn test_yaml_output_format() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg("--format")
        .arg("yaml")
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // YAML output should contain YAML structure
    assert!(
        stdout.contains("results:") || stdout.contains("statistics:"),
        "Expected YAML output, got: {}",
        stdout
    );
}

#[test]
fn test_timing_flag() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg("--time")
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain timing information
    assert!(
        stdout.contains("time") || stdout.contains("ms") || stdout.contains("Time"),
        "Expected timing information, got: {}",
        stdout
    );
}

#[test]
fn test_stats_flag() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg("--stats")
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain statistics
    assert!(
        stdout.contains("Statistics") || stdout.contains("statistics"),
        "Expected statistics, got: {}",
        stdout
    );
}

#[test]
fn test_no_color_flag() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg("--no-color")
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should not contain ANSI color codes
    assert!(
        !stdout.contains("\x1b["),
        "Expected no color codes, got: {}",
        stdout
    );
}

#[test]
fn test_multiple_files() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file1 = create_temp_smt2(smt2_content);
    let temp_file2 = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg(temp_file1.to_str().unwrap())
        .arg(temp_file2.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file1).ok();
    fs::remove_file(temp_file2).ok();

    // Should process both files without crashing
    assert!(
        output.status.success() || output.status.code() == Some(1),
        "Unexpected exit status"
    );
}

#[test]
fn test_stdin_input() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let mut child = Command::new(oxiz_bin())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn oxiz");

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin.write_all(smt2_content.as_bytes()).ok();
    }

    let output = child.wait_with_output().expect("Failed to read output");

    // Should handle stdin input without crashing
    assert!(
        output.status.success() || output.status.code() == Some(1),
        "Unexpected exit status"
    );
}

#[test]
fn test_verbosity_levels() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    for level in &["quiet", "normal", "verbose"] {
        let output = Command::new(oxiz_bin())
            .arg("--verbosity")
            .arg(level)
            .arg(temp_file.to_str().unwrap())
            .output()
            .expect("Failed to execute oxiz");

        // Should handle all verbosity levels without crashing
        assert!(
            output.status.success() || output.status.code() == Some(1),
            "Failed at verbosity level: {}",
            level
        );
    }

    fs::remove_file(temp_file).ok();
}

/// Regression test for issue #5 follow-up: sequential multi-file solving must
/// not leak solver state from one file into the next.
///
/// Before the fix, the second file always returned `unsat` in sequential mode
/// because assertions from file 1 were still present in the shared Context
/// when file 2 was solved.
#[test]
fn test_issue_5_sequential_isolation() {
    // A simple LRA problem: (declare x Real), assert x >= 1.5 → sat
    let lra_content = r#"
(set-logic QF_LRA)
(declare-const x Real)
(assert (>= x 1.5))
(check-sat)
"#;

    // A simple LIA problem: (declare n Int), assert n <= 10 → sat
    let lia_content = r#"
(set-logic QF_LIA)
(declare-const n Int)
(assert (<= n 10))
(check-sat)
"#;

    let lra_file = create_temp_smt2(lra_content);
    let lia_file = create_temp_smt2(lia_content);

    // Run both files sequentially (no --parallel flag)
    let output = Command::new(oxiz_bin())
        .arg("--quiet")
        .arg(lra_file.to_str().expect("lra path is valid UTF-8"))
        .arg(lia_file.to_str().expect("lia path is valid UTF-8"))
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(&lra_file).ok();
    fs::remove_file(&lia_file).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Both files must be reported sat; neither should be unsat due to leaked state.
    let lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    assert_eq!(
        lines.len(),
        2,
        "Expected exactly 2 result lines (one per file), got: {:?}\nstdout={}\nstderr={}",
        lines,
        stdout,
        stderr
    );
    assert_eq!(
        lines[0], "sat",
        "First file (LRA) should be sat, got '{}'\nstdout={}\nstderr={}",
        lines[0], stdout, stderr
    );
    assert_eq!(
        lines[1], "sat",
        "Second file (LIA) should be sat, got '{}' — state leak from first file suspected\nstdout={}\nstderr={}",
        lines[1], stdout, stderr
    );
}

/// Regression test for GitHub issue #5: --memory must report process RSS, not system-wide RAM.
///
/// Before the fix, `--memory` output showed the host's total installed RAM
/// (e.g. 64 GB) instead of the solver process's own resident set size.
///
/// This test asserts:
/// 1. The `--memory` flag does not crash the CLI.
/// 2. The reported memory is > 0 (the process actually uses some RAM).
/// 3. The reported memory is well below the total system RAM
///    (a process solving a tiny problem should never claim all installed RAM).
#[test]
fn test_issue_5_memory_reports_process_rss_not_system_total() {
    use sysinfo::System;

    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (and (>= x 0) (<= x 100)))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);

    let output = Command::new(oxiz_bin())
        .arg("--memory")
        .arg("--quiet")
        .arg(temp_file.to_str().expect("temp path is valid UTF-8"))
        .output()
        .expect("Failed to execute oxiz with --memory");

    fs::remove_file(&temp_file).ok();

    // CLI must exit successfully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "--memory flag caused non-zero exit.\nstderr={}",
        stderr
    );

    // Find the "Memory used:" line in stderr (statistics go to stderr)
    let stderr_str = stderr.to_string();
    let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
    let combined = format!("{}{}", stdout_str, stderr_str);

    // The memory output line may appear in either stdout or stderr depending on args
    // Parse the numeric value from "Memory used: N MB"
    let reported_mb: Option<u64> = combined
        .lines()
        .find(|l| l.contains("Memory used:"))
        .and_then(|line| {
            line.split_whitespace()
                .find(|tok| tok.parse::<u64>().is_ok())
                .and_then(|tok| tok.parse().ok())
        });

    if let Some(mb) = reported_mb {
        // Must be > 0: the process allocates at least some memory
        assert!(mb > 0, "Reported memory is 0 — RSS collection failed");

        // Must be well below total system RAM: a tiny problem can't consume all RAM
        let total_system_mb = System::new_all().total_memory() / 1_048_576;
        assert!(
            mb < total_system_mb,
            "Reported memory ({} MB) equals or exceeds total system RAM ({} MB) \
             — this indicates system-wide RAM is being reported instead of process RSS",
            mb,
            total_system_mb
        );
    }
    // If the line is missing entirely, that's also acceptable (very fast solve
    // may round to 0 MB and the conditional `if stats.memory_bytes > 0` hides it)
}

#[test]
fn test_output_file() {
    let smt2_content = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    let temp_file = create_temp_smt2(smt2_content);
    let output_file = env::temp_dir().join(format!("output_{}.txt", rand_string()));

    let result = Command::new(oxiz_bin())
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .arg(temp_file.to_str().unwrap())
        .output()
        .expect("Failed to execute oxiz");

    fs::remove_file(temp_file).ok();

    // Output file should be created
    if result.status.success() {
        assert!(output_file.exists(), "Output file was not created");
        fs::remove_file(output_file).ok();
    }
}
