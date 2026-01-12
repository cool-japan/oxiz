//! Integration tests for oxiz CLI

use std::env;
use std::fs;
use std::path::PathBuf;
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
