//! Integration tests for SV-COMP benchmark suite discovery
//!
//! Uses a synthetic SV-COMP layout in a temporary directory to verify that
//! `SvCompReader::discover` correctly categorises tasks and produces `BenchmarkMeta`.

use oxiz_smtcomp::svcomp::SvCompReader;
use std::fs;
use tempfile::TempDir;

/// Create a temporary directory containing a synthetic SV-COMP benchmark layout:
///
/// - `task_smt2.yml` — valid task pointing at a `.smt2` file (should be accepted)
/// - `task_smt2.smt2` — the SMT-LIB file referenced by `task_smt2.yml`
/// - `task_c.yml` — valid task pointing at a `.c` file (should be skipped: NotSmtLib)
/// - `malformed.yml` — YAML with unbalanced structure that fails parsing (should be skipped: YamlParseError)
fn setup_synthetic_svcomp() -> TempDir {
    let dir = tempfile::TempDir::new().expect("temp dir should be created");
    let root = dir.path();

    // 1. A valid SMT-LIB2 benchmark file
    let smt2_content =
        "(set-logic QF_LIA)\n(declare-fun x () Int)\n(assert (> x 0))\n(check-sat)\n";
    let smt2_path = root.join("benchmark.smt2");
    fs::write(&smt2_path, smt2_content).expect("writing smt2 fixture should succeed");

    // 2. A valid SV-COMP YAML task pointing at the .smt2 file
    let task_smt2_yaml =
        "format_version: \"2.0\"\ninput_files:\n  - benchmark.smt2\nexpected_verdict: \"true\"\n"
            .to_string();
    fs::write(root.join("task_smt2.yml"), &task_smt2_yaml)
        .expect("writing task_smt2.yml should succeed");

    // 3. A valid SV-COMP YAML task pointing at a .c file (no SMT-LIB → skipped)
    let task_c_yaml =
        "format_version: \"2.0\"\ninput_files:\n  - program.c\nexpected_verdict: \"false\"\n";
    fs::write(root.join("task_c.yml"), task_c_yaml).expect("writing task_c.yml should succeed");

    // 4. A malformed YAML file (unterminated flow mapping → parse failure)
    let malformed_yaml = "format_version: {[broken\n";
    fs::write(root.join("malformed.yml"), malformed_yaml)
        .expect("writing malformed.yml should succeed");

    dir
}

#[test]
fn test_discover_succeeds() {
    let dir = setup_synthetic_svcomp();
    let reader =
        SvCompReader::discover(dir.path()).expect("discover should succeed on synthetic layout");
    drop(dir); // explicit: keep dir alive until assertions done

    // We just want to confirm it doesn't error
    let _ = reader;
}

#[test]
fn test_discover_tasks_count() {
    let dir = setup_synthetic_svcomp();
    let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");

    // Exactly one task: the SMT-LIB task
    assert_eq!(
        reader.tasks().len(),
        1,
        "expected exactly 1 accepted task (the SMT-LIB one)"
    );
}

#[test]
fn test_discover_skipped_count() {
    let dir = setup_synthetic_svcomp();
    let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");

    // Two skipped: task_c.yml (NotSmtLib) + malformed.yml (YamlParseError)
    assert_eq!(
        reader.skipped_count(),
        2,
        "expected exactly 2 skipped files: task_c.yml and malformed.yml"
    );
}

#[test]
fn test_discover_to_benchmark_meta_count() {
    let dir = setup_synthetic_svcomp();
    let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");
    let metas = reader.to_benchmark_meta();

    // One meta per .smt2 source → one meta from the single accepted task
    assert_eq!(
        metas.len(),
        1,
        "expected exactly 1 BenchmarkMeta from the SMT-LIB task"
    );
}

#[test]
fn test_discover_task_has_correct_name() {
    let dir = setup_synthetic_svcomp();
    let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");

    let task = &reader.tasks()[0];
    assert_eq!(
        task.name, "task_smt2",
        "task name should be the YAML filename stem"
    );
}

#[test]
fn test_discover_task_expected_verdict_true_maps_to_unsat() {
    use oxiz_smtcomp::loader::ExpectedStatus;

    let dir = setup_synthetic_svcomp();
    let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");

    let task = &reader.tasks()[0];
    assert_eq!(
        task.expected_verdict.as_deref(),
        Some("true"),
        "raw verdict should be 'true'"
    );
    assert_eq!(
        task.expected_status(),
        Some(ExpectedStatus::Unsat),
        "'true' verdict should map to Unsat"
    );
}

#[test]
fn test_discover_meta_category_is_svcomp() {
    let dir = setup_synthetic_svcomp();
    let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");
    let metas = reader.to_benchmark_meta();

    assert_eq!(
        metas[0].category.as_deref(),
        Some("sv-comp"),
        "category should be 'sv-comp'"
    );
}

#[test]
fn test_discover_meta_logic_is_none() {
    let dir = setup_synthetic_svcomp();
    let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");
    let metas = reader.to_benchmark_meta();

    assert!(
        metas[0].logic.is_none(),
        "logic should be None (discovered via path, not parsed)"
    );
}

#[test]
fn test_discover_empty_directory() {
    let dir = tempfile::TempDir::new().expect("temp dir should be created");
    let reader = SvCompReader::discover(dir.path()).expect("discover on empty dir should succeed");
    assert_eq!(reader.tasks().len(), 0);
    assert_eq!(reader.skipped_count(), 0);
}

#[test]
fn test_discover_nonexistent_root_returns_error() {
    let result = SvCompReader::discover(std::path::Path::new(
        "/nonexistent/svcomp/path/oxiz-test-99999",
    ));
    assert!(result.is_err(), "discover on nonexistent root should fail");
}

#[test]
fn test_discover_yaml_extension_case_insensitive() {
    let dir = tempfile::TempDir::new().expect("temp dir should be created");
    let root = dir.path();

    // Write a .smt2 file
    fs::write(root.join("bench.smt2"), "(set-logic QF_LIA)\n(check-sat)\n")
        .expect("write smt2 should succeed");

    // Use .YAML (uppercase) extension
    let yaml = "format_version: \"2.0\"\ninput_files:\n  - bench.smt2\n";
    fs::write(root.join("task.YAML"), yaml).expect("write task.YAML should succeed");

    let reader = SvCompReader::discover(root).expect("discover should succeed");
    assert_eq!(
        reader.tasks().len(),
        1,
        "uppercase .YAML should be accepted"
    );
}

#[test]
fn test_discover_yaml_recursive() {
    let dir = tempfile::TempDir::new().expect("temp dir should be created");
    let root = dir.path();

    // Create subdirectory
    let sub = root.join("category1");
    fs::create_dir_all(&sub).expect("create sub dir should succeed");

    fs::write(sub.join("bench.smt2"), "(check-sat)\n").expect("write smt2 should succeed");
    let yaml = "format_version: \"2.0\"\ninput_files:\n  - bench.smt2\n";
    fs::write(sub.join("task.yml"), yaml).expect("write task.yml should succeed");

    let reader = SvCompReader::discover(root).expect("discover should succeed");
    assert_eq!(
        reader.tasks().len(),
        1,
        "recursive discovery should find tasks in subdirectories"
    );
}
