use std::fs;

fn write_history_snapshot(dir: &std::path::Path, geomean: f64, count: usize) {
    let snapshot = serde_json::json!({
        "schema_version": 1,
        "oxiz_version": "0.2.1",
        "summary": {
            "geomean_ratio": geomean,
            "p50_ratio": geomean,
            "p95_ratio": geomean * 1.2,
            "count": count
        },
        "entries": [],
        "by_logic_summary": {}
    });
    let path = dir.join("2026-04-19_test.json");
    fs::write(path, serde_json::to_string(&snapshot).unwrap()).unwrap();
}

#[test]
fn test_geomean_gate_fails_when_ratio_too_high() {
    let dir = tempfile::tempdir().unwrap();
    write_history_snapshot(dir.path(), 1.5, 10);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_regression"))
        .args(["--check-geomean", "--max", "1.2", "--history-dir"])
        .arg(dir.path())
        .status()
        .expect("binary should run");
    assert_ne!(
        status.code(),
        Some(0),
        "should exit non-zero when geomean 1.5 > max 1.2"
    );
}

#[test]
fn test_geomean_gate_passes_when_ratio_ok() {
    let dir = tempfile::tempdir().unwrap();
    write_history_snapshot(dir.path(), 1.0, 10);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_regression"))
        .args(["--check-geomean", "--max", "1.2", "--history-dir"])
        .arg(dir.path())
        .status()
        .expect("binary should run");
    assert_eq!(
        status.code(),
        Some(0),
        "should exit 0 when geomean 1.0 <= max 1.2"
    );
}

#[test]
fn test_geomean_gate_passes_when_no_z3_data() {
    let dir = tempfile::tempdir().unwrap();
    write_history_snapshot(dir.path(), 0.0, 0); // count=0
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_regression"))
        .args(["--check-geomean", "--max", "1.2", "--history-dir"])
        .arg(dir.path())
        .status()
        .expect("binary should run");
    assert_eq!(
        status.code(),
        Some(0),
        "soft-gate: should pass when count=0"
    );
}
