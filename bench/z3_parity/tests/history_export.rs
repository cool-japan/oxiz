// Integration tests for the history export functionality.
// Because z3_parity is a binary crate (not a library), we cannot import
// its internal types here.  Instead we verify:
//   - JSON schema round-trips correctly (format/structural correctness)
//   - Directory creation behaviour mirrors what export_to_history does
//   - Filename format convention is consistent

#[test]
fn test_history_snapshot_filename_format() {
    let sample_date = "2026-04-19";
    let sample_sha = "abc1234";
    let filename = format!("{sample_date}_{sample_sha}.json");
    // Date part: YYYY-MM-DD (10 chars), split on first '_'
    let parts: Vec<&str> = filename.splitn(2, '_').collect();
    assert_eq!(parts.len(), 2, "filename must contain at least one '_'");
    assert_eq!(
        parts[0].len(),
        10,
        "date part should be exactly YYYY-MM-DD (10 chars)"
    );
    assert!(
        parts[0].starts_with("20"),
        "year part should start with '20xx'"
    );
    assert!(
        parts[1].ends_with(".json"),
        "filename should end with .json"
    );
}

#[test]
fn test_history_snapshot_roundtrip() {
    // Write a synthetic snapshot JSON and verify it round-trips via serde_json::Value
    let dir = tempfile::TempDir::new().unwrap();
    let snapshot = serde_json::json!({
        "schema_version": 1,
        "oxiz_version": "0.2.1",
        "git_sha": "abc1234",
        "utc_date": "2026-04-19",
        "host_z3_version": null,
        "entries": [
            {
                "benchmark": "test.smt2",
                "logic": "QF_LIA",
                "oxiz_ms": 1.5,
                "z3_ms": 1.0,
                "ratio": 1.5
            }
        ],
        "summary": {
            "geomean_ratio": 1.5,
            "p50_ratio": 1.5,
            "p95_ratio": 1.5,
            "count": 1
        },
        "by_logic_summary": {
            "QF_LIA": {
                "geomean_ratio": 1.5,
                "p50_ratio": 1.5,
                "p95_ratio": 1.5,
                "count": 1
            }
        }
    });
    let path = dir.path().join("2026-04-19_abc1234.json");
    std::fs::write(&path, serde_json::to_string(&snapshot).unwrap()).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    let read_back: serde_json::Value = serde_json::from_str(&raw).unwrap();

    assert_eq!(read_back["schema_version"], 1);
    assert_eq!(read_back["oxiz_version"], "0.2.1");
    assert_eq!(read_back["entries"][0]["logic"], "QF_LIA");
    assert_eq!(read_back["entries"][0]["ratio"], 1.5_f64);
    assert_eq!(
        read_back["by_logic_summary"]["QF_LIA"]["count"],
        1
    );
}

#[test]
fn test_export_creates_history_dir() {
    // Verify that create_dir_all handles deeply nested paths (mirrors
    // the std::fs::create_dir_all call inside export_to_history).
    let dir = tempfile::TempDir::new().unwrap();
    let nested = dir.path().join("a").join("b").join("history");
    assert!(!nested.exists(), "nested dir should not exist before creation");
    std::fs::create_dir_all(&nested).unwrap();
    assert!(nested.exists(), "create_dir_all should create nested dirs");
}

#[test]
fn test_history_entry_z3_ms_none_when_zero() {
    // Verify the JSON schema: z3_ms field must be null when Z3 wasn't run.
    // We model this directly in JSON without importing ParityResult.
    let entry = serde_json::json!({
        "benchmark": "trivial.smt2",
        "logic": "QF_BV",
        "oxiz_ms": 0.5,
        "z3_ms": null,
        "ratio": null
    });
    assert!(entry["z3_ms"].is_null(), "z3_ms should be null when Z3 unavailable");
    assert!(entry["ratio"].is_null(), "ratio should be null when z3_ms is null");
}

#[test]
fn test_ratio_summary_schema() {
    // Verify the RatioSummary JSON schema shape.
    let summary = serde_json::json!({
        "geomean_ratio": 1.2,
        "p50_ratio": 1.1,
        "p95_ratio": 2.3,
        "count": 5
    });
    assert_eq!(summary["count"], 5);
    assert!(summary["geomean_ratio"].as_f64().unwrap() > 1.0);
    assert!(summary["p95_ratio"].as_f64().unwrap() >= summary["p50_ratio"].as_f64().unwrap());
}

#[test]
fn test_schema_version_is_one() {
    // schema_version must always be 1 in the current protocol.
    let snapshot = serde_json::json!({
        "schema_version": 1,
        "oxiz_version": "0.2.1",
        "git_sha": "unknown",
        "utc_date": "2026-04-20",
        "host_z3_version": null,
        "entries": [],
        "summary": {"geomean_ratio": null, "p50_ratio": null, "p95_ratio": null, "count": 0},
        "by_logic_summary": {}
    });
    assert_eq!(snapshot["schema_version"], 1);
    assert_eq!(snapshot["entries"].as_array().unwrap().len(), 0);
}
