#![cfg(feature = "profiling")]

use oxiz_sat::profiling::{ProfilingCategory, ProfilingStats, ScopedTimer};

#[test]
fn each_category_records_sample() {
    for &category in ProfilingCategory::all() {
        {
            let _timer = ScopedTimer::new(category);
        }
        let snapshot = ProfilingStats::snapshot();
        assert!(
            snapshot.count(category) >= 1,
            "expected at least one sample for {category}"
        );
    }
}

#[test]
fn json_summary_is_parseable() {
    let snapshot = ProfilingStats::snapshot();
    let json = snapshot.to_json();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
    assert!(parsed.is_ok(), "profiling json should parse: {json}");
}
