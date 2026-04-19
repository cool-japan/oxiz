use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, Serialize, Deserialize)]
pub struct Z3ComparisonEntry {
    pub benchmark: String,
    pub oxiz_ms: f64,
    pub z3_ms: Option<f64>,
    pub ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RatioSummary {
    pub geomean_ratio: Option<f64>,
    pub p50_ratio: Option<f64>,
    pub p95_ratio: Option<f64>,
    pub count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Z3ComparisonReport {
    pub entries: Vec<Z3ComparisonEntry>,
    pub z3_available: bool,
    pub z3_version: Option<String>,
    pub within_target: bool,
    #[serde(default)]
    pub geomean_ratio: Option<f64>,
    #[serde(default)]
    pub p50_ratio: Option<f64>,
    #[serde(default)]
    pub p95_ratio: Option<f64>,
    #[serde(default)]
    pub ratio_count: usize,
}

pub fn detect_z3() -> Option<String> {
    let output = Command::new("z3").arg("--version").output().ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

pub fn run_z3_on_smt2(path: &Path, timeout_secs: u64) -> Option<Duration> {
    let start = Instant::now();
    let output = Command::new("z3")
        .arg(path)
        .arg(format!("-T:{timeout_secs}"))
        .output()
        .ok()?;
    if output.status.success() {
        Some(start.elapsed())
    } else {
        None
    }
}

pub fn compute_ratio(oxiz_ms: f64, z3_ms: f64) -> Option<f64> {
    if z3_ms > 0.0 {
        Some(oxiz_ms / z3_ms)
    } else {
        None
    }
}

pub fn summarize_ratios(entries: &[Z3ComparisonEntry]) -> RatioSummary {
    let mut ratios: Vec<f64> = entries.iter().filter_map(|e| e.ratio).collect();
    let count = ratios.len();
    if count == 0 {
        return RatioSummary::default();
    }
    let log_sum: f64 = ratios.iter().map(|&r| r.ln()).sum();
    let geomean = (log_sum / count as f64).exp();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = ratios[count / 2];
    let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
    let p95 = ratios[p95_idx];
    RatioSummary {
        geomean_ratio: Some(geomean),
        p50_ratio: Some(p50),
        p95_ratio: Some(p95),
        count,
    }
}

pub fn compare_with_z3(
    oxiz_timings: &[(String, f64)],
    smt2_paths: &HashMap<String, PathBuf>,
) -> Z3ComparisonReport {
    let z3_version = detect_z3();
    let z3_available = z3_version.is_some();

    let entries = oxiz_timings
        .iter()
        .map(|(benchmark, oxiz_ms)| {
            let z3_ms = if z3_available {
                smt2_paths
                    .get(benchmark)
                    .and_then(|path| run_z3_on_smt2(path, 30))
                    .map(|duration| duration.as_secs_f64() * 1000.0)
            } else {
                None
            };

            let ratio = z3_ms.and_then(|ms| compute_ratio(*oxiz_ms, ms));

            Z3ComparisonEntry {
                benchmark: benchmark.clone(),
                oxiz_ms: *oxiz_ms,
                z3_ms,
                ratio,
            }
        })
        .collect::<Vec<_>>();

    let summary = summarize_ratios(&entries);
    let within_target = summary.geomean_ratio.map(|g| g <= 1.2).unwrap_or(true);

    Z3ComparisonReport {
        entries,
        z3_available,
        z3_version,
        within_target,
        geomean_ratio: summary.geomean_ratio,
        p50_ratio: summary.p50_ratio,
        p95_ratio: summary.p95_ratio,
        ratio_count: summary.count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(benchmark: &str, ratio: Option<f64>) -> Z3ComparisonEntry {
        Z3ComparisonEntry {
            benchmark: benchmark.to_string(),
            oxiz_ms: 1.0,
            z3_ms: ratio.map(|r| 1.0 / r),
            ratio,
        }
    }

    #[test]
    fn test_geomean_basic() {
        let entries = vec![
            make_entry("a", Some(1.0)),
            make_entry("b", Some(2.0)),
            make_entry("c", Some(4.0)),
        ];
        let summary = summarize_ratios(&entries);
        let geomean = summary.geomean_ratio.expect("geomean should be Some");
        assert!((geomean - 2.0).abs() < 1e-9, "geomean([1,2,4]) should be 2.0, got {geomean}");
    }

    #[test]
    fn test_geomean_skips_none() {
        let entries = vec![
            make_entry("a", Some(1.0)),
            make_entry("b", None),
            make_entry("c", Some(4.0)),
        ];
        let summary = summarize_ratios(&entries);
        // Only entries with Some(ratio) are included: [1.0, 4.0]
        // geomean = exp((ln(1) + ln(4)) / 2) = exp(ln(4)/2) = 2.0
        let geomean = summary.geomean_ratio.expect("geomean should be Some");
        assert_eq!(summary.count, 2, "count should only include Some entries");
        assert!((geomean - 2.0).abs() < 1e-9, "geomean([1,4]) should be 2.0, got {geomean}");
    }

    #[test]
    fn test_p50_p95_ordering() {
        let entries = vec![
            make_entry("a", Some(1.0)),
            make_entry("b", Some(2.0)),
            make_entry("c", Some(3.0)),
            make_entry("d", Some(4.0)),
            make_entry("e", Some(100.0)),
        ];
        let summary = summarize_ratios(&entries);
        let p50 = summary.p50_ratio.expect("p50 should be Some");
        let p95 = summary.p95_ratio.expect("p95 should be Some");
        assert!((p50 - 3.0).abs() < f64::EPSILON, "p50 should be index 2 (3.0) for 5 items, got {p50}");
        assert!(p95 >= 50.0, "p95 should be >= 50.0 (near the 100.0 outlier), got {p95}");
    }

    #[test]
    fn test_within_target_uses_geomean() {
        // Individual ratios [2.5, 0.5, 0.5, 0.5, 0.5]
        // geomean = exp((ln(2.5) + 4*ln(0.5)) / 5) which is < 1.2
        // So within_target should be true even though one entry is 2.5x
        let entries = vec![
            make_entry("a", Some(2.5)),
            make_entry("b", Some(0.5)),
            make_entry("c", Some(0.5)),
            make_entry("d", Some(0.5)),
            make_entry("e", Some(0.5)),
        ];
        let report = compare_with_z3(&[], &HashMap::new());
        // Use summarize_ratios directly since compare_with_z3 requires z3 binary
        let summary = summarize_ratios(&entries);
        let geomean = summary.geomean_ratio.expect("geomean should be Some");
        let within_target = geomean <= 1.2;
        assert!(within_target, "geomean {geomean:.4} should be <= 1.2 (single outlier should not fail gate)");
        // Also verify the report from an empty compare uses true (no z3 data)
        assert!(report.within_target, "empty compare_with_z3 should return within_target=true");
    }

    #[test]
    fn test_within_target_true_when_count_zero() {
        let entries = vec![
            make_entry("a", None),
            make_entry("b", None),
        ];
        let summary = summarize_ratios(&entries);
        assert_eq!(summary.count, 0, "count should be 0 when all ratios are None");
        assert!(summary.geomean_ratio.is_none(), "geomean should be None when count is 0");
        // within_target = geomean_ratio.map(|g| g <= 1.2).unwrap_or(true) = true
        let within_target = summary.geomean_ratio.map(|g| g <= 1.2).unwrap_or(true);
        assert!(within_target, "within_target should be true when count=0 (no Z3 data)");
    }
}
