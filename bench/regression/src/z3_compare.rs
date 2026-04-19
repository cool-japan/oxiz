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

#[derive(Debug, Serialize, Deserialize)]
pub struct Z3ComparisonReport {
    pub entries: Vec<Z3ComparisonEntry>,
    pub z3_available: bool,
    pub z3_version: Option<String>,
    pub within_target: bool,
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

    let within_target = entries
        .iter()
        .filter_map(|entry| entry.ratio)
        .all(|ratio| ratio <= 1.2);

    Z3ComparisonReport {
        entries,
        z3_available,
        z3_version,
        within_target,
    }
}
