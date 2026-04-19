use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Per-entry view for history (matches Z3ComparisonEntry schema in bench/regression)
#[derive(Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub benchmark: String,
    pub logic: String,
    pub oxiz_ms: f64,
    pub z3_ms: Option<f64>,  // None if z3_time is zero (z3 was unavailable)
    pub ratio: Option<f64>,  // oxiz_ms / z3_ms; None if z3_ms is None or 0
}

/// Geomean/p50/p95 summary over a set of ratios
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct RatioSummary {
    pub geomean_ratio: Option<f64>,
    pub p50_ratio: Option<f64>,
    pub p95_ratio: Option<f64>,
    pub count: usize,
}

/// A versioned snapshot of parity results for a single run
#[derive(Debug, Serialize, Deserialize)]
pub struct HistorySnapshot {
    pub schema_version: u32,    // always 1
    pub oxiz_version: String,   // env!("CARGO_PKG_VERSION")
    pub git_sha: String,        // "git rev-parse --short HEAD" or "unknown"
    pub utc_date: String,       // "YYYY-MM-DD" from SystemTime or "unknown"
    pub host_z3_version: Option<String>, // "z3 --version" output or None
    pub entries: Vec<HistoryEntry>,
    pub summary: RatioSummary,
    pub by_logic_summary: HashMap<String, RatioSummary>,
}

/// Compute geomean/p50/p95 from a slice of ratios (None entries are skipped)
pub fn compute_ratio_summary(ratios: &[Option<f64>]) -> RatioSummary {
    let mut valid: Vec<f64> = ratios.iter().filter_map(|r| *r).collect();
    let count = valid.len();
    if count == 0 {
        return RatioSummary::default();
    }
    let log_sum: f64 = valid.iter().map(|&r| r.ln()).sum();
    let geomean = (log_sum / count as f64).exp();
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = valid[count / 2];
    let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
    let p95 = valid[p95_idx];
    RatioSummary {
        geomean_ratio: Some(geomean),
        p50_ratio: Some(p50),
        p95_ratio: Some(p95),
        count,
    }
}

/// Convert ParityResults to HistoryEntries
pub fn to_history_entries(results: &[super::ParityResult]) -> Vec<HistoryEntry> {
    results
        .iter()
        .map(|r| {
            let oxiz_ms = r.oxiz_time.as_secs_f64() * 1000.0;
            // z3_time == Duration::ZERO means Z3 wasn't run (or had error)
            let z3_ms = if r.z3_time == std::time::Duration::ZERO {
                None
            } else {
                Some(r.z3_time.as_secs_f64() * 1000.0)
            };
            let ratio = z3_ms.filter(|&z| z > 0.0).map(|z| oxiz_ms / z);
            HistoryEntry {
                benchmark: r.benchmark.clone(),
                logic: r.logic.clone(),
                oxiz_ms,
                z3_ms,
                ratio,
            }
        })
        .collect()
}

/// Get current date as "YYYY-MM-DD" from SystemTime (no chrono dep)
fn utc_date_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Julian Day Number approach: correct for 1970–2100
    let days_since_epoch = secs / 86400;
    // 1970-01-01 is Julian Day 2440588
    let jdn = days_since_epoch as i64 + 2440588;
    // Convert JDN to Gregorian
    let a = jdn + 32044;
    let b = (4 * a + 3) / 146097;
    let c = a - (146097 * b) / 4;
    let d = (4 * c + 3) / 1461;
    let e = c - (1461 * d) / 4;
    let m = (5 * e + 2) / 153;
    let day = e - (153 * m + 2) / 5 + 1;
    let month = m + 3 - 12 * (m / 10);
    let year = 100 * b + d - 4800 + m / 10;
    format!("{year:04}-{month:02}-{day:02}")
}

/// Get short git SHA ("git rev-parse --short HEAD")
fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Query Z3 version if available
fn z3_version() -> Option<String> {
    let output = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .ok()?;
    if output.status.success() {
        String::from_utf8(output.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

/// Build a HistorySnapshot from parity results
pub fn build_snapshot(results: &[super::ParityResult]) -> HistorySnapshot {
    let entries = to_history_entries(results);
    let all_ratios: Vec<Option<f64>> = entries.iter().map(|e| e.ratio).collect();
    let summary = compute_ratio_summary(&all_ratios);

    // Per-logic summaries
    let mut by_logic: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    for entry in &entries {
        by_logic
            .entry(entry.logic.clone())
            .or_default()
            .push(entry.ratio);
    }
    let by_logic_summary = by_logic
        .into_iter()
        .map(|(logic, ratios)| (logic, compute_ratio_summary(&ratios)))
        .collect();

    HistorySnapshot {
        schema_version: 1,
        oxiz_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: git_sha(),
        utc_date: utc_date_string(),
        host_z3_version: z3_version(),
        entries,
        summary,
        by_logic_summary,
    }
}

/// Export a HistorySnapshot to `<dir>/<date>_<git_sha>.json`
/// Creates the directory if it does not exist.
pub fn export_to_history(
    results: &[super::ParityResult],
    dir: &Path,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let snapshot = build_snapshot(results);
    let filename = format!("{}_{}.json", snapshot.utc_date, snapshot.git_sha);
    let path = dir.join(&filename);
    let json = serde_json::to_string_pretty(&snapshot)?;
    std::fs::write(&path, json)?;
    Ok(path)
}
