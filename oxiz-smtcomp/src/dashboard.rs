//! Web dashboard for result visualization
//!
//! This module provides functionality to generate a simple web dashboard
//! for visualizing benchmark results using HTML/JS.

use crate::benchmark::{RunSummary, SingleResult};
use crate::statistics::FullAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use thiserror::Error;

/// Error type for dashboard operations
#[derive(Error, Debug)]
pub enum DashboardError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for dashboard operations
pub type DashboardResult<T> = Result<T, DashboardError>;

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard title
    pub title: String,
    /// Auto-refresh interval in seconds (0 = disabled)
    pub refresh_interval: u32,
    /// Show individual benchmarks table
    pub show_details: bool,
    /// Show cactus plot
    pub show_cactus: bool,
    /// Show per-logic breakdown
    pub show_logic_breakdown: bool,
    /// Maximum detail rows to show
    pub max_detail_rows: usize,
    /// Theme (light/dark)
    pub theme: String,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            title: "SMT Benchmark Dashboard".to_string(),
            refresh_interval: 0,
            show_details: true,
            show_cactus: true,
            show_logic_breakdown: true,
            max_detail_rows: 500,
            theme: "light".to_string(),
        }
    }
}

impl DashboardConfig {
    /// Create config with title
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Enable auto-refresh
    #[must_use]
    pub fn with_refresh(mut self, seconds: u32) -> Self {
        self.refresh_interval = seconds;
        self
    }

    /// Use dark theme
    #[must_use]
    pub fn with_dark_theme(mut self) -> Self {
        self.theme = "dark".to_string();
        self
    }
}

/// Dashboard data for JSON export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    /// Summary statistics
    pub summary: RunSummary,
    /// Per-logic statistics
    pub by_logic: HashMap<String, LogicData>,
    /// Cactus plot data
    pub cactus_data: Vec<[f64; 2]>,
    /// Recent results (limited)
    pub recent_results: Vec<ResultData>,
    /// Generation timestamp
    pub timestamp: u64,
}

/// Logic-specific data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicData {
    /// Logic name
    pub logic: String,
    /// Total benchmarks
    pub total: usize,
    /// Solved
    pub solved: usize,
    /// SAT
    pub sat: usize,
    /// UNSAT
    pub unsat: usize,
    /// Timeout
    pub timeout: usize,
    /// Error
    pub error: usize,
    /// Solve rate
    pub solve_rate: f64,
}

/// Result data for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultData {
    /// Benchmark name
    pub name: String,
    /// Logic
    pub logic: Option<String>,
    /// Status
    pub status: String,
    /// Time in seconds
    pub time_secs: f64,
    /// Correct
    pub correct: Option<bool>,
}

impl DashboardData {
    /// Create from results
    #[must_use]
    pub fn from_results(results: &[SingleResult], max_recent: usize) -> Self {
        let analysis = FullAnalysis::from_results(results);
        let cactus = crate::statistics::cactus_data(results);

        let by_logic: HashMap<_, _> = analysis
            .by_logic
            .iter()
            .map(|(logic, stats)| {
                (
                    logic.clone(),
                    LogicData {
                        logic: logic.clone(),
                        total: stats.total,
                        solved: stats.solved,
                        sat: stats.sat,
                        unsat: stats.unsat,
                        timeout: stats.timeouts,
                        error: stats.errors,
                        solve_rate: stats.solve_rate(),
                    },
                )
            })
            .collect();

        let cactus_data: Vec<[f64; 2]> = cactus.iter().map(|p| [p.solved as f64, p.time]).collect();

        let recent_results: Vec<_> = results
            .iter()
            .take(max_recent)
            .map(|r| ResultData {
                name: r
                    .path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                logic: r.logic.clone(),
                status: r.status.as_str().to_string(),
                time_secs: r.time.as_secs_f64(),
                correct: r.correct,
            })
            .collect();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            summary: analysis.summary,
            by_logic,
            cactus_data,
            recent_results,
            timestamp,
        }
    }
}

/// Dashboard generator
pub struct DashboardGenerator {
    config: DashboardConfig,
}

impl DashboardGenerator {
    /// Create a new generator
    #[must_use]
    pub fn new(config: DashboardConfig) -> Self {
        Self { config }
    }

    /// Generate dashboard HTML
    pub fn generate(&self, results: &[SingleResult]) -> DashboardResult<String> {
        let data = DashboardData::from_results(results, self.config.max_detail_rows);
        let data_json = serde_json::to_string(&data)?;

        let theme_vars = if self.config.theme == "dark" {
            r#"
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --border-color: #0f3460;
            --success-color: #4ade80;
            --danger-color: #f87171;
            --warning-color: #fbbf24;
            --info-color: #60a5fa;
            "#
        } else {
            r#"
            --bg-color: #f5f5f5;
            --card-bg: #ffffff;
            --text-color: #333;
            --border-color: #ddd;
            --success-color: #22c55e;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #3b82f6;
            "#
        };

        let refresh_meta = if self.config.refresh_interval > 0 {
            format!(
                r#"<meta http-equiv="refresh" content="{}">"#,
                self.config.refresh_interval
            )
        } else {
            String::new()
        };

        Ok(format!(
            r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {refresh_meta}
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{ {theme_vars} }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }}
        .header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 1rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .card h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: var(--text-color); opacity: 0.8; }}
        .stat {{ text-align: center; }}
        .stat .value {{ font-size: 2.5rem; font-weight: bold; }}
        .stat .label {{ font-size: 0.9rem; opacity: 0.7; }}
        .stat.success .value {{ color: var(--success-color); }}
        .stat.danger .value {{ color: var(--danger-color); }}
        .stat.warning .value {{ color: var(--warning-color); }}
        .stat.info .value {{ color: var(--info-color); }}
        .chart-container {{ position: relative; height: 300px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ font-weight: 600; opacity: 0.8; }}
        .status-sat {{ color: var(--success-color); }}
        .status-unsat {{ color: var(--info-color); }}
        .status-timeout {{ color: var(--warning-color); }}
        .status-error {{ color: var(--danger-color); }}
        .badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }}
        .badge-success {{ background: var(--success-color); color: white; }}
        .badge-danger {{ background: var(--danger-color); color: white; }}
        .progress-bar {{ height: 8px; background: var(--border-color); border-radius: 4px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: var(--success-color); transition: width 0.5s; }}
        footer {{ text-align: center; padding: 2rem; opacity: 0.6; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Last updated: <span id="timestamp"></span></p>
    </div>

    <div class="container">
        <div class="grid">
            <div class="card stat">
                <div class="value" id="total"></div>
                <div class="label">Total Benchmarks</div>
            </div>
            <div class="card stat success">
                <div class="value" id="solved"></div>
                <div class="label">Solved</div>
            </div>
            <div class="card stat info">
                <div class="value" id="rate"></div>
                <div class="label">Solve Rate</div>
            </div>
            <div class="card stat warning">
                <div class="value" id="timeout"></div>
                <div class="label">Timeout</div>
            </div>
        </div>

        <div class="grid" style="grid-template-columns: 1fr 1fr;">
            <div class="card">
                <h2>Cactus Plot</h2>
                <div class="chart-container">
                    <canvas id="cactusChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>Results by Logic</h2>
                <div class="chart-container">
                    <canvas id="logicChart"></canvas>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Recent Results</h2>
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Logic</th>
                        <th>Status</th>
                        <th>Time</th>
                        <th>Correct</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <footer>
        Generated by OxiZ SMT-COMP Infrastructure
    </footer>

    <script>
    const data = {data_json};

    // Update stats
    document.getElementById('total').textContent = data.summary.total;
    document.getElementById('solved').textContent = data.summary.sat + data.summary.unsat;
    document.getElementById('rate').textContent = ((data.summary.sat + data.summary.unsat) / data.summary.total * 100).toFixed(1) + '%';
    document.getElementById('timeout').textContent = data.summary.timeouts;
    document.getElementById('timestamp').textContent = new Date(data.timestamp * 1000).toLocaleString();

    // Cactus chart
    new Chart(document.getElementById('cactusChart'), {{
        type: 'line',
        data: {{
            labels: data.cactus_data.map(p => p[0]),
            datasets: [{{
                label: 'Solved',
                data: data.cactus_data.map(p => ({{ x: p[0], y: p[1] }})),
                borderColor: '#667eea',
                fill: false,
                tension: 0.1
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{ title: {{ display: true, text: 'Solved' }} }},
                y: {{ title: {{ display: true, text: 'Time (s)' }} }}
            }}
        }}
    }});

    // Logic chart
    const logics = Object.values(data.by_logic);
    new Chart(document.getElementById('logicChart'), {{
        type: 'bar',
        data: {{
            labels: logics.map(l => l.logic),
            datasets: [
                {{ label: 'SAT', data: logics.map(l => l.sat), backgroundColor: '#22c55e' }},
                {{ label: 'UNSAT', data: logics.map(l => l.unsat), backgroundColor: '#3b82f6' }},
                {{ label: 'Timeout', data: logics.map(l => l.timeout), backgroundColor: '#f59e0b' }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{ x: {{ stacked: true }}, y: {{ stacked: true }} }}
        }}
    }});

    // Results table
    const tbody = document.querySelector('#resultsTable tbody');
    data.recent_results.forEach(r => {{
        const tr = document.createElement('tr');
        const statusClass = 'status-' + r.status;
        const correctBadge = r.correct === true ? '<span class="badge badge-success">OK</span>' :
                            r.correct === false ? '<span class="badge badge-danger">WRONG</span>' : '-';
        tr.innerHTML = `
            <td>${{r.name}}</td>
            <td>${{r.logic || '-'}}</td>
            <td class="${{statusClass}}">${{r.status}}</td>
            <td>${{r.time_secs.toFixed(3)}}s</td>
            <td>${{correctBadge}}</td>
        `;
        tbody.appendChild(tr);
    }});
    </script>
</body>
</html>
"##,
            title = self.config.title,
            refresh_meta = refresh_meta,
            theme_vars = theme_vars,
            data_json = data_json,
        ))
    }

    /// Write dashboard to file
    pub fn write_to_file(
        &self,
        results: &[SingleResult],
        path: impl AsRef<Path>,
    ) -> DashboardResult<()> {
        let html = self.generate(results)?;
        fs::write(path, html)?;
        Ok(())
    }

    /// Generate and write data JSON file (for AJAX updates)
    pub fn write_data_file(
        &self,
        results: &[SingleResult],
        path: impl AsRef<Path>,
    ) -> DashboardResult<()> {
        let data = DashboardData::from_results(results, self.config.max_detail_rows);
        let json = serde_json::to_string_pretty(&data)?;
        fs::write(path, json)?;
        Ok(())
    }
}

/// Generate a simple static dashboard
pub fn generate_dashboard(results: &[SingleResult], title: &str) -> DashboardResult<String> {
    let config = DashboardConfig::new(title);
    let generator = DashboardGenerator::new(config);
    generator.generate(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkStatus;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_result(name: &str, status: BenchmarkStatus, time_ms: u64) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/{}", name)),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    #[test]
    fn test_dashboard_config() {
        let config = DashboardConfig::new("Test Dashboard")
            .with_refresh(30)
            .with_dark_theme();

        assert_eq!(config.title, "Test Dashboard");
        assert_eq!(config.refresh_interval, 30);
        assert_eq!(config.theme, "dark");
    }

    #[test]
    fn test_dashboard_data() {
        let results = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 100),
            make_result("b.smt2", BenchmarkStatus::Unsat, 200),
            make_result("c.smt2", BenchmarkStatus::Timeout, 60000),
        ];

        let data = DashboardData::from_results(&results, 10);

        assert_eq!(data.summary.total, 3);
        assert_eq!(data.recent_results.len(), 3);
        assert!(!data.cactus_data.is_empty());
    }

    #[test]
    fn test_dashboard_generation() {
        let results = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 100),
            make_result("b.smt2", BenchmarkStatus::Unsat, 200),
        ];

        let config = DashboardConfig::new("Test");
        let generator = DashboardGenerator::new(config);
        let html = generator.generate(&results).unwrap();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test"));
        assert!(html.contains("cactusChart"));
    }

    #[test]
    fn test_dark_theme() {
        let results = vec![make_result("a.smt2", BenchmarkStatus::Sat, 100)];

        let config = DashboardConfig::new("Dark Test").with_dark_theme();
        let generator = DashboardGenerator::new(config);
        let html = generator.generate(&results).unwrap();

        assert!(html.contains("#1a1a2e")); // Dark theme color
    }

    #[test]
    fn test_auto_refresh() {
        let results = vec![make_result("a.smt2", BenchmarkStatus::Sat, 100)];

        let config = DashboardConfig::new("Refresh Test").with_refresh(60);
        let generator = DashboardGenerator::new(config);
        let html = generator.generate(&results).unwrap();

        assert!(html.contains("http-equiv=\"refresh\""));
        assert!(html.contains("content=\"60\""));
    }
}
