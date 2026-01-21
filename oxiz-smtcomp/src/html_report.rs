//! HTML report generation with tables and charts
//!
//! This module provides functionality to generate comprehensive HTML reports
//! for benchmark results, including tables, charts, and statistics.

use crate::benchmark::{BenchmarkStatus, RunSummary, SingleResult};
use crate::plotting::{SvgPlot, generate_cactus_plot};
use crate::statistics::{CategoryStats, FullAnalysis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use thiserror::Error;

/// Error type for HTML report operations
#[derive(Error, Debug)]
pub enum HtmlReportError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Plot generation error
    #[error("Plot error: {0}")]
    PlotError(#[from] crate::plotting::PlotError),
}

/// Result type for HTML report operations
pub type HtmlReportResult<T> = Result<T, HtmlReportError>;

/// Configuration for HTML report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlReportConfig {
    /// Report title
    pub title: String,
    /// Solver name
    pub solver_name: String,
    /// Solver version
    pub solver_version: String,
    /// Include individual benchmark results table
    pub include_details: bool,
    /// Include cactus plot
    pub include_cactus: bool,
    /// Include per-logic breakdown
    pub include_logic_breakdown: bool,
    /// Include difficulty analysis
    pub include_difficulty: bool,
    /// Maximum results to show in details table
    pub max_detail_rows: usize,
    /// CSS theme (light/dark)
    pub theme: String,
}

impl Default for HtmlReportConfig {
    fn default() -> Self {
        Self {
            title: "SMT Benchmark Results".to_string(),
            solver_name: "OxiZ".to_string(),
            solver_version: env!("CARGO_PKG_VERSION").to_string(),
            include_details: true,
            include_cactus: true,
            include_logic_breakdown: true,
            include_difficulty: true,
            max_detail_rows: 1000,
            theme: "light".to_string(),
        }
    }
}

impl HtmlReportConfig {
    /// Create a new config with title
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Set solver info
    #[must_use]
    pub fn with_solver(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.solver_name = name.into();
        self.solver_version = version.into();
        self
    }

    /// Set detail level
    #[must_use]
    pub fn with_details(mut self, include: bool) -> Self {
        self.include_details = include;
        self
    }
}

/// HTML report generator
pub struct HtmlReportGenerator {
    config: HtmlReportConfig,
}

impl HtmlReportGenerator {
    /// Create a new generator
    #[must_use]
    pub fn new(config: HtmlReportConfig) -> Self {
        Self { config }
    }

    /// Generate HTML report from results
    pub fn generate(&self, results: &[SingleResult]) -> HtmlReportResult<String> {
        let analysis = FullAnalysis::from_results(results);
        let mut html = String::new();

        // HTML header
        html.push_str(&self.generate_header());

        // Navigation
        html.push_str(&self.generate_nav());

        // Main content
        html.push_str(r#"<main class="container">"#);

        // Summary section
        html.push_str(&self.generate_summary_section(&analysis.summary));

        // Cactus plot
        if self.config.include_cactus {
            html.push_str(&self.generate_cactus_section(results)?);
        }

        // Logic breakdown
        if self.config.include_logic_breakdown && analysis.by_logic.len() > 1 {
            html.push_str(&self.generate_logic_section(&analysis.by_logic));
        }

        // Difficulty analysis
        if self.config.include_difficulty {
            html.push_str(&self.generate_difficulty_section(&analysis.by_difficulty));
        }

        // Detailed results
        if self.config.include_details {
            html.push_str(&self.generate_details_section(results));
        }

        html.push_str("</main>");

        // Footer
        html.push_str(&self.generate_footer());

        Ok(html)
    }

    /// Generate HTML header
    fn generate_header(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        :root {{
            --bg-color: #ffffff;
            --text-color: #333333;
            --card-bg: #f8f9fa;
            --border-color: #dee2e6;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            background: var(--bg-color);
            color: var(--text-color);
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        nav {{ background: #343a40; color: white; padding: 1rem; margin-bottom: 2rem; }}
        nav h1 {{ font-size: 1.5rem; }}
        .card {{ background: var(--card-bg); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ margin-bottom: 1rem; color: #495057; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }}
        .stat-box {{ text-align: center; padding: 1rem; background: white; border-radius: 8px; }}
        .stat-box .value {{ font-size: 2rem; font-weight: bold; }}
        .stat-box .label {{ color: #6c757d; font-size: 0.875rem; }}
        .stat-box.success .value {{ color: var(--success-color); }}
        .stat-box.danger .value {{ color: var(--danger-color); }}
        .stat-box.warning .value {{ color: var(--warning-color); }}
        .stat-box.info .value {{ color: var(--info-color); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: #e9ecef; font-weight: 600; }}
        tr:hover {{ background: #f1f3f5; }}
        .status-sat {{ color: var(--success-color); }}
        .status-unsat {{ color: var(--info-color); }}
        .status-timeout {{ color: var(--warning-color); }}
        .status-error {{ color: var(--danger-color); }}
        .correct {{ color: var(--success-color); }}
        .wrong {{ color: var(--danger-color); font-weight: bold; }}
        .plot-container {{ text-align: center; margin: 1rem 0; }}
        .progress-bar {{ height: 24px; background: #e9ecef; border-radius: 4px; overflow: hidden; }}
        .progress-fill {{ height: 100%; transition: width 0.3s; }}
        footer {{ text-align: center; padding: 2rem; color: #6c757d; }}
    </style>
</head>
<body>
"#,
            self.config.title
        )
    }

    /// Generate navigation
    fn generate_nav(&self) -> String {
        format!(
            r#"<nav>
    <div class="container">
        <h1>{}</h1>
        <p>{} v{}</p>
    </div>
</nav>
"#,
            self.config.title, self.config.solver_name, self.config.solver_version
        )
    }

    /// Generate summary section
    fn generate_summary_section(&self, summary: &RunSummary) -> String {
        let solve_rate = summary.solve_rate();
        let sound_class = if summary.is_sound() {
            "success"
        } else {
            "danger"
        };

        format!(
            r#"<section class="card" id="summary">
    <h2>Summary</h2>
    <div class="stats-grid">
        <div class="stat-box">
            <div class="value">{}</div>
            <div class="label">Total Benchmarks</div>
        </div>
        <div class="stat-box success">
            <div class="value">{}</div>
            <div class="label">Solved</div>
        </div>
        <div class="stat-box info">
            <div class="value">{:.1}%</div>
            <div class="label">Solve Rate</div>
        </div>
        <div class="stat-box">
            <div class="value">{}</div>
            <div class="label">SAT</div>
        </div>
        <div class="stat-box">
            <div class="value">{}</div>
            <div class="label">UNSAT</div>
        </div>
        <div class="stat-box warning">
            <div class="value">{}</div>
            <div class="label">Timeout</div>
        </div>
        <div class="stat-box danger">
            <div class="value">{}</div>
            <div class="label">Errors</div>
        </div>
        <div class="stat-box {}">
            <div class="value">{}</div>
            <div class="label">Sound</div>
        </div>
    </div>
    <div style="margin-top: 1rem;">
        <p>Total time: {:.2}s | Average: {:.3}s per benchmark</p>
    </div>
</section>
"#,
            summary.total,
            summary.solved(),
            solve_rate,
            summary.sat,
            summary.unsat,
            summary.timeouts,
            summary.errors,
            sound_class,
            if summary.is_sound() { "Yes" } else { "NO" },
            summary.total_time.as_secs_f64(),
            summary.avg_time.as_secs_f64()
        )
    }

    /// Generate cactus plot section
    fn generate_cactus_section(&self, results: &[SingleResult]) -> HtmlReportResult<String> {
        let mut plot = SvgPlot::cactus("Cactus Plot");
        plot.add_results(&self.config.solver_name, results);

        let svg = plot.to_svg()?;

        Ok(format!(
            r#"<section class="card" id="cactus">
    <h2>Cactus Plot</h2>
    <div class="plot-container">
        {}
    </div>
</section>
"#,
            svg
        ))
    }

    /// Generate logic breakdown section
    fn generate_logic_section(&self, by_logic: &HashMap<String, CategoryStats>) -> String {
        let mut logics: Vec<_> = by_logic.iter().collect();
        logics.sort_by_key(|(k, _)| k.as_str());

        let mut rows = String::new();
        for (logic, stats) in logics {
            let solve_rate = stats.solve_rate();
            rows.push_str(&format!(
                r#"<tr>
    <td>{}</td>
    <td>{}</td>
    <td>{}</td>
    <td>{:.1}%</td>
    <td>{}</td>
    <td>{}</td>
    <td>{}</td>
    <td>{}</td>
</tr>
"#,
                logic,
                stats.total,
                stats.solved,
                solve_rate,
                stats.sat,
                stats.unsat,
                stats.timeouts,
                stats.errors
            ));
        }

        format!(
            r#"<section class="card" id="by-logic">
    <h2>Results by Logic</h2>
    <table>
        <thead>
            <tr>
                <th>Logic</th>
                <th>Total</th>
                <th>Solved</th>
                <th>Rate</th>
                <th>SAT</th>
                <th>UNSAT</th>
                <th>Timeout</th>
                <th>Error</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
    </table>
</section>
"#,
            rows
        )
    }

    /// Generate difficulty analysis section
    fn generate_difficulty_section(
        &self,
        difficulty: &crate::statistics::DifficultyAnalysis,
    ) -> String {
        format!(
            r#"<section class="card" id="difficulty">
    <h2>Difficulty Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Difficulty</th>
                <th>Time Range</th>
                <th>Total</th>
                <th>Solved</th>
                <th>Rate</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Easy</td>
                <td>&lt; 1s</td>
                <td>{}</td>
                <td>{}</td>
                <td>{:.1}%</td>
            </tr>
            <tr>
                <td>Medium</td>
                <td>1-10s</td>
                <td>{}</td>
                <td>{}</td>
                <td>{:.1}%</td>
            </tr>
            <tr>
                <td>Hard</td>
                <td>10-60s</td>
                <td>{}</td>
                <td>{}</td>
                <td>{:.1}%</td>
            </tr>
            <tr>
                <td>Very Hard</td>
                <td>&gt; 60s</td>
                <td>{}</td>
                <td>{}</td>
                <td>{:.1}%</td>
            </tr>
        </tbody>
    </table>
</section>
"#,
            difficulty.easy.total,
            difficulty.easy.solved,
            difficulty.easy.solve_rate(),
            difficulty.medium.total,
            difficulty.medium.solved,
            difficulty.medium.solve_rate(),
            difficulty.hard.total,
            difficulty.hard.solved,
            difficulty.hard.solve_rate(),
            difficulty.very_hard.total,
            difficulty.very_hard.solved,
            difficulty.very_hard.solve_rate()
        )
    }

    /// Generate detailed results section
    fn generate_details_section(&self, results: &[SingleResult]) -> String {
        let display_count = results.len().min(self.config.max_detail_rows);
        let mut rows = String::new();

        for result in results.iter().take(display_count) {
            let filename = result
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            let status_class = match result.status {
                BenchmarkStatus::Sat => "status-sat",
                BenchmarkStatus::Unsat => "status-unsat",
                BenchmarkStatus::Timeout => "status-timeout",
                BenchmarkStatus::Error => "status-error",
                _ => "",
            };

            let correct_cell = match result.correct {
                Some(true) => r#"<span class="correct">OK</span>"#,
                Some(false) => r#"<span class="wrong">WRONG</span>"#,
                None => "-",
            };

            rows.push_str(&format!(
                r#"<tr>
    <td title="{}">{}</td>
    <td>{}</td>
    <td class="{}">{}</td>
    <td>{}</td>
    <td>{:.3}s</td>
</tr>
"#,
                result.path.display(),
                filename,
                result.logic.as_deref().unwrap_or("-"),
                status_class,
                result.status.as_str(),
                correct_cell,
                result.time.as_secs_f64()
            ));
        }

        let truncation_note = if results.len() > display_count {
            format!(
                r#"<p style="margin-top: 1rem; color: #6c757d;">Showing {} of {} results</p>"#,
                display_count,
                results.len()
            )
        } else {
            String::new()
        };

        format!(
            r#"<section class="card" id="details">
    <h2>Individual Results</h2>
    <table>
        <thead>
            <tr>
                <th>Benchmark</th>
                <th>Logic</th>
                <th>Status</th>
                <th>Correct</th>
                <th>Time</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
    </table>
    {}
</section>
"#,
            rows, truncation_note
        )
    }

    /// Generate footer
    fn generate_footer(&self) -> String {
        let now = chrono_lite();
        format!(
            r#"<footer>
    <p>Generated by {} v{} on {}</p>
</footer>
</body>
</html>
"#,
            self.config.solver_name, self.config.solver_version, now
        )
    }

    /// Write report to file
    pub fn write_to_file(
        &self,
        results: &[SingleResult],
        path: impl AsRef<Path>,
    ) -> HtmlReportResult<()> {
        let html = self.generate(results)?;
        fs::write(path, html)?;
        Ok(())
    }
}

/// Simple date/time formatting without chrono dependency
fn chrono_lite() -> String {
    use std::time::SystemTime;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();

    // Simple formatting: just show seconds since epoch for now
    // In production, you'd use chrono or time crate
    let secs = now.as_secs();
    let days = secs / 86400;
    let years_since_1970 = days / 365;
    let year = 1970 + years_since_1970;

    format!("{}", year)
}

/// Generate a comparison report for multiple solvers
pub fn generate_comparison_report(
    solver_results: &HashMap<String, Vec<SingleResult>>,
    title: &str,
) -> HtmlReportResult<String> {
    let mut html = String::new();

    // Header
    html.push_str(&format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{}</title>
    <style>
        body {{ font-family: sans-serif; margin: 2rem; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f4f4f4; }}
        .best {{ font-weight: bold; color: #28a745; }}
    </style>
</head>
<body>
<h1>{}</h1>
"#,
        title, title
    ));

    // Cactus plot
    if let Ok(svg) = generate_cactus_plot(solver_results, "Cactus Plot Comparison") {
        html.push_str(&format!(r#"<div style="margin: 2rem 0;">{}</div>"#, svg));
    }

    // Comparison table
    html.push_str("<h2>Summary Comparison</h2><table><tr><th>Solver</th><th>Solved</th><th>SAT</th><th>UNSAT</th><th>Timeout</th><th>Rate</th></tr>");

    let mut best_solved = 0;
    for results in solver_results.values() {
        let summary = RunSummary::from_results(results);
        best_solved = best_solved.max(summary.solved());
    }

    for (name, results) in solver_results {
        let summary = RunSummary::from_results(results);
        let best_class = if summary.solved() == best_solved {
            r#" class="best""#
        } else {
            ""
        };
        html.push_str(&format!(
            "<tr><td>{}</td><td{}>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td></tr>",
            name,
            best_class,
            summary.solved(),
            summary.sat,
            summary.unsat,
            summary.timeouts,
            summary.solve_rate()
        ));
    }

    html.push_str("</table></body></html>");

    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_result(status: BenchmarkStatus, time_ms: u64) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/test_{}.smt2", time_ms)),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    #[test]
    fn test_html_report_config() {
        let config = HtmlReportConfig::new("Test Report")
            .with_solver("TestSolver", "1.0.0")
            .with_details(false);

        assert_eq!(config.title, "Test Report");
        assert_eq!(config.solver_name, "TestSolver");
        assert!(!config.include_details);
    }

    #[test]
    fn test_html_report_generation() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, 100),
            make_result(BenchmarkStatus::Unsat, 200),
            make_result(BenchmarkStatus::Timeout, 60000),
        ];

        let config = HtmlReportConfig::new("Test Report");
        let generator = HtmlReportGenerator::new(config);
        let html = generator.generate(&results).unwrap();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Report"));
        assert!(html.contains("Total Benchmarks"));
    }

    #[test]
    fn test_comparison_report() {
        let mut solver_results = HashMap::new();
        solver_results.insert(
            "Solver A".to_string(),
            vec![
                make_result(BenchmarkStatus::Sat, 100),
                make_result(BenchmarkStatus::Sat, 200),
            ],
        );
        solver_results.insert(
            "Solver B".to_string(),
            vec![
                make_result(BenchmarkStatus::Sat, 150),
                make_result(BenchmarkStatus::Timeout, 60000),
            ],
        );

        let html = generate_comparison_report(&solver_results, "Solver Comparison").unwrap();
        assert!(html.contains("Solver A"));
        assert!(html.contains("Solver B"));
    }
}
