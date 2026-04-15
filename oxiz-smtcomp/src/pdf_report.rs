//! PDF report generation for SMT-COMP benchmark results.
//!
//! This module mirrors the high-level API of [`crate::html_report`] but emits a
//! compact PDF summary using the Pure Rust [`pdf_writer`] crate. It is gated
//! behind the `pdf-report` Cargo feature so default builds stay lean.
//!
//! # Layout
//!
//! The generated document uses a simple tabular layout rendered with the
//! standard `Helvetica` Type 1 font (no embedded fonts required):
//!
//! 1. A summary page with the overall totals (SAT / UNSAT / timeout / error,
//!    solve rate, total and average time).
//! 2. An optional per-logic breakdown (total / solved / rate / SAT / UNSAT /
//!    timeout / error columns).
//! 3. An optional per-benchmark table, clipped to
//!    [`PdfReportConfig::max_detail_rows`] (default 100). A truncation note is
//!    emitted when entries are omitted.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "pdf-report")] {
//! use oxiz_smtcomp::pdf_report::{PdfReportConfig, PdfReportGenerator};
//! use oxiz_smtcomp::SingleResult;
//!
//! let results: Vec<SingleResult> = Vec::new();
//! let config = PdfReportConfig::new("OxiZ Benchmark Run");
//! let generator = PdfReportGenerator::new(config);
//! generator
//!     .write_to_file(&results, "/tmp/oxiz_report.pdf")
//!     .expect("write pdf report");
//! # }
//! ```

use crate::benchmark::{RunSummary, SingleResult};
use crate::statistics::{CategoryStats, FullAnalysis};
use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref, Str};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;
use thiserror::Error;

/// Error type for PDF report operations.
#[derive(Error, Debug)]
pub enum PdfReportError {
    /// IO error while writing the PDF buffer.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

/// Result alias for PDF report operations.
pub type PdfReportResult<T> = Result<T, PdfReportError>;

/// Configuration for the PDF report generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfReportConfig {
    /// Report title, rendered at the top of the summary page.
    pub title: String,
    /// Solver name, rendered in the document header and footer.
    pub solver_name: String,
    /// Solver version, rendered beside the solver name.
    pub solver_version: String,
    /// Include the per-logic breakdown table.
    pub include_logic_breakdown: bool,
    /// Include the per-benchmark details table.
    pub include_details: bool,
    /// Maximum number of rows displayed in the per-benchmark details table.
    pub max_detail_rows: usize,
}

impl Default for PdfReportConfig {
    fn default() -> Self {
        Self {
            title: "SMT Benchmark Results".to_string(),
            solver_name: "OxiZ".to_string(),
            solver_version: env!("CARGO_PKG_VERSION").to_string(),
            include_logic_breakdown: true,
            include_details: true,
            max_detail_rows: 100,
        }
    }
}

impl PdfReportConfig {
    /// Create a new configuration with a custom title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Set the solver identification fields.
    #[must_use]
    pub fn with_solver(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.solver_name = name.into();
        self.solver_version = version.into();
        self
    }

    /// Toggle inclusion of the per-logic breakdown table.
    #[must_use]
    pub fn with_logic_breakdown(mut self, include: bool) -> Self {
        self.include_logic_breakdown = include;
        self
    }

    /// Toggle inclusion of the per-benchmark details table.
    #[must_use]
    pub fn with_details(mut self, include: bool) -> Self {
        self.include_details = include;
        self
    }

    /// Set the maximum number of rows displayed in the per-benchmark table.
    #[must_use]
    pub fn with_max_detail_rows(mut self, rows: usize) -> Self {
        self.max_detail_rows = rows;
        self
    }
}

// --- A4 page geometry (PostScript points) ----------------------------------
//
// PDF coordinates put the origin at the bottom-left; Y grows upward. We keep
// generous margins so that long paths/logic strings do not overflow off the
// page.
const PAGE_WIDTH: f32 = 595.0;
const PAGE_HEIGHT: f32 = 842.0;
const MARGIN_LEFT: f32 = 50.0;
const MARGIN_RIGHT: f32 = 50.0;
const MARGIN_TOP: f32 = 60.0;
const MARGIN_BOTTOM: f32 = 50.0;
const TITLE_FONT_SIZE: f32 = 16.0;
const SECTION_FONT_SIZE: f32 = 12.0;
const BODY_FONT_SIZE: f32 = 10.0;
const LINE_HEIGHT: f32 = 14.0;

/// Simple monotonic allocator for PDF indirect references.
struct RefAllocator {
    next: i32,
}

impl RefAllocator {
    fn new() -> Self {
        // id 1 is reserved below for the document catalog.
        Self { next: 1 }
    }

    fn alloc(&mut self) -> Ref {
        let id = self.next;
        self.next += 1;
        Ref::new(id)
    }
}

/// A single text line to emit on a PDF page.
struct Line {
    text: String,
    font_size: f32,
    bold: bool,
}

impl Line {
    fn body(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            font_size: BODY_FONT_SIZE,
            bold: false,
        }
    }

    fn section(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            font_size: SECTION_FONT_SIZE,
            bold: true,
        }
    }

    fn title(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            font_size: TITLE_FONT_SIZE,
            bold: true,
        }
    }

    fn blank() -> Self {
        Self::body("")
    }
}

/// Generator that serialises benchmark results into a PDF buffer.
pub struct PdfReportGenerator {
    config: PdfReportConfig,
}

impl PdfReportGenerator {
    /// Build a new generator with the supplied configuration.
    #[must_use]
    pub fn new(config: PdfReportConfig) -> Self {
        Self { config }
    }

    /// Render the benchmark results into a freshly allocated PDF byte buffer.
    #[must_use]
    pub fn generate(&self, results: &[SingleResult]) -> Vec<u8> {
        let analysis = FullAnalysis::from_results(results);
        let lines = self.build_lines(&analysis, results);
        self.render_pdf(&lines)
    }

    /// Render the benchmark results and persist them to `path`.
    pub fn write_to_file(
        &self,
        results: &[SingleResult],
        path: impl AsRef<Path>,
    ) -> PdfReportResult<()> {
        let buffer = self.generate(results);
        fs::write(path, buffer)?;
        Ok(())
    }

    // ----- content assembly -------------------------------------------------

    fn build_lines(&self, analysis: &FullAnalysis, results: &[SingleResult]) -> Vec<Line> {
        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::title(self.config.title.clone()));
        lines.push(Line::body(format!(
            "{} v{}",
            self.config.solver_name, self.config.solver_version
        )));
        lines.push(Line::blank());

        self.push_summary_lines(&mut lines, &analysis.summary);

        if self.config.include_logic_breakdown && !analysis.by_logic.is_empty() {
            lines.push(Line::blank());
            self.push_logic_lines(&mut lines, &analysis.by_logic);
        }

        if self.config.include_details {
            lines.push(Line::blank());
            self.push_detail_lines(&mut lines, results);
        }

        lines
    }

    fn push_summary_lines(&self, lines: &mut Vec<Line>, summary: &RunSummary) {
        lines.push(Line::section("Summary"));
        lines.push(Line::body(format!("Total benchmarks: {}", summary.total)));
        lines.push(Line::body(format!(
            "Solved: {} ({:.1}%)",
            summary.solved(),
            summary.solve_rate()
        )));
        lines.push(Line::body(format!("SAT: {}", summary.sat)));
        lines.push(Line::body(format!("UNSAT: {}", summary.unsat)));
        lines.push(Line::body(format!("Unknown: {}", summary.unknown)));
        lines.push(Line::body(format!("Timeouts: {}", summary.timeouts)));
        lines.push(Line::body(format!("Errors: {}", summary.errors)));
        lines.push(Line::body(format!("Memory outs: {}", summary.memouts)));
        lines.push(Line::body(format!(
            "Sound: {}",
            if summary.is_sound() { "yes" } else { "NO" }
        )));
        lines.push(Line::body(format!(
            "Total time: {:.3} s",
            summary.total_time.as_secs_f64()
        )));
        lines.push(Line::body(format!(
            "Average time: {:.3} s",
            summary.avg_time.as_secs_f64()
        )));
    }

    fn push_logic_lines(
        &self,
        lines: &mut Vec<Line>,
        by_logic: &std::collections::HashMap<String, CategoryStats>,
    ) {
        lines.push(Line::section("Per-logic breakdown"));
        lines.push(Line::body(format_logic_row(
            "Logic", "Total", "Solved", "Rate", "SAT", "UNSAT", "TO", "ERR",
        )));

        let mut logics: Vec<_> = by_logic.iter().collect();
        logics.sort_by(|a, b| a.0.cmp(b.0));
        for (logic, stats) in logics {
            lines.push(Line::body(format_logic_row(
                logic,
                &stats.total.to_string(),
                &stats.solved.to_string(),
                &format!("{:.1}%", stats.solve_rate()),
                &stats.sat.to_string(),
                &stats.unsat.to_string(),
                &stats.timeouts.to_string(),
                &stats.errors.to_string(),
            )));
        }
    }

    fn push_detail_lines(&self, lines: &mut Vec<Line>, results: &[SingleResult]) {
        lines.push(Line::section("Individual results"));
        lines.push(Line::body(format_detail_row(
            "#",
            "Benchmark",
            "Logic",
            "Status",
            "Time",
        )));

        let limit = self.config.max_detail_rows;
        let shown = results.len().min(limit);
        for (idx, result) in results.iter().take(shown).enumerate() {
            let filename = result
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            lines.push(Line::body(format_detail_row(
                &format!("{}", idx + 1),
                &filename,
                result.logic.as_deref().unwrap_or("-"),
                result.status.as_str(),
                &format!("{:.3}s", result.time.as_secs_f64()),
            )));
        }

        if results.len() > shown {
            lines.push(Line::body(format!(
                "(truncated: showing {} of {} results)",
                shown,
                results.len()
            )));
        }
    }

    // ----- rendering --------------------------------------------------------

    fn render_pdf(&self, lines: &[Line]) -> Vec<u8> {
        let mut allocator = RefAllocator::new();
        let catalog_id = allocator.alloc();
        let page_tree_id = allocator.alloc();
        let font_id = allocator.alloc();
        let font_bold_id = allocator.alloc();

        let font_regular = Name(b"F1");
        let font_bold = Name(b"F2");

        let paginated = paginate(lines);
        let mut page_ids: Vec<Ref> = Vec::with_capacity(paginated.len());
        let mut contents: Vec<(Ref, Vec<u8>)> = Vec::with_capacity(paginated.len());
        let total_pages = paginated.len().max(1);

        // Guarantee at least one page, even when the caller passed no lines.
        let empty_page: Vec<&Line> = Vec::new();
        let iter: Vec<Vec<&Line>> = if paginated.is_empty() {
            vec![empty_page]
        } else {
            paginated
        };

        for (page_index, chunk) in iter.iter().enumerate() {
            let page_id = allocator.alloc();
            let content_id = allocator.alloc();
            page_ids.push(page_id);

            let content_bytes = build_page_content(
                chunk,
                font_regular,
                font_bold,
                page_index + 1,
                total_pages,
                &self.config,
            );
            contents.push((content_id, content_bytes));
        }

        // --- write document structure --------------------------------------
        let mut pdf = Pdf::new();
        pdf.catalog(catalog_id).pages(page_tree_id);

        let kids: Vec<Ref> = page_ids.clone();
        pdf.pages(page_tree_id)
            .kids(kids)
            .count(page_ids.len() as i32);

        for (i, page_id) in page_ids.iter().enumerate() {
            let (content_id, _) = contents[i];
            let mut page = pdf.page(*page_id);
            page.media_box(Rect::new(0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT));
            page.parent(page_tree_id);
            page.contents(content_id);
            let mut resources = page.resources();
            let mut fonts = resources.fonts();
            fonts.pair(font_regular, font_id);
            fonts.pair(font_bold, font_bold_id);
            fonts.finish();
            resources.finish();
            page.finish();
        }

        for (content_id, content_bytes) in &contents {
            pdf.stream(*content_id, content_bytes);
        }

        pdf.type1_font(font_id).base_font(Name(b"Helvetica"));
        pdf.type1_font(font_bold_id)
            .base_font(Name(b"Helvetica-Bold"));

        pdf.finish()
    }
}

// --- helpers ---------------------------------------------------------------

fn format_logic_row(
    logic: &str,
    total: &str,
    solved: &str,
    rate: &str,
    sat: &str,
    unsat: &str,
    timeouts: &str,
    errors: &str,
) -> String {
    format!(
        "{:<18} {:>6} {:>7} {:>7} {:>6} {:>6} {:>5} {:>5}",
        truncate(logic, 18),
        total,
        solved,
        rate,
        sat,
        unsat,
        timeouts,
        errors
    )
}

fn format_detail_row(idx: &str, benchmark: &str, logic: &str, status: &str, time: &str) -> String {
    format!(
        "{:>4} {:<36} {:<12} {:<8} {:>10}",
        idx,
        truncate(benchmark, 36),
        truncate(logic, 12),
        status,
        time
    )
}

/// Truncate a string to at most `max` characters, appending an ellipsis when
/// truncation occurs.
fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.to_string()
    } else {
        let head: String = value.chars().take(max.saturating_sub(1)).collect();
        format!("{}~", head)
    }
}

/// Split the line list into page-sized chunks.
fn paginate(lines: &[Line]) -> Vec<Vec<&Line>> {
    let usable_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;
    let lines_per_page = (usable_height / LINE_HEIGHT).floor() as usize;
    let lines_per_page = lines_per_page.max(1);

    let mut pages: Vec<Vec<&Line>> = Vec::new();
    let mut current: Vec<&Line> = Vec::with_capacity(lines_per_page);
    for line in lines {
        if current.len() == lines_per_page {
            pages.push(std::mem::take(&mut current));
        }
        current.push(line);
    }
    if !current.is_empty() {
        pages.push(current);
    }
    pages
}

fn build_page_content(
    lines: &[&Line],
    font_regular: Name,
    font_bold: Name,
    page_num: usize,
    total_pages: usize,
    config: &PdfReportConfig,
) -> Vec<u8> {
    let mut content = Content::new();
    content.begin_text();

    let start_x = MARGIN_LEFT;
    let start_y = PAGE_HEIGHT - MARGIN_TOP;
    // Absolute positioning via `next_line` uses offsets, so we jump to the
    // anchor explicitly the first time and then advance line-by-line.
    content.set_font(font_regular, BODY_FONT_SIZE);
    content.next_line(start_x, start_y);

    for (idx, line) in lines.iter().enumerate() {
        let name = if line.bold { font_bold } else { font_regular };
        content.set_font(name, line.font_size);
        content.show(Str(line.text.as_bytes()));
        if idx + 1 < lines.len() {
            content.next_line(0.0, -LINE_HEIGHT);
        }
    }

    content.end_text();

    // Footer: page count + solver identity.
    let footer_text = format!(
        "{} v{} - page {} of {}",
        config.solver_name, config.solver_version, page_num, total_pages
    );
    content.begin_text();
    content.set_font(font_regular, BODY_FONT_SIZE - 2.0);
    content.next_line(MARGIN_LEFT, MARGIN_BOTTOM - 20.0);
    content.show(Str(footer_text.as_bytes()));
    content.end_text();

    content.finish().to_vec()
}

/// Convenience wrapper exposing a simple non-builder API that closely mirrors
/// the plan's original `PdfReport::from_runs(...)` shape while reusing the
/// generator internally.
pub struct PdfReport {
    buffer: Vec<u8>,
}

impl PdfReport {
    /// Build a PDF report from a slice of [`SingleResult`]s using default
    /// configuration.
    #[must_use]
    pub fn from_runs(results: &[SingleResult]) -> Self {
        let generator = PdfReportGenerator::new(PdfReportConfig::default());
        Self {
            buffer: generator.generate(results),
        }
    }

    /// Build a PDF report with an explicit configuration.
    #[must_use]
    pub fn from_runs_with_config(results: &[SingleResult], config: PdfReportConfig) -> Self {
        let generator = PdfReportGenerator::new(config);
        Self {
            buffer: generator.generate(results),
        }
    }

    /// Borrow the raw PDF bytes backing this report.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Consume this report and return the owned byte buffer.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    /// Persist the report to `path`.
    pub fn write(&self, path: impl AsRef<Path>) -> PdfReportResult<()> {
        fs::write(path, &self.buffer)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "pdf-report"))]
mod tests {
    use super::*;
    use crate::benchmark::{BenchmarkStatus, SingleResult};
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_result(logic: &str, status: BenchmarkStatus, time_ms: u64, id: u64) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/bench_{}_{}.smt2", logic, id)),
            logic: Some(logic.to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    fn assert_pdf_magic(path: &Path, min_size: u64) {
        let bytes = fs::read(path).expect("read generated pdf");
        assert!(
            bytes.starts_with(b"%PDF-"),
            "missing %PDF- magic; first bytes: {:?}",
            &bytes.get(..8)
        );
        let size = bytes.len() as u64;
        assert!(
            size >= min_size,
            "pdf smaller than expected: {} < {}",
            size,
            min_size
        );
    }

    #[test]
    fn test_pdf_report_minimal() {
        let path = std::env::temp_dir().join("oxiz_pdf_test_minimal.pdf");
        let _ = fs::remove_file(&path);

        let report = PdfReport::from_runs(&[]);
        report
            .write(&path)
            .expect("minimal pdf should write successfully");

        assert_pdf_magic(&path, 1024);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_pdf_report_multi_logic() {
        let path = std::env::temp_dir().join("oxiz_pdf_test_multi_logic.pdf");
        let _ = fs::remove_file(&path);

        let mut results: Vec<SingleResult> = Vec::new();
        for id in 0..4 {
            results.push(make_result("QF_LIA", BenchmarkStatus::Sat, 100 + id, id));
            results.push(make_result("QF_BV", BenchmarkStatus::Unsat, 200 + id, id));
            results.push(make_result("QF_LRA", BenchmarkStatus::Timeout, 60_000, id));
        }

        let config = PdfReportConfig::new("OxiZ Multi-Logic PDF Test")
            .with_solver("OxiZ-test", env!("CARGO_PKG_VERSION"))
            .with_max_detail_rows(50);

        let report = PdfReport::from_runs_with_config(&results, config);
        report
            .write(&path)
            .expect("multi-logic pdf should write successfully");

        assert_pdf_magic(&path, 2048);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_pdf_report_generator_write_to_file() {
        let path = std::env::temp_dir().join("oxiz_pdf_test_generator_api.pdf");
        let _ = fs::remove_file(&path);

        let results = vec![
            make_result("QF_LIA", BenchmarkStatus::Sat, 10, 1),
            make_result("QF_LIA", BenchmarkStatus::Unsat, 20, 2),
        ];
        let config = PdfReportConfig::default();
        let generator = PdfReportGenerator::new(config);
        generator
            .write_to_file(&results, &path)
            .expect("generator should write pdf");

        assert_pdf_magic(&path, 1024);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_truncate_helper() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("verylongstring", 5).chars().count(), 5);
    }
}
