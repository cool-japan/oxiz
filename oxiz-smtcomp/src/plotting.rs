//! Cactus plot generation (SVG output)
//!
//! This module provides functionality to generate cactus plots and scatter plots
//! for visualizing benchmark results, with SVG output.

use crate::benchmark::SingleResult;
use crate::statistics::{CactusPoint, cactus_data};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use thiserror::Error;

/// Error type for plotting operations
#[derive(Error, Debug)]
pub enum PlotError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// No data to plot
    #[error("No data to plot")]
    NoData,
}

/// Result type for plotting operations
pub type PlotResult<T> = Result<T, PlotError>;

/// Color for plot elements
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
}

impl Color {
    /// Create a new color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to CSS color string
    #[must_use]
    pub fn to_css(&self) -> String {
        format!("rgb({},{},{})", self.r, self.g, self.b)
    }

    /// Blue color - Google Blue
    pub const BLUE: Color = Color::new(66, 133, 244);
    /// Red color - Google Red
    pub const RED: Color = Color::new(234, 67, 53);
    /// Green color - Google Green
    pub const GREEN: Color = Color::new(52, 168, 83);
    /// Orange color - Google Yellow/Orange
    pub const ORANGE: Color = Color::new(251, 188, 4);
    /// Purple color - Material Purple
    pub const PURPLE: Color = Color::new(103, 58, 183);
    /// Cyan color - Material Cyan
    pub const CYAN: Color = Color::new(0, 188, 212);
    /// Pink color - Material Pink
    pub const PINK: Color = Color::new(233, 30, 99);
    /// Gray color - Material Gray
    pub const GRAY: Color = Color::new(158, 158, 158);
}

/// Default color palette for multiple solvers
pub const DEFAULT_COLORS: [Color; 8] = [
    Color::BLUE,
    Color::RED,
    Color::GREEN,
    Color::ORANGE,
    Color::PURPLE,
    Color::CYAN,
    Color::PINK,
    Color::GRAY,
];

/// Plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotConfig {
    /// Plot width in pixels
    pub width: u32,
    /// Plot height in pixels
    pub height: u32,
    /// Left margin
    pub margin_left: u32,
    /// Right margin
    pub margin_right: u32,
    /// Top margin
    pub margin_top: u32,
    /// Bottom margin
    pub margin_bottom: u32,
    /// Title
    pub title: Option<String>,
    /// X-axis label
    pub x_label: Option<String>,
    /// Y-axis label
    pub y_label: Option<String>,
    /// Show grid
    pub show_grid: bool,
    /// Show legend
    pub show_legend: bool,
    /// Use logarithmic Y-axis
    pub log_y: bool,
    /// Line width
    pub line_width: f32,
    /// Point radius
    pub point_radius: f32,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            margin_left: 80,
            margin_right: 40,
            margin_top: 50,
            margin_bottom: 60,
            title: None,
            x_label: Some("Number of solved benchmarks".to_string()),
            y_label: Some("Time (seconds)".to_string()),
            show_grid: true,
            show_legend: true,
            log_y: false,
            line_width: 2.0,
            point_radius: 0.0, // No points by default for cactus
        }
    }
}

impl PlotConfig {
    /// Create a new config with title
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: Some(title.into()),
            ..Default::default()
        }
    }

    /// Set dimensions
    #[must_use]
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Enable logarithmic Y-axis
    #[must_use]
    pub fn with_log_y(mut self, log_y: bool) -> Self {
        self.log_y = log_y;
        self
    }

    /// Set labels
    #[must_use]
    pub fn with_labels(mut self, x: impl Into<String>, y: impl Into<String>) -> Self {
        self.x_label = Some(x.into());
        self.y_label = Some(y.into());
        self
    }

    /// Get plot area dimensions
    #[must_use]
    pub fn plot_area(&self) -> (u32, u32, u32, u32) {
        (
            self.margin_left,
            self.margin_top,
            self.width - self.margin_left - self.margin_right,
            self.height - self.margin_top - self.margin_bottom,
        )
    }
}

/// Data series for plotting
#[derive(Debug, Clone)]
pub struct DataSeries {
    /// Series name
    pub name: String,
    /// Data points (x, y)
    pub points: Vec<(f64, f64)>,
    /// Series color
    pub color: Color,
}

impl DataSeries {
    /// Create a new data series
    #[must_use]
    pub fn new(name: impl Into<String>, points: Vec<(f64, f64)>, color: Color) -> Self {
        Self {
            name: name.into(),
            points,
            color,
        }
    }

    /// Create from cactus data
    #[must_use]
    pub fn from_cactus(name: impl Into<String>, data: &[CactusPoint], color: Color) -> Self {
        let points = data.iter().map(|p| (p.solved as f64, p.time)).collect();
        Self::new(name, points, color)
    }

    /// Create from results
    #[must_use]
    pub fn from_results(name: impl Into<String>, results: &[SingleResult], color: Color) -> Self {
        let cactus = cactus_data(results);
        Self::from_cactus(name, &cactus, color)
    }
}

/// SVG plot generator
pub struct SvgPlot {
    config: PlotConfig,
    series: Vec<DataSeries>,
}

impl SvgPlot {
    /// Create a new plot
    #[must_use]
    pub fn new(config: PlotConfig) -> Self {
        Self {
            config,
            series: Vec::new(),
        }
    }

    /// Create a cactus plot from results
    #[must_use]
    pub fn cactus(title: impl Into<String>) -> Self {
        Self::new(PlotConfig::new(title))
    }

    /// Add a data series
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Add results as a series
    pub fn add_results(&mut self, name: impl Into<String>, results: &[SingleResult]) {
        let color = DEFAULT_COLORS[self.series.len() % DEFAULT_COLORS.len()];
        self.add_series(DataSeries::from_results(name, results, color));
    }

    /// Generate SVG content
    pub fn to_svg(&self) -> PlotResult<String> {
        if self.series.is_empty() || self.series.iter().all(|s| s.points.is_empty()) {
            return Err(PlotError::NoData);
        }

        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
<style>
    .title {{ font: bold 16px sans-serif; }}
    .label {{ font: 12px sans-serif; }}
    .axis {{ font: 10px sans-serif; }}
    .grid {{ stroke: #e0e0e0; stroke-width: 1; }}
    .legend {{ font: 11px sans-serif; }}
</style>
<rect width="100%" height="100%" fill="white"/>
"#,
            self.config.width, self.config.height, self.config.width, self.config.height
        ));

        let (x_off, y_off, plot_w, plot_h) = self.config.plot_area();

        // Calculate data bounds
        let (x_min, x_max, y_min, y_max) = self.calculate_bounds();

        // Draw grid
        if self.config.show_grid {
            svg.push_str(&self.generate_grid(x_off, y_off, plot_w, plot_h, x_max, y_max));
        }

        // Draw axes
        svg.push_str(&self.generate_axes(x_off, y_off, plot_w, plot_h, x_max, y_max));

        // Draw data series
        for series in &self.series {
            svg.push_str(&self.generate_series(
                series, x_off, y_off, plot_w, plot_h, x_min, x_max, y_min, y_max,
            ));
        }

        // Draw title
        if let Some(ref title) = self.config.title {
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="title" text-anchor="middle">{}</text>
"#,
                self.config.width / 2,
                25,
                title
            ));
        }

        // Draw axis labels
        if let Some(ref label) = self.config.x_label {
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="label" text-anchor="middle">{}</text>
"#,
                x_off + plot_w / 2,
                self.config.height - 15,
                label
            ));
        }

        if let Some(ref label) = self.config.y_label {
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="label" text-anchor="middle" transform="rotate(-90,{},{})">{}</text>
"#,
                20,
                y_off + plot_h / 2,
                20,
                y_off + plot_h / 2,
                label
            ));
        }

        // Draw legend
        if self.config.show_legend && !self.series.is_empty() {
            svg.push_str(&self.generate_legend(x_off + plot_w - 150, y_off + 10));
        }

        svg.push_str("</svg>\n");

        Ok(svg)
    }

    /// Write SVG to file
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> PlotResult<()> {
        let svg = self.to_svg()?;
        fs::write(path, svg)?;
        Ok(())
    }

    /// Write SVG to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> PlotResult<()> {
        let svg = self.to_svg()?;
        writer.write_all(svg.as_bytes())?;
        Ok(())
    }

    /// Calculate data bounds
    fn calculate_bounds(&self) -> (f64, f64, f64, f64) {
        let mut x_min = f64::MAX;
        let mut x_max = f64::MIN;
        let mut y_min = f64::MAX;
        let mut y_max = f64::MIN;

        for series in &self.series {
            for (x, y) in &series.points {
                x_min = x_min.min(*x);
                x_max = x_max.max(*x);
                y_min = y_min.min(*y);
                y_max = y_max.max(*y);
            }
        }

        // Add some padding
        x_min = 0.0;
        y_min = 0.0;
        x_max *= 1.05;
        y_max *= 1.1;

        (x_min, x_max, y_min, y_max)
    }

    /// Generate grid lines
    fn generate_grid(
        &self,
        x_off: u32,
        y_off: u32,
        plot_w: u32,
        plot_h: u32,
        x_max: f64,
        y_max: f64,
    ) -> String {
        let mut svg = String::new();

        // Horizontal grid lines
        let y_steps = 5;
        for i in 0..=y_steps {
            let y = y_off + (plot_h as f64 * i as f64 / y_steps as f64) as u32;
            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
                x_off,
                y,
                x_off + plot_w,
                y
            ));
        }

        // Vertical grid lines
        let x_steps = 5;
        for i in 0..=x_steps {
            let x = x_off + (plot_w as f64 * i as f64 / x_steps as f64) as u32;
            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
                x,
                y_off,
                x,
                y_off + plot_h
            ));
        }

        svg
    }

    /// Generate axes with labels
    fn generate_axes(
        &self,
        x_off: u32,
        y_off: u32,
        plot_w: u32,
        plot_h: u32,
        x_max: f64,
        y_max: f64,
    ) -> String {
        let mut svg = String::new();

        // X-axis
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>
"#,
            x_off,
            y_off + plot_h,
            x_off + plot_w,
            y_off + plot_h
        ));

        // Y-axis
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>
"#,
            x_off,
            y_off,
            x_off,
            y_off + plot_h
        ));

        // X-axis labels
        let x_steps = 5;
        for i in 0..=x_steps {
            let x = x_off + (plot_w as f64 * i as f64 / x_steps as f64) as u32;
            let value = (x_max * i as f64 / x_steps as f64) as i32;
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="axis" text-anchor="middle">{}</text>
"#,
                x,
                y_off + plot_h + 20,
                value
            ));
        }

        // Y-axis labels
        let y_steps = 5;
        for i in 0..=y_steps {
            let y = y_off + plot_h - (plot_h as f64 * i as f64 / y_steps as f64) as u32;
            let value = y_max * i as f64 / y_steps as f64;
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="axis" text-anchor="end">{:.1}</text>
"#,
                x_off - 10,
                y + 4,
                value
            ));
        }

        svg
    }

    /// Generate data series path
    #[allow(clippy::too_many_arguments)]
    fn generate_series(
        &self,
        series: &DataSeries,
        x_off: u32,
        y_off: u32,
        plot_w: u32,
        plot_h: u32,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    ) -> String {
        if series.points.is_empty() {
            return String::new();
        }

        let mut svg = String::new();
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // Build path
        let mut path = String::new();
        for (i, (x, y)) in series.points.iter().enumerate() {
            let px = x_off as f64 + ((*x - x_min) / x_range) * plot_w as f64;
            let py = y_off as f64 + plot_h as f64 - ((*y - y_min) / y_range) * plot_h as f64;

            if i == 0 {
                path.push_str(&format!("M {:.1} {:.1}", px, py));
            } else {
                path.push_str(&format!(" L {:.1} {:.1}", px, py));
            }
        }

        svg.push_str(&format!(
            r#"<path d="{}" fill="none" stroke="{}" stroke-width="{}"/>
"#,
            path,
            series.color.to_css(),
            self.config.line_width
        ));

        // Draw points if configured
        if self.config.point_radius > 0.0 {
            for (x, y) in &series.points {
                let px = x_off as f64 + ((*x - x_min) / x_range) * plot_w as f64;
                let py = y_off as f64 + plot_h as f64 - ((*y - y_min) / y_range) * plot_h as f64;
                svg.push_str(&format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{}" fill="{}"/>
"#,
                    px,
                    py,
                    self.config.point_radius,
                    series.color.to_css()
                ));
            }
        }

        svg
    }

    /// Generate legend
    fn generate_legend(&self, x: u32, y: u32) -> String {
        let mut svg = String::new();

        // Legend background
        let legend_height = 25 * self.series.len() as u32 + 10;
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"140\" height=\"{}\" fill=\"white\" stroke=\"#ccc\" rx=\"5\"/>\n",
            x, y, legend_height
        ));

        for (i, series) in self.series.iter().enumerate() {
            let ly = y + 20 + i as u32 * 25;

            // Color line
            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="3"/>
"#,
                x + 10,
                ly,
                x + 35,
                ly,
                series.color.to_css()
            ));

            // Name
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="legend">{}</text>
"#,
                x + 45,
                ly + 4,
                series.name
            ));
        }

        svg
    }
}

/// Generate a cactus plot comparing multiple solvers
pub fn generate_cactus_plot(
    solver_results: &HashMap<String, Vec<SingleResult>>,
    title: &str,
) -> PlotResult<String> {
    let mut plot = SvgPlot::cactus(title);

    for (i, (name, results)) in solver_results.iter().enumerate() {
        let color = DEFAULT_COLORS[i % DEFAULT_COLORS.len()];
        plot.add_series(DataSeries::from_results(name, results, color));
    }

    plot.to_svg()
}

/// Generate a scatter plot comparing two solvers
pub fn generate_scatter_plot(
    results_a: &[SingleResult],
    results_b: &[SingleResult],
    name_a: &str,
    name_b: &str,
    title: &str,
) -> PlotResult<String> {
    let mut config = PlotConfig::new(title);
    config.x_label = Some(format!("{} time (s)", name_a));
    config.y_label = Some(format!("{} time (s)", name_b));
    config.point_radius = 3.0;
    config.line_width = 0.0;

    let scatter = crate::statistics::scatter_data(results_a, results_b);
    let points: Vec<(f64, f64)> = scatter
        .iter()
        .filter_map(|p| Some((p.time_a?, p.time_b?)))
        .collect();

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("Comparison", points, Color::BLUE));

    // Add diagonal line for reference
    plot.to_svg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkStatus;
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
    fn test_color() {
        let color = Color::new(255, 128, 64);
        assert_eq!(color.to_css(), "rgb(255,128,64)");
    }

    #[test]
    fn test_plot_config() {
        let config = PlotConfig::new("Test Plot")
            .with_size(1024, 768)
            .with_log_y(true);

        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert!(config.log_y);
    }

    #[test]
    fn test_data_series() {
        let points = vec![(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)];
        let series = DataSeries::new("Test", points.clone(), Color::BLUE);

        assert_eq!(series.name, "Test");
        assert_eq!(series.points.len(), 3);
    }

    #[test]
    fn test_svg_plot_generation() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, 100),
            make_result(BenchmarkStatus::Sat, 200),
            make_result(BenchmarkStatus::Sat, 300),
        ];

        let mut plot = SvgPlot::cactus("Test Cactus Plot");
        plot.add_results("Solver A", &results);

        let svg = plot.to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Test Cactus Plot"));
        assert!(svg.contains("Solver A"));
    }

    #[test]
    fn test_empty_plot_error() {
        let plot = SvgPlot::cactus("Empty Plot");
        assert!(matches!(plot.to_svg(), Err(PlotError::NoData)));
    }

    #[test]
    fn test_multi_solver_cactus() {
        let results_a = vec![
            make_result(BenchmarkStatus::Sat, 100),
            make_result(BenchmarkStatus::Sat, 200),
        ];
        let results_b = vec![
            make_result(BenchmarkStatus::Sat, 150),
            make_result(BenchmarkStatus::Sat, 250),
        ];

        let mut solver_results = HashMap::new();
        solver_results.insert("Solver A".to_string(), results_a);
        solver_results.insert("Solver B".to_string(), results_b);

        let svg = generate_cactus_plot(&solver_results, "Comparison").unwrap();
        assert!(svg.contains("Solver A"));
        assert!(svg.contains("Solver B"));
    }
}
