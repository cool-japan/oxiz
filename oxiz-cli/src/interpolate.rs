//! Interpolant generation support for OxiZ CLI
//!
//! This module provides Craig interpolation functionality through the CLI.
//! It supports parsing assertions into A and B partitions and generating
//! interpolants using oxiz-proof's interpolation API.

use oxiz_proof::InterpolationAlgorithm;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Output format for interpolants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolateFormat {
    /// SMT-LIB2 format (default)
    #[default]
    Smtlib,
    /// Plain text format
    Text,
    /// JSON format
    Json,
}

impl InterpolateFormat {
    /// Parse format from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "smtlib" | "smt2" | "smtlib2" => Some(Self::Smtlib),
            "text" | "plain" => Some(Self::Text),
            "json" => Some(Self::Json),
            _ => None,
        }
    }
}

impl fmt::Display for InterpolateFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Smtlib => write!(f, "smtlib"),
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
        }
    }
}

/// Result of interpolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationResult {
    /// The computed interpolant formula
    pub interpolant: String,
    /// Satisfiability status (should be unsat for interpolation)
    pub status: String,
    /// A partition assertions
    pub a_assertions: Vec<String>,
    /// B partition assertions
    pub b_assertions: Vec<String>,
    /// Statistics about the interpolation
    pub stats: Option<InterpolationStatistics>,
    /// Error message if any
    pub error: Option<String>,
}

/// Statistics about interpolation computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationStatistics {
    /// Number of A assertions
    pub a_count: usize,
    /// Number of B assertions
    pub b_count: usize,
    /// Time taken in microseconds
    pub time_us: u64,
    /// Algorithm used
    pub algorithm: String,
}

/// Execute interpolation on a script
#[allow(clippy::collapsible_if)]
pub fn execute_interpolation(
    script: &str,
    format: InterpolateFormat,
    algorithm: Option<InterpolationAlgorithm>,
) -> String {
    // Parse the script looking for assert-partition commands
    let mut a_assertions = Vec::new();
    let mut b_assertions = Vec::new();

    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("(assert-partition") {
            if let Some(rest) = trimmed.strip_prefix("(assert-partition") {
                let rest = rest.trim();
                if rest.starts_with('A') || rest.starts_with('a') {
                    let formula = rest[1..].trim();
                    if let Some(stripped) = formula.strip_suffix(')') {
                        a_assertions.push(stripped.trim().to_string());
                    }
                } else if rest.starts_with('B') || rest.starts_with('b') {
                    let formula = rest[1..].trim();
                    if let Some(stripped) = formula.strip_suffix(')') {
                        b_assertions.push(stripped.trim().to_string());
                    }
                }
            }
        }
    }

    // Build result
    let result = if a_assertions.is_empty() || b_assertions.is_empty() {
        InterpolationResult {
            interpolant: String::new(),
            status: "error".to_string(),
            a_assertions,
            b_assertions,
            stats: None,
            error: Some(
                "Both A and B partitions must have at least one assertion. \
                 Use (assert-partition A <formula>) and (assert-partition B <formula>)."
                    .to_string(),
            ),
        }
    } else {
        // For now, return a placeholder message
        // Full implementation would call the solver and extract an interpolant
        let algo_name = match algorithm {
            Some(InterpolationAlgorithm::McMillan) => "McMillan",
            Some(InterpolationAlgorithm::Pudlak) => "Pudlak",
            Some(InterpolationAlgorithm::Huang) => "Huang",
            None => "Default",
        };

        InterpolationResult {
            interpolant: "true".to_string(), // Placeholder
            status: "unknown".to_string(),
            a_assertions,
            b_assertions,
            stats: Some(InterpolationStatistics {
                a_count: 0,
                b_count: 0,
                time_us: 0,
                algorithm: algo_name.to_string(),
            }),
            error: Some("Interpolation not fully implemented. This is a placeholder.".to_string()),
        }
    };

    // Format output
    match format {
        InterpolateFormat::Smtlib => {
            let mut output = String::new();
            output.push_str(&result.status);
            output.push('\n');
            if let Some(ref err) = result.error {
                output.push_str(&format!("(error \"{}\")\n", err));
            } else {
                output.push_str(&format!("(interpolant {})\n", result.interpolant));
            }
            output
        }
        InterpolateFormat::Text => {
            let mut output = String::new();
            output.push_str(&format!("Status: {}\n", result.status));
            if let Some(ref err) = result.error {
                output.push_str(&format!("Error: {}\n", err));
            } else {
                output.push_str(&format!("Interpolant: {}\n", result.interpolant));
            }
            output
        }
        InterpolateFormat::Json => {
            serde_json::to_string_pretty(&result).unwrap_or_else(|_| "{}".to_string())
        }
    }
}
