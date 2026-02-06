use crate::SolverResult;
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

const Z3_TIMEOUT_SECS: u64 = 60;

pub fn run_z3(smt2_file: &Path) -> Result<SolverResult> {
    // Try common Z3 installation paths
    let z3_paths = [
        "/opt/homebrew/bin/z3", // macOS Homebrew
        "/usr/local/bin/z3",    // macOS/Linux manual install
        "/usr/bin/z3",          // Linux package manager
        "z3",                   // PATH
    ];

    let mut z3_cmd = None;
    for path in &z3_paths {
        if Command::new(path).arg("--version").output().is_ok() {
            z3_cmd = Some(path);
            break;
        }
    }

    let z3_path = z3_cmd.context("Z3 not found. Please install Z3 and ensure it's in PATH")?;

    let output = Command::new(z3_path)
        .arg("-smt2")
        .arg(smt2_file)
        .arg(format!("-T:{}", Z3_TIMEOUT_SECS))
        .output()
        .context("Failed to execute Z3")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for errors in stdout (Z3 outputs errors to stdout with "(error ...)" prefix)
    // Note: Z3 may still output "sat"/"unsat" even when there are errors
    if stdout.contains("(error ") || !output.status.success() {
        // Extract error messages from stdout
        let error_lines: Vec<&str> = stdout
            .lines()
            .filter(|line| line.starts_with("(error "))
            .collect();

        let error_msg = if !error_lines.is_empty() {
            error_lines.join("\n")
        } else if !stderr.is_empty() {
            stderr.to_string()
        } else {
            format!("Z3 failed with exit code: {:?}", output.status.code())
        };

        return Ok(SolverResult::Error(error_msg));
    }

    // Parse Z3 output for result
    let result = if stdout.contains("unsat") {
        SolverResult::Unsat
    } else if stdout.contains("sat") && !stdout.contains("unsat") {
        SolverResult::Sat
    } else if stdout.contains("unknown") || stdout.contains("timeout") {
        SolverResult::Unknown
    } else {
        SolverResult::Error(format!("Unexpected Z3 output: {}", stdout.trim()))
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    #[ignore] // Only run if Z3 is installed
    fn test_z3_sat() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "(set-logic QF_LIA)")?;
        writeln!(file, "(declare-const x Int)")?;
        writeln!(file, "(assert (= x 42))")?;
        writeln!(file, "(check-sat)")?;

        let result = run_z3(file.path())?;
        assert_eq!(result, SolverResult::Sat);
        Ok(())
    }

    #[test]
    #[ignore] // Only run if Z3 is installed
    fn test_z3_unsat() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "(set-logic QF_LIA)")?;
        writeln!(file, "(declare-const x Int)")?;
        writeln!(file, "(assert (< x 0))")?;
        writeln!(file, "(assert (> x 0))")?;
        writeln!(file, "(check-sat)")?;

        let result = run_z3(file.path())?;
        assert_eq!(result, SolverResult::Unsat);
        Ok(())
    }
}
