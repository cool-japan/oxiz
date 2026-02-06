use crate::SolverResult;
use anyhow::Result;
use oxiz_solver::Context;
use std::fs;
use std::path::Path;
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::thread;
use std::time::Duration;

/// Execute an SMT2 file using OxiZ solver with timeout protection
pub fn run_oxiz(smt2_file: &Path) -> Result<SolverResult> {
    // Read the SMT2 file
    let script = fs::read_to_string(smt2_file)
        .map_err(|e| anyhow::anyhow!("Failed to read SMT2 file {}: {}", smt2_file.display(), e))?;

    // Create a channel for communication between threads
    let (tx, rx) = channel();

    // Spawn a thread to run the solver
    // This allows us to enforce timeout and handle crashes gracefully
    thread::spawn(move || {
        let mut ctx = Context::new();
        let result = ctx.execute_script(&script);
        let _ = tx.send(result);
    });

    // Wait for the result with a 60-second timeout
    match rx.recv_timeout(Duration::from_secs(60)) {
        // Solver completed successfully
        Ok(result) => match result {
            Ok(output) => parse_output(&output),
            Err(e) => Ok(SolverResult::Error(format!("Solver error: {}", e))),
        },

        // Solver exceeded timeout
        Err(RecvTimeoutError::Timeout) => Ok(SolverResult::Timeout),

        // Solver thread crashed (panic or disconnect)
        Err(RecvTimeoutError::Disconnected) => Ok(SolverResult::Error(
            "Solver crashed or panicked".to_string(),
        )),
    }
}

/// Parse the output from execute_script to extract the check-sat result
///
/// The execute_script function returns a Vec<String> containing responses
/// to all commands. We need to find the check-sat response(s) and return
/// the appropriate SolverResult.
fn parse_output(output: &[String]) -> Result<SolverResult> {
    // Iterate through the output in reverse order to find the last check-sat result
    // This handles cases where there might be multiple check-sat commands
    for line in output.iter().rev() {
        let line_trimmed = line.trim().to_lowercase();

        // Check for exact matches of SMT-LIB2 responses
        if line_trimmed == "sat" {
            return Ok(SolverResult::Sat);
        }
        if line_trimmed == "unsat" {
            return Ok(SolverResult::Unsat);
        }
        if line_trimmed == "unknown" {
            return Ok(SolverResult::Unknown);
        }
    }

    // If no check-sat result was found in the output, return Unknown
    // This can happen if:
    // - The SMT2 file has no check-sat command
    // - The output format is unexpected
    // - There was an error during execution that was handled silently
    Ok(SolverResult::Unknown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_oxiz_sat() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "(set-logic QF_LIA)")?;
        writeln!(file, "(declare-const x Int)")?;
        writeln!(file, "(assert (= x 42))")?;
        writeln!(file, "(check-sat)")?;

        let result = run_oxiz(file.path())?;
        assert_eq!(result, SolverResult::Sat);
        Ok(())
    }

    #[test]
    fn test_oxiz_unsat() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "(set-logic QF_LIA)")?;
        writeln!(file, "(declare-const x Int)")?;
        writeln!(file, "(assert (< x 0))")?;
        writeln!(file, "(assert (> x 0))")?;
        writeln!(file, "(check-sat)")?;

        let result = run_oxiz(file.path())?;
        assert_eq!(result, SolverResult::Unsat);
        Ok(())
    }
}
