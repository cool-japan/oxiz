//! SMT-COMP 2026 solver binary entry point.
//!
//! This binary reads SMT-LIB2 input from stdin (or a file argument), runs
//! the OxiZ solver, and writes exactly one of "sat", "unsat", or "unknown"
//! to stdout, following the SMT-COMP 2026 output specification.
//!
//! # Exit Codes
//!
//! | Code | Meaning                          |
//! |------|----------------------------------|
//! |  0   | sat                              |
//! |  10  | unsat                            |
//! |  20  | unknown / timeout / error        |
//!
//! # Usage
//!
//! ```text
//! smtcomp2026 [OPTIONS] [FILE]
//!
//! If FILE is omitted, reads from stdin.
//!
//! Options:
//!   --track <TRACK>   Competition track (default: single).
//!                     Allowed values: single, incremental, unsat-core, model, proof.
//! ```

use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use oxiz_solver::Context;

/// Exit code for a SAT result (StarExec convention for SMT-COMP).
const EXIT_SAT: i32 = 0;
/// Exit code for an UNSAT result.
const EXIT_UNSAT: i32 = 10;
/// Exit code for UNKNOWN / error / timeout.
const EXIT_UNKNOWN: i32 = 20;

/// SMT-COMP 2026 track selection for the binary.
///
/// Each variant changes the set of post-`check-sat` queries that are issued
/// and printed after the primary result.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SolverTrack {
    /// Single-query track: one `check-sat` per file, no follow-up queries.
    #[default]
    Single,
    /// Incremental track: `push`/`pop` are handled natively; no extra output.
    Incremental,
    /// Unsat-core track: emit `(get-unsat-core)` after an `unsat` result.
    UnsatCore,
    /// Model-validation track: emit `(get-model)` after a `sat` result.
    Model,
    /// Proof-exhibition track: emit `(get-proof)` after an `unsat` result.
    Proof,
}

impl SolverTrack {
    /// Parse a `--track` argument value into a `SolverTrack`.
    ///
    /// Accepted strings: `"single"`, `"incremental"`, `"unsat-core"`,
    /// `"unsat_core"`, `"model"`, `"model_validation"`, `"proof"`,
    /// `"proof_exhibition"`.  Returns `None` for unrecognised values.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "single" | "default" => Some(Self::Single),
            "incremental" => Some(Self::Incremental),
            "unsat-core" | "unsat_core" => Some(Self::UnsatCore),
            "model" | "model_validation" => Some(Self::Model),
            "proof" | "proof_exhibition" => Some(Self::Proof),
            _ => None,
        }
    }
}

/// CLI argument container for SMT-COMP mode.
#[derive(Debug)]
struct Args {
    /// Path to the SMT-LIB2 input file, or `None` for stdin.
    input_file: Option<PathBuf>,
    /// Print version and exit.
    print_version: bool,
    /// Verbose/debug mode (writes to stderr).
    verbose: bool,
    /// Competition track selection.
    track: SolverTrack,
}

impl Args {
    /// Parse arguments from `std::env::args`.
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut input_file = None;
        let mut print_version = false;
        let mut verbose = false;
        let mut track = SolverTrack::default();
        let mut i = 1;

        while i < args.len() {
            match args[i].as_str() {
                "--version" | "-V" => {
                    print_version = true;
                }
                "--verbose" | "-v" => {
                    verbose = true;
                }
                // StarExec passes --smtcomp as a mode flag; consume silently.
                "--smtcomp" => {}
                // Track selection flag.
                "--track" => {
                    i += 1;
                    if i < args.len() {
                        match SolverTrack::parse(&args[i]) {
                            Some(t) => track = t,
                            None => {
                                eprintln!(
                                    "smtcomp2026: unknown track '{}'; \
                                     valid: single, incremental, unsat-core, model, proof",
                                    args[i]
                                );
                            }
                        }
                    } else {
                        eprintln!("smtcomp2026: --track requires an argument");
                    }
                }
                // Ignore time/memory limit flags (enforced by StarExec externally).
                "--time" | "-t" | "--memory" | "-m" => {
                    i += 1; // skip the following value argument
                }
                arg if !arg.starts_with('-') => {
                    input_file = Some(PathBuf::from(arg));
                }
                other => {
                    eprintln!("smtcomp2026: unrecognised option '{}'", other);
                }
            }
            i += 1;
        }

        Args {
            input_file,
            print_version,
            verbose,
            track,
        }
    }
}

fn main() {
    let args = Args::parse();

    if args.print_version {
        println!("smtcomp2026 {} (OxiZ)", env!("CARGO_PKG_VERSION"));
        process::exit(0);
    }

    let input = match read_input(&args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("smtcomp2026: failed to read input: {}", e);
            println!("unknown");
            process::exit(EXIT_UNKNOWN);
        }
    };

    if args.verbose {
        eprintln!("smtcomp2026: read {} bytes of input", input.len());
        eprintln!("smtcomp2026: track = {:?}", args.track);
    }

    process::exit(run_solver(&input, &args.track, args.verbose));
}

/// Read the full SMT-LIB2 input from a file or stdin.
fn read_input(args: &Args) -> io::Result<String> {
    match &args.input_file {
        Some(path) => std::fs::read_to_string(path),
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            Ok(buf)
        }
    }
}

/// Execute the SMT-LIB2 script, apply track-specific post-processing, and
/// return the appropriate exit code.
///
/// For the `UnsatCore`, `Model`, and `Proof` tracks a second `execute_script`
/// call is made with the relevant get-* command so that its output reaches
/// stdout before we exit.  The primary check-sat result (sat/unsat/unknown)
/// and the corresponding exit code follow the StarExec convention and are
/// unaffected by track selection.
fn run_solver(input: &str, track: &SolverTrack, verbose: bool) -> i32 {
    let mut ctx = Context::new();

    let output_lines = match ctx.execute_script(input) {
        Ok(lines) => lines,
        Err(e) => {
            if verbose {
                eprintln!("smtcomp2026: solver error: {}", e);
            }
            println!("unknown");
            return EXIT_UNKNOWN;
        }
    };

    // The last check-sat result is the authoritative answer.
    // Scan in reverse to find the most recent check-sat outcome.
    let mut last_result: Option<&str> = None;
    for line in output_lines.iter().rev() {
        let trimmed = line.trim();
        match trimmed {
            "sat" | "unsat" | "unknown" => {
                last_result = Some(trimmed);
                break;
            }
            _ => {}
        }
    }

    let exit_code = match last_result {
        Some("sat") => {
            println!("sat");
            EXIT_SAT
        }
        Some("unsat") => {
            println!("unsat");
            EXIT_UNSAT
        }
        _ => {
            println!("unknown");
            EXIT_UNKNOWN
        }
    };

    // Track-specific post-processing: issue follow-up SMT-LIB commands and
    // print their output.  We reuse the same solver context so that the
    // accumulated solver state (assertions, etc.) is still available.
    match (track, last_result) {
        (SolverTrack::UnsatCore, Some("unsat")) => {
            match ctx.execute_script("(get-unsat-core)") {
                Ok(core_lines) => {
                    for line in &core_lines {
                        println!("{}", line);
                    }
                }
                Err(e) => {
                    if verbose {
                        eprintln!("smtcomp2026: get-unsat-core error: {}", e);
                    }
                }
            }
        }
        (SolverTrack::Model, Some("sat")) => {
            match ctx.execute_script("(get-model)") {
                Ok(model_lines) => {
                    for line in &model_lines {
                        println!("{}", line);
                    }
                }
                Err(e) => {
                    if verbose {
                        eprintln!("smtcomp2026: get-model error: {}", e);
                    }
                }
            }
        }
        (SolverTrack::Proof, Some("unsat")) => {
            match ctx.execute_script("(get-proof)") {
                Ok(proof_lines) => {
                    for line in &proof_lines {
                        println!("{}", line);
                    }
                }
                Err(e) => {
                    if verbose {
                        eprintln!("smtcomp2026: get-proof error: {}", e);
                    }
                }
            }
        }
        // Single / Incremental / mismatched result — nothing extra to emit.
        _ => {}
    }

    exit_code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_parse_single() {
        assert_eq!(SolverTrack::parse("single"), Some(SolverTrack::Single));
        assert_eq!(SolverTrack::parse("default"), Some(SolverTrack::Single));
    }

    #[test]
    fn test_track_parse_incremental() {
        assert_eq!(
            SolverTrack::parse("incremental"),
            Some(SolverTrack::Incremental)
        );
    }

    #[test]
    fn test_track_parse_unsat_core() {
        assert_eq!(
            SolverTrack::parse("unsat-core"),
            Some(SolverTrack::UnsatCore)
        );
        assert_eq!(
            SolverTrack::parse("unsat_core"),
            Some(SolverTrack::UnsatCore)
        );
    }

    #[test]
    fn test_track_parse_model() {
        assert_eq!(SolverTrack::parse("model"), Some(SolverTrack::Model));
        assert_eq!(
            SolverTrack::parse("model_validation"),
            Some(SolverTrack::Model)
        );
    }

    #[test]
    fn test_track_parse_proof() {
        assert_eq!(SolverTrack::parse("proof"), Some(SolverTrack::Proof));
        assert_eq!(
            SolverTrack::parse("proof_exhibition"),
            Some(SolverTrack::Proof)
        );
    }

    #[test]
    fn test_track_parse_unknown_returns_none() {
        assert_eq!(SolverTrack::parse("bogus"), None);
        assert_eq!(SolverTrack::parse(""), None);
    }

    #[test]
    fn test_track_default_is_single() {
        let t = SolverTrack::default();
        assert_eq!(t, SolverTrack::Single);
    }
}
