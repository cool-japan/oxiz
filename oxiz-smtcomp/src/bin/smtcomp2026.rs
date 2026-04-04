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

/// Simple CLI argument container for SMT-COMP mode.
#[derive(Debug)]
struct Args {
    /// Path to the SMT-LIB2 input file, or `None` for stdin.
    input_file: Option<PathBuf>,
    /// Print version and exit.
    print_version: bool,
    /// Verbose/debug mode (writes to stderr).
    verbose: bool,
}

impl Args {
    /// Parse arguments from `std::env::args`.
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut input_file = None;
        let mut print_version = false;
        let mut verbose = false;
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
    }

    process::exit(run_solver(&input, args.verbose));
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

/// Execute the SMT-LIB2 script and return the appropriate exit code.
///
/// Inspects the output lines from `Context::execute_script` to determine the
/// final check-sat result. Returns `EXIT_SAT`, `EXIT_UNSAT`, or `EXIT_UNKNOWN`.
fn run_solver(input: &str, verbose: bool) -> i32 {
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
    // We scan in reverse so we find the most recent check-sat outcome.
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

    match last_result {
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
    }
}
