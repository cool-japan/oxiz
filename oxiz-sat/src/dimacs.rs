//! DIMACS CNF format parser and writer
//!
//! DIMACS CNF is the standard format for SAT problems.
//! Format:
//! - Comments start with 'c'
//! - Problem line: p cnf <num_vars> <num_clauses>
//! - Clause lines: space-separated literals ending with 0
//! - Positive literal i represents variable i, negative -i represents NOT i

use crate::literal::{Lit, Var};
use crate::solver::{Solver, SolverResult};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

/// Error type for DIMACS parsing
#[derive(Debug)]
pub enum DimacsError {
    /// I/O error
    Io(io::Error),
    /// Parse error
    Parse(String),
    /// Invalid problem line
    InvalidProblem,
    /// Literal out of range
    LiteralOutOfRange(i32),
}

impl fmt::Display for DimacsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Parse(msg) => write!(f, "Parse error: {msg}"),
            Self::InvalidProblem => write!(f, "Invalid problem line"),
            Self::LiteralOutOfRange(lit) => write!(f, "Literal out of range: {lit}"),
        }
    }
}

impl std::error::Error for DimacsError {}

impl From<io::Error> for DimacsError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// DIMACS CNF parser
pub struct DimacsParser {
    num_vars: usize,
    num_clauses: usize,
}

impl DimacsParser {
    /// Create a new DIMACS parser
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_vars: 0,
            num_clauses: 0,
        }
    }

    /// Parse a DIMACS file and load into solver
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed
    pub fn parse_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        solver: &mut Solver,
    ) -> Result<(), DimacsError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.parse_reader(reader, solver)
    }

    /// Parse from a reader
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails
    pub fn parse_reader<R: BufRead>(
        &mut self,
        reader: R,
        solver: &mut Solver,
    ) -> Result<(), DimacsError> {
        let mut current_clause = Vec::new();
        let mut clauses_read = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines
            if line.is_empty() {
                continue;
            }

            // Skip comments
            if line.starts_with('c') {
                continue;
            }

            // Parse problem line
            if line.starts_with('p') {
                self.parse_problem_line(line)?;
                solver.ensure_vars(self.num_vars);
                continue;
            }

            // Parse clause
            for token in line.split_whitespace() {
                let lit_val: i32 = token
                    .parse()
                    .map_err(|_| DimacsError::Parse(format!("Invalid literal: {token}")))?;

                if lit_val == 0 {
                    // End of clause
                    if !current_clause.is_empty() {
                        solver.add_clause(current_clause.iter().copied());
                        current_clause.clear();
                        clauses_read += 1;
                    }
                } else {
                    // Add literal to current clause
                    let lit = self.dimacs_to_lit(lit_val)?;
                    current_clause.push(lit);
                }
            }
        }

        // Handle case where last clause doesn't end with 0
        if !current_clause.is_empty() {
            solver.add_clause(current_clause.iter().copied());
            clauses_read += 1;
        }

        // Verify we read the expected number of clauses
        if self.num_clauses > 0 && clauses_read != self.num_clauses {
            eprintln!(
                "Warning: Expected {} clauses but read {}",
                self.num_clauses, clauses_read
            );
        }

        Ok(())
    }

    /// Parse the problem line: p cnf <num_vars> <num_clauses>
    fn parse_problem_line(&mut self, line: &str) -> Result<(), DimacsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 4 || parts[0] != "p" || parts[1] != "cnf" {
            return Err(DimacsError::InvalidProblem);
        }

        self.num_vars = parts[2]
            .parse()
            .map_err(|_| DimacsError::Parse("Invalid number of variables".to_string()))?;
        self.num_clauses = parts[3]
            .parse()
            .map_err(|_| DimacsError::Parse("Invalid number of clauses".to_string()))?;

        Ok(())
    }

    /// Convert DIMACS literal (1-indexed, negative for negation) to internal Lit
    fn dimacs_to_lit(&self, dimacs_lit: i32) -> Result<Lit, DimacsError> {
        if dimacs_lit == 0 {
            return Err(DimacsError::Parse("Literal cannot be 0".to_string()));
        }

        let abs_val = dimacs_lit.unsigned_abs();
        if abs_val as usize > self.num_vars {
            return Err(DimacsError::LiteralOutOfRange(dimacs_lit));
        }

        // DIMACS uses 1-indexed variables, we use 0-indexed
        let var = Var::new(abs_val - 1);
        Ok(if dimacs_lit > 0 {
            Lit::pos(var)
        } else {
            Lit::neg(var)
        })
    }

    /// Get number of variables
    #[must_use]
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get number of clauses
    #[must_use]
    pub const fn num_clauses(&self) -> usize {
        self.num_clauses
    }
}

impl Default for DimacsParser {
    fn default() -> Self {
        Self::new()
    }
}

/// DIMACS CNF writer
pub struct DimacsWriter;

impl DimacsWriter {
    /// Write a SAT problem to a file in DIMACS format
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn write_cnf<P: AsRef<Path>>(
        path: P,
        num_vars: usize,
        clauses: &[Vec<Lit>],
    ) -> Result<(), DimacsError> {
        let mut file = File::create(path)?;
        Self::write_cnf_to(&mut file, num_vars, clauses)
    }

    /// Write CNF to a writer
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails
    pub fn write_cnf_to<W: Write>(
        writer: &mut W,
        num_vars: usize,
        clauses: &[Vec<Lit>],
    ) -> Result<(), DimacsError> {
        // Write header
        writeln!(writer, "c DIMACS CNF")?;
        writeln!(writer, "p cnf {} {}", num_vars, clauses.len())?;

        // Write clauses
        for clause in clauses {
            for &lit in clause {
                write!(writer, "{} ", Self::lit_to_dimacs(lit))?;
            }
            writeln!(writer, "0")?;
        }

        Ok(())
    }

    /// Write a model (satisfying assignment) to a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn write_model<P: AsRef<Path>>(
        path: P,
        solver: &Solver,
        result: SolverResult,
    ) -> Result<(), DimacsError> {
        let mut file = File::create(path)?;
        Self::write_model_to(&mut file, solver, result)
    }

    /// Write model to a writer
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails
    pub fn write_model_to<W: Write>(
        writer: &mut W,
        solver: &Solver,
        result: SolverResult,
    ) -> Result<(), DimacsError> {
        use crate::literal::LBool;

        match result {
            SolverResult::Sat => {
                writeln!(writer, "s SATISFIABLE")?;
                write!(writer, "v ")?;
                for i in 0..solver.num_vars() {
                    let var = Var::new(i as u32);
                    let value = solver.model_value(var);
                    let dimacs_lit = if value == LBool::True {
                        (i + 1) as i32
                    } else {
                        -((i + 1) as i32)
                    };
                    write!(writer, "{dimacs_lit} ")?;
                }
                writeln!(writer, "0")?;
            }
            SolverResult::Unsat => {
                writeln!(writer, "s UNSATISFIABLE")?;
            }
            SolverResult::Unknown => {
                writeln!(writer, "s UNKNOWN")?;
            }
        }
        Ok(())
    }

    /// Convert internal Lit to DIMACS literal
    fn lit_to_dimacs(lit: Lit) -> i32 {
        let var_index = lit.var().index() as i32;
        if lit.is_pos() {
            var_index + 1
        } else {
            -(var_index + 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_cnf() {
        let cnf = "c Simple test\n\
                   p cnf 3 2\n\
                   1 -3 0\n\
                   2 3 -1 0\n";

        let mut parser = DimacsParser::new();
        let mut solver = Solver::new();

        parser.parse_reader(cnf.as_bytes(), &mut solver).unwrap();

        assert_eq!(parser.num_vars(), 3);
        assert_eq!(parser.num_clauses(), 2);
    }

    #[test]
    fn test_parse_with_comments() {
        let cnf = "c This is a comment\n\
                   c Another comment\n\
                   p cnf 2 1\n\
                   c Comment in the middle\n\
                   1 2 0\n";

        let mut parser = DimacsParser::new();
        let mut solver = Solver::new();

        parser.parse_reader(cnf.as_bytes(), &mut solver).unwrap();

        assert_eq!(parser.num_vars(), 2);
        assert_eq!(parser.num_clauses(), 1);
    }

    #[test]
    fn test_write_cnf() {
        let mut buffer = Vec::new();
        let clauses = vec![
            vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(2))],
            vec![
                Lit::pos(Var::new(1)),
                Lit::pos(Var::new(2)),
                Lit::neg(Var::new(0)),
            ],
        ];

        DimacsWriter::write_cnf_to(&mut buffer, 3, &clauses).unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("p cnf 3 2"));
        assert!(output.contains("1 -3 0"));
        assert!(output.contains("2 3 -1 0"));
    }

    #[test]
    fn test_roundtrip() {
        // Create a simple formula
        let original_clauses = vec![
            vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))],
            vec![Lit::pos(Var::new(1)), Lit::neg(Var::new(2))],
            vec![Lit::pos(Var::new(2)), Lit::neg(Var::new(0))],
        ];

        // Write to buffer
        let mut buffer = Vec::new();
        DimacsWriter::write_cnf_to(&mut buffer, 3, &original_clauses).unwrap();

        // Parse back
        let mut parser = DimacsParser::new();
        let mut solver = Solver::new();
        parser.parse_reader(buffer.as_slice(), &mut solver).unwrap();

        assert_eq!(parser.num_vars(), 3);
        assert_eq!(parser.num_clauses(), 3);
    }
}
