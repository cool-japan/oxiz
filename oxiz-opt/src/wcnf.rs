//! WCNF (Weighted CNF) format parser for MaxSAT competition instances.
//!
//! The WCNF format is the standard input format for MaxSAT solvers in competitions.
//!
//! Format:
//! ```text
//! c comment lines start with 'c'
//! p wcnf <num_vars> <num_clauses> <top_weight>
//! <weight> <lit1> <lit2> ... <litn> 0
//! ...
//! ```
//!
//! Where:
//! - `top_weight`: Weight used for hard clauses (infinity)
//! - Each clause line: `<weight> <literals...> 0`
//! - Hard clauses: weight = top_weight
//! - Soft clauses: weight < top_weight
//!
//! Reference: MaxSAT Evaluation format specification

use crate::maxsat::{MaxSatSolver, Weight};
use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_sat::Lit;
use std::io::{BufRead, BufReader, Read};
use thiserror::Error;

/// Errors from WCNF parsing
#[derive(Error, Debug)]
pub enum WcnfError {
    /// IO error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Parse error
    #[error("parse error: {0}")]
    Parse(String),
    /// Invalid format
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    /// Missing problem line
    #[error("missing problem line")]
    MissingProblem,
}

/// WCNF instance representation
#[derive(Debug, Clone)]
pub struct WcnfInstance {
    /// Number of variables
    pub num_vars: u32,
    /// Number of clauses
    pub num_clauses: u32,
    /// Top weight (used for hard clauses)
    pub top_weight: Weight,
    /// Hard clauses
    pub hard_clauses: Vec<Vec<Lit>>,
    /// Soft clauses with weights
    pub soft_clauses: Vec<(Weight, Vec<Lit>)>,
}

impl WcnfInstance {
    /// Parse a WCNF instance from a reader
    pub fn parse<R: Read>(reader: R) -> Result<Self, WcnfError> {
        let buf_reader = BufReader::new(reader);
        let lines = buf_reader.lines();

        let mut num_vars = 0;
        let mut num_clauses = 0;
        let mut top_weight = Weight::Infinite;
        let mut hard_clauses = Vec::new();
        let mut soft_clauses = Vec::new();
        let mut found_problem = false;

        for line in lines {
            let line = line?;
            let line = line.trim();

            // Skip empty lines
            if line.is_empty() {
                continue;
            }

            // Skip comment lines
            if line.starts_with('c') {
                continue;
            }

            // Parse problem line
            if line.starts_with("p wcnf") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 5 {
                    return Err(WcnfError::InvalidFormat(
                        "problem line must have format: p wcnf <vars> <clauses> <top>".to_string(),
                    ));
                }

                num_vars = parts[2]
                    .parse()
                    .map_err(|_| WcnfError::Parse(format!("invalid num_vars: {}", parts[2])))?;

                num_clauses = parts[3]
                    .parse()
                    .map_err(|_| WcnfError::Parse(format!("invalid num_clauses: {}", parts[3])))?;

                // Parse top weight
                let top_str = parts[4];
                top_weight = Self::parse_weight(top_str)?;

                found_problem = true;
                continue;
            }

            // Parse clause line
            if found_problem {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }

                // First element is the weight
                let weight_str = parts[0];
                let weight = Self::parse_weight(weight_str)?;

                // Parse literals (skip first element which is weight, and last element which should be 0)
                let mut lits = Vec::new();
                for lit_str in parts.iter().skip(1) {
                    let lit_int: i32 = lit_str
                        .parse()
                        .map_err(|_| WcnfError::Parse(format!("invalid literal: {}", lit_str)))?;

                    if lit_int == 0 {
                        break;
                    }

                    lits.push(Lit::from_dimacs(lit_int));
                }

                // Categorize as hard or soft clause
                if weight == top_weight {
                    hard_clauses.push(lits);
                } else {
                    soft_clauses.push((weight, lits));
                }
            }
        }

        if !found_problem {
            return Err(WcnfError::MissingProblem);
        }

        Ok(Self {
            num_vars,
            num_clauses,
            top_weight,
            hard_clauses,
            soft_clauses,
        })
    }

    /// Parse a weight from string
    fn parse_weight(s: &str) -> Result<Weight, WcnfError> {
        // Check for special infinity markers
        if s == "h" || s == "H" || s == "top" {
            return Ok(Weight::Infinite);
        }

        // Try parsing as integer
        if let Ok(n) = s.parse::<u64>() {
            // Very large numbers are treated as infinity
            if n >= u64::MAX / 2 {
                return Ok(Weight::Infinite);
            }
            // Convert to BigInt for weights
            return Ok(Weight::from(BigInt::from(n)));
        }

        // Try parsing as rational (numerator/denominator)
        if s.contains('/') {
            let parts: Vec<&str> = s.split('/').collect();
            if parts.len() == 2 {
                let num: i64 = parts[0].parse().map_err(|_| {
                    WcnfError::Parse(format!("invalid rational numerator: {}", parts[0]))
                })?;
                let den: i64 = parts[1].parse().map_err(|_| {
                    WcnfError::Parse(format!("invalid rational denominator: {}", parts[1]))
                })?;
                let rational = BigRational::new(BigInt::from(num), BigInt::from(den));
                return Ok(Weight::Rational(rational));
            }
        }

        Err(WcnfError::Parse(format!("invalid weight: {}", s)))
    }

    /// Load into a MaxSAT solver
    pub fn load_into_solver(&self, solver: &mut MaxSatSolver) {
        // Add hard clauses
        for clause in &self.hard_clauses {
            solver.add_hard(clause.iter().copied());
        }

        // Add soft clauses
        for (weight, clause) in &self.soft_clauses {
            solver.add_soft_weighted(clause.iter().copied(), weight.clone());
        }
    }

    /// Create a solver from this instance
    pub fn to_solver(&self) -> MaxSatSolver {
        let mut solver = MaxSatSolver::new();
        self.load_into_solver(&mut solver);
        solver
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_wcnf() {
        let input = r#"c Simple WCNF instance
p wcnf 3 4 10
10 1 2 0
10 -1 3 0
5 1 0
3 -2 -3 0
"#;

        let instance = WcnfInstance::parse(input.as_bytes()).unwrap();
        assert_eq!(instance.num_vars, 3);
        assert_eq!(instance.num_clauses, 4);
        assert_eq!(instance.top_weight, Weight::from(10));
        assert_eq!(instance.hard_clauses.len(), 2);
        assert_eq!(instance.soft_clauses.len(), 2);
    }

    #[test]
    fn test_parse_infinite_top() {
        let input = r#"p wcnf 2 3 h
h 1 2 0
5 1 0
3 -1 -2 0
"#;

        let instance = WcnfInstance::parse(input.as_bytes()).unwrap();
        assert_eq!(instance.num_vars, 2);
        assert_eq!(instance.top_weight, Weight::Infinite);
        assert_eq!(instance.hard_clauses.len(), 1);
        assert_eq!(instance.soft_clauses.len(), 2);
    }

    #[test]
    fn test_parse_large_top_as_infinite() {
        let input = format!("p wcnf 2 2 {}\n{} 1 0\n5 -1 0\n", u64::MAX, u64::MAX);

        let instance = WcnfInstance::parse(input.as_bytes()).unwrap();
        assert_eq!(instance.top_weight, Weight::Infinite);
        assert_eq!(instance.hard_clauses.len(), 1);
        assert_eq!(instance.soft_clauses.len(), 1);
    }

    #[test]
    fn test_parse_with_comments() {
        let input = r#"c This is a comment
c Another comment
p wcnf 2 2 10
c Comment in the middle
10 1 2 0
5 -1 0
"#;

        let instance = WcnfInstance::parse(input.as_bytes()).unwrap();
        assert_eq!(instance.num_vars, 2);
        assert_eq!(instance.hard_clauses.len(), 1);
        assert_eq!(instance.soft_clauses.len(), 1);
    }

    #[test]
    fn test_parse_empty_clauses() {
        let input = r#"p wcnf 1 2 10
10 0
5 1 0
"#;

        let instance = WcnfInstance::parse(input.as_bytes()).unwrap();
        assert_eq!(instance.hard_clauses.len(), 1);
        assert_eq!(instance.hard_clauses[0].len(), 0); // Empty clause
        assert_eq!(instance.soft_clauses.len(), 1);
    }

    #[test]
    fn test_load_into_solver() {
        let input = r#"p wcnf 3 3 10
10 1 2 0
5 1 0
3 -2 -3 0
"#;

        let instance = WcnfInstance::parse(input.as_bytes()).unwrap();
        let mut solver = MaxSatSolver::new();
        instance.load_into_solver(&mut solver);

        // Solver should have the clauses loaded
        // We can't directly check the internal state, but we can solve
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_solver() {
        let input = r#"p wcnf 2 2 10
10 1 0
5 -1 0
"#;

        let instance = WcnfInstance::parse(input.as_bytes()).unwrap();
        let mut solver = instance.to_solver();
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_missing_problem() {
        let input = "5 1 0\n";
        let result = WcnfInstance::parse(input.as_bytes());
        assert!(matches!(result, Err(WcnfError::MissingProblem)));
    }

    #[test]
    fn test_parse_invalid_problem_line() {
        let input = "p wcnf 2\n";
        let result = WcnfInstance::parse(input.as_bytes());
        assert!(matches!(result, Err(WcnfError::InvalidFormat(_))));
    }
}
