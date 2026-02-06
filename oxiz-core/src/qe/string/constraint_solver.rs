//! String Constraint Solver for Quantifier Elimination.
//!
//! Implements constraint solving for string theory including:
//! - Length constraints propagation
//! - Regular expression constraints
//! - Prefix/suffix/contains constraints
//! - Concatenation reasoning

use num_bigint::BigInt;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// String constraint solver for QE.
pub struct StringConstraintSolver {
    /// Variable to length bounds mapping
    length_bounds: FxHashMap<String, LengthBound>,
    /// Concatenation constraints
    concat_constraints: Vec<ConcatConstraint>,
    /// Regular expression constraints
    regex_constraints: Vec<RegexConstraint>,
    /// Contains constraints
    contains_constraints: Vec<ContainsConstraint>,
    /// Statistics
    stats: StringSolverStats,
}

/// Length bound for a string variable.
#[derive(Debug, Clone)]
pub struct LengthBound {
    /// Minimum length (inclusive)
    pub min_length: BigInt,
    /// Maximum length (inclusive, None = unbounded)
    pub max_length: Option<BigInt>,
}

/// Concatenation constraint: x = y · z
#[derive(Debug, Clone)]
pub struct ConcatConstraint {
    /// Result variable
    pub result: String,
    /// Left operand
    pub left: String,
    /// Right operand
    pub right: String,
}

/// Regular expression constraint.
#[derive(Debug, Clone)]
pub struct RegexConstraint {
    /// Variable constrained by regex
    pub var: String,
    /// Regular expression pattern (simplified representation)
    pub pattern: RegexPattern,
}

/// Simplified regex pattern.
#[derive(Debug, Clone)]
pub enum RegexPattern {
    /// Empty string
    Empty,
    /// Single character
    Char(char),
    /// Character class [a-z]
    CharClass(Vec<(char, char)>),
    /// Concatenation
    Concat(Vec<RegexPattern>),
    /// Alternation (|)
    Alt(Vec<RegexPattern>),
    /// Kleene star (*)
    Star(Box<RegexPattern>),
    /// Plus (+)
    Plus(Box<RegexPattern>),
    /// Optional (?)
    Optional(Box<RegexPattern>),
}

/// Contains constraint: x contains y
#[derive(Debug, Clone)]
pub struct ContainsConstraint {
    /// Haystack variable
    pub haystack: String,
    /// Needle variable or constant
    pub needle: String,
}

/// String solver statistics.
#[derive(Debug, Clone, Default)]
pub struct StringSolverStats {
    /// Length constraints propagated
    pub length_propagations: usize,
    /// Concatenations resolved
    pub concat_resolutions: usize,
    /// Regex constraints checked
    pub regex_checks: usize,
    /// Contains checks
    pub contains_checks: usize,
    /// Conflicts detected
    pub conflicts_detected: usize,
}

impl StringConstraintSolver {
    /// Create a new string constraint solver.
    pub fn new() -> Self {
        Self {
            length_bounds: FxHashMap::default(),
            concat_constraints: Vec::new(),
            regex_constraints: Vec::new(),
            contains_constraints: Vec::new(),
            stats: StringSolverStats::default(),
        }
    }

    /// Add a length constraint: len(var) OP bound.
    pub fn add_length_constraint(
        &mut self,
        var: String,
        min: Option<BigInt>,
        max: Option<BigInt>,
    ) -> Result<(), String> {
        let entry = self
            .length_bounds
            .entry(var.clone())
            .or_insert(LengthBound {
                min_length: BigInt::zero(),
                max_length: None,
            });

        // Update minimum
        if let Some(min) = min
            && min > entry.min_length
        {
            entry.min_length = min;
            self.stats.length_propagations += 1;
        }

        // Update maximum
        if let Some(max) = max {
            match &entry.max_length {
                None => {
                    entry.max_length = Some(max);
                    self.stats.length_propagations += 1;
                }
                Some(existing_max) => {
                    if max < *existing_max {
                        entry.max_length = Some(max);
                        self.stats.length_propagations += 1;
                    }
                }
            }
        }

        // Check for conflicts
        if let Some(ref max_len) = entry.max_length
            && entry.min_length > *max_len
        {
            self.stats.conflicts_detected += 1;
            return Err(format!(
                "Length conflict for {}: min {} > max {}",
                var, entry.min_length, max_len
            ));
        }

        Ok(())
    }

    /// Add concatenation constraint: result = left · right.
    pub fn add_concat_constraint(
        &mut self,
        result: String,
        left: String,
        right: String,
    ) -> Result<(), String> {
        self.concat_constraints.push(ConcatConstraint {
            result: result.clone(),
            left: left.clone(),
            right: right.clone(),
        });

        // Propagate length constraints
        self.propagate_concat_lengths(&result, &left, &right)?;

        Ok(())
    }

    /// Propagate length constraints for concatenation.
    fn propagate_concat_lengths(
        &mut self,
        result: &str,
        left: &str,
        right: &str,
    ) -> Result<(), String> {
        // len(result) = len(left) + len(right)

        let left_bound = self.get_length_bound(left);
        let right_bound = self.get_length_bound(right);
        let result_bound = self.get_length_bound(result);

        // Forward propagation: result's bounds from left and right
        let new_result_min = &left_bound.min_length + &right_bound.min_length;
        let new_result_max = match (&left_bound.max_length, &right_bound.max_length) {
            (Some(l_max), Some(r_max)) => Some(l_max + r_max),
            _ => None, // Unbounded
        };

        self.add_length_constraint(
            result.to_string(),
            Some(new_result_min.clone()),
            new_result_max.clone(),
        )?;

        // Backward propagation: constrain left and right from result
        if let Some(ref result_max) = result_bound.max_length {
            // len(left) ≤ len(result) - len(right).min
            let left_max = result_max - &right_bound.min_length;
            if left_max >= BigInt::zero() {
                self.add_length_constraint(left.to_string(), None, Some(left_max))?;
            }

            // len(right) ≤ len(result) - len(left).min
            let right_max = result_max - &left_bound.min_length;
            if right_max >= BigInt::zero() {
                self.add_length_constraint(right.to_string(), None, Some(right_max))?;
            }
        }

        // len(left) ≥ len(result).min - len(right).max (if right bounded)
        if let Some(ref right_max) = right_bound.max_length {
            let left_min = &result_bound.min_length - right_max;
            if left_min > BigInt::zero() {
                self.add_length_constraint(left.to_string(), Some(left_min), None)?;
            }
        }

        // len(right) ≥ len(result).min - len(left).max (if left bounded)
        if let Some(ref left_max) = left_bound.max_length {
            let right_min = &result_bound.min_length - left_max;
            if right_min > BigInt::zero() {
                self.add_length_constraint(right.to_string(), Some(right_min), None)?;
            }
        }

        self.stats.concat_resolutions += 1;
        Ok(())
    }

    /// Get length bound for a variable.
    fn get_length_bound(&self, var: &str) -> LengthBound {
        self.length_bounds.get(var).cloned().unwrap_or(LengthBound {
            min_length: BigInt::zero(),
            max_length: None,
        })
    }

    /// Add regex constraint: var ∈ L(regex).
    pub fn add_regex_constraint(&mut self, var: String, pattern: RegexPattern) {
        // Compute length bounds from regex
        let (min_len, max_len) = self.regex_length_bounds(&pattern);

        if let Err(_e) = self.add_length_constraint(var.clone(), Some(min_len), max_len) {
            // Conflict detected
            self.stats.conflicts_detected += 1;
        }

        self.regex_constraints
            .push(RegexConstraint { var, pattern });
        self.stats.regex_checks += 1;
    }

    /// Compute length bounds from regex pattern.
    fn regex_length_bounds(&self, pattern: &RegexPattern) -> (BigInt, Option<BigInt>) {
        match pattern {
            RegexPattern::Empty => (BigInt::zero(), Some(BigInt::zero())),
            RegexPattern::Char(_) => (BigInt::one(), Some(BigInt::one())),
            RegexPattern::CharClass(_) => (BigInt::one(), Some(BigInt::one())),
            RegexPattern::Concat(patterns) => {
                let mut min = BigInt::zero();
                let mut max: Option<BigInt> = Some(BigInt::zero());
                for p in patterns {
                    let (p_min, p_max) = self.regex_length_bounds(p);
                    min += p_min;
                    max = match (max, p_max) {
                        (Some(m), Some(pm)) => Some(m + pm),
                        _ => None, // Unbounded
                    };
                }
                (min, max)
            }
            RegexPattern::Alt(patterns) => {
                if patterns.is_empty() {
                    return (BigInt::zero(), Some(BigInt::zero()));
                }
                let mut min = BigInt::zero();
                let mut max: Option<BigInt> = None;
                for (i, p) in patterns.iter().enumerate() {
                    let (p_min, p_max) = self.regex_length_bounds(p);
                    if i == 0 {
                        min = p_min.clone();
                        max = p_max;
                    } else {
                        if p_min < min {
                            min = p_min;
                        }
                        match (&max, p_max) {
                            (Some(m), Some(pm)) => {
                                if pm > *m {
                                    max = Some(pm);
                                }
                            }
                            _ => max = None, // Unbounded
                        }
                    }
                }
                (min, max)
            }
            RegexPattern::Star(_) => (BigInt::zero(), None), // Unbounded
            RegexPattern::Plus(inner) => {
                let (inner_min, _) = self.regex_length_bounds(inner);
                (inner_min, None) // Unbounded
            }
            RegexPattern::Optional(inner) => {
                let (_, inner_max) = self.regex_length_bounds(inner);
                (BigInt::zero(), inner_max)
            }
        }
    }

    /// Add contains constraint: haystack contains needle.
    pub fn add_contains_constraint(&mut self, haystack: String, needle: String) {
        // Length constraint: len(haystack) ≥ len(needle)
        let needle_bound = self.get_length_bound(&needle);
        if self
            .add_length_constraint(
                haystack.clone(),
                Some(needle_bound.min_length.clone()),
                None,
            )
            .is_err()
        {
            self.stats.conflicts_detected += 1;
        }

        self.contains_constraints
            .push(ContainsConstraint { haystack, needle });
        self.stats.contains_checks += 1;
    }

    /// Check if constraints are satisfiable.
    pub fn is_satisfiable(&self) -> bool {
        // Check for length conflicts
        for bound in self.length_bounds.values() {
            if let Some(ref max) = bound.max_length
                && bound.min_length > *max
            {
                return false;
            }
        }

        // Additional consistency checks would go here
        true
    }

    /// Get all variables with their length bounds.
    pub fn get_length_bounds(&self) -> &FxHashMap<String, LengthBound> {
        &self.length_bounds
    }

    /// Get statistics.
    pub fn stats(&self) -> &StringSolverStats {
        &self.stats
    }
}

impl Default for StringConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_constraint_solver() {
        let solver = StringConstraintSolver::new();
        assert_eq!(solver.stats.length_propagations, 0);
    }

    #[test]
    fn test_length_constraint() {
        let mut solver = StringConstraintSolver::new();

        solver
            .add_length_constraint(
                "x".to_string(),
                Some(BigInt::from(5)),
                Some(BigInt::from(10)),
            )
            .unwrap();

        let bound = solver.get_length_bound("x");
        assert_eq!(bound.min_length, BigInt::from(5));
        assert_eq!(bound.max_length, Some(BigInt::from(10)));
    }

    #[test]
    fn test_length_conflict() {
        let mut solver = StringConstraintSolver::new();

        solver
            .add_length_constraint("x".to_string(), Some(BigInt::from(10)), None)
            .unwrap();

        let result = solver.add_length_constraint("x".to_string(), None, Some(BigInt::from(5)));

        assert!(result.is_err());
        assert_eq!(solver.stats.conflicts_detected, 1);
    }

    #[test]
    fn test_concat_constraint() {
        let mut solver = StringConstraintSolver::new();

        // x = "hello" (len 5), y = "world" (len 5)
        solver
            .add_length_constraint(
                "x".to_string(),
                Some(BigInt::from(5)),
                Some(BigInt::from(5)),
            )
            .unwrap();
        solver
            .add_length_constraint(
                "y".to_string(),
                Some(BigInt::from(5)),
                Some(BigInt::from(5)),
            )
            .unwrap();

        // z = x · y
        solver
            .add_concat_constraint("z".to_string(), "x".to_string(), "y".to_string())
            .unwrap();

        let z_bound = solver.get_length_bound("z");
        assert_eq!(z_bound.min_length, BigInt::from(10));
        assert_eq!(z_bound.max_length, Some(BigInt::from(10)));
    }

    #[test]
    fn test_regex_char() {
        let mut solver = StringConstraintSolver::new();

        solver.add_regex_constraint("x".to_string(), RegexPattern::Char('a'));

        let bound = solver.get_length_bound("x");
        assert_eq!(bound.min_length, BigInt::one());
        assert_eq!(bound.max_length, Some(BigInt::one()));
    }

    #[test]
    fn test_regex_star() {
        let mut solver = StringConstraintSolver::new();

        let pattern = RegexPattern::Star(Box::new(RegexPattern::Char('a')));
        solver.add_regex_constraint("x".to_string(), pattern);

        let bound = solver.get_length_bound("x");
        assert_eq!(bound.min_length, BigInt::zero());
        assert_eq!(bound.max_length, None); // Unbounded
    }

    #[test]
    fn test_contains_constraint() {
        let mut solver = StringConstraintSolver::new();

        solver
            .add_length_constraint(
                "needle".to_string(),
                Some(BigInt::from(3)),
                Some(BigInt::from(3)),
            )
            .unwrap();
        solver.add_contains_constraint("haystack".to_string(), "needle".to_string());

        let haystack_bound = solver.get_length_bound("haystack");
        assert!(haystack_bound.min_length >= BigInt::from(3));
    }

    #[test]
    fn test_is_satisfiable() {
        let mut solver = StringConstraintSolver::new();

        solver
            .add_length_constraint(
                "x".to_string(),
                Some(BigInt::from(5)),
                Some(BigInt::from(10)),
            )
            .unwrap();
        assert!(solver.is_satisfiable());

        let mut solver2 = StringConstraintSolver::new();
        solver2
            .add_length_constraint("y".to_string(), Some(BigInt::from(10)), None)
            .unwrap();
        let _ = solver2.add_length_constraint("y".to_string(), None, Some(BigInt::from(5)));
        assert!(!solver2.is_satisfiable());
    }
}
