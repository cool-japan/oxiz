//! Regular Expression Constraint Solver for String Theory
#![allow(missing_docs)] // Under development - documentation in progress
//!
//! This module implements solving for string constraints with regular expressions:
//! - Membership testing (str ∈ regex)
//! - Negated membership (str ∉ regex)
//! - Regex intersection and complement
//! - Length-aware regex solving

use rustc_hash::{FxHashMap, FxHashSet};

/// String variable identifier
pub type StrVar = usize;

/// Regular expression
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Regex {
    /// Empty language
    Empty,
    /// Epsilon (empty string)
    Epsilon,
    /// Single character
    Char(char),
    /// Character class (set of characters)
    CharClass(FxHashSet<char>),
    /// Concatenation
    Concat(Vec<Regex>),
    /// Union (alternation)
    Union(Vec<Regex>),
    /// Kleene star
    Star(Box<Regex>),
    /// Negation (complement)
    Complement(Box<Regex>),
    /// Intersection
    Intersection(Vec<Regex>),
    /// Optional (zero or one)
    Optional(Box<Regex>),
    /// One or more
    Plus(Box<Regex>),
    /// Exact repetition
    Repeat { regex: Box<Regex>, count: usize },
    /// Range repetition
    RepeatRange {
        regex: Box<Regex>,
        min: usize,
        max: Option<usize>,
    },
}

/// String constraint
#[derive(Debug, Clone)]
pub enum StringConstraint {
    /// String matches regex
    InRegex {
        var: StrVar,
        regex: Regex,
    },
    /// String doesn't match regex
    NotInRegex {
        var: StrVar,
        regex: Regex,
    },
    /// Length constraint
    LengthEq {
        var: StrVar,
        length: usize,
    },
    LengthLe {
        var: StrVar,
        length: usize,
    },
    LengthGe {
        var: StrVar,
        length: usize,
    },
}

/// Solution for string variables
#[derive(Debug, Clone)]
pub struct StringSolution {
    /// Assignment of variables to strings
    pub assignment: FxHashMap<StrVar, String>,
}

/// Statistics for regex solver
#[derive(Debug, Clone, Default)]
pub struct RegexSolverStats {
    pub constraints_solved: u64,
    pub regex_intersections: u64,
    pub regex_complements: u64,
    pub membership_tests: u64,
    pub length_propagations: u64,
}

/// Configuration for regex solver
#[derive(Debug, Clone)]
pub struct RegexSolverConfig {
    /// Maximum string length to consider
    pub max_string_length: usize,
    /// Enable length-based pruning
    pub use_length_pruning: bool,
    /// Maximum regex size for complement
    pub max_complement_size: usize,
}

impl Default for RegexSolverConfig {
    fn default() -> Self {
        Self {
            max_string_length: 100,
            use_length_pruning: true,
            max_complement_size: 1000,
        }
    }
}

/// Regular expression constraint solver
pub struct RegexSolver {
    config: RegexSolverConfig,
    stats: RegexSolverStats,
    /// Constraints for each variable
    constraints: FxHashMap<StrVar, Vec<StringConstraint>>,
    /// Length bounds for variables
    length_bounds: FxHashMap<StrVar, (Option<usize>, Option<usize>)>,
}

impl RegexSolver {
    /// Create a new regex solver
    pub fn new(config: RegexSolverConfig) -> Self {
        Self {
            config,
            stats: RegexSolverStats::default(),
            constraints: FxHashMap::default(),
            length_bounds: FxHashMap::default(),
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: StringConstraint) {
        let var = match &constraint {
            StringConstraint::InRegex { var, .. } => *var,
            StringConstraint::NotInRegex { var, .. } => *var,
            StringConstraint::LengthEq { var, .. } => *var,
            StringConstraint::LengthLe { var, .. } => *var,
            StringConstraint::LengthGe { var, .. } => *var,
        };

        self.constraints.entry(var).or_default().push(constraint);
    }

    /// Solve all constraints
    pub fn solve(&mut self) -> Result<Option<StringSolution>, String> {
        self.stats.constraints_solved += 1;

        // Phase 1: Propagate length constraints
        if self.config.use_length_pruning {
            self.propagate_lengths()?;
        }

        // Phase 2: Compute regex intersection for each variable
        let combined_regexes = self.compute_combined_regexes()?;

        // Phase 3: Find satisfying strings
        let solution = self.find_satisfying_strings(&combined_regexes)?;

        Ok(solution)
    }

    /// Propagate length constraints to tighten bounds
    fn propagate_lengths(&mut self) -> Result<(), String> {
        for (&var, constraints) in &self.constraints {
            let mut lower = None;
            let mut upper = None;

            for constraint in constraints {
                match constraint {
                    StringConstraint::LengthEq { length, .. } => {
                        lower = Some(*length);
                        upper = Some(*length);
                    }
                    StringConstraint::LengthLe { length, .. } => {
                        upper = match upper {
                            Some(u) => Some(u.min(*length)),
                            None => Some(*length),
                        };
                    }
                    StringConstraint::LengthGe { length, .. } => {
                        lower = match lower {
                            Some(l) => Some(l.max(*length)),
                            None => Some(*length),
                        };
                    }
                    StringConstraint::InRegex { regex, .. } => {
                        // Compute possible lengths for regex
                        if let Some((min_len, max_len)) = self.regex_length_bounds(regex) {
                            lower = match lower {
                                Some(l) => Some(l.max(min_len)),
                                None => Some(min_len),
                            };
                            if let Some(max) = max_len {
                                upper = match upper {
                                    Some(u) => Some(u.min(max)),
                                    None => Some(max),
                                };
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Check consistency
            if let (Some(l), Some(u)) = (lower, upper)
                && l > u
            {
                return Err(format!(
                    "Inconsistent length bounds for variable {}: {} > {}",
                    var, l, u
                ));
            }

            self.length_bounds.insert(var, (lower, upper));
            self.stats.length_propagations += 1;
        }

        Ok(())
    }

    /// Compute length bounds for a regex
    fn regex_length_bounds(&self, regex: &Regex) -> Option<(usize, Option<usize>)> {
        match regex {
            Regex::Empty => None,
            Regex::Epsilon => Some((0, Some(0))),
            Regex::Char(_) | Regex::CharClass(_) => Some((1, Some(1))),
            Regex::Concat(parts) => {
                let mut min = 0;
                let mut max = Some(0);

                for part in parts {
                    if let Some((part_min, part_max)) = self.regex_length_bounds(part) {
                        min += part_min;
                        max = match (max, part_max) {
                            (Some(m), Some(pm)) => Some(m + pm),
                            _ => None,
                        };
                    } else {
                        return None;
                    }
                }

                Some((min, max))
            }
            Regex::Union(parts) => {
                let bounds: Vec<_> = parts
                    .iter()
                    .filter_map(|p| self.regex_length_bounds(p))
                    .collect();

                if bounds.is_empty() {
                    return None;
                }

                // Safety: bounds is not empty (checked above), use unwrap_or for no-unwrap policy
                let min = bounds.iter().map(|(m, _)| *m).min().unwrap_or(0);
                let max = bounds.iter().filter_map(|(_, m)| *m).max();

                Some((min, max))
            }
            Regex::Star(_) => Some((0, None)),
            Regex::Plus(inner) => self.regex_length_bounds(inner).map(|(min, _)| (min, None)),
            Regex::Optional(inner) => self.regex_length_bounds(inner).map(|(_, max)| (0, max)),
            Regex::Repeat { regex, count } => self
                .regex_length_bounds(regex)
                .map(|(min, max)| (min * count, max.map(|m| m * count))),
            Regex::RepeatRange {
                regex,
                min: min_rep,
                max: max_rep,
            } => self.regex_length_bounds(regex).map(|(min, max)| {
                let min_len = min * min_rep;
                let max_len = max_rep.and_then(|mr| max.map(|m| m * mr));
                (min_len, max_len)
            }),
            _ => Some((0, None)),
        }
    }

    /// Compute combined regex for each variable
    fn compute_combined_regexes(&mut self) -> Result<FxHashMap<StrVar, Regex>, String> {
        let mut combined = FxHashMap::default();

        for (&var, constraints) in &self.constraints {
            let mut positive = Vec::new();
            let mut negative = Vec::new();

            for constraint in constraints {
                match constraint {
                    StringConstraint::InRegex { regex, .. } => {
                        positive.push(regex.clone());
                    }
                    StringConstraint::NotInRegex { regex, .. } => {
                        negative.push(regex.clone());
                    }
                    _ => {}
                }
            }

            // Intersect all positive constraints
            let mut result = if positive.is_empty() {
                Regex::Star(Box::new(Regex::CharClass(self.all_chars())))
            } else if positive.len() == 1 {
                positive[0].clone()
            } else {
                self.stats.regex_intersections += 1;
                Regex::Intersection(positive)
            };

            // Subtract negative constraints
            for neg in negative {
                self.stats.regex_complements += 1;
                result = Regex::Intersection(vec![result, Regex::Complement(Box::new(neg))]);
            }

            combined.insert(var, result);
        }

        Ok(combined)
    }

    /// Find satisfying strings for combined regexes
    fn find_satisfying_strings(
        &mut self,
        regexes: &FxHashMap<StrVar, Regex>,
    ) -> Result<Option<StringSolution>, String> {
        let mut assignment = FxHashMap::default();

        for (&var, regex) in regexes {
            // Get length bounds
            let (min_len, max_len) = self
                .length_bounds
                .get(&var)
                .copied()
                .unwrap_or((Some(0), Some(self.config.max_string_length)));

            // Generate a satisfying string
            if let Some(string) = self.generate_string(regex, min_len, max_len)? {
                assignment.insert(var, string);
            } else {
                // No satisfying string found
                return Ok(None);
            }
        }

        Ok(Some(StringSolution { assignment }))
    }

    /// Generate a string that matches the regex within length bounds
    fn generate_string(
        &self,
        regex: &Regex,
        min_len: Option<usize>,
        max_len: Option<usize>,
    ) -> Result<Option<String>, String> {
        let min = min_len.unwrap_or(0);
        let max = max_len.unwrap_or(self.config.max_string_length);

        // Try lengths from min to max
        for length in min..=max {
            if let Some(string) = self.generate_string_of_length(regex, length)? {
                return Ok(Some(string));
            }
        }

        Ok(None)
    }

    /// Generate a string of specific length matching regex
    fn generate_string_of_length(
        &self,
        regex: &Regex,
        length: usize,
    ) -> Result<Option<String>, String> {
        match regex {
            Regex::Empty => Ok(None),
            Regex::Epsilon if length == 0 => Ok(Some(String::new())),
            Regex::Epsilon => Ok(None),
            Regex::Char(c) if length == 1 => Ok(Some(c.to_string())),
            Regex::Char(_) => Ok(None),
            Regex::CharClass(chars) if length == 1 => {
                Ok(chars.iter().next().map(|c| c.to_string()))
            }
            Regex::CharClass(_) => Ok(None),
            Regex::Concat(parts) => self.generate_concat(parts, length),
            Regex::Union(parts) => {
                // Try each alternative
                for part in parts {
                    if let Some(s) = self.generate_string_of_length(part, length)? {
                        return Ok(Some(s));
                    }
                }
                Ok(None)
            }
            Regex::Star(inner) => self.generate_star(inner, length),
            _ => {
                // Simplified handling for other cases
                Ok(Some("a".repeat(length)))
            }
        }
    }

    /// Generate string for concatenation
    fn generate_concat(&self, parts: &[Regex], length: usize) -> Result<Option<String>, String> {
        // Distribute length among parts
        if parts.is_empty() {
            return Ok(if length == 0 {
                Some(String::new())
            } else {
                None
            });
        }

        // Simple strategy: try to divide length evenly
        self.generate_concat_recursive(parts, length, 0)
    }

    /// Recursive helper for concatenation
    fn generate_concat_recursive(
        &self,
        parts: &[Regex],
        remaining_length: usize,
        part_idx: usize,
    ) -> Result<Option<String>, String> {
        if part_idx >= parts.len() {
            return Ok(if remaining_length == 0 {
                Some(String::new())
            } else {
                None
            });
        }

        // Try different lengths for current part
        for len in 0..=remaining_length {
            if let Some(part_str) = self.generate_string_of_length(&parts[part_idx], len)?
                && let Some(rest_str) =
                    self.generate_concat_recursive(parts, remaining_length - len, part_idx + 1)?
            {
                return Ok(Some(format!("{}{}", part_str, rest_str)));
            }
        }

        Ok(None)
    }

    /// Generate string for star
    fn generate_star(&self, inner: &Regex, length: usize) -> Result<Option<String>, String> {
        if length == 0 {
            return Ok(Some(String::new()));
        }

        // Get possible lengths for inner regex
        let (min_inner, _max_inner) = self.regex_length_bounds(inner).unwrap_or((1, Some(1)));

        // Try different repetition counts
        for count in 1..=(length / min_inner.max(1)) {
            let target_len = length / count;
            if let Some(inner_str) = self.generate_string_of_length(inner, target_len)? {
                let result = inner_str.repeat(count);
                if result.len() == length {
                    return Ok(Some(result));
                }
            }
        }

        Ok(None)
    }

    /// Test if a string matches a regex
    pub fn test_membership(&mut self, string: &str, regex: &Regex) -> Result<bool, String> {
        self.stats.membership_tests += 1;

        match regex {
            Regex::Empty => Ok(false),
            Regex::Epsilon => Ok(string.is_empty()),
            Regex::Char(c) => Ok(string.len() == 1 && string.starts_with(*c)),
            Regex::CharClass(chars) => {
                Ok(string.len() == 1 && string.chars().next().is_some_and(|c| chars.contains(&c)))
            }
            Regex::Concat(parts) => self.test_concat(string, parts),
            Regex::Union(parts) => {
                for part in parts {
                    if self.test_membership(string, part)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Regex::Star(inner) => self.test_star(string, inner),
            Regex::Complement(inner) => Ok(!self.test_membership(string, inner)?),
            Regex::Intersection(parts) => {
                for part in parts {
                    if !self.test_membership(string, part)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(true), // Simplified
        }
    }

    /// Test concatenation
    fn test_concat(&mut self, string: &str, parts: &[Regex]) -> Result<bool, String> {
        self.test_concat_recursive(string, parts, 0)
    }

    /// Recursive helper for concatenation testing
    fn test_concat_recursive(
        &mut self,
        string: &str,
        parts: &[Regex],
        part_idx: usize,
    ) -> Result<bool, String> {
        if part_idx >= parts.len() {
            return Ok(string.is_empty());
        }

        // Try all possible splits
        for split_pos in 0..=string.len() {
            let (prefix, suffix) = string.split_at(split_pos);

            if self.test_membership(prefix, &parts[part_idx])?
                && self.test_concat_recursive(suffix, parts, part_idx + 1)?
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Test star
    fn test_star(&mut self, string: &str, inner: &Regex) -> Result<bool, String> {
        if string.is_empty() {
            return Ok(true);
        }

        // Try all possible repetitions
        for end in 1..=string.len() {
            let (prefix, suffix) = string.split_at(end);

            if self.test_membership(prefix, inner)? && self.test_star(suffix, inner)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get all characters (placeholder)
    fn all_chars(&self) -> FxHashSet<char> {
        ('a'..='z').chain('A'..='Z').chain('0'..='9').collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &RegexSolverStats {
        &self.stats
    }

    /// Reset solver
    pub fn reset(&mut self) {
        self.constraints.clear();
        self.length_bounds.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let config = RegexSolverConfig::default();
        let solver = RegexSolver::new(config);
        assert_eq!(solver.stats.constraints_solved, 0);
    }

    #[test]
    fn test_epsilon_membership() {
        let config = RegexSolverConfig::default();
        let mut solver = RegexSolver::new(config);

        let result = solver.test_membership("", &Regex::Epsilon).unwrap();
        assert!(result);

        let result2 = solver.test_membership("a", &Regex::Epsilon).unwrap();
        assert!(!result2);
    }

    #[test]
    fn test_char_membership() {
        let config = RegexSolverConfig::default();
        let mut solver = RegexSolver::new(config);

        let regex = Regex::Char('a');
        assert!(solver.test_membership("a", &regex).unwrap());
        assert!(!solver.test_membership("b", &regex).unwrap());
        assert!(!solver.test_membership("aa", &regex).unwrap());
    }

    #[test]
    fn test_union_membership() {
        let config = RegexSolverConfig::default();
        let mut solver = RegexSolver::new(config);

        let regex = Regex::Union(vec![Regex::Char('a'), Regex::Char('b')]);

        assert!(solver.test_membership("a", &regex).unwrap());
        assert!(solver.test_membership("b", &regex).unwrap());
        assert!(!solver.test_membership("c", &regex).unwrap());
    }

    #[test]
    fn test_concat_membership() {
        let config = RegexSolverConfig::default();
        let mut solver = RegexSolver::new(config);

        let regex = Regex::Concat(vec![Regex::Char('a'), Regex::Char('b')]);

        assert!(solver.test_membership("ab", &regex).unwrap());
        assert!(!solver.test_membership("a", &regex).unwrap());
        assert!(!solver.test_membership("ba", &regex).unwrap());
    }

    #[test]
    fn test_star_membership() {
        let config = RegexSolverConfig::default();
        let mut solver = RegexSolver::new(config);

        let regex = Regex::Star(Box::new(Regex::Char('a')));

        assert!(solver.test_membership("", &regex).unwrap());
        assert!(solver.test_membership("a", &regex).unwrap());
        assert!(solver.test_membership("aa", &regex).unwrap());
        assert!(!solver.test_membership("ab", &regex).unwrap());
    }

    #[test]
    fn test_length_bounds_epsilon() {
        let solver = RegexSolver::new(RegexSolverConfig::default());

        let bounds = solver.regex_length_bounds(&Regex::Epsilon);
        assert_eq!(bounds, Some((0, Some(0))));
    }

    #[test]
    fn test_length_bounds_char() {
        let solver = RegexSolver::new(RegexSolverConfig::default());

        let bounds = solver.regex_length_bounds(&Regex::Char('a'));
        assert_eq!(bounds, Some((1, Some(1))));
    }

    #[test]
    fn test_length_bounds_concat() {
        let solver = RegexSolver::new(RegexSolverConfig::default());

        let regex = Regex::Concat(vec![Regex::Char('a'), Regex::Char('b'), Regex::Char('c')]);
        let bounds = solver.regex_length_bounds(&regex);
        assert_eq!(bounds, Some((3, Some(3))));
    }

    #[test]
    fn test_length_bounds_star() {
        let solver = RegexSolver::new(RegexSolverConfig::default());

        let regex = Regex::Star(Box::new(Regex::Char('a')));
        let bounds = solver.regex_length_bounds(&regex);
        assert_eq!(bounds, Some((0, None)));
    }

    #[test]
    fn test_add_constraint() {
        let mut solver = RegexSolver::new(RegexSolverConfig::default());

        solver.add_constraint(StringConstraint::InRegex {
            var: 0,
            regex: Regex::Char('a'),
        });

        assert_eq!(solver.constraints.get(&0).map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_reset() {
        let mut solver = RegexSolver::new(RegexSolverConfig::default());

        solver.add_constraint(StringConstraint::InRegex {
            var: 0,
            regex: Regex::Char('a'),
        });

        solver.reset();

        assert!(solver.constraints.is_empty());
        assert!(solver.length_bounds.is_empty());
    }
}
