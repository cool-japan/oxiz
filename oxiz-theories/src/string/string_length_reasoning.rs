//! String Length Constraint Reasoning
//!
//! Advanced length constraint propagation and solving for string theory.
//! Integrates with linear arithmetic solver for length abstraction.
//!
//! ## Features
//!
//! - **Length bounds propagation**: Deduce length bounds from string operations
//! - **Conflict detection**: Detect unsatisfiable length constraints
//! - **Length abstraction**: Convert string constraints to arithmetic constraints
//! - **Operation-aware reasoning**: Handle concat, substring, replace, etc.
//!
//! ## SMT-LIB2 Support
//!
//! ```smt2
//! (assert (= (str.len s) 5))
//! (assert (>= (str.len t) 10))
//! (assert (= (str.len (str.++ s t)) 15))
//! ```

#![allow(missing_docs)]

use oxiz_core::ast::TermId;
use oxiz_core::error::{OxizError, Result};
use rustc_hash::FxHashMap;
use std::cmp::{max, min};
use std::collections::VecDeque;

/// Length variable representing str.len(string_var)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LengthVar {
    /// The string variable
    pub string_var: TermId,
    /// The integer variable representing its length
    pub length_var: TermId,
}

impl LengthVar {
    /// Create a new length variable
    pub fn new(string_var: TermId, length_var: TermId) -> Self {
        Self {
            string_var,
            length_var,
        }
    }
}

/// Length bound for a string variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthBound {
    /// Lower bound (inclusive)
    pub lower: i64,
    /// Upper bound (inclusive), None if unbounded
    pub upper: Option<i64>,
}

impl LengthBound {
    /// Create an exact length bound
    pub fn exact(len: i64) -> Self {
        Self {
            lower: len,
            upper: Some(len),
        }
    }

    /// Create a range bound
    pub fn range(lower: i64, upper: Option<i64>) -> Self {
        Self { lower, upper }
    }

    /// Create a lower bound only
    pub fn at_least(lower: i64) -> Self {
        Self { lower, upper: None }
    }

    /// Create an upper bound only
    pub fn at_most(upper: i64) -> Self {
        Self {
            lower: 0,
            upper: Some(upper),
        }
    }

    /// Unbounded length
    pub fn unbounded() -> Self {
        Self {
            lower: 0,
            upper: None,
        }
    }

    /// Check if this is an exact length
    pub fn is_exact(&self) -> bool {
        self.upper == Some(self.lower)
    }

    /// Get exact length if available
    pub fn exact_value(&self) -> Option<i64> {
        if self.is_exact() {
            Some(self.lower)
        } else {
            None
        }
    }

    /// Intersect with another bound
    pub fn intersect(&self, other: &LengthBound) -> Option<LengthBound> {
        let lower = max(self.lower, other.lower);
        let upper = match (self.upper, other.upper) {
            (Some(a), Some(b)) => Some(min(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        // Check if intersection is empty
        if let Some(u) = upper
            && lower > u
        {
            return None;
        }

        Some(LengthBound { lower, upper })
    }

    /// Check if this bound is satisfiable
    pub fn is_satisfiable(&self) -> bool {
        if self.lower < 0 {
            return false;
        }
        if let Some(u) = self.upper
            && (u < 0 || self.lower > u)
        {
            return false;
        }
        true
    }

    /// Tighten the bound
    pub fn tighten(&mut self, other: &LengthBound) -> bool {
        let old = *self;
        if let Some(new_bound) = self.intersect(other) {
            *self = new_bound;
            *self != old
        } else {
            false
        }
    }
}

/// String operation for length reasoning
#[derive(Debug, Clone)]
pub enum StringOp {
    /// Concatenation: result = s1 ++ s2 ++ ... ++ sn
    Concat {
        result: TermId,
        operands: Vec<TermId>,
    },
    /// Substring: result = substr(s, offset, length)
    Substr {
        result: TermId,
        source: TermId,
        offset: TermId,
        length: TermId,
    },
    /// Replace: result = replace(s, pattern, replacement)
    Replace {
        result: TermId,
        source: TermId,
        pattern: TermId,
        replacement: TermId,
    },
    /// At (character at position): result = at(s, pos)
    At {
        result: TermId,
        source: TermId,
        position: TermId,
    },
    /// Contains: source contains pattern
    Contains { source: TermId, pattern: TermId },
    /// Prefix: pattern is prefix of source
    Prefix { pattern: TermId, source: TermId },
    /// Suffix: pattern is suffix of source
    Suffix { pattern: TermId, source: TermId },
    /// IndexOf: result = indexOf(source, pattern, offset)
    IndexOf {
        result: TermId,
        source: TermId,
        pattern: TermId,
        offset: TermId,
    },
}

/// Length constraint
#[derive(Debug, Clone)]
pub enum LengthConstraint {
    /// Exact length: len(s) = n
    Exact(TermId, i64),
    /// Lower bound: len(s) >= n
    LowerBound(TermId, i64),
    /// Upper bound: len(s) <= n
    UpperBound(TermId, i64),
    /// Equality: len(s1) = len(s2)
    Equal(TermId, TermId),
    /// Inequality: len(s1) + offset = len(s2)
    LinearCombo {
        lhs: Vec<(TermId, i64)>, // (var, coefficient)
        rhs: i64,
    },
}

/// Length reasoning solver
#[derive(Debug)]
pub struct LengthSolver {
    /// Length variables
    length_vars: FxHashMap<TermId, LengthVar>,
    /// Length bounds for each string variable
    bounds: FxHashMap<TermId, LengthBound>,
    /// String operations
    operations: Vec<StringOp>,
    /// Length constraints
    constraints: Vec<LengthConstraint>,
    /// Propagation queue
    propagation_queue: VecDeque<TermId>,
    /// Conflict clause (if unsat)
    conflict: Option<Vec<TermId>>,
}

impl LengthSolver {
    /// Create a new length solver
    pub fn new() -> Self {
        Self {
            length_vars: FxHashMap::default(),
            bounds: FxHashMap::default(),
            operations: Vec::new(),
            constraints: Vec::new(),
            propagation_queue: VecDeque::new(),
            conflict: None,
        }
    }

    /// Register a length variable
    pub fn add_length_var(&mut self, string_var: TermId, length_var: TermId) {
        let lv = LengthVar::new(string_var, length_var);
        self.length_vars.insert(string_var, lv);
        self.bounds.insert(string_var, LengthBound::unbounded());
    }

    /// Add an exact length constraint
    pub fn add_exact_length(&mut self, string_var: TermId, length: i64) -> Result<()> {
        if length < 0 {
            return Err(OxizError::Internal("negative string length".to_string()));
        }

        self.constraints
            .push(LengthConstraint::Exact(string_var, length));
        self.update_bound(string_var, LengthBound::exact(length))?;
        Ok(())
    }

    /// Add a lower bound constraint
    pub fn add_lower_bound(&mut self, string_var: TermId, length: i64) -> Result<()> {
        self.constraints
            .push(LengthConstraint::LowerBound(string_var, length));
        self.update_bound(string_var, LengthBound::at_least(length))?;
        Ok(())
    }

    /// Add an upper bound constraint
    pub fn add_upper_bound(&mut self, string_var: TermId, length: i64) -> Result<()> {
        if length < 0 {
            return Err(OxizError::Internal("negative upper bound".to_string()));
        }

        self.constraints
            .push(LengthConstraint::UpperBound(string_var, length));
        self.update_bound(string_var, LengthBound::at_most(length))?;
        Ok(())
    }

    /// Add a string operation
    pub fn add_operation(&mut self, op: StringOp) {
        self.operations.push(op);
    }

    /// Update a length bound
    fn update_bound(&mut self, var: TermId, new_bound: LengthBound) -> Result<()> {
        let current = self.bounds.entry(var).or_insert(LengthBound::unbounded());

        if let Some(intersection) = current.intersect(&new_bound) {
            if !intersection.is_satisfiable() {
                self.conflict = Some(vec![var]);
                return Err(OxizError::Internal(
                    "unsatisfiable length bound".to_string(),
                ));
            }

            let changed = *current != intersection;
            *current = intersection;

            if changed {
                self.propagation_queue.push_back(var);
            }
        } else {
            self.conflict = Some(vec![var]);
            return Err(OxizError::Internal("conflicting length bounds".to_string()));
        }

        Ok(())
    }

    /// Propagate length constraints
    pub fn propagate(&mut self) -> Result<Vec<(TermId, LengthBound)>> {
        let mut deductions = Vec::new();

        // If queue is empty, add all variables to ensure at least one propagation pass
        if self.propagation_queue.is_empty() {
            for &var in self.bounds.keys() {
                self.propagation_queue.push_back(var);
            }
        }

        while let Some(var) = self.propagation_queue.pop_front() {
            // Propagate through operations
            for op in &self.operations.clone() {
                if let Some(new_deductions) = self.propagate_operation(op, var)? {
                    deductions.extend(new_deductions);
                }
            }
        }

        Ok(deductions)
    }

    /// Propagate through a single operation
    fn propagate_operation(
        &mut self,
        op: &StringOp,
        changed_var: TermId,
    ) -> Result<Option<Vec<(TermId, LengthBound)>>> {
        let mut deductions = Vec::new();

        match op {
            StringOp::Concat { result, operands } => {
                // len(result) = len(op1) + len(op2) + ... + len(opn)
                if operands.contains(&changed_var) || *result == changed_var {
                    // Forward: compute result length from operands
                    let mut total_lower = 0i64;
                    let mut total_upper = Some(0i64);
                    let mut _all_exact = true;

                    for &op_var in operands {
                        let bound = self.get_bound(op_var);
                        total_lower += bound.lower;
                        total_upper = match (total_upper, bound.upper) {
                            (Some(a), Some(b)) => Some(a + b),
                            _ => None,
                        };
                        if !bound.is_exact() {
                            _all_exact = false;
                        }
                    }

                    let result_bound = LengthBound {
                        lower: total_lower,
                        upper: total_upper,
                    };

                    if self.update_bound(*result, result_bound).is_ok() {
                        deductions.push((*result, result_bound));
                    }

                    // Backward: if result length is known, constrain operands
                    if let Some(result_bound) = self.bounds.get(result)
                        && let Some(exact_result) = result_bound.exact_value()
                    {
                        // If all operands except one have exact lengths, deduce the last one
                        let mut known_sum = 0i64;
                        let mut unknown_vars = Vec::new();

                        for &op_var in operands {
                            let bound = self.get_bound(op_var);
                            if let Some(exact) = bound.exact_value() {
                                known_sum += exact;
                            } else {
                                unknown_vars.push(op_var);
                            }
                        }

                        if unknown_vars.len() == 1 {
                            let remaining = exact_result - known_sum;
                            if remaining >= 0 {
                                let deduced_bound = LengthBound::exact(remaining);
                                if self.update_bound(unknown_vars[0], deduced_bound).is_ok() {
                                    deductions.push((unknown_vars[0], deduced_bound));
                                }
                            }
                        }
                    }
                }
            }

            StringOp::Substr {
                result,
                source,
                offset: _,
                length: _,
            } => {
                // len(result) = length (if length is concrete)
                // len(result) <= len(source)
                // len(result) <= len(source) - offset (if offset is concrete)
                if *source == changed_var || *result == changed_var {
                    let source_bound = self.get_bound(*source);

                    // Result length is at most source length
                    let result_bound = LengthBound::at_most(source_bound.upper.unwrap_or(i64::MAX));
                    if self.update_bound(*result, result_bound).is_ok() {
                        deductions.push((*result, result_bound));
                    }

                    // If result length is known, source must be at least that long
                    if let Some(result_bound) = self.bounds.get(result)
                        && result_bound.lower > 0
                    {
                        let min_source = LengthBound::at_least(result_bound.lower);
                        if self.update_bound(*source, min_source).is_ok() {
                            deductions.push((*source, min_source));
                        }
                    }
                }
            }

            StringOp::Replace {
                result,
                source,
                pattern,
                replacement,
            } => {
                // len(result) depends on how many times pattern occurs in source
                // Lower bound: len(result) >= len(source) - len(pattern) + len(replacement)
                // (if pattern occurs once)
                // Upper bound: len(result) <= len(source) (if pattern doesn't occur)
                if *source == changed_var || *result == changed_var {
                    let source_bound = self.get_bound(*source);
                    let pattern_bound = self.get_bound(*pattern);
                    let replacement_bound = self.get_bound(*replacement);

                    // Conservative: result is at most source length + possible expansions
                    // At minimum, if no replacement happens, result = source
                    // At maximum, if pattern is replaced, result changes by (len(repl) - len(pat))

                    // If pattern doesn't occur, len(result) = len(source)
                    // If pattern occurs once, len(result) = len(source) - len(pattern) + len(replacement)

                    // Conservative bounds:
                    let min_result = if let (Some(p_len), Some(r_len)) =
                        (pattern_bound.exact_value(), replacement_bound.exact_value())
                    {
                        if p_len <= source_bound.lower {
                            source_bound.lower - p_len + r_len
                        } else {
                            source_bound.lower
                        }
                    } else {
                        0
                    };

                    let max_result = source_bound.upper.map(|s_len| {
                        if let Some(p_len) = pattern_bound.exact_value() {
                            if let Some(r_len) = replacement_bound.exact_value() {
                                if r_len > p_len {
                                    // Replacement is longer - worst case is all chars are pattern
                                    s_len / p_len * (r_len - p_len) + s_len
                                } else {
                                    s_len
                                }
                            } else {
                                s_len * 2 // Conservative upper bound
                            }
                        } else {
                            s_len * 2 // Conservative
                        }
                    });

                    let result_bound = LengthBound {
                        lower: min_result.max(0),
                        upper: max_result,
                    };

                    if self.update_bound(*result, result_bound).is_ok() {
                        deductions.push((*result, result_bound));
                    }
                }
            }

            StringOp::At {
                result,
                source: _,
                position: _,
            } => {
                // len(result) = 1 (character at position is always length 1)
                // Always propagate this constraint since result length is always 1
                let result_bound = LengthBound::exact(1);
                if self.update_bound(*result, result_bound).is_ok() {
                    deductions.push((*result, result_bound));
                }
            }

            StringOp::Contains { source, pattern } => {
                // If source contains pattern, len(source) >= len(pattern)
                if *source == changed_var || *pattern == changed_var {
                    let pattern_bound = self.get_bound(*pattern);
                    let source_bound = self.get_bound(*source);

                    if pattern_bound.lower > 0 {
                        let min_source = LengthBound::at_least(pattern_bound.lower);
                        if self.update_bound(*source, min_source).is_ok() {
                            deductions.push((*source, min_source));
                        }
                    }

                    // If source is shorter than pattern, contradiction
                    if let (Some(s_max), p_min) = (source_bound.upper, pattern_bound.lower)
                        && s_max < p_min
                    {
                        self.conflict = Some(vec![*source, *pattern]);
                        return Err(OxizError::Internal(
                            "source too short to contain pattern".to_string(),
                        ));
                    }
                }
            }

            StringOp::Prefix { pattern, source } => {
                // If pattern is prefix of source, len(source) >= len(pattern)
                if *source == changed_var || *pattern == changed_var {
                    let pattern_bound = self.get_bound(*pattern);
                    let source_bound = self.get_bound(*source);

                    let min_source = LengthBound::at_least(pattern_bound.lower);
                    if self.update_bound(*source, min_source).is_ok() {
                        deductions.push((*source, min_source));
                    }

                    // Check conflict
                    if let (Some(s_max), p_min) = (source_bound.upper, pattern_bound.lower)
                        && s_max < p_min
                    {
                        self.conflict = Some(vec![*source, *pattern]);
                        return Err(OxizError::Internal(
                            "source too short for prefix".to_string(),
                        ));
                    }
                }
            }

            StringOp::Suffix { pattern, source } => {
                // Same as prefix for length reasoning
                if *source == changed_var || *pattern == changed_var {
                    let pattern_bound = self.get_bound(*pattern);
                    let source_bound = self.get_bound(*source);

                    let min_source = LengthBound::at_least(pattern_bound.lower);
                    if self.update_bound(*source, min_source).is_ok() {
                        deductions.push((*source, min_source));
                    }

                    // Check conflict
                    if let (Some(s_max), p_min) = (source_bound.upper, pattern_bound.lower)
                        && s_max < p_min
                    {
                        self.conflict = Some(vec![*source, *pattern]);
                        return Err(OxizError::Internal(
                            "source too short for suffix".to_string(),
                        ));
                    }
                }
            }

            StringOp::IndexOf {
                result: _,
                source,
                pattern,
                offset: _,
            } => {
                // result is the position of pattern in source (or -1 if not found)
                // If found, 0 <= result < len(source) - len(pattern) + 1
                // This doesn't directly constrain lengths much, but:
                // - source must be at least len(pattern) if result >= 0
                if *source == changed_var || *pattern == changed_var {
                    let pattern_bound = self.get_bound(*pattern);
                    let _source_bound = self.get_bound(*source);

                    // If indexOf succeeds, source >= pattern
                    let min_source = LengthBound::at_least(pattern_bound.lower);
                    if self.update_bound(*source, min_source).is_ok() {
                        deductions.push((*source, min_source));
                    }
                }
            }
        }

        if deductions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(deductions))
        }
    }

    /// Get the bound for a variable
    fn get_bound(&self, var: TermId) -> LengthBound {
        self.bounds
            .get(&var)
            .cloned()
            .unwrap_or_else(LengthBound::unbounded)
    }

    /// Check for conflicts
    pub fn check(&mut self) -> Result<()> {
        if self.conflict.is_some() {
            return Err(OxizError::Internal(
                "length constraint conflict".to_string(),
            ));
        }

        // Check all bounds are satisfiable
        for (&var, bound) in &self.bounds {
            if !bound.is_satisfiable() {
                self.conflict = Some(vec![var]);
                return Err(OxizError::Internal(
                    "unsatisfiable length bound".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get the conflict clause
    pub fn get_conflict(&self) -> Option<&[TermId]> {
        self.conflict.as_deref()
    }

    /// Get length bound for a variable
    pub fn get_length_bound(&self, var: TermId) -> Option<&LengthBound> {
        self.bounds.get(&var)
    }

    /// Generate arithmetic constraints for a length abstraction
    pub fn abstract_to_arithmetic(&self) -> Vec<ArithmeticConstraint> {
        let mut constraints = Vec::new();

        // For each length variable, create corresponding arithmetic constraints
        for (&string_var, lv) in &self.length_vars {
            if let Some(bound) = self.bounds.get(&string_var) {
                if bound.is_exact() {
                    // Exact constraint (lower == upper)
                    constraints.push(ArithmeticConstraint::Exact {
                        var: lv.length_var,
                        value: bound.lower,
                    });
                } else {
                    // Range constraints
                    constraints.push(ArithmeticConstraint::LowerBound {
                        var: lv.length_var,
                        value: bound.lower,
                    });

                    // len(s) <= upper (if bounded)
                    if let Some(upper) = bound.upper {
                        constraints.push(ArithmeticConstraint::UpperBound {
                            var: lv.length_var,
                            value: upper,
                        });
                    }
                }
            }
        }

        // For each operation, create arithmetic constraints
        for op in &self.operations {
            match op {
                StringOp::Concat { result, operands } => {
                    // len(result) = sum(len(operands))
                    if let Some(result_lv) = self.length_vars.get(result) {
                        let mut lhs = vec![(result_lv.length_var, 1)];
                        for &operand in operands {
                            if let Some(op_lv) = self.length_vars.get(&operand) {
                                lhs.push((op_lv.length_var, -1));
                            }
                        }
                        constraints.push(ArithmeticConstraint::Linear { lhs, rhs: 0 });
                    }
                }

                StringOp::Substr { result, source, .. } => {
                    // len(result) <= len(source)
                    if let (Some(result_lv), Some(source_lv)) =
                        (self.length_vars.get(result), self.length_vars.get(source))
                    {
                        constraints.push(ArithmeticConstraint::LessOrEqual {
                            lhs: result_lv.length_var,
                            rhs: source_lv.length_var,
                        });
                    }
                }

                StringOp::At {
                    result,
                    source: _,
                    position: _,
                } => {
                    // len(result) = 1
                    if let Some(result_lv) = self.length_vars.get(result) {
                        constraints.push(ArithmeticConstraint::Exact {
                            var: result_lv.length_var,
                            value: 1,
                        });
                    }
                }

                StringOp::Contains { source, pattern }
                | StringOp::Prefix { pattern, source }
                | StringOp::Suffix { pattern, source } => {
                    // len(source) >= len(pattern)
                    if let (Some(source_lv), Some(pattern_lv)) =
                        (self.length_vars.get(source), self.length_vars.get(pattern))
                    {
                        constraints.push(ArithmeticConstraint::GreaterOrEqual {
                            lhs: source_lv.length_var,
                            rhs: pattern_lv.length_var,
                        });
                    }
                }

                _ => {}
            }
        }

        constraints
    }

    /// Statistics
    pub fn stats(&self) -> LengthSolverStats {
        LengthSolverStats {
            num_length_vars: self.length_vars.len(),
            num_operations: self.operations.len(),
            num_constraints: self.constraints.len(),
            num_exact_bounds: self.bounds.values().filter(|b| b.is_exact()).count(),
        }
    }
}

impl Default for LengthSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Arithmetic constraint for length abstraction
#[derive(Debug, Clone)]
pub enum ArithmeticConstraint {
    /// var = value
    Exact { var: TermId, value: i64 },
    /// var >= value
    LowerBound { var: TermId, value: i64 },
    /// var <= value
    UpperBound { var: TermId, value: i64 },
    /// var1 <= var2
    LessOrEqual { lhs: TermId, rhs: TermId },
    /// var1 >= var2
    GreaterOrEqual { lhs: TermId, rhs: TermId },
    /// Linear combination: sum(coeff * var) = rhs
    Linear { lhs: Vec<(TermId, i64)>, rhs: i64 },
}

/// Statistics for length solver
#[derive(Debug, Clone, Copy)]
pub struct LengthSolverStats {
    /// Number of length variables
    pub num_length_vars: usize,
    /// Number of operations
    pub num_operations: usize,
    /// Number of constraints
    pub num_constraints: usize,
    /// Number of exact bounds deduced
    pub num_exact_bounds: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_bound_exact() {
        let bound = LengthBound::exact(5);
        assert!(bound.is_exact());
        assert_eq!(bound.exact_value(), Some(5));
        assert!(bound.is_satisfiable());
    }

    #[test]
    fn test_length_bound_range() {
        let bound = LengthBound::range(2, Some(10));
        assert!(!bound.is_exact());
        assert_eq!(bound.exact_value(), None);
        assert!(bound.is_satisfiable());
    }

    #[test]
    fn test_length_bound_intersection() {
        let b1 = LengthBound::range(0, Some(10));
        let b2 = LengthBound::range(5, Some(15));
        let intersection = b1.intersect(&b2).expect("should intersect");
        assert_eq!(intersection.lower, 5);
        assert_eq!(intersection.upper, Some(10));
    }

    #[test]
    fn test_length_bound_empty_intersection() {
        let b1 = LengthBound::range(0, Some(5));
        let b2 = LengthBound::range(10, Some(15));
        assert!(b1.intersect(&b2).is_none());
    }

    #[test]
    fn test_length_bound_negative_unsatisfiable() {
        let bound = LengthBound::exact(-1);
        assert!(!bound.is_satisfiable());
    }

    #[test]
    fn test_add_exact_length() {
        let mut solver = LengthSolver::new();
        let var = TermId(0);
        solver.add_length_var(var, TermId(1));

        assert!(solver.add_exact_length(var, 5).is_ok());
        let bound = solver.get_length_bound(var).expect("should exist");
        assert_eq!(bound.exact_value(), Some(5));
    }

    #[test]
    fn test_add_negative_length_fails() {
        let mut solver = LengthSolver::new();
        let var = TermId(0);
        solver.add_length_var(var, TermId(1));

        assert!(solver.add_exact_length(var, -1).is_err());
    }

    #[test]
    fn test_conflicting_bounds() {
        let mut solver = LengthSolver::new();
        let var = TermId(0);
        solver.add_length_var(var, TermId(1));

        assert!(solver.add_exact_length(var, 5).is_ok());
        assert!(solver.add_exact_length(var, 10).is_err());
    }

    #[test]
    fn test_concat_forward_propagation() {
        let mut solver = LengthSolver::new();
        let s1 = TermId(0);
        let s2 = TermId(1);
        let result = TermId(2);

        solver.add_length_var(s1, TermId(10));
        solver.add_length_var(s2, TermId(11));
        solver.add_length_var(result, TermId(12));

        solver.add_exact_length(s1, 3).unwrap();
        solver.add_exact_length(s2, 4).unwrap();

        solver.add_operation(StringOp::Concat {
            result,
            operands: vec![s1, s2],
        });

        let deductions = solver.propagate().unwrap();
        assert!(!deductions.is_empty());

        let result_bound = solver.get_length_bound(result).expect("should exist");
        assert_eq!(result_bound.exact_value(), Some(7));
    }

    #[test]
    fn test_concat_backward_propagation() {
        let mut solver = LengthSolver::new();
        let s1 = TermId(0);
        let s2 = TermId(1);
        let result = TermId(2);

        solver.add_length_var(s1, TermId(10));
        solver.add_length_var(s2, TermId(11));
        solver.add_length_var(result, TermId(12));

        solver.add_exact_length(s1, 3).unwrap();
        solver.add_exact_length(result, 7).unwrap();

        solver.add_operation(StringOp::Concat {
            result,
            operands: vec![s1, s2],
        });

        let _deductions = solver.propagate().unwrap();

        let s2_bound = solver.get_length_bound(s2).expect("should exist");
        assert_eq!(s2_bound.exact_value(), Some(4));
    }

    #[test]
    fn test_substr_propagation() {
        let mut solver = LengthSolver::new();
        let source = TermId(0);
        let result = TermId(1);
        let offset = TermId(2);
        let length = TermId(3);

        solver.add_length_var(source, TermId(10));
        solver.add_length_var(result, TermId(11));

        solver.add_exact_length(source, 10).unwrap();

        solver.add_operation(StringOp::Substr {
            result,
            source,
            offset,
            length,
        });

        let _ = solver.propagate();

        let result_bound = solver.get_length_bound(result).expect("should exist");
        assert!(result_bound.upper.is_some());
        assert!(result_bound.upper.unwrap() <= 10);
    }

    #[test]
    fn test_at_propagation() {
        let mut solver = LengthSolver::new();
        let source = TermId(0);
        let result = TermId(1);
        let position = TermId(2);

        solver.add_length_var(source, TermId(10));
        solver.add_length_var(result, TermId(11));

        solver.add_operation(StringOp::At {
            result,
            source,
            position,
        });

        let _ = solver.propagate();

        let result_bound = solver.get_length_bound(result).expect("should exist");
        assert_eq!(result_bound.exact_value(), Some(1));
    }

    #[test]
    fn test_contains_length_constraint() {
        let mut solver = LengthSolver::new();
        let source = TermId(0);
        let pattern = TermId(1);

        solver.add_length_var(source, TermId(10));
        solver.add_length_var(pattern, TermId(11));

        solver.add_exact_length(pattern, 5).unwrap();

        solver.add_operation(StringOp::Contains { source, pattern });

        let _ = solver.propagate();

        let source_bound = solver.get_length_bound(source).expect("should exist");
        assert!(source_bound.lower >= 5);
    }

    #[test]
    fn test_contains_conflict() {
        let mut solver = LengthSolver::new();
        let source = TermId(0);
        let pattern = TermId(1);

        solver.add_length_var(source, TermId(10));
        solver.add_length_var(pattern, TermId(11));

        solver.add_exact_length(source, 3).unwrap();
        solver.add_exact_length(pattern, 5).unwrap();

        solver.add_operation(StringOp::Contains { source, pattern });

        assert!(solver.propagate().is_err());
        assert!(solver.get_conflict().is_some());
    }

    #[test]
    fn test_prefix_propagation() {
        let mut solver = LengthSolver::new();
        let pattern = TermId(0);
        let source = TermId(1);

        solver.add_length_var(pattern, TermId(10));
        solver.add_length_var(source, TermId(11));

        solver.add_exact_length(pattern, 4).unwrap();

        solver.add_operation(StringOp::Prefix { pattern, source });

        let _ = solver.propagate();

        let source_bound = solver.get_length_bound(source).expect("should exist");
        assert!(source_bound.lower >= 4);
    }

    #[test]
    fn test_arithmetic_abstraction() {
        let mut solver = LengthSolver::new();
        let s1 = TermId(0);
        let s2 = TermId(1);
        let result = TermId(2);

        solver.add_length_var(s1, TermId(10));
        solver.add_length_var(s2, TermId(11));
        solver.add_length_var(result, TermId(12));

        solver.add_exact_length(s1, 5).unwrap();

        solver.add_operation(StringOp::Concat {
            result,
            operands: vec![s1, s2],
        });

        let arith_constraints = solver.abstract_to_arithmetic();
        assert!(!arith_constraints.is_empty());

        // Should have exact constraint for s1
        let has_exact = arith_constraints.iter().any(|c| {
            matches!(c, ArithmeticConstraint::Exact { var, value } if *var == TermId(10) && *value == 5)
        });
        assert!(has_exact);

        // Should have linear constraint for concat
        let has_linear = arith_constraints
            .iter()
            .any(|c| matches!(c, ArithmeticConstraint::Linear { .. }));
        assert!(has_linear);
    }

    #[test]
    fn test_multiple_concat() {
        let mut solver = LengthSolver::new();
        let s1 = TermId(0);
        let s2 = TermId(1);
        let s3 = TermId(2);
        let result = TermId(3);

        solver.add_length_var(s1, TermId(10));
        solver.add_length_var(s2, TermId(11));
        solver.add_length_var(s3, TermId(12));
        solver.add_length_var(result, TermId(13));

        solver.add_exact_length(s1, 2).unwrap();
        solver.add_exact_length(s2, 3).unwrap();
        solver.add_exact_length(s3, 4).unwrap();

        solver.add_operation(StringOp::Concat {
            result,
            operands: vec![s1, s2, s3],
        });

        let _ = solver.propagate().unwrap();

        let result_bound = solver.get_length_bound(result).expect("should exist");
        assert_eq!(result_bound.exact_value(), Some(9));
    }

    #[test]
    fn test_stats() {
        let mut solver = LengthSolver::new();
        let s1 = TermId(0);
        let s2 = TermId(1);

        solver.add_length_var(s1, TermId(10));
        solver.add_length_var(s2, TermId(11));
        solver.add_exact_length(s1, 5).unwrap();

        let stats = solver.stats();
        assert_eq!(stats.num_length_vars, 2);
        assert_eq!(stats.num_exact_bounds, 1);
    }

    #[test]
    fn test_bound_tighten() {
        let mut b1 = LengthBound::range(0, Some(10));
        let b2 = LengthBound::range(5, Some(8));
        assert!(b1.tighten(&b2));
        assert_eq!(b1.lower, 5);
        assert_eq!(b1.upper, Some(8));
    }

    #[test]
    fn test_unbounded_intersection() {
        let b1 = LengthBound::unbounded();
        let b2 = LengthBound::range(5, Some(10));
        let intersection = b1.intersect(&b2).expect("should intersect");
        assert_eq!(intersection.lower, 5);
        assert_eq!(intersection.upper, Some(10));
    }
}
