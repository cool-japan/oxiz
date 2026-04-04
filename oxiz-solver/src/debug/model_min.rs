//! Model Minimization for Debugging.
//!
//! Given a satisfying model, finds a minimal model by identifying which
//! variable assignments are essential (required for satisfiability) and
//! which are optional (can be removed without breaking satisfiability).
//!
//! ## Strategies
//!
//! - **Linear scan**: Remove assignments one by one, check if still satisfying.
//! - **Binary search**: Use binary partitioning to find minimal set faster.
//!
//! ## References
//!
//! - Z3's `smt/smt_model_generator.cpp`

#[allow(unused_imports)]
use crate::prelude::*;

/// A variable assignment in the model.
#[derive(Debug, Clone)]
pub struct ModelAssignment {
    /// Variable identifier.
    pub var_id: u32,
    /// Variable name (for display).
    pub name: String,
    /// The value assigned (as a string representation).
    pub value: String,
    /// Whether this is a boolean variable.
    pub is_bool: bool,
}

/// Result of model minimization.
#[derive(Debug, Clone)]
pub struct ModelMinResult {
    /// Variables that are essential (must be assigned for satisfiability).
    pub essential_vars: Vec<ModelAssignment>,
    /// Variables that are optional (can be removed).
    pub optional_vars: Vec<ModelAssignment>,
    /// Statistics about the minimization process.
    pub stats: MinStats,
}

impl ModelMinResult {
    /// Get the total number of variables in the original model.
    pub fn total_vars(&self) -> usize {
        self.essential_vars.len() + self.optional_vars.len()
    }

    /// Get the reduction ratio (0.0 = no reduction, 1.0 = all optional).
    pub fn reduction_ratio(&self) -> f64 {
        let total = self.total_vars();
        if total == 0 {
            return 0.0;
        }
        self.optional_vars.len() as f64 / total as f64
    }

    /// Format the result as human-readable text.
    pub fn format(&self) -> String {
        let mut out = String::new();

        out.push_str("=== Model Minimization Result ===\n\n");
        out.push_str(&format!(
            "Original model size: {} variables\n",
            self.total_vars()
        ));
        out.push_str(&format!(
            "Essential variables:  {}\n",
            self.essential_vars.len()
        ));
        out.push_str(&format!(
            "Optional variables:   {}\n",
            self.optional_vars.len()
        ));
        out.push_str(&format!(
            "Reduction:            {:.1}%\n\n",
            self.reduction_ratio() * 100.0
        ));

        out.push_str(&format!(
            "Checks performed:     {}\n",
            self.stats.checks_performed
        ));
        out.push_str(&format!(
            "Removals attempted:   {}\n",
            self.stats.removals_attempted
        ));
        out.push_str(&format!(
            "Successful removals:  {}\n\n",
            self.stats.successful_removals
        ));

        if !self.essential_vars.is_empty() {
            out.push_str("Essential variables:\n");
            for v in &self.essential_vars {
                out.push_str(&format!("  {} = {}\n", v.name, v.value));
            }
            out.push('\n');
        }

        if !self.optional_vars.is_empty() {
            out.push_str("Optional variables:\n");
            for v in &self.optional_vars {
                out.push_str(&format!("  {} = {} (removable)\n", v.name, v.value));
            }
        }

        out
    }
}

/// Statistics for the minimization process.
#[derive(Debug, Clone, Default)]
pub struct MinStats {
    /// Number of satisfiability checks performed.
    pub checks_performed: u64,
    /// Number of removal attempts.
    pub removals_attempted: u64,
    /// Number of successful removals.
    pub successful_removals: u64,
}

/// A checker function that determines if a subset of assignments is still satisfying.
///
/// Takes a set of (var_id, value) pairs and returns true if satisfying.
pub type SatisfactionChecker = Box<dyn Fn(&[(u32, String)]) -> bool>;

/// Model minimizer.
#[derive(Debug)]
pub struct ModelMinimizer {
    /// Original model assignments.
    assignments: Vec<ModelAssignment>,
    /// Maximum number of checks before giving up.
    max_checks: u64,
}

impl ModelMinimizer {
    /// Create a new model minimizer.
    pub fn new() -> Self {
        Self {
            assignments: Vec::new(),
            max_checks: 10_000,
        }
    }

    /// Set the maximum number of satisfiability checks.
    pub fn set_max_checks(&mut self, max: u64) {
        self.max_checks = max;
    }

    /// Add assignments from the model.
    pub fn add_assignment(&mut self, assignment: ModelAssignment) {
        self.assignments.push(assignment);
    }

    /// Add multiple assignments.
    pub fn add_assignments(&mut self, assignments: impl IntoIterator<Item = ModelAssignment>) {
        self.assignments.extend(assignments);
    }

    /// Clear all assignments.
    pub fn clear(&mut self) {
        self.assignments.clear();
    }

    /// Get the number of assignments.
    pub fn num_assignments(&self) -> usize {
        self.assignments.len()
    }

    /// Minimize the model using linear scan.
    ///
    /// For each assignment, try removing it and check if the remaining
    /// assignments still satisfy the formula. If so, mark it as optional.
    ///
    /// The `checker` function takes a list of (var_id, value) pairs and
    /// returns true if they form a satisfying assignment.
    pub fn minimize_linear<F>(&self, checker: F) -> ModelMinResult
    where
        F: Fn(&[(u32, String)]) -> bool,
    {
        let mut stats = MinStats::default();
        let mut essential = Vec::new();
        let mut optional = Vec::new();

        // Track which indices are still active.
        let mut active: Vec<bool> = vec![true; self.assignments.len()];

        for i in 0..self.assignments.len() {
            if stats.checks_performed >= self.max_checks {
                // Budget exceeded: mark remaining as essential.
                for j in i..self.assignments.len() {
                    essential.push(self.assignments[j].clone());
                }
                break;
            }

            // Try removing assignment i.
            active[i] = false;
            stats.removals_attempted += 1;

            let subset: Vec<(u32, String)> = self
                .assignments
                .iter()
                .enumerate()
                .filter(|(j, _)| active[*j])
                .map(|(_, a)| (a.var_id, a.value.clone()))
                .collect();

            stats.checks_performed += 1;
            if checker(&subset) {
                // Still satisfying without this assignment.
                stats.successful_removals += 1;
                optional.push(self.assignments[i].clone());
            } else {
                // Needed: restore it.
                active[i] = true;
                essential.push(self.assignments[i].clone());
            }
        }

        ModelMinResult {
            essential_vars: essential,
            optional_vars: optional,
            stats,
        }
    }

    /// Minimize the model using binary search partitioning.
    ///
    /// Splits the assignments in half and checks each half. If one half
    /// alone is satisfying, recurse into it. Otherwise, both are needed.
    /// This can be faster than linear scan for models with many optional vars.
    pub fn minimize_binary<F>(&self, checker: F) -> ModelMinResult
    where
        F: Fn(&[(u32, String)]) -> bool,
    {
        let mut stats = MinStats::default();
        let n = self.assignments.len();

        if n == 0 {
            return ModelMinResult {
                essential_vars: Vec::new(),
                optional_vars: Vec::new(),
                stats,
            };
        }

        // Start with all assignments.
        let all: Vec<usize> = (0..n).collect();
        let mut essential_indices: FxHashSet<usize> = FxHashSet::default();

        // Binary search for minimal set.
        self.binary_search_minimal(
            &all,
            &checker,
            &mut essential_indices,
            &mut stats,
        );

        let mut essential = Vec::new();
        let mut optional = Vec::new();

        for (i, assignment) in self.assignments.iter().enumerate() {
            if essential_indices.contains(&i) {
                essential.push(assignment.clone());
            } else {
                optional.push(assignment.clone());
            }
        }

        ModelMinResult {
            essential_vars: essential,
            optional_vars: optional,
            stats,
        }
    }

    /// Recursive binary search for minimal satisfying set.
    fn binary_search_minimal<F>(
        &self,
        indices: &[usize],
        checker: &F,
        essential: &mut FxHashSet<usize>,
        stats: &mut MinStats,
    ) where
        F: Fn(&[(u32, String)]) -> bool,
    {
        if indices.is_empty() {
            return;
        }

        if indices.len() == 1 {
            // Single element: check if it is essential.
            let idx = indices[0];
            let without: Vec<(u32, String)> = self
                .assignments
                .iter()
                .enumerate()
                .filter(|(i, _)| essential.contains(i) && *i != idx)
                .map(|(_, a)| (a.var_id, a.value.clone()))
                .collect();

            stats.checks_performed += 1;
            stats.removals_attempted += 1;

            if !checker(&without) {
                essential.insert(idx);
            } else {
                stats.successful_removals += 1;
            }
            return;
        }

        if stats.checks_performed >= self.max_checks {
            // Budget exceeded: mark all as essential.
            for &idx in indices {
                essential.insert(idx);
            }
            return;
        }

        let mid = indices.len() / 2;
        let left = &indices[..mid];
        let right = &indices[mid..];

        // Try with only the left half + current essential.
        let left_set: FxHashSet<usize> = left.iter().copied().collect();
        let left_subset: Vec<(u32, String)> = self
            .assignments
            .iter()
            .enumerate()
            .filter(|(i, _)| left_set.contains(i) || essential.contains(i))
            .map(|(_, a)| (a.var_id, a.value.clone()))
            .collect();

        stats.checks_performed += 1;
        if checker(&left_subset) {
            // Left half alone is sufficient: right half is optional.
            stats.successful_removals += right.len() as u64;
            // Recurse into left to minimize further.
            self.binary_search_minimal(left, checker, essential, stats);
        } else {
            // Try right half alone.
            let right_set: FxHashSet<usize> = right.iter().copied().collect();
            let right_subset: Vec<(u32, String)> = self
                .assignments
                .iter()
                .enumerate()
                .filter(|(i, _)| right_set.contains(i) || essential.contains(i))
                .map(|(_, a)| (a.var_id, a.value.clone()))
                .collect();

            stats.checks_performed += 1;
            if checker(&right_subset) {
                // Right half alone is sufficient: left half is optional.
                stats.successful_removals += left.len() as u64;
                self.binary_search_minimal(right, checker, essential, stats);
            } else {
                // Both halves are needed: recurse into each.
                // First add all of right to essential, then minimize left.
                for &idx in right {
                    essential.insert(idx);
                }
                self.binary_search_minimal(left, checker, essential, stats);

                // Now minimize right with left essential.
                let right_owned: Vec<usize> = right.to_vec();
                // Remove right from essential to re-check.
                for &idx in &right_owned {
                    essential.remove(&idx);
                }
                // Add left to essential.
                for &idx in left {
                    essential.insert(idx);
                }
                self.binary_search_minimal(&right_owned, checker, essential, stats);
            }
        }
    }
}

impl Default for ModelMinimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_assignment(id: u32, name: &str, value: &str) -> ModelAssignment {
        ModelAssignment {
            var_id: id,
            name: name.to_string(),
            value: value.to_string(),
            is_bool: true,
        }
    }

    #[test]
    fn test_empty_model() {
        let minimizer = ModelMinimizer::new();
        let result = minimizer.minimize_linear(|_| true);
        assert_eq!(result.essential_vars.len(), 0);
        assert_eq!(result.optional_vars.len(), 0);
        assert_eq!(result.total_vars(), 0);
    }

    #[test]
    fn test_all_essential() {
        let mut minimizer = ModelMinimizer::new();
        minimizer.add_assignment(make_assignment(1, "x", "true"));
        minimizer.add_assignment(make_assignment(2, "y", "false"));

        // Both are essential: removing either makes it unsatisfying.
        let result = minimizer.minimize_linear(|assignments| {
            assignments.len() >= 2
        });

        assert_eq!(result.essential_vars.len(), 2);
        assert_eq!(result.optional_vars.len(), 0);
    }

    #[test]
    fn test_all_optional() {
        let mut minimizer = ModelMinimizer::new();
        minimizer.add_assignment(make_assignment(1, "x", "true"));
        minimizer.add_assignment(make_assignment(2, "y", "false"));
        minimizer.add_assignment(make_assignment(3, "z", "true"));

        // Always satisfying regardless of assignments.
        let result = minimizer.minimize_linear(|_| true);

        assert_eq!(result.essential_vars.len(), 0);
        assert_eq!(result.optional_vars.len(), 3);
        assert!((result.reduction_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mixed_essential_optional() {
        let mut minimizer = ModelMinimizer::new();
        minimizer.add_assignment(make_assignment(1, "x", "true"));
        minimizer.add_assignment(make_assignment(2, "y", "false"));
        minimizer.add_assignment(make_assignment(3, "z", "true"));

        // Only var 1 is essential: need at least one assignment with var_id=1.
        let result = minimizer.minimize_linear(|assignments| {
            assignments.iter().any(|(id, _)| *id == 1)
        });

        assert_eq!(result.essential_vars.len(), 1);
        assert_eq!(result.essential_vars[0].var_id, 1);
        assert_eq!(result.optional_vars.len(), 2);
    }

    #[test]
    fn test_binary_minimization() {
        let mut minimizer = ModelMinimizer::new();
        for i in 0..8 {
            minimizer.add_assignment(make_assignment(
                i,
                &format!("v{}", i),
                "true",
            ));
        }

        // Only vars 0 and 4 are essential.
        let result = minimizer.minimize_binary(|assignments| {
            let has_0 = assignments.iter().any(|(id, _)| *id == 0);
            let has_4 = assignments.iter().any(|(id, _)| *id == 4);
            has_0 && has_4
        });

        // Essential set should contain vars 0 and 4.
        let essential_ids: Vec<u32> = result.essential_vars.iter().map(|v| v.var_id).collect();
        assert!(essential_ids.contains(&0), "var 0 should be essential");
        assert!(essential_ids.contains(&4), "var 4 should be essential");
    }

    #[test]
    fn test_model_min_result_format() {
        let result = ModelMinResult {
            essential_vars: vec![make_assignment(1, "x", "true")],
            optional_vars: vec![
                make_assignment(2, "y", "false"),
                make_assignment(3, "z", "true"),
            ],
            stats: MinStats {
                checks_performed: 5,
                removals_attempted: 3,
                successful_removals: 2,
            },
        };

        let text = result.format();
        assert!(text.contains("Model Minimization Result"));
        assert!(text.contains("Original model size: 3"));
        assert!(text.contains("Essential variables:  1"));
        assert!(text.contains("Optional variables:   2"));
        assert!(text.contains("x = true"));
        assert!(text.contains("y = false (removable)"));
    }

    #[test]
    fn test_max_checks_limit() {
        let mut minimizer = ModelMinimizer::new();
        minimizer.set_max_checks(2);

        for i in 0..10 {
            minimizer.add_assignment(make_assignment(
                i,
                &format!("v{}", i),
                "true",
            ));
        }

        let result = minimizer.minimize_linear(|_| true);

        // Should stop after max_checks.
        assert!(result.stats.checks_performed <= 2);
        // Remaining should be marked essential (budget exceeded).
        assert!(result.essential_vars.len() + result.optional_vars.len() == 10);
    }
}
