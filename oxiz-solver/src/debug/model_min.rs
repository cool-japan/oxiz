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
        self.binary_search_minimal(all, &checker, &mut essential_indices, &mut stats);

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

    /// Binary search for the minimal satisfying set, driven by an explicit
    /// heap stack.
    ///
    /// This used to be a two-call recursion (`Both halves are needed` below
    /// descends into the left half, mutates the essential set, then descends
    /// into the right half). Its return type is `()`, so it has no channel
    /// through which a depth cap could report truncation — a cap could only
    /// silently return a wrong `ModelMinResult`. The recursion is therefore
    /// converted to an explicit stack that preserves the original traversal
    /// order, the original check counts, and the original mutations of
    /// `essential` exactly; the only behavioural difference is that the native
    /// call stack is no longer the depth bound.
    fn binary_search_minimal<F>(
        &self,
        root: Vec<usize>,
        checker: &F,
        essential: &mut FxHashSet<usize>,
        stats: &mut MinStats,
    ) where
        F: Fn(&[(u32, String)]) -> bool,
    {
        let mut stack: Vec<MinFrame> = vec![MinFrame::Descend(root)];

        while let Some(frame) = stack.pop() {
            match frame {
                MinFrame::ResumeRight { left, right } => {
                    // Continuation of the `both halves are needed` case: the
                    // left descent has finished, so hand the right half back
                    // its own turn with the left half pinned as essential.
                    for idx in &right {
                        essential.remove(idx);
                    }
                    for idx in &left {
                        essential.insert(*idx);
                    }
                    stack.push(MinFrame::Descend(right));
                }
                MinFrame::Descend(indices) => {
                    if indices.is_empty() {
                        continue;
                    }

                    if let [idx] = indices[..] {
                        // Single element: check if it is essential.
                        let without = self.subset_where(|i| essential.contains(&i) && i != idx);

                        stats.checks_performed += 1;
                        stats.removals_attempted += 1;

                        if checker(&without) {
                            stats.successful_removals += 1;
                        } else {
                            essential.insert(idx);
                        }
                        continue;
                    }

                    if stats.checks_performed >= self.max_checks {
                        // Budget exceeded: mark all as essential.
                        essential.extend(indices.iter().copied());
                        continue;
                    }

                    let mid = indices.len() / 2;
                    let right: Vec<usize> = indices[mid..].to_vec();
                    let mut left = indices;
                    left.truncate(mid);

                    // Try with only the left half + current essential.
                    let left_set: FxHashSet<usize> = left.iter().copied().collect();
                    let left_subset =
                        self.subset_where(|i| left_set.contains(&i) || essential.contains(&i));

                    stats.checks_performed += 1;
                    if checker(&left_subset) {
                        // Left half alone is sufficient: right half is optional.
                        stats.successful_removals += right.len() as u64;
                        // Descend into left to minimize further.
                        stack.push(MinFrame::Descend(left));
                        continue;
                    }

                    // Try right half alone.
                    let right_set: FxHashSet<usize> = right.iter().copied().collect();
                    let right_subset =
                        self.subset_where(|i| right_set.contains(&i) || essential.contains(&i));

                    stats.checks_performed += 1;
                    if checker(&right_subset) {
                        // Right half alone is sufficient: left half is optional.
                        stats.successful_removals += left.len() as u64;
                        stack.push(MinFrame::Descend(right));
                        continue;
                    }

                    // Both halves are needed: descend into each.
                    // First add all of right to essential, then minimize left;
                    // the resume frame then swaps the roles and takes right.
                    essential.extend(right.iter().copied());
                    stack.push(MinFrame::ResumeRight {
                        left: left.clone(),
                        right,
                    });
                    stack.push(MinFrame::Descend(left));
                }
            }
        }
    }

    /// Project the assignments whose index satisfies `keep` into the
    /// `(var_id, value)` pair list the checker expects.
    fn subset_where(&self, keep: impl Fn(usize) -> bool) -> Vec<(u32, String)> {
        self.assignments
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(_, a)| (a.var_id, a.value.clone()))
            .collect()
    }
}

/// One pending step of the iterative [`ModelMinimizer::binary_search_minimal`]
/// driver.
///
/// The recursion this replaces had a resume point in the middle of one arm
/// (descend left, mutate the essential set, descend right), so that arm needs
/// its own frame rather than a plain worklist entry.
#[derive(Debug)]
enum MinFrame {
    /// Process this index range as the original recursive call would.
    Descend(Vec<usize>),
    /// Continuation for the `both halves are needed` arm: unpin `right`,
    /// pin `left`, then process `right`.
    ResumeRight {
        /// Left half, pinned as essential before the right half is processed.
        left: Vec<usize>,
        /// Right half, processed after the swap.
        right: Vec<usize>,
    },
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
        let result = minimizer.minimize_linear(|assignments| assignments.len() >= 2);

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
        let result =
            minimizer.minimize_linear(|assignments| assignments.iter().any(|(id, _)| *id == 1));

        assert_eq!(result.essential_vars.len(), 1);
        assert_eq!(result.essential_vars[0].var_id, 1);
        assert_eq!(result.optional_vars.len(), 2);
    }

    #[test]
    fn test_binary_minimization() {
        let mut minimizer = ModelMinimizer::new();
        for i in 0..8 {
            minimizer.add_assignment(make_assignment(i, &format!("v{}", i), "true"));
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

    /// Semantic pin for the recursive -> iterative conversion of
    /// `binary_search_minimal`: the *exact* essential set, optional set and
    /// check counts produced by the original recursion on a hand-checked
    /// input.
    #[test]
    fn test_binary_minimization_exact_semantics_pin() {
        let mut minimizer = ModelMinimizer::new();
        for i in 0..8 {
            minimizer.add_assignment(make_assignment(i, &format!("v{}", i), "true"));
        }

        // Every variable is required: this drives the `both halves are needed`
        // arm at every internal node, i.e. the one arm that had a resume point
        // in the middle of the old recursion.
        let result = minimizer.minimize_binary(|assignments| assignments.len() == 8);

        let mut essential_ids: Vec<u32> = result.essential_vars.iter().map(|v| v.var_id).collect();
        essential_ids.sort_unstable();
        assert_eq!(essential_ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(result.optional_vars.is_empty());
        assert_eq!(result.stats.checks_performed, 22);
        assert_eq!(result.stats.removals_attempted, 8);
        assert_eq!(result.stats.successful_removals, 0);
    }

    /// Semantic pin for the mixed case (one half sufficient at the root).
    #[test]
    fn test_binary_minimization_left_half_sufficient_pin() {
        let mut minimizer = ModelMinimizer::new();
        for i in 0..8 {
            minimizer.add_assignment(make_assignment(i, &format!("v{}", i), "true"));
        }

        // Only var 0 matters, and it sits in the left half at every level.
        let result = minimizer.minimize_binary(|a| a.iter().any(|(id, _)| *id == 0));

        let essential_ids: Vec<u32> = result.essential_vars.iter().map(|v| v.var_id).collect();
        assert_eq!(essential_ids, vec![0]);
        assert_eq!(result.optional_vars.len(), 7);
        assert_eq!(result.stats.checks_performed, 4);
        assert_eq!(result.stats.removals_attempted, 1);
        assert_eq!(result.stats.successful_removals, 4 + 2 + 1);
    }

    /// The converted walk must return rather than overflow the native stack.
    ///
    /// Honest scaling note: unlike a term walk, this driver's *nesting depth*
    /// is logarithmic in the input (each frame halves its index range), so a
    /// 50 000-level nesting is not constructible here at all — 2^50000
    /// assignments do not fit in memory. The deepest tree an input of `n`
    /// assignments can produce is `log2(n)` levels, and it is produced by
    /// forcing the two-way `both halves are needed` arm at every node, which
    /// is exactly what this test does. The input is sized so the O(n^2) subset
    /// rebuilding stays fast, and it runs on a deliberately small 1 MiB stack
    /// so that a regression back to recursion would have to fit its whole
    /// tree there.
    ///
    /// The stack stays at 1 MiB, unlike the crate's other deep-nesting tests
    /// that were scaled down to 128 KiB together with their depths: this one
    /// nests only `log2(1024) = 10` levels, so there is no stack/depth ratio
    /// to preserve and nothing quadratic to shrink.
    #[cfg(feature = "std")]
    #[test]
    fn test_binary_minimization_returns_on_small_stack() {
        const N: u32 = 1024;
        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — nests
        // only log2(1024) = 10 levels (sub-10,000-depth), so 1 MiB is
        // intentionally generous. See TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let mut minimizer = ModelMinimizer::new();
                minimizer.set_max_checks(u64::MAX);
                for i in 0..N {
                    minimizer.add_assignment(make_assignment(i, &format!("v{}", i), "true"));
                }
                // Every variable required => the two-way arm fires at every
                // internal node, giving the deepest tree this input admits.
                let result = minimizer.minimize_binary(|a| a.len() >= N as usize);
                (result.total_vars(), result.essential_vars.len())
            })
            .expect("spawn minimization thread");

        let (total, essential) = handle.join().expect("minimization thread panicked");
        assert_eq!(total, N as usize);
        assert_eq!(essential, N as usize);
    }

    #[test]
    fn test_max_checks_limit() {
        let mut minimizer = ModelMinimizer::new();
        minimizer.set_max_checks(2);

        for i in 0..10 {
            minimizer.add_assignment(make_assignment(i, &format!("v{}", i), "true"));
        }

        let result = minimizer.minimize_linear(|_| true);

        // Should stop after max_checks.
        assert!(result.stats.checks_performed <= 2);
        // Remaining should be marked essential (budget exceeded).
        assert!(result.essential_vars.len() + result.optional_vars.len() == 10);
    }
}
