//! Cardinality constraint encoding
//!
//! This module implements efficient encoding of cardinality constraints into CNF.
//! Cardinality constraints express conditions like:
//! - At-most-k: at most k of the given literals can be true
//! - At-least-k: at least k of the given literals must be true
//! - Exactly-k: exactly k of the given literals must be true
//!
//! We use the Totalizer encoding which provides:
//! - Efficient incremental strengthening
//! - Good propagation
//! - Reasonable clause count

use crate::literal::{Lit, Var};
use crate::solver::Solver;
use smallvec::SmallVec;

/// Cardinality constraint encoder
pub struct CardinalityEncoder;

impl CardinalityEncoder {
    /// Encode an at-most-k constraint: sum(lits) <= k
    ///
    /// # Arguments
    ///
    /// * `solver` - The SAT solver
    /// * `lits` - The literals in the constraint
    /// * `k` - The upper bound
    ///
    /// Returns true if the constraint was successfully encoded
    pub fn encode_at_most_k(solver: &mut Solver, lits: &[Lit], k: usize) -> bool {
        if k >= lits.len() {
            return true; // Constraint is trivially satisfied
        }

        if k == 0 {
            // None of the literals can be true
            for &lit in lits {
                solver.add_clause([lit.negate()]);
            }
            return true;
        }

        if lits.len() <= 4 {
            // For small constraints, use direct encoding
            Self::encode_at_most_k_direct(solver, lits, k)
        } else {
            // For larger constraints, use totalizer encoding
            Self::encode_at_most_k_totalizer(solver, lits, k)
        }
    }

    /// Encode an at-least-k constraint: sum(lits) >= k
    ///
    /// Equivalent to: at-most-(n-k) of the negations
    pub fn encode_at_least_k(solver: &mut Solver, lits: &[Lit], k: usize) -> bool {
        if k == 0 {
            return true; // Trivially satisfied
        }

        if k > lits.len() {
            return false; // Unsatisfiable
        }

        if k == 1 {
            // At least one must be true - simple clause
            solver.add_clause(lits.iter().copied());
            return true;
        }

        // Transform to at-most constraint on negations
        let negated: Vec<Lit> = lits.iter().map(|&l| l.negate()).collect();
        Self::encode_at_most_k(solver, &negated, lits.len() - k)
    }

    /// Encode an exactly-k constraint: sum(lits) == k
    pub fn encode_exactly_k(solver: &mut Solver, lits: &[Lit], k: usize) -> bool {
        if k > lits.len() {
            return false;
        }

        // Combine at-most-k and at-least-k
        Self::encode_at_most_k(solver, lits, k) && Self::encode_at_least_k(solver, lits, k)
    }

    /// Direct encoding for small at-most-k constraints
    fn encode_at_most_k_direct(solver: &mut Solver, lits: &[Lit], k: usize) -> bool {
        // Generate all subsets of size k+1 and forbid them
        let n = lits.len();
        if k >= n {
            return true;
        }

        // Generate all combinations of k+1 literals
        Self::generate_combinations(lits, k + 1, &mut |combo| {
            // Add clause: at least one of these must be false
            let negated: SmallVec<[Lit; 8]> = combo.iter().map(|&&l| l.negate()).collect();
            solver.add_clause(negated.iter().copied());
        });

        true
    }

    /// Helper function to generate all k-combinations
    fn generate_combinations<F>(lits: &[Lit], k: usize, callback: &mut F)
    where
        F: FnMut(&[&Lit]),
    {
        let mut indices = vec![0; k];
        let n = lits.len();

        if k > n {
            return;
        }

        // Initialize first combination
        for (i, item) in indices.iter_mut().enumerate().take(k) {
            *item = i;
        }

        loop {
            // Call callback with current combination
            let combo: Vec<&Lit> = indices.iter().map(|&i| &lits[i]).collect();
            callback(&combo);

            // Find the rightmost index that can be incremented
            let mut i = k;
            loop {
                if i == 0 {
                    return; // No more combinations
                }
                i -= 1;
                if indices[i] < n - k + i {
                    break;
                }
            }

            // Increment this index and reset all following indices
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
        }
    }

    /// Totalizer encoding for at-most-k constraints
    ///
    /// The totalizer builds a tree of adder circuits that count the number
    /// of true literals. It introduces auxiliary variables representing
    /// "at least i literals are true" for various i.
    fn encode_at_most_k_totalizer(solver: &mut Solver, lits: &[Lit], k: usize) -> bool {
        if lits.is_empty() || k >= lits.len() {
            return true;
        }

        // Build totalizer tree
        let root_vars = Self::build_totalizer_tree(solver, lits, k);

        // Add constraint: at most k can be true
        // This means the (k+1)-th totalizer variable must be false
        if k < root_vars.len() {
            solver.add_clause([Lit::neg(root_vars[k])]);
        }

        true
    }

    /// Build the totalizer tree and return the root counting variables
    ///
    /// Returns a vector where output[i] represents "at least i+1 literals are true"
    fn build_totalizer_tree(solver: &mut Solver, lits: &[Lit], bound: usize) -> Vec<Var> {
        if lits.len() == 1 {
            // Leaf node: return the variable of the literal
            return vec![lits[0].var()];
        }

        // Split the literals into two halves
        let mid = lits.len() / 2;
        let left_lits = &lits[..mid];
        let right_lits = &lits[mid..];

        // Recursively build subtrees
        let left_vars = Self::build_totalizer_tree(solver, left_lits, bound);
        let right_vars = Self::build_totalizer_tree(solver, right_lits, bound);

        // Merge the two subtrees
        let max_count = (left_vars.len() + right_vars.len()).min(bound + 1);
        let mut output = Vec::with_capacity(max_count);

        for _ in 0..max_count {
            output.push(solver.new_var());
        }

        // Add clauses for the totalizer merge
        Self::add_totalizer_clauses(solver, &left_vars, &right_vars, &output);

        output
    }

    /// Add clauses for merging two totalizer trees
    ///
    /// Implements the totalizer merge operation:
    /// output[i] is true iff at least i+1 of the input literals are true
    fn add_totalizer_clauses(solver: &mut Solver, left: &[Var], right: &[Var], output: &[Var]) {
        // For each output position i (representing "at least i+1 are true")
        for (i, &out_var) in output.iter().enumerate() {
            let count = i + 1; // Number of true literals needed

            // If left has >= j and right has >= k where j+k >= count, then output[i] is true
            for j in 0..=left.len() {
                for k in 0..=right.len() {
                    if j + k >= count && j + k > 0 {
                        let mut clause = SmallVec::<[Lit; 4]>::new();

                        // If left[j-1] is true and right[k-1] is true, then output[i] is true
                        if j > 0 && j <= left.len() {
                            clause.push(Lit::neg(left[j - 1]));
                        }
                        if k > 0 && k <= right.len() {
                            clause.push(Lit::neg(right[k - 1]));
                        }

                        if !clause.is_empty() {
                            clause.push(Lit::pos(out_var));
                            solver.add_clause(clause.iter().copied());
                        }
                    }
                }
            }

            // Reverse direction: if output[i] is true, then sufficient input must be true
            // output[i] => (left[j] or right[k]) for all valid j, k where j+k+2 == count+1
            // This is captured by the contrapositive of the above clauses
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolverResult;

    #[test]
    fn test_at_most_0() {
        let mut solver = Solver::new();
        let vars: Vec<Var> = (0..3).map(|_| solver.new_var()).collect();
        let lits: Vec<Lit> = vars.iter().map(|&v| Lit::pos(v)).collect();

        CardinalityEncoder::encode_at_most_k(&mut solver, &lits, 0);

        let result = solver.solve();
        // At most 0 means all must be false, which is satisfiable
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_at_most_1() {
        let mut solver = Solver::new();
        let vars: Vec<Var> = (0..3).map(|_| solver.new_var()).collect();
        let lits: Vec<Lit> = vars.iter().map(|&v| Lit::pos(v)).collect();

        CardinalityEncoder::encode_at_most_k(&mut solver, &lits, 1);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_at_least_1() {
        let mut solver = Solver::new();
        let vars: Vec<Var> = (0..3).map(|_| solver.new_var()).collect();
        let lits: Vec<Lit> = vars.iter().map(|&v| Lit::pos(v)).collect();

        CardinalityEncoder::encode_at_least_k(&mut solver, &lits, 1);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_exactly_2() {
        let mut solver = Solver::new();
        let vars: Vec<Var> = (0..3).map(|_| solver.new_var()).collect();
        let lits: Vec<Lit> = vars.iter().map(|&v| Lit::pos(v)).collect();

        CardinalityEncoder::encode_exactly_k(&mut solver, &lits, 2);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_at_most_exceeds_length() {
        let mut solver = Solver::new();
        let vars: Vec<Var> = (0..3).map(|_| solver.new_var()).collect();
        let lits: Vec<Lit> = vars.iter().map(|&v| Lit::pos(v)).collect();

        let success = CardinalityEncoder::encode_at_most_k(&mut solver, &lits, 5);
        assert!(success);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
    }
}
