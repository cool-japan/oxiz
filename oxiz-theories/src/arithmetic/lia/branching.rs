//\! Branch-and-bound methods for LIA solver

use super::super::simplex::{LinExpr, VarId};
use super::types::{BranchNode, LiaSolver};
use crate::config::BranchingHeuristic;
use num_rational::Rational64;
use num_traits::One;
use oxiz_core::error::{OxizError, Result};
impl LiaSolver {

    /// Check satisfiability with branch-and-bound
    pub fn check(&mut self) -> Result<bool> {
        // First check if the LP relaxation is feasible
        match self.simplex.check() {
            Ok(()) => {
                // LP is feasible, now check integrality
                self.branch_and_bound(0)
            }
            Err(_reasons) => {
                // LP is infeasible, so LIA is also infeasible
                Ok(false)
            }
        }
    }

    /// Branch-and-bound algorithm
    fn branch_and_bound(&mut self, depth: usize) -> Result<bool> {
        if depth > self.max_depth {
            return Err(OxizError::Internal(
                "branch-and-bound depth limit exceeded".to_string(),
            ));
        }

        // Check if current solution is integer
        if let Some((var, value)) = self.find_fractional_var() {
            // Generate cuts before branching (if enabled and under limit)
            if self.cuts_generated < self.config.max_cuts {
                // Try generating cuts in order of strength: MIR > CG > Gomory
                let cut = if self.config.enable_mir_cuts {
                    self.generate_mir_cut(var, value)
                } else if self.config.enable_cg_cuts {
                    self.generate_cg_cut(var, value)
                } else if self.config.enable_gomory_cuts {
                    self.generate_gomory_cut(var, value)
                } else {
                    None
                };

                if let Some(cut) = cut {
                    self.simplex.add_le(cut, 0);
                    self.cuts_generated += 1;

                    // Re-solve after adding cut
                    match self.simplex.check() {
                        Ok(()) => {
                            // After adding cuts, try to fix variables with tight bounds
                            // This can significantly reduce the branching tree
                            let _ = self.fix_tight_bounds();
                            return self.branch_and_bound(depth);
                        }
                        Err(_) => return Ok(false),
                    }
                }
            }

            // Before branching, try to fix variables with tight bounds
            // This reduces the number of variables we need to branch on
            let _ = self.fix_tight_bounds();

            // Branch on the fractional variable
            self.branch_stack.push(BranchNode {
                var,
                branch_up: true,
                fractional_value: value,
            });

            // Try branch up: x >= ceil(value)
            let ceil_value = value.ceil().to_integer();
            let mut expr = LinExpr::new();
            expr.add_term(var, Rational64::one());
            expr.add_constant(-Rational64::from_integer(ceil_value));
            self.simplex.add_ge(expr.clone(), 0);

            let up_cost = match self.simplex.check() {
                Ok(()) => {
                    let result = self.branch_and_bound(depth + 1)?;
                    let cost = (depth + 1) as f64; // Simple depth-based cost
                    if result {
                        // Update pseudo-cost for successful up-branch
                        self.update_pseudo_cost(var, true, cost);
                        return Ok(true);
                    }
                    cost
                }
                Err(_) => {
                    // Branch up is infeasible - high cost
                    0.1 // Infeasible branches have low cost (pruned immediately)
                }
            };

            // Update pseudo-cost for up-branch
            self.update_pseudo_cost(var, true, up_cost);

            // Backtrack and try branch down: x <= floor(value)
            self.simplex.reset();
            let floor_value = value.floor().to_integer();
            let mut expr_down = LinExpr::new();
            expr_down.add_term(var, Rational64::one());
            expr_down.add_constant(-Rational64::from_integer(floor_value));
            self.simplex.add_le(expr_down, 0);

            let down_cost = match self.simplex.check() {
                Ok(()) => {
                    let result = self.branch_and_bound(depth + 1)?;
                    let cost = (depth + 1) as f64;
                    if result {
                        // Update pseudo-cost for successful down-branch
                        self.update_pseudo_cost(var, false, cost);
                        return Ok(true);
                    }
                    cost
                }
                Err(_) => {
                    // Branch down is infeasible
                    0.1
                }
            };

            // Update pseudo-cost for down-branch
            self.update_pseudo_cost(var, false, down_cost);

            self.branch_stack.pop();
            Ok(false)
        } else {
            // All variables are integer-valued
            Ok(true)
        }
    }

    /// Find a variable with fractional value using the configured branching heuristic
    pub(super) fn find_fractional_var(&self) -> Option<(VarId, Rational64)> {
        match self.config.branching_heuristic {
            BranchingHeuristic::FirstFractional => self.find_first_fractional(),
            BranchingHeuristic::MostFractional => self.find_most_fractional(),
            BranchingHeuristic::PseudoCost => self.find_pseudo_cost_var(),
            BranchingHeuristic::StrongBranching => self.find_strong_branching_var(),
        }
    }

    /// Find the first fractional variable (fastest, but may not be optimal)
    pub fn find_first_fractional(&self) -> Option<(VarId, Rational64)> {
        for &var in self.int_vars.keys() {
            let value = self.simplex.value(var);
            if !value.is_integer() {
                return Some((var, value));
            }
        }
        None
    }

    /// Find the most fractional variable (closest to 0.5)
    /// This heuristic prefers variables that are "most uncertain"
    pub fn find_most_fractional(&self) -> Option<(VarId, Rational64)> {
        let mut best_var = None;
        let mut best_fractionality = 0.0;

        for &var in self.int_vars.keys() {
            let value = self.simplex.value(var);
            if !value.is_integer() {
                let frac = value - value.floor();
                let frac_f64 = (*frac.numer() as f64) / (*frac.denom() as f64);
                // Fractionality is how close to 0.5 the fractional part is
                let fractionality = 0.5 - (frac_f64 - 0.5).abs();

                if fractionality > best_fractionality {
                    best_fractionality = fractionality;
                    best_var = Some((var, value));
                }
            }
        }
        best_var
    }

    /// Find variable using pseudo-cost heuristic
    /// Pseudo-cost estimates the expected objective change when branching on a variable
    pub fn find_pseudo_cost_var(&self) -> Option<(VarId, Rational64)> {
        let mut best_var = None;
        let mut best_score = -1.0;

        for &var in self.int_vars.keys() {
            let value = self.simplex.value(var);
            if !value.is_integer() {
                let frac = value - value.floor();
                let frac_f64 = (*frac.numer() as f64) / (*frac.denom() as f64);

                // Get pseudo-costs (default to 1.0 if no history)
                let (down_cost, up_cost, _count) = self
                    .pseudo_costs
                    .get(&var)
                    .copied()
                    .unwrap_or((1.0, 1.0, 0));

                // Score is the product of estimated down and up costs
                // This balances between variables that are expensive to branch on
                let down_score = down_cost * frac_f64;
                let up_score = up_cost * (1.0 - frac_f64);
                let score = down_score * up_score;

                if score > best_score {
                    best_score = score;
                    best_var = Some((var, value));
                }
            }
        }
        best_var
    }

    /// Update pseudo-cost estimates after branching
    ///
    /// Pseudo-costs estimate how much "work" is required when branching on a variable.
    /// We track the number of nodes explored in each branch direction.
    fn update_pseudo_cost(&mut self, var: VarId, branch_up: bool, cost: f64) {
        let entry = self.pseudo_costs.entry(var).or_insert((1.0, 1.0, 0));

        if branch_up {
            // Update up-branch cost using exponential moving average
            let alpha = 0.1; // Learning rate
            entry.1 = (1.0 - alpha) * entry.1 + alpha * cost;
        } else {
            // Update down-branch cost
            let alpha = 0.1;
            entry.0 = (1.0 - alpha) * entry.0 + alpha * cost;
        }

        entry.2 += 1; // Increment observation count
    }

    /// Strong branching: evaluate both branch directions before selecting
    ///
    /// For each fractional variable candidate:
    /// 1. Tentatively add x <= floor(value) and solve for a few iterations
    /// 2. Tentatively add x >= ceil(value) and solve for a few iterations
    /// 3. Score based on the dual bound improvement in both directions
    /// 4. Select the variable with best score
    ///
    /// This is more expensive than other heuristics but often reduces the
    /// branch-and-bound tree size by 20-50%, leading to faster overall solving.
    ///
    /// Reference: Achterberg (2007) "Constraint Integer Programming"
    pub fn find_strong_branching_var(&self) -> Option<(VarId, Rational64)> {
        // Collect all fractional variables with their values
        let mut candidates: Vec<(VarId, Rational64)> = self
            .int_vars
            .keys()
            .filter_map(|&var| {
                let value = self.simplex.value(var);
                if !value.is_integer() {
                    Some((var, value))
                } else {
                    None
                }
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Limit number of candidates if configured
        let max_candidates = self.config.strong_branching_candidates;
        if max_candidates > 0 && candidates.len() > max_candidates {
            // Use most fractional as tiebreaker for limiting candidates
            candidates.sort_by(|(_, val_a), (_, val_b)| {
                let frac_a = val_a - val_a.floor();
                let frac_b = val_b - val_b.floor();
                let frac_a_f64 = (*frac_a.numer() as f64) / (*frac_a.denom() as f64);
                let frac_b_f64 = (*frac_b.numer() as f64) / (*frac_b.denom() as f64);
                let dist_a = (frac_a_f64 - 0.5).abs();
                let dist_b = (frac_b_f64 - 0.5).abs();
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(max_candidates);
        }

        // Evaluate each candidate
        let mut best_var = None;
        let mut best_score = -1.0;

        for &(var, value) in &candidates {
            // Note: We can't actually modify self.simplex here since this is &self
            // In a full implementation, we would:
            // 1. Clone the simplex state (or use push/pop)
            // 2. Add constraint x <= floor(value), solve limited iterations
            // 3. Record bound improvement (down_gain)
            // 4. Restore state
            // 5. Add constraint x >= ceil(value), solve limited iterations
            // 6. Record bound improvement (up_gain)
            // 7. Score = min(down_gain, up_gain) * max(down_gain, up_gain)
            //
            // For now, we fall back to a simplified heuristic that combines
            // pseudo-costs with fractionality (hybrid approach)

            let frac = value - value.floor();
            let frac_f64 = (*frac.numer() as f64) / (*frac.denom() as f64);

            // Get pseudo-costs (default to 1.0 if no history)
            let (down_cost, up_cost, count) = self
                .pseudo_costs
                .get(&var)
                .copied()
                .unwrap_or((1.0, 1.0, 0));

            // If we have pseudo-cost history, use it; otherwise use fractionality
            let score = if count > 0 {
                // Weighted combination of pseudo-cost and fractionality
                let pc_score = down_cost * frac_f64 * up_cost * (1.0 - frac_f64);
                let frac_score = 0.5 - (frac_f64 - 0.5).abs();
                0.8 * pc_score + 0.2 * frac_score
            } else {
                // No history: use fractionality
                0.5 - (frac_f64 - 0.5).abs()
            };

            if score > best_score {
                best_score = score;
                best_var = Some((var, value));
            }
        }

        best_var
    }

}
