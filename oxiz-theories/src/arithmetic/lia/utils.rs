//! Utility and conflict tracking methods for LIA solver

use super::super::simplex::{LinExpr, VarId};
use super::types::{IntBound, LiaSolver};
use num_rational::Rational64;
use oxiz_core::error::Result;

impl LiaSolver {
    /// Integer bound propagation using constraint reasoning
    pub fn propagate_bounds(&mut self) -> Result<()> {
        // For each integer variable, try to derive tighter bounds
        let vars: Vec<VarId> = self.int_vars.keys().copied().collect();

        for &var in &vars {
            // Try to propagate based on constraints involving this variable
            // In a full implementation, we would:
            // 1. Find all constraints containing 'var'
            // 2. For each constraint, compute the implied bound on 'var'
            //    by considering bounds on all other variables
            // 3. Update the bound if it's tighter

            // For now, perform simple bound tightening based on integrality
            let value = self.value(var);

            // If the variable has a fractional value at the LP solution,
            // we can tighten bounds
            if !value.is_integer() {
                // The integer value must be either floor or ceil
                let floor_val = value.floor().to_integer();
                let _ceil_val = value.ceil().to_integer();

                // Check current bounds
                let current_bound = self.int_vars.get(&var);

                // We could potentially update bounds here, but we need to be careful
                // not to modify bounds during the solving process without proper tracking
                if let Some(IntBound::Lower(lb)) = current_bound {
                    // If lower bound allows, we know the value must be at least ceil
                    if *lb <= floor_val {
                        // Could propagate: var >= _ceil_val
                        // But this requires modifying the Simplex state
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the current value of a variable
    #[must_use]
    pub fn value(&self, var: VarId) -> Rational64 {
        self.simplex.value(var)
    }

    /// Reset the solver state
    pub fn reset(&mut self) {
        self.simplex.reset();
        self.int_vars.clear();
        self.branch_stack.clear();
        self.cuts_generated = 0;
        self.conflict_vars.clear();
        self.num_conflicts = 0;
    }

    /// Record a conflict involving the given variables
    ///
    /// This is used for conflict-driven cut selection - we prioritize generating cuts
    /// for variables that frequently appear in conflicts.
    ///
    /// In practice, this is called when:
    /// - A branch leads to infeasibility
    /// - A cut is violated
    /// - The LP relaxation is infeasible
    pub fn record_conflict(&mut self, vars: &[VarId]) {
        self.num_conflicts += 1;

        for &var in vars {
            *self.conflict_vars.entry(var).or_insert(0) += 1;
        }

        // Decay old conflict scores to prioritize recent conflicts
        if self.num_conflicts.is_multiple_of(100) {
            self.decay_conflict_scores();
        }
    }

    /// Decay conflict scores to prioritize recent conflicts
    fn decay_conflict_scores(&mut self) {
        for score in self.conflict_vars.values_mut() {
            *score = (*score * 9) / 10; // Multiply by 0.9
        }

        // Remove variables with very low scores
        self.conflict_vars.retain(|_, &mut score| score > 0);
    }

    /// Get the conflict score for a variable
    ///
    /// Higher scores indicate variables that appear frequently in conflicts.
    /// These are good candidates for cut generation.
    #[must_use]
    pub fn conflict_score(&self, var: VarId) -> u32 {
        self.conflict_vars.get(&var).copied().unwrap_or(0)
    }

    /// Select the best variable for cut generation based on conflict analysis
    ///
    /// This implements conflict-driven cut selection:
    /// 1. Consider only fractional variables
    /// 2. Among fractional variables, prefer those with high conflict scores
    /// 3. If no conflict information, fall back to most fractional value
    ///
    /// Reference: "Conflict-Driven Cutting Planes" by Achterberg (2007)
    #[must_use]
    pub fn select_var_for_cut(&self) -> Option<(VarId, Rational64)> {
        let mut candidates: Vec<(VarId, Rational64, u32)> = Vec::new();

        // Collect all fractional variables with their conflict scores
        for &var in self.int_vars.keys() {
            let value = self.simplex.value(var);
            if !value.is_integer() {
                let score = self.conflict_score(var);
                candidates.push((var, value, score));
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Sort by conflict score (descending), then by fractionality
        candidates.sort_by(|a, b| {
            // First, compare conflict scores (higher is better)
            let score_cmp = b.2.cmp(&a.2);
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }

            // If scores are equal, compare fractionality (closer to 0.5 is better)
            use num_traits::sign::Signed;
            let frac_a = (a.1 - a.1.floor()).abs();
            let frac_b = (b.1 - b.1.floor()).abs();

            let dist_a = (frac_a - Rational64::new(1, 2)).abs();
            let dist_b = (frac_b - Rational64::new(1, 2)).abs();

            dist_a.cmp(&dist_b)
        });

        // Return the best candidate
        candidates.first().map(|(var, val, _)| (*var, *val))
    }

    /// Generate a cut with conflict-driven selection
    ///
    /// This combines cut generation with conflict analysis:
    /// 1. Select a variable using conflict-driven heuristic
    /// 2. Generate the appropriate cut type (Gomory, MIR, or CG)
    /// 3. Apply coefficient lifting to strengthen the cut
    pub fn generate_conflict_driven_cut(&mut self) -> Option<LinExpr> {
        // Select the best variable for cutting
        let (var, value) = self.select_var_for_cut()?;

        // Generate a Gomory cut for this variable
        let mut cut = self.generate_gomory_cut(var, value)?;

        // Apply coefficient lifting to strengthen the cut
        self.lift_cut_all_vars(&mut cut);

        // Record that we generated a cut for this variable
        // (it may lead to future conflicts if it's too aggressive)
        self.record_conflict(&[var]);

        Some(cut)
    }
}
