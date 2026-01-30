//! Bounds Normalization Tactic for Arithmetic.
//!
//! Normalizes arithmetic bounds into canonical form and propagates
//! derived inequalities.

use crate::ast::{TermId, TermKind, TermManager};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// Bounds normalization tactic.
pub struct NormalizeBoundsTactic {
    /// Variable bounds
    bounds: FxHashMap<String, BoundInfo>,
    /// Statistics
    stats: NormalizeBoundsStats,
}

/// Bound information for a variable.
#[derive(Debug, Clone)]
pub struct BoundInfo {
    /// Lower bound (if any)
    pub lower: Option<Bound>,
    /// Upper bound (if any)
    pub upper: Option<Bound>,
}

/// A single bound constraint.
#[derive(Debug, Clone)]
pub struct Bound {
    /// Bound value
    pub value: BigRational,
    /// Is the bound strict (< or >) vs non-strict (≤ or ≥)
    pub strict: bool,
}

/// Normalization statistics.
#[derive(Debug, Clone, Default)]
pub struct NormalizeBoundsStats {
    /// Number of bounds extracted
    pub bounds_extracted: usize,
    /// Number of redundant bounds eliminated
    pub redundant_eliminated: usize,
    /// Number of conflicts detected
    pub conflicts_detected: usize,
    /// Number of derived inequalities
    pub derived_inequalities: usize,
}

impl NormalizeBoundsTactic {
    /// Create a new bounds normalization tactic.
    pub fn new() -> Self {
        Self {
            bounds: FxHashMap::default(),
            stats: NormalizeBoundsStats::default(),
        }
    }

    /// Apply normalization to a formula.
    pub fn apply(&mut self, formula: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        // Phase 1: Extract all bounds
        self.extract_bounds(formula, tm)?;

        // Phase 2: Check for conflicts
        if let Some(conflict_var) = self.check_conflicts() {
            self.stats.conflicts_detected += 1;
            return Ok(tm.mk_false());
        }

        // Phase 3: Generate normalized inequalities
        let normalized_ineqs = self.generate_normalized_inequalities(tm)?;

        // Phase 4: Derive additional inequalities
        let derived = self.derive_inequalities(tm)?;
        self.stats.derived_inequalities += derived.len();

        // Combine all
        let mut all_ineqs = normalized_ineqs;
        all_ineqs.extend(derived);

        if all_ineqs.is_empty() {
            Ok(tm.mk_true())
        } else {
            tm.mk_and(all_ineqs)
        }
    }

    /// Extract bounds from formula.
    fn extract_bounds(&mut self, formula: TermId, tm: &TermManager) -> Result<(), String> {
        let mut stack = vec![formula];

        while let Some(tid) = stack.pop() {
            let term = tm.get(tid).ok_or("term not found")?;

            match &term.kind {
                TermKind::And(args) => {
                    stack.extend(args);
                }
                TermKind::Le(lhs, rhs) => {
                    self.extract_bound_from_le(*lhs, *rhs, false, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                TermKind::Lt(lhs, rhs) => {
                    self.extract_bound_from_le(*lhs, *rhs, true, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                TermKind::Ge(lhs, rhs) => {
                    self.extract_bound_from_le(*rhs, *lhs, false, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                TermKind::Gt(lhs, rhs) => {
                    self.extract_bound_from_le(*rhs, *lhs, true, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Extract bound from ≤ constraint: lhs ≤ rhs.
    fn extract_bound_from_le(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        strict: bool,
        tm: &TermManager,
    ) -> Result<(), String> {
        // Check if lhs is a variable
        if let Some(var_name) = self.get_var_name(lhs, tm) {
            // rhs is upper bound: var ≤ rhs
            if let Some(value) = self.evaluate_to_constant(rhs, tm) {
                let bound = Bound { value, strict };
                self.update_upper_bound(var_name, bound);
            }
        }

        // Check if rhs is a variable
        if let Some(var_name) = self.get_var_name(rhs, tm) {
            // lhs is lower bound: lhs ≤ var
            if let Some(value) = self.evaluate_to_constant(lhs, tm) {
                let bound = Bound { value, strict };
                self.update_lower_bound(var_name, bound);
            }
        }

        Ok(())
    }

    /// Update lower bound for variable.
    fn update_lower_bound(&mut self, var: String, new_bound: Bound) {
        let entry = self.bounds.entry(var).or_insert(BoundInfo {
            lower: None,
            upper: None,
        });

        // Keep the tighter bound
        let should_update = match &entry.lower {
            None => true,
            Some(existing) => new_bound.value > existing.value
                || (new_bound.value == existing.value && new_bound.strict && !existing.strict),
        };

        if should_update {
            if entry.lower.is_some() {
                self.stats.redundant_eliminated += 1;
            }
            entry.lower = Some(new_bound);
        } else {
            self.stats.redundant_eliminated += 1;
        }
    }

    /// Update upper bound for variable.
    fn update_upper_bound(&mut self, var: String, new_bound: Bound) {
        let entry = self.bounds.entry(var).or_insert(BoundInfo {
            lower: None,
            upper: None,
        });

        // Keep the tighter bound
        let should_update = match &entry.upper {
            None => true,
            Some(existing) => new_bound.value < existing.value
                || (new_bound.value == existing.value && new_bound.strict && !existing.strict),
        };

        if should_update {
            if entry.upper.is_some() {
                self.stats.redundant_eliminated += 1;
            }
            entry.upper = Some(new_bound);
        } else {
            self.stats.redundant_eliminated += 1;
        }
    }

    /// Check for conflicting bounds.
    fn check_conflicts(&self) -> Option<String> {
        for (var, info) in &self.bounds {
            if let (Some(lower), Some(upper)) = (&info.lower, &info.upper) {
                // Check if lower > upper
                if lower.value > upper.value {
                    return Some(var.clone());
                }

                // Check if lower == upper but both strict
                if lower.value == upper.value && (lower.strict || upper.strict) {
                    return Some(var.clone());
                }
            }
        }

        None
    }

    /// Generate normalized inequalities from bounds.
    fn generate_normalized_inequalities(&self, tm: &mut TermManager) -> Result<Vec<TermId>, String> {
        let mut ineqs = Vec::new();

        for (var, info) in &self.bounds {
            let var_term = tm.mk_var(var.clone(), tm.sorts.int_sort);

            if let Some(lower) = &info.lower {
                let lower_const = tm.mk_rat(lower.value.clone())?;
                let ineq = if lower.strict {
                    tm.mk_gt(var_term, lower_const)?
                } else {
                    tm.mk_ge(var_term, lower_const)?
                };
                ineqs.push(ineq);
            }

            if let Some(upper) = &info.upper {
                let upper_const = tm.mk_rat(upper.value.clone())?;
                let ineq = if upper.strict {
                    tm.mk_lt(var_term, upper_const)?
                } else {
                    tm.mk_le(var_term, upper_const)?
                };
                ineqs.push(ineq);
            }
        }

        Ok(ineqs)
    }

    /// Derive additional inequalities from bounds.
    fn derive_inequalities(&self, tm: &mut TermManager) -> Result<Vec<TermId>, String> {
        let mut derived = Vec::new();

        // For each pair of variables with bounds, derive transitivity
        let vars: Vec<_> = self.bounds.keys().cloned().collect();

        for i in 0..vars.len() {
            for j in (i + 1)..vars.len() {
                let var1 = &vars[i];
                let var2 = &vars[j];

                if let (Some(info1), Some(info2)) = (self.bounds.get(var1), self.bounds.get(var2)) {
                    // If var1 ≤ c1 and c2 ≤ var2, and c1 ≤ c2, then var1 ≤ var2
                    if let (Some(upper1), Some(lower2)) = (&info1.upper, &info2.lower) {
                        if upper1.value <= lower2.value {
                            let var1_term = tm.mk_var(var1.clone(), tm.sorts.int_sort);
                            let var2_term = tm.mk_var(var2.clone(), tm.sorts.int_sort);
                            let derived_ineq = tm.mk_le(var1_term, var2_term)?;
                            derived.push(derived_ineq);
                        }
                    }
                }
            }
        }

        Ok(derived)
    }

    /// Get variable name if term is a variable.
    fn get_var_name(&self, tid: TermId, tm: &TermManager) -> Option<String> {
        let term = tm.get(tid)?;
        match &term.kind {
            TermKind::Var(name) => Some(name.to_string()),
            _ => None,
        }
    }

    /// Evaluate term to constant if possible.
    fn evaluate_to_constant(&self, tid: TermId, tm: &TermManager) -> Option<BigRational> {
        let term = tm.get(tid)?;
        match &term.kind {
            TermKind::IntConst(val) => Some(BigRational::from_integer(val.clone())),
            TermKind::Rat(val) => Some(val.clone()),
            _ => None,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &NormalizeBoundsStats {
        &self.stats
    }
}

impl Default for NormalizeBoundsTactic {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_bounds_tactic() {
        let tactic = NormalizeBoundsTactic::new();
        assert_eq!(tactic.stats.bounds_extracted, 0);
    }

    #[test]
    fn test_update_bounds() {
        let mut tactic = NormalizeBoundsTactic::new();

        let bound1 = Bound {
            value: BigRational::from_integer(BigInt::from(5)),
            strict: false,
        };

        tactic.update_lower_bound("x".to_string(), bound1);

        assert!(tactic.bounds.contains_key("x"));
        assert!(tactic.bounds.get("x").unwrap().lower.is_some());
    }
}
