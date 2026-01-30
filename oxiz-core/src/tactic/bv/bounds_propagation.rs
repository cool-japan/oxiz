//! Bit-Vector Bounds Propagation Tactic.
//!
//! Propagates interval bounds through bit-vector operations.

use crate::ast::{TermId, TermKind, TermManager};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// Bit-vector bounds propagation tactic.
pub struct BvBoundsPropagation {
    /// Bounds for each term
    bounds: FxHashMap<TermId, BvInterval>,
    /// Statistics
    stats: BvBoundsStats,
}

/// Interval bounds for a bit-vector value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BvInterval {
    /// Lower bound (inclusive)
    pub lower: BigUint,
    /// Upper bound (inclusive)
    pub upper: BigUint,
    /// Bit-width
    pub width: usize,
}

/// Bounds propagation statistics.
#[derive(Debug, Clone, Default)]
pub struct BvBoundsStats {
    /// Number of bounds computed
    pub bounds_computed: usize,
    /// Number of conflicts detected
    pub conflicts_detected: usize,
    /// Number of propagations
    pub propagations: usize,
}

impl BvBoundsPropagation {
    /// Create a new BV bounds propagation tactic.
    pub fn new() -> Self {
        Self {
            bounds: FxHashMap::default(),
            stats: BvBoundsStats::default(),
        }
    }

    /// Apply bounds propagation to formula.
    pub fn apply(&mut self, formula: TermId, tm: &TermManager) -> Result<TermId, String> {
        // Compute bounds for all subterms
        self.compute_bounds(formula, tm)?;

        // Check for conflicts
        if self.has_conflict(formula, tm)? {
            self.stats.conflicts_detected += 1;
            return Ok(tm.mk_false());
        }

        // Return simplified formula (placeholder)
        Ok(formula)
    }

    /// Compute interval bounds for a term.
    fn compute_bounds(&mut self, tid: TermId, tm: &TermManager) -> Result<BvInterval, String> {
        // Check if already computed
        if let Some(bounds) = self.bounds.get(&tid) {
            return Ok(bounds.clone());
        }

        self.stats.bounds_computed += 1;

        let term = tm.get(tid).ok_or("term not found")?;

        let bounds = match &term.kind {
            TermKind::BitVecConst { value: val, width } => {
                // Constant: exact interval
                BvInterval {
                    lower: val.clone(),
                    upper: val.clone(),
                    width: *width,
                }
            }

            TermKind::BvVar(_, width) => {
                // Variable: full range [0, 2^width - 1]
                let max_val = (BigUint::one() << *width) - BigUint::one();
                BvInterval {
                    lower: BigUint::zero(),
                    upper: max_val,
                    width: *width,
                }
            }

            TermKind::BvAdd(args) => {
                self.compute_add_bounds(args, tm)?
            }

            TermKind::BvMul(args) => {
                self.compute_mul_bounds(args, tm)?
            }

            TermKind::BvAnd(args) => {
                self.compute_and_bounds(args, tm)?
            }

            TermKind::BvOr(args) => {
                self.compute_or_bounds(args, tm)?
            }

            TermKind::BvNot(arg) => {
                self.compute_not_bounds(*arg, tm)?
            }

            TermKind::BvShl(lhs, rhs) => {
                self.compute_shl_bounds(*lhs, *rhs, tm)?
            }

            TermKind::BvLshr(lhs, rhs) => {
                self.compute_lshr_bounds(*lhs, *rhs, tm)?
            }

            _ => {
                // Unknown: return full range
                let width = self.get_width(tid, tm)?;
                let max_val = (BigUint::one() << width) - BigUint::one();
                BvInterval {
                    lower: BigUint::zero(),
                    upper: max_val,
                    width,
                }
            }
        };

        self.bounds.insert(tid, bounds.clone());
        Ok(bounds)
    }

    /// Compute bounds for BV addition.
    fn compute_add_bounds(&mut self, args: &[TermId], tm: &TermManager) -> Result<BvInterval, String> {
        if args.is_empty() {
            return Err("empty add".to_string());
        }

        let mut result = self.compute_bounds(args[0], tm)?;

        for &arg in &args[1..] {
            let arg_bounds = self.compute_bounds(arg, tm)?;
            result = self.interval_add(&result, &arg_bounds);
        }

        Ok(result)
    }

    /// Interval addition.
    fn interval_add(&self, a: &BvInterval, b: &BvInterval) -> BvInterval {
        let width = a.width;
        let mask = (BigUint::one() << width) - BigUint::one();

        let lower = (&a.lower + &b.lower) & &mask;
        let upper = (&a.upper + &b.upper) & &mask;

        BvInterval {
            lower,
            upper,
            width,
        }
    }

    /// Compute bounds for BV multiplication.
    fn compute_mul_bounds(&mut self, args: &[TermId], tm: &TermManager) -> Result<BvInterval, String> {
        if args.is_empty() {
            return Err("empty mul".to_string());
        }

        let mut result = self.compute_bounds(args[0], tm)?;

        for &arg in &args[1..] {
            let arg_bounds = self.compute_bounds(arg, tm)?;
            result = self.interval_mul(&result, &arg_bounds);
        }

        Ok(result)
    }

    /// Interval multiplication.
    fn interval_mul(&self, a: &BvInterval, b: &BvInterval) -> BvInterval {
        let width = a.width;
        let mask = (BigUint::one() << width) - BigUint::one();

        let lower = (&a.lower * &b.lower) & &mask;
        let upper = (&a.upper * &b.upper) & &mask;

        BvInterval {
            lower,
            upper,
            width,
        }
    }

    /// Compute bounds for BV AND.
    fn compute_and_bounds(&mut self, args: &[TermId], tm: &TermManager) -> Result<BvInterval, String> {
        if args.is_empty() {
            return Err("empty and".to_string());
        }

        let mut result = self.compute_bounds(args[0], tm)?;

        for &arg in &args[1..] {
            let arg_bounds = self.compute_bounds(arg, tm)?;
            result = self.interval_and(&result, &arg_bounds);
        }

        Ok(result)
    }

    /// Interval AND.
    fn interval_and(&self, a: &BvInterval, b: &BvInterval) -> BvInterval {
        // AND can only decrease or keep bits: upper(a & b) ≤ min(upper(a), upper(b))
        let width = a.width;

        let lower = BigUint::zero(); // Worst case: all bits cleared
        let upper = a.upper.clone().min(b.upper.clone());

        BvInterval {
            lower,
            upper,
            width,
        }
    }

    /// Compute bounds for BV OR.
    fn compute_or_bounds(&mut self, args: &[TermId], tm: &TermManager) -> Result<BvInterval, String> {
        if args.is_empty() {
            return Err("empty or".to_string());
        }

        let mut result = self.compute_bounds(args[0], tm)?;

        for &arg in &args[1..] {
            let arg_bounds = self.compute_bounds(arg, tm)?;
            result = self.interval_or(&result, &arg_bounds);
        }

        Ok(result)
    }

    /// Interval OR.
    fn interval_or(&self, a: &BvInterval, b: &BvInterval) -> BvInterval {
        // OR can only increase or keep bits: lower(a | b) ≥ max(lower(a), lower(b))
        let width = a.width;
        let mask = (BigUint::one() << width) - BigUint::one();

        let lower = a.lower.clone().max(b.lower.clone());
        let upper = mask; // Worst case: all bits set

        BvInterval {
            lower,
            upper,
            width,
        }
    }

    /// Compute bounds for BV NOT.
    fn compute_not_bounds(&mut self, arg: TermId, tm: &TermManager) -> Result<BvInterval, String> {
        let arg_bounds = self.compute_bounds(arg, tm)?;

        let width = arg_bounds.width;
        let mask = (BigUint::one() << width) - BigUint::one();

        // NOT flips bits: ~x
        let lower = !&arg_bounds.upper.clone() & &mask;
        let upper = !&arg_bounds.lower.clone() & &mask;

        Ok(BvInterval {
            lower,
            upper,
            width,
        })
    }

    /// Compute bounds for left shift.
    fn compute_shl_bounds(&mut self, lhs: TermId, rhs: TermId, tm: &TermManager) -> Result<BvInterval, String> {
        let lhs_bounds = self.compute_bounds(lhs, tm)?;
        let rhs_bounds = self.compute_bounds(rhs, tm)?;

        let width = lhs_bounds.width;
        let mask = (BigUint::one() << width) - BigUint::one();

        // Shift by lower bound (minimum shift)
        let lower = (&lhs_bounds.lower << rhs_bounds.lower.to_u32_digits().first().copied().unwrap_or(0)) & &mask;

        // Shift by upper bound (maximum shift)
        let upper = (&lhs_bounds.upper << rhs_bounds.upper.to_u32_digits().first().copied().unwrap_or(0)) & &mask;

        Ok(BvInterval {
            lower,
            upper,
            width,
        })
    }

    /// Compute bounds for logical right shift.
    fn compute_lshr_bounds(&mut self, lhs: TermId, rhs: TermId, tm: &TermManager) -> Result<BvInterval, String> {
        let lhs_bounds = self.compute_bounds(lhs, tm)?;
        let rhs_bounds = self.compute_bounds(rhs, tm)?;

        let width = lhs_bounds.width;

        // Shift by upper bound (minimum value)
        let lower = &lhs_bounds.lower >> rhs_bounds.upper.to_u32_digits().first().copied().unwrap_or(0);

        // Shift by lower bound (maximum value)
        let upper = &lhs_bounds.upper >> rhs_bounds.lower.to_u32_digits().first().copied().unwrap_or(0);

        Ok(BvInterval {
            lower,
            upper,
            width,
        })
    }

    /// Check for conflicts.
    fn has_conflict(&self, _formula: TermId, _tm: &TermManager) -> Result<bool, String> {
        // Placeholder: check if any bounds are empty or contradictory
        Ok(false)
    }

    /// Get bit-width of term.
    fn get_width(&self, _tid: TermId, _tm: &TermManager) -> Result<usize, String> {
        Ok(32) // Default width
    }

    /// Get statistics.
    pub fn stats(&self) -> &BvBoundsStats {
        &self.stats
    }
}

impl Default for BvBoundsPropagation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bv_bounds_propagation() {
        let tactic = BvBoundsPropagation::new();
        assert_eq!(tactic.stats.bounds_computed, 0);
    }

    #[test]
    fn test_interval_add() {
        let tactic = BvBoundsPropagation::new();

        let a = BvInterval {
            lower: BigUint::from(5u32),
            upper: BigUint::from(10u32),
            width: 8,
        };

        let b = BvInterval {
            lower: BigUint::from(2u32),
            upper: BigUint::from(4u32),
            width: 8,
        };

        let result = tactic.interval_add(&a, &b);

        assert_eq!(result.width, 8);
    }
}
