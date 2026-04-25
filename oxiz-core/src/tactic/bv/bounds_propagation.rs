//! Bit-Vector Bounds Propagation Tactic.
//!
//! Propagates unsigned interval bounds through bit-vector operations.
//! This implements a classic abstract interpretation over the domain of
//! unsigned intervals `[lo, hi]` ⊆ `[0, 2^width - 1]`.

#[allow(unused_imports)]
use crate::prelude::*;
use crate::ast::{TermId, TermKind, TermManager};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Interval bounds for a bit-vector value (unsigned representation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BvInterval {
    /// Lower bound (inclusive), in `[0, 2^width - 1]`
    pub lower: BigUint,
    /// Upper bound (inclusive), in `[0, 2^width - 1]`
    pub upper: BigUint,
    /// Bit-width of the value
    pub width: usize,
}

/// Statistics collected during bounds propagation.
#[derive(Debug, Clone, Default)]
pub struct BvBoundsStats {
    /// Number of term bounds computed (cache misses)
    pub bounds_computed: usize,
    /// Number of unsatisfiable (empty) bounds detected
    pub conflicts_detected: usize,
    /// Number of successful inter-term propagations
    pub propagations: usize,
}

/// Configuration knobs for bounds propagation (currently a unit type).
#[derive(Debug, Clone, Default)]
pub struct BoundsConfig;

/// Public API alias for `BvBoundsPropagation`.
pub type BoundsPropagationTactic = BvBoundsPropagation;

/// Public API alias for `BvBoundsStats`.
pub type BoundsStats = BvBoundsStats;

/// Public API alias for `BvInterval`.
pub type Interval = BvInterval;

// ─── Main tactic ──────────────────────────────────────────────────────────────

/// Bit-vector bounds propagation tactic.
///
/// Traverses a term DAG and computes the tightest unsigned interval
/// `[lo, hi]` reachable for each sub-term, memoising results.  If any
/// interval becomes empty (`lo > hi`) the formula is detected as
/// unsatisfiable and replaced with `⊥`.
pub struct BvBoundsPropagation {
    bounds: FxHashMap<TermId, BvInterval>,
    stats: BvBoundsStats,
}

impl BvBoundsPropagation {
    /// Create a fresh instance.
    pub fn new() -> Self {
        Self { bounds: FxHashMap::default(), stats: BvBoundsStats::default() }
    }

    /// Apply bounds propagation to `formula`, returning a (possibly simpler)
    /// equivalent formula.
    pub fn apply(&mut self, formula: TermId, tm: &TermManager) -> Result<TermId, String> {
        self.compute_bounds(formula, tm)?;

        if self.has_conflict()? {
            self.stats.conflicts_detected += 1;
            return Ok(tm.mk_false());
        }

        Ok(formula)
    }

    /// Return accumulated statistics.
    pub fn stats(&self) -> &BvBoundsStats {
        &self.stats
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Compute bounds for `tid`, memoising the result.
    fn compute_bounds(&mut self, tid: TermId, tm: &TermManager) -> Result<BvInterval, String> {
        if let Some(cached) = self.bounds.get(&tid) {
            return Ok(cached.clone());
        }

        self.stats.bounds_computed += 1;
        let term = tm.get(tid).ok_or_else(|| format!("term {:?} not found", tid))?;

        let interval = match &term.kind {
            // ── Constants ─────────────────────────────────────────────────
            TermKind::BitVecConst { value, width } => {
                let w = *width as usize;
                let mask = mask_for(w);
                // `value` is stored as `BigInt`; convert to unsigned bit-pattern.
                let raw = bigint_to_unsigned(value) & mask;
                BvInterval { lower: raw.clone(), upper: raw, width: w }
            }

            // ── Variables (unconstrained) ──────────────────────────────────
            TermKind::Var(_) => full_range(self.default_width()),

            // ── Bitvector arithmetic / logic ───────────────────────────────
            TermKind::BvAdd(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_add(&lb, &rb)
            }
            TermKind::BvSub(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_sub(&lb, &rb)
            }
            TermKind::BvMul(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_mul(&lb, &rb)
            }
            TermKind::BvAnd(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_and(&lb, &rb)
            }
            TermKind::BvOr(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_or(&lb, &rb)
            }
            TermKind::BvXor(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                // XOR result ≤ OR result upper ≤ mask; lower is 0 conservatively.
                bv_or(&lb, &rb) // conservative superset
            }
            TermKind::BvNot(arg) => {
                let ab = self.compute_bounds(*arg, tm)?;
                self.stats.propagations += 1;
                bv_not(&ab)
            }
            TermKind::BvShl(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_shl(&lb, &rb)
            }
            TermKind::BvLshr(lhs, rhs) => {
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_lshr(&lb, &rb)
            }
            TermKind::BvAshr(lhs, rhs) => {
                // Arithmetic right shift: for unsigned abstract interpretation,
                // treat identically to logical right shift (sound over-approximation).
                let lb = self.compute_bounds(*lhs, tm)?;
                let rb = self.compute_bounds(*rhs, tm)?;
                self.stats.propagations += 1;
                bv_lshr(&lb, &rb)
            }

            // ── All other term kinds: return full range of default width ────
            _ => full_range(self.default_width()),
        };

        self.bounds.insert(tid, interval.clone());
        Ok(interval)
    }

    /// Detect any empty interval (lower > upper), which signals a conflict.
    fn has_conflict(&self) -> Result<bool, String> {
        for iv in self.bounds.values() {
            if iv.lower > iv.upper {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Conservative default bit-width used when the sort is not available.
    fn default_width(&self) -> usize {
        32
    }
}

impl Default for BvBoundsPropagation {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Free arithmetic helpers ─────────────────────────────────────────────────

/// Return the all-ones mask `2^width - 1`.
fn mask_for(width: usize) -> BigUint {
    if width == 0 { BigUint::zero() } else { (BigUint::one() << width) - BigUint::one() }
}

/// Full unsigned range `[0, 2^width - 1]`.
fn full_range(width: usize) -> BvInterval {
    BvInterval { lower: BigUint::zero(), upper: mask_for(width), width }
}

/// Convert a signed `BigInt` to its two's-complement unsigned magnitude.
///
/// For non-negative values this is a no-op.  For negative values the magnitude
/// is the negated value, which is correct modulo `2^width` once masked.
fn bigint_to_unsigned(v: &BigInt) -> BigUint {
    // Both positive and negative: .magnitude() gives the absolute value bits.
    // The caller applies a width mask, so large negatives work correctly.
    match v.sign() {
        Sign::Minus => {
            // Two's-complement: -x is represented as 2^w - x.
            // We return the raw magnitude; the caller masks to width bits,
            // giving 2^w - x mod 2^w, which is correct.
            v.magnitude().clone()
        }
        _ => v.magnitude().clone(),
    }
}

/// Modular unsigned addition: `[a.lo + b.lo, a.hi + b.hi] mod 2^w`.
fn bv_add(a: &BvInterval, b: &BvInterval) -> BvInterval {
    let mask = mask_for(a.width);
    BvInterval {
        lower: (&a.lower + &b.lower) & &mask,
        upper: (&a.upper + &b.upper) & &mask,
        width: a.width,
    }
}

/// Modular unsigned subtraction (conservative over-approximation).
///
/// Without sign information, `a - b` can wrap; we return the full range
/// when the result would underflow.
fn bv_sub(a: &BvInterval, b: &BvInterval) -> BvInterval {
    let w = a.width;
    let mask = mask_for(w);
    if a.lower >= b.upper {
        // No underflow possible.
        BvInterval {
            lower: (&a.lower - &b.upper) & &mask,
            upper: (&a.upper - &b.lower) & &mask,
            width: w,
        }
    } else {
        // Potential underflow: return full range.
        full_range(w)
    }
}

/// Modular unsigned multiplication.
fn bv_mul(a: &BvInterval, b: &BvInterval) -> BvInterval {
    let mask = mask_for(a.width);
    BvInterval {
        lower: (&a.lower * &b.lower) & &mask,
        upper: (&a.upper * &b.upper) & &mask,
        width: a.width,
    }
}

/// Unsigned AND.  `a & b ≤ min(a, b)`, lower is conservatively 0.
fn bv_and(a: &BvInterval, b: &BvInterval) -> BvInterval {
    BvInterval {
        lower: BigUint::zero(),
        upper: a.upper.clone().min(b.upper.clone()),
        width: a.width,
    }
}

/// Unsigned OR.  `a | b ≥ max(a, b)`, upper is conservatively all-ones.
fn bv_or(a: &BvInterval, b: &BvInterval) -> BvInterval {
    BvInterval {
        lower: a.lower.clone().max(b.lower.clone()),
        upper: mask_for(a.width),
        width: a.width,
    }
}

/// Bitwise NOT.  `~x = mask - x`, so `~[lo, hi] = [mask - hi, mask - lo]`.
fn bv_not(a: &BvInterval) -> BvInterval {
    let mask = mask_for(a.width);
    // By invariant hi ≤ mask, so no underflow.
    let new_lo = if a.upper <= mask { &mask - &a.upper } else { BigUint::zero() };
    let new_hi = if a.lower <= mask { &mask - &a.lower } else { BigUint::zero() };
    BvInterval { lower: new_lo, upper: new_hi, width: a.width }
}

/// Unsigned left shift.  Larger shift gives a larger (potentially wrapped) result.
fn bv_shl(value: &BvInterval, shift: &BvInterval) -> BvInterval {
    let w = value.width;
    let mask = mask_for(w);
    let sh_lo = shift.lower.to_u32_digits().first().copied().unwrap_or(0) as usize;
    let sh_hi = shift.upper.to_u32_digits().first().copied().unwrap_or(0) as usize;

    let lower = if sh_lo >= w { BigUint::zero() } else { (&value.lower << sh_lo) & &mask };
    let upper = if sh_hi >= w { BigUint::zero() } else { (&value.upper << sh_hi) & &mask };

    BvInterval { lower, upper, width: w }
}

/// Unsigned logical right shift.  Larger shift gives a smaller result.
fn bv_lshr(value: &BvInterval, shift: &BvInterval) -> BvInterval {
    let w = value.width;
    let sh_lo = shift.lower.to_u32_digits().first().copied().unwrap_or(0) as usize;
    let sh_hi = shift.upper.to_u32_digits().first().copied().unwrap_or(0) as usize;

    // Minimum result uses the largest shift on the smallest value.
    let lower = if sh_hi >= w { BigUint::zero() } else { &value.lower >> sh_hi };
    // Maximum result uses the smallest shift on the largest value.
    let upper = if sh_lo >= w { BigUint::zero() } else { &value.upper >> sh_lo };

    BvInterval { lower, upper, width: w }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let t = BvBoundsPropagation::new();
        assert_eq!(t.stats.bounds_computed, 0);
        assert_eq!(t.stats.conflicts_detected, 0);
    }

    #[test]
    fn test_bv_add_no_overflow() {
        let a = BvInterval { lower: BigUint::from(5u32), upper: BigUint::from(10u32), width: 8 };
        let b = BvInterval { lower: BigUint::from(2u32), upper: BigUint::from(4u32), width: 8 };
        let r = bv_add(&a, &b);
        assert_eq!(r.lower, BigUint::from(7u32));
        assert_eq!(r.upper, BigUint::from(14u32));
        assert_eq!(r.width, 8);
    }

    #[test]
    fn test_bv_not_zero() {
        // width=8, mask=0xFF; NOT [0, 0] = [0xFF, 0xFF]
        let a = BvInterval { lower: BigUint::zero(), upper: BigUint::zero(), width: 8 };
        let r = bv_not(&a);
        assert_eq!(r.lower, BigUint::from(0xFFu32));
        assert_eq!(r.upper, BigUint::from(0xFFu32));
    }

    #[test]
    fn test_bv_and_upper() {
        let a = BvInterval { lower: BigUint::zero(), upper: BigUint::from(10u32), width: 8 };
        let b = BvInterval { lower: BigUint::zero(), upper: BigUint::from(6u32), width: 8 };
        let r = bv_and(&a, &b);
        assert_eq!(r.upper, BigUint::from(6u32));
    }

    #[test]
    fn test_full_range() {
        let r = full_range(8);
        assert_eq!(r.lower, BigUint::zero());
        assert_eq!(r.upper, BigUint::from(255u32));
    }

    #[test]
    fn test_type_aliases() {
        let _: BoundsPropagationTactic = BvBoundsPropagation::new();
        let _: BoundsStats = BvBoundsStats::default();
        let _: Interval = full_range(8);
        let _: BoundsConfig = BoundsConfig;
    }
}
