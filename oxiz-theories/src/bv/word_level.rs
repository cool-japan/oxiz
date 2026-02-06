//! Word-Level Reasoning for Bit-Vectors
//!
//! This module implements word-level reasoning techniques that avoid bit-blasting
//! when possible, operating on bitvectors as atomic values. This is crucial for
//! performance on formulas with large bitvector widths.
//!
//! # Key Techniques
//!
//! ## 1. Interval Analysis
//! - Track upper and lower bounds for bitvector values
//! - Propagate bounds through operations
//! - Detect conflicts at word level before bit-blasting
//! - Implement abstract interpretation for BV operations
//!
//! ## 2. Sign Detection
//! - Determine if a bitvector is always positive, always negative, or mixed
//! - Use sign information for optimization
//! - Track sign bits through operations
//! - Signed vs unsigned reasoning
//!
//! ## 3. Overflow Analysis
//! - Detect when operations cannot overflow
//! - Simplify operations when overflow is impossible
//! - Track overflow conditions explicitly
//! - Use overflow information for bounds refinement
//!
//! ## 4. Constant Propagation
//! - Eagerly evaluate operations on constants
//! - Partial evaluation when some operands are constants
//! - Strength reduction (e.g., multiply by 2^n → shift)
//!
//! ## 5. Equality Reasoning
//! - Maintain equivalence classes of bitvectors
//! - Congruence closure over BV operations
//! - Extract equalities before bit-blasting
//!
//! ## 6. Bit-Width Reduction
//! - Detect unused high bits
//! - Narrow bitvector widths when safe
//! - Zero-extension and sign-extension elimination
//!
//! # References
//!
//! - Z3: `src/ast/rewriter/bv_rewriter.cpp`
//! - "Abstract Conflict Driven Learning" (D'Silva et al.)
//! - "Bit-Vector Rewriting with Automatic Rule Generation" (Niemetz et al.)
//! - "Precise Widening Operators for Bit-Vectors" (Brauer et al.)

#![allow(missing_docs)]

use super::propagator::Interval;
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Sign information for a bitvector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignInfo {
    /// Always non-negative (sign bit = 0)
    NonNegative,
    /// Always negative (sign bit = 1)
    Negative,
    /// Sign is unknown (could be either)
    Unknown,
}

impl SignInfo {
    /// Merge two sign infos (for different paths)
    #[must_use]
    pub fn merge(self, other: SignInfo) -> SignInfo {
        match (self, other) {
            (SignInfo::NonNegative, SignInfo::NonNegative) => SignInfo::NonNegative,
            (SignInfo::Negative, SignInfo::Negative) => SignInfo::Negative,
            _ => SignInfo::Unknown,
        }
    }

    /// Propagate through negation
    #[must_use]
    pub fn negate(self) -> SignInfo {
        match self {
            SignInfo::NonNegative => SignInfo::Negative,
            SignInfo::Negative => SignInfo::NonNegative,
            SignInfo::Unknown => SignInfo::Unknown,
        }
    }

    /// Check if definitely non-negative
    #[must_use]
    pub fn is_non_negative(self) -> bool {
        matches!(self, SignInfo::NonNegative)
    }

    /// Check if definitely negative
    #[must_use]
    pub fn is_negative(self) -> bool {
        matches!(self, SignInfo::Negative)
    }
}

/// Overflow information for an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowInfo {
    /// Operation cannot overflow
    NoOverflow,
    /// Operation may overflow
    MayOverflow,
    /// Operation definitely overflows
    MustOverflow,
}

/// Bit-width usage information
#[derive(Debug, Clone)]
pub struct WidthInfo {
    /// Original width
    pub original_width: usize,
    /// Minimum width needed (high bits are known to be 0 or sign-extension)
    pub effective_width: usize,
    /// High bits are zero-extension
    pub is_zero_extended: bool,
    /// High bits are sign-extension
    pub is_sign_extended: bool,
}

impl WidthInfo {
    /// Create width info for full width (no reduction possible)
    #[must_use]
    pub fn full(width: usize) -> Self {
        Self {
            original_width: width,
            effective_width: width,
            is_zero_extended: false,
            is_sign_extended: false,
        }
    }

    /// Check if width reduction is possible
    #[must_use]
    pub fn can_reduce(&self) -> bool {
        self.effective_width < self.original_width
    }

    /// Get the number of bits that can be eliminated
    #[must_use]
    pub fn reduction_amount(&self) -> usize {
        self.original_width - self.effective_width
    }
}

/// Word-level reasoner for bitvectors
pub struct WordLevelReasoner {
    /// Interval bounds for each term
    intervals: FxHashMap<TermId, Interval>,
    /// Sign information for each term
    signs: FxHashMap<TermId, SignInfo>,
    /// Overflow information for operations
    overflows: FxHashMap<TermId, OverflowInfo>,
    /// Width information for terms
    widths: FxHashMap<TermId, WidthInfo>,
    /// Constant values (if known)
    constants: FxHashMap<TermId, u64>,
    /// Equivalence classes (union-find structure)
    eq_classes: UnionFind,
    /// Terms that are equal
    equalities: Vec<(TermId, TermId)>,
    /// Pending propagation queue
    propagation_queue: VecDeque<TermId>,
    /// Statistics
    stats: WordLevelStats,
}

/// Statistics for word-level reasoning
#[derive(Debug, Clone, Default)]
pub struct WordLevelStats {
    /// Number of bounds refined
    pub bounds_refined: usize,
    /// Number of signs detected
    pub signs_detected: usize,
    /// Number of overflows detected
    pub overflows_detected: usize,
    /// Number of widths reduced
    pub widths_reduced: usize,
    /// Number of constants propagated
    pub constants_propagated: usize,
    /// Number of equalities detected
    pub equalities_detected: usize,
    /// Number of conflicts detected at word level
    pub word_level_conflicts: usize,
}

impl Default for WordLevelReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl WordLevelReasoner {
    /// Create a new word-level reasoner
    #[must_use]
    pub fn new() -> Self {
        Self {
            intervals: FxHashMap::default(),
            signs: FxHashMap::default(),
            overflows: FxHashMap::default(),
            widths: FxHashMap::default(),
            constants: FxHashMap::default(),
            eq_classes: UnionFind::new(),
            equalities: Vec::new(),
            propagation_queue: VecDeque::new(),
            stats: WordLevelStats::default(),
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &WordLevelStats {
        &self.stats
    }

    /// Reset the reasoner
    pub fn reset(&mut self) {
        self.intervals.clear();
        self.signs.clear();
        self.overflows.clear();
        self.widths.clear();
        self.constants.clear();
        self.eq_classes = UnionFind::new();
        self.equalities.clear();
        self.propagation_queue.clear();
        self.stats = WordLevelStats::default();
    }

    /// Set interval for a term
    pub fn set_interval(&mut self, term: TermId, interval: Interval) {
        // Check for refinement
        if let Some(old_interval) = self.intervals.get(&term) {
            if let Some(new_interval) = old_interval.intersect(&interval) {
                if new_interval.is_empty() {
                    self.stats.word_level_conflicts += 1;
                } else if new_interval != *old_interval {
                    // Check if interval is now a singleton (constant) - extract before moving
                    let is_singleton = new_interval.is_singleton();
                    let lower_value = new_interval.lower;

                    self.intervals.insert(term, new_interval);
                    self.propagation_queue.push_back(term);
                    self.stats.bounds_refined += 1;

                    if is_singleton {
                        self.constants.insert(term, lower_value);
                        self.stats.constants_propagated += 1;
                    }
                }
            } else {
                // Empty intersection = conflict
                self.stats.word_level_conflicts += 1;
            }
        } else {
            self.intervals.insert(term, interval.clone());
            self.propagation_queue.push_back(term);

            if interval.is_singleton() {
                self.constants.insert(term, interval.lower);
                self.stats.constants_propagated += 1;
            }
        }
    }

    /// Get interval for a term
    #[must_use]
    pub fn get_interval(&self, term: TermId) -> Option<&Interval> {
        self.intervals.get(&term)
    }

    /// Set sign information for a term
    pub fn set_sign(&mut self, term: TermId, sign: SignInfo) {
        let existing = self.signs.entry(term).or_insert(sign);
        let merged = existing.merge(sign);

        if merged != *existing {
            *existing = merged;
            self.propagation_queue.push_back(term);
            self.stats.signs_detected += 1;
        }
    }

    /// Get sign information for a term
    #[must_use]
    pub fn get_sign(&self, term: TermId) -> SignInfo {
        self.signs.get(&term).copied().unwrap_or(SignInfo::Unknown)
    }

    /// Infer sign from interval
    #[must_use]
    pub fn infer_sign_from_interval(&self, interval: &Interval) -> SignInfo {
        let width = interval.width as usize;
        if width == 0 {
            return SignInfo::Unknown;
        }

        let sign_bit = 1u64 << (width - 1);

        // Check if upper bound has sign bit clear → non-negative
        if interval.upper < sign_bit {
            return SignInfo::NonNegative;
        }

        // Check if lower bound has sign bit set → negative
        if interval.lower >= sign_bit {
            return SignInfo::Negative;
        }

        SignInfo::Unknown
    }

    /// Set overflow information for a term
    pub fn set_overflow(&mut self, term: TermId, overflow: OverflowInfo) {
        if let Some(existing) = self.overflows.get(&term) {
            if *existing != overflow {
                self.overflows.insert(term, overflow);
                self.stats.overflows_detected += 1;
            }
        } else {
            self.overflows.insert(term, overflow);
            self.stats.overflows_detected += 1;
        }
    }

    /// Get overflow information for a term
    #[must_use]
    pub fn get_overflow(&self, term: TermId) -> OverflowInfo {
        self.overflows
            .get(&term)
            .copied()
            .unwrap_or(OverflowInfo::MayOverflow)
    }

    /// Detect overflow for addition
    #[must_use]
    pub fn detect_add_overflow(&self, a: &Interval, b: &Interval) -> OverflowInfo {
        let max_value = if a.width == 64 {
            u64::MAX
        } else {
            (1u64 << a.width) - 1
        };

        // Check if a.upper + b.upper can overflow
        let sum_upper = a.upper.wrapping_add(b.upper);

        // If wrapping occurred and sum is less than either operand, overflow happened
        if sum_upper > max_value || (sum_upper < a.upper && sum_upper < b.upper) {
            // May overflow
            if a.lower.wrapping_add(b.lower) > max_value {
                OverflowInfo::MustOverflow
            } else {
                OverflowInfo::MayOverflow
            }
        } else {
            OverflowInfo::NoOverflow
        }
    }

    /// Detect overflow for multiplication
    #[must_use]
    pub fn detect_mul_overflow(&self, a: &Interval, b: &Interval) -> OverflowInfo {
        let max_value = if a.width == 64 {
            u64::MAX
        } else {
            (1u64 << a.width) - 1
        };

        // Conservative check: if a.upper * b.upper would need more than width bits
        if a.upper > 0 && b.upper > max_value / a.upper {
            if a.lower > 0 && b.lower > max_value / a.lower {
                OverflowInfo::MustOverflow
            } else {
                OverflowInfo::MayOverflow
            }
        } else {
            OverflowInfo::NoOverflow
        }
    }

    /// Set width information for a term
    pub fn set_width_info(&mut self, term: TermId, width_info: WidthInfo) {
        if let Some(existing) = self.widths.get(&term) {
            // Take the more restrictive width
            if width_info.effective_width < existing.effective_width {
                self.widths.insert(term, width_info);
                self.stats.widths_reduced += 1;
            }
        } else {
            self.widths.insert(term, width_info);
        }
    }

    /// Get width information for a term
    #[must_use]
    pub fn get_width_info(&self, term: TermId) -> Option<&WidthInfo> {
        self.widths.get(&term)
    }

    /// Detect effective width from interval
    #[must_use]
    pub fn detect_effective_width(&self, interval: &Interval) -> WidthInfo {
        let original_width = interval.width as usize;

        // Find the highest bit set in upper bound
        let highest_bit = if interval.upper == 0 {
            0
        } else {
            64 - interval.upper.leading_zeros() as usize
        };

        let effective_width = highest_bit.max(1);

        // Check if high bits are zero
        let is_zero_extended = effective_width < original_width;

        // Check if high bits are sign extension
        let sign_bit_pos = effective_width.saturating_sub(1);
        let sign_bit = if sign_bit_pos < 64 {
            (interval.lower >> sign_bit_pos) & 1
        } else {
            0
        };

        let is_sign_extended = if sign_bit == 1 {
            // Check if all high bits match sign bit
            let high_mask = if original_width >= 64 {
                0
            } else {
                (!0u64) << effective_width
            };

            (interval.lower & high_mask) == high_mask && (interval.upper & high_mask) == high_mask
        } else {
            false
        };

        WidthInfo {
            original_width,
            effective_width,
            is_zero_extended,
            is_sign_extended,
        }
    }

    /// Set constant value for a term
    pub fn set_constant(&mut self, term: TermId, value: u64, width: u32) {
        self.constants.insert(term, value);
        self.set_interval(term, Interval::singleton(value, width));
        // Note: constants_propagated is incremented in set_interval for singletons
    }

    /// Get constant value if known
    #[must_use]
    pub fn get_constant(&self, term: TermId) -> Option<u64> {
        self.constants.get(&term).copied()
    }

    /// Assert equality between two terms
    pub fn assert_equal(&mut self, a: TermId, b: TermId) {
        self.eq_classes.union(a, b);
        self.equalities.push((a, b));
        self.stats.equalities_detected += 1;

        // Merge intervals
        if let (Some(ia), Some(ib)) = (
            self.intervals.get(&a).cloned(),
            self.intervals.get(&b).cloned(),
        ) && let Some(merged) = ia.intersect(&ib)
        {
            self.set_interval(a, merged.clone());
            self.set_interval(b, merged);
        }

        // Propagate constants
        if let Some(val) = self.constants.get(&a).copied() {
            if let std::collections::hash_map::Entry::Vacant(e) = self.constants.entry(b) {
                e.insert(val);
                self.stats.constants_propagated += 1;
            }
        } else if let Some(val) = self.constants.get(&b).copied() {
            self.constants.insert(a, val);
            self.stats.constants_propagated += 1;
        }
    }

    /// Check if two terms are in the same equivalence class
    #[must_use]
    pub fn are_equal(&self, a: TermId, b: TermId) -> bool {
        self.eq_classes.find(a) == self.eq_classes.find(b)
    }

    /// Propagate addition: c = a + b
    pub fn propagate_add(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // If both operands are constants, compute result
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            let result = va.wrapping_add(vb);
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            self.set_constant(c, result & mask, width);
            return Ok(());
        }

        // Propagate intervals - clone before using to avoid borrow checker issues
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ia = ia.clone();
            let ib = ib.clone();
            let ic = Interval::propagate_add(&ia, &ib);
            self.set_interval(c, ic);

            // Detect overflow
            let overflow = self.detect_add_overflow(&ia, &ib);
            self.set_overflow(c, overflow);
        }

        // Propagate signs
        let sign_a = self.get_sign(a);
        let sign_b = self.get_sign(b);

        let sign_c = match (sign_a, sign_b) {
            (SignInfo::NonNegative, SignInfo::NonNegative) => SignInfo::NonNegative,
            (SignInfo::Negative, SignInfo::Negative) => SignInfo::Negative,
            _ => SignInfo::Unknown,
        };

        self.set_sign(c, sign_c);

        Ok(())
    }

    /// Propagate subtraction: c = a - b
    pub fn propagate_sub(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // Constants
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            let result = va.wrapping_sub(vb);
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            self.set_constant(c, result & mask, width);
            return Ok(());
        }

        // Intervals
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ic = Interval::propagate_sub(ia, ib);
            self.set_interval(c, ic);
        }

        Ok(())
    }

    /// Propagate multiplication: c = a * b
    pub fn propagate_mul(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // Constants
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            let result = va.wrapping_mul(vb);
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            self.set_constant(c, result & mask, width);
            return Ok(());
        }

        // Intervals - clone before using to avoid borrow checker issues
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ia = ia.clone();
            let ib = ib.clone();
            let ic = Interval::propagate_mul(&ia, &ib);
            self.set_interval(c, ic);

            // Detect overflow
            let overflow = self.detect_mul_overflow(&ia, &ib);
            self.set_overflow(c, overflow);
        }

        // Special cases: multiply by 0 or 1
        if let Some(val) = self.get_constant(b) {
            if val == 0 {
                self.set_constant(c, 0, width);
            } else if val == 1 {
                // c = a * 1 = a
                if let Some(va) = self.get_constant(a) {
                    self.set_constant(c, va, width);
                }
            }
        }

        Ok(())
    }

    /// Propagate bitwise AND: c = a & b
    pub fn propagate_and(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // Constants
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            self.set_constant(c, va & vb, width);
            return Ok(());
        }

        // Intervals
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ic = Interval::propagate_and(ia, ib);
            self.set_interval(c, ic);
        }

        // Special case: AND with 0
        if self.get_constant(a) == Some(0) || self.get_constant(b) == Some(0) {
            self.set_constant(c, 0, width);
        }

        // Sign propagation: AND of two non-negatives is non-negative
        if self.get_sign(a).is_non_negative() && self.get_sign(b).is_non_negative() {
            self.set_sign(c, SignInfo::NonNegative);
        }

        Ok(())
    }

    /// Propagate bitwise OR: c = a | b
    pub fn propagate_or(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // Constants
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            self.set_constant(c, va | vb, width);
            return Ok(());
        }

        // Intervals
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ic = Interval::propagate_or(ia, ib);
            self.set_interval(c, ic);
        }

        Ok(())
    }

    /// Propagate bitwise XOR: c = a ^ b
    pub fn propagate_xor(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // Constants
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            self.set_constant(c, va ^ vb, width);
            return Ok(());
        }

        // Intervals
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ic = Interval::propagate_xor(ia, ib);
            self.set_interval(c, ic);
        }

        // XOR with self is 0
        if a == b {
            self.set_constant(c, 0, width);
        }

        Ok(())
    }

    /// Propagate bitwise NOT: c = ~a
    pub fn propagate_not(&mut self, c: TermId, a: TermId, width: u32) -> Result<()> {
        // Constants
        if let Some(va) = self.get_constant(a) {
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            self.set_constant(c, (!va) & mask, width);
            return Ok(());
        }

        // Intervals
        if let Some(ia) = self.get_interval(a) {
            let ic = Interval::propagate_not(ia);
            self.set_interval(c, ic);
        }

        // Sign propagation
        let sign_a = self.get_sign(a);
        self.set_sign(c, sign_a.negate());

        Ok(())
    }

    /// Propagate left shift: c = a << b
    pub fn propagate_shl(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // If shift amount is constant
        if let Some(shift) = self.get_constant(b) {
            if let Some(va) = self.get_constant(a) {
                let result = va << shift;
                let mask = if width == 64 {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                };
                self.set_constant(c, result & mask, width);
                return Ok(());
            }

            // Interval propagation with constant shift
            if let Some(ia) = self.get_interval(a) {
                let shift_interval = Interval::singleton(shift, width);
                let ic = Interval::propagate_shl(ia, &shift_interval);
                self.set_interval(c, ic);
            }
        }

        Ok(())
    }

    /// Propagate logical right shift: c = a >> b
    pub fn propagate_lshr(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // If shift amount is constant
        if let Some(shift) = self.get_constant(b) {
            if let Some(va) = self.get_constant(a) {
                self.set_constant(c, va >> shift, width);
                return Ok(());
            }

            // Interval propagation with constant shift
            if let Some(ia) = self.get_interval(a) {
                let shift_interval = Interval::singleton(shift, width);
                let ic = Interval::propagate_lshr(ia, &shift_interval);
                self.set_interval(c, ic);
            }
        }

        // LSHR always produces non-negative result
        self.set_sign(c, SignInfo::NonNegative);

        Ok(())
    }

    /// Propagate unsigned division: c = a / b
    pub fn propagate_udiv(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // Constants
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            if vb != 0 {
                self.set_constant(c, va / vb, width);
            } else {
                // Division by zero: SMT-LIB semantics = all 1s
                let all_ones = if width == 64 {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                };
                self.set_constant(c, all_ones, width);
            }
            return Ok(());
        }

        // Intervals
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ic = Interval::propagate_udiv(ia, ib);
            self.set_interval(c, ic);
        }

        Ok(())
    }

    /// Propagate unsigned remainder: c = a % b
    pub fn propagate_urem(&mut self, c: TermId, a: TermId, b: TermId, width: u32) -> Result<()> {
        // Constants
        if let (Some(va), Some(vb)) = (self.get_constant(a), self.get_constant(b)) {
            if vb != 0 {
                self.set_constant(c, va % vb, width);
            } else {
                // Remainder by zero: SMT-LIB semantics = a
                self.set_constant(c, va, width);
            }
            return Ok(());
        }

        // Intervals
        if let (Some(ia), Some(ib)) = (self.get_interval(a), self.get_interval(b)) {
            let ic = Interval::propagate_urem(ia, ib);
            self.set_interval(c, ic);
        }

        Ok(())
    }

    /// Propagate extract: c = a\[high:low\]
    pub fn propagate_extract(
        &mut self,
        c: TermId,
        a: TermId,
        high: usize,
        low: usize,
    ) -> Result<()> {
        let extract_width = (high - low + 1) as u32;

        // Constants
        if let Some(va) = self.get_constant(a) {
            let mask = if extract_width == 64 {
                u64::MAX
            } else {
                (1u64 << extract_width) - 1
            };
            let extracted = (va >> low) & mask;
            self.set_constant(c, extracted, extract_width);
            return Ok(());
        }

        // Intervals
        if let Some(ia) = self.get_interval(a) {
            let mask = if extract_width == 64 {
                u64::MAX
            } else {
                (1u64 << extract_width) - 1
            };
            let lower = (ia.lower >> low) & mask;
            let upper = (ia.upper >> low) & mask;
            self.set_interval(c, Interval::new(lower, upper, extract_width));
        }

        Ok(())
    }

    /// Propagate concat: c = high ++ low
    pub fn propagate_concat(
        &mut self,
        c: TermId,
        high: TermId,
        low: TermId,
        high_width: u32,
        low_width: u32,
    ) -> Result<()> {
        let total_width = high_width + low_width;

        // Constants
        if let (Some(vh), Some(vl)) = (self.get_constant(high), self.get_constant(low)) {
            let result = (vh << low_width) | vl;
            self.set_constant(c, result, total_width);
            return Ok(());
        }

        // Intervals
        if let (Some(ih), Some(il)) = (self.get_interval(high), self.get_interval(low)) {
            let lower = (ih.lower << low_width) | il.lower;
            let upper = (ih.upper << low_width) | il.upper;
            self.set_interval(c, Interval::new(lower, upper, total_width));
        }

        Ok(())
    }

    /// Run full propagation to fixpoint
    pub fn propagate_to_fixpoint(&mut self, max_iterations: usize) -> Result<bool> {
        let mut iterations = 0;

        while !self.propagation_queue.is_empty() && iterations < max_iterations {
            iterations += 1;

            // Process all pending terms
            while let Some(term) = self.propagation_queue.pop_front() {
                // Update sign from interval if available - clone to avoid borrow checker issues
                if let Some(interval) = self.get_interval(term) {
                    let interval = interval.clone();
                    let inferred_sign = self.infer_sign_from_interval(&interval);
                    self.set_sign(term, inferred_sign);

                    // Update width info
                    let width_info = self.detect_effective_width(&interval);
                    self.set_width_info(term, width_info);
                }
            }
        }

        Ok(iterations < max_iterations)
    }

    /// Check for word-level conflicts
    #[must_use]
    pub fn has_conflict(&self) -> bool {
        self.stats.word_level_conflicts > 0
    }

    /// Extract all detected equalities
    #[must_use]
    pub fn get_equalities(&self) -> &[(TermId, TermId)] {
        &self.equalities
    }

    /// Get all terms with known constants
    #[must_use]
    pub fn get_constants(&self) -> &FxHashMap<TermId, u64> {
        &self.constants
    }

    /// Simplify based on word-level analysis
    pub fn simplify_term(&self, term: TermId) -> Option<SimplifiedTerm> {
        // If term is constant, return constant
        if let Some(value) = self.get_constant(term) {
            return Some(SimplifiedTerm::Constant(value));
        }

        // If term has narrow effective width, return narrowed term
        if let Some(width_info) = self.get_width_info(term)
            && width_info.can_reduce()
        {
            return Some(SimplifiedTerm::Narrowed {
                original: term,
                new_width: width_info.effective_width,
                is_zero_extended: width_info.is_zero_extended,
                is_sign_extended: width_info.is_sign_extended,
            });
        }

        // If term is equal to another term, return canonical representative
        let canonical = self.eq_classes.find(term);
        if canonical != term {
            return Some(SimplifiedTerm::EqualTo(canonical));
        }

        None
    }
}

/// Simplified term result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimplifiedTerm {
    /// Term is a constant
    Constant(u64),
    /// Term is equal to another term
    EqualTo(TermId),
    /// Term can be narrowed to smaller width
    Narrowed {
        original: TermId,
        new_width: usize,
        is_zero_extended: bool,
        is_sign_extended: bool,
    },
}

/// Union-Find data structure for equivalence classes
#[derive(Debug, Clone)]
struct UnionFind {
    parent: FxHashMap<TermId, TermId>,
    rank: FxHashMap<TermId, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
        }
    }

    fn find(&self, x: TermId) -> TermId {
        match self.parent.get(&x) {
            None => x,
            Some(&p) if p == x => x,
            Some(&p) => self.find(p),
        }
    }

    fn union(&mut self, x: TermId, y: TermId) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if x_root == y_root {
            return;
        }

        let x_rank = *self.rank.get(&x_root).unwrap_or(&0);
        let y_rank = *self.rank.get(&y_root).unwrap_or(&0);

        if x_rank < y_rank {
            self.parent.insert(x_root, y_root);
        } else if x_rank > y_rank {
            self.parent.insert(y_root, x_root);
        } else {
            self.parent.insert(y_root, x_root);
            self.rank.insert(x_root, x_rank + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_info_merge() {
        assert_eq!(
            SignInfo::NonNegative.merge(SignInfo::NonNegative),
            SignInfo::NonNegative
        );
        assert_eq!(
            SignInfo::NonNegative.merge(SignInfo::Negative),
            SignInfo::Unknown
        );
        assert_eq!(
            SignInfo::Negative.merge(SignInfo::Negative),
            SignInfo::Negative
        );
    }

    #[test]
    fn test_sign_info_negate() {
        assert_eq!(SignInfo::NonNegative.negate(), SignInfo::Negative);
        assert_eq!(SignInfo::Negative.negate(), SignInfo::NonNegative);
        assert_eq!(SignInfo::Unknown.negate(), SignInfo::Unknown);
    }

    #[test]
    fn test_width_info() {
        let info = WidthInfo::full(32);
        assert!(!info.can_reduce());

        let info2 = WidthInfo {
            original_width: 32,
            effective_width: 16,
            is_zero_extended: true,
            is_sign_extended: false,
        };
        assert!(info2.can_reduce());
        assert_eq!(info2.reduction_amount(), 16);
    }

    #[test]
    fn test_word_level_reasoner_constants() {
        let mut reasoner = WordLevelReasoner::new();

        let a = TermId::new(1);
        reasoner.set_constant(a, 42, 8);

        assert_eq!(reasoner.get_constant(a), Some(42));
        assert_eq!(reasoner.stats().constants_propagated, 1);
    }

    #[test]
    fn test_propagate_add_constants() {
        let mut reasoner = WordLevelReasoner::new();

        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        reasoner.set_constant(a, 5, 8);
        reasoner.set_constant(b, 3, 8);

        reasoner.propagate_add(c, a, b, 8).unwrap();

        assert_eq!(reasoner.get_constant(c), Some(8));
    }

    #[test]
    fn test_propagate_mul_by_zero() {
        let mut reasoner = WordLevelReasoner::new();

        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        reasoner.set_constant(b, 0, 8);

        reasoner.propagate_mul(c, a, b, 8).unwrap();

        assert_eq!(reasoner.get_constant(c), Some(0));
    }

    #[test]
    fn test_equality_propagation() {
        let mut reasoner = WordLevelReasoner::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        reasoner.assert_equal(a, b);

        assert!(reasoner.are_equal(a, b));
        assert_eq!(reasoner.stats().equalities_detected, 1);
    }

    #[test]
    fn test_interval_refinement() {
        let mut reasoner = WordLevelReasoner::new();

        let a = TermId::new(1);

        reasoner.set_interval(a, Interval::new(0, 100, 8));
        reasoner.set_interval(a, Interval::new(50, 150, 8));

        // Should be refined to [50, 100]
        if let Some(interval) = reasoner.get_interval(a) {
            assert_eq!(interval.lower, 50);
            assert_eq!(interval.upper, 100);
        }
    }

    #[test]
    fn test_overflow_detection_add() {
        let reasoner = WordLevelReasoner::new();

        let a = Interval::new(200, 250, 8);
        let b = Interval::new(50, 100, 8);

        let overflow = reasoner.detect_add_overflow(&a, &b);
        assert_ne!(overflow, OverflowInfo::NoOverflow);
    }

    #[test]
    fn test_sign_inference_from_interval() {
        let reasoner = WordLevelReasoner::new();

        // Non-negative interval
        let i1 = Interval::new(0, 127, 8);
        assert_eq!(
            reasoner.infer_sign_from_interval(&i1),
            SignInfo::NonNegative
        );

        // Negative interval
        let i2 = Interval::new(128, 255, 8);
        assert_eq!(reasoner.infer_sign_from_interval(&i2), SignInfo::Negative);

        // Mixed interval
        let i3 = Interval::new(100, 200, 8);
        assert_eq!(reasoner.infer_sign_from_interval(&i3), SignInfo::Unknown);
    }

    #[test]
    fn test_effective_width_detection() {
        let reasoner = WordLevelReasoner::new();

        // Value fits in 4 bits
        let i = Interval::new(0, 15, 32);
        let width_info = reasoner.detect_effective_width(&i);

        assert_eq!(width_info.effective_width, 4);
        assert!(width_info.is_zero_extended);
    }

    #[test]
    fn test_xor_self_simplification() {
        let mut reasoner = WordLevelReasoner::new();

        let a = TermId::new(1);
        let c = TermId::new(2);

        reasoner.propagate_xor(c, a, a, 8).unwrap();

        assert_eq!(reasoner.get_constant(c), Some(0));
    }

    #[test]
    fn test_lshr_sign_propagation() {
        let mut reasoner = WordLevelReasoner::new();

        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        reasoner.set_constant(b, 1, 8);

        reasoner.propagate_lshr(c, a, b, 8).unwrap();

        assert_eq!(reasoner.get_sign(c), SignInfo::NonNegative);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new();

        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        assert_eq!(uf.find(a), a);
        assert_eq!(uf.find(b), b);

        uf.union(a, b);
        assert_eq!(uf.find(a), uf.find(b));

        uf.union(b, c);
        assert_eq!(uf.find(a), uf.find(c));
    }
}
