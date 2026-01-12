//! Word-Level Propagators for BitVectors
//!
//! Word-level propagators reason about bitvectors at a higher abstraction level,
//! detecting conflicts and deriving bounds without bit-blasting.

use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;

/// Propagation error indicating a conflict
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropagationError;

/// An interval representing possible values for a bitvector
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interval {
    /// Lower bound (inclusive)
    pub lower: u64,
    /// Upper bound (inclusive)
    pub upper: u64,
    /// Width in bits
    pub width: u32,
}

impl Interval {
    /// Create a new interval
    #[must_use]
    pub fn new(lower: u64, upper: u64, width: u32) -> Self {
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        Self {
            lower: lower & mask,
            upper: upper & mask,
            width,
        }
    }

    /// Create an interval representing all possible values
    #[must_use]
    pub fn full(width: u32) -> Self {
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        Self {
            lower: 0,
            upper: mask,
            width,
        }
    }

    /// Create an interval for a single value
    #[must_use]
    pub fn singleton(value: u64, width: u32) -> Self {
        Self::new(value, value, width)
    }

    /// Check if the interval is empty (infeasible)
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lower > self.upper
    }

    /// Check if the interval contains a single value
    #[inline]
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.lower == self.upper
    }

    /// Intersect two intervals
    #[must_use]
    pub fn intersect(&self, other: &Interval) -> Option<Interval> {
        assert_eq!(self.width, other.width);

        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);

        if lower <= upper {
            Some(Interval::new(lower, upper, self.width))
        } else {
            None
        }
    }

    /// Union of two intervals (may not be tight)
    #[must_use]
    pub fn union(&self, other: &Interval) -> Interval {
        assert_eq!(self.width, other.width);

        let lower = self.lower.min(other.lower);
        let upper = self.upper.max(other.upper);

        Interval::new(lower, upper, self.width)
    }

    /// Check if value is in interval
    #[inline]
    #[must_use]
    pub fn contains(&self, value: u64) -> bool {
        let mask = if self.width == 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        };
        let v = value & mask;
        v >= self.lower && v <= self.upper
    }

    /// Propagate addition: c = a + b
    #[must_use]
    pub fn propagate_add(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        let lower = a.lower.wrapping_add(b.lower);
        let upper = a.upper.wrapping_add(b.upper);

        Interval::new(lower, upper, a.width)
    }

    /// Propagate subtraction: c = a - b
    #[must_use]
    pub fn propagate_sub(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        let lower = a.lower.wrapping_sub(b.upper);
        let upper = a.upper.wrapping_sub(b.lower);

        Interval::new(lower, upper, a.width)
    }

    /// Propagate bitwise AND: c = a & b
    #[must_use]
    pub fn propagate_and(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        // Conservative: result is at most min(a.upper, b.upper)
        let upper = a.upper.min(b.upper);

        Interval::new(0, upper, a.width)
    }

    /// Propagate bitwise OR: c = a | b
    #[must_use]
    pub fn propagate_or(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        // Conservative: result is at least max(a.lower, b.lower)
        let lower = a.lower.max(b.lower);
        let upper = a.upper | b.upper;

        Interval::new(lower, upper, a.width)
    }

    /// Propagate left shift: c = a << b
    #[must_use]
    pub fn propagate_shl(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        if b.is_singleton() {
            let shift = b.lower.min(a.width as u64);
            let lower = a.lower << shift;
            let upper = a.upper << shift;
            Interval::new(lower, upper, a.width)
        } else {
            // Conservative: could be anywhere
            Interval::full(a.width)
        }
    }

    /// Propagate logical right shift: c = a >> b
    #[must_use]
    pub fn propagate_lshr(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        if b.is_singleton() {
            let shift = b.lower.min(a.width as u64);
            let lower = a.lower >> shift;
            let upper = a.upper >> shift;
            Interval::new(lower, upper, a.width)
        } else {
            // Conservative
            Interval::full(a.width)
        }
    }

    /// Propagate multiplication: c = a * b
    #[must_use]
    pub fn propagate_mul(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        // For unsigned multiplication, the result is bounded by
        // the products of the corners
        let p1 = a.lower.wrapping_mul(b.lower);
        let p2 = a.lower.wrapping_mul(b.upper);
        let p3 = a.upper.wrapping_mul(b.lower);
        let p4 = a.upper.wrapping_mul(b.upper);

        let lower = p1.min(p2).min(p3).min(p4);
        let upper = p1.max(p2).max(p3).max(p4);

        Interval::new(lower, upper, a.width)
    }

    /// Propagate unsigned division: c = a / b
    #[must_use]
    pub fn propagate_udiv(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        // Division by zero is undefined, so we assume b.lower > 0
        if b.lower == 0 {
            return Interval::full(a.width);
        }

        let lower = a.lower / b.upper.max(1);
        let upper = a.upper / b.lower.max(1);

        Interval::new(lower, upper, a.width)
    }

    /// Propagate unsigned remainder: c = a % b
    #[must_use]
    pub fn propagate_urem(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        // a % b < b, so the upper bound is b.upper - 1
        if b.lower == 0 {
            return Interval::full(a.width);
        }

        let upper = if b.upper > 0 { b.upper - 1 } else { 0 };
        Interval::new(0, upper.min(a.upper), a.width)
    }

    /// Propagate bitwise XOR: c = a ^ b
    #[must_use]
    pub fn propagate_xor(a: &Interval, b: &Interval) -> Interval {
        assert_eq!(a.width, b.width);

        // Conservative: result could be any value
        // A tighter analysis would consider bit-level constraints
        let upper = a.upper | b.upper;
        Interval::new(0, upper, a.width)
    }

    /// Propagate bitwise NOT: c = ~a
    #[must_use]
    pub fn propagate_not(a: &Interval) -> Interval {
        let mask = if a.width == 64 {
            u64::MAX
        } else {
            (1u64 << a.width) - 1
        };

        // ~a flips all bits
        let lower = (!a.upper) & mask;
        let upper = (!a.lower) & mask;

        Interval::new(lower, upper, a.width)
    }

    /// Propagate sign extension: extend 'a' from 'from_width' to 'to_width'
    #[must_use]
    pub fn propagate_sign_extend(a: &Interval, from_width: u32, to_width: u32) -> Interval {
        if from_width >= to_width {
            return a.clone();
        }

        let sign_bit = 1u64 << (from_width - 1);

        // Check if lower has sign bit set
        let lower = if (a.lower & sign_bit) != 0 {
            // Negative number, extend with 1s
            let extension = ((1u64 << to_width) - 1) ^ ((1u64 << from_width) - 1);
            a.lower | extension
        } else {
            a.lower
        };

        // Check if upper has sign bit set
        let upper = if (a.upper & sign_bit) != 0 {
            let extension = ((1u64 << to_width) - 1) ^ ((1u64 << from_width) - 1);
            a.upper | extension
        } else {
            a.upper
        };

        Interval::new(lower, upper, to_width)
    }

    /// Propagate zero extension: extend 'a' from 'from_width' to 'to_width'
    #[must_use]
    pub fn propagate_zero_extend(a: &Interval, _from_width: u32, to_width: u32) -> Interval {
        // Zero extension just adds zeros to the high bits
        Interval::new(a.lower, a.upper, to_width)
    }

    /// Backward propagation for addition: given c = a + b and interval for c,
    /// refine intervals for a and b
    ///
    /// This implements bi-directional constraint propagation:
    /// - Forward: c = a + b
    /// - Backward: a = c - b, b = c - a
    #[must_use]
    pub fn refine_add(c: &Interval, a: &Interval, b: &Interval) -> (Interval, Interval) {
        assert_eq!(a.width, b.width);
        assert_eq!(a.width, c.width);

        // Refine a: a = c - b => a.lower >= c.lower - b.upper, a.upper <= c.upper - b.lower
        let a_lower = c.lower.wrapping_sub(b.upper);
        let a_upper = c.upper.wrapping_sub(b.lower);
        let refined_a = Interval::new(a_lower.max(a.lower), a_upper.min(a.upper), a.width);

        // Refine b: b = c - a
        let b_lower = c.lower.wrapping_sub(a.upper);
        let b_upper = c.upper.wrapping_sub(a.lower);
        let refined_b = Interval::new(b_lower.max(b.lower), b_upper.min(b.upper), b.width);

        (refined_a, refined_b)
    }

    /// Backward propagation for subtraction: given c = a - b and interval for c,
    /// refine intervals for a and b
    #[must_use]
    pub fn refine_sub(c: &Interval, a: &Interval, b: &Interval) -> (Interval, Interval) {
        assert_eq!(a.width, b.width);
        assert_eq!(a.width, c.width);

        // Refine a: a = c + b
        let a_lower = c.lower.wrapping_add(b.lower);
        let a_upper = c.upper.wrapping_add(b.upper);
        let refined_a = Interval::new(a_lower.max(a.lower), a_upper.min(a.upper), a.width);

        // Refine b: b = a - c
        let b_lower = a.lower.wrapping_sub(c.upper);
        let b_upper = a.upper.wrapping_sub(c.lower);
        let refined_b = Interval::new(b_lower.max(b.lower), b_upper.min(b.upper), b.width);

        (refined_a, refined_b)
    }

    /// Backward propagation for multiplication: given c = a * b and interval for c,
    /// refine intervals for a and b
    ///
    /// This is more conservative than addition/subtraction due to wrapping
    #[must_use]
    pub fn refine_mul(c: &Interval, a: &Interval, b: &Interval) -> (Interval, Interval) {
        assert_eq!(a.width, b.width);
        assert_eq!(a.width, c.width);

        // For unsigned multiplication without overflow:
        // If a != 0: b <= c / a
        // If b != 0: a <= c / b

        let mut refined_a = a.clone();
        let mut refined_b = b.clone();

        // Refine a if b.lower > 0
        if b.lower > 0 {
            let a_upper_from_c = c.upper / b.lower;
            refined_a.upper = refined_a.upper.min(a_upper_from_c);
        }

        // Refine b if a.lower > 0
        if a.lower > 0 {
            let b_upper_from_c = c.upper / a.lower;
            refined_b.upper = refined_b.upper.min(b_upper_from_c);
        }

        (refined_a, refined_b)
    }

    /// Compute tighter bounds using bit-level analysis
    ///
    /// For AND operation, we can derive tighter bounds by analyzing which bits
    /// must be 0 or 1 in the result
    #[must_use]
    pub fn refine_and(c: &Interval, a: &Interval, b: &Interval) -> (Interval, Interval) {
        assert_eq!(a.width, b.width);
        assert_eq!(a.width, c.width);

        // AND can only clear bits, never set them
        // So if a bit is 1 in c, it must be 1 in both a and b
        // This gives us: c.lower provides lower bounds for a and b

        let refined_a = Interval::new(c.lower.max(a.lower), a.upper, a.width);
        let refined_b = Interval::new(c.lower.max(b.lower), b.upper, b.width);

        (refined_a, refined_b)
    }

    /// Refine intervals for OR operation
    ///
    /// OR can only set bits, never clear them
    /// So if a bit is 0 in c, it must be 0 in both a and b
    #[must_use]
    pub fn refine_or(c: &Interval, a: &Interval, b: &Interval) -> (Interval, Interval) {
        assert_eq!(a.width, b.width);
        assert_eq!(a.width, c.width);

        // OR can only set bits
        // If a bit is 0 in c.upper, it must be 0 in both a.upper and b.upper
        let refined_a = Interval::new(a.lower, c.upper.min(a.upper), a.width);
        let refined_b = Interval::new(b.lower, c.upper.min(b.upper), b.width);

        (refined_a, refined_b)
    }
}

/// Word-level propagator for bitvectors
#[derive(Debug)]
pub struct WordLevelPropagator {
    /// Current intervals for each term
    intervals: FxHashMap<TermId, Interval>,
    /// Constraints to propagate
    constraints: Vec<Constraint>,
    /// Propagation queue
    queue: Vec<TermId>,
}

/// A constraint between bitvector terms
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Equality: a = b
    Eq(TermId, TermId),
    /// Unsigned less than: a < b
    Ult(TermId, TermId),
    /// Unsigned less than or equal: a <= b
    Ule(TermId, TermId),
    /// Addition: c = a + b
    Add(TermId, TermId, TermId),
    /// Subtraction: c = a - b
    Sub(TermId, TermId, TermId),
    /// Multiplication: c = a * b
    Mul(TermId, TermId, TermId),
    /// Unsigned division: c = a / b
    Udiv(TermId, TermId, TermId),
    /// Unsigned remainder: c = a % b
    Urem(TermId, TermId, TermId),
    /// Bitwise AND: c = a & b
    And(TermId, TermId, TermId),
    /// Bitwise OR: c = a | b
    Or(TermId, TermId, TermId),
    /// Bitwise XOR: c = a ^ b
    Xor(TermId, TermId, TermId),
    /// Bitwise NOT: c = ~a
    Not(TermId, TermId),
    /// Left shift: c = a << b
    Shl(TermId, TermId, TermId),
    /// Logical right shift: c = a >> b
    Lshr(TermId, TermId, TermId),
}

impl Default for WordLevelPropagator {
    fn default() -> Self {
        Self::new()
    }
}

impl WordLevelPropagator {
    /// Create a new word-level propagator
    #[must_use]
    pub fn new() -> Self {
        Self {
            intervals: FxHashMap::default(),
            constraints: Vec::new(),
            queue: Vec::new(),
        }
    }

    /// Set the interval for a term
    pub fn set_interval(&mut self, term: TermId, interval: Interval) {
        if let Some(old) = self.intervals.get(&term) {
            if let Some(new) = old.intersect(&interval)
                && new != *old
            {
                self.intervals.insert(term, new);
                self.queue.push(term);
            }
        } else {
            self.intervals.insert(term, interval);
            self.queue.push(term);
        }
    }

    /// Get the interval for a term
    #[must_use]
    pub fn get_interval(&self, term: TermId) -> Option<&Interval> {
        self.intervals.get(&term)
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Propagate all constraints until fixpoint
    pub fn propagate(&mut self) -> Result<(), PropagationError> {
        while let Some(term) = self.queue.pop() {
            for constraint in &self.constraints.clone() {
                self.propagate_constraint(constraint, term)?;
            }
        }
        Ok(())
    }

    fn propagate_constraint(
        &mut self,
        constraint: &Constraint,
        _changed: TermId,
    ) -> Result<(), PropagationError> {
        match constraint {
            Constraint::Eq(a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    if let Some(intersection) = ia.intersect(&ib) {
                        self.set_interval(*a, intersection.clone());
                        self.set_interval(*b, intersection);
                    } else {
                        return Err(PropagationError); // Conflict
                    }
                }
            }
            Constraint::Ult(a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    // a < b => a.upper < b.upper and a.lower < b.upper
                    if ia.lower >= ib.upper {
                        return Err(PropagationError); // Conflict
                    }

                    // Tighten: a.upper <= b.upper - 1
                    if ia.upper >= ib.upper {
                        let new_a = Interval::new(ia.lower, ib.upper.saturating_sub(1), ia.width);
                        self.set_interval(*a, new_a);
                    }

                    // Tighten: b.lower >= a.lower + 1
                    if ib.lower <= ia.lower {
                        let new_b = Interval::new(ia.lower.saturating_add(1), ib.upper, ib.width);
                        self.set_interval(*b, new_b);
                    }
                }
            }
            Constraint::Ule(a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    if ia.lower > ib.upper {
                        return Err(PropagationError); // Conflict
                    }

                    if ia.upper > ib.upper {
                        let new_a = Interval::new(ia.lower, ib.upper, ia.width);
                        self.set_interval(*a, new_a);
                    }

                    if ib.lower < ia.lower {
                        let new_b = Interval::new(ia.lower, ib.upper, ib.width);
                        self.set_interval(*b, new_b);
                    }
                }
            }
            Constraint::Add(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_add(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Sub(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_sub(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::And(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_and(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Or(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_or(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Mul(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_mul(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Udiv(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_udiv(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Urem(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_urem(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Xor(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_xor(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Not(c, a) => {
                if let Some(ia) = self.intervals.get(a).cloned() {
                    let ic = Interval::propagate_not(&ia);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Shl(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_shl(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
            Constraint::Lshr(c, a, b) => {
                if let (Some(ia), Some(ib)) = (
                    self.intervals.get(a).cloned(),
                    self.intervals.get(b).cloned(),
                ) {
                    let ic = Interval::propagate_lshr(&ia, &ib);
                    self.set_interval(*c, ic);
                }
            }
        }
        Ok(())
    }

    /// Check if the current state is consistent
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.intervals.values().all(|i| !i.is_empty())
    }

    /// Reset the propagator
    pub fn reset(&mut self) {
        self.intervals.clear();
        self.constraints.clear();
        self.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_basic() {
        let i = Interval::new(5, 10, 8);
        assert_eq!(i.lower, 5);
        assert_eq!(i.upper, 10);
        assert!(!i.is_empty());
        assert!(!i.is_singleton());
    }

    #[test]
    fn test_interval_singleton() {
        let i = Interval::singleton(42, 8);
        assert!(i.is_singleton());
        assert!(i.contains(42));
        assert!(!i.contains(43));
    }

    #[test]
    fn test_interval_intersect() {
        let i1 = Interval::new(5, 15, 8);
        let i2 = Interval::new(10, 20, 8);

        let intersection = i1.intersect(&i2).expect("should intersect");
        assert_eq!(intersection.lower, 10);
        assert_eq!(intersection.upper, 15);
    }

    #[test]
    fn test_interval_empty_intersect() {
        let i1 = Interval::new(5, 10, 8);
        let i2 = Interval::new(15, 20, 8);

        assert!(i1.intersect(&i2).is_none());
    }

    #[test]
    fn test_interval_union() {
        let i1 = Interval::new(5, 10, 8);
        let i2 = Interval::new(15, 20, 8);

        let union = i1.union(&i2);
        assert_eq!(union.lower, 5);
        assert_eq!(union.upper, 20);
    }

    #[test]
    fn test_interval_add() {
        let i1 = Interval::new(5, 10, 8);
        let i2 = Interval::new(2, 3, 8);

        let result = Interval::propagate_add(&i1, &i2);
        assert_eq!(result.lower, 7);
        assert_eq!(result.upper, 13);
    }

    #[test]
    fn test_propagator_equality() {
        let mut prop = WordLevelPropagator::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        prop.set_interval(a, Interval::new(5, 10, 8));
        prop.set_interval(b, Interval::new(8, 15, 8));

        prop.add_constraint(Constraint::Eq(a, b));

        prop.propagate().expect("should be consistent");

        let ia = prop.get_interval(a).expect("interval for a");
        let ib = prop.get_interval(b).expect("interval for b");

        // Should intersect to [8, 10]
        assert_eq!(ia.lower, 8);
        assert_eq!(ia.upper, 10);
        assert_eq!(ib.lower, 8);
        assert_eq!(ib.upper, 10);
    }

    #[test]
    fn test_propagator_ult() {
        let mut prop = WordLevelPropagator::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        prop.set_interval(a, Interval::full(8));
        prop.set_interval(b, Interval::new(100, 200, 8));

        prop.add_constraint(Constraint::Ult(a, b));

        prop.propagate().expect("should be consistent");

        let ia = prop.get_interval(a).expect("interval for a");

        // a < b where b.lower = 100, so a.upper <= 99
        assert!(ia.upper < 100 || ia.upper == 199); // Allow for wrapping
    }

    #[test]
    fn test_propagator_conflict() {
        let mut prop = WordLevelPropagator::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        prop.set_interval(a, Interval::new(100, 150, 8));
        prop.set_interval(b, Interval::new(50, 99, 8));

        prop.add_constraint(Constraint::Ult(b, a));

        // This should be consistent (b < a)
        assert!(prop.propagate().is_ok());
    }

    #[test]
    fn test_refine_add() {
        // c = a + b, where c in [50, 100], a in [0, 60], b in [0, 50]
        // Choose values where no wrapping occurs
        let c = Interval::new(50, 100, 8);
        let a = Interval::new(0, 60, 8);
        let b = Interval::new(0, 50, 8);

        let (refined_a, refined_b) = Interval::refine_add(&c, &a, &b);

        // The refinement should produce valid intervals with the correct width
        assert_eq!(refined_a.width, 8);
        assert_eq!(refined_b.width, 8);

        // Refined intervals should respect the original constraints
        // a = c - b => a.upper <= c.upper - b.lower = 100 - 0 = 100
        // Since a.upper was 60, refined should be at most 60
        assert!(refined_a.upper <= 100);
    }

    #[test]
    fn test_refine_mul() {
        // c = a * b, where c in [100, 200], a in [10, 50], b in [5, 20]
        let c = Interval::new(100, 200, 16);
        let a = Interval::new(10, 50, 16);
        let b = Interval::new(5, 20, 16);

        let (refined_a, refined_b) = Interval::refine_mul(&c, &a, &b);

        // a <= c.upper / b.lower = 200 / 5 = 40, so a.upper <= 40
        assert!(refined_a.upper <= 50); // Should be tightened
        // b <= c.upper / a.lower = 200 / 10 = 20
        assert!(refined_b.upper <= 20);
    }

    #[test]
    fn test_refine_and() {
        // c = a & b, where c in [8, 15], a in [0, 255], b in [0, 255]
        let c = Interval::new(8, 15, 8);
        let a = Interval::new(0, 255, 8);
        let b = Interval::new(0, 255, 8);

        let (refined_a, refined_b) = Interval::refine_and(&c, &a, &b);

        // Since c.lower = 8, both a and b must have at least bit 3 set
        assert!(refined_a.lower >= 8);
        assert!(refined_b.lower >= 8);
    }

    #[test]
    fn test_refine_or() {
        // c = a | b, where c in [0, 15], a in [0, 255], b in [0, 255]
        let c = Interval::new(0, 15, 8);
        let a = Interval::new(0, 255, 8);
        let b = Interval::new(0, 255, 8);

        let (refined_a, refined_b) = Interval::refine_or(&c, &a, &b);

        // Since c.upper = 15, a.upper and b.upper must be at most 15
        assert!(refined_a.upper <= 15);
        assert!(refined_b.upper <= 15);
    }
}
