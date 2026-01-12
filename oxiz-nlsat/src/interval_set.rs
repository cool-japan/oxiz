//! Interval sets for NLSAT.
//!
//! This module provides interval set representation for tracking feasible
//! regions in the NLSAT solver. An interval set is a union of disjoint
//! intervals, used to represent the possible values for a variable.
//!
//! Reference: Z3's `nlsat/nlsat_interval_set.cpp`

use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::interval::{Bound, Interval};
use std::fmt;

/// Kind of polynomial constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// Equality: p = 0
    Eq,
    /// Less than: p < 0
    Lt,
    /// Greater than: p > 0
    Gt,
    /// Less or equal: p <= 0
    Le,
    /// Greater or equal: p >= 0
    Ge,
    /// Not equal: p != 0
    Ne,
}

/// An interval set is a union of disjoint intervals.
#[derive(Clone, Debug)]
pub struct IntervalSet {
    /// Disjoint intervals in sorted order (by lower bound).
    intervals: Vec<Interval>,
}

impl IntervalSet {
    /// Create an empty interval set.
    #[inline]
    pub fn empty() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// Create the full interval set (-∞, +∞).
    pub fn reals() -> Self {
        Self {
            intervals: vec![Interval::reals()],
        }
    }

    /// Create a singleton interval set containing just one point.
    pub fn point(a: BigRational) -> Self {
        Self {
            intervals: vec![Interval::point(a)],
        }
    }

    /// Alias for `point()`.
    #[inline]
    pub fn from_point(a: BigRational) -> Self {
        Self::point(a)
    }

    /// Create interval set for x < a: (-∞, a).
    pub fn lt(a: BigRational) -> Self {
        Self {
            intervals: vec![Interval::less_than(a)],
        }
    }

    /// Create interval set for x <= a: (-∞, a].
    pub fn le(a: BigRational) -> Self {
        Self {
            intervals: vec![Interval::at_most(a)],
        }
    }

    /// Create interval set for x > a: (a, +∞).
    pub fn gt(a: BigRational) -> Self {
        Self {
            intervals: vec![Interval::greater_than(a)],
        }
    }

    /// Create interval set for x >= a: [a, +∞).
    pub fn ge(a: BigRational) -> Self {
        Self {
            intervals: vec![Interval::at_least(a)],
        }
    }

    /// Create from a single interval.
    pub fn from_interval(interval: Interval) -> Self {
        if interval.is_empty() {
            Self::empty()
        } else {
            Self {
                intervals: vec![interval],
            }
        }
    }

    /// Create from multiple intervals (normalizes them).
    pub fn from_intervals(intervals: impl IntoIterator<Item = Interval>) -> Self {
        let mut set = Self::empty();
        for interval in intervals {
            set = set.union(&Self::from_interval(interval));
        }
        set
    }

    /// Check if the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Check if the set is the full real line.
    pub fn is_reals(&self) -> bool {
        self.intervals.len() == 1
            && self.intervals[0].lo.is_neg_inf()
            && self.intervals[0].hi.is_pos_inf()
    }

    /// Check if the set contains a single point.
    pub fn is_singleton(&self) -> bool {
        self.intervals.len() == 1 && self.intervals[0].is_point()
    }

    /// Get the number of intervals.
    #[inline]
    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }

    /// Get the intervals.
    #[inline]
    pub fn intervals(&self) -> &[Interval] {
        &self.intervals
    }

    /// Check if the set contains a value.
    pub fn contains(&self, x: &BigRational) -> bool {
        self.intervals.iter().any(|i| i.contains(x))
    }

    /// Check if the set contains zero.
    pub fn contains_zero(&self) -> bool {
        self.contains(&BigRational::zero())
    }

    /// Get the lower bound of the set.
    pub fn lower_bound(&self) -> Option<&Bound> {
        self.intervals.first().map(|i| &i.lo)
    }

    /// Get the upper bound of the set.
    pub fn upper_bound(&self) -> Option<&Bound> {
        self.intervals.last().map(|i| &i.hi)
    }

    /// Compute the union of two interval sets.
    pub fn union(&self, other: &IntervalSet) -> IntervalSet {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        // Merge the two sorted lists
        let mut result = Vec::with_capacity(self.intervals.len() + other.intervals.len());
        let mut i = 0;
        let mut j = 0;

        while i < self.intervals.len() && j < other.intervals.len() {
            if self.intervals[i].lo <= other.intervals[j].lo {
                result.push(self.intervals[i].clone());
                i += 1;
            } else {
                result.push(other.intervals[j].clone());
                j += 1;
            }
        }
        result.extend_from_slice(&self.intervals[i..]);
        result.extend_from_slice(&other.intervals[j..]);

        // Merge overlapping intervals
        let merged = merge_intervals(result);
        IntervalSet { intervals: merged }
    }

    /// Compute the intersection of two interval sets.
    pub fn intersect(&self, other: &IntervalSet) -> IntervalSet {
        if self.is_empty() || other.is_empty() {
            return IntervalSet::empty();
        }

        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.intervals.len() && j < other.intervals.len() {
            let intersection = self.intervals[i].intersect(&other.intervals[j]);
            if !intersection.is_empty() {
                result.push(intersection);
            }

            // Advance the iterator with the smaller upper bound
            if self.intervals[i].hi <= other.intervals[j].hi {
                i += 1;
            } else {
                j += 1;
            }
        }

        IntervalSet { intervals: result }
    }

    /// Compute the complement of the interval set.
    pub fn complement(&self) -> IntervalSet {
        if self.is_empty() {
            return IntervalSet::reals();
        }

        let mut result = Vec::new();
        let mut prev_hi = Bound::NegInf;
        let mut prev_hi_open = true;

        for interval in &self.intervals {
            // Add gap before this interval
            let gap_hi = interval.lo.clone();
            let gap_hi_open = !interval.lo_open;

            // Create gap from previous interval's end to this interval's start
            let gap = Interval {
                lo: prev_hi.clone(),
                hi: gap_hi,
                lo_open: !prev_hi_open,
                hi_open: gap_hi_open,
            };
            if !gap.is_empty() {
                result.push(gap);
            }

            prev_hi = interval.hi.clone();
            prev_hi_open = interval.hi_open;
        }

        // Add final gap to +∞
        if prev_hi < Bound::PosInf {
            result.push(Interval {
                lo: prev_hi,
                hi: Bound::PosInf,
                lo_open: !prev_hi_open,
                hi_open: true,
            });
        }

        IntervalSet { intervals: result }
    }

    /// Compute the difference: self - other.
    pub fn difference(&self, other: &IntervalSet) -> IntervalSet {
        self.intersect(&other.complement())
    }

    /// Get a sample point from the interval set (if non-empty).
    pub fn sample(&self) -> Option<BigRational> {
        if self.is_empty() {
            return None;
        }

        // Try to find a nice point (preferably an integer)
        for interval in &self.intervals {
            if let Some(mid) = interval.midpoint() {
                // Round to nearest integer if possible
                let floor = mid.floor();
                let ceil = mid.ceil();

                if interval.contains(&floor) {
                    return Some(floor);
                }
                if interval.contains(&ceil) {
                    return Some(ceil);
                }
                return Some(mid);
            }
        }

        // Handle unbounded intervals
        for interval in &self.intervals {
            // If unbounded below, try 0 or a negative integer
            if interval.lo.is_neg_inf()
                && let Bound::Finite(hi) = &interval.hi
            {
                let val = hi.floor() - BigRational::one();
                if interval.contains(&val) {
                    return Some(val);
                }
            }
            // If unbounded above, try 0 or a positive integer
            if interval.hi.is_pos_inf()
                && let Bound::Finite(lo) = &interval.lo
            {
                let val = lo.ceil() + BigRational::one();
                if interval.contains(&val) {
                    return Some(val);
                }
            }
        }

        // Just return 0 if it's in the set
        if self.contains_zero() {
            return Some(BigRational::zero());
        }

        None
    }

    /// Get all finite endpoints in the interval set.
    pub fn endpoints(&self) -> Vec<BigRational> {
        let mut result = Vec::new();
        for interval in &self.intervals {
            if let Bound::Finite(lo) = &interval.lo {
                result.push(lo.clone());
            }
            if let Bound::Finite(hi) = &interval.hi
                && Some(hi) != interval.lo.as_finite()
            {
                result.push(hi.clone());
            }
        }
        result.sort();
        result.dedup();
        result
    }

    /// Intersect with the roots of a polynomial.
    ///
    /// Returns an interval set containing only the points in this set that are roots
    /// of the given polynomial. This is useful for constraint solving.
    pub fn intersect_with_roots(&self, roots: &[BigRational]) -> IntervalSet {
        if self.is_empty() || roots.is_empty() {
            return IntervalSet::empty();
        }

        let mut result_intervals = Vec::new();

        for root in roots {
            if self.contains(root) {
                result_intervals.push(Interval::point(root.clone()));
            }
        }

        IntervalSet {
            intervals: result_intervals,
        }
    }

    /// Filter this interval set by a predicate on polynomial signs.
    ///
    /// Given the roots of a polynomial and the signs between roots,
    /// returns the subset of this interval set where the polynomial has the given sign.
    pub fn filter_by_sign(
        &self,
        roots: &[BigRational],
        signs_between: &[i8],
        target_sign: i8,
    ) -> IntervalSet {
        // Create the sign-based interval set
        let sign_set = IntervalSet::sign_set(roots, signs_between, target_sign);

        // Intersect with our current set
        self.intersect(&sign_set)
    }

    /// Compute the interval set where a polynomial constraint is satisfied.
    ///
    /// Given:
    /// - roots: The roots of the polynomial
    /// - signs: The signs of the polynomial in intervals between roots
    /// - constraint: The kind of constraint (Eq, Lt, Gt)
    ///
    /// Returns the interval set satisfying the constraint.
    pub fn from_constraint(
        roots: &[BigRational],
        signs_between: &[i8],
        constraint_kind: ConstraintKind,
    ) -> IntervalSet {
        match constraint_kind {
            ConstraintKind::Eq => {
                // p = 0: just the roots
                IntervalSet::sign_set(roots, signs_between, 0)
            }
            ConstraintKind::Lt => {
                // p < 0: negative regions
                IntervalSet::sign_set(roots, signs_between, -1)
            }
            ConstraintKind::Gt => {
                // p > 0: positive regions
                IntervalSet::sign_set(roots, signs_between, 1)
            }
            ConstraintKind::Le => {
                // p <= 0: negative regions + roots
                let neg = IntervalSet::sign_set(roots, signs_between, -1);
                let zero = IntervalSet::sign_set(roots, signs_between, 0);
                neg.union(&zero)
            }
            ConstraintKind::Ge => {
                // p >= 0: positive regions + roots
                let pos = IntervalSet::sign_set(roots, signs_between, 1);
                let zero = IntervalSet::sign_set(roots, signs_between, 0);
                pos.union(&zero)
            }
            ConstraintKind::Ne => {
                // p != 0: complement of roots
                let zero = IntervalSet::sign_set(roots, signs_between, 0);
                zero.complement()
            }
        }
    }

    /// Create the interval set where a polynomial has a given sign.
    /// sign: 1 for positive, -1 for negative, 0 for zero
    pub fn sign_set(roots: &[BigRational], signs_between: &[i8], target_sign: i8) -> IntervalSet {
        if roots.is_empty() {
            // No roots, constant sign
            if signs_between.len() == 1 && signs_between[0] == target_sign {
                return IntervalSet::reals();
            }
            return IntervalSet::empty();
        }

        let mut intervals = Vec::new();
        let n = roots.len();

        // Check region before first root
        if !signs_between.is_empty() && signs_between[0] == target_sign {
            intervals.push(Interval::less_than(roots[0].clone()));
        }

        // Check each root
        for (i, root) in roots.iter().enumerate() {
            if target_sign == 0 {
                intervals.push(Interval::point(root.clone()));
            }

            // Check region after this root
            if i + 1 < signs_between.len() && signs_between[i + 1] == target_sign {
                if i + 1 < n {
                    intervals.push(Interval::open(root.clone(), roots[i + 1].clone()));
                } else {
                    intervals.push(Interval::greater_than(root.clone()));
                }
            }
        }

        IntervalSet::from_intervals(intervals)
    }

    /// Restrict to integers (for integer arithmetic).
    /// Returns the interval set of integers within this set.
    pub fn restrict_to_integers(&self) -> IntervalSet {
        let mut result = Vec::new();

        for interval in &self.intervals {
            match (&interval.lo, &interval.hi) {
                (Bound::Finite(lo), Bound::Finite(hi)) => {
                    let lo_int = if interval.lo_open {
                        lo.ceil()
                    } else {
                        lo.floor()
                    };
                    let hi_int = if interval.hi_open {
                        hi.floor()
                    } else {
                        hi.ceil()
                    };

                    if lo_int <= hi_int {
                        result.push(Interval::closed(lo_int, hi_int));
                    }
                }
                (Bound::NegInf, Bound::Finite(hi)) => {
                    let hi_int = if interval.hi_open {
                        hi.floor()
                    } else {
                        hi.ceil()
                    };
                    result.push(Interval::at_most(hi_int));
                }
                (Bound::Finite(lo), Bound::PosInf) => {
                    let lo_int = if interval.lo_open {
                        lo.ceil()
                    } else {
                        lo.floor()
                    };
                    result.push(Interval::at_least(lo_int));
                }
                (Bound::NegInf, Bound::PosInf) => {
                    return IntervalSet::reals();
                }
                _ => {}
            }
        }

        IntervalSet { intervals: result }
    }
}

/// Merge overlapping or adjacent intervals.
fn merge_intervals(mut intervals: Vec<Interval>) -> Vec<Interval> {
    if intervals.is_empty() {
        return intervals;
    }

    // Sort by lower bound
    intervals.sort_by(|a, b| a.lo.cmp(&b.lo));

    let mut result = Vec::with_capacity(intervals.len());
    let mut current = intervals[0].clone();

    for interval in intervals.into_iter().skip(1) {
        // Check if intervals overlap or are adjacent
        if let Some(merged) = current.union(&interval) {
            current = merged;
        } else {
            result.push(current);
            current = interval;
        }
    }
    result.push(current);

    result
}

impl Default for IntervalSet {
    fn default() -> Self {
        Self::reals()
    }
}

impl fmt::Display for IntervalSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else {
            for (i, interval) in self.intervals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∪ ")?;
                }
                write!(f, "{}", interval)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(num_bigint::BigInt::from(n))
    }

    #[test]
    fn test_interval_set_empty() {
        let set = IntervalSet::empty();
        assert!(set.is_empty());
        assert!(!set.contains(&rat(0)));
    }

    #[test]
    fn test_interval_set_reals() {
        let set = IntervalSet::reals();
        assert!(!set.is_empty());
        assert!(set.is_reals());
        assert!(set.contains(&rat(-1000)));
        assert!(set.contains(&rat(0)));
        assert!(set.contains(&rat(1000)));
    }

    #[test]
    fn test_interval_set_union() {
        let a = IntervalSet::from_interval(Interval::closed(rat(1), rat(3)));
        let b = IntervalSet::from_interval(Interval::closed(rat(5), rat(7)));
        let c = a.union(&b);

        assert_eq!(c.num_intervals(), 2);
        assert!(c.contains(&rat(2)));
        assert!(c.contains(&rat(6)));
        assert!(!c.contains(&rat(4)));
    }

    #[test]
    fn test_interval_set_union_merge() {
        let a = IntervalSet::from_interval(Interval::closed(rat(1), rat(5)));
        let b = IntervalSet::from_interval(Interval::closed(rat(3), rat(7)));
        let c = a.union(&b);

        // Should merge into [1, 7]
        assert_eq!(c.num_intervals(), 1);
        assert!(c.contains(&rat(1)));
        assert!(c.contains(&rat(7)));
    }

    #[test]
    fn test_interval_set_intersect() {
        let a = IntervalSet::from_interval(Interval::closed(rat(1), rat(5)));
        let b = IntervalSet::from_interval(Interval::closed(rat(3), rat(7)));
        let c = a.intersect(&b);

        // Should be [3, 5]
        assert_eq!(c.num_intervals(), 1);
        assert!(!c.contains(&rat(2)));
        assert!(c.contains(&rat(3)));
        assert!(c.contains(&rat(5)));
        assert!(!c.contains(&rat(6)));
    }

    #[test]
    fn test_interval_set_complement() {
        let set = IntervalSet::from_interval(Interval::closed(rat(1), rat(5)));
        let comp = set.complement();

        // Complement should be (-∞, 1) ∪ (5, +∞)
        assert_eq!(comp.num_intervals(), 2);
        assert!(comp.contains(&rat(0)));
        assert!(!comp.contains(&rat(1)));
        assert!(!comp.contains(&rat(3)));
        assert!(!comp.contains(&rat(5)));
        assert!(comp.contains(&rat(6)));
    }

    #[test]
    fn test_interval_set_sample() {
        let set = IntervalSet::from_interval(Interval::closed(rat(1), rat(5)));
        let sample = set.sample();
        assert!(sample.is_some());
        let s = sample.unwrap();
        assert!(set.contains(&s));
    }

    #[test]
    fn test_interval_set_sign_set() {
        // Polynomial with roots at 1 and 3
        // Signs: (-∞, 1): +, (1, 3): -, (3, +∞): +
        let roots = vec![rat(1), rat(3)];
        let signs = vec![1, -1, 1];

        let positive_set = IntervalSet::sign_set(&roots, &signs, 1);
        assert!(positive_set.contains(&rat(0)));
        assert!(!positive_set.contains(&rat(2)));
        assert!(positive_set.contains(&rat(4)));

        let negative_set = IntervalSet::sign_set(&roots, &signs, -1);
        assert!(!negative_set.contains(&rat(0)));
        assert!(negative_set.contains(&rat(2)));
        assert!(!negative_set.contains(&rat(4)));

        let zero_set = IntervalSet::sign_set(&roots, &signs, 0);
        assert!(zero_set.contains(&rat(1)));
        assert!(zero_set.contains(&rat(3)));
        assert!(!zero_set.contains(&rat(2)));
    }

    #[test]
    fn test_intersect_with_roots() {
        // Interval set [0, 10]
        let set = IntervalSet::from_interval(Interval::closed(rat(0), rat(10)));

        // Roots at -1, 2, 5, 12
        let roots = vec![rat(-1), rat(2), rat(5), rat(12)];

        let intersection = set.intersect_with_roots(&roots);

        // Should contain only the roots that are in [0, 10]
        assert!(!intersection.contains(&rat(-1))); // Outside
        assert!(intersection.contains(&rat(2))); // Inside
        assert!(intersection.contains(&rat(5))); // Inside
        assert!(!intersection.contains(&rat(12))); // Outside
        assert!(!intersection.contains(&rat(3))); // Not a root
    }

    #[test]
    fn test_filter_by_sign() {
        // Interval set [0, 10]
        let set = IntervalSet::from_interval(Interval::closed(rat(0), rat(10)));

        // Polynomial with roots at 2 and 8
        // Signs: (-∞, 2): +, (2, 8): -, (8, +∞): +
        let roots = vec![rat(2), rat(8)];
        let signs = vec![1, -1, 1];

        // Filter for positive regions
        let positive = set.filter_by_sign(&roots, &signs, 1);
        assert!(positive.contains(&rat(1))); // [0, 2) ∩ [0, 10]
        assert!(!positive.contains(&rat(5))); // (2, 8) is negative
        assert!(positive.contains(&rat(9))); // (8, 10] ∩ [0, 10]

        // Filter for negative regions
        let negative = set.filter_by_sign(&roots, &signs, -1);
        assert!(!negative.contains(&rat(1)));
        assert!(negative.contains(&rat(5))); // (2, 8) ∩ [0, 10]
        assert!(!negative.contains(&rat(9)));
    }

    #[test]
    fn test_complement_point_interval() {
        // Regression test: complement of complement should be identity for point intervals
        let point = IntervalSet::from_interval(Interval::closed(rat(5), rat(5)));
        let comp = point.complement();
        let double_comp = comp.complement();

        assert!(point.contains(&rat(5)));
        assert!(!comp.contains(&rat(5)));
        assert!(double_comp.contains(&rat(5)));

        // Check equality
        assert!(point.contains(&rat(5)) == double_comp.contains(&rat(5)));
        assert!(!point.contains(&rat(4)) && !double_comp.contains(&rat(4)));
        assert!(!point.contains(&rat(6)) && !double_comp.contains(&rat(6)));
    }

    #[test]
    fn test_from_constraint() {
        // Polynomial with roots at 1 and 3
        // Signs: (-∞, 1): +, (1, 3): -, (3, +∞): +
        let roots = vec![rat(1), rat(3)];
        let signs = vec![1, -1, 1];

        // p = 0
        let eq = IntervalSet::from_constraint(&roots, &signs, ConstraintKind::Eq);
        assert!(eq.contains(&rat(1)));
        assert!(eq.contains(&rat(3)));
        assert!(!eq.contains(&rat(2)));

        // p < 0
        let lt = IntervalSet::from_constraint(&roots, &signs, ConstraintKind::Lt);
        assert!(!lt.contains(&rat(0)));
        assert!(!lt.contains(&rat(1)));
        assert!(lt.contains(&rat(2)));
        assert!(!lt.contains(&rat(3)));
        assert!(!lt.contains(&rat(4)));

        // p > 0
        let gt = IntervalSet::from_constraint(&roots, &signs, ConstraintKind::Gt);
        assert!(gt.contains(&rat(0)));
        assert!(!gt.contains(&rat(2)));
        assert!(gt.contains(&rat(4)));

        // p <= 0
        let le = IntervalSet::from_constraint(&roots, &signs, ConstraintKind::Le);
        assert!(!le.contains(&rat(0)));
        assert!(le.contains(&rat(1))); // Root included
        assert!(le.contains(&rat(2)));
        assert!(le.contains(&rat(3))); // Root included
        assert!(!le.contains(&rat(4)));

        // p >= 0
        let ge = IntervalSet::from_constraint(&roots, &signs, ConstraintKind::Ge);
        assert!(ge.contains(&rat(0)));
        assert!(ge.contains(&rat(1))); // Root included
        assert!(!ge.contains(&rat(2)));
        assert!(ge.contains(&rat(3))); // Root included
        assert!(ge.contains(&rat(4)));

        // p != 0
        let ne = IntervalSet::from_constraint(&roots, &signs, ConstraintKind::Ne);
        assert!(ne.contains(&rat(0)));
        assert!(!ne.contains(&rat(1))); // Root excluded
        assert!(ne.contains(&rat(2)));
        assert!(!ne.contains(&rat(3))); // Root excluded
        assert!(ne.contains(&rat(4)));
    }

    // Property-based tests using proptest
    use proptest::prelude::*;

    proptest! {
        /// Property: Union is commutative
        #[test]
        fn prop_union_commutative(a in -1000i64..1000, b in -1000i64..1000,
                                   c in -1000i64..1000, d in -1000i64..1000) {
            let (lo_a, hi_a) = if a <= b { (a, b) } else { (b, a) };
            let (lo_c, hi_c) = if c <= d { (c, d) } else { (d, c) };

            let set1 = IntervalSet::from_interval(Interval::closed(rat(lo_a), rat(hi_a)));
            let set2 = IntervalSet::from_interval(Interval::closed(rat(lo_c), rat(hi_c)));

            let union1 = set1.union(&set2);
            let union2 = set2.union(&set1);

            // Check that both unions contain the same elements
            let mid_ab = (lo_a + hi_a) / 2;
            let mid_cd = (lo_c + hi_c) / 2;
            for x in [lo_a-1, lo_a, mid_ab, hi_a, lo_c-1, lo_c, mid_cd, hi_c, hi_c+1].iter() {
                prop_assert_eq!(union1.contains(&rat(*x)), union2.contains(&rat(*x)));
            }
        }

        /// Property: Intersection is commutative
        #[test]
        fn prop_intersect_commutative(a in -1000i64..1000, b in -1000i64..1000,
                                       c in -1000i64..1000, d in -1000i64..1000) {
            let (lo_a, hi_a) = if a <= b { (a, b) } else { (b, a) };
            let (lo_c, hi_c) = if c <= d { (c, d) } else { (d, c) };

            let set1 = IntervalSet::from_interval(Interval::closed(rat(lo_a), rat(hi_a)));
            let set2 = IntervalSet::from_interval(Interval::closed(rat(lo_c), rat(hi_c)));

            let inter1 = set1.intersect(&set2);
            let inter2 = set2.intersect(&set1);

            // Check that both intersections contain the same elements
            let mid_ab = (lo_a + hi_a) / 2;
            let mid_cd = (lo_c + hi_c) / 2;
            for x in [lo_a-1, lo_a, mid_ab, hi_a, lo_c-1, lo_c, mid_cd, hi_c, hi_c+1].iter() {
                prop_assert_eq!(inter1.contains(&rat(*x)), inter2.contains(&rat(*x)));
            }
        }

        /// Property: Complement of complement is identity
        #[test]
        fn prop_complement_involutive(a in -1000i64..1000, b in -1000i64..1000) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };

            let set = IntervalSet::from_interval(Interval::closed(rat(lo), rat(hi)));
            let double_complement = set.complement().complement();

            // Check points within and at boundaries
            let mid = (lo + hi) / 2;
            for x in [lo, mid, hi].iter() {
                prop_assert_eq!(set.contains(&rat(*x)), double_complement.contains(&rat(*x)),
                    "Failed at x={}, set.contains={}, double_complement.contains={}",
                    x, set.contains(&rat(*x)), double_complement.contains(&rat(*x)));
            }
        }

        /// Property: Union with empty set is identity
        #[test]
        fn prop_union_empty_identity(a in -1000i64..1000, b in -1000i64..1000) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };

            let set = IntervalSet::from_interval(Interval::closed(rat(lo), rat(hi)));
            let empty = IntervalSet::empty();
            let result = set.union(&empty);

            let mid = (lo + hi) / 2;
            for x in [lo, mid, hi].iter() {
                prop_assert_eq!(set.contains(&rat(*x)), result.contains(&rat(*x)));
            }
        }

        /// Property: Intersection with empty set is empty
        #[test]
        fn prop_intersect_empty(a in -1000i64..1000, b in -1000i64..1000) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };

            let set = IntervalSet::from_interval(Interval::closed(rat(lo), rat(hi)));
            let empty = IntervalSet::empty();
            let result = set.intersect(&empty);

            prop_assert!(result.is_empty());
        }

        /// Property: De Morgan's law for intersection
        #[test]
        fn prop_demorgan_intersect(a in -1000i64..1000, b in -1000i64..1000,
                                     c in -1000i64..1000, d in -1000i64..1000) {
            let (lo_a, hi_a) = if a <= b { (a, b) } else { (b, a) };
            let (lo_c, hi_c) = if c <= d { (c, d) } else { (d, c) };

            let set1 = IntervalSet::from_interval(Interval::closed(rat(lo_a), rat(hi_a)));
            let set2 = IntervalSet::from_interval(Interval::closed(rat(lo_c), rat(hi_c)));

            // complement(A ∩ B) = complement(A) ∪ complement(B)
            let left = set1.intersect(&set2).complement();
            let right = set1.complement().union(&set2.complement());

            // Check several points
            let mid_ab = (lo_a + hi_a) / 2;
            let mid_cd = (lo_c + hi_c) / 2;
            for x in [lo_a-1, lo_a, mid_ab, hi_a, lo_c-1, lo_c, mid_cd, hi_c, hi_c+1].iter() {
                prop_assert_eq!(left.contains(&rat(*x)), right.contains(&rat(*x)));
            }
        }

        /// Property: Intersection is subset of both operands
        #[test]
        fn prop_intersect_subset(a in -1000i64..1000, b in -1000i64..1000,
                                  c in -1000i64..1000, d in -1000i64..1000,
                                  x in -2000i64..2000) {
            let (lo_a, hi_a) = if a <= b { (a, b) } else { (b, a) };
            let (lo_c, hi_c) = if c <= d { (c, d) } else { (d, c) };

            let set1 = IntervalSet::from_interval(Interval::closed(rat(lo_a), rat(hi_a)));
            let set2 = IntervalSet::from_interval(Interval::closed(rat(lo_c), rat(hi_c)));
            let inter = set1.intersect(&set2);

            let point = rat(x);
            // If x is in intersection, it must be in both sets
            if inter.contains(&point) {
                prop_assert!(set1.contains(&point));
                prop_assert!(set2.contains(&point));
            }
        }

        /// Property: Union contains both operands
        #[test]
        fn prop_union_superset(a in -1000i64..1000, b in -1000i64..1000,
                                c in -1000i64..1000, d in -1000i64..1000,
                                x in -2000i64..2000) {
            let (lo_a, hi_a) = if a <= b { (a, b) } else { (b, a) };
            let (lo_c, hi_c) = if c <= d { (c, d) } else { (d, c) };

            let set1 = IntervalSet::from_interval(Interval::closed(rat(lo_a), rat(hi_a)));
            let set2 = IntervalSet::from_interval(Interval::closed(rat(lo_c), rat(hi_c)));
            let union = set1.union(&set2);

            let point = rat(x);
            // If x is in either set, it must be in union
            if set1.contains(&point) || set2.contains(&point) {
                prop_assert!(union.contains(&point));
            }
        }
    }
}
