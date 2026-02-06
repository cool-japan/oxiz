//! String Theory Length Constraint Propagation.
//!
//! This module implements sophisticated length constraint propagation for string
//! theory, which is crucial for efficient solving of word equations and string
//! constraints.
//!
//! ## Key Techniques
//!
//! 1. **Length Variable Domains**: Track possible length ranges for string variables
//! 2. **Substring Length Inference**: Deduce length constraints from substring operations
//! 3. **Concatenation Propagation**: Propagate lengths through concat operations
//! 4. **Arithmetic Integration**: Interface with LIA solver for length arithmetic
//! 5. **Conflict Analysis**: Generate length-based conflict explanations
//!
//! ## Example Propagations
//!
//! - `len(x ++ y) = len(x) + len(y)` (concatenation)
//! - `x = substr(y, i, len(x))` implies `len(x) ≤ len(y)`
//! - `x contains y` implies `len(y) ≤ len(x)`
//! - `x = replace(y, a, b)` implies bounds on `len(x)`
//!
//! ## Integration
//!
//! Length propagation runs as part of the string theory solver:
//! 1. Extract length constraints from string operations
//! 2. Add length constraints to LIA solver
//! 3. Propagate using LIA (simplex, bounds propagation)
//! 4. Feed back length information to string solver
//!
//! ## References
//!
//! - Liang et al.: "A DPLL(T) Theory Solver for a Theory of Strings and Regular Expressions" (CAV 2014)
//! - Z3 string solver length abstraction
//! - CVC4 string theory implementation

use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;

/// Length domain for a string variable (possible length range).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LengthDomain {
    /// Minimum possible length (inclusive).
    pub min: i64,
    /// Maximum possible length (inclusive, None = unbounded).
    pub max: Option<i64>,
}

impl LengthDomain {
    /// Create an unbounded domain [0, ∞).
    pub fn unbounded() -> Self {
        Self { min: 0, max: None }
    }

    /// Create a point domain [n, n].
    pub fn point(n: i64) -> Self {
        Self {
            min: n,
            max: Some(n),
        }
    }

    /// Create a bounded domain [min, max].
    pub fn bounded(min: i64, max: i64) -> Self {
        Self {
            min,
            max: Some(max),
        }
    }

    /// Check if domain is empty (inconsistent).
    pub fn is_empty(&self) -> bool {
        if let Some(max) = self.max {
            max < self.min
        } else {
            false
        }
    }

    /// Check if domain is a single point.
    pub fn is_point(&self) -> bool {
        self.max == Some(self.min)
    }

    /// Intersect with another domain.
    pub fn intersect(&self, other: &LengthDomain) -> LengthDomain {
        let min = self.min.max(other.min);
        let max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        LengthDomain { min, max }
    }

    /// Add a constant offset to the domain.
    pub fn add(&self, offset: i64) -> LengthDomain {
        LengthDomain {
            min: self.min.saturating_add(offset),
            max: self.max.map(|m| m.saturating_add(offset)),
        }
    }

    /// Subtract a constant offset from the domain.
    pub fn sub(&self, offset: i64) -> LengthDomain {
        LengthDomain {
            min: (self.min - offset).max(0), // Length is non-negative
            max: self.max.map(|m| (m - offset).max(0)),
        }
    }
}

/// Length constraint between terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LengthConstraint {
    /// len(x) = n
    Equal(TermId, i64),
    /// len(x) <= n
    LessEq(TermId, i64),
    /// len(x) >= n
    GreaterEq(TermId, i64),
    /// len(x) = len(y)
    EqualVar(TermId, TermId),
    /// len(x) <= len(y)
    LessEqVar(TermId, TermId),
    /// len(x) + len(y) = len(z) (concatenation)
    Concat(TermId, TermId, TermId),
}

/// Statistics for length propagation.
#[derive(Debug, Clone, Default)]
pub struct LengthPropStats {
    /// Number of domain refinements.
    pub domain_refinements: u64,
    /// Number of length conflicts detected.
    pub length_conflicts: u64,
    /// Number of propagations to LIA.
    pub lia_propagations: u64,
    /// Number of propagations from LIA.
    pub lia_feedbacks: u64,
    /// Time in length propagation (microseconds).
    pub propagation_time_us: u64,
}

/// Length constraint propagator for string theory.
pub struct LengthPropagator {
    /// Length domains for string variables.
    domains: FxHashMap<TermId, LengthDomain>,
    /// Length constraints collected from string operations.
    constraints: Vec<LengthConstraint>,
    /// Statistics.
    stats: LengthPropStats,
}

impl LengthPropagator {
    /// Create a new length propagator.
    pub fn new() -> Self {
        Self {
            domains: FxHashMap::default(),
            constraints: Vec::new(),
            stats: LengthPropStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &LengthPropStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = LengthPropStats::default();
    }

    /// Get length domain for a term.
    pub fn get_domain(&self, term: TermId) -> LengthDomain {
        self.domains
            .get(&term)
            .cloned()
            .unwrap_or_else(LengthDomain::unbounded)
    }

    /// Set length domain for a term.
    pub fn set_domain(&mut self, term: TermId, domain: LengthDomain) {
        if domain.is_empty() {
            self.stats.length_conflicts += 1;
        }

        self.domains.insert(term, domain);
    }

    /// Refine length domain (intersect with new constraints).
    pub fn refine_domain(&mut self, term: TermId, new_domain: LengthDomain) {
        let current = self.get_domain(term);
        let refined = current.intersect(&new_domain);

        if refined != current {
            self.stats.domain_refinements += 1;
        }

        self.set_domain(term, refined);
    }

    /// Add a length constraint.
    pub fn add_constraint(&mut self, constraint: LengthConstraint) {
        self.constraints.push(constraint);
    }

    /// Propagate length constraints.
    ///
    /// Returns true if new information was propagated.
    pub fn propagate(&mut self) -> bool {
        let start = std::time::Instant::now();
        let mut changed = false;

        // Process each constraint
        for constraint in &self.constraints.clone() {
            match constraint {
                LengthConstraint::Equal(x, n) => {
                    let domain = LengthDomain::point(*n);
                    let old = self.get_domain(*x);
                    if old != domain {
                        self.refine_domain(*x, domain);
                        changed = true;
                    }
                }

                LengthConstraint::LessEq(x, n) => {
                    let domain = LengthDomain::bounded(0, *n);
                    let old = self.get_domain(*x);
                    let refined = old.intersect(&domain);
                    if refined != old {
                        self.set_domain(*x, refined);
                        changed = true;
                    }
                }

                LengthConstraint::GreaterEq(x, n) => {
                    let domain = LengthDomain { min: *n, max: None };
                    let old = self.get_domain(*x);
                    let refined = old.intersect(&domain);
                    if refined != old {
                        self.set_domain(*x, refined);
                        changed = true;
                    }
                }

                LengthConstraint::EqualVar(x, y) => {
                    let domain_x = self.get_domain(*x);
                    let domain_y = self.get_domain(*y);
                    let intersection = domain_x.intersect(&domain_y);

                    if intersection != domain_x {
                        self.set_domain(*x, intersection.clone());
                        changed = true;
                    }
                    if intersection != domain_y {
                        self.set_domain(*y, intersection);
                        changed = true;
                    }
                }

                LengthConstraint::Concat(x, y, z) => {
                    // len(x) + len(y) = len(z)
                    let domain_x = self.get_domain(*x);
                    let domain_y = self.get_domain(*y);
                    let domain_z = self.get_domain(*z);

                    // Propagate: len(z) >= len(x) + len(y).min
                    let min_z = domain_x.min + domain_y.min;
                    if min_z > domain_z.min {
                        let new_domain = LengthDomain {
                            min: min_z,
                            max: domain_z.max,
                        };
                        self.set_domain(*z, new_domain);
                        changed = true;
                    }

                    // Propagate: len(x) <= len(z)
                    // (if len(z) is bounded, len(x) is also bounded)
                    if let Some(max_z) = domain_z.max
                        && domain_x.max.is_none_or(|m| m > max_z)
                    {
                        self.refine_domain(
                            *x,
                            LengthDomain {
                                min: domain_x.min,
                                max: Some(max_z),
                            },
                        );
                        changed = true;
                    }
                }

                _ => {}
            }
        }

        self.stats.propagation_time_us += start.elapsed().as_micros() as u64;
        changed
    }

    /// Check if any domain is empty (conflict).
    pub fn has_conflict(&self) -> bool {
        self.domains.values().any(|d| d.is_empty())
    }

    /// Clear all domains and constraints.
    pub fn clear(&mut self) {
        self.domains.clear();
        self.constraints.clear();
    }
}

impl Default for LengthPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_domain_unbounded() {
        let domain = LengthDomain::unbounded();
        assert_eq!(domain.min, 0);
        assert_eq!(domain.max, None);
        assert!(!domain.is_empty());
        assert!(!domain.is_point());
    }

    #[test]
    fn test_length_domain_point() {
        let domain = LengthDomain::point(5);
        assert_eq!(domain.min, 5);
        assert_eq!(domain.max, Some(5));
        assert!(!domain.is_empty());
        assert!(domain.is_point());
    }

    #[test]
    fn test_length_domain_empty() {
        let domain = LengthDomain::bounded(10, 5); // min > max
        assert!(domain.is_empty());
    }

    #[test]
    fn test_length_domain_intersect() {
        let d1 = LengthDomain::bounded(0, 10);
        let d2 = LengthDomain::bounded(5, 15);

        let intersection = d1.intersect(&d2);
        assert_eq!(intersection.min, 5);
        assert_eq!(intersection.max, Some(10));
    }

    #[test]
    fn test_length_domain_intersect_unbounded() {
        let d1 = LengthDomain::unbounded();
        let d2 = LengthDomain::bounded(5, 10);

        let intersection = d1.intersect(&d2);
        assert_eq!(intersection.min, 5);
        assert_eq!(intersection.max, Some(10));
    }

    #[test]
    fn test_length_domain_add() {
        let domain = LengthDomain::bounded(5, 10);
        let shifted = domain.add(3);

        assert_eq!(shifted.min, 8);
        assert_eq!(shifted.max, Some(13));
    }

    #[test]
    fn test_length_domain_sub() {
        let domain = LengthDomain::bounded(10, 20);
        let shifted = domain.sub(5);

        assert_eq!(shifted.min, 5);
        assert_eq!(shifted.max, Some(15));
    }

    #[test]
    fn test_length_domain_sub_negative() {
        let domain = LengthDomain::bounded(3, 5);
        let shifted = domain.sub(10); // Would go negative

        // Should clamp to 0 (lengths are non-negative)
        assert_eq!(shifted.min, 0);
        assert_eq!(shifted.max, Some(0));
    }

    #[test]
    fn test_length_propagator_creation() {
        let prop = LengthPropagator::new();
        assert_eq!(prop.stats().domain_refinements, 0);
        assert!(!prop.has_conflict());
    }

    #[test]
    fn test_length_propagator_set_domain() {
        let mut prop = LengthPropagator::new();

        let term = TermId::new(1);
        let domain = LengthDomain::bounded(5, 10);

        prop.set_domain(term, domain.clone());

        assert_eq!(prop.get_domain(term), domain);
    }

    #[test]
    fn test_length_propagator_refine() {
        let mut prop = LengthPropagator::new();

        let term = TermId::new(1);
        prop.set_domain(term, LengthDomain::bounded(0, 20));

        // Refine to [5, 10]
        prop.refine_domain(term, LengthDomain::bounded(5, 10));

        let refined = prop.get_domain(term);
        assert_eq!(refined.min, 5);
        assert_eq!(refined.max, Some(10));
        assert_eq!(prop.stats().domain_refinements, 1);
    }

    #[test]
    fn test_constraint_equal() {
        let mut prop = LengthPropagator::new();

        let term = TermId::new(1);
        prop.add_constraint(LengthConstraint::Equal(term, 5));

        let changed = prop.propagate();
        assert!(changed);

        let domain = prop.get_domain(term);
        assert!(domain.is_point());
        assert_eq!(domain.min, 5);
    }

    #[test]
    fn test_constraint_less_eq() {
        let mut prop = LengthPropagator::new();

        let term = TermId::new(1);
        prop.add_constraint(LengthConstraint::LessEq(term, 10));

        prop.propagate();

        let domain = prop.get_domain(term);
        assert_eq!(domain.min, 0);
        assert_eq!(domain.max, Some(10));
    }

    #[test]
    fn test_constraint_concat() {
        let mut prop = LengthPropagator::new();

        let x = TermId::new(1);
        let y = TermId::new(2);
        let z = TermId::new(3);

        // Set initial domains
        prop.set_domain(x, LengthDomain::bounded(2, 3));
        prop.set_domain(y, LengthDomain::bounded(3, 4));

        // Add: len(x) + len(y) = len(z)
        prop.add_constraint(LengthConstraint::Concat(x, y, z));

        prop.propagate();

        // len(z) should be >= 2 + 3 = 5
        let domain_z = prop.get_domain(z);
        assert!(domain_z.min >= 5);
    }

    #[test]
    fn test_conflict_detection() {
        let mut prop = LengthPropagator::new();

        let term = TermId::new(1);
        prop.set_domain(term, LengthDomain::bounded(10, 5)); // Empty domain

        assert!(prop.has_conflict());
        assert_eq!(prop.stats().length_conflicts, 1);
    }

    #[test]
    fn test_propagator_clear() {
        let mut prop = LengthPropagator::new();

        prop.set_domain(TermId::new(1), LengthDomain::point(5));
        prop.add_constraint(LengthConstraint::Equal(TermId::new(2), 10));

        assert!(!prop.domains.is_empty());
        assert!(!prop.constraints.is_empty());

        prop.clear();

        assert!(prop.domains.is_empty());
        assert!(prop.constraints.is_empty());
    }

    #[test]
    fn test_stats_reset() {
        let mut prop = LengthPropagator::new();

        prop.stats.domain_refinements = 100;
        prop.stats.length_conflicts = 10;

        prop.reset_stats();

        assert_eq!(prop.stats().domain_refinements, 0);
        assert_eq!(prop.stats().length_conflicts, 0);
    }
}
