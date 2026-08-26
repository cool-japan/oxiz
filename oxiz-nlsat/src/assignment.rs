//! Variable assignment for NLSAT.
//!
//! This module provides the assignment data structure that tracks:
//! - Current values of arithmetic variables (x0, x1, ...)
//! - Current truth values of boolean variables (b0, b1, ...)
//! - Decision levels and justifications
//!
//! Reference: Z3's `nlsat/nlsat_assignment.h`

use crate::cad::CadPoint;
use crate::interval_set::IntervalSet;
use crate::types::{BoolVar, Lbool, Literal, NULL_BOOL_VAR};
use num_rational::BigRational;
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;

/// Justification for why a literal was assigned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Justification {
    /// Decision by the solver.
    Decision,
    /// Propagated from a clause.
    Propagation(u32),
    /// Justified by theory (CAD).
    Theory,
    /// Unit clause.
    Unit,
}

/// An entry in the assignment trail.
#[derive(Debug, Clone)]
pub struct TrailEntry {
    /// The literal that was assigned.
    pub literal: Literal,
    /// Decision level at which this assignment was made.
    pub level: u32,
    /// Justification for this assignment.
    pub justification: Justification,
}

/// Variable assignment for the NLSAT solver.
///
/// ## Rational and algebraic values
///
/// An arithmetic variable's value is a [`CadPoint`]: a plain rational, or an
/// exact real-algebraic number (defining polynomial + isolating interval) for a
/// witness such as `√2` that no rational can stand in for. The two are
/// deliberately *not* interchangeable at the accessor level:
///
/// * [`Assignment::arith_value`] answers only for a rational slot and reports
///   an algebraic one as `None`. Every reader that predates algebraic witnesses
///   is written as `if let Some(value) = …` or `…?`, so this degrades each of
///   them to "this variable is not usable here" — the sound direction (a wider
///   feasible region, a `Lbool::Undef` evaluation), never a wrong value.
/// * [`Assignment::arith_point`] answers for both, and is what code that can
///   handle an algebraic number exactly (`CadPoint::try_sign_of`) must call.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Values of arithmetic variables.
    ///
    /// `None` means unassigned; `Some(CadPoint::Rational(_))` and
    /// `Some(CadPoint::Algebraic { .. })` are both *assigned*.
    arith_values: Vec<Option<CadPoint>>,
    /// Feasible regions for each arithmetic variable.
    feasible: Vec<IntervalSet>,
    /// Values of boolean variables.
    bool_values: Vec<Lbool>,
    /// Decision levels for boolean variables.
    bool_levels: Vec<u32>,
    /// Trail of assignments.
    trail: Vec<TrailEntry>,
    /// Indices in trail where each decision level starts.
    trail_lim: Vec<usize>,
    /// Current decision level.
    level: u32,
}

impl Assignment {
    /// Create a new empty assignment.
    pub fn new() -> Self {
        Self {
            arith_values: Vec::new(),
            feasible: Vec::new(),
            bool_values: Vec::new(),
            bool_levels: Vec::new(),
            trail: Vec::new(),
            trail_lim: Vec::new(),
            level: 0,
        }
    }

    /// Create a new assignment with specified capacity.
    pub fn with_capacity(num_arith: usize, num_bool: usize) -> Self {
        Self {
            arith_values: vec![None; num_arith],
            feasible: vec![IntervalSet::reals(); num_arith],
            bool_values: vec![Lbool::Undef; num_bool],
            bool_levels: vec![0; num_bool],
            trail: Vec::with_capacity(num_bool),
            trail_lim: Vec::new(),
            level: 0,
        }
    }

    /// Ensure we have enough arithmetic variables.
    pub fn ensure_arith_var(&mut self, var: Var) {
        let idx = var as usize;
        if idx >= self.arith_values.len() {
            self.arith_values.resize(idx + 1, None);
            self.feasible.resize(idx + 1, IntervalSet::reals());
        }
    }

    /// Ensure we have enough boolean variables.
    pub fn ensure_bool_var(&mut self, var: BoolVar) {
        let idx = var as usize;
        if idx >= self.bool_values.len() {
            self.bool_values.resize(idx + 1, Lbool::Undef);
            self.bool_levels.resize(idx + 1, 0);
        }
    }

    /// Get the current decision level.
    #[inline]
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Get the number of arithmetic variables.
    #[inline]
    pub fn num_arith_vars(&self) -> usize {
        self.arith_values.len()
    }

    /// Get the number of boolean variables.
    #[inline]
    pub fn num_bool_vars(&self) -> usize {
        self.bool_values.len()
    }

    /// Get the trail.
    #[inline]
    pub fn trail(&self) -> &[TrailEntry] {
        &self.trail
    }

    // ========== Arithmetic Variable Operations ==========

    /// Check if an arithmetic variable is assigned.
    ///
    /// # THIS IS TRUE FOR AN ALGEBRAIC VALUE TOO
    ///
    /// A variable holding an algebraic point is **assigned**, even though
    /// [`Self::arith_value`] answers `None` for it. Never re-derive this
    /// predicate as `arith_value(var).is_some()`: `NlsatSolver::next_arith_var`
    /// picks "the first variable this says is unassigned" and
    /// `NlsatSolver::is_complete` (see `solver/conflict.rs`) decides `Sat` on
    /// "this says every variable is assigned". Reading an algebraic slot as
    /// unassigned makes the search hand the same variable back to
    /// `pick_arith_value` forever — a hang, not a wrong answer, but a hang in
    /// the one case this whole feature exists to solve.
    pub fn is_arith_assigned(&self, var: Var) -> bool {
        let assigned = self
            .arith_values
            .get(var as usize)
            .is_some_and(|v| v.is_some());
        debug_assert_eq!(
            assigned,
            self.arith_value(var).is_some() || self.arith_point(var).is_some(),
            "assignedness must cover the algebraic slot as well as the rational one"
        );
        assigned
    }

    /// Get the value of an arithmetic variable, **only if it is rational**.
    ///
    /// A variable holding an algebraic value answers `None` here even though
    /// [`Self::is_arith_assigned`] is `true` for it — see the type-level docs.
    /// Use [`Self::arith_point`] when an algebraic value must be seen.
    pub fn arith_value(&self, var: Var) -> Option<&BigRational> {
        self.arith_values
            .get(var as usize)
            .and_then(|v| v.as_ref())
            .and_then(CadPoint::as_rational)
    }

    /// Get the value of an arithmetic variable as a [`CadPoint`], rational or
    /// algebraic alike.
    pub fn arith_point(&self, var: Var) -> Option<&CadPoint> {
        self.arith_values.get(var as usize).and_then(|v| v.as_ref())
    }

    /// Set the value of an arithmetic variable.
    pub fn set_arith(&mut self, var: Var, value: BigRational) {
        self.ensure_arith_var(var);
        self.arith_values[var as usize] = Some(CadPoint::rational(value));
    }

    /// Set the value of an arithmetic variable to an exact point, which may be
    /// an algebraic number.
    pub fn set_arith_point(&mut self, var: Var, point: CadPoint) {
        self.ensure_arith_var(var);
        self.arith_values[var as usize] = Some(point);
    }

    /// Whether any arithmetic variable currently holds a non-rational
    /// (algebraic) value.
    ///
    /// The search uses this as a guard rather than an observation: a
    /// polynomial over *two* algebraic points needs multivariate
    /// real-algebraic arithmetic, which this crate does not have, so
    /// `crate::solver::witness_algebraic` refuses to install a second one.
    pub fn has_algebraic_value(&self) -> bool {
        self.arith_values
            .iter()
            .any(|v| v.as_ref().is_some_and(|p| !p.is_rational()))
    }

    /// Unset the value of an arithmetic variable.
    pub fn unset_arith(&mut self, var: Var) {
        if (var as usize) < self.arith_values.len() {
            self.arith_values[var as usize] = None;
        }
    }

    /// Get the feasible region for an arithmetic variable.
    pub fn feasible(&self, var: Var) -> Option<&IntervalSet> {
        self.feasible.get(var as usize)
    }

    /// Get the feasible region for an arithmetic variable (mutable).
    pub fn feasible_mut(&mut self, var: Var) -> Option<&mut IntervalSet> {
        self.feasible.get_mut(var as usize)
    }

    /// Set the feasible region for an arithmetic variable.
    pub fn set_feasible(&mut self, var: Var, region: IntervalSet) {
        self.ensure_arith_var(var);
        self.feasible[var as usize] = region;
    }

    /// Restrict the feasible region of a variable.
    pub fn restrict_feasible(&mut self, var: Var, constraint: &IntervalSet) {
        self.ensure_arith_var(var);
        let current = &self.feasible[var as usize];
        self.feasible[var as usize] = current.intersect(constraint);
    }

    /// Reset the feasible region of a variable to the entire real line.
    pub fn reset_feasible(&mut self, var: Var) {
        if (var as usize) < self.feasible.len() {
            self.feasible[var as usize] = IntervalSet::reals();
        }
    }

    /// Find the highest unassigned arithmetic variable.
    pub fn max_unassigned_arith(&self) -> Option<Var> {
        for i in (0..self.arith_values.len()).rev() {
            if self.arith_values[i].is_none() {
                return Some(i as Var);
            }
        }
        None
    }

    /// Get the number of assigned arithmetic variables.
    pub fn num_arith_assigned(&self) -> usize {
        self.arith_values.iter().filter(|v| v.is_some()).count()
    }

    // ========== Boolean Variable Operations ==========

    /// Check if a boolean variable is assigned.
    pub fn is_bool_assigned(&self, var: BoolVar) -> bool {
        if var == NULL_BOOL_VAR {
            return false;
        }
        let idx = var as usize;
        idx < self.bool_values.len() && !self.bool_values[idx].is_undef()
    }

    /// Get the value of a boolean variable.
    pub fn bool_value(&self, var: BoolVar) -> Lbool {
        if var == NULL_BOOL_VAR {
            return Lbool::Undef;
        }
        self.bool_values
            .get(var as usize)
            .copied()
            .unwrap_or(Lbool::Undef)
    }

    /// Get the truth value of a literal.
    pub fn lit_value(&self, lit: Literal) -> Lbool {
        let val = self.bool_value(lit.var());
        if lit.is_negated() { val.negate() } else { val }
    }

    /// Get the decision level at which a boolean variable was assigned.
    pub fn bool_level(&self, var: BoolVar) -> u32 {
        self.bool_levels.get(var as usize).copied().unwrap_or(0)
    }

    /// Assign a literal with justification.
    pub fn assign(&mut self, lit: Literal, justification: Justification) {
        let var = lit.var();
        self.ensure_bool_var(var);

        let value = if lit.is_negated() {
            Lbool::False
        } else {
            Lbool::True
        };

        self.bool_values[var as usize] = value;
        self.bool_levels[var as usize] = self.level;

        self.trail.push(TrailEntry {
            literal: lit,
            level: self.level,
            justification,
        });
    }

    /// Unassign a boolean variable.
    fn unassign_bool(&mut self, var: BoolVar) {
        if (var as usize) < self.bool_values.len() {
            self.bool_values[var as usize] = Lbool::Undef;
            self.bool_levels[var as usize] = 0;
        }
    }

    // ========== Decision Level Operations ==========

    /// Push a new decision level.
    pub fn push_level(&mut self) {
        self.level += 1;
        self.trail_lim.push(self.trail.len());
    }

    /// Pop to a previous decision level.
    /// Returns the literals that were unassigned.
    pub fn pop_level(&mut self, target_level: u32) -> Vec<Literal> {
        let mut unassigned = Vec::new();

        while self.level > target_level {
            let trail_start = self.trail_lim.pop().unwrap_or(0);

            // Unassign all literals from this level
            while self.trail.len() > trail_start {
                let entry = self
                    .trail
                    .pop()
                    .expect("collection validated to be non-empty");
                self.unassign_bool(entry.literal.var());
                unassigned.push(entry.literal);
            }

            self.level -= 1;
        }

        unassigned
    }

    /// Clear all assignments and reset to level 0.
    pub fn clear(&mut self) {
        self.arith_values.fill(None);
        self.feasible.fill(IntervalSet::reals());
        self.bool_values.fill(Lbool::Undef);
        self.bool_levels.fill(0);
        self.trail.clear();
        self.trail_lim.clear();
        self.level = 0;
    }

    /// Get the trail entries at a specific level.
    pub fn trail_at_level(&self, level: u32) -> &[TrailEntry] {
        if level == 0 {
            if self.trail_lim.is_empty() {
                &self.trail
            } else {
                &self.trail[..self.trail_lim[0]]
            }
        } else {
            let lvl = level as usize;
            if lvl > self.trail_lim.len() {
                return &[];
            }
            let start = if lvl > 0 { self.trail_lim[lvl - 1] } else { 0 };
            let end = self.trail_lim.get(lvl).copied().unwrap_or(self.trail.len());
            &self.trail[start..end]
        }
    }

    /// Find the first unassigned literal among the given literals.
    pub fn first_unassigned(&self, literals: &[Literal]) -> Option<Literal> {
        literals
            .iter()
            .find(|lit| self.lit_value(**lit).is_undef())
            .copied()
    }

    /// Evaluate an arithmetic value at the current assignment.
    ///
    /// Returns `None` if any required variable is unassigned **or holds an
    /// algebraic value**: this returns a `BigRational`, and there is no
    /// rational that is `√2`. Rounding to the isolating interval's midpoint
    /// here would silently turn `x² − 2` into a non-zero value at its own root,
    /// which is precisely the unsoundness `CadPoint` exists to prevent. Use
    /// [`Self::eval_poly_sign`] when only the sign is needed — that one *can*
    /// answer exactly at an algebraic point.
    pub fn eval_poly(&self, poly: &Polynomial) -> Option<BigRational> {
        // Build assignment map from current values
        let mut assignment = FxHashMap::default();

        // Get all variables in the polynomial
        let vars = poly.vars();

        // Check that all variables are assigned to a *rational* and collect
        // their values.
        for &var in &vars {
            let value = self.arith_value(var)?;
            assignment.insert(var, value.clone());
        }

        // Evaluate the polynomial
        Some(poly.eval(&assignment))
    }

    /// Exact sign (`-1`, `0`, `1`) of `poly` under the current assignment, or
    /// `None` when it cannot be *proved*.
    ///
    /// Three cases, and the honest answer to everything else is `None`:
    ///
    /// 1. Every variable of `poly` holds a rational value — the sign of
    ///    [`Self::eval_poly`].
    /// 2. Every variable of `poly` is assigned and exactly one of them holds
    ///    an algebraic value — substitute the rationals, leaving a univariate
    ///    polynomial in that one variable, and ask
    ///    [`CadPoint::try_sign_of`], which decides `p(α) = 0` exactly by a gcd
    ///    test and otherwise refines the isolating interval until the sign is
    ///    constant on it.
    /// 3. Anything else — an unassigned variable, *two* algebraic variables
    ///    (that is multivariate real-algebraic arithmetic, which this crate
    ///    does not have), or a refinement budget that ran out before the sign
    ///    was proved.
    ///
    /// `None` never means "zero" and never means "some sign we could not pin
    /// down": callers must treat it as undecided.
    pub fn eval_poly_sign(&self, poly: &Polynomial) -> Option<i8> {
        // One pass over the polynomial's variables: collect the rational
        // values as an evaluation map and note the single algebraic one, if
        // any. This runs for every factor of every atom on each theory
        // propagation sweep, so it deliberately does not re-walk `vars()` per
        // case.
        let vars = poly.vars();
        let mut rationals = FxHashMap::default();
        let mut algebraic: Option<Var> = None;
        for &var in &vars {
            match self.arith_values.get(var as usize).and_then(|v| v.as_ref()) {
                // Case 3a: unassigned.
                None => return None,
                Some(CadPoint::Rational(value)) => {
                    rationals.insert(var, value.clone());
                }
                Some(CadPoint::Algebraic { .. }) => {
                    if algebraic.is_some() {
                        return None; // Case 3b: two algebraic variables.
                    }
                    algebraic = Some(var);
                }
            }
        }

        let Some(alpha_var) = algebraic else {
            // Case 1: fully rational.
            return Some(rational_sign(&poly.eval(&rationals)));
        };

        // Case 2: substitute every rational value, leaving a univariate
        // polynomial in `alpha_var`.
        let mut residual = poly.clone();
        for (&var, value) in &rationals {
            residual = residual.substitute(var, &Polynomial::constant(value.clone()));
        }
        self.arith_point(alpha_var)?
            .try_sign_of(&residual, alpha_var)
    }
}

/// Sign of `x` as `-1`, `0`, or `1`.
fn rational_sign(x: &BigRational) -> i8 {
    use num_traits::Zero;
    if x.is_zero() {
        0
    } else if x > &BigRational::zero() {
        1
    } else {
        -1
    }
}

impl Default for Assignment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assignment_basic() {
        let mut a = Assignment::new();

        // Ensure variables
        a.ensure_bool_var(5);
        assert_eq!(a.num_bool_vars(), 6);

        a.ensure_arith_var(3);
        assert_eq!(a.num_arith_vars(), 4);
    }

    #[test]
    fn test_assignment_bool() {
        let mut a = Assignment::with_capacity(0, 10);

        // Initial state
        assert!(a.bool_value(5).is_undef());
        assert!(!a.is_bool_assigned(5));

        // Assign
        a.assign(Literal::positive(5), Justification::Decision);
        assert!(a.bool_value(5).is_true());
        assert!(a.is_bool_assigned(5));

        // Literal value
        assert!(a.lit_value(Literal::positive(5)).is_true());
        assert!(a.lit_value(Literal::negative(5)).is_false());
    }

    #[test]
    fn test_assignment_levels() {
        let mut a = Assignment::with_capacity(0, 10);

        // Level 0
        a.assign(Literal::positive(0), Justification::Unit);
        assert_eq!(a.level(), 0);

        // Push level 1
        a.push_level();
        assert_eq!(a.level(), 1);
        a.assign(Literal::positive(1), Justification::Decision);

        // Push level 2
        a.push_level();
        assert_eq!(a.level(), 2);
        a.assign(Literal::positive(2), Justification::Propagation(0));

        // Pop back to level 1
        let unassigned = a.pop_level(1);
        assert_eq!(a.level(), 1);
        assert!(a.bool_value(2).is_undef());
        assert!(a.bool_value(1).is_true());
        assert!(a.bool_value(0).is_true());
        assert_eq!(unassigned.len(), 1);

        // Pop back to level 0
        let unassigned = a.pop_level(0);
        assert_eq!(a.level(), 0);
        assert!(a.bool_value(1).is_undef());
        assert!(a.bool_value(0).is_true()); // Level 0 assignment remains
        assert_eq!(unassigned.len(), 1);
    }

    #[test]
    fn test_assignment_arith() {
        let mut a = Assignment::with_capacity(5, 0);

        // Initial state
        assert!(!a.is_arith_assigned(2));
        assert!(a.arith_value(2).is_none());

        // Assign
        let val = BigRational::from_integer(num_bigint::BigInt::from(42));
        a.set_arith(2, val.clone());
        assert!(a.is_arith_assigned(2));
        assert_eq!(a.arith_value(2), Some(&val));

        // Unset
        a.unset_arith(2);
        assert!(!a.is_arith_assigned(2));
    }

    /// An algebraic slot is **assigned** but has no rational value: the exact
    /// combination `next_arith_var` / `is_complete` depend on.
    #[test]
    fn test_algebraic_slot_is_assigned_but_has_no_rational_value() {
        use oxiz_math::polynomial::Polynomial;

        let mut a = Assignment::with_capacity(2, 0);
        let x2_minus_2 = Polynomial::univariate(
            0,
            &[
                BigRational::from_integer(num_bigint::BigInt::from(-2)),
                BigRational::from_integer(num_bigint::BigInt::from(0)),
                BigRational::from_integer(num_bigint::BigInt::from(1)),
            ],
        );
        let sqrt2 = CadPoint::algebraic(
            BigRational::from_integer(num_bigint::BigInt::from(1)),
            BigRational::from_integer(num_bigint::BigInt::from(2)),
            x2_minus_2.clone(),
            2,
        );
        a.set_arith_point(0, sqrt2);

        assert!(
            a.is_arith_assigned(0),
            "an algebraic slot must count as assigned, or the search never terminates"
        );
        assert!(a.arith_value(0).is_none(), "no rational is √2");
        assert!(a.arith_point(0).is_some());
        assert!(a.has_algebraic_value());

        // Sign is exact at the root, and `eval_poly` refuses to round.
        assert_eq!(a.eval_poly_sign(&x2_minus_2), Some(0));
        assert!(a.eval_poly(&x2_minus_2).is_none());

        // x - 1 is positive at √2 ≈ 1.414.
        let x_minus_1 = Polynomial::univariate(
            0,
            &[
                BigRational::from_integer(num_bigint::BigInt::from(-1)),
                BigRational::from_integer(num_bigint::BigInt::from(1)),
            ],
        );
        assert_eq!(a.eval_poly_sign(&x_minus_1), Some(1));

        // Unsetting clears it, and a rational overwrite clears the flag.
        a.unset_arith(0);
        assert!(!a.is_arith_assigned(0));
        assert!(!a.has_algebraic_value());
    }

    /// Two algebraic values leave a genuinely multivariate real-algebraic
    /// problem, which must answer "undecided" rather than guess.
    #[test]
    fn test_two_algebraic_values_make_a_product_undecidable() {
        use oxiz_math::polynomial::Polynomial;

        fn rat(n: i64) -> BigRational {
            BigRational::from_integer(num_bigint::BigInt::from(n))
        }

        let mut a = Assignment::with_capacity(2, 0);
        a.set_arith_point(
            0,
            CadPoint::algebraic(
                rat(1),
                rat(2),
                Polynomial::univariate(0, &[rat(-2), rat(0), rat(1)]),
                2,
            ),
        );
        a.set_arith_point(
            1,
            CadPoint::algebraic(
                rat(1),
                rat(2),
                Polynomial::univariate(1, &[rat(-3), rat(0), rat(1)]),
                2,
            ),
        );

        let xy = Polynomial::mul(&Polynomial::from_var(0), &Polynomial::from_var(1));
        assert_eq!(
            a.eval_poly_sign(&xy),
            None,
            "√2 · √3 needs multivariate algebraic arithmetic this crate does not have"
        );
    }

    #[test]
    fn test_assignment_feasible() {
        let mut a = Assignment::with_capacity(5, 0);

        // Initial state is reals
        assert!(
            a.feasible(0)
                .expect("test operation should succeed")
                .is_reals()
        );

        // Restrict
        let constraint = IntervalSet::from_interval(oxiz_math::interval::Interval::closed(
            BigRational::from_integer(num_bigint::BigInt::from(0)),
            BigRational::from_integer(num_bigint::BigInt::from(10)),
        ));
        a.restrict_feasible(0, &constraint);
        assert!(
            !a.feasible(0)
                .expect("test operation should succeed")
                .is_reals()
        );

        // Reset
        a.reset_feasible(0);
        assert!(
            a.feasible(0)
                .expect("test operation should succeed")
                .is_reals()
        );
    }
}
