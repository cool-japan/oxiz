//! Unit tests for [`super::ArithSolver`].
//!
//! Relocated out of `solver.rs` as a child module (rather than left inline)
//! once that file approached the workspace's 2000-line-per-file ceiling --
//! the same treatment `euf::solver` already gives its own tests. Pure
//! relocation: the module path `arithmetic::solver::tests` is unchanged, so
//! every test name and `use super::*` import resolves exactly as before.

use super::*;
use num_traits::{One, Zero};

#[test]
fn test_arith_basic() {
    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let reason = TermId::new(100);

    // x >= 0
    solver.assert_ge(
        &[(x, Rational64::one())],
        Rational64::from_integer(0),
        reason,
    );

    // y >= 0
    solver.assert_ge(
        &[(y, Rational64::one())],
        Rational64::from_integer(0),
        reason,
    );

    // x + y <= 10
    solver.assert_le(
        &[(x, Rational64::one()), (y, Rational64::one())],
        Rational64::from_integer(10),
        reason,
    );

    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

#[test]
fn test_arith_unsat() {
    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let reason = TermId::new(100);

    // x >= 10
    solver.assert_ge(
        &[(x, Rational64::one())],
        Rational64::from_integer(10),
        reason,
    );

    // x <= 5
    solver.assert_le(
        &[(x, Rational64::one())],
        Rational64::from_integer(5),
        reason,
    );

    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_arith_strict_inequality() {
    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let reason = TermId::new(100);

    // x > 0 (strict)
    solver.assert_gt(
        &[(x, Rational64::one())],
        Rational64::from_integer(0),
        reason,
    );

    // x < 10 (strict)
    solver.assert_lt(
        &[(x, Rational64::one())],
        Rational64::from_integer(10),
        reason,
    );

    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

#[test]
fn test_arith_strict_unsat() {
    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let reason = TermId::new(100);

    // x >= 5
    solver.assert_ge(
        &[(x, Rational64::one())],
        Rational64::from_integer(5),
        reason,
    );

    // x < 5 (strict) - should be unsatisfiable with x >= 5
    solver.assert_lt(
        &[(x, Rational64::one())],
        Rational64::from_integer(5),
        reason,
    );

    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_coefficient_normalization_lia() {
    let mut solver = ArithSolver::lia();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let reason = TermId::new(100);

    // 2x + 4y <= 10 should be normalized to x + 2y <= 5 (GCD = 2)
    solver.assert_le(
        &[
            (x, Rational64::from_integer(2)),
            (y, Rational64::from_integer(4)),
        ],
        Rational64::from_integer(10),
        reason,
    );

    // The solver should handle this correctly
    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

#[test]
fn test_coefficient_normalization_sign() {
    let solver = ArithSolver::lra();

    let _x = TermId::new(1);
    let _y = TermId::new(2);

    // Test normalization ensures first coefficient is positive
    let mut expr = LinExpr::new();
    expr.add_term(0, Rational64::from_integer(-3));
    expr.add_term(1, Rational64::from_integer(2));

    solver.normalize_expr(&mut expr);

    // After normalization, first coefficient should be positive
    if let Some((_, c)) = expr.terms.first() {
        assert!(c > &Rational64::zero());
    }
}

#[test]
fn test_gcd_computation() {
    assert_eq!(gcd_i64(12, 8), 4);
    assert_eq!(gcd_i64(15, 25), 5);
    assert_eq!(gcd_i64(7, 13), 1);
    assert_eq!(gcd_i64(0, 5), 5);
    assert_eq!(gcd_i64(5, 0), 5);
    assert_eq!(gcd_i64(-12, 8), 4);
    assert_eq!(gcd_i64(12, -8), 4);
}

// Audit regression (theories-arith): the GCD-infeasibility path in
// `assert_eq` used to fabricate its contradictory bounds with a
// hardcoded `reason` id of `0`, so the resulting UNSAT conflict always
// cited whatever the FIRST reason ever added happened to be, instead of
// the actual assertion that caused the contradiction. Assert an
// unrelated, satisfiable constraint first (populating reason id `0`
// with an unrelated term), then a GCD-infeasible equality with a
// DIFFERENT reason term, and confirm the conflict cites the real
// culprit.
#[test]
fn audit_gcd_infeasibility_conflict_cites_real_reason() {
    let mut solver = ArithSolver::lia();

    let x = TermId::new(10);
    let y = TermId::new(20);
    let unrelated_reason = TermId::new(1);
    let real_reason = TermId::new(2);

    // x >= 0: satisfiable, unrelated to the GCD conflict. If the old
    // hardcoded-reason-0 bug were still present, this becomes
    // `self.reasons[0]`, and the GCD conflict below would wrongly cite
    // it instead of `real_reason`.
    solver.assert_ge(
        &[(x, Rational64::one())],
        Rational64::zero(),
        unrelated_reason,
    );

    // 2y = 7 has no integer solution: gcd(2) = 2 does not divide 7.
    solver.assert_eq(
        &[(y, Rational64::from_integer(2))],
        Rational64::from_integer(7),
        real_reason,
    );

    let result = solver.check().expect("check should succeed");
    match result {
        TheoryResult::Unsat(conflict) => {
            assert!(
                conflict.contains(&real_reason),
                "GCD-infeasibility conflict must cite the actual violating \
                     assertion {real_reason:?}, got {conflict:?}"
            );
        }
        other => panic!("expected Unsat (2y=7 is GCD-infeasible over integers), got {other:?}"),
    }
}

#[test]
fn test_bound_tightening_lia() {
    let solver = ArithSolver::lia();

    // Upper bound tightening: x <= 5.7 -> x <= 5
    let tightened = solver.tighten_bound(Rational64::new(57, 10), true);
    assert_eq!(tightened, Rational64::from_integer(5));

    // Lower bound tightening: x >= 2.3 -> x >= 3
    let tightened = solver.tighten_bound(Rational64::new(23, 10), false);
    assert_eq!(tightened, Rational64::from_integer(3));

    // Integer bounds don't change
    let tightened = solver.tighten_bound(Rational64::from_integer(5), true);
    assert_eq!(tightened, Rational64::from_integer(5));
}

#[test]
fn test_bound_tightening_lra() {
    let solver = ArithSolver::lra();

    // No tightening for real arithmetic
    let bound = Rational64::new(57, 10);
    let tightened = solver.tighten_bound(bound, true);
    assert_eq!(tightened, bound);
}

#[test]
fn test_tighten_constraints() {
    let mut solver_lia = ArithSolver::lia();
    let mut solver_lra = ArithSolver::lra();

    // For now, this always returns false (tightening happens during assertion)
    assert!(!solver_lia.tighten_constraints());
    assert!(!solver_lra.tighten_constraints());
}

/// Test that x > 5 AND x < 6 is UNSAT for integers (no integer in open interval (5,6))
/// This is the bug report test case: strict inequalities must be transformed for LIA
#[test]
fn test_lia_strict_inequality_empty_interval() {
    let mut solver = ArithSolver::lia();

    let x = TermId::new(1);
    let reason = TermId::new(100);

    // x > 5 (for integers, this becomes x >= 6)
    solver.assert_gt(
        &[(x, Rational64::one())],
        Rational64::from_integer(5),
        reason,
    );

    // x < 6 (for integers, this becomes x <= 5)
    solver.assert_lt(
        &[(x, Rational64::one())],
        Rational64::from_integer(6),
        reason,
    );

    // Should be UNSAT: x >= 6 AND x <= 5 is impossible
    let result = solver.check().expect("test operation should succeed");
    assert!(
        matches!(result, TheoryResult::Unsat(_)),
        "Expected UNSAT for x > 5 AND x < 6 in LIA, got {:?}",
        result
    );
}

/// Test that x > 5 AND x < 6 is SAT for reals (5.5 is a valid solution)
#[test]
fn test_lra_strict_inequality_has_solution() {
    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let reason = TermId::new(100);

    // x > 5
    solver.assert_gt(
        &[(x, Rational64::one())],
        Rational64::from_integer(5),
        reason,
    );

    // x < 6
    solver.assert_lt(
        &[(x, Rational64::one())],
        Rational64::from_integer(6),
        reason,
    );

    // Should be SAT for reals: x = 5.5 is a valid solution
    let result = solver.check().expect("test operation should succeed");
    assert!(
        matches!(result, TheoryResult::Sat),
        "Expected SAT for x > 5 AND x < 6 in LRA, got {:?}",
        result
    );
}

/// Test x >= 5 AND x <= 5 with strict bounds in LIA
#[test]
fn test_lia_strict_at_boundary() {
    let mut solver = ArithSolver::lia();

    let x = TermId::new(1);
    let reason = TermId::new(100);

    // x >= 5
    solver.assert_ge(
        &[(x, Rational64::one())],
        Rational64::from_integer(5),
        reason,
    );

    // x < 6 (becomes x <= 5)
    solver.assert_lt(
        &[(x, Rational64::one())],
        Rational64::from_integer(6),
        reason,
    );

    // Should be SAT: x = 5 is the only solution
    let result = solver.check().expect("test operation should succeed");
    assert!(
        matches!(result, TheoryResult::Sat),
        "Expected SAT for x >= 5 AND x < 6 in LIA, got {:?}",
        result
    );
}

// ---- Nelson-Oppen tests ----

/// x <= y AND y <= x should yield an entailed equality.
/// `entailed_disequal_reason` must report a *reason* exactly when
/// arithmetic's own bounds already rule out `x = y`, and report nothing
/// when they leave the equality open.
///
/// Pinning `x` to the single point `[3, 3]` and `y` to `[5, 5]` makes
/// `x = y` infeasible on both sides at once, so the probe must return the
/// bound atoms that force it. Widening `y` to `[2, 5]` -- which now
/// overlaps `x`'s point -- leaves `x = y` perfectly satisfiable, so the
/// probe must decline: reporting a disequality there would manufacture a
/// cross-theory conflict out of nothing.
#[test]
fn test_pr30_entailed_disequal_reason_fires_only_on_disjoint_bounds() {
    let x = TermId::new(1);
    let y = TermId::new(2);
    // Distinct reason terms so the returned reason can be checked for
    // *which* bound atoms it names, not merely that it is non-empty.
    let x_lo = TermId::new(101);
    let x_hi = TermId::new(102);
    let y_lo = TermId::new(103);
    let y_hi = TermId::new(104);

    let mut solver = ArithSolver::lra();
    solver.intern(x);
    solver.intern(y);

    // x pinned to [3, 3].
    solver.assert_ge(&[(x, Rational64::one())], Rational64::from_integer(3), x_lo);
    solver.assert_le(&[(x, Rational64::one())], Rational64::from_integer(3), x_hi);
    // y pinned to [5, 5]: disjoint from x.
    solver.assert_ge(&[(y, Rational64::one())], Rational64::from_integer(5), y_lo);
    solver.assert_le(&[(y, Rational64::one())], Rational64::from_integer(5), y_hi);

    assert!(
        matches!(
            solver.check().expect("check should succeed"),
            TheoryResult::Sat
        ),
        "the bounds themselves are consistent; only x = y is not"
    );

    let reason = solver
        .entailed_disequal_reason(x, y)
        .expect("x in [3,3] and y in [5,5] entails x != y");
    assert!(
        !reason.is_empty(),
        "an entailed disequality must be justified by the bound atoms that force it"
    );
    assert!(
        reason.iter().all(|t| [x_lo, x_hi, y_lo, y_hi].contains(t)),
        "the reason must name only the asserted bound atoms, got: {reason:?}"
    );
    // The gap is closed from below by x's upper bound and from above by
    // y's lower bound; those two must be among the cited atoms.
    assert!(
        reason.contains(&x_hi) || reason.contains(&x_lo),
        "the reason must cite a bound on x, got: {reason:?}"
    );
    assert!(
        reason.contains(&y_hi) || reason.contains(&y_lo),
        "the reason must cite a bound on y, got: {reason:?}"
    );

    // Same probe, overlapping ranges: y in [2, 5] admits y = 3 = x.
    let mut solver = ArithSolver::lra();
    solver.intern(x);
    solver.intern(y);
    solver.assert_ge(&[(x, Rational64::one())], Rational64::from_integer(3), x_lo);
    solver.assert_le(&[(x, Rational64::one())], Rational64::from_integer(3), x_hi);
    solver.assert_ge(&[(y, Rational64::one())], Rational64::from_integer(2), y_lo);
    solver.assert_le(&[(y, Rational64::one())], Rational64::from_integer(5), y_hi);
    assert!(matches!(
        solver.check().expect("check should succeed"),
        TheoryResult::Sat
    ));

    assert!(
        solver.entailed_disequal_reason(x, y).is_none(),
        "x = 3 lies inside y's range [2, 5], so x != y is NOT entailed"
    );
}

/// A term the solver has never interned has no bounds to reason from, so
/// the probe must decline rather than treat "unknown" as "disequal".
#[test]
fn test_pr30_entailed_disequal_reason_declines_unknown_terms() {
    let x = TermId::new(1);
    let stranger = TermId::new(9_999);
    let bound = TermId::new(101);

    let mut solver = ArithSolver::lra();
    solver.intern(x);
    solver.assert_ge(
        &[(x, Rational64::one())],
        Rational64::from_integer(3),
        bound,
    );
    solver.assert_le(
        &[(x, Rational64::one())],
        Rational64::from_integer(3),
        bound,
    );
    solver.check().expect("check should succeed");

    assert!(
        solver.entailed_disequal_reason(x, stranger).is_none(),
        "an uninterned term must never yield an entailed disequality"
    );
}

#[test]
fn test_no_entailed_equality_bidirectional() {
    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let reason = TermId::new(100);

    // Intern both so they appear in var_to_term.
    solver.intern(x);
    solver.intern(y);

    // x <= y
    solver.assert_le(
        &[(x, Rational64::one()), (y, -Rational64::one())],
        Rational64::from_integer(0),
        reason,
    );
    // y <= x
    solver.assert_le(
        &[(y, Rational64::one()), (x, -Rational64::one())],
        Rational64::from_integer(0),
        reason,
    );

    let sat = solver.check().expect("check should succeed");
    assert!(matches!(sat, TheoryResult::Sat), "Expected SAT");

    // Both x < y and x > y should be infeasible — equality is entailed.
    let eqs = solver.derive_shared_equalities();
    let has_xy = eqs
        .iter()
        .any(|e| (e.lhs == x && e.rhs == y) || (e.lhs == y && e.rhs == x));
    assert!(
        has_xy,
        "Expected entailed equality between x and y, got: {:?}",
        eqs
    );
}

/// x <= y alone should NOT yield an entailed equality (y could be > x).
#[test]
fn test_no_entailed_equality_one_direction_only() {
    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let reason = TermId::new(100);

    solver.intern(x);
    solver.intern(y);

    // x <= y only (one direction)
    solver.assert_le(
        &[(x, Rational64::one()), (y, -Rational64::one())],
        Rational64::from_integer(0),
        reason,
    );

    solver.check().expect("check should succeed");

    let eqs = solver.derive_shared_equalities();
    let has_xy = eqs
        .iter()
        .any(|e| (e.lhs == x && e.rhs == y) || (e.lhs == y && e.rhs == x));
    assert!(
        !has_xy,
        "Should NOT derive x=y from x<=y alone; got: {:?}",
        eqs
    );
}

/// notify_equality(x, y) followed by check should enforce x = y:
/// asserting x < y should then be UNSAT.
#[test]
fn test_notify_equality_enforces_equality() {
    use crate::theory::{EqualityNotification, TheoryCombination};

    let mut solver = ArithSolver::lra();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let reason = TermId::new(100);

    solver.intern(x);
    solver.intern(y);

    // Notify x = y
    let eq = EqualityNotification {
        lhs: x,
        rhs: y,
        reason: Some(reason),
    };
    let accepted = solver.notify_equality(eq);
    assert!(accepted, "notify_equality should accept x=y");

    // After asserting x=y, adding x < y should yield UNSAT.
    solver.push();
    solver.assert_lt(
        &[(x, Rational64::one()), (y, -Rational64::one())],
        Rational64::from_integer(0),
        reason,
    );
    let result = solver.check().expect("check should not error");
    assert!(
        matches!(result, TheoryResult::Unsat(_)),
        "Expected UNSAT when x=y is enforced and x<y is added; got {:?}",
        result
    );
    solver.pop();
}

// ---- push/pop state-rollback regression (term_to_var / var_to_term) ----

/// `pop()` must roll back `term_to_var` in lockstep with `var_to_term`.
///
/// Before the fix, `pop()` truncated `var_to_term` but left stale
/// `term_to_var` entries behind. Because the simplex recycles VarIds across
/// a pop, those stale entries made `intern()` replay indices that now belong
/// to a different (or not-yet-created) variable. This test inspects the
/// internal maps directly to prove the two stay consistent.
#[test]
fn regression_pop_rolls_back_term_to_var() {
    let mut solver = ArithSolver::lra();
    let a = TermId::new(1);
    let b = TermId::new(2);
    let c = TermId::new(3);

    // Intern `a` at the base level.
    let va = solver.intern(a);
    assert_eq!(va, 0);

    solver.push();
    // Intern two more terms inside the scope.
    let vb = solver.intern(b);
    let vc = solver.intern(c);
    assert_eq!(vb, 1);
    assert_eq!(vc, 2);
    assert_eq!(solver.var_to_term.len(), 3);
    assert_eq!(solver.term_to_var.len(), 3);

    solver.pop();

    // `var_to_term` is truncated back to just `[a]`.
    assert_eq!(solver.var_to_term.len(), 1);
    // `term_to_var` must be rolled back in lockstep: the scope-local terms
    // `b` and `c` are gone, only `a` remains.
    assert_eq!(solver.term_to_var.len(), 1);
    assert!(solver.term_to_var.contains_key(&a));
    assert!(!solver.term_to_var.contains_key(&b));
    assert!(!solver.term_to_var.contains_key(&c));

    // The core invariant: NO surviving mapping points at a truncated
    // (out-of-range) variable index.
    let live = solver.var_to_term.len() as VarId;
    for (&term, &var) in &solver.term_to_var {
        assert!(
            var < live,
            "term {term:?} maps to stale var {var} >= live var count {live}"
        );
    }

    // Re-interning the truncated terms yields FRESH valid indices.
    let vb2 = solver.intern(b);
    assert_eq!(vb2, 1, "re-interned `b` should take the next fresh index");
    assert!((vb2 as usize) < solver.var_to_term.len());
    let vc2 = solver.intern(c);
    assert_eq!(vc2, 2, "re-interned `c` should take the next fresh index");
    assert_ne!(vb2, vc2);
}

/// A fresh term interned after a pop must NOT collide with a stale-but-since-
/// re-interned term that used to hold the recycled index.
///
/// This is the recycled-index hazard the fix removes, observable purely
/// through the public `intern()` API: intern `a`, push, intern `b`, pop —
/// then intern a brand-new `c` (which the simplex hands the index `b` used
/// to occupy) and finally re-intern `b`. With the stale mapping still
/// present, `intern(b)` would return the same index as `c`.
#[test]
fn regression_pop_no_recycled_index_collision() {
    let mut solver = ArithSolver::lra();
    let a = TermId::new(11);
    let b = TermId::new(22);
    let c = TermId::new(33);

    let _va = solver.intern(a);
    solver.push();
    let _vb = solver.intern(b);
    solver.pop();

    // `c` is new: the simplex hands it the index `b` used to occupy.
    let vc = solver.intern(c);
    // `b` was truncated: re-interning must allocate a *different* fresh index.
    let vb2 = solver.intern(b);
    assert_ne!(
        vc, vb2,
        "recycled var index {vc} collided with re-interned truncated term"
    );
}

/// Regression (GitHub issue #12): in LRA the assignment for a variable
/// pinned at a *strict* bound is a delta-rational `r ± δ`.  `value()` must
/// instantiate `δ` with a concrete positive rational, otherwise it reports
/// `x = 0` for `x > 0` — a witness that violates the asserted constraint.
#[test]
fn regression_lra_strict_bound_model_instantiates_delta() {
    let mut solver = ArithSolver::lra();
    let x = TermId::new(1);
    let reason = TermId::new(100);

    // x > 0
    solver.assert_gt(&[(x, Rational64::one())], Rational64::zero(), reason);
    assert!(matches!(solver.check(), Ok(TheoryResult::Sat)));

    let value = solver.value(x).expect("x must have a model value");
    assert!(
        value > Rational64::zero(),
        "model x = {value} violates x > 0"
    );
}

/// Both ends of a strict range must be respected simultaneously: the
/// instantiated delta has to keep `0 < x < 1/2` genuinely inside the range.
#[test]
fn regression_lra_strict_range_model_inside_bounds() {
    let mut solver = ArithSolver::lra();
    let x = TermId::new(1);
    let lo = TermId::new(100);
    let hi = TermId::new(101);
    let half = Rational64::new(1, 2);

    solver.assert_gt(&[(x, Rational64::one())], Rational64::zero(), lo);
    solver.assert_lt(&[(x, Rational64::one())], half, hi);
    assert!(matches!(solver.check(), Ok(TheoryResult::Sat)));

    let value = solver.value(x).expect("x must have a model value");
    assert!(
        value > Rational64::zero() && value < half,
        "model x = {value} is outside the strict range (0, 1/2)"
    );
}

/// `2x = 1` over an `Int` `x`, with the integrality deferred to the caller
/// (re-fix pass 16, `mixed` module "Deferred integrality"): the solver runs
/// as with no mark — the relaxation `x = 1/2` is feasible, `value()` reads
/// it unrounded — and names `x` as the fractional marked term with its split
/// bounds; taken back, the same constraint is refuted.
#[test]
fn deferred_integrality_leaves_the_relaxation_to_the_caller() {
    let x = TermId::new(1);
    let reason = TermId::new(100);
    let two = Rational64::from_integer(2);

    let mut deferred = ArithSolver::lra();
    deferred.mark_int_term(x);
    deferred.set_defer_integrality(true);
    assert!(deferred.integrality_deferred());
    deferred.assert_eq(&[(x, two)], Rational64::one(), reason);
    let result = deferred
        .check_integrality()
        .expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Sat), "{result:?}");
    assert_eq!(deferred.value(x), Some(Rational64::new(1, 2)));
    assert_eq!(
        deferred.first_non_integral_int_term(),
        Some((x, Rational64::zero(), Rational64::one()))
    );

    let mut decided = ArithSolver::lra();
    decided.mark_int_term(x);
    decided.assert_eq(&[(x, two)], Rational64::one(), reason);
    let result = decided
        .check_integrality()
        .expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)), "{result:?}");
}

/// A strict bound over an `Int` term is not tightened while the integrality is
/// deferred: `2 < x < 3` keeps `x` at `2 + δ` or `3 − δ`, which the caller
/// reads as the split `x ≤ 2 ∨ x ≥ 3`.
#[test]
fn deferred_integrality_keeps_a_strict_bound_at_its_infinitesimal() {
    let x = TermId::new(1);
    let reason = TermId::new(100);
    let mut solver = ArithSolver::lra();
    solver.mark_int_term(x);
    solver.set_defer_integrality(true);
    solver.assert_lt(
        &[(x, Rational64::one())],
        Rational64::from_integer(3),
        reason,
    );
    solver.assert_gt(
        &[(x, Rational64::one())],
        Rational64::from_integer(2),
        reason,
    );
    let result = solver
        .check_integrality()
        .expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Sat), "{result:?}");
    assert_eq!(
        solver.first_non_integral_int_term(),
        Some((x, Rational64::from_integer(2), Rational64::from_integer(3)))
    );
}
