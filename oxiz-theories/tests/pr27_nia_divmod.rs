//! Regression tests for Euclidean `div`/`mod` support in the QF_NIA
//! dispatcher (`oxiz_theories::nlsat::dispatch_nia_constraints` /
//! `TermPolyTranslator::ensure_divmod_witness`).
//!
//! ## Scope
//!
//! This file exercises the *nonlinear* dispatch path specifically: every
//! case here needs genuine nonlinearity (a product of two non-constant
//! factors) alongside a `div`/`mod` term, because `dispatch_nia_constraints`
//! only engages when `term_is_nonlinear` sees one — a purely linear
//! `div`/`mod` problem is deliberately left to `oxiz-solver`'s
//! `arith_axioms` ground-lemma encoder (the simplex/CDCL(T) path), which
//! already handles it and is not touched here. The Euclidean convention
//! itself (which value `div`/`mod` actually produce, including `get-value`)
//! is pinned end-to-end on that LIA path instead, in
//! `oxiz-solver/tests/pr27_divmod_semantics.rs` — this file only checks that
//! the *nonlinear* dispatcher reaches the correct sat/unsat verdict.
//!
//! Each case here was hand-verified against the pre-fix tree: before the
//! Euclidean encoding, `contains_non_polynomial_ops` treated every
//! `div`/`mod` occurrence as unsupported, so `dispatch_nia_constraints`
//! either never trusted its own `Unsat` (`has_unsupported_ops` stayed set)
//! or fell through to `None` (`extract_poly_atoms` marked the atom
//! `incomplete` when `translate` returned `None` for the `div`/`mod`
//! sub-term) and the caller fell back to CDCL(T), which cannot handle the
//! nonlinear part either — so these formulas answered `unknown` end to end.
//!
//! ## Feature gate
//!
//! The whole file exercises `oxiz_theories::nlsat`, which exists only under the
//! `nlsat` feature (on by default) -- that feature is what pulls the
//! `oxiz-nlsat` crate into the graph. A `--no-default-features --features std`
//! build has no such module, so this file compiles to nothing there.
#![cfg(feature = "nlsat")]

use oxiz_core::ast::TermManager;
use oxiz_theories::nlsat::{NlDispatchResult, dispatch_nia_constraints};

/// `x*y = 12 ∧ x mod 5 = 2`: nonlinear (product of two distinct variables)
/// plus a positive-divisor `mod`. Satisfiable, e.g. `x=2, y=6`.
#[test]
fn test_pr27_nia_divmod_positive_divisor_is_sat() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let y = tm.mk_var("y", int_sort);
    let xy = tm.mk_mul([x, y]);
    let twelve = tm.mk_int(12);
    let eq_product = tm.mk_eq(xy, twelve);
    let five = tm.mk_int(5);
    let mod_x = tm.mk_mod(x, five);
    let two = tm.mk_int(2);
    let eq_mod = tm.mk_eq(mod_x, two);
    let assertion = tm.mk_and([eq_product, eq_mod]);

    assert!(
        matches!(
            dispatch_nia_constraints(&[assertion], &tm, true),
            Some(NlDispatchResult::Sat { .. })
        ),
        "x*y=12 ∧ x mod 5=2 is satisfiable (x=2, y=6)"
    );
}

/// `x*x >= 100 ∧ -5<=x<=5` is unsatisfiable on its own (max `x^2` in range is
/// 25); `x mod 3 = 1` is added to exercise the Euclidean witness atoms
/// coexisting with (not being the cause of) the conflict. Every *problem*
/// atom here is univariate in `x`, so the trustworthy-`Unsat` gate is not
/// blocked by the pre-existing "no multivariate problem atom" restriction —
/// only the synthetic witness identity is multivariate, and that is exactly
/// what `PolyAtom::synthetic` exists to permit.
#[test]
fn test_pr27_nia_divmod_alongside_bound_conflict_is_unsat() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let xx = tm.mk_mul([x, x]);
    let hundred = tm.mk_int(100);
    let ge_bound = tm.mk_ge(xx, hundred);
    let five = tm.mk_int(5);
    let neg_five = tm.mk_neg(five);
    let x_ge = tm.mk_ge(x, neg_five);
    let x_le = tm.mk_le(x, five);
    let three = tm.mk_int(3);
    let mod_x = tm.mk_mod(x, three);
    let one = tm.mk_int(1);
    let eq_mod = tm.mk_eq(mod_x, one);
    let assertion = tm.mk_and([ge_bound, x_ge, x_le, eq_mod]);

    assert_eq!(
        dispatch_nia_constraints(&[assertion], &tm, true),
        Some(NlDispatchResult::Unsat),
        "x^2>=100 has no solution with -5<=x<=5"
    );
}

/// Negative constant divisor: `x*x = 9 ∧ x mod (-4) = 3`. `x` must be `3` or
/// `-3` (real/integer roots of `x^2=9`); Euclidean semantics give
/// `3 mod (-4) = 3` (`3 = (-4)*0 + 3`) but `(-3) mod (-4) = 1`
/// (`-3 = (-4)*1 + 1`), so only `x=3` extends. Satisfiable.
#[test]
fn test_pr27_nia_divmod_negative_divisor_is_sat() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let xx = tm.mk_mul([x, x]);
    let nine = tm.mk_int(9);
    let eq_square = tm.mk_eq(xx, nine);
    let four = tm.mk_int(4);
    let neg_four = tm.mk_neg(four);
    let mod_x = tm.mk_mod(x, neg_four);
    let three = tm.mk_int(3);
    let eq_mod = tm.mk_eq(mod_x, three);
    let assertion = tm.mk_and([eq_square, eq_mod]);

    assert!(
        matches!(
            dispatch_nia_constraints(&[assertion], &tm, true),
            Some(NlDispatchResult::Sat { .. })
        ),
        "x^2=9 ∧ x mod (-4)=3 is satisfiable (x=3)"
    );
}

/// Negative constant divisor, unsatisfiable side: same square constraint but
/// requiring the remainder Euclidean semantics rule out for *both* roots.
/// `x^2=9 ∧ x mod (-4) = 2`: `3 mod (-4) = 3` and `(-3) mod (-4) = 1`,
/// neither is `2`.
#[test]
fn test_pr27_nia_divmod_negative_divisor_unreachable_remainder_is_unsat() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let xx = tm.mk_mul([x, x]);
    let nine = tm.mk_int(9);
    let eq_square = tm.mk_eq(xx, nine);
    let four = tm.mk_int(4);
    let neg_four = tm.mk_neg(four);
    let mod_x = tm.mk_mod(x, neg_four);
    let two = tm.mk_int(2);
    let eq_mod = tm.mk_eq(mod_x, two);
    let assertion = tm.mk_and([eq_square, eq_mod]);

    let result = dispatch_nia_constraints(&[assertion], &tm, true);
    // Whether this specific shape reaches a *trustworthy* certified `Unsat`
    // depends on the underlying CAD/branch-and-bound search's own power
    // (root isolation interacting with an extra linear identity is not
    // guaranteed to converge -- see the module-level note in
    // `oxiz-theories/src/nlsat.rs` and the deferred bounded-enumeration
    // discussion in the PR27 report). The one outcome that would be a
    // soundness bug is `Sat`: neither root's remainder is `2`.
    assert!(
        !matches!(result, Some(NlDispatchResult::Sat { .. })),
        "x^2=9 ∧ x mod (-4)=2 has no solution and must never be reported Sat"
    );
}

/// A zero divisor must not be given a (fabricated) Euclidean meaning --
/// SMT-LIB leaves `div`/`mod` by zero uninterpreted. The dispatcher must
/// fall through (`None`), never panic and never assert a wrong verdict, even
/// though `x*x=9` alone is easily decided.
#[test]
fn test_pr27_nia_divmod_zero_divisor_is_not_encoded() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let xx = tm.mk_mul([x, x]);
    let nine = tm.mk_int(9);
    let eq_square = tm.mk_eq(xx, nine);
    let zero = tm.mk_int(0);
    let mod_x = tm.mk_mod(x, zero);
    let five = tm.mk_int(5);
    let eq_mod = tm.mk_eq(mod_x, five);
    let assertion = tm.mk_and([eq_square, eq_mod]);

    // Must not panic (checked implicitly by reaching this point) and must
    // not fabricate a verdict either way.
    let result = dispatch_nia_constraints(&[assertion], &tm, true);
    assert_eq!(
        result, None,
        "a zero divisor has no polynomial encoding; the dispatcher must fall \
         through to CDCL(T) rather than guess"
    );
}

/// A symbolic (non-constant) divisor has no polynomial encoding either --
/// the defining identity `m = n*q + r` would itself be nonlinear in `n` and
/// `q`. Must fall through cleanly, not panic.
#[test]
fn test_pr27_nia_divmod_symbolic_divisor_is_not_encoded() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let n = tm.mk_var("n", int_sort);
    let xx = tm.mk_mul([x, x]);
    let nine = tm.mk_int(9);
    let eq_square = tm.mk_eq(xx, nine);
    let mod_x = tm.mk_mod(x, n);
    let two = tm.mk_int(2);
    let eq_mod = tm.mk_eq(mod_x, two);
    let assertion = tm.mk_and([eq_square, eq_mod]);

    let result = dispatch_nia_constraints(&[assertion], &tm, true);
    assert_eq!(
        result, None,
        "a symbolic divisor has no polynomial encoding; the dispatcher must \
         fall through to CDCL(T) rather than guess"
    );
}

/// The divisor-constant folding in `resolve_int_divisor` must agree with
/// `oxiz-solver`'s `arith_axioms::int_constant` on a *folded* (not bare
/// literal) divisor expression, since both encoders are asserted to treat
/// "the divisor is the constant `n`" identically (see `resolve_int_divisor`'s
/// doc comment). `(mod x (- (* 2 3) 1))` folds to divisor `5`, the same as
/// the bare-literal case above.
#[test]
fn test_pr27_nia_divmod_folded_divisor_expression_is_sat() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let y = tm.mk_var("y", int_sort);
    let xy = tm.mk_mul([x, y]);
    let twelve = tm.mk_int(12);
    let eq_product = tm.mk_eq(xy, twelve);
    let two = tm.mk_int(2);
    let three = tm.mk_int(3);
    let six = tm.mk_mul([two, three]);
    let one = tm.mk_int(1);
    let folded_five = tm.mk_sub(six, one); // (2*3) - 1 = 5
    let mod_x = tm.mk_mod(x, folded_five);
    let expected_rem = tm.mk_int(2);
    let eq_mod = tm.mk_eq(mod_x, expected_rem);
    let assertion = tm.mk_and([eq_product, eq_mod]);

    assert!(
        matches!(
            dispatch_nia_constraints(&[assertion], &tm, true),
            Some(NlDispatchResult::Sat { .. })
        ),
        "x*y=12 ∧ x mod ((2*3)-1)=2 is satisfiable (x=2, y=6), same divisor as the bare-literal case"
    );
}

/// `TermKind::Div` is shared by two unrelated operators: `Ints`' Euclidean
/// `div` and exact rational `/`, distinguished only by the node's own sort
/// (`mk_div`/`mk_mod` inherit it from the dividend). `ensure_divmod_witness`
/// must gate on that sort, not merely on whether the divisor resolves to an
/// integer constant -- a bare numeral divisor like `2` always parses as an
/// `IntConst` (see `oxiz-core/src/smtlib/parser/terms.rs`'s numeral leaf),
/// so `(/ v 2)` with `v` a `Real` reaches this dispatcher with an
/// `IntConst` divisor exactly like a genuine `Int` `(div n 2)` would.
///
/// `v = 1.5` satisfies `v*v = 2.25` (real, nonlinear, forces
/// `dispatch_nia_constraints` to engage) and, under real division,
/// `v/2 = 0.75`. If `(/ v 2)` were wrongly given the `Ints` Euclidean
/// encoding, the witnesses `q`, `r` it introduces are `Integer`-typed, so
/// the identity `v = 2*q + r` would force `v` itself to an integer value --
/// impossible for `v = 1.5` -- turning a genuinely satisfiable problem into
/// a false `Unsat`. This must never happen; falling through to `None`
/// (real division has no polynomial encoding here, same as before any
/// `div`/`mod` support existed) is the correct, sound outcome.
#[test]
fn test_pr27_nia_real_division_is_not_euclidean_encoded() {
    let mut tm = TermManager::new();
    let real_sort = tm.sorts.real_sort;
    let v = tm.mk_var("v", real_sort);
    let vv = tm.mk_mul([v, v]);
    let target_sq = tm.mk_real(num_rational::Rational64::new(9, 4)); // 2.25
    let eq_square = tm.mk_eq(vv, target_sq);
    let two = tm.mk_int(2); // bare numeral divisor: parses as IntConst
    let half_v = tm.mk_div(v, two);
    let target_half = tm.mk_real(num_rational::Rational64::new(3, 4)); // 0.75
    let eq_half = tm.mk_eq(half_v, target_half);
    let assertion = tm.mk_and([eq_square, eq_half]);

    let result = dispatch_nia_constraints(&[assertion], &tm, true);
    assert_ne!(
        result,
        Some(NlDispatchResult::Unsat),
        "v=1.5 genuinely satisfies v*v=2.25 and (real) v/2=0.75; a Real-sorted \
         Div must never be given Ints' Euclidean encoding"
    );
}
