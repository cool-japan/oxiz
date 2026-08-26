//! Tests for the quantifier tactics.
//!
//! Split out of the former single-file `tactic/quantifier.rs`. Pure code
//! motion.

use super::subst::substitute_single_var;
use super::*;
use crate::ast::{TermId, TermKind, TermManager};
use crate::interner::Spur;
use crate::sort::SortId;
use crate::tactic::{Goal, TacticResult};
use smallvec::SmallVec;

fn setup_manager() -> TermManager {
    TermManager::new()
}

/// Does `name` occur as a *free* variable anywhere in `term`?
///
/// Used by the substitution regression tests below to detect the failure mode
/// they exist for: a tactic reporting that it eliminated a variable while the
/// variable is still there, now unbound.
fn mentions_free_var(manager: &TermManager, term: TermId, name: &str) -> bool {
    manager
        .free_vars_including_patterns(term)
        .into_iter()
        .filter_map(|v| match manager.get(v).map(|t| &t.kind) {
            Some(TermKind::Var(n)) => Some(manager.resolve_str(*n).to_string()),
            _ => None,
        })
        .any(|n| n == name)
}

/// Run `DerTactic` over a one-assertion goal and return the rewritten
/// assertion (or the input, when the tactic reports `NotApplicable`).
fn der_once(manager: &mut TermManager, assertion: TermId) -> TermId {
    let goal = Goal::new(vec![assertion]);
    let result = DerTactic::new(manager)
        .apply_mut(&goal)
        .expect("DER must not fail");
    match result {
        TacticResult::SubGoals(goals) => goals
            .first()
            .and_then(|g| g.assertions.first().copied())
            .expect("one sub-goal with one assertion"),
        TacticResult::NotApplicable => assertion,
        other => panic!("unexpected DER result: {other:?}"),
    }
}

#[test]
fn test_ground_term_collector() {
    let mut manager = setup_manager();
    let int_sort = manager.sorts.int_sort;

    // Create some terms
    let x = manager.mk_var("x", int_sort);
    let one = manager.mk_int(1);
    let two = manager.mk_int(2);
    let sum = manager.mk_add([one, two]);

    // Collect from multiple terms
    let term_with_var = manager.mk_add([x, one]);

    let mut collector = GroundTermCollector::new();
    // Collect from ground sum (1 + 2)
    collector.collect(sum, &manager);
    // Collect from non-ground term (x + 1)
    collector.collect(term_with_var, &manager);

    // Should have collected 1, 2, and (1 + 2) from ground term
    assert!(collector.all_terms().contains(&one));
    assert!(collector.all_terms().contains(&two));
    assert!(collector.all_terms().contains(&sum)); // ground term (1 + 2)
    // Should NOT have collected x or (x + 1) since they contain free vars
    assert!(!collector.all_terms().contains(&x));
    assert!(!collector.all_terms().contains(&term_with_var));
}

#[test]
fn test_pattern_matching_simple() {
    let mut manager = setup_manager();
    let int_sort = manager.sorts.int_sort;

    // Create pattern: f(x) where x is bound
    let x = manager.mk_var("x", int_sort);
    let f_x = manager.mk_apply("f", [x], int_sort);

    // Create ground term: f(1)
    let one = manager.mk_int(1);
    let f_one = manager.mk_apply("f", [one], int_sort);

    // Create quantifier: forall x. P(f(x)) with pattern f(x)
    let x_name = manager.intern_str("x");
    let bound_vars: SmallVec<[(Spur, SortId); 2]> = smallvec::smallvec![(x_name, int_sort)];

    let matcher = PatternMatcher::new();
    let result = matcher.try_match_term(f_x, f_one, &bound_vars, &manager);

    assert!(result.is_some());
    let bindings = result.expect("should have bindings");
    assert_eq!(bindings.get(&x_name), Some(&one));
}

// ===========================================================================
// `substitute_single_var` regression tests
//
// The helper used to match only a 21-arm whitelist of `TermKind`s and end in
// `_ => term_id`, silently returning the term *unchanged* for every operator
// outside it (bit-vector, string, floating-point, `Mod`, `Xor`, `Distinct`,
// `Let`, `Match`, datatypes). Because `DerTactic` and `SkolemizationTactic`
// drop the binder they were eliminating, a dropped substitution does not
// merely degrade the result -- it leaves the eliminated variable *free* in a
// formula the tactic reports as an equisatisfiable rewrite, which can flip
// UNSAT to SAT (and, with a name collision, SAT to UNSAT).
// ===========================================================================

/// End-to-end: `∃x:BV8. (x = #x05 ∧ x <u #x01)` is UNSAT, but DER used to
/// rewrite it to `x <u #x01` with `x` free -- which is SAT (take `x = 0`).
#[test]
fn der_substitutes_through_a_bitvector_operator() {
    let mut m = setup_manager();
    let bv8 = m.sorts.bitvec(8);
    let x = m.mk_var("x", bv8);
    let five = m.mk_bitvec(5, 8);
    let one = m.mk_bitvec(1, 8);

    let eq = m.mk_eq(x, five);
    let lt = m.mk_bv_ult(x, one);
    let body = m.mk_and([eq, lt]);
    let exists = m.mk_exists([("x", bv8)], body);

    let result = der_once(&mut m, exists);

    assert!(
        !mentions_free_var(&m, result, "x"),
        "DER dropped the ∃ binder, so x must have been substituted away"
    );
    let expected = m.mk_bv_ult(five, one);
    assert_eq!(result, expected, "x must be replaced by #x05 inside bvult");
}

/// Same shape with integer `Mod`, reached *through* an `Eq` (which the old
/// whitelist did handle) so the gap is specifically the unlisted inner
/// operator: `∃x. (x = 4 ∧ x mod 2 = 1)` is UNSAT; the old rewrite produced
/// `x mod 2 = 1` with `x` free, which is SAT (take `x = 1`).
#[test]
fn der_substitutes_through_mod_under_an_equality() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let four = m.mk_int(4);
    let two = m.mk_int(2);
    let one = m.mk_int(1);

    let eq = m.mk_eq(x, four);
    let modulo = m.mk_mod(x, two);
    let cmp = m.mk_eq(modulo, one);
    let body = m.mk_and([eq, cmp]);
    let exists = m.mk_exists([("x", int_sort)], body);

    let result = der_once(&mut m, exists);

    assert!(!mentions_free_var(&m, result, "x"));
    let expected_mod = m.mk_mod(four, two);
    let expected = m.mk_eq(expected_mod, one);
    assert_eq!(result, expected);
}

/// `Distinct` was another unlisted n-ary operator.
#[test]
fn der_substitutes_through_distinct() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let y = m.mk_var("y", int_sort);
    let five = m.mk_int(5);

    let eq = m.mk_eq(x, five);
    let distinct = m.mk_distinct([x, y]);
    let body = m.mk_and([eq, distinct]);
    let exists = m.mk_exists([("x", int_sort)], body);

    let result = der_once(&mut m, exists);

    assert!(!mentions_free_var(&m, result, "x"));
    let expected = m.mk_distinct([five, y]);
    assert_eq!(result, expected);
}

/// A string operator: `str.len` was unlisted.
#[test]
fn der_substitutes_through_a_string_operator() {
    let mut m = setup_manager();
    let str_sort = m.sorts.string_sort();
    let x = m.mk_var("x", str_sort);
    let ab = m.mk_string_lit("ab");
    let three = m.mk_int(3);

    let eq = m.mk_eq(x, ab);
    let len = m.mk_str_len(x);
    let cmp = m.mk_eq(len, three);
    let body = m.mk_and([eq, cmp]);
    let exists = m.mk_exists([("x", str_sort)], body);

    let result = der_once(&mut m, exists);

    assert!(!mentions_free_var(&m, result, "x"));
    let expected_len = m.mk_str_len(ab);
    let expected = m.mk_eq(expected_len, three);
    assert_eq!(result, expected);
}

/// A datatype constructor argument: `DtConstructor` was unlisted, and the
/// `DtSelector` wrapping it too.
#[test]
fn der_substitutes_through_a_datatype_constructor() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let dt_sort = m.sorts.mk_datatype_sort("Pair");
    let x = m.mk_var("x", int_sort);
    let seven = m.mk_int(7);
    let zero = m.mk_int(0);

    let eq = m.mk_eq(x, seven);
    let pair = m.mk_dt_constructor("Pair", vec![x, zero], dt_sort);
    let first = m.mk_dt_selector("first", pair, int_sort);
    let cmp = m.mk_eq(first, zero);
    let body = m.mk_and([eq, cmp]);
    let exists = m.mk_exists([("x", int_sort)], body);

    let result = der_once(&mut m, exists);

    assert!(!mentions_free_var(&m, result, "x"));
    let expected_pair = m.mk_dt_constructor("Pair", vec![seven, zero], dt_sort);
    let expected_first = m.mk_dt_selector("first", expected_pair, int_sort);
    let expected = m.mk_eq(expected_first, zero);
    assert_eq!(result, expected);
}

// ===========================================================================
// `DerTactic` existential-path regression tests
//
// `remove_from_and_and_substitute` used to drop *every* conjunct that was an
// equality mentioning the eliminated variable (`is_equality_for_var`), not
// just the one it substituted. The ∀ counterpart
// (`remove_from_or_and_substitute`) was already precise, matching only the
// exact disequality being resolved via `is_target_diseq`; the ∃ path never
// got the same treatment, so every *other* equality on x was silently lost.
// Losing a conjunct weakens the formula, so this turns UNSAT into SAT.
// ===========================================================================

/// `∃x. (x = 5 ∧ x = 6)` is UNSAT (nothing equals both). DER used to drop
/// *both* equalities -- the one it resolved and the surviving constraint --
/// leaving an empty conjunction, i.e. `true`.
#[test]
fn der_keeps_a_second_equality_on_the_eliminated_variable() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let five = m.mk_int(5);
    let six = m.mk_int(6);

    let eq5 = m.mk_eq(x, five);
    let eq6 = m.mk_eq(x, six);
    let body = m.mk_and([eq5, eq6]);
    let exists = m.mk_exists([("x", int_sort)], body);

    let result = der_once(&mut m, exists);

    let kind = m.get(result).map(|t| t.kind.clone());
    assert!(
        matches!(kind, Some(TermKind::False)),
        "∃x.(x = 5 ∧ x = 6) is UNSAT; DER produced {kind:?}"
    );
}

/// The surviving equality must be *kept and substituted into*, not merely
/// kept: `∃x. (x = 5 ∧ x = f(y))` must become `5 = f(y)`, which constrains
/// `y`. Dropping it yielded `true` -- again UNSAT-to-SAT for any context
/// that forces `f(y) ≠ 5`.
#[test]
fn der_substitutes_into_a_kept_equality_on_the_eliminated_variable() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let y = m.mk_var("y", int_sort);
    let five = m.mk_int(5);
    let f_y = m.mk_apply("f", [y], int_sort);

    let eq5 = m.mk_eq(x, five);
    let eq_f = m.mk_eq(x, f_y);
    let body = m.mk_and([eq5, eq_f]);
    let exists = m.mk_exists([("x", int_sort)], body);

    let result = der_once(&mut m, exists);

    assert!(!mentions_free_var(&m, result, "x"));
    let expected = m.mk_eq(five, f_y);
    assert_eq!(
        result, expected,
        "the kept equality must have x replaced by 5, giving 5 = f(y)"
    );
}

/// An equality on x whose other side *also* mentions x (`x = g(x)`) is not
/// eliminable, but `is_equality_for_var` matched it and threw it away.
/// `∃x. (x = 5 ∧ x = g(x))` must become `5 = g(5)`.
#[test]
fn der_keeps_a_non_eliminable_equality_on_the_eliminated_variable() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let five = m.mk_int(5);
    let g_x = m.mk_apply("g", [x], int_sort);

    let eq5 = m.mk_eq(x, five);
    let eq_g = m.mk_eq(x, g_x);
    let body = m.mk_and([eq5, eq_g]);
    let exists = m.mk_exists([("x", int_sort)], body);

    let result = der_once(&mut m, exists);

    assert!(!mentions_free_var(&m, result, "x"));
    let g_five = m.mk_apply("g", [five], int_sort);
    let expected = m.mk_eq(five, g_five);
    assert_eq!(result, expected, "x = g(x) must survive as 5 = g(5)");
}

/// `StatelessDerTactic` is a separate public entry point; it must inherit the
/// fix (it delegates to `DerTactic`, so this pins that it keeps doing so).
#[test]
fn stateless_der_keeps_a_second_equality_too() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let five = m.mk_int(5);
    let six = m.mk_int(6);

    let eq5 = m.mk_eq(x, five);
    let eq6 = m.mk_eq(x, six);
    let body = m.mk_and([eq5, eq6]);
    let exists = m.mk_exists([("x", int_sort)], body);
    let goal = Goal::new(vec![exists]);

    let result = StatelessDerTactic::new()
        .apply(&goal, &mut m)
        .expect("DER must not fail");
    let TacticResult::SubGoals(goals) = result else {
        panic!("expected sub-goals, got {result:?}");
    };
    let rewritten = goals
        .first()
        .and_then(|g| g.assertions.first().copied())
        .expect("one sub-goal with one assertion");

    let kind = m.get(rewritten).map(|t| t.kind.clone());
    assert!(
        matches!(kind, Some(TermKind::False)),
        "∃x.(x = 5 ∧ x = 6) is UNSAT; StatelessDerTactic produced {kind:?}"
    );
}

/// The resolved equality itself must still be dropped rather than kept as the
/// tautology `t = t` -- pins that the precision fix did not turn DER into a
/// no-op on its own rule. `∃x. (x = 5 ∧ x > 3)` must become `5 > 3`, i.e.
/// `true`, with no residual `5 = 5` conjunct.
#[test]
fn der_still_drops_the_resolved_equality() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let five = m.mk_int(5);
    let three = m.mk_int(3);

    let eq5 = m.mk_eq(x, five);
    let gt = m.mk_gt(x, three);
    let body = m.mk_and([eq5, gt]);
    let exists = m.mk_exists([("x", int_sort)], body);

    let result = der_once(&mut m, exists);

    let expected = m.mk_gt(five, three);
    assert_eq!(result, expected, "expected just 5 > 3");
}

/// `Let` was unlisted, so a `let` body was skipped whole. Called directly
/// (rather than through DER) because the substituted variable occurs only in
/// the `let`'s *bound value*, which is the position a binder-unaware walk is
/// most likely to get wrong.
#[test]
fn substitute_descends_into_a_let() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let a = m.mk_var("a", int_sort);
    let zero = m.mk_int(0);
    let nine = m.mk_int(9);

    // (let ((a x)) (> a 0))
    let inner = m.mk_gt(a, zero);
    let let_term = m.mk_let([("a", x)], inner);

    let x_name = m.intern_str("x");
    let result = substitute_single_var(&mut m, let_term, x_name, nine);

    assert!(!mentions_free_var(&m, result, "x"));
    let expected = m.mk_let([("a", nine)], inner);
    assert_eq!(result, expected);
}

/// Shadowing must still be respected: a `let`-bound occurrence of the
/// substituted name is a *different* variable and must not be replaced.
#[test]
fn substitute_respects_let_shadowing() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let zero = m.mk_int(0);
    let one = m.mk_int(1);
    let nine = m.mk_int(9);

    // (let ((x 1)) (> x 0)) -- the inner x is bound by the let.
    let inner = m.mk_gt(x, zero);
    let let_term = m.mk_let([("x", one)], inner);

    let x_name = m.intern_str("x");
    let result = substitute_single_var(&mut m, let_term, x_name, nine);

    assert_eq!(
        result, let_term,
        "the let-bound x is a different variable and must be left alone"
    );
}

/// Shadowing under a quantifier (this the old code did get right; pinned so
/// the delegation cannot regress it).
#[test]
fn substitute_respects_quantifier_shadowing() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let zero = m.mk_int(0);
    let nine = m.mk_int(9);

    let body = m.mk_gt(x, zero);
    let forall = m.mk_forall([("x", int_sort)], body);

    let x_name = m.intern_str("x");
    let result = substitute_single_var(&mut m, forall, x_name, nine);

    assert_eq!(result, forall, "x is re-bound here; nothing to substitute");
}

/// Capture avoidance: `(forall ((y Int)) (P x y))[x := y]` must alpha-rename
/// the bound `y`, otherwise the substituted free `y` is captured and the
/// rewrite is not equisatisfiable. The old walk descended into the body with
/// no renaming at all and produced `(forall ((y Int)) (P y y))`.
#[test]
fn substitute_under_a_binder_avoids_capture() {
    let mut m = setup_manager();
    let int_sort = m.sorts.int_sort;
    let bool_sort = m.sorts.bool_sort;
    let x = m.mk_var("x", int_sort);
    let y = m.mk_var("y", int_sort);

    let body = m.mk_apply("P", [x, y], bool_sort);
    let forall = m.mk_forall([("y", int_sort)], body);

    let x_name = m.intern_str("x");
    let result = substitute_single_var(&mut m, forall, x_name, y);

    let kind = m
        .get(result)
        .map(|t| t.kind.clone())
        .expect("result term must exist");
    let TermKind::Forall {
        vars,
        body: new_body,
        ..
    } = kind
    else {
        panic!("expected a Forall, got {kind:?}");
    };
    let (renamed, sort) = vars
        .first()
        .map(|&(n, s)| (m.resolve_str(n).to_string(), s))
        .expect("exactly one bound variable");
    assert_ne!(renamed, "y", "the bound y must be alpha-renamed");

    // Body was `P(x, y)`; after `x := y` with the binder renamed to `y!N` it
    // must be `P(y, y!N)` -- the first argument is the substituted *free* y,
    // the second the renamed bound one.
    let fresh = m.mk_var(&renamed, sort);
    let expected_body = m.mk_apply("P", [y, fresh], bool_sort);
    assert_eq!(
        new_body, expected_body,
        "the substituted free y must not be captured by the binder"
    );
}

/// Skolemization must not leave the existential variable behind: the binder
/// is dropped, so a skipped substitution turns a bound variable into a free
/// one that can collide with an unrelated free variable of the same name
/// elsewhere in the goal (SAT -> UNSAT).
#[test]
fn skolemization_substitutes_through_a_bitvector_operator() {
    let mut m = setup_manager();
    let bv8 = m.sorts.bitvec(8);
    let x = m.mk_var("x", bv8);
    let one = m.mk_bitvec(1, 8);

    let body = m.mk_bv_ult(x, one);
    let exists = m.mk_exists([("x", bv8)], body);
    let goal = Goal::new(vec![exists]);

    let result = SkolemizationTactic::new(&mut m)
        .apply_mut(&goal)
        .expect("skolemization must not fail");
    let TacticResult::SubGoals(goals) = result else {
        panic!("expected sub-goals, got {result:?}");
    };
    let skolemized = goals
        .first()
        .and_then(|g| g.assertions.first().copied())
        .expect("one sub-goal with one assertion");

    assert!(
        !mentions_free_var(&m, skolemized, "x"),
        "the existential x must be replaced by a Skolem constant, not left free"
    );
}

/// Run `f` on a dedicated thread with a 128 KiB stack -- far smaller than the
/// default main-thread stack.
///
/// A stack overflow aborts the whole process rather than failing one test, so
/// for the deep-input test below the call *returning at all* is itself the
/// assertion.
fn run_on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(f)
        .expect("spawning the constrained-stack test thread should succeed")
        .join()
        .expect("the constrained-stack thread must not panic")
}

/// The old helper recursed natively once per level of term nesting with no
/// depth guard of any kind, so a deep (but perfectly valid) term aborted the
/// process.
#[test]
fn substitute_survives_a_deeply_nested_term_on_a_tiny_stack() {
    const DEPTH: usize = 12_500;

    let (reached_leaf, old_leaf_gone) = run_on_small_stack(|| {
        let mut m = setup_manager();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let nine = m.mk_int(9);

        // f(f(f(...f(x)...))): uninterpreted application never folds, and is
        // built iteratively so that construction itself cannot overflow.
        let mut chain = x;
        for _ in 0..DEPTH {
            chain = m.mk_apply("f", [chain], int_sort);
        }

        let x_name = m.intern_str("x");
        let result = substitute_single_var(&mut m, chain, x_name, nine);

        // Peel the layers back off and check the leaf really changed.
        let mut current = result;
        for _ in 0..DEPTH {
            let kind = m.get(current).map(|t| t.kind.clone());
            match kind {
                Some(TermKind::Apply { args, .. }) if args.len() == 1 => current = args[0],
                _ => break,
            }
        }
        (current == nine, current != x)
    });

    assert!(
        reached_leaf,
        "substitution must reach the bottom of the chain"
    );
    assert!(old_leaf_gone, "the old leaf x must not remain");
}

// ---------------------------------------------------------------------------
// Group C1: explicit-stack conversions (DER, Skolemization, instantiation)
// ---------------------------------------------------------------------------

/// `DerTactic::apply_der` used to recurse once per level of *term* nesting and
/// guard that with `DerConfig::max_depth` (default 10). It now walks with an
/// explicit heap stack, so a Boolean skeleton far deeper than any cap must
/// simply return. The assertion is that the call *returns at all*: a native
/// stack overflow aborts the whole process rather than failing the test.
#[test]
fn der_survives_a_deeply_nested_boolean_skeleton_on_a_tiny_stack() {
    const DEPTH: usize = 7_500;

    let handle = std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let mut m = setup_manager();
            let bool_sort = m.sorts.bool_sort;
            let int_sort = m.sorts.int_sort;

            // At the bottom: forall x. (x != 5 or P(x)) -- a genuine DER site.
            let x = m.mk_var("x", int_sort);
            let five = m.mk_int(5);
            let eq = m.mk_eq(x, five);
            let diseq = m.mk_not(eq);
            let p_x = m.mk_apply("P", [x], bool_sort);
            let body = m.mk_or([diseq, p_x]);
            let mut current = m.mk_forall([("x", int_sort)], body);

            // Bury it under a deep Not/And alternation.
            for i in 0..DEPTH {
                current = if i % 2 == 0 {
                    m.mk_not(current)
                } else {
                    let t = m.mk_true();
                    m.mk_and([current, t])
                };
            }

            let goal = Goal::new(vec![current]);
            let mut tactic = DerTactic::new(&mut m);
            tactic.apply_mut(&goal).is_ok()
        })
        .expect("test thread must spawn");
    let finished = handle.join();

    assert!(
        matches!(finished, Ok(true)),
        "DER must return on a deeply nested goal instead of overflowing"
    );
}

/// Semantic pin for the cap removal: DER now reaches an eliminable
/// disequality that sits far below the old `max_depth = 10` term-nesting
/// bound. `not(not(...(forall x. (x != 5 or P(x)))...))` with 40 negations
/// must still become `not(not(...P(5)...))`.
#[test]
fn der_eliminates_below_the_old_depth_cap() {
    let mut m = setup_manager();
    let bool_sort = m.sorts.bool_sort;
    let int_sort = m.sorts.int_sort;

    let x = m.mk_var("x", int_sort);
    let five = m.mk_int(5);
    let eq = m.mk_eq(x, five);
    let diseq = m.mk_not(eq);
    let p_x = m.mk_apply("P", [x], bool_sort);
    let body = m.mk_or([diseq, p_x]);
    let quant = m.mk_forall([("x", int_sort)], body);

    // 40 negations -- well past the old cap of 10.
    let mut nested = quant;
    for _ in 0..40 {
        nested = m.mk_not(nested);
    }

    let goal = Goal::new(vec![nested]);
    let result = {
        let mut tactic = DerTactic::new(&mut m);
        tactic
            .apply_mut(&goal)
            .expect("test operation should succeed")
    };

    let TacticResult::SubGoals(goals) = result else {
        panic!("DER should have fired below the old depth cap, got {result:?}");
    };
    let produced = goals[0].assertions[0];

    // Expected: the same 40 negations wrapped around P(5).
    let p_five = m.mk_apply("P", [five], bool_sort);
    let mut expected = p_five;
    for _ in 0..40 {
        expected = m.mk_not(expected);
    }
    assert_eq!(produced, expected);
    assert!(!mentions_free_var(&m, produced, "x"));
}

/// Skolemization's three-way mutual recursion is now an explicit stack. A
/// deeply nested Boolean skeleton with an existential at the bottom must
/// return (and actually Skolemize).
#[test]
fn skolemization_survives_a_deeply_nested_formula_on_a_tiny_stack() {
    const DEPTH: usize = 7_500;

    let handle = std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let mut m = setup_manager();
            let bool_sort = m.sorts.bool_sort;
            let int_sort = m.sorts.int_sort;

            let y = m.mk_var("y", int_sort);
            let p_y = m.mk_apply("P", [y], bool_sort);
            let mut current = m.mk_exists([("y", int_sort)], p_y);

            for i in 0..DEPTH {
                current = if i % 2 == 0 {
                    let t = m.mk_true();
                    m.mk_and([current, t])
                } else {
                    let f = m.mk_false();
                    m.mk_or([current, f])
                };
            }

            let goal = Goal::new(vec![current]);
            let mut tactic = SkolemizationTactic::new(&mut m);
            matches!(tactic.apply_mut(&goal), Ok(TacticResult::SubGoals(_)))
        })
        .expect("test thread must spawn");
    let outcome = handle.join();

    assert!(
        matches!(outcome, Ok(true)),
        "Skolemization must return on a deeply nested goal instead of overflowing"
    );
}

/// Semantic pin for the Skolemization conversion: the polarity rules,
/// governing-universal arguments and per-goal fresh-name counter must be
/// exactly what the recursive version produced.
#[test]
fn skolemization_iterative_matches_the_recursive_semantics() {
    let mut m = setup_manager();
    let bool_sort = m.sorts.bool_sort;
    let int_sort = m.sorts.int_sort;

    // forall u. exists v. P(u, v)   ==>   forall u. P(u, sk!0(u))
    let u = m.mk_var("u", int_sort);
    let v = m.mk_var("v", int_sort);
    let p_uv = m.mk_apply("P", [u, v], bool_sort);
    let inner = m.mk_exists([("v", int_sort)], p_uv);
    let outer = m.mk_forall([("u", int_sort)], inner);

    let goal = Goal::new(vec![outer]);
    let result = {
        let mut tactic = SkolemizationTactic::new(&mut m);
        tactic
            .apply_mut(&goal)
            .expect("test operation should succeed")
    };
    let TacticResult::SubGoals(goals) = result else {
        panic!("expected Skolemization to fire, got {result:?}");
    };
    let produced = goals[0].assertions[0];

    let sk = m.mk_apply("sk!0", [u], int_sort);
    let p_u_sk = m.mk_apply("P", [u, sk], bool_sort);
    let expected = m.mk_forall([("u", int_sort)], p_u_sk);
    assert_eq!(produced, expected);
    assert!(!mentions_free_var(&m, produced, "v"));
}

/// `remove_from_or_and_substitute`'s `Not(_) => false` arm is now guarded by
/// `is_target_diseq`. A bare non-target negation must be substituted into,
/// never collapsed to `false`.
#[test]
fn der_does_not_collapse_a_non_target_negation_to_false() {
    let mut m = setup_manager();
    let bool_sort = m.sorts.bool_sort;
    let int_sort = m.sorts.int_sort;

    // forall x. (x != 5 or not P(x))
    let x = m.mk_var("x", int_sort);
    let five = m.mk_int(5);
    let eq = m.mk_eq(x, five);
    let diseq = m.mk_not(eq);
    let p_x = m.mk_apply("P", [x], bool_sort);
    let not_p_x = m.mk_not(p_x);
    let body = m.mk_or([diseq, not_p_x]);
    let quant = m.mk_forall([("x", int_sort)], body);

    let goal = Goal::new(vec![quant]);
    let result = {
        let mut tactic = DerTactic::new(&mut m);
        tactic
            .apply_mut(&goal)
            .expect("test operation should succeed")
    };
    let TacticResult::SubGoals(goals) = result else {
        panic!("expected DER to fire, got {result:?}");
    };

    let p_five = m.mk_apply("P", [five], bool_sort);
    let expected = m.mk_not(p_five);
    let false_id = m.mk_false();
    assert_eq!(goals[0].assertions[0], expected);
    assert_ne!(goals[0].assertions[0], false_id);
}

/// `collect_positive_foralls` is memoized on `(TermId, polarity)`. A shared
/// subformula doubled 55 times would otherwise be re-expanded 2^55 times.
#[test]
fn quantifier_instantiation_handles_a_shared_dag_quickly() {
    let mut m = setup_manager();
    let bool_sort = m.sorts.bool_sort;
    let int_sort = m.sorts.int_sort;

    let x = m.mk_var("x", int_sort);
    let p_x = m.mk_apply("P", [x], bool_sort);
    let mut current = m.mk_forall([("x", int_sort)], p_x);

    // Each level references the previous one twice: a DAG of 55 nodes whose
    // tree unfolding has 2^55 leaves.
    for _ in 0..55 {
        current = m.mk_and([current, current]);
        // `mk_and` may dedup identical arguments; force real sharing with a
        // disjunction of the same node under two different parents.
        current = m.mk_or([current, current]);
    }

    let goal = Goal::new(vec![current]);
    let start = oxiz_time::Instant::now();
    let mut tactic = QuantifierInstantiationTactic::new(&mut m);
    let _ = tactic.apply_mut(&goal);
    assert!(
        start.elapsed() < std::time::Duration::from_secs(10),
        "shared-DAG traversal must not re-expand exponentially"
    );
}
