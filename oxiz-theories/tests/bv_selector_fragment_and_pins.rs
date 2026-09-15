//! Unit-level regressions for the two `ite`-selector findings fixed in 0.3.4
//! (`#P2b-24` and `#P2b-25` in `TODO.md`), driven against `BvSolver`
//! directly so a future regression is localised to this crate rather than
//! only showing up as a wrong verdict in `oxiz-solver`.
//!
//! * `#P2b-24`: `encode_bool_node` accepted only `not`/`and`/`or` over
//!   `=`/`bvult`/`bvule`/`bvslt`/`bvsle`, so a selector using `distinct`,
//!   `xor`, `=>`, a Bool `ite` or a Bool `=` had no circuit — and `bv_ite`
//!   returned `()` without saying so, leaving the caller to abstract the term
//!   into free bits.
//! * `#P2b-25`: a truth value pinned into a boolean node by
//!   `assert_bool_value` was resolved against by the embedded SAT solver but
//!   never named in the conflict explanation, so the CDCL(T) core learned a
//!   clause the theory never derived.

use oxiz_core::ast::{TermId, TermManager};
use oxiz_theories::bv::BvSolver;
use oxiz_theories::{Theory, TheoryCheckResult};

/// `x = (ite <selector> #x01 #x02)` over the 8-bit constants `a = 3`,
/// `b = 4`, both pinned; returns the model value of `x`.
fn value_under_selector(build: impl FnOnce(&mut TermManager, TermId, TermId) -> TermId) -> u64 {
    let mut tm = TermManager::new();
    let bv8 = tm.sorts.bitvec(8);
    let a = tm.mk_var("a", bv8);
    let b = tm.mk_var("b", bv8);
    let one = tm.mk_bitvec(1i64, 8);
    let two = tm.mk_bitvec(2i64, 8);
    let cond = build(&mut tm, a, b);
    let x = tm.mk_var("x", bv8);

    let mut solver = BvSolver::new();
    assert!(solver.assert_const(a, 3, 8));
    assert!(solver.assert_const(b, 4, 8));
    assert!(solver.assert_const(one, 1, 8));
    assert!(solver.assert_const(two, 2, 8));
    assert!(
        solver.bv_ite(x, cond, one, two, &tm),
        "the selector is inside the supported fragment"
    );
    assert!(matches!(
        solver.check().expect("check"),
        TheoryCheckResult::Sat
    ));
    solver.get_value(x).expect("x has a circuit")
}

#[test]
fn distinct_over_bv_operands_is_a_selector() {
    assert_eq!(
        value_under_selector(|tm, a, b| tm.mk_distinct([a, b])),
        1,
        "3 ≠ 4 selects the then-branch"
    );
    assert_eq!(
        value_under_selector(|tm, a, _b| tm.mk_distinct([a, a])),
        2,
        "(distinct a a) is folded or false either way: the else-branch"
    );
}

#[test]
fn xor_of_two_comparisons_is_a_selector() {
    // (3 <u 4) xor (3 = 4)  ≡  true xor false  ≡  true.
    assert_eq!(
        value_under_selector(|tm, a, b| {
            let lt = tm.mk_bv_ult(a, b);
            let eq = tm.mk_eq(a, b);
            tm.mk_xor(lt, eq)
        }),
        1
    );
}

#[test]
fn implication_is_a_selector() {
    // (3 <u 4) => (3 = 4)  ≡  false.
    assert_eq!(
        value_under_selector(|tm, a, b| {
            let lt = tm.mk_bv_ult(a, b);
            let eq = tm.mk_eq(a, b);
            tm.mk_implies(lt, eq)
        }),
        2
    );
    // (4 <u 3) => (3 = 4)  ≡  true (vacuously).
    assert_eq!(
        value_under_selector(|tm, a, b| {
            let gt = tm.mk_bv_ult(b, a);
            let eq = tm.mk_eq(a, b);
            tm.mk_implies(gt, eq)
        }),
        1
    );
}

#[test]
fn bool_ite_is_a_selector() {
    // (ite (3 <u 4) (3 <=u 4) (4 <u 3))  ≡  (3 <=u 4)  ≡  true.
    assert_eq!(
        value_under_selector(|tm, a, b| {
            let lt = tm.mk_bv_ult(a, b);
            let t = tm.mk_bv_ule(a, b);
            let e = tm.mk_bv_ult(b, a);
            tm.mk_ite(lt, t, e)
        }),
        1
    );
    // (ite (4 <u 3) (3 <=u 4) (4 <u 3))  ≡  (4 <u 3)  ≡  false.
    assert_eq!(
        value_under_selector(|tm, a, b| {
            let gt = tm.mk_bv_ult(b, a);
            let t = tm.mk_bv_ule(a, b);
            tm.mk_ite(gt, t, gt)
        }),
        2
    );
}

#[test]
fn bool_equality_and_bool_distinct_are_selectors() {
    // (3 <u 4) = (3 <=u 4)  ≡  true = true  ≡  true.
    assert_eq!(
        value_under_selector(|tm, a, b| {
            let lt = tm.mk_bv_ult(a, b);
            let le = tm.mk_bv_ule(a, b);
            tm.mk_eq(lt, le)
        }),
        1
    );
    // (distinct (3 <u 4) (3 <=u 4))  ≡  false.
    assert_eq!(
        value_under_selector(|tm, a, b| {
            let lt = tm.mk_bv_ult(a, b);
            let le = tm.mk_bv_ule(a, b);
            tm.mk_distinct([lt, le])
        }),
        2
    );
}

/// A selector outside the fragment (an arithmetic comparison) makes `bv_ite`
/// report failure and build nothing; before the fix it returned `()` and the
/// caller went on as if the `ite` had a circuit.
#[test]
fn bv_ite_reports_a_selector_it_cannot_encode() {
    let mut tm = TermManager::new();
    let bv8 = tm.sorts.bitvec(8);
    let int_sort = tm.sorts.int_sort;
    let i = tm.mk_var("i", int_sort);
    let j = tm.mk_var("j", int_sort);
    let cond = tm.mk_lt(i, j);
    let one = tm.mk_bitvec(1i64, 8);
    let two = tm.mk_bitvec(2i64, 8);
    let x = tm.mk_var("x", bv8);

    let mut solver = BvSolver::new();
    assert!(solver.assert_const(one, 1, 8));
    assert!(solver.assert_const(two, 2, 8));
    assert!(!solver.bv_ite(x, cond, one, two, &tm));
    assert!(
        solver.get_bv(x).is_none(),
        "a failed bv_ite must not leave a free bit-vector behind"
    );
}

/// The `#P2b-25` shape at its smallest: `x = (ite p 1 2)` with `x = 2`
/// asserted is refuted only once `p` is pinned true, and the explanation
/// must say so.
#[test]
fn a_pinned_selector_is_named_in_the_conflict_explanation() {
    let mut tm = TermManager::new();
    let bv8 = tm.sorts.bitvec(8);
    let bool_sort = tm.sorts.bool_sort;
    let p = tm.mk_var("p", bool_sort);
    let one = tm.mk_bitvec(1i64, 8);
    let two = tm.mk_bitvec(2i64, 8);
    let x = tm.mk_var("x", bv8);
    // Stands in for the outer `(= x (ite p 1 2))` atom the manager records.
    let atom = tm.mk_var("atom", bool_sort);

    let mut solver = BvSolver::new();
    assert!(solver.assert_const(one, 1, 8));
    assert!(solver.assert_const(two, 2, 8));
    assert!(solver.bv_ite(x, p, one, two, &tm));
    solver.record_constraint_term(atom);
    assert!(solver.assert_const(x, 2, 8));
    assert!(matches!(
        solver.check().expect("check"),
        TheoryCheckResult::Sat
    ));
    assert!(solver.pinned_terms().is_empty());

    // The pin arrives after the last check — the shape whose refutation went
    // unexamined before `assert_bool_value` reported "pinned a live node".
    assert!(solver.assert_bool_value(p, true));
    assert_eq!(solver.pinned_terms(), &[p]);
    match solver.check().expect("check") {
        TheoryCheckResult::Unsat(explanation) => {
            assert!(
                explanation.contains(&atom),
                "the recorded atom is blamed: {explanation:?}"
            );
            assert!(
                explanation.contains(&p),
                "0.3.3/0.3.4 omitted the pinned selector from the explanation, \
                 which made the CDCL(T) core learn a clause without `p` — a \
                 false proof: {explanation:?}"
            );
        }
        other => panic!("expected Unsat, got {other:?}"),
    }
}

/// A pin installed inside a scope is retracted by `pop`, together with the
/// unit clause it stood for, so a later conflict cannot blame it.
#[test]
fn a_pin_is_retracted_with_its_scope() {
    let mut tm = TermManager::new();
    let bv8 = tm.sorts.bitvec(8);
    let bool_sort = tm.sorts.bool_sort;
    let p = tm.mk_var("p", bool_sort);
    let one = tm.mk_bitvec(1i64, 8);
    let two = tm.mk_bitvec(2i64, 8);
    let x = tm.mk_var("x", bv8);

    let mut solver = BvSolver::new();
    assert!(solver.assert_const(one, 1, 8));
    assert!(solver.assert_const(two, 2, 8));
    assert!(solver.bv_ite(x, p, one, two, &tm));
    assert!(solver.assert_const(x, 2, 8));

    solver.push();
    assert!(solver.assert_bool_value(p, true));
    assert_eq!(solver.pinned_terms(), &[p]);
    assert!(matches!(
        solver.check().expect("check"),
        TheoryCheckResult::Unsat(_)
    ));
    solver.pop();

    assert!(solver.pinned_terms().is_empty());
    assert!(
        matches!(solver.check().expect("check"), TheoryCheckResult::Sat),
        "with the pin gone, x = 2 is satisfiable again (p = false)"
    );
    assert_eq!(solver.bool_value(p), Some(false));
}

/// A value remembered *before* the node exists is applied on creation and
/// counts as a pin from then on.
#[test]
fn a_value_recorded_before_the_node_exists_is_pinned_on_creation() {
    let mut tm = TermManager::new();
    let bv8 = tm.sorts.bitvec(8);
    let bool_sort = tm.sorts.bool_sort;
    let p = tm.mk_var("p", bool_sort);
    let one = tm.mk_bitvec(1i64, 8);
    let two = tm.mk_bitvec(2i64, 8);
    let x = tm.mk_var("x", bv8);

    let mut solver = BvSolver::new();
    assert!(
        !solver.assert_bool_value(p, false),
        "no node yet: the value is only remembered"
    );
    assert!(solver.pinned_terms().is_empty());
    assert!(solver.assert_const(one, 1, 8));
    assert!(solver.assert_const(two, 2, 8));
    assert!(solver.bv_ite(x, p, one, two, &tm));
    assert_eq!(solver.pinned_terms(), &[p]);
    assert!(matches!(
        solver.check().expect("check"),
        TheoryCheckResult::Sat
    ));
    assert_eq!(solver.get_value(x), Some(2));
}

// ---------------------------------------------------------------------------
// `#P2b-27` / `#P2b-29` — what the circuit exposes to the theory manager
// ---------------------------------------------------------------------------

/// The value the circuit chose for a selector the enclosing search never
/// assigned is readable through `bool_node_terms` + `bool_value` — the pair
/// `oxiz-solver`'s model builder publishes (`#P2b-27`): `x = (ite p 1 2)`
/// with `x = 1` pinned leaves `p = true` as the only model.
#[test]
fn a_selector_the_outer_search_never_assigned_is_readable_from_the_circuit() {
    let mut tm = TermManager::new();
    let bv8 = tm.sorts.bitvec(8);
    let bool_sort = tm.sorts.bool_sort;
    let p = tm.mk_var("p", bool_sort);
    let one = tm.mk_bitvec(1i64, 8);
    let two = tm.mk_bitvec(2i64, 8);
    let x = tm.mk_var("x", bv8);

    let mut solver = BvSolver::new();
    assert!(solver.assert_const(one, 1, 8));
    assert!(solver.assert_const(two, 2, 8));
    assert!(solver.bv_ite(x, p, one, two, &tm));
    assert!(solver.assert_const(x, 1, 8));
    assert!(matches!(
        solver.check().expect("check"),
        TheoryCheckResult::Sat
    ));
    let nodes: Vec<TermId> = solver.bool_node_terms().collect();
    assert!(
        nodes.contains(&p),
        "the selector has a boolean node: {nodes:?}"
    );
    assert_eq!(solver.bool_value(p), Some(true), "only p = true selects 1");
    assert_eq!(solver.get_value(x), Some(1));
}

/// An opaque leaf is journalled with its circuit and retracted by the same
/// `pop` (`#P2b-29`): the theory manager reads this list to intern every
/// leaf into congruence closure, so a stale entry would name a term whose
/// circuit is gone.
#[test]
fn opaque_leaves_are_journalled_and_retracted_with_their_scope() {
    let mut tm = TermManager::new();
    let bv8 = tm.sorts.bitvec(8);
    let fa = tm.mk_var("fa", bv8);
    let fb = tm.mk_var("fb", bv8);
    let fc = tm.mk_var("fc", bv8);

    let mut solver = BvSolver::new();
    assert!(solver.opaque_leaves().is_empty());
    solver.new_opaque_leaf(fa, 8);
    assert_eq!(solver.opaque_leaves(), &[fa]);
    solver.push();
    solver.new_opaque_leaf(fb, 8);
    solver.new_opaque_leaf(fb, 8);
    assert_eq!(
        solver.opaque_leaves(),
        &[fa, fb],
        "a re-creation is not journalled twice"
    );
    assert!(solver.get_bv(fb).is_some());
    solver.pop();
    assert_eq!(solver.opaque_leaves(), &[fa]);
    assert!(
        solver.get_bv(fb).is_none(),
        "the leaf's circuit goes with the same pop as its record"
    );
    solver.new_opaque_leaf(fc, 8);
    assert_eq!(solver.opaque_leaves(), &[fa, fc]);
    solver.reset();
    assert!(solver.opaque_leaves().is_empty());
    assert!(solver.get_bv(fa).is_none());
}
