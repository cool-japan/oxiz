//! Regression tests for the arithmetic-disequality false-`sat` family.
//!
//! Fifteen `:status unsat` SMT-LIB benchmarks (QF_LIA rings, QF_IDL job-shop,
//! QF_UFLIA Wisa, QF_AUFLIA storecomm/Rodin) answered `sat` in under a second.
//! They shared two root causes, both of which are fixed by the code these
//! tests pin down:
//!
//! **A1 -- an arithmetic disequality never reached the arithmetic solver.**
//! When CDCL assigns an Int/Real `Eq` atom *false*, `TheoryManager::
//! process_constraint`'s negative branch speaks only to EUF and BV;
//! `ArithSolver` has no `assert_neq` (a disequality is not convex, so the
//! tableau cannot hold one). The atom therefore constrained nothing at all and
//! the LP was free to give both sides the same value. The only rescue was a
//! *syntactic* pre-pass, `Solver::add_arith_diseq_split`, which walked the
//! assertion looking for `Not(Eq(..))`/`Distinct(..)` and enumerated only a
//! handful of connectives -- so an `Eq` reachable only through a shape it did
//! not enumerate stayed a free Boolean. The fix moves the trichotomy
//! `(a = b) OR (a < b) OR (a > b)` onto the **atom**, emitted from the
//! encoder's `TermKind::Eq` arm, which every numeric equality must pass
//! through to get a SAT variable at all.
//!
//! **A2 -- `let` was a blindfold.** The SMT-LIB parser wrapped each assertion
//! in a `Let` whose bindings it had *already substituted* into the body, so
//! the node bound nothing -- but every assert-time pre-pass treats `Let` as an
//! opaque binder and refuses to descend. One vacuous wrapper silently disabled
//! `eliminate_nonbool_ite`, `abstract_compound_bool_args`,
//! `flatten_lookup_spines`, `purify_numeric_uf_args`, the quantified model
//! gate and `add_arith_diseq_split` for the whole assertion beneath it. Two
//! formulas differing only by an *unused* `(let ((q 0)) ...)` got different
//! verdicts, one of them a wrong `sat`. The parser now returns the substituted
//! body directly and emits no `Let`.
//!
//! Every formula below is written **in-house**: minimal shapes reproduced from
//! the diagnosis, never a copy of an upstream benchmark file.
//!
//! Each unsatisfiable case is paired with a satisfiable control that differs
//! by one constant. Those controls are the real safety net: the trichotomy
//! clause is a tautology and the gate answers `Unknown`, so a mistake in
//! either direction shows up as a *wrong `unsat`* or a lost `sat` here rather
//! than as a silently weaker solver.

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
        .expect("script should parse and run")
}

// ---------------------------------------------------------------------
// A1: a negated arithmetic equality buried in a shape the syntactic
//     pre-pass never enumerated.
// ---------------------------------------------------------------------

/// The seven-line A1 minimal repro: no `let`, no UF, no arrays.
///
/// `a` is pinned to `0` by `a >= 0 /\ a <= 0`, and `b >= 1`. The last
/// assertion `(not (and (< 0 b) (= a 0)))` must then be false, since both
/// conjuncts hold. The `Eq` sits under `Not(And(..))`, and with `(< 0 b)`
/// already true the clause forces `(= a 0)` false -- but nothing told
/// arithmetic that, so the solver reported `sat` with the very model
/// (`a = 0`) that falsifies the assertion.
#[test]
fn negated_equality_under_not_and_is_refuted() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun a () Int)
        (declare-fun b () Int)
        (assert (>= a 0))
        (assert (<= a 0))
        (assert (>= b 1))
        (assert (not (and (< 0 b) (= a 0))))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "a is pinned to 0 and b >= 1, so both conjuncts hold and the negation is false"
    );
}

/// Satisfiable control for the shape above: drop the lower bound on `b`, so
/// the negation can be satisfied through `(< 0 b)` being false.
#[test]
fn negated_equality_under_not_and_stays_sat_when_satisfiable() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun a () Int)
        (declare-fun b () Int)
        (assert (>= a 0))
        (assert (<= a 0))
        (assert (not (and (< 0 b) (= a 0))))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["sat"],
        "b <= 0 satisfies the negation without disturbing a = 0"
    );
}

/// A Bool-Bool `Eq` wrapping an arithmetic `Eq`: `(= p (= (+ x 1) 6))` with
/// `p` false. The inner numeric equality is forced false through the iff, a
/// path the syntactic walk could not see at all -- it looked for `Not(Eq)` and
/// this is `Eq(Bool, Eq)`.
#[test]
fn arithmetic_equality_forced_false_through_a_boolean_iff() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun p () Bool)
        (assert (= x 5))
        (assert (= p (= (+ x 1) 6)))
        (assert (not p))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "x = 5 makes (+ x 1) = 6 true, so p cannot be false"
    );
}

/// Satisfiable control: `x = 4` makes the inner equality genuinely false, so
/// `p` false is consistent.
#[test]
fn boolean_iff_over_arithmetic_equality_stays_sat_when_satisfiable() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun p () Bool)
        (assert (= x 4))
        (assert (= p (= (+ x 1) 6)))
        (assert (not p))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["sat"],
        "x = 4 makes (+ x 1) = 5, so the inner equality really is false"
    );
}

/// An `Eq` reachable only through an `Implies` **antecedent**. The pre-pass
/// descended into the consequent only, on the theory that instantiation
/// lemmas carry the disequality there.
#[test]
fn arithmetic_equality_in_an_implies_antecedent_is_refuted() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun q () Bool)
        (assert (= x 5))
        (assert (=> (= (+ x 1) 6) q))
        (assert (not q))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "the antecedent is true under x = 5, so q is forced and cannot be false"
    );
}

/// An `Eq` under `Xor`, another shape the walk never enumerated.
#[test]
fn arithmetic_equality_under_xor_is_refuted() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun r () Bool)
        (assert (= x 5))
        (assert r)
        (assert (xor (= (+ x 1) 6) r))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "both xor operands are true under x = 5 and r, so the xor is false"
    );
}

// ---------------------------------------------------------------------
// The trichotomy itself: a disequality must reach the tableau as a strict
// ordering constraint.
// ---------------------------------------------------------------------

/// The mechanism in isolation. `x` is pinned to `5`, and `(+ x 1)` is
/// asserted *different* from `6`. Refuting this requires the disequality to
/// become a strict ordering constraint the simplex tableau can hold: with
/// `(= (+ x 1) 6)` forced false, the trichotomy leaves
/// `(< (+ x 1) 6) OR (> (+ x 1) 6)`, and both branches conflict with `x = 5`.
#[test]
fn trichotomy_turns_a_disequality_into_a_strict_ordering_constraint() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (assert (= x 5))
        (assert (not (= (+ x 1) 6)))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "x = 5 forces (+ x 1) = 6, contradicting the asserted disequality"
    );
}

/// Satisfiable control: the same disequality against a value arithmetic does
/// *not* entail. Both strict branches must stay open, so the trichotomy must
/// not over-constrain.
#[test]
fn trichotomy_leaves_a_genuine_disequality_satisfiable() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (assert (>= x 0))
        (assert (not (= (+ x 1) 6)))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["sat"],
        "any x other than 5 satisfies this; the split must not force a value"
    );
}

/// The unguarded split `(a < b) OR (a > b)` would be *unsound* here: it would
/// force `x != 0` even though the formula is satisfied through `p`. Only the
/// three-way trichotomy, which keeps `(= x 0)` as a disjunct, is safe in a
/// context that does not force the equality false.
#[test]
fn disequality_split_does_not_force_a_disequality_in_a_disjunctive_context() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun p () Bool)
        (assert p)
        (assert (= x 0))
        (assert (or p (not (= x 0))))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["sat"],
        "the disjunction is satisfied through p; x = 0 must remain allowed"
    );
}

// ---------------------------------------------------------------------
// A2: an unused `let` wrapper must not change the verdict.
//
// Each pair below is the SAME formula with and without a binding whose
// variable never occurs in the body. Both must agree.
// ---------------------------------------------------------------------

/// A2 in its most direct form: the disequality itself sits under a `let`.
#[test]
fn disequality_under_a_let_is_refuted() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (assert (= x 5))
        (assert (let ((t (+ x 1))) (not (= t 6))))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "the let binds t to (+ x 1), which x = 5 forces to 6"
    );
}

/// The A7 pair: a non-Bool `ite` in a UF argument position. Without the
/// wrapper the solver was already correct; with an *unused* `(let ((q 0)) ..)`
/// around the identical formula it answered `sat`, because the wrapper
/// disabled `eliminate_nonbool_ite`.
#[test]
fn ite_uf_argument_is_refuted_without_a_let_wrapper() {
    let output = run(r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun c () Bool)
        (assert (= (f 1) 10))
        (assert (= (f 2) 20))
        (assert (= (f (ite c 1 2)) 30))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "either branch of the ite makes f's value 10 or 20, never 30"
    );
}

/// The A7 partner: byte-identical logic, wrapped in a binding that is never
/// used. It must reach the same verdict as the unwrapped form above.
#[test]
fn ite_uf_argument_is_refuted_through_an_unused_let_wrapper() {
    let output = run(r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun c () Bool)
        (assert (let ((q 0))
          (and (= (f 1) 10) (and (= (f 2) 20) (= (f (ite c 1 2)) 30)))))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "an unused let binding must not change the verdict of the A7 shape"
    );
}

/// The A8 pair: a compound numeric UF argument, `(f (- a b))`, whose value
/// arithmetic entails. Correct without a wrapper...
#[test]
fn compound_uf_argument_is_refuted_without_a_let_wrapper() {
    let output = run(r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun a () Int)
        (declare-fun b () Int)
        (assert (and (= a 6) (and (= b 2) (and (= (f 4) 1) (= (f (- a b)) 2)))))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "a - b is entailed to be 4, so f(a-b) and f(4) are the same application"
    );
}

/// ...and it must stay correct with one.
#[test]
fn compound_uf_argument_is_refuted_through_an_unused_let_wrapper() {
    let output = run(r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun a () Int)
        (declare-fun b () Int)
        (assert (let ((q 0))
          (and (= a 6) (and (= b 2) (and (= (f 4) 1) (= (f (- a b)) 2))))))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "an unused let binding must not change the verdict of the A8 shape"
    );
}

/// The satisfiable direction of the same pair: `a - b` is entailed to be `3`,
/// not `4`, so an uninterpreted `f` may map them differently. Both the
/// wrapped and unwrapped forms must stay `sat` -- this is the check that the
/// `let` fix did not buy soundness with a wrong `unsat`.
#[test]
fn compound_uf_argument_pair_stays_sat_in_both_forms_when_satisfiable() {
    let bare = run(r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun a () Int)
        (declare-fun b () Int)
        (assert (and (= a 5) (and (= b 2) (and (= (f 4) 1) (= (f (- a b)) 2)))))
        (check-sat)
    "#);
    let wrapped = run(r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun a () Int)
        (declare-fun b () Int)
        (assert (let ((q 0))
          (and (= a 5) (and (= b 2) (and (= (f 4) 1) (= (f (- a b)) 2))))))
        (check-sat)
    "#);
    assert_eq!(bare, vec!["sat"], "a - b = 3 != 4, so f may differ");
    assert_eq!(
        wrapped, bare,
        "the let-wrapped form must agree with the bare form in the sat direction too"
    );
}

/// A `let` whose binding *is* used, in the satisfiable direction: the
/// substituted body must still be solved correctly, and `get-value` must
/// resolve through it.
#[test]
fn a_used_let_binding_still_solves_and_yields_a_model() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (assert (let ((t (+ x 1))) (= t 6)))
        (check-sat)
        (get-value (x))
    "#);
    assert_eq!(output[0], "sat", "x = 5 satisfies this");
    assert!(
        output[1].contains('5'),
        "the model must resolve x to 5 through the let body, got: {}",
        output[1]
    );
}

// ---------------------------------------------------------------------
// The model-verification gate.
// ---------------------------------------------------------------------

/// The gate's trail-polarity half, exercised end-to-end.
///
/// `combine_eq` answers `Undetermined` -- never `Bool(true)` -- when two
/// numeric operands are *equal*, deliberately: a collision in the LP model is
/// not by itself evidence of a violation, since the tableau enforces
/// disequalities by case splitting rather than by pinning distinct witnesses.
/// So a violated disequality was invisible to the gate. The missing
/// information is the trail polarity, which
/// `model_violates_negated_equality` now consults: if the core assigned
/// `(= a b)` false and the model makes both sides equal, that is a definite
/// refutation.
///
/// Two chained disequalities over a three-element range force `x`, `y` and
/// `z` to be pairwise distinct in `{0, 1}` -- impossible by pigeonhole. The
/// answer must be `unsat`; what must never happen is `sat` with a model that
/// collides two of them.
#[test]
fn pigeonhole_over_disequalities_is_never_answered_sat() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (declare-fun z () Int)
        (assert (and (>= x 0) (<= x 1)))
        (assert (and (>= y 0) (<= y 1)))
        (assert (and (>= z 0) (<= z 1)))
        (assert (not (= x y)))
        (assert (not (= y z)))
        (assert (not (= x z)))
        (check-sat)
    "#);
    assert_ne!(
        output,
        vec!["sat"],
        "three pairwise-distinct values cannot fit in {{0, 1}}; sat here would be a false model"
    );
    assert_eq!(output, vec!["unsat"], "the pigeonhole must be refuted");
}

/// The same shape with room for all three values must stay `sat`: the gate
/// must not downgrade a genuine model in which the disequalities hold.
#[test]
fn pairwise_disequalities_stay_sat_when_the_range_is_wide_enough() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (declare-fun z () Int)
        (assert (and (>= x 0) (<= x 2)))
        (assert (and (>= y 0) (<= y 2)))
        (assert (and (>= z 0) (<= z 2)))
        (assert (not (= x y)))
        (assert (not (= y z)))
        (assert (not (= x z)))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["sat"],
        "0, 1, 2 is a genuine model; the gate must not refuse it"
    );
}

/// `distinct` must behave like the chained disequalities above: the pairwise
/// expansion goes through the same numeric `Eq` atoms.
#[test]
fn distinct_over_a_too_small_range_is_refuted() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (declare-fun z () Int)
        (assert (and (>= x 0) (<= x 1)))
        (assert (and (>= y 0) (<= y 1)))
        (assert (and (>= z 0) (<= z 1)))
        (assert (distinct x y z))
        (check-sat)
    "#);
    assert_eq!(
        output,
        vec!["unsat"],
        "distinct over three variables needs three values, and only two exist"
    );
}

/// Real (LRA) arithmetic takes the same path as LIA through
/// `Constraint::Eq`, and a strict ordering over the reals is satisfiable
/// where the integer version is not -- so this pair pins that the trichotomy
/// is emitted for `Real` operands too, without over-constraining them.
#[test]
fn real_sorted_disequality_is_handled_without_over_constraining() {
    let unsat = run(r#"
        (set-logic QF_LRA)
        (declare-fun u () Real)
        (assert (= u 1.5))
        (assert (not (= (+ u 0.5) 2.0)))
        (check-sat)
    "#);
    assert_eq!(
        unsat,
        vec!["unsat"],
        "u = 1.5 forces (+ u 0.5) = 2.0, contradicting the disequality"
    );

    let sat = run(r#"
        (set-logic QF_LRA)
        (declare-fun u () Real)
        (declare-fun v () Real)
        (assert (< u v))
        (assert (not (= u v)))
        (check-sat)
    "#);
    assert_eq!(
        sat,
        vec!["sat"],
        "u < v already implies u != v; this must stay satisfiable"
    );
}

// ---------------------------------------------------------------------
// Lazy theory mode must agree with eager mode.
//
// `:theory-mode lazy` batches theory assignments and processes them at
// `final_check` instead of on every propagation. Two defects in that path
// were exposed by making the arithmetic case split explicit (the trichotomy
// gives the SAT core real branches, so lazy mode started backtracking where
// it previously almost never did):
//
// 1. `TheoryManager::on_backtrack` cleared the whole `pending_assignments`
//    queue instead of pruning it by level, discarding assignments the SAT
//    core had *not* undone and would never re-send. The facts simply never
//    reached the theory solvers.
// 2. The `final_check` soundness backstop that rebuilds theory state from the
//    shadow trail was gated to eager mode, because lazy mode kept no shadow
//    trail at all. It does now, and the gate is gone.
//
// Every case below is checked in BOTH modes: the verdicts must agree, which
// is the property that actually matters. A mode that answers `unknown` where
// the other proves `unsat` is a completeness bug; one that answers `sat`
// there is a soundness bug — defect 1 produced exactly that.
// ---------------------------------------------------------------------

/// Run the same formula under both theory modes and require the same verdict.
fn run_both_modes(body: &str) -> Vec<String> {
    let eager = run(&format!("(set-option :theory-mode eager)\n{body}"));
    let lazy = run(&format!("(set-option :theory-mode lazy)\n{body}"));
    assert_eq!(
        eager, lazy,
        "eager and lazy theory modes must agree on this formula"
    );
    eager
}

/// A two-step arithmetic chain entailing a disequality's negation.
///
/// `x = 2` and `y = x + 1` force `y = 3`, contradicting `y != 3`. Deriving it
/// needs *both* equalities, and defect 1 dropped whichever of them landed
/// below the SAT core's backtrack point — after which lazy mode answered a
/// wrong `sat`.
#[test]
fn lazy_mode_refutes_an_arithmetic_chain_like_eager_mode() {
    let output = run_both_modes(
        r#"
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (= x 2))
        (assert (= y (+ x 1)))
        (assert (not (= y 3)))
        (check-sat)
    "#,
    );
    assert_eq!(output, vec!["unsat"], "x = 2 and y = x + 1 force y = 3");
}

/// The same chain, with the contradiction reached through EUF *congruence*
/// rather than arithmetic alone: `y = 3` must cross the theory boundary so
/// that `f(y)` and `f(3)` become congruent.
///
/// This is the script named in `TheoryManager::final_check`'s own comment as
/// the reason the rebuild backstop existed; it now passes in lazy mode too.
#[test]
fn lazy_mode_reaches_euf_congruence_through_an_arithmetic_chain() {
    let output = run_both_modes(
        r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (= x 2))
        (assert (= y (+ x 1)))
        (assert (not (= (f y) (f 3))))
        (check-sat)
    "#,
    );
    assert_eq!(
        output,
        vec!["unsat"],
        "arithmetic entails y = 3, so congruence forces f(y) = f(3)"
    );
}

/// Satisfiable control: the chain entails `y = 4`, so `f(y) != f(3)` is
/// perfectly consistent. Neither the level-pruned queue nor the newly
/// un-gated rebuild may be achieved by making lazy mode over-constrain.
#[test]
fn lazy_mode_stays_sat_when_the_chain_entails_no_contradiction() {
    let output = run_both_modes(
        r#"
        (set-logic QF_UFLIA)
        (declare-fun f (Int) Int)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (= x 2))
        (assert (= y (+ x 2)))
        (assert (not (= (f y) (f 3))))
        (check-sat)
    "#,
    );
    assert_eq!(output, vec!["sat"], "y = 4, so f(y) != f(3) is consistent");
}

/// The A1 shape itself under both modes: the disequality is reached only
/// through `Not(And(..))`, so it exercises the trichotomy and the lazy queue
/// together.
#[test]
fn lazy_mode_refutes_the_a1_shape_like_eager_mode() {
    let output = run_both_modes(
        r#"
        (set-logic QF_LIA)
        (declare-fun a () Int)
        (declare-fun b () Int)
        (assert (>= a 0))
        (assert (<= a 0))
        (assert (>= b 1))
        (assert (not (and (< 0 b) (= a 0))))
        (check-sat)
    "#,
    );
    assert_eq!(
        output,
        vec!["unsat"],
        "both conjuncts hold, so the negation fails"
    );
}
