//! Regression tests for finding U-Z10: wrong `sat` on QF_BV formulas with
//! Boolean structure over bit-vector atoms.
//!
//! # What was broken
//!
//! `BvSolver::pop()` (`oxiz-theories/src/bv/solver.rs`) rolled the embedded SAT
//! solver's clause database back to the matching `push()` but left the three
//! term→circuit memo caches — `term_to_bv`, `ult_cache` and `bool_node` —
//! populated.  Every circuit node first created *above* a level that was later
//! popped therefore survived in the cache as a bundle of SAT variables whose
//! defining and constant-pinning clauses had just been deleted: a completely
//! unconstrained bit-vector.  The idempotence guard in
//! `oxiz-solver/src/solver/theory_bv_encode.rs` (`if bv.get_bv(tid).is_some()
//! { continue; }`) then refused to re-encode it, so the next `BvSolver::check()`
//! found a "model" for a circuit that no longer said anything and returned
//! `Sat`.  Nothing downstream recovered the lost conflict: `final_check` never
//! calls the BV solver, its resync backstop is gated off for BV problems, and
//! the model gate cannot evaluate bit-vector terms.
//!
//! The fix gives each of the three caches an undo journal, truncated in `pop()`
//! **before** `sat.pop()` and cleared in `reset()`, restoring the invariant the
//! solver had silently assumed: *a term has a cache entry iff the clauses
//! defining it are live in the embedded SAT solver*.
//!
//! A second, independent defect lived in `theory_manager.rs`: both inner
//! `match constraint` blocks of the BV comparison arm handled only
//! `Constraint::Lt` and `Le`, ending in a silent `_ => {}`.  A `Constraint::Gt`
//! or `Ge` over bit-vector operands therefore asserted **nothing** and survived
//! as a free Boolean.  Both arms are now present, and an atom that reaches the
//! circuit without anything being asserted is recorded so the owning solver
//! answers `unknown` rather than trusting the circuit's `Sat`.
//!
//! That second defect has two repairs, and section 3 and section 4 below pin
//! one each: the *primary* one is U-Z14's parser sort check, which rejects an
//! ill-sorted `(> bv bv)` script outright, and the theory-manager arms are the
//! second half, for a `Gt`/`Ge` term an embedder builds through the
//! `TermManager` API, where no parser ever sees it.  Section 3 therefore takes
//! the API route and section 4 pins the parse error.
//!
//! # Measured before / after
//!
//! Every `unsat` expectation below was answered **`sat`** by OxiZ 0.3.3 and by
//! the unmodified 0.3.4 tree (commit `6bdf958`); the two "control" tests and
//! the three satisfiable scripts answered correctly both before and after, and
//! are here to pin that the fix does not overshoot.  Across 12 000 random
//! width-8 differential trials the wrong-`sat` rate went from 34/703 unsat
//! instances (4.8 %) to 0/703, with 0 wrong `unsat` in both builds.

use oxiz_core::ast::{TermId, TermManager};
use oxiz_solver::{Context, Solver, SolverResult};

/// Run a single SMT-LIB2 script and return the solver result.
///
/// The verdict is the last `sat` / `unsat` / `unknown` token in the output.
fn run_script(script: &str) -> SolverResult {
    let mut ctx = Context::new();
    let outputs = ctx.execute_script(script).unwrap_or_default();
    for tok in outputs.iter().rev() {
        match tok.trim() {
            "sat" => return SolverResult::Sat,
            "unsat" => return SolverResult::Unsat,
            "unknown" => return SolverResult::Unknown,
            _ => {}
        }
    }
    SolverResult::Unknown
}

/// Every output line of a script, joined, so `(get-value ...)` can be inspected.
fn run_script_output(script: &str) -> String {
    let mut ctx = Context::new();
    ctx.execute_script(script).unwrap_or_default().join("\n")
}

// ─────────────────────────────────────────────────────────────────────────
// 1. The cargo-formal conformance fixtures
// ─────────────────────────────────────────────────────────────────────────

/// `crates/formal-conformance/fixtures/u01_boolean_structure_and_ule.smt2`.
///
/// `a = 0x0f` and both `bvule` conjuncts hold there, so the negated
/// conjunction is unsatisfiable.  0.3.3/0.3.4: **`sat`**, with the model
/// `a = #b00001111` — a counterexample that falsifies nothing.
#[test]
fn u01_negated_conjunction_of_ule_is_unsat() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (not (and (bvule a #x0f) (bvule a #x10))))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// `u08_boolean_structure_ult_ugt.smt2`: the same shape over `bvult`/`bvugt`,
/// showing the bug is not specific to one comparison operator.  `0x0f <u 0x10`
/// and `0x0f >u 0x00` both hold.  0.3.3/0.3.4: **`sat`** (in 0.476 ms).
#[test]
fn u08_negated_conjunction_of_ult_ugt_is_unsat() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (not (and (bvult a #x10) (bvugt a #x00))))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// `u09_boolean_structure_ugt_ult.smt2`: u08 with the conjunction's operands
/// swapped, which is why the failure looked order-sensitive — whether a
/// circuit node happens to be created above a level that is later popped
/// depends on the search order.  0.3.3/0.3.4: **`sat`** (in 0.125 ms).
#[test]
fn u09_negated_conjunction_operands_swapped_is_unsat() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (not (and (bvugt a #x00) (bvult a #x10))))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// The De Morgan form of u01, `(or (not P) (not Q))`.
///
/// This one corrects the capability memo, which recorded the disjunctive form
/// as *passing* — on these `bvule` atoms it fails identically.  0.3.3/0.3.4:
/// **`sat`**.
#[test]
fn demorgan_disjunction_of_negations_is_unsat() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (or (not (bvule a #x0f)) (not (bvule a #x10))))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// The signed variant (`p2b/w0/probe4.smt2`): `bvsle` instead of `bvule`, so a
/// different `assert_*` and a different circuit shape reach the same stale
/// cache.  `0x0f <=s 0x0f` and `0x0f <=s 0x10` both hold.  0.3.3/0.3.4:
/// **`sat`**.
#[test]
fn signed_negated_conjunction_of_sle_is_unsat() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (not (and (bvsle a #x0f) (bvsle a #x10))))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// The two-variable variant (`p2b/w0/probe5.smt2`): the stale entry belongs to
/// a declared constant rather than a literal, so the loss is not an artefact of
/// `BitVecConst` pinning alone.  `0x0f <=u 0x10` and `0x0f <=u 0xf0` both hold.
/// 0.3.3/0.3.4: **`sat`**.
#[test]
fn two_variable_negated_conjunction_is_unsat() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (= a #x0f))
(assert (= b #x10))
(assert (not (and (bvule a b) (bvule a #xf0))))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

// ─────────────────────────────────────────────────────────────────────────
// 2. Controls: single-atom shapes that never push, correct before and after
// ─────────────────────────────────────────────────────────────────────────

/// One negated atom, no Boolean structure, so no decision level is opened and
/// no cache entry outlives a pop.  `unsat` before the fix and after.
#[test]
fn control_single_negated_ule_is_unsat() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (not (bvule a #x0f)))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

// ─────────────────────────────────────────────────────────────────────────
// 3. `Constraint::Lt` / `Gt` / `Ge` over bit-vector operands, built through
//    the `TermManager` API (independent defect)
// ─────────────────────────────────────────────────────────────────────────
//
// # Why these are API-level and not scripts
//
// They were scripts (`(assert (> a #x0f))`) when they were written, because
// that is how finding U-Z10's sibling family was first observed.  U-Z14 —
// landed in the same wave — gives `<`, `<=`, `>`, `>=` a sort check in
// `oxiz-core/src/smtlib/parser/build.rs`, so an ill-sorted `(> bv bv)` is now
// an `Err` from `Context::execute_script` and never reaches the solver at all.
// That parser rejection is the **primary** repair (report `I0-a` §4.1(b) calls
// it "the better repair"), and section 5 below pins it.
//
// The `Constraint::Gt`/`Ge` arms in `theory_manager.rs` are the second half:
// `TermManager::mk_gt`/`mk_ge` intern a `TermKind::Gt`/`Ge` with a Bool sort
// and **no sort check** on the operands (`oxiz-core/src/ast/manager/builder.rs`),
// and `encode.rs` records `Constraint::Gt`/`Ge` for them unconditionally, so an
// embedder building terms through the API still reaches those arms.  These
// tests take that route: `Solver` + `TermManager` directly, no parser.  Both
// halves are needed and neither subsumes the other — the parser cannot police
// terms it never sees, and the theory manager cannot recover a signedness the
// operator never carried.
//
// (Case recorded, as the Gate 1 ruling asks: the API route **does** reach the
// arms — `mk_gt`/`mk_ge` do not refuse bit-vector operands — so every one of
// these is a genuine API-level test, not a fallback parse-error test.)
//
// # These tests were checked against the arms being absent
//
// A test that passes with the fix removed proves nothing.  With the four
// `Constraint::Gt`/`Ge` arms of `theory_manager.rs` temporarily replaced by
// `=> false`, **all six** `Gt`/`Ge` rows below fail (`Unknown` instead of
// `Unsat`/`Sat`, because `bv_run_check` then records the atom as unmodelled
// and `resource_exhausted()` downgrades the verdict).  The `<` control passes
// either way, which is exactly its job: `Constraint::Lt` always had its arm.
//
// # Measured before / after
//
// Every `unsat` expectation below was answered **`sat`** by OxiZ 0.3.3 and by
// the unmodified 0.3.4 tree (commit `6bdf958`), through the script route these
// tests used to take; the `<` control and the two satisfiable twins answered
// `sat`/`unsat` correctly both before and after.

/// An 8-bit constant `a`, a `Solver` and the `TermManager` that owns them.
///
/// Returns `(solver, manager, a)`, with `a = 0x0f` already asserted — the
/// premise every case below contradicts or satisfies.
fn pinned_a_solver() -> (Solver, TermManager, TermId) {
    let mut manager = TermManager::new();
    let mut solver = Solver::new();
    let bv8 = manager.sorts.bitvec(8);
    let a = manager.mk_var("a", bv8);
    let fifteen = manager.mk_bitvec(0x0f_i64, 8);
    let premise = manager.mk_eq(a, fifteen);
    solver.assert(premise, &mut manager);
    (solver, manager, a)
}

/// The `<` control.  `Constraint::Lt` always had its arm, which is exactly
/// what isolated the missing `Gt`/`Ge` arms below: `unsat` before the fix and
/// after, on the script route and on this one.
#[test]
fn control_lt_over_bv_operands_is_unsat() {
    let (mut solver, mut manager, a) = pinned_a_solver();
    let fifteen = manager.mk_bitvec(0x0f_i64, 8);
    let atom = manager.mk_lt(a, fifteen);
    solver.assert(atom, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
}

/// `p2b/w0/probe3.smt2` through the API: `a = 0x0f ∧ a > 0x0f`.  Both inner
/// `match constraint` blocks of the BV comparison arm handled only `Lt`/`Le`,
/// so nothing was asserted and the atom stayed a free Boolean.  0.3.3/0.3.4:
/// **`sat`** — and still `sat` on a tree carrying only the scope-rollback fix,
/// which is what proves the two defects are independent.
#[test]
fn positive_gt_over_bv_operands_is_unsat() {
    let (mut solver, mut manager, a) = pinned_a_solver();
    let fifteen = manager.mk_bitvec(0x0f_i64, 8);
    let atom = manager.mk_gt(a, fifteen);
    solver.assert(atom, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
}

/// The `>=` twin: `a = 0x0f ∧ a >= 0x10`.  0.3.3/0.3.4: **`sat`**.
#[test]
fn positive_ge_over_bv_operands_is_unsat() {
    let (mut solver, mut manager, a) = pinned_a_solver();
    let sixteen = manager.mk_bitvec(0x10_i64, 8);
    let atom = manager.mk_ge(a, sixteen);
    solver.assert(atom, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
}

/// The negative `Gt` arm, which the positive tests cannot reach:
/// `¬(a >u b) ≡ a <=u b`, so `a = 0x0f ∧ ¬(a > 0x00)` is unsatisfiable.
/// Getting the direction backwards here would be a silent wrong `unsat`, so it
/// is tested in both directions (see the satisfiable twin below).
/// 0.3.3/0.3.4: **`sat`**.
#[test]
fn negative_gt_over_bv_operands_is_unsat() {
    let (mut solver, mut manager, a) = pinned_a_solver();
    let zero = manager.mk_bitvec(0x00_i64, 8);
    let atom = manager.mk_gt(a, zero);
    let negated = manager.mk_not(atom);
    solver.assert(negated, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
}

/// The negative `Ge` arm: `¬(a >=u b) ≡ a <u b`, so `a = 0x0f ∧ ¬(a >= 0x0f)`
/// is unsatisfiable.  0.3.3/0.3.4: **`sat`**.
#[test]
fn negative_ge_over_bv_operands_is_unsat() {
    let (mut solver, mut manager, a) = pinned_a_solver();
    let fifteen = manager.mk_bitvec(0x0f_i64, 8);
    let atom = manager.mk_ge(a, fifteen);
    let negated = manager.mk_not(atom);
    solver.assert(negated, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
}

/// The satisfiable twin of the `Gt` arm: `a = 0x0f ∧ a > 0x0e` holds, and the
/// model must still pin `a = 0x0f`.  Before the fix this was `sat` too — but
/// vacuously, because the atom asserted nothing; the point of the row is that
/// the new arm asserts `0x0e <u a` and not its converse, which would make this
/// a wrong `unsat`.
#[test]
fn satisfiable_gt_over_bv_operands_keeps_its_model() {
    let (mut solver, mut manager, a) = pinned_a_solver();
    let fourteen = manager.mk_bitvec(0x0e_i64, 8);
    let atom = manager.mk_gt(a, fourteen);
    solver.assert(atom, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    assert_model_pins_a(&solver, &mut manager, a);
}

/// The satisfiable twin of the `Ge` arm: `a = 0x0f ∧ a >= 0x0f`.
#[test]
fn satisfiable_ge_over_bv_operands_keeps_its_model() {
    let (mut solver, mut manager, a) = pinned_a_solver();
    let fifteen = manager.mk_bitvec(0x0f_i64, 8);
    let atom = manager.mk_ge(a, fifteen);
    solver.assert(atom, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    assert_model_pins_a(&solver, &mut manager, a);
}

/// `a` is entailed to be `0x0f` by the premise, so the reported model must say
/// so — the value is *entailed*, not one witness among several, and a search
/// order change cannot legitimately move it.
///
/// `Solver`'s `Model` maps a term to the `TermId` of its value term, and terms
/// are hash-consed, so interning `0x0f` at width 8 again yields exactly the
/// `TermId` the model must carry.
fn assert_model_pins_a(solver: &Solver, manager: &mut TermManager, a: TermId) {
    let fifteen = manager.mk_bitvec(0x0f_i64, 8);
    let value = solver
        .model()
        .expect("a `Sat` verdict must carry a model")
        .get(a);
    assert_eq!(
        value,
        Some(fifteen),
        "model must still pin a = 0x0f, got {:?}",
        value.and_then(|v| manager.get(v).map(|t| t.kind.clone()))
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 4. The primary repair: the parser rejects `(> bv bv)` outright (U-Z14)
// ─────────────────────────────────────────────────────────────────────────
//
// The two tests above's script ancestors are these.  `<`, `<=`, `>` and `>=`
// are arithmetic predicates; SMT-LIB has no bit-vector overload for them, so an
// ill-sorted `(> a #x0f)` over an 8-bit `a` is not a well-sorted script and the
// parser now says so.  `oxiz-core/tests/parser_sort_checks.rs` covers all four
// operators at the `parse_script` level; these two pin the end-to-end shape
// U-Z10's sibling family was reported as, through `Context::execute_script`,
// which is where an embedder meets it.
//
// The rejection surfaces as `Err(..)` from `execute_script` — its first
// statement parses the whole script — and **not** as an `(error ...)` line in
// the returned response vector, so `run_script`'s `unwrap_or_default()` would
// silently turn it into an empty output.  These tests therefore call
// `execute_script` directly.

/// `(= a #x0f) ∧ (> a #x0f)` is the shape that answered `sat` on 0.3.3/0.3.4
/// (report `I0-a` §4.1(b)).  It is now rejected before the solver sees it.
#[test]
fn a_greater_than_over_bit_vectors_is_a_parse_error() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (> a #x0f))
(check-sat)
";
    let message = script_rejection(script);
    assert!(message.contains("operands of >"), "{message}");
    assert!(message.contains("Int or Real"), "{message}");
    assert!(message.contains("(_ BitVec 8)"), "{message}");
}

/// The `>=` twin of the test above.
#[test]
fn a_greater_or_equal_over_bit_vectors_is_a_parse_error() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(assert (= a #x0f))
(assert (>= a #x0f))
(check-sat)
";
    let message = script_rejection(script);
    assert!(message.contains("operands of >="), "{message}");
    assert!(message.contains("Int or Real"), "{message}");
    assert!(message.contains("(_ BitVec 8)"), "{message}");
}

/// Execute `script` and return the error message it is rejected with, or panic
/// if it was accepted.
fn script_rejection(script: &str) -> String {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(outputs) => panic!(
            "expected an ill-sorted script to be rejected, got outputs {outputs:?}\n{script}"
        ),
        Err(err) => err.to_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 5. Model completeness after the rollback
// ─────────────────────────────────────────────────────────────────────────
//
// The rollback removes cache entries, and `get_model` builds bit-vector values
// by iterating `term_to_bv`.  The risk it creates is the mirror image of the
// bug: a genuinely satisfiable instance coming back with a missing or defaulted
// model entry.  These three scripts each force Boolean structure *and* a
// backjump; the `(get-value ...)` output below is the one both the unmodified
// 0.3.4 tree and the fixed tree produce, byte for byte.
//
// Which components are entailed by the script and which are one witness among
// several is called out per test: a later search-order change may legitimately
// move a witness, and that must not read as a soundness regression.

/// `a = 0x0f` (entailed) with a disjunction whose first branch is excluded, so
/// `b >u 0xf0` must hold.  `b = 0xff` is a **measured witness** — one of the
/// fifteen values above `0xf0` — recorded here because it is the value both
/// builds produce.
#[test]
fn satisfiable_disjunction_model_is_complete() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (= a #x0f))
(assert (or (and (bvult a #x05) (= b #x01)) (and (bvule a #x0f) (bvugt b #xf0))))
(check-sat)
(get-value (a b))
";
    let printed = run_script_output(script);
    assert!(
        printed.contains("(a #x0f)"),
        "a = 0x0f is entailed, got: {printed}"
    );
    assert!(
        printed.contains("(b #xff)"),
        "recorded witness b = 0xff, got: {printed}"
    );
}

/// u01's negated conjunction made satisfiable, plus `y = x + 1` (entailed by
/// the script given `x`) and `y <u 0xff`.  `x = 0x10` is a **measured
/// witness** — any `x` in `[0x11, 0xfd]` also works, and `0x10` is what both
/// builds pick; `y = 0x11` then follows from it.
#[test]
fn satisfiable_negated_conjunction_with_arithmetic_model_is_complete() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))
(assert (not (and (bvule x #x0f) (bvule x #x10))))
(assert (= y (bvadd x #x01)))
(assert (bvult y #xff))
(check-sat)
(get-value (x y))
";
    let printed = run_script_output(script);
    assert!(
        printed.contains("(x #x10)"),
        "recorded witness x = 0x10, got: {printed}"
    );
    assert!(
        printed.contains("(y #x11)"),
        "y = x + 1 is entailed by the witness, got: {printed}"
    );
}

/// Two two-way disjunctions with both first branches excluded, so
/// `a = 0x22` and `b = 0x44` are **entailed**: the model is unique and a
/// missing or defaulted entry would be caught outright.
#[test]
fn satisfiable_two_disjunctions_model_is_entailed() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (or (= a #x11) (= a #x22)))
(assert (or (= b #x33) (= b #x44)))
(assert (not (= a #x11)))
(assert (not (= b #x33)))
(check-sat)
(get-value (a b))
";
    let printed = run_script_output(script);
    assert!(
        printed.contains("(a #x22)"),
        "a = 0x22 is entailed, got: {printed}"
    );
    assert!(
        printed.contains("(b #x44)"),
        "b = 0x44 is entailed, got: {printed}"
    );
}
