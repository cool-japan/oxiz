//! Tests for the model-verification gate's evaluator ([`super`]).
//!
//! Split out of `model_eval.rs` to keep that file inside the 2000-line policy
//! limit; `super` still resolves to the evaluator module, so every private item
//! these tests reach stays reachable.

use super::{ENCODE_DEPTH_LIMIT, EvalOutcome, EvalVal};
use crate::solver::Solver;
use crate::solver::types::Model;
use num_rational::Rational64;
use oxiz_core::ast::{TermId, TermManager};

/// `2^62` fits `i64`, but `2^62 + 2^62 = 2^63` does not — the smallest
/// round number that makes `Rational64` addition overflow.
const HALF_MAX: i64 = 1 << 62;

/// The stack the in-budget regression tests in this module run their
/// evaluation on.  1 MiB is what an embedder's worker thread typically
/// gets, and a native stack overflow aborts the process — so "the closure
/// returned at all" is itself part of each assertion.
// STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — pins the
// realistic embedder worker-thread budget, not a scaled test depth.
// See TODO.md "v0.3.2 backlog".
const WORKER_STACK: usize = 1 << 20;

/// The stack the *past-the-budget* test below runs on.  It is an eighth of
/// [`WORKER_STACK`], paired with an eighth of that test's depth, so the
/// bytes-per-frame threshold the test really pins (~21 B per level) is
/// unchanged while the term the test has to build — and keep interned —
/// shrinks by 8x.  Never change one of the two without the other.
const DEEP_WORKER_STACK: usize = 1 << 17;

/// Run `body` on a fresh thread with `stack_size` bytes of stack.
fn on_stack<T: Send + 'static>(stack_size: usize, body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(stack_size)
        .spawn(body)
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not abort")
}

/// Run `body` on a fresh [`WORKER_STACK`] thread and return its result.
fn on_worker_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    on_stack(WORKER_STACK, body)
}

/// A solver holding exactly `assertions`, with an empty (but present)
/// model, ready for the gate.
fn solver_with(assertions: Vec<TermId>) -> Solver {
    let mut solver = Solver::new();
    solver.assertions = assertions;
    solver.model = Some(Model::new());
    solver
}

/// The gate's answer for a single assertion.
fn gate_refuses(manager: &TermManager, assertion: TermId) -> bool {
    solver_with(vec![assertion]).model_refutes_assertions(manager)
}

/// The evaluator's outcome for a single term.
fn outcome(manager: &TermManager, term: TermId) -> EvalOutcome {
    let solver = solver_with(Vec::new());
    let model = Model::new();
    solver.eval_in_model_outcome(term, &model, manager, 0)
}

/// An overflowing `+` under an assertion the model genuinely **violates**.
///
/// `(< (+ 2^62 2^62) 0)` is false — `2^63` is positive — so the gate must
/// refuse the model.  Unchecked, this wrapped to `i64::MIN < 0` in release
/// and reported `true`, hiding the violation; in debug it aborted with
/// `attempt to add with overflow` before answering anything at all.
#[test]
fn overflowing_addition_never_hides_a_violated_assertion() {
    let mut manager = TermManager::new();
    let half = manager.mk_int(HALF_MAX);
    let sum = manager.mk_add([half, half]);
    let zero = manager.mk_int(0);
    let assertion = manager.mk_lt(sum, zero);

    assert_eq!(outcome(&manager, sum), EvalOutcome::Unrepresentable);
    assert!(gate_refuses(&manager, assertion));
}

/// The same overflow under an assertion the model **satisfies**.
///
/// `(>= (+ 2^62 2^62) 0)` is true, so refusing the model costs precision —
/// the caller answers `Unknown` for a formula it could have called `Sat`.
/// That is the deliberate direction: a `false` answer from the gate is
/// consumed as "report `Sat`", and an assertion the evaluator could not
/// evaluate is no evidence that the model satisfies it.  Unchecked, this
/// wrapped the other way and refuted the model on garbage.
#[test]
fn overflowing_addition_never_vouches_for_a_model() {
    let mut manager = TermManager::new();
    let half = manager.mk_int(HALF_MAX);
    let sum = manager.mk_add([half, half]);
    let zero = manager.mk_int(0);
    let assertion = manager.mk_ge(sum, zero);

    assert!(gate_refuses(&manager, assertion));
}

/// Every arithmetic operator the evaluator folds is checked, not just `+`.
#[test]
fn every_arithmetic_operator_reports_overflow() {
    let mut manager = TermManager::new();
    let half = manager.mk_int(HALF_MAX);
    let two = manager.mk_int(2);
    let min = manager.mk_int(i64::MIN);
    let max = manager.mk_int(i64::MAX);

    let product = manager.mk_mul([half, two]);
    let difference = manager.mk_sub(min, max);
    let negation = manager.mk_neg(min);

    for term in [product, difference, negation] {
        assert_eq!(outcome(&manager, term), EvalOutcome::Unrepresentable);
    }
}

/// An overflow the surrounding formula never depends on must not leak out.
///
/// `(or true (< (+ 2^62 2^62) 0))` is decided by its first disjunct, and
/// `(and false …)` by its first conjunct, so neither may be downgraded.
/// This is what the three-valued outcome buys over a "saw an overflow
/// anywhere" flag.
#[test]
fn short_circuited_overflow_does_not_downgrade() {
    let mut manager = TermManager::new();
    let half = manager.mk_int(HALF_MAX);
    let sum = manager.mk_add([half, half]);
    let zero = manager.mk_int(0);
    let overflowing = manager.mk_lt(sum, zero);
    // `mk_or` / `mk_and` drop a literal `true` / `false` operand outright,
    // so the deciding operand has to be a comparison the *evaluator*
    // folds rather than one the builder does.
    let one = manager.mk_int(1);
    let truth = manager.mk_lt(zero, one);

    let disjunction = manager.mk_or([truth, overflowing]);
    assert_eq!(
        outcome(&manager, disjunction),
        EvalOutcome::Value(EvalVal::Bool(true))
    );
    assert!(!gate_refuses(&manager, disjunction));

    // `(and <overflow> false)` is `false` whatever the overflow was: the
    // gate refuses, but as a genuine refutation rather than a shrug.
    let falsehood = manager.mk_lt(one, zero);
    let conjunction = manager.mk_and([overflowing, falsehood]);
    assert_eq!(
        outcome(&manager, conjunction),
        EvalOutcome::Value(EvalVal::Bool(false))
    );
    assert!(gate_refuses(&manager, conjunction));
}

/// An unevaluable term that is *not* an overflow stays inconclusive.
///
/// `distinct`, a numeric equality collision and a strict comparison at its
/// boundary are all `Undetermined`, and none of them may downgrade a `Sat`
/// — that distinction is the whole reason the outcome is three-valued and
/// not two.
#[test]
fn ordinary_inconclusiveness_never_downgrades() {
    let mut manager = TermManager::new();
    let one = manager.mk_int(1);
    let zero = manager.mk_int(0);
    // Terms are hash-consed, so `mk_int(1)` twice is the *same* term and
    // `mk_eq` would fold it to `true`.  `(+ 0 1)` is a distinct term with
    // the same value, which is exactly the collision the gate distrusts.
    let other_one = manager.mk_add([zero, one]);

    let collision = manager.mk_eq(one, other_one);
    let boundary = manager.mk_lt(one, other_one);
    let unconstrained = manager.mk_var("x", manager.sorts.int_sort);
    let opaque = manager.mk_ge(unconstrained, one);

    for term in [collision, boundary, opaque] {
        assert_eq!(outcome(&manager, term), EvalOutcome::Undetermined);
        assert!(!gate_refuses(&manager, term));
    }
}

/// The evaluator still computes what it always did for ordinary terms.
#[test]
fn arithmetic_and_comparisons_still_fold() {
    let mut manager = TermManager::new();
    let two = manager.mk_int(2);
    let three = manager.mk_int(3);
    let seven = manager.mk_int(7);

    let sum = manager.mk_add([two, three]);
    assert_eq!(
        outcome(&manager, sum),
        EvalOutcome::Value(EvalVal::Num(Rational64::from_integer(5)))
    );
    let product = manager.mk_mul([two, three]);
    assert_eq!(
        outcome(&manager, product),
        EvalOutcome::Value(EvalVal::Num(Rational64::from_integer(6)))
    );
    let difference = manager.mk_sub(three, seven);
    assert_eq!(
        outcome(&manager, difference),
        EvalOutcome::Value(EvalVal::Num(Rational64::from_integer(-4)))
    );
    let negated = manager.mk_neg(seven);
    assert_eq!(
        outcome(&manager, negated),
        EvalOutcome::Value(EvalVal::Num(Rational64::from_integer(-7)))
    );

    let less = manager.mk_lt(sum, product);
    assert_eq!(
        outcome(&manager, less),
        EvalOutcome::Value(EvalVal::Bool(true))
    );
    let at_least = manager.mk_ge(difference, seven);
    assert_eq!(
        outcome(&manager, at_least),
        EvalOutcome::Value(EvalVal::Bool(false))
    );
    let implication = manager.mk_implies(less, at_least);
    assert_eq!(
        outcome(&manager, implication),
        EvalOutcome::Value(EvalVal::Bool(false))
    );
    let choice = manager.mk_ite(less, difference, seven);
    assert_eq!(
        outcome(&manager, choice),
        EvalOutcome::Value(EvalVal::Num(Rational64::from_integer(-4)))
    );
}

/// A deeply nested assertion is evaluated on the heap, not the native
/// stack, and still produces the *exact* right verdict.
///
/// The chain is built with an iterative loop (a recursive test helper would
/// move the overflow into the test itself) and evaluated on a 1 MiB thread.
/// Each level is one `Sub` frame, which the recursive evaluator paid for
/// with a native frame; 1900 of them are well inside the depth budget and
/// were comfortably enough to exhaust that stack.
#[test]
fn deeply_nested_assertion_evaluates_on_a_worker_stack() {
    // Track the real budget so the pin survives future limit changes:
    // stay just inside ENCODE_DEPTH_LIMIT (the chain is DEPTH levels deep,
    // and the `(>= chain 1)` assertion adds one more).
    const DEPTH: i64 = ENCODE_DEPTH_LIMIT as i64 - 50;

    let refused = on_worker_stack(|| {
        let mut manager = TermManager::new();
        let one = manager.mk_int(1);
        let mut chain = manager.mk_int(0);
        for _ in 0..DEPTH {
            chain = manager.mk_sub(chain, one);
        }
        // `chain` is exactly `-DEPTH`, so `(>= chain 0)` is false: the gate
        // must refute, and refute for the right reason.
        let value = outcome(&manager, chain);
        let assertion = manager.mk_ge(chain, one);
        (value, gate_refuses(&manager, assertion))
    });

    assert_eq!(
        refused.0,
        EvalOutcome::Value(EvalVal::Num(Rational64::from_integer(-DEPTH)))
    );
    assert!(refused.1);
}

/// A chain past the evaluator's depth budget answers `Undetermined` — the
/// same answer the recursive version gave — rather than aborting the
/// process on the way there.
///
/// Stack and depth scale together (1 MiB/50k -> 128 KiB/6.25k): the
/// ~21 B-per-frame threshold is the pin, so never raise one alone.
#[test]
fn assertion_past_the_depth_budget_stays_inconclusive() {
    const DEPTH: usize = 6_250;

    let (value, refused) = on_stack(DEEP_WORKER_STACK, || {
        let mut manager = TermManager::new();
        let one = manager.mk_int(1);
        let mut chain = manager.mk_int(0);
        for _ in 0..DEPTH {
            chain = manager.mk_sub(chain, one);
        }
        let assertion = manager.mk_ge(chain, one);
        (outcome(&manager, chain), gate_refuses(&manager, assertion))
    });

    assert_eq!(value, EvalOutcome::Undetermined);
    assert!(!refused);
}

// -----------------------------------------------------------------
// The trail-polarity half of the gate.
// -----------------------------------------------------------------

/// Build a solver whose SAT core has committed `eq_term` to **false**, and
/// whose model gives `lhs` and `rhs` the same integer value.
///
/// `lhs`/`rhs` are deliberately *uninterpreted applications*, not Int
/// `Var`s: the evaluator reads an Int `Var` from the arithmetic tableau
/// (`arith.value`), which a unit test cannot populate without running a
/// solve, whereas an opaque leaf is read straight from the model witness.
fn solver_with_false_equality(
    manager: &mut TermManager,
    eq_term: TermId,
    lhs: TermId,
    rhs: TermId,
    value: TermId,
) -> Solver {
    use crate::solver::types::Constraint;
    use oxiz_sat::Lit;

    let mut solver = Solver::new();
    let var = solver.get_or_create_var(eq_term);
    solver.record_constraint(var, Constraint::Eq(lhs, rhs));
    // Force the atom false and solve, so `sat.model_value(var)` really is
    // `LBool::False` rather than `Undef`.
    solver.sat.add_clause([Lit::neg(var)]);
    let _ = solver.sat.solve();

    let mut model = Model::new();
    model.set(lhs, value);
    model.set(rhs, value);
    solver.model = Some(model);
    let _ = manager;
    solver
}

/// The witness the false-`sat` family left behind: the core committed
/// `(= (f 1) (g 1))` to **false**, then produced a model giving both sides
/// `7`. The assignment and the model contradict each other outright, so
/// the gate must refuse the verdict.
///
/// `combine_eq` structurally cannot catch this — it sees two equal numbers
/// and answers `Undetermined` by design, because a collision in the LP
/// model is not by itself evidence of anything. The missing information is
/// the trail polarity, which only this gate has.
#[test]
fn a_trail_false_equality_whose_sides_collide_refutes_the_model() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let one = manager.mk_int(1);
    let f1 = manager.mk_apply("f", [one], int_sort);
    let g1 = manager.mk_apply("g", [one], int_sort);
    let eq = manager.mk_eq(f1, g1);
    let seven = manager.mk_int(7);

    let solver = solver_with_false_equality(&mut manager, eq, f1, g1, seven);
    assert!(
        solver.model_refutes_assertions(&manager),
        "an `Eq` assigned false whose sides the model makes equal is a \
         definite refutation, not a coincidence"
    );
}

/// The same shape with the model giving the two sides *different* values
/// is a perfectly good model of the disequality, and must pass the gate.
/// Without this control the test above would also pass if the gate simply
/// refused every trail-false equality.
#[test]
fn a_trail_false_equality_with_distinct_values_passes_the_gate() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let one = manager.mk_int(1);
    let f1 = manager.mk_apply("f", [one], int_sort);
    let g1 = manager.mk_apply("g", [one], int_sort);
    let eq = manager.mk_eq(f1, g1);
    let seven = manager.mk_int(7);
    let eight = manager.mk_int(8);

    let mut solver = solver_with_false_equality(&mut manager, eq, f1, g1, seven);
    let Some(model) = solver.model.as_mut() else {
        panic!("the helper always installs a model");
    };
    model.set(g1, eight);
    assert!(
        !solver.model_refutes_assertions(&manager),
        "7 != 8 satisfies the disequality the core committed to"
    );
}

/// A *Bool*-sorted equality assigned false must be ignored by this gate
/// even when both sides carry the same model witness: Booleans are the EUF
/// / SAT layer's business, and `Constraint::Eq` over them is also used to
/// feed congruence closure. Firing here would cost legitimate `sat`
/// verdicts.
#[test]
fn a_trail_false_boolean_equality_is_not_this_gates_business() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let p = manager.mk_var("p", bool_sort);
    let q = manager.mk_var("q", bool_sort);
    let eq = manager.mk_eq(p, q);
    let t = manager.mk_true();

    let solver = solver_with_false_equality(&mut manager, eq, p, q, t);
    assert!(
        !solver.model_refutes_assertions(&manager),
        "a Bool-sorted equality has no arithmetic value to collide"
    );
}

// ─────────────────────────────────────────────────────────────────
// `#P2b-22` — the structural pre-pass
// ─────────────────────────────────────────────────────────────────
//
// End-to-end evidence that the pre-pass really is the second line of defence
// `#P2b-20` lacked, measured on a scratch copy of this workspace on
// 2026-09-14 with the `#P2b-20` fix in `oxiz_sat::Solver::pop` deliberately
// reverted to its unconditional `self.trivially_unsat = false;`:
//
// | script (`(assert (distinct x x)) (push 1) (pop 1) (check-sat)`) | pre-pass off | pre-pass on |
// |---|---|---|
// | `x : (_ BitVec 8)` | `sat`  (wrong) | `unknown` |
// | `x : Int`          | `sat`  (wrong) | `unknown` |
// | `x : Bool`         | `sat`  (wrong) | `unknown` |
//
// `unknown` is the most the gate can ever produce — it inspects a finished
// candidate model and can only downgrade a verdict, never correct one — and it
// is the difference between a solver that publishes a wrong `sat` and one that
// declines to answer.  On this tree the same scripts answer `unsat`, because
// `#P2b-20` is fixed and the gate is never reached; the unit tests below are
// therefore what pins the pre-pass here.

/// `(distinct b b)` over a bit-vector variable the model does not pin.
///
/// This is the exact assertion that got past the gate in `#P2b-20`:
/// `mk_eq(b, b)` folds to `true` at construction, so `b` never reaches the
/// bit-blaster and the published model has no value for it — a *value*-based
/// evaluator can only answer `Undetermined`, and the gate then waved through
/// a `sat` any reader can see is wrong.  Deciding it needs no value at all:
/// `x` is `x`, so `(distinct x x)` is false whatever the model says.
#[test]
fn a_distinct_over_one_repeated_bit_vector_operand_is_refuted() {
    let mut manager = TermManager::new();
    let sort = manager.sorts.bitvec(8);
    let b = manager.mk_var("b", sort);
    let assertion = manager.mk_distinct([b, b]);

    assert_eq!(
        outcome(&manager, assertion),
        EvalOutcome::Value(EvalVal::Bool(false)),
        "`(distinct b b)` is false structurally, with no model value for `b`"
    );
    assert!(
        gate_refuses(&manager, assertion),
        "the gate must refute a candidate model of `(distinct b b)` (#P2b-22)"
    );
}

/// The same for every other sort the gate can be handed, since the pre-pass
/// is deliberately sort-independent: an operand is identical to itself in
/// any interpretation, so no sort-specific reasoning is involved.
///
/// `Int` matters most here: the value half of `Op::Distinct` decides *only*
/// a uniform-width bit-vector tuple and bails out to `Undetermined` on
/// anything else (deliberately — the LP model collides arithmetic values
/// routinely), so without the structural pre-pass an arithmetic
/// `(distinct i i)` is exactly as invisible as the bit-vector one.
#[test]
fn a_distinct_with_a_repeated_operand_is_refuted_for_every_sort() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let real_sort = manager.sorts.real_sort;
    let bool_sort = manager.sorts.bool_sort;
    let i = manager.mk_var("i", int_sort);
    let r = manager.mk_var("r", real_sort);
    let p = manager.mk_var("p", bool_sort);

    for (name, term) in [("Int", i), ("Real", r), ("Bool", p)] {
        let assertion = manager.mk_distinct([term, term]);
        assert_eq!(
            outcome(&manager, assertion),
            EvalOutcome::Value(EvalVal::Bool(false)),
            "`(distinct x x)` over {name} is false structurally"
        );
        assert!(
            gate_refuses(&manager, assertion),
            "the gate must refute `(distinct x x)` over {name} (#P2b-22)"
        );
    }
}

/// A repeated operand buried in a longer `distinct` decides it too — the
/// pre-pass looks for *any* colliding pair, not just a two-operand term —
/// and it fires even though the other operands have no model value.
#[test]
fn a_repeated_operand_anywhere_in_a_distinct_decides_it() {
    let mut manager = TermManager::new();
    let sort = manager.sorts.bitvec(8);
    let a = manager.mk_var("a", sort);
    let b = manager.mk_var("b", sort);
    let c = manager.mk_var("c", sort);
    let assertion = manager.mk_distinct([a, b, c, b]);

    assert_eq!(
        outcome(&manager, assertion),
        EvalOutcome::Value(EvalVal::Bool(false)),
        "`b` cannot differ from itself, so the whole `distinct` is false"
    );
    assert!(gate_refuses(&manager, assertion));
}

/// The control, and the property the pre-pass must not break: a `distinct`
/// over *different* operands the model does not pin stays inconclusive.
///
/// This is the behaviour the gate has always had for arithmetic `distinct`
/// and must keep — the linear-arithmetic solver enforces disequalities by
/// case splitting rather than by pinning distinct witnesses, so refusing
/// here would turn correct `sat` verdicts into spurious `unknown`s.
#[test]
fn a_distinct_over_different_operands_stays_inconclusive() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let assertion = manager.mk_distinct([x, y]);

    assert_eq!(
        outcome(&manager, assertion),
        EvalOutcome::Undetermined,
        "two different variables may or may not collide; the gate declines"
    );
    assert!(
        !gate_refuses(&manager, assertion),
        "refusing here would cost legitimate `sat` verdicts"
    );
}

/// Why the `Eq(x, x)` half of the pre-pass has no reachability test of its
/// own: [`TermManager::mk_eq`] folds `x = x` to `true` before interning, so
/// no `TermKind::Eq` with identical operands can be built through the
/// builder or the SMT-LIB parser at all.
///
/// The arm exists as defence in depth for a term interned some other way,
/// and for the day that fold changes.  This test pins the fold itself, so
/// the arm's "unreachable through the builder" rationale cannot silently
/// stop being true.
#[test]
fn an_equality_of_a_term_with_itself_is_folded_away_before_the_gate() {
    let mut manager = TermManager::new();
    let sort = manager.sorts.bitvec(8);
    let b = manager.mk_var("b", sort);
    let eq = manager.mk_eq(b, b);

    assert_eq!(
        eq,
        manager.mk_true(),
        "`mk_eq` folds `x = x` to `true` at construction"
    );
    assert_eq!(
        outcome(&manager, eq),
        EvalOutcome::Value(EvalVal::Bool(true))
    );
    assert!(
        !gate_refuses(&manager, eq),
        "`true` is satisfied by every model"
    );
}

/// The gate's other refusal (`#P2b-27`): an assertion that evaluates
/// `Undetermined` *because a Boolean variable in it has no model entry* is
/// not skipped — the published model prints a default for that variable and
/// may falsify the assertion with it.  `(= x (ite p #x01 #x02))` with
/// `x = #x01` in the model and no entry for `p` is the shape that let the
/// pre-`#P2b-24` free-bit model through; with `p` present the same
/// assertion evaluates and the refusal does not fire.
#[test]
fn an_undetermined_assertion_over_an_unassigned_boolean_is_refused() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let bool_sort = manager.sorts.bool_sort;
    let p = manager.mk_var("p", bool_sort);
    let x = manager.mk_var("x", bv8);
    let one = manager.mk_bitvec(1, 8);
    let two = manager.mk_bitvec(2, 8);
    let ite = manager.mk_ite(p, one, two);
    let assertion = manager.mk_eq(x, ite);

    let mut solver = solver_with(vec![assertion]);
    let mut model = Model::new();
    model.set(x, one);
    solver.model = Some(model);
    assert!(
        !solver.model_refutes_assertions(&manager),
        "without p the assertion is Undetermined, which the refuting gate skips"
    );
    assert!(
        solver.model_leaves_a_boolean_undetermined(&manager),
        "with p completed by the printed default `false` the assertion is false: refused"
    );

    let mut model = Model::new();
    model.set(x, one);
    let true_term = manager.mk_true();
    model.set(p, true_term);
    solver.model = Some(model);
    assert!(
        !solver.model_leaves_a_boolean_undetermined(&manager),
        "with p published the assertion evaluates (to true) and nothing is refused"
    );
    assert!(!solver.model_refutes_assertions(&manager));
}

/// A tautology over a Boolean the encoder folded away — `(or a (not a))`,
/// the shape the model counter enumerates — holds under the printed default
/// and is *not* refused: the refusal is about models the user would see
/// falsify an assertion, not about missing entries as such.
#[test]
fn a_tautology_over_an_unassigned_boolean_is_not_refused() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let a = manager.mk_var("a", bool_sort);
    let not_a = manager.mk_not(a);
    let assertion = manager.mk_or([a, not_a]);
    let solver = solver_with(vec![assertion]);
    assert!(!solver.model_leaves_a_boolean_undetermined(&manager));
}

/// A numeric variable without a tableau value is *not* what the Boolean
/// refusal is about: `(distinct i j)` over two unconstrained integers stays
/// `Undetermined` and passes both gates, exactly as before.
#[test]
fn an_undetermined_assertion_over_numeric_variables_is_not_refused() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let i = manager.mk_var("i", int_sort);
    let j = manager.mk_var("j", int_sort);
    let assertion = manager.mk_distinct([i, j]);
    let solver = solver_with(vec![assertion]);
    assert!(!solver.model_refutes_assertions(&manager));
    assert!(!solver.model_leaves_a_boolean_undetermined(&manager));
}

// ---------------------------------------------------------------------------
// `#P2b-32` — a `select` over a `store` is read as read-over-write by the
// gate, and as its published leaf by the array-axiom instantiator.
// ---------------------------------------------------------------------------

/// The gate's half of `#P2b-32`: a model whose published leaf for a read
/// contradicts the store it reads over is refused.  Before the `select` arm
/// the read fell into the opaque-leaf arm and evaluated to whatever the
/// free bits said — `(distinct (bvadd (select (store arr i #x05) i) #x01)
/// #x06)` was published `sat` with the read at `#xfe`.
#[test]
fn a_read_over_write_leaf_contradicting_the_store_is_refuted() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let array_sort = manager.sorts.array(bv8, bv8);
    let arr = manager.mk_var("arr", array_sort);
    let i = manager.mk_var("i", bv8);
    let five = manager.mk_bitvec(5, 8);
    let one = manager.mk_bitvec(1, 8);
    let six = manager.mk_bitvec(6, 8);
    let store = manager.mk_store(arr, i, five);
    let read = manager.mk_select(store, i);
    let sum = manager.mk_bv_add(read, one);
    let assertion = manager.mk_distinct([sum, six]);

    let zero = manager.mk_bitvec(0, 8);
    let free_leaf = manager.mk_bitvec(0xfe, 8);
    let mut model = Model::new();
    model.set(i, zero);
    model.set(read, free_leaf);
    let mut solver = solver_with(vec![assertion]);
    solver.model = Some(model.clone());
    assert!(
        solver.model_refutes_assertions(&manager),
        "read-over-write makes the read 5, the sum 6, and the distinct false"
    );

    // The instantiator's reading is the published leaf: the read-over-write
    // instance `(= i i) ⇒ (= read 5)` is *not* satisfied by this model, so it
    // is asserted — which is what closes the free leaf for good.
    let index_equal = manager.mk_eq(i, i);
    let hit = manager.mk_eq(read, five);
    let instance = manager.mk_implies(index_equal, hit);
    assert_ne!(
        solver.eval_in_model(instance, &model, &manager, 0),
        Some(EvalVal::Bool(true)),
        "read by its published leaf the instance is violated, so the instantiator asserts it"
    );
    assert_eq!(
        solver.eval_in_model_outcome(instance, &model, &manager, 0),
        EvalOutcome::boolean(true),
        "read as read-over-write the instance is a tautology, which is why the gate must not be the instantiator's reader"
    );

    // The satisfiable twin `(= (bvadd read 1) 6)` is judged on the model the
    // user sees — the array as printed, read by read-over-write — and holds
    // under it whatever the circuit's leaf said: the leaf is not part of the
    // printed model, so neither a consistent nor a stale entry for it can
    // make the gate refuse.
    let satisfiable = manager.mk_eq(sum, six);
    let mut solver = solver_with(vec![satisfiable]);
    for leaf in [five, free_leaf] {
        let mut model = Model::new();
        model.set(i, zero);
        model.set(read, leaf);
        solver.model = Some(model);
        assert!(
            !solver.model_refutes_assertions(&manager),
            "the printed model satisfies the twin whatever the leaf entry says"
        );
    }
}

/// A miss on every level reads the published entry of the innermost base
/// at the same index — the term the read-over-write lemma interns — and
/// refutes a model that contradicts it; without such an entry the read is
/// inconclusive.
#[test]
fn a_read_over_write_miss_reads_the_base_entry() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let array_sort = manager.sorts.array(bv8, bv8);
    let arr = manager.mk_var("arr", array_sort);
    let i = manager.mk_var("i", bv8);
    let j = manager.mk_var("j", bv8);
    let v = manager.mk_var("v", bv8);
    let one = manager.mk_bitvec(1, 8);
    let six = manager.mk_bitvec(6, 8);
    let store = manager.mk_store(arr, i, v);
    let read = manager.mk_select(store, j);
    let sum = manager.mk_bv_add(read, one);
    let assertion = manager.mk_eq(sum, six);
    let base_read = manager.mk_select(arr, j);

    let zero = manager.mk_bitvec(0, 8);
    let seven = manager.mk_bitvec(7, 8);
    let five = manager.mk_bitvec(5, 8);
    let mut model = Model::new();
    model.set(i, zero);
    model.set(j, one);
    model.set(base_read, seven);
    let mut solver = solver_with(vec![assertion]);
    solver.model = Some(model);
    assert!(
        solver.model_refutes_assertions(&manager),
        "i ≠ j misses the store, the base reads 7, the sum is 8, not 6"
    );

    let mut model = Model::new();
    model.set(i, zero);
    model.set(j, one);
    model.set(base_read, five);
    solver.model = Some(model);
    assert!(!solver.model_refutes_assertions(&manager));

    let mut model = Model::new();
    model.set(i, zero);
    model.set(j, one);
    solver.model = Some(model);
    assert!(
        !solver.model_refutes_assertions(&manager),
        "no entry for the base read: inconclusive, never a refutation"
    );
}

/// An index the model does not pin keeps the read inconclusive, and a
/// numeric index collision is not a hit: the gate never manufactures a
/// read-over-write refutation out of a defaulted or colliding value.
#[test]
fn a_read_over_write_with_an_unpinned_or_colliding_index_stays_inconclusive() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let array_sort = manager.sorts.array(int_sort, int_sort);
    let arr = manager.mk_var("arr", array_sort);
    let i = manager.mk_var("i", int_sort);
    let j = manager.mk_var("j", int_sort);
    let five = manager.mk_int(5);
    let one = manager.mk_int(1);
    let six = manager.mk_int(6);
    let store = manager.mk_store(arr, i, five);
    let read = manager.mk_select(store, j);
    let sum = manager.mk_add([read, one]);
    let assertion = manager.mk_distinct([sum, six]);

    // No tableau value for `i` or `j`: the index is `Undetermined`.
    let solver = solver_with(vec![assertion]);
    assert_eq!(
        solver.eval_in_model_outcome(read, &Model::new(), &manager, 0),
        EvalOutcome::UNDETERMINED
    );
    assert!(!solver.model_refutes_assertions(&manager));
}

/// A read with no store under it is the leaf it always was: the model's
/// entry for the `select` term, under both readings.
#[test]
fn a_read_with_no_store_is_the_published_leaf() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let array_sort = manager.sorts.array(bv8, bv8);
    let arr = manager.mk_var("arr", array_sort);
    let i = manager.mk_var("i", bv8);
    let read = manager.mk_select(arr, i);
    let five = manager.mk_bitvec(5, 8);
    let mut model = Model::new();
    model.set(read, five);
    let solver = solver_with(Vec::new());
    assert_eq!(
        solver.eval_in_model_outcome(read, &model, &manager, 0),
        EvalOutcome::bits(5.into(), 8)
    );
    assert_eq!(
        solver.eval_in_model(read, &model, &manager, 0),
        Some(EvalVal::Bv {
            value: 5.into(),
            width: 8
        })
    );
    assert_eq!(
        solver.eval_in_model_outcome(read, &Model::new(), &manager, 0),
        EvalOutcome::UNDETERMINED
    );
}

/// `(get-value)`'s reading (`Solver::model_value_in`) folds numeric leaves
/// from the **published model**, never from the tableau: a variable the
/// nonlinear engine decided has a model entry the tableau knows nothing
/// about, and the old routing printed the tableau's stale `0` for `x` where
/// the model said `-2` (`nlsat_feature_gate::a_decided_goal_still_carries_its_model`).
/// A term with a direct entry is answered with that entry verbatim; a
/// compound term folds over the model's leaves; the gate's own reading of
/// the same variable stays `Undetermined`.
#[test]
fn get_value_reads_numeric_leaves_from_the_model_not_the_tableau() {
    let mut manager = TermManager::new();
    let real_sort = manager.sorts.real_sort;
    let x = manager.mk_var("x", real_sort);
    let minus_two = manager.mk_real(Rational64::from_integer(-2));
    let square = manager.mk_mul([x, x]);
    let mut model = Model::new();
    model.set(x, minus_two);
    let mut solver = solver_with(Vec::new());
    solver.model = Some(model);

    let published = solver.model.clone().expect("model");
    assert_eq!(
        solver.model_value_in(x, &published, &mut manager),
        Some(minus_two),
        "a direct entry is printed as it is"
    );
    let four = manager.mk_real(Rational64::from_integer(4));
    assert_eq!(
        solver.model_value_in(square, &published, &mut manager),
        Some(four),
        "x * x folds over the model's -2, not the tableau's nothing"
    );
    assert_eq!(
        solver.eval_in_model_outcome(square, solver.model.as_ref().expect("model"), &manager, 0),
        EvalOutcome::UNDETERMINED,
        "the gate keeps reading the tableau, which never constrained x"
    );
}
