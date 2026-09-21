//! Round 4, adversarial recheck pass 7 — pins for what this pass found.
//!
//! # What this file is for
//!
//! Re-fix pass 6 closed the *named* mechanism of finding R5-1: an array read
//! that is first ground inside a quantifier instance.  It closed it for the
//! spelling the finding used — a quantifier that **is** the assertion, or one
//! conjunct of it.  This pass asked what happens to the same quantifier in
//! every other Boolean position a script may put it in, and found that the
//! answer is a wrong `sat`, with no array in sight:
//!
//! ```text
//! (declare-sort U 0)
//! (declare-fun f (U) U)
//! (assert (not (forall ((x U)) (= (f x) (f x)))))
//! (check-sat)          ; sat — the negation of a tautology
//! ```
//!
//! A quantifier reaches the SAT core as one Tseitin literal.  Asserted
//! positively — at the top level or under an `and` — that literal is unit and
//! the MBQI fixpoint has to justify it.  In any other position (`not`, `or`,
//! `=>`, `ite`, a Boolean `=`) nothing ties the literal to the quantifier's
//! meaning, so the SAT solver may pick whichever value closes the search, and
//! `check-sat` answers `sat` for a formula that has no model.  Every hole
//! pinned in section 1 is one spelling of that.
//!
//! The family predates this round: `c4b04b7` and crates.io 0.3.3 answer the
//! same wrong `sat` to all of them.  What the round changed is its *reach* —
//! `encode::finite_expand`'s new whole-sort expansion (`#P2b-47`) decides the
//! narrow-bit-vector spellings outright, so the family now begins one index
//! bit above the 64-point expansion budget instead of at width 1.
//!
//! # The two conventions this file follows
//!
//! * A test whose doc carries **THE HOLE IS CLOSED** asserts the answer this
//!   tree gives *today*, which is the wrong one.  It is green now and it goes
//!   red on the pass that fixes the defect; that pass inverts it and deletes
//!   the note.  It never asserts that the wrong answer is correct.
//! * Every other test asserts the **correct** answer and is an ordinary
//!   regression guard.  Those are the shapes re-fix pass 6 fixed, plus the
//!   controls that make each hole above a statement about the Boolean context
//!   rather than about arrays, widths or sorts.
//!
//! No test here puts a verdict behind a wall clock and none carries a
//! `(set-option :timeout N)` (decision (16)).

use oxiz_solver::Context;

/// The all-zero array over `(_ BitVec 7)`, plus the declaration that pins it.
///
/// Index width 7 is 128 points, one bit above `encode::finite_expand`'s
/// 64-point budget, so every script built on this header keeps the MBQI path
/// and none of them is decided by the whole-sort expansion.
const PINNED_ARRAY: &str = "(set-logic ALL)\n\
     (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
     (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))\n";

/// The quantifier every script in section 1 argues about.
///
/// `a` is the all-zero array, so at `i = #b0000000` the `store` misses index 1
/// and the read is `#b0000000`, not `#b0000101`.  The body is therefore false
/// at that point and the quantifier is **unsatisfiable** — which
/// [`the_quantifier_is_unsatisfiable_at_a_ground_index`] establishes without
/// any quantifier at all, so nothing here rests on reasoning about binders.
const ROW_FORALL: &str = "(forall ((i (_ BitVec 7))) \
     (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7)))";

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// The last `sat`/`unsat`/`unknown` line, or `"none"`.
fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// Every `sat`/`unsat`/`unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

/// The whole response, joined, for an assertion message.
fn joined(lines: &[String]) -> String {
    lines.join("\n")
}

fn assert_verdict(script: &str, expected: &str, why: &str) {
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        expected,
        "{why}\n--- script ---\n{script}--- response ---\n{}",
        joined(&lines)
    );
}

// ---------------------------------------------------------------------------
// 0. THE TRUTH, ESTABLISHED WITHOUT A BINDER
// ---------------------------------------------------------------------------

/// One ground instance of [`ROW_FORALL`]'s body is already unsatisfiable.
///
/// `(select (store a #b0000000 #b0000101) #b0000001)` reads `a` at index 1,
/// and `a` is the all-zero array.  So the body is false at `i = #b0000000`,
/// the quantifier has no model, and every `sat` in section 1 is wrong —
/// established here by the ground solver, with no quantifier, no MBQI and no
/// external oracle.
#[test]
fn the_quantifier_is_unsatisfiable_at_a_ground_index() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (assert (= (select (store a (_ bv0 7) (_ bv5 7)) (_ bv1 7)) (_ bv5 7)))\n\
             (check-sat)\n"
        ),
        "unsat",
        "the body's own ground instance at index 0 must be refuted; if this \
         ever answers `sat` the rest of this file is arguing about the wrong \
         formula",
    );
}

// ---------------------------------------------------------------------------
// 1. THE HOLE — A QUANTIFIER OUTSIDE A CONJUNCTIVE POSITION IS A FREE BOOLEAN
//
//    Every test in this section is a HOLE pin: it asserts the wrong answer
//    this tree gives today.  The controls that make each one a statement
//    about the Boolean *context* are in section 2.
// ---------------------------------------------------------------------------

/// The negation of a tautology answers `sat` (`#P2b-54`).
///
/// `(forall ((x U)) (= (f x) (f x)))` is valid in every structure, so its
/// negation is unsatisfiable in every structure — no theory reasoning, no
/// array, no bit-vector, no cardinality argument.  The tree answers `sat`,
/// and so do `c4b04b7` and crates.io 0.3.3.
///
/// This is the whole defect in three lines: the quantifier occurs only under a
/// `not`, so its Tseitin literal is never made to mean anything, and the SAT
/// solver is free to set it false.
///
/// **THE HOLE IS CLOSED** when this answers `unsat` — invert it then.
#[test]
fn a_negated_valid_quantifier_is_a_wrong_sat() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun f (U) U)\n\
         (assert (not (forall ((x U)) (= (f x) (f x)))))\n\
         (check-sat)\n",
        "sat",
        "PIN (#P2b-54): the negation of a tautology is unsatisfiable; this \
         tree answers `sat` because a quantifier in a negative position is an \
         unconstrained Boolean",
    );
}

/// The same hole with `Int` and an uninterpreted function (`#P2b-54`).
///
/// Included so the family is not read as a bit-vector or an array defect: it
/// is the Boolean position, and nothing else.
///
/// **THE HOLE IS CLOSED** when this answers `unsat`.
#[test]
fn a_negated_valid_quantifier_over_int_is_a_wrong_sat() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-fun f (Int) Int)\n\
         (assert (not (forall ((x Int)) (= (f x) (f x)))))\n\
         (check-sat)\n",
        "sat",
        "PIN (#P2b-54): `∀x:Int. f(x) = f(x)` is valid, so its negation has no \
         model",
    );
}

/// A quantifier under an implication whose antecedent is asserted true
/// (`#P2b-54`).
///
/// `p` is asserted, so `(=> p Q)` forces `Q`, and `Q` is [`ROW_FORALL`], which
/// section 0 shows is unsatisfiable.  The identical `Q` asserted at the top
/// level is refuted in 0.7 ms — see
/// [`the_same_quantifier_asserted_at_the_top_level_is_refuted`] — so this test
/// and that one differ in nothing but the Boolean context.
///
/// **THE HOLE IS CLOSED** when this answers `unsat`.
#[test]
fn a_quantifier_under_an_implication_is_a_wrong_sat() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (declare-const p Bool)\n\
             (assert p)\n\
             (assert (=> p {ROW_FORALL}))\n\
             (check-sat)\n"
        ),
        "sat",
        "PIN (#P2b-54): `p` is true, so the implication forces the quantifier, \
         which has no model",
    );
}

/// The wrong `sat` serves a model that contradicts itself inside one response
/// (`#P2b-54`).
///
/// This is what makes the finding a soundness claim rather than an
/// incompleteness one: the solver answers `sat`, publishes `p = true` — so the
/// implication's antecedent holds and the quantifier is asserted — and then
/// answers `#b0000000` for `(select (store a #b0000000 #b0000101) #b0000001)`,
/// which is the quantified body's own instance at `i = #b0000000` evaluating
/// to `false`.  No oracle is involved; the response refutes itself.
///
/// **THE HOLE IS CLOSED** when the verdict becomes `unsat` — invert this then.
#[test]
fn the_wrong_sat_publishes_a_model_that_refutes_its_own_assertion() {
    let script = format!(
        "{PINNED_ARRAY}\
         (set-option :produce-models true)\n\
         (declare-const p Bool)\n\
         (assert p)\n\
         (assert (=> p {ROW_FORALL}))\n\
         (check-sat)\n\
         (get-value ((select (store a (_ bv0 7) (_ bv5 7)) (_ bv1 7)) p))\n"
    );
    let lines = run(&script);
    let text = joined(&lines);
    assert_eq!(
        verdict(&lines),
        "sat",
        "PIN (#P2b-54): the verdict is the wrong `sat`\n{text}"
    );
    assert!(
        text.contains("(p true)"),
        "PIN (#P2b-54): the published model makes the antecedent true, so the \
         quantifier is asserted in it\n{text}"
    );
    assert!(
        text.contains("((select (store a (_ bv0 7) (_ bv5 7)) (_ bv1 7)) #b0000000)"),
        "PIN (#P2b-54): and the same model reads `#b0000000` where the \
         quantifier demands `#b0000101`, so it falsifies the very assertion it \
         was published for\n{text}"
    );
}

/// A quantifier under a disjunction whose other arm is asserted false
/// (`#P2b-54`).
///
/// **THE HOLE IS CLOSED** when this answers `unsat`.
#[test]
fn a_quantifier_under_a_disjunction_is_a_wrong_sat() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (declare-const p Bool)\n\
             (assert (not p))\n\
             (assert (or p {ROW_FORALL}))\n\
             (check-sat)\n"
        ),
        "sat",
        "PIN (#P2b-54): `p` is false, so the disjunction forces the quantifier",
    );
}

/// A quantifier under an `ite`, and under a Boolean `=` (`#P2b-54`).
///
/// Two more spellings in one test so the family cannot be closed one operator
/// at a time and still look finished.
///
/// **THE HOLE IS CLOSED** when either answers `unsat`.
#[test]
fn a_quantifier_under_an_ite_or_a_boolean_equality_is_a_wrong_sat() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (declare-const p Bool)\n\
             (assert p)\n\
             (assert (ite p {ROW_FORALL} false))\n\
             (check-sat)\n"
        ),
        "sat",
        "PIN (#P2b-54): the `ite` spelling",
    );
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (declare-const p Bool)\n\
             (assert p)\n\
             (assert (= p {ROW_FORALL}))\n\
             (check-sat)\n"
        ),
        "sat",
        "PIN (#P2b-54): the Boolean-equality spelling",
    );
}

/// `(not (exists …))` — the same hole written as a negated existential
/// (`#P2b-54`).
///
/// `¬∃i. (select (store a i #b0000101) #b0000001) ≠ #b0000101` is `ROW_FORALL`
/// by De Morgan, so this is unsatisfiable; the NNF spelling
/// [`the_positive_existential_spelling_is_decided_correctly`] is answered
/// correctly, which is what places the defect in the negation and not in the
/// existential.
///
/// **THE HOLE IS CLOSED** when this answers `unsat`.
#[test]
fn a_negated_existential_is_a_wrong_sat() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (assert (not (exists ((i (_ BitVec 7))) \
             (distinct (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7)))))\n\
             (check-sat)\n"
        ),
        "sat",
        "PIN (#P2b-54): `¬∃` is `∀¬`, which has no model here",
    );
}

/// An `Int` index sort in the same Boolean context (`#P2b-54`).
///
/// `encode::finite_expand` can never enumerate `Int`, so this spelling has no
/// width above which it starts: it is the family's permanent member.
///
/// **THE HOLE IS CLOSED** when this answers `unsat`.
#[test]
fn an_int_index_sort_under_an_implication_is_a_wrong_sat() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array Int Int))\n\
         (declare-const p Bool)\n\
         (assert (= a ((as const (Array Int Int)) 0)))\n\
         (assert p)\n\
         (assert (=> p (forall ((i Int)) (= (select (store a i 5) 1) 5))))\n\
         (check-sat)\n",
        "sat",
        "PIN (#P2b-54): the `Int` member of the family",
    );
}

// ---------------------------------------------------------------------------
// 1b. TWO SHAPES THAT EVADE `encode::binder_row`'s OWN GUARDS
//
//     These are conjunctive — the position re-fix pass 6 fixed — so they are
//     not section 1's defect.  They are the two guards `binder_row` declines
//     on, and declining leaves the pass-5 wrong `sat` exactly as it was.
// ---------------------------------------------------------------------------

/// A dead quantifier that merely *binds the name* of a free constant elsewhere
/// in the same assertion disables the fix (`#P2b-55`).
///
/// `binder_row::rewritable_quantifiers` keeps a quantifier only when none of
/// its free variables shares a name with a binder anywhere in the assertion —
/// `finite_expand`'s capture guard, reused.  The guard compares *names*, so
/// `(forall ((a (_ BitVec 7))) (= a a))`, which is valid and says nothing,
/// puts the name `a` in the bound set and the real quantifier — whose only
/// free variable is the array `a` — is declined.  The same two quantifiers as
/// two separate assertions are refuted, which is
/// [`the_same_two_quantifiers_as_two_assertions_are_refuted`].
///
/// `c4b04b7` and 0.3.3 answer `unknown` here, so this spelling is a wrong
/// answer the round introduced where the base had an honest one.
///
/// **THE HOLE IS CLOSED** when this answers `unsat`.
#[test]
fn a_binder_name_collision_inside_one_assertion_is_a_wrong_sat() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (assert (and (forall ((a (_ BitVec 7))) (= a a)) {ROW_FORALL}))\n\
             (check-sat)\n"
        ),
        "sat",
        "PIN (#P2b-55): a valid, contentless quantifier that happens to bind \
         the name `a` disables the read-over-write expansion for the \
         quantifier beside it",
    );
}

/// A quantifier whose body *is* another quantifier is never rewritten
/// (`#P2b-55`).
///
/// `binder_row::read_over_write_map` stops at a deeper binder, so the outer
/// quantifier yields no rewrite; the inner one does, but splicing it back
/// crosses the outer binder and is dropped.  `(forall ((j …)) (forall ((i …))
/// C(i)))` is `(forall ((i …)) C(i))`, so the answer must match
/// [`the_same_quantifier_asserted_at_the_top_level_is_refuted`] — and the
/// two-variable spelling of exactly the same formula,
/// [`one_binder_list_with_two_variables_is_refuted`], is `unsat`.
///
/// `c4b04b7` answers `unknown`.
///
/// **THE HOLE IS CLOSED** when this answers `unsat`.
#[test]
fn a_quantifier_nested_directly_inside_another_is_a_wrong_sat() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (assert (forall ((j (_ BitVec 7))) {ROW_FORALL}))\n\
             (check-sat)\n"
        ),
        "sat",
        "PIN (#P2b-55): an unused outer binder makes the inner quantifier \
         unreachable for the rewrite",
    );
}

// ---------------------------------------------------------------------------
// 2. THE CONTROLS — what makes section 1 a statement about the *context*
//
//    Every test below asserts the CORRECT answer and is an ordinary
//    regression guard.
// ---------------------------------------------------------------------------

/// The same quantifier as the whole assertion is refuted.
///
/// This is re-fix pass 6's fix working, and it is the control that turns every
/// pin in section 1 into a statement about the Boolean position: the formula
/// is character-for-character the same.
#[test]
fn the_same_quantifier_asserted_at_the_top_level_is_refuted() {
    assert_verdict(
        &format!("{PINNED_ARRAY}(assert {ROW_FORALL})\n(check-sat)\n"),
        "unsat",
        "the top-level spelling of the same quantifier must be refuted",
    );
}

/// And under an `and`, which is the other conjunctive position.
#[test]
fn the_conjunctive_spelling_is_refuted() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (declare-const p Bool)\n\
             (assert (and p {ROW_FORALL}))\n\
             (check-sat)\n"
        ),
        "unsat",
        "a conjunct is asserted unconditionally, so it must be refuted",
    );
}

/// The same two quantifiers as two assertions rather than one conjunction.
///
/// The control for [`a_binder_name_collision_inside_one_assertion_is_a_wrong_sat`]:
/// `binder_row`'s guard is scoped to one assertion, so splitting the `and`
/// removes the name collision and the refutation comes back.
#[test]
fn the_same_two_quantifiers_as_two_assertions_are_refuted() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (assert (forall ((a (_ BitVec 7))) (= a a)))\n\
             (assert {ROW_FORALL})\n\
             (check-sat)\n"
        ),
        "unsat",
        "two assertions carry no shared binder-name set, so the expansion \
         applies and the script is refuted",
    );
}

/// `(forall ((j …) (i …)) C(i))` — the same formula with one binder list.
///
/// The control for [`a_quantifier_nested_directly_inside_another_is_a_wrong_sat`].
#[test]
fn one_binder_list_with_two_variables_is_refuted() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (assert (forall ((j (_ BitVec 7)) (i (_ BitVec 7))) \
             (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7))))\n\
             (check-sat)\n"
        ),
        "unsat",
        "one binder list instead of two nested ones must not change the answer",
    );
}

/// Inside `encode::finite_expand`'s 64-point budget the Boolean context does
/// not matter.
///
/// At index width 2 the whole-sort expansion (`#P2b-47`) turns the quantifier
/// into a ground conjunction before any of section 1's machinery is reached,
/// so the `=>` spelling is refuted and the negated tautology is refuted.  That
/// is what bounds the family from below: it starts one index bit above the
/// budget, and it is why every pin in section 1 uses width 7.
#[test]
fn inside_the_finite_expansion_budget_the_boolean_context_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 2) (_ BitVec 2)))\n\
         (declare-const p Bool)\n\
         (assert (= a ((as const (Array (_ BitVec 2) (_ BitVec 2))) (_ bv0 2))))\n\
         (assert p)\n\
         (assert (=> p (forall ((i (_ BitVec 2))) \
         (= (select (store a i (_ bv1 2)) (_ bv1 2)) (_ bv1 2)))))\n\
         (check-sat)\n",
        "unsat",
        "at width 2 the whole-sort expansion decides this before the Boolean \
         context can matter",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 2) (_ BitVec 2)))\n\
         (assert (not (forall ((i (_ BitVec 2))) (= (select a i) (select a i)))))\n\
         (check-sat)\n",
        "unsat",
        "and the negated tautology, at a width the expansion reaches; \
         `c4b04b7` answers a wrong `sat` to this one",
    );
}

/// The NNF spelling of [`a_negated_existential_is_a_wrong_sat`]'s formula.
///
/// `(exists ((x U)) (distinct (f x) (f x)))` is the same formula with the
/// negation pushed in, and it is refuted.  So the defect is the negation, not
/// the existential quantifier.
#[test]
fn the_positive_existential_spelling_is_decided_correctly() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun f (U) U)\n\
         (assert (exists ((x U)) (distinct (f x) (f x))))\n\
         (check-sat)\n",
        "unsat",
        "the NNF spelling of the negated tautology must be refuted",
    );
}

/// A quantifier literal that a positive assertion already constrains is
/// refuted.
///
/// `(assert Q)` and `(assert (not Q))` share one Tseitin literal, so the SAT
/// core alone refutes them.  This is the shape that makes section 1's
/// mechanism precise: the literal is only unconstrained when *nothing* asserts
/// the quantifier positively.
#[test]
fn a_quantifier_literal_shared_with_a_positive_assertion_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun p (U) Bool)\n\
         (assert (forall ((x U)) (p x)))\n\
         (assert (not (forall ((x U)) (p x))))\n\
         (check-sat)\n",
        "unsat",
        "one literal asserted both ways is refuted by the SAT core alone",
    );
}

/// A satisfiable existential over the same body still answers `sat`.
///
/// So "refute everything" does not pass section 2 either: at `i = #b0000001`
/// the `store` hits index 1 and the read is `#b0000101`.
#[test]
fn the_satisfiable_existential_over_the_same_body_is_still_sat() {
    assert_verdict(
        &format!(
            "{PINNED_ARRAY}\
             (assert (exists ((i (_ BitVec 7))) \
             (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7))))\n\
             (check-sat)\n"
        ),
        "sat",
        "the existential has a witness at index 1 and must stay `sat`",
    );
}

// ---------------------------------------------------------------------------
// 3. REGRESSION GUARDS FOR WHAT RE-FIX PASS 6 CLOSED
//
//    Each of these answers `unknown` on `c4b04b7` and on crates.io 0.3.3, and
//    the correct `unsat` here.
// ---------------------------------------------------------------------------

/// A read of a constant array under a binder, above the expansion budget —
/// and the unrelated declaration that decides whether it is refuted
/// (`#P2b-56`).
///
/// `((as const …) #b0000000)` reads `#b0000000` at every index, so `distinct`
/// from `#b0000000` has no model.  There is no `store`, so `binder_row` cannot
/// reach this one: it is the seam (`Solver::prepare_ground_instance`) doing the
/// work, and it is the width-7 spelling of the pass-5 minimal repro.
///
/// The seam refutes it — but only when the script *also* declares a constant
/// of the index sort that occurs in no assertion at all.  Delete
/// `(declare-const d (_ BitVec 7))` and the same formula answers `unknown`.
/// The refutation therefore depends on a ground term of the index sort
/// existing somewhere in the script for the instantiation to seed itself
/// from, which is a property of the *spelling* and not of the formula.
/// `c4b04b7` answers `unknown` to both, so neither half is a lost verdict; the
/// finding is that the closure of the pass-5 family is narrower than the
/// family.
///
/// **THE HOLE IS CLOSED** when the second half below answers `unsat` too —
/// replace this test with a plain guard then.
#[test]
fn a_constant_array_read_under_a_binder_needs_an_unrelated_declaration() {
    const BODY: &str = "(assert (forall ((i (_ BitVec 7))) \
         (distinct (_ bv0 7) \
         (select ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7)) i))))\n\
         (check-sat)\n";
    assert_verdict(
        &format!("(set-logic ALL)\n(declare-const d (_ BitVec 7))\n{BODY}"),
        "unsat",
        "with a spare constant of the index sort the seam refutes it",
    );
    assert_verdict(
        &format!("(set-logic ALL)\n{BODY}"),
        "unknown",
        "PIN (#P2b-56): and without that unused declaration the identical \
         formula is undecided",
    );
}

/// The same with an `Int` index sort, which no expansion can enumerate.
#[test]
fn a_constant_array_read_under_an_int_binder_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (assert (forall ((i Int)) \
         (distinct 0 (select ((as const (Array Int Int)) 0) i))))\n\
         (check-sat)\n",
        "unsat",
        "the `Int` spelling of the same constant-array read",
    );
}

/// An array-sorted `ite` under a binder, both arms pinned to the same constant
/// array.
#[test]
fn an_array_sorted_ite_under_a_binder_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (declare-const b (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (declare-const c Bool)\n\
         (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))\n\
         (assert (= b ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select (ite c a b) i) (_ bv5 7))))\n\
         (check-sat)\n",
        "unsat",
        "both arms are the all-zero array, so the read is `#b0000000` \
         whichever way `c` goes",
    );
}

/// A store chain deeper than `binder_row::MAX_STORE_PEEL_DEPTH`.
///
/// Seventeen stores, one more than the peel cap, so the rewrite truncates and
/// leaves a `select` over the remaining chain.  Truncating is an equivalence,
/// not an approximation, and the script is still refuted.
#[test]
fn a_store_chain_past_the_peel_cap_is_still_refuted() {
    let mut inner = String::from("(store a i (_ bv5 7))");
    for index in 10..26u32 {
        inner = format!("(store {inner} (_ bv{index} 7) (_ bv0 7))");
    }
    let script = format!(
        "{PINNED_ARRAY}\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select {inner} (_ bv1 7)) (_ bv5 7))))\n\
         (check-sat)\n"
    );
    assert_verdict(
        &script,
        "unsat",
        "a chain past the peel cap truncates to an equivalent term, so the \
         verdict must not change",
    );
}

/// `push`/`pop` around the quantified assertion.
///
/// The derived read-over-write lemma is added with the same `add_clause` the
/// assertion uses, so a `pop` must retract both: `unsat` inside the scope and
/// `sat` after it.  A lemma installed at the root scope would leave the second
/// `check-sat` `unsat`, which is a wrong `unsat` and the one way this pass's
/// design could have gone wrong.
#[test]
fn a_pop_retracts_the_derived_lemma_with_its_assertion() {
    let lines = run(&format!(
        "{PINNED_ARRAY}\
         (push 1)\n\
         (assert {ROW_FORALL})\n\
         (check-sat)\n\
         (pop 1)\n\
         (check-sat)\n"
    ));
    assert_eq!(
        verdicts(&lines),
        vec!["unsat".to_string(), "sat".to_string()],
        "the scoped assertion must be refuted and the popped context must be \
         satisfiable again\n{}",
        joined(&lines)
    );
}

// ---------------------------------------------------------------------------
// 4. A NOTE ON WHAT IS NOT PINNED HERE
//
//    * `TODO.md` `#P2b-51` (a published model that falsifies its own
//      quantified assertion) keeps its pin in
//      `round4_pass6_recheck_pins.rs`; this pass re-measured it rather than
//      re-pinning it.
//    * `TODO.md` `#P2b-46 (f)` (the SAT-variable leak across `push`/`pop`) is
//      a cost defect whose cheapest repro costs minutes in the profile
//      `cargo nextest` builds.  Its two release-calibrated cost pins in
//      `round4_pass4_recheck_pins.rs` are the right place for it and are
//      unchanged.
//    * `rf6/cal/st10_w3.smt2` — the calibration ladder's lost verdict — is a
//      release-only measurement for the same reason.
// ---------------------------------------------------------------------------
