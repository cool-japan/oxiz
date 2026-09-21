//! Round 4, **recheck pass 6**, inverted by re-fix pass 7: the array-read
//! family above the finite-expansion budget.
//!
//! # The short version
//!
//! Pass 6 closed R5-1 — "an array read that is first ground inside a
//! quantifier instance is a free value" — for every index sort
//! `encode::finite_expand` can *enumerate*, and the paired corpora it was
//! measured on (`rc5/corpus/{q,qnoite}`, 800 scripts) contain index widths 1
//! and 2 only.  A binder over `(_ BitVec 1)` or `(_ BitVec 2)` is rewritten
//! into its own ground expansion before the encoder runs, so those scripts
//! never reach MBQI at all.  Section 1 of this file is that half's regression
//! suite: capture, nested binders, `let` under a binder, patterns, the budget
//! edge on both sides, a quantifier in a term position, `push`/`pop`.
//!
//! The expansion declines a sort it cannot spell out — `(_ BitVec w)` for
//! `w >= 7` (128 points against a 64-point budget), `Int`, `Real`, and any
//! declared sort.  Recheck pass 6 found R5-1 fully open on that side, with the
//! bit-vector member *worse* than `c4b04b7` (the base answered `unknown`, the
//! tree answered a wrong `sat`), and section 2 pinned six spellings of it.
//!
//! **Re-fix pass 7 closed that side at its root** and section 2 is inverted
//! here.  The root is `solver::encode::binder_row`: the read-over-write axiom
//!
//! ```text
//! (select (store a i v) k)  ≡  (ite (= i k) v (select a k))
//! ```
//!
//! is applied under the binder, so the read is no longer an opaque value of
//! the element sort and the refutation no longer depends on MBQI guessing an
//! index the script never mentions.  The rewritten assertion is added *beside*
//! the original, never in its place — see `Solver::assert_binder_row_lemma`.
//!
//! # The shape
//!
//! ```text
//! (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))
//! (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))
//! (assert (forall ((i (_ BitVec 7)))
//!           (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7))))
//! ```
//!
//! `a` is the all-zero array, pinned by an equality — there is nothing left
//! for a model to choose.  At `i = #b0000000` the `store` misses index
//! `#b0000001`, so the read is `a[1] = #b0000000`, which is not `#b0000101`:
//! the assertion is **unsatisfiable**.  This tree answered `sat` and now
//! answers `unsat`.
//!
//! # What is still pinned open
//!
//! Two holes survive and keep their `THE HOLE IS CLOSED` notes:
//!
//! * `a_declared_index_sort_with_a_free_cardinality_is_undecided` — a
//!   *completeness* hole this pass introduced knowingly.  An uninterpreted
//!   sort may be a singleton, so the script is satisfiable; the earlier `sat`
//!   came from the very defect the rest of this section guards, and showing it
//!   properly needs finite-model finding over an uninterpreted sort.
//!   `TODO.md` `#P2b-50`.
//! * `a_published_model_still_falsifies_its_own_quantified_assertion` — the
//!   *model* half of R5-1, a different mechanism in a different module
//!   (`context::model_fmt::array_model` chooses an array's default from the
//!   sort rather than from the quantified assertions).  `TODO.md` `#P2b-51`.
//! * `two_nested_arrays_held_apart_still_print_identically` — `#P2b-49`,
//!   untouched by this pass.
//!
//! # How to read a pin
//!
//! A test that asserts the **correct** answer is a regression guard: a failure
//! there is a new defect.
//!
//! A test carrying a `THE HOLE IS CLOSED` note asserts the answer this tree
//! actually gives, which is the wrong one; when the defect is fixed the test
//! goes red and the correct response is to invert it — not to weaken it.  That
//! is the only way an open hole can be carried in a green gate without being
//! forgotten.
//!
//! Every script here runs in milliseconds and none of them asserts anything
//! behind a wall clock (decision (16)): no `(set-option :timeout N)` appears
//! anywhere in this file.

use oxiz_solver::Context;

/// Run a script and return the response lines, folding an `Err` into a single
/// `(error …)` line the way the conformance runner does.
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

/// Every `sat`/`unsat`/`unknown` line, in order, for a multi-`check-sat`
/// script.
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

/// Assert that `script` answers exactly `expected`.
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
// 1. REGRESSION GUARDS — the finite-sort expansion is an equivalence.
//
//    `#P2b-47` rewrites `(forall ((i S)) C(i))` into the conjunction over `S`
//    whenever `S` is `Bool` or a bit-vector sort inside the 64-point budget.
//    A rewrite that substitutes into a body is exactly where capture, binder
//    shadowing and `let` handling go wrong, and none of the module's own unit
//    tests exercises any of them.  Every case below was checked against
//    `c4b04b7`: the base answers a wrong `sat` to three of them and `unknown`
//    to six, so these are guards for behaviour this round created.
// ---------------------------------------------------------------------------

/// An inner binder that re-binds the outer binder's **name** keeps its own
/// meaning.
///
/// `∀i. ∃i. i = #b0` is true: the inner `∃` shadows the outer `i` completely,
/// so the body does not mention the outer variable at all and a witness
/// exists.  A substitution that walked into the inner binder would produce
/// `∃i. #b1 = #b0` on the second outer point and answer `unsat`.
///
/// **This one is a smoke test, not a discriminating guard**, and it is labelled
/// so rather than left reading as coverage it does not give: `sat` is also what
/// a solver that declined to expand anything answers, so only the `unsat`
/// direction would be evidence.  The discriminating shadowing guards are
/// [`a_let_that_shadows_the_binder_name_is_not_substituted_into`] and
/// [`a_free_constant_sharing_the_binder_name_keeps_its_own_value`].
#[test]
fn an_inner_binder_that_shadows_the_outer_name_is_not_substituted_into() {
    assert_verdict(
        "(set-logic ALL)\n\
         (assert (forall ((i (_ BitVec 1))) (exists ((i (_ BitVec 1))) (= i #b0))))\n\
         (check-sat)\n",
        "sat",
        "the inner `exists` shadows `i`, so the outer expansion must not reach it",
    );
}

/// A `let` that re-binds the binder's name shadows it in the same way.
///
/// `∀i. (let ((i #b0)) (= i #b1))` is `#b0 = #b1`, which is false at every
/// point, so the script is `unsat` — and it is `unsat` for the *body's* reason,
/// not because the expansion substituted `i` under the `let`.
#[test]
fn a_let_that_shadows_the_binder_name_is_not_substituted_into() {
    assert_verdict(
        "(set-logic ALL)\n\
         (assert (forall ((i (_ BitVec 1))) (let ((i #b0)) (= i #b1))))\n\
         (check-sat)\n",
        "unsat",
        "the `let` binding shadows `i`; the body is `#b0 = #b1`",
    );
}

/// A top-level constant that happens to carry the binder's name is a different
/// symbol and must keep its own value.
///
/// `a[i] = #b1` for the *free* `i`, and `∀i. a[i] = #b0`: the array is
/// everywhere `#b0` and one of its entries is `#b1`, so the script is `unsat`.
/// An expansion that confused the two `i`s would drop the first assertion's
/// constraint and answer `sat`.
#[test]
fn a_free_constant_sharing_the_binder_name_keeps_its_own_value() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const i (_ BitVec 1))\n\
         (assert (= (select a i) #b1))\n\
         (assert (forall ((i (_ BitVec 1))) (= (select a i) #b0)))\n\
         (check-sat)\n",
        "unsat",
        "the free `i` and the bound `i` are different symbols",
    );
}

/// An `∃` under a `∀` that genuinely *reads* the outer bound variable.
///
/// `∀i:(_ BitVec 2). ∃j:(_ BitVec 1). a[j] = i` asks two array entries to
/// cover four values, so it is `unsat` by counting.  The `sat` twin — the same
/// shape with a one-bit element sort, where two entries cover two values — is
/// asserted beside it so that "answer `unsat` to everything" does not pass.
/// `c4b04b7` answers `unknown` to both.
#[test]
fn an_exists_under_a_forall_reading_the_outer_variable_counts_correctly() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 2)))\n\
         (assert (forall ((i (_ BitVec 2))) (exists ((j (_ BitVec 1))) (= (select a j) i))))\n\
         (check-sat)\n",
        "unsat",
        "two entries cannot take four values",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 1))) (exists ((j (_ BitVec 1))) (= (select a j) i))))\n\
         (check-sat)\n",
        "sat",
        "two entries can take two values",
    );
}

/// A `let` **body** under a binder is expanded with the rest of the body.
///
/// `∀i. (let ((x a[i])) x = #b0)` with `a[#b1] = #b1` asserted is `unsat`; the
/// expansion has to substitute the point into the `let`'s *bound term*, which
/// is where a rewrite that stops at a binder node silently loses the
/// constraint.
#[test]
fn a_let_under_a_binder_is_expanded_with_its_bound_term() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 1))) (let ((x (select a i))) (= x #b0))))\n\
         (assert (= (select a #b1) #b1))\n\
         (check-sat)\n",
        "unsat",
        "the `let`-bound term mentions the binder and must be expanded too",
    );
}

/// An array-sorted `ite` read under a binder (`#P2b-41`'s family, under a
/// quantifier).
///
/// Whichever branch the `ite` takes, index `#b0` reads `#b1`, and the
/// quantified assertion says every entry is `#b0`.  `c4b04b7` answers a wrong
/// `sat`.
#[test]
fn an_array_sorted_ite_read_under_a_binder_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const c Bool)\n\
         (assert (forall ((i (_ BitVec 1))) (= (select (ite c a b) i) #b0)))\n\
         (assert (= (select a #b0) #b1))\n\
         (assert (= (select b #b0) #b1))\n\
         (check-sat)\n",
        "unsat",
        "both branches of the array `ite` contradict the quantified assertion",
    );
}

/// A quantifier in a **term** position and a quantifier under a **negation**.
///
/// The expansion emits the whole substituted body, so it is an equivalence and
/// therefore polarity-independent — but nothing else in the suite checks a
/// quantifier that is not a top-level assertion.  `c4b04b7` answers a wrong
/// `sat` to both.
#[test]
fn a_quantifier_off_the_top_level_is_still_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (ite p (forall ((i (_ BitVec 1))) (= (select a i) #b0)) false))\n\
         (assert (= (select a #b1) #b1))\n\
         (check-sat)\n",
        "unsat",
        "`p` must be true, and then the `forall` is contradicted",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (not (exists ((i (_ BitVec 1))) (= (select a i) #b1))))\n\
         (assert (= (select a #b0) #b1))\n\
         (check-sat)\n",
        "unsat",
        "the negated `exists` forbids the entry the ground assertion demands",
    );
}

/// The budget edge, from both sides, on a shape that only the expansion
/// decides.
///
/// A binder over `(_ BitVec 6)` is 64 points — exactly the
/// `finite_expansion_budget` — and is expanded; `(_ BitVec 7)` is 128 and
/// keeps its MBQI path.  Both scripts read the array at a *ground* index, so
/// MBQI has the instance it needs and both are `unsat` here.  The pair exists
/// so that a future change to the budget shows up as a verdict, and so that
/// section 2's `(_ BitVec 7)` pin cannot be dismissed as "width 7 is simply
/// not supported".
#[test]
fn both_sides_of_the_expansion_budget_decide_a_ground_indexed_read() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 6) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 6))) (= (select a i) #b0)))\n\
         (assert (= (select a (_ bv5 6)) #b1))\n\
         (check-sat)\n",
        "unsat",
        "64 points is inside the budget, so the binder is expanded",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b0)))\n\
         (assert (= (select a (_ bv5 7)) #b1))\n\
         (check-sat)\n",
        "unsat",
        "128 points declines the expansion, and MBQI still has the ground index",
    );
}

/// A two-variable binder at exactly the budget, in both directions.
///
/// `(_ BitVec 3) x (_ BitVec 3)` is 64 points.  Injectivity of an
/// eight-element index sort into an eight-element element sort is satisfiable;
/// into a four-element one it is not.  `c4b04b7` answers `unknown` to both, so
/// the whole pair is behaviour this round created.
#[test]
fn a_two_variable_binder_at_the_budget_decides_both_ways() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 3) (_ BitVec 3)))\n\
         (assert (forall ((i (_ BitVec 3)) (j (_ BitVec 3)))\n\
         \x20 (=> (distinct i j) (distinct (select a i) (select a j)))))\n\
         (check-sat)\n",
        "sat",
        "eight indices into eight values can be injective",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 3) (_ BitVec 2)))\n\
         (assert (forall ((i (_ BitVec 3)) (j (_ BitVec 3)))\n\
         \x20 (=> (distinct i j) (distinct (select a i) (select a j)))))\n\
         (check-sat)\n",
        "unsat",
        "eight indices into four values cannot be injective",
    );
}

/// `Bool` is a finite sort too, and a `:pattern` annotation does not change
/// what the binder means.
#[test]
fn a_bool_index_sort_and_a_pattern_annotation_are_both_expanded() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array Bool (_ BitVec 1)))\n\
         (assert (forall ((b Bool)) (= (select a b) #b0)))\n\
         (assert (= (select a true) #b1))\n\
         (check-sat)\n",
        "unsat",
        "`Bool` is a two-element sort and is its own box",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 1)))\n\
         \x20 (! (= (select a i) #b0) :pattern ((select a i)))))\n\
         (assert (= (select a #b1) #b1))\n\
         (check-sat)\n",
        "unsat",
        "a trigger annotation is dropped with the quantifier it annotates",
    );
}

/// A quantified assertion is retracted by `pop`, expansion and all.
///
/// Inside the scope the script is `unsat`; after `pop` the same ground
/// assertion alone is `sat`.  An expansion cached across the scope — or a
/// ground array root that outlived it — would keep the `unsat`.
#[test]
fn a_quantified_assertion_is_retracted_by_pop() {
    let lines = run("(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (push 1)\n\
         (assert (forall ((i (_ BitVec 1))) (= (select a i) #b0)))\n\
         (assert (= (select a #b1) #b1))\n\
         (check-sat)\n\
         (pop 1)\n\
         (assert (= (select a #b1) #b1))\n\
         (check-sat)\n");
    assert_eq!(
        verdicts(&lines),
        vec!["unsat".to_string(), "sat".to_string()],
        "the quantified assertion must not survive its own scope\n{}",
        joined(&lines)
    );
}

/// Nested universals, and an `∃` under a `∀` with a ground contradiction.
#[test]
fn nested_binders_over_arrays_are_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 1)))\n\
         \x20 (forall ((j (_ BitVec 1))) (= (select a i) (select a j)))))\n\
         (assert (distinct (select a #b0) (select a #b1)))\n\
         (check-sat)\n",
        "unsat",
        "the nested universals make `a` constant",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 1)))\n\
         \x20 (exists ((j (_ BitVec 1))) (distinct (select a i) (select a j)))))\n\
         (assert (= (select a #b0) (select a #b1)))\n\
         (check-sat)\n",
        "unsat",
        "a constant array has no differing pair",
    );
}

/// An array of arrays under a binder, and a declared index sort, both keep
/// their ordinary path and are still refuted from a ground index.
///
/// The declared-sort case is the control for
/// [`a_declared_index_sort_is_a_wrong_sat`] below: an uninterpreted sort is
/// not enumerable either way, so a wrong `sat` there is about the *store*, not
/// about the sort.
#[test]
fn an_array_of_arrays_and_a_declared_index_sort_are_refuted_from_a_ground_index() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const n (Array (_ BitVec 1) (Array (_ BitVec 1) (_ BitVec 1))))\n\
         (assert (forall ((i (_ BitVec 1))) (= (select (select n i) #b0) #b1)))\n\
         (assert (= (select (select n #b0) #b0) #b0))\n\
         (check-sat)\n",
        "unsat",
        "arrays of arrays under a binder",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-const a (Array U (_ BitVec 1)))\n\
         (declare-const k U)\n\
         (assert (= (select a k) #b1))\n\
         (assert (forall ((x U)) (= (select a x) #b0)))\n\
         (check-sat)\n",
        "unsat",
        "`k` is a ground term of the declared sort, so MBQI has its instance",
    );
}

/// The bounded-`Int` fragment still works, and an unguarded `Int` binder is
/// still refuted when the falsifying index *is* a ground term of the script.
///
/// The shared odometer that `#P2b-47` refactored `expand_one` onto serves both
/// fragments, so the `Int` half needs a guard of its own.
#[test]
fn the_bounded_int_fragment_is_undisturbed() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array Int Int))\n\
         (assert (forall ((i Int)) (=> (and (>= i 0) (<= i 2)) (= (select a i) 0))))\n\
         (assert (= (select a 1) 7))\n\
         (check-sat)\n",
        "unsat",
        "the guard pins `i` to [0, 2] and the expansion covers index 1",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array Int Int))\n\
         (declare-const k Int)\n\
         (assert (= (select a k) 7))\n\
         (assert (forall ((i Int)) (= (select a i) 0)))\n\
         (check-sat)\n",
        "unsat",
        "`k` is a ground index term, so MBQI instantiates at it",
    );
}

// ---------------------------------------------------------------------------
// 2. REGRESSION GUARDS — the family R5-1 named above the expansion budget.
//
//    Every test here used to be a PIN asserting the wrong answer this tree
//    gave, carrying a `THE HOLE IS CLOSED` note.  Re-fix pass 7 closed the
//    family at its root (`solver::encode::binder_row`: the read-over-write
//    axiom is applied under the binder, so the read is no longer an opaque
//    value of the element sort), and the pins are inverted here rather than
//    relaxed.  Two holes survive and are pinned as such, each with its own
//    note and its own `TODO.md` entry.
// ---------------------------------------------------------------------------

/// **REGRESSION GUARD — the blocker of recheck pass 6, closed.**
///
/// The array is pinned to the all-zero constant array by an equality, so no
/// model has anything left to choose.  At `i = #b0000000` the `store` misses
/// index `#b0000001` and the read is `a[1] = #b0000000`, which is not
/// `#b0000101`, so the script is **unsatisfiable**.
///
/// | build | answer |
/// |---|---|
/// | this tree | `unsat` (0.3 ms) |
/// | this tree before re-fix pass 7 | `sat` — wrong |
/// | `c4b04b7` (the round's base) | `unknown` |
/// | crates.io 0.3.3 | `unknown` |
///
/// 128 points is past `encode::finite_expand`'s 64-point budget, so the
/// quantifier is *not* expanded and this is the MBQI path, not the expansion.
/// What decides it is the read-over-write expansion under the binder: the body
/// becomes `(= (ite (= i #b1) #b5 (select a #b1)) #b5)`, whose only array term
/// is ground, `a`'s definition pins it to `#b0`, and every `i` but `#b1`
/// refutes the body.
///
/// A failure here is a new defect, not a hole to re-pin.
#[test]
fn an_unenumerable_index_sort_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))\n\
         (assert (forall ((i (_ BitVec 7)))\n\
         \x20 (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7))))\n\
         (check-sat)\n",
        "unsat",
        "the read-over-write under the binder is expanded, so `i = #b0` refutes the body",
    );
}

/// **CONTROL** for [`an_unenumerable_index_sort_is_refuted`].
///
/// Byte-for-byte the same script with every `7` replaced by `6`, so the index
/// sort has 64 points and `encode::finite_expand` enumerates it.  It was
/// already `unsat` before the re-fix, which is what made the pin above a
/// statement about the budget boundary rather than about the `store` shape;
/// both sides of the boundary now agree.
#[test]
fn the_same_shape_one_bit_narrower_is_decided_correctly() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 6) (_ BitVec 6)))\n\
         (assert (= a ((as const (Array (_ BitVec 6) (_ BitVec 6))) (_ bv0 6))))\n\
         (assert (forall ((i (_ BitVec 6)))\n\
         \x20 (= (select (store a i (_ bv5 6)) (_ bv1 6)) (_ bv5 6))))\n\
         (check-sat)\n",
        "unsat",
        "64 points is inside the expansion budget, so the script is its own expansion",
    );
}

/// **REGRESSION GUARD.**  The same family one bit wider again.
///
/// `(_ BitVec 8)` is 256 points.  `c4b04b7` answers `sat` here and so did this
/// tree before re-fix pass 7, so this spelling was *pre-existing* rather than a
/// regression; it is kept because it shows the fix does not stop at one width.
#[test]
fn the_family_does_not_stop_at_one_width() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))\n\
         (assert (= (select a (_ bv1 8)) (_ bv0 8)))\n\
         (assert (forall ((i (_ BitVec 8)))\n\
         \x20 (= (select (store a i (_ bv5 8)) (_ bv1 8)) (_ bv5 8))))\n\
         (check-sat)\n",
        "unsat",
        "the width is irrelevant once the read-over-write is expanded under the binder",
    );
}

/// **REGRESSION GUARD — `#P2b-48`, the `Int` member.**
///
/// `(select a 1) = 0` and `∀i. (select (store a i 5) 1) = 5`.  At `i = 2` the
/// `store` misses index 1, so the read is `(select a 1) = 0 ≠ 5` and the script
/// is **unsatisfiable**.  This tree and `c4b04b7` both answered `sat` before
/// re-fix pass 7 — pre-existing, and closed by the same rewrite as the
/// bit-vector members, which is the point: enumerability was what *masked* the
/// defect below width 7, never what caused it.
#[test]
fn an_int_index_sort_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array Int Int))\n\
         (assert (= (select a 1) 0))\n\
         (assert (forall ((i Int)) (= (select (store a i 5) 1) 5)))\n\
         (check-sat)\n",
        "unsat",
        "`#P2b-48`: an unenumerable index sort is refuted by the rewrite, not by enumeration",
    );
}

/// **REGRESSION GUARD.**  `Real` index sort, same family (`i = 2.0` misses
/// index `1.0`).  Both this tree and `c4b04b7` answered `sat` before re-fix
/// pass 7.
#[test]
fn a_real_index_sort_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array Real Real))\n\
         (assert (= (select a 1.0) 0.0))\n\
         (assert (forall ((i Real)) (= (select (store a i 5.0) 1.0) 5.0)))\n\
         (check-sat)\n",
        "unsat",
        "the rewrite is sort-agnostic: `Real` is refuted exactly like `Int`",
    );
}

/// **REGRESSION GUARD — a declared index sort whose cardinality is pinned.**
///
/// `(distinct k k2)` forces `|U| >= 2`, so `i := k2` leaves `a[k] = 0 ≠ 5` and
/// the script is **unsatisfiable**.  `c4b04b7` and this tree before re-fix
/// pass 7 both answered `sat`, which was a genuine wrong `sat`.
///
/// This spelling replaces recheck pass 6's `a_declared_index_sort_is_a_wrong_sat`,
/// whose script left `U`'s cardinality free.  That pin's stated truth was
/// wrong: an uninterpreted sort is only required to be non-empty, so a model
/// with `|U| = 1` satisfies `∀i:U. i = k`, and `sat` was the *correct* answer
/// there.  The singleton spelling is pinned separately just below, as the
/// completeness hole it now is.
#[test]
fn a_declared_index_sort_with_a_pinned_cardinality_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-const k U)\n\
         (declare-const k2 U)\n\
         (assert (distinct k k2))\n\
         (declare-const a (Array U Int))\n\
         (assert (= a ((as const (Array U Int)) 0)))\n\
         (assert (forall ((i U)) (= (select (store a i 5) k) 5)))\n\
         (check-sat)\n",
        "unsat",
        "`i := k2` is a ground index term distinct from `k`, so the body is refuted",
    );
}

/// **PIN — COMPLETENESS, and a regression this pass introduced knowingly.**
///
/// The same script with `U`'s cardinality left free.  An uninterpreted sort is
/// required only to be non-empty, so `|U| = 1` is a model: `i = k` always
/// holds, `(select (store a k 5) k) = 5`, and `a = ((as const …) 0)` is
/// untouched.  **The correct answer is `sat`.**
///
/// | build | answer |
/// |---|---|
/// | this tree | `unknown` |
/// | this tree before re-fix pass 7 | `sat` — right answer, wrong reason |
/// | `c4b04b7`, crates.io 0.3.3 | `sat` |
///
/// The earlier `sat` came from treating the read under the binder as an opaque
/// value of the element sort, i.e. from exactly the defect
/// [`an_unenumerable_index_sort_is_refuted`] pins — the same mechanism that
/// answered `sat` to the `(distinct k k2)` spelling above, where `sat` is
/// wrong.  With the read-over-write expanded the body reduces to `∀i:U. i = k`,
/// which needs finite-model finding over an uninterpreted sort to be *shown*
/// satisfiable, and this solver has none.  So the answer moved from a right
/// answer for a wrong reason to an honest `unknown`.
///
/// **THE HOLE IS CLOSED** when this answers `sat` — invert it then, and never
/// relax it to "not `unknown`".  `TODO.md` `#P2b-50` carries it.
#[test]
fn a_declared_index_sort_with_a_free_cardinality_is_undecided() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-const k U)\n\
         (declare-const a (Array U Int))\n\
         (assert (= a ((as const (Array U Int)) 0)))\n\
         (assert (forall ((i U)) (= (select (store a i 5) k) 5)))\n\
         (check-sat)\n",
        "unknown",
        "PIN: the correct answer is `sat` (`|U| = 1`); see this test's doc and `#P2b-50`",
    );
}

/// **REGRESSION GUARD — a `(check-sat)` answer no longer depends on a command
/// issued after it.**
///
/// The two scripts differ by one trailing line.  Before re-fix pass 7 the first
/// answered `sat` and the second `unsat`: the ground `store` term inside the
/// trailing `(get-value)` entered the term universe before the solve and handed
/// MBQI the instantiation it could not find for itself.  A `(get-value)` cannot
/// change what the assertions mean, so at most one of the two could be right.
///
/// Both now answer `unsat`, which is the right one, and the test asserts the
/// agreement as well as the answers — the agreement is the property that was
/// broken, and it must not be possible to satisfy this test by breaking both
/// spellings in the same direction.
#[test]
fn a_trailing_get_value_does_not_change_the_preceding_check_sat() {
    let head = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (assert (= (select a (_ bv1 7)) (_ bv0 7)))\n\
         (assert (forall ((i (_ BitVec 7)))\n\
         \x20 (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7))))\n\
         (check-sat)\n";
    let without = head.to_string();
    let with = format!("{head}(get-value ((select (store a (_ bv2 7) (_ bv5 7)) (_ bv1 7))))\n");
    let plain = run(&without);
    let extended = run(&with);
    assert_eq!(
        verdict(&plain),
        "unsat",
        "the script without the trailing command\n{}",
        joined(&plain)
    );
    assert_eq!(
        verdict(&extended),
        "unsat",
        "the same script with a trailing `(get-value)`\n{}",
        joined(&extended)
    );
    assert_eq!(
        verdict(&plain),
        verdict(&extended),
        "a later command must not change an earlier `(check-sat)`"
    );
}

/// **REGRESSION GUARD — the lemma reaches a *named* assertion too.**
///
/// `Solver::assert` and `Solver::assert_named` are two separate pre-pass
/// chains and a fix landed in only one of them would leave every script that
/// asks for unsat cores on the old behaviour.  Same script as
/// [`an_unenumerable_index_sort_is_refuted`], written with `:named`
/// assertions; `sat` before re-fix pass 7, `unsat` now.
#[test]
fn the_read_over_write_lemma_reaches_a_named_assertion() {
    assert_verdict(
        "(set-logic ALL)\n\
         (set-option :produce-unsat-cores true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (assert (! (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7)))\n\
         \x20 :named base))\n\
         (assert (! (forall ((i (_ BitVec 7)))\n\
         \x20 (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7))) :named q))\n\
         (check-sat)\n",
        "unsat",
        "the named chain earns the same read-over-write lemma as the anonymous one",
    );
}

/// **PIN — a published model that falsifies its own script.**
///
/// `a1[#b0000011] = #b0`, so `(bvxor (select a1 #b0000011) #b1)` is `#b1` and
/// the binder demands that `a0` be constantly `#b1`.  **The verdict `sat` is
/// correct** — `a0 = ((as const …) #b1)` is a model — but the model this tree
/// publishes is
///
/// ```text
/// a0 = (store (store ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0)
///                    #b1111111 #b1) #b0000011 #b1)
/// ```
///
/// whose value at `#b0000000` is `#b0`.  The model falsifies its own `forall`.
///
/// This is the *model* half of R5-1 and it is a different mechanism from the
/// verdict half the rest of this section guards: the renderer
/// (`context::model_fmt::array_model`) chooses an array's default from the sort
/// rather than from the quantified assertions, and the read-over-write rewrite
/// cannot reach it — the binder here carries no `store` at all.  Measured on
/// the 300 paired width-7/8 scripts of `rk6/corpus/qmbqi`: 26 falsifying models
/// before re-fix pass 7, 24 after.
///
/// **THE HOLE IS CLOSED** when the published `a0` reads `#b1` at
/// `#b0000000` — invert this test then.  `TODO.md` `#P2b-51` carries it.
#[test]
fn a_published_model_still_falsifies_its_own_quantified_assertion() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (= (select a1 (_ bv3 7)) #b0))\n\
         (assert (forall ((i (_ BitVec 7)))\n\
         \x20 (= (select a0 i) (bvxor (select a1 (_ bv3 7)) #b1))))\n\
         (check-sat)\n\
         (get-model)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", joined(&lines));
    let text = joined(&lines);
    let rendered = text
        .lines()
        .find(|line| line.trim_start().starts_with("(define-fun a0 "))
        .unwrap_or_default()
        .to_string();
    assert!(
        rendered.contains("((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0)"),
        "PIN: `a0`'s published default is the sort's `#b0` and the binder demands \
         `#b1`, so the model falsifies its own assertion; when the default \
         becomes `#b1`, invert this test\n{text}"
    );
}

// ---------------------------------------------------------------------------
// 3. A NOTE ON WHAT IS NOT PINNED HERE
//
//    * `TODO.md` `#P2b-46 (f)` — the SAT-variable leak in `oxiz_sat::Solver::
//      pop` — is a cost defect, and every pin for it costs minutes in the
//      profile `cargo nextest` builds.  The two release-calibrated cost pins
//      that exist (`round4_pass4_recheck_pins`) are the right place for it and
//      they are unchanged.
//    * `array_uf_combination::report`'s missing floor on `unknown_decided` was
//      reported from here rather than pinned, because the constant is private
//      to another integration-test crate.  Re-fix pass 7 fixed it in that
//      crate instead: `report` now asserts both a ceiling on `unknown_decided`
//      and a floor on decided verdicts, and at `HARNESS_EMBEDDED_CHECK_BUDGET
//      = 1` all three gates go red where two of three used to pass.
//    * `match` under a binder is unreachable: the parser rejects `match`
//      outright (`unknown function/constant match`), on this tree and on
//      `c4b04b7` alike, so there is no behaviour to pin.
// ---------------------------------------------------------------------------
