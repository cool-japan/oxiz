//! Round 4, **recheck pass 8**: what re-fix pass 8 closed, and the one thing it
//! opened.
//!
//! # Why this file exists
//!
//! Re-fix pass 8 closed `#P2b-54` (a quantifier in a non-conjunctive Boolean
//! position was an unconstrained Boolean) and `#P2b-55` (`binder_row`'s two
//! evadable decline guards) at the root, with
//! [`encode::quant_guard`](../src/solver/encode/quant_guard.rs).  That is
//! confirmed here by regression guards rather than taken on trust.
//!
//! It also had a **price**, which re-fix pass 9 closed (`#P2b-57`).  A
//! quantifier at *positive* polarity under a non-conjunctive connective earns
//! the universal obligation `∀x. (g → φ)`, and that obligation used to be
//! handed to MBQI as an ordinary universal to verify — even when the search
//! had set `g` to `false`, which satisfies it outright at every point of the
//! domain.  Scripts that are *satisfiable with the quantifier playing no part
//! at all* answered `unknown`:
//!
//! ```text
//! (declare-fun r ((_ BitVec 7)) Bool)
//! (declare-const p Bool)
//! (assert p)
//! (assert (or p (forall ((x (_ BitVec 7))) (r x))))
//! (check-sat)      ; c4b04b7, 0.3.3, and now this tree: `sat`.
//! ```
//!
//! Two root fixes closed it, and section 1 is the regression guard for both:
//!
//! * `MBQIIntegration::quantifier_vacuous_under_model` discharges, for one
//!   round, a quantifier whose body reduces to `true` once the *published*
//!   partial model's ground Boolean variables are substituted.  A guard the
//!   search set `false` therefore costs no verification at all — which is what
//!   decision (29)(ii) meant by "a `g` the search sets FALSE discharges every
//!   instance vacuously".
//! * `MBQIIntegration::generate_blind_instantiations` no longer drops every
//!   lemma that is still `Implies`-headed after simplification.  That filter
//!   was the reason a guard the search set *true* was not verified either: the
//!   blind lemmas are the only thing that seeds the relevant-term set for a
//!   guarded universal, and with no seed `sat_certify` never becomes eligible.
//!   It now skips only a residual guard that still mentions a bound variable,
//!   which is the unsound case its own comment described.
//!
//! Section 2 places the fix precisely — the same script one index bit *below*
//! the expansion budget, the same quantifier at *negative* polarity, and the
//! same universal also asserted unconditionally were `sat` throughout, so
//! section 1 is a statement about the positive obligation and not about
//! quantifiers, widths or sorts.
//!
//! Section 3 is the regression half of re-fix pass 8: the eight `#P2b-54`
//! spellings and both `#P2b-55` shapes answer correctly, a *satisfiable*
//! guarded script publishes a model that does not contradict itself, an unsat
//! core names only the assertion the user wrote, and the guarding pass's own
//! decline paths cost `unknown` rather than a verdict.
//!
//! Every script here is fixed, carries no `(set-option :timeout N)` and no wall
//! clock, and asserts a verdict rather than a duration.

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

// ---------------------------------------------------------------------------
// SECTION 1 — REGRESSION GUARDS for `#P2b-57`: satisfiable scripts whose
// quantifier is irrelevant to the verdict are decided, above the expansion
// budget and over sorts that have no budget at all.
// ---------------------------------------------------------------------------

/// A quantifier at positive polarity under `or`, above the finite-expansion
/// budget, does not cost the verdict of a trivially satisfiable script.
///
/// `(assert p)` already satisfies `(or p Q)` whatever `Q` is, so no sound
/// solver needs to look at the quantifier.  `c4b04b7` and crates.io 0.3.3
/// answer `sat`; re-fix pass 8 answered `unknown`, because `quant_guard`
/// replaces `Q` by a guard constant `g` and asserts the universal obligation
/// `∀x. (g → (r x))` — which was handed to MBQI to verify even on the
/// candidate models that set `g` to `false`, and at index width 7 (128 points
/// against the 64-point `DEFAULT_FINITE_EXPANSION_BUDGET`) it could not be.
///
/// `MBQIIntegration::quantifier_vacuous_under_model` is the fix: a `g` the
/// search sets `false` reduces `∀x. (g → φ)` to `true` uniformly in `x`, so
/// the obligation is discharged with no instantiation at all.  That is
/// decision (29)(ii)'s "a `g` the search sets FALSE discharges every instance
/// vacuously", expressed on the obligation rather than on the instances.
///
/// An `unsat` here would be a soundness defect — the script is satisfiable.
#[test]
fn a_satisfiable_script_with_an_irrelevant_positive_quantifier_is_decided() {
    let script = "(set-logic ALL)\n\
         (declare-fun r ((_ BitVec 7)) Bool)\n\
         (declare-const p Bool)\n\
         (assert p)\n\
         (assert (or p (forall ((x (_ BitVec 7))) (r x))))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "`(assert p)` alone satisfies `(or p Q)`, so this script is \
         satisfiable with the quantifier playing no part.  c4b04b7 and 0.3.3 \
         answer `sat`; an `unknown` here is `#P2b-57` reopened, and an \
         `unsat` would be a soundness defect.\n{script}"
    );
}

/// The same shape over an uninterpreted sort, where there is no budget to
/// widen at all.
///
/// The bit-vector spelling above is decided only *above* a budget, so a `sat`
/// there could be read as the expansion doing the work.  This one cannot: an
/// uninterpreted sort has no cardinality the whole-sort expansion could ever
/// enumerate, so the verdict rests entirely on the vacuity discharge.
#[test]
fn the_same_shape_over_an_uninterpreted_sort_is_decided_without_a_budget() {
    let script = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun r (U) Bool)\n\
         (declare-const p Bool)\n\
         (assert p)\n\
         (assert (or p (forall ((x U)) (r x))))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "uninterpreted-sort spelling of the same irrelevant-quantifier \
         script; no widening could ever buy this one back, so an `unknown` \
         here is the vacuity discharge gone.\n{script}"
    );
}

/// An `exists` in the antecedent of `=>` does not cost the verdict either,
/// when the whole implication is discharged by its consequent.
///
/// `(assert p)` makes `(=> Q p)` true for every `Q`.  An `exists` at negative
/// polarity earns the universal obligation `∀x. (g ∨ ¬φ)`; with `φ` a
/// tautology that reduces to `g`, and the vacuity discharge reads `g` off the
/// candidate model instead of asking MBQI to enumerate `U`.
#[test]
fn an_exists_in_an_implication_antecedent_is_decided() {
    let script = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun f (U) U)\n\
         (declare-const p Bool)\n\
         (assert p)\n\
         (assert (=> (exists ((x U)) (= (f x) (f x))) p))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "the consequent `p` is asserted, so the implication holds whatever \
         the existential is worth.  c4b04b7 and 0.3.3 answer `sat`.\n{script}"
    );
}

/// A guarded universal that is *true* in the intended model is confirmed, so
/// the family is not confined to quantifiers the verdict can ignore.
///
/// `a` can be the constant-5 array and `p` true; the universal then holds
/// outright, and `p` being asserted forces the guard `g` **true**, so the
/// vacuity discharge cannot fire.  This one is the other root fix: the blind
/// instantiations that seed `sat_certify`'s relevant-term set were all dropped
/// by the `Implies`-headed lemma filter, so the obligation was never
/// certifiable at any width.
///
/// The verdict is what is pinned here.  The model this now publishes is
/// `((as const …) #b0000000)`, which falsifies the assertion — that is
/// `#P2b-51`, measured to be *identical* on the unconditional spelling
/// `(assert (forall ((i …)) (= (select a i) #b0000101)))`, which answers `sat`
/// with the same wrong model on `c4b04b7`, on `00add07` and here.  So it is a
/// pre-existing model-completion defect this verdict now meets, not one this
/// shape introduces; see `round4_pass6_recheck_pins`.
#[test]
fn a_guarded_universal_that_holds_is_confirmed() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (declare-const p Bool)\n\
         (assert (=> p (forall ((i (_ BitVec 7))) (= (select a i) (_ bv5 7)))))\n\
         (assert p)\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "the constant-5 array satisfies this outright, and the guard is \
         forced true so no vacuity argument applies.\n{script}"
    );
}

// ---------------------------------------------------------------------------
// SECTION 2 — the controls that make section 1 a statement about the POSITIVE
// obligation rather than about quantifiers, widths or sorts.
// ---------------------------------------------------------------------------

/// **CONTROL.**  One index bit *below* the budget the same script is `sat`.
///
/// Width 2 is 4 index points, inside `DEFAULT_FINITE_EXPANSION_BUDGET`, so
/// `finite_expand` replaces the quantifier by its whole-sort conjunction
/// before `quant_guard` ever sees it and there is no obligation to verify.
/// This is what bounds section 1's family from below.
#[test]
fn below_the_expansion_budget_the_same_script_is_decided() {
    let script = "(set-logic ALL)\n\
         (declare-fun r ((_ BitVec 2)) Bool)\n\
         (declare-const p Bool)\n\
         (assert p)\n\
         (assert (or p (forall ((x (_ BitVec 2))) (r x))))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "inside the finite-expansion budget the quantifier never reaches \
         `quant_guard`, so the loss pinned above must not appear here.\n{script}"
    );
}

/// **CONTROL.**  A `forall` at *negative* polarity, over the same
/// uninterpreted sort, is decided.
///
/// A negative universal is an existential obligation, discharged by
/// Skolemisation into a **ground** formula the ordinary solver decides.  Only
/// the universal half needs verifying, which is why section 1's loss is about
/// polarity and not about quantifiers.
#[test]
fn a_negative_universal_over_the_same_sort_is_decided() {
    let script = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun f (U) U)\n\
         (declare-const p Bool)\n\
         (assert p)\n\
         (assert (=> (forall ((x U)) (= (f x) (f x))) p))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "a `forall` at negative polarity is Skolemised into a ground \
         obligation, so it costs no verification and no verdict.\n{script}"
    );
}

/// **CONTROL.**  The unconditional spelling of the same universal is `sat`.
///
/// Asserted on its own spine the quantifier is an ordinary MBQI registration
/// and the search confirms it.  So section 1 is not "this solver cannot answer
/// `sat` to a `forall` over an uninterpreted sort".
#[test]
fn the_unconditional_spelling_of_the_same_universal_is_decided() {
    let script = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun r (U) Bool)\n\
         (declare-const p Bool)\n\
         (assert (=> p (forall ((x U)) (r x))))\n\
         (assert p)\n\
         (assert (forall ((y U)) (r y)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "the same universal, also asserted unconditionally, is confirmed — so \
         the loss above is the CONDITIONAL placement and nothing else.  \
         (c4b04b7 answers `unknown` to this one.)\n{script}"
    );
}

// ---------------------------------------------------------------------------
// SECTION 3 — REGRESSION GUARDS for what re-fix pass 8 closed.
// ---------------------------------------------------------------------------

/// **REGRESSION.**  `#P2b-54`'s minimal repro, in four lines of pure UF, is
/// refuted.
///
/// `∀x:U. f(x) = f(x)` is a tautology in every structure, so its negation is
/// unsatisfiable in every structure.  c4b04b7 and crates.io 0.3.3 answer
/// `sat`.
#[test]
fn the_negation_of_a_valid_universal_over_uf_is_refuted() {
    let script = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun f (U) U)\n\
         (assert (not (forall ((x U)) (= (f x) (f x)))))\n\
         (check-sat)\n";
    assert_eq!(verdict(&run(script)), "unsat", "#P2b-54\n{script}");
}

/// **REGRESSION.**  A quantifier under `xor` and one under `distinct` — two
/// polarity boundaries `quant_guard` answers with `Both` — are refuted.
///
/// Neither spelling appears in the pass-7 battery, and both are positions the
/// `Pol` lattice has to get right in *both* directions at once: the guard
/// constant carries the full definition `g ↔ Q` there or the answer is wrong.
#[test]
fn a_quantifier_under_xor_or_distinct_is_refuted() {
    let xor = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun r (U) Bool)\n\
         (declare-const q Bool)\n\
         (assert (forall ((x U)) (r x)))\n\
         (assert (xor q (forall ((y U)) (r y))))\n\
         (assert q)\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(xor)),
        "unsat",
        "`q` and `∀y. r y` are both true, so their `xor` is false\n{xor}"
    );
    let distinct = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun r (U) Bool)\n\
         (declare-const q Bool)\n\
         (assert q)\n\
         (assert (distinct q (forall ((y U)) (or (r y) (not (r y))))))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(distinct)),
        "unsat",
        "a Boolean `distinct` between `true` and a valid universal\n{distinct}"
    );
}

/// **REGRESSION.**  `#P2b-55` (a) and (b): a binder-name collision inside one
/// assertion, and a `forall` nested directly inside another, are refuted.
///
/// Both were a wrong `sat` on the pass-6 checkpoint where c4b04b7 answers an
/// honest `unknown`, i.e. soundness regressions against the base.  The array is
/// pinned to the all-zero constant array, so `(select a #b0000001)` is `#b0…0`
/// and the quantified body fails at every index other than `#b0000001`.
#[test]
fn both_binder_row_guard_evasions_are_refuted() {
    let header = "(set-logic AUFBV)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))\n";
    let collision = format!(
        "{header}(assert (and (forall ((a (_ BitVec 7))) (= a a))\n\
         (forall ((i (_ BitVec 7))) (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7)))))\n\
         (check-sat)\n"
    );
    assert_eq!(
        verdict(&run(&collision)),
        "unsat",
        "#P2b-55 (a): a contentless conjunct sharing the array's NAME must not \
         decline the real quantifier\n{collision}"
    );
    let nested = format!(
        "{header}(assert (forall ((j (_ BitVec 7))) (forall ((i (_ BitVec 7))) \
         (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7)))))\n\
         (check-sat)\n"
    );
    assert_eq!(
        verdict(&run(&nested)),
        "unsat",
        "#P2b-55 (b): a `forall` nested directly inside another\n{nested}"
    );
}

/// **REGRESSION.**  A *satisfiable* script whose quantifier was guarded
/// publishes a model that does not contradict its own assertions.
///
/// `#P2b-54`'s self-contradicting `(get-value)` is only vacuously fixed on the
/// scripts that became `unsat` — those publish no model at all.  This is the
/// other half: a script that still answers `sat` with a guard constant in it,
/// where the model IS published and can be read back.  `p` must be `false`,
/// because `(select a #b0000000)` is asserted different from `#b0000101` and
/// `p` would force it equal.
///
/// It also pins that the guard and Skolem symbols stay out of the answer: a
/// `(get-value)` response naming `g`, `sk` or `qwit` would be a solver-minted
/// symbol leaking into a user-visible one.
#[test]
fn a_satisfiable_guarded_script_publishes_a_self_consistent_model() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (declare-const p Bool)\n\
         (assert (=> p (forall ((i (_ BitVec 7))) (= (select a i) (_ bv5 7)))))\n\
         (assert (not (= (select a (_ bv0 7)) (_ bv5 7))))\n\
         (check-sat)\n\
         (get-value ((select a (_ bv0 7)) p))\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let values = lines
        .iter()
        .find(|line| line.contains("(select a"))
        .cloned()
        .unwrap_or_default();
    assert!(
        values.contains("(p false)"),
        "`p` true would force `(select a #b0000000)` to `#b0000101`, which the \
         second assertion forbids: the published value of `p` must be `false`. \
         Got: {values}"
    );
    assert!(
        !values.contains("oxiz."),
        "no solver-minted reserved symbol may appear in a `(get-value)` \
         response.  Got: {values}"
    );
}

/// **REGRESSION.**  An unsat core of a refutation that runs through a
/// `quant_guard` obligation names only the assertion the user wrote.
///
/// The obligations are derived terms asserted beside the assertion and are
/// deliberately not pushed onto `Solver::assertions`.  If one ever were, it
/// would show up here as a second core element the user never wrote.
#[test]
fn the_unsat_core_of_a_guarded_refutation_names_only_the_user_assertion() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-unsat-cores true)\n\
         (declare-sort U 0)\n\
         (declare-fun f (U) U)\n\
         (assert (! (not (forall ((x U)) (= (f x) (f x)))) :named A1))\n\
         (check-sat)\n\
         (get-unsat-core)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "unsat", "{}", lines.join("\n"));
    let core = lines
        .iter()
        .find(|line| line.starts_with('(') && line.contains("A1"))
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        core.trim(),
        "(A1)",
        "the core must be exactly the named assertion; a derived obligation \
         appearing here would mean `assert_quantifier_obligation` pushed onto \
         `Solver::assertions`.  Got: {core}"
    );
}

/// **REGRESSION.**  `(get-assertions)` reports the script's own assertions and
/// nothing the guarding pass minted.
#[test]
fn get_assertions_does_not_show_the_derived_obligations() {
    let script = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun r (U) Bool)\n\
         (assert (not (forall ((x U)) (or (r x) (not (r x))))))\n\
         (check-sat)\n\
         (get-assertions)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "unsat", "{}", lines.join("\n"));
    let printed = lines
        .iter()
        .filter(|line| line.contains("forall") || line.contains("oxiz."))
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !printed.contains("oxiz."),
        "`(get-assertions)` must show only what the user asserted.  Got:\n{printed}"
    );
}

/// **REGRESSION (the honesty net).**  `quant_guard`'s own decline path costs
/// `unknown`, never a verdict — measured on the path that is easiest to reach
/// from a script, `MAX_GUARDED_QUANTIFIERS`.
///
/// Sixty-five distinct quantifiers in one assertion is one past the cap, so
/// `guard_conditional_quantifiers` returns `None` and every one of the
/// sixty-five reaches the encoder with a free Boolean literal.  The assertion
/// is unsatisfiable (each disjunct is the negation of a valid universal), and
/// the answer must be an honest `unknown` rather than the `sat` the free
/// literals would allow.
///
/// The companion below shows the same assertion IS refuted once the same
/// quantifiers are also asserted positively, which pins that the `unknown` is
/// the cap and not an inability to reason about sixty-five quantifiers.
#[test]
fn one_quantifier_past_the_guard_cap_costs_unknown_and_never_a_verdict() {
    let mut script =
        String::from("(set-logic ALL)\n(declare-sort U 0)\n(declare-fun p (U) Bool)\n(assert (or");
    for index in 0..65 {
        script.push_str(&format!(" (not (forall ((x{index} U)) (p x{index})))"));
    }
    script.push_str("))\n(check-sat)\n");
    assert_eq!(
        verdict(&run(&script)),
        "unknown",
        "sixty-five candidates is one past `MAX_GUARDED_QUANTIFIERS`, so \
         nothing in this assertion is guarded.  A `sat` here would mean the \
         honesty net (`Solver::quantifier_literal_unconstrained`) does not \
         cover the cap, which is `#P2b-54` again.\n{script}"
    );
}

/// **REGRESSION.**  The same over-the-cap assertion, with every one of its
/// quantifiers also asserted positively, is refuted.
///
/// This is the shape that would exploit a *global* `justified_quantifiers`
/// set: the same hash-consed quantifier marked justified by one assertion
/// while another assertion leaves its literal free.  It cannot, because an
/// assertion that justifies a quantifier also forces its literal true, and
/// this test is what says so.
#[test]
fn the_same_assertion_is_refuted_when_its_quantifiers_are_also_asserted() {
    let mut script =
        String::from("(set-logic ALL)\n(declare-sort U 0)\n(declare-fun p (U) Bool)\n");
    for index in 0..65 {
        script.push_str(&format!("(assert (forall ((x{index} U)) (p x{index})))\n"));
    }
    script.push_str("(assert (or");
    for index in 0..65 {
        script.push_str(&format!(" (not (forall ((x{index} U)) (p x{index})))"));
    }
    script.push_str("))\n(check-sat)\n");
    assert_eq!(
        verdict(&run(&script)),
        "unsat",
        "each disjunct negates a universal the script also asserts\n{script}"
    );
}

/// **REGRESSION.**  `push` / `pop` around a negative-polarity quantifier
/// retracts its obligations with its assertion.
///
/// The Skolem witness and the guard constant are minted inside the scope and
/// the `JustifiedQuantifierAdded` trail entry is what retracts the mark.  A
/// leak would show as the third `check-sat` disagreeing with the first.
#[test]
fn push_and_pop_around_a_guarded_quantifier_leaves_no_residue() {
    let script = "(set-logic ALL)\n\
         (declare-sort U 0)\n\
         (declare-fun f (U) U)\n\
         (push 1)\n\
         (assert (not (forall ((x U)) (= (f x) (f x)))))\n\
         (check-sat)\n\
         (pop 1)\n\
         (assert (forall ((x U)) (= (f x) (f x))))\n\
         (check-sat)\n\
         (push 1)\n\
         (assert (not (forall ((x U)) (= (f x) (f x)))))\n\
         (check-sat)\n\
         (pop 1)\n\
         (check-sat)\n";
    let lines = run(script);
    let verdicts: Vec<&str> = lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .map(|line| line.as_str())
        .collect();
    assert_eq!(
        verdicts,
        vec!["unsat", "sat", "unsat", "sat"],
        "the sequence is the whole test: a leaked obligation would make the \
         final `check-sat` disagree with the second.\n{}",
        lines.join("\n")
    );
}
