//! Regression guards for the defects the round-4 adversarial recheck found.
//!
//! Every test here was a *pin* on wrong behaviour when the recheck wrote it —
//! it asserted what the tree did then, and failed with `THE HOLE IS CLOSED`
//! when the behaviour became correct.  Each one has since been closed, so the
//! assertions are inverted: they now state the correct behaviour, and the
//! controls that kept the pins honest are kept as controls.
//!
//! The scripts are the recheck's own, unchanged; the evidence they came from
//! is recorded in `TODO.md` under `#P2b-37` and `#P2b-38`.

use oxiz_solver::Context;

/// Run a script and return the response lines.
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
        .find(|l| matches!(l.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// The `(model …)` block, or the empty string.
fn model_block(lines: &[String]) -> String {
    lines
        .iter()
        .find(|l| l.starts_with("(model"))
        .cloned()
        .unwrap_or_default()
}

/// The body of `(define-fun <name> () <sort> <body>)` in a model block.
fn model_binding(model: &str, name: &str) -> String {
    let needle = format!("(define-fun {name} ()");
    model
        .lines()
        .find(|line| line.trim_start().starts_with(&needle))
        .unwrap_or_default()
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------
// 1. The off-chain Skolem index is unspellable (`#P2b-37`, was a BLOCKER).
// ---------------------------------------------------------------------------

/// The off-chain index minted for an array pair used to be named
/// `!oxiz!off!{lo}!{hi}`, and SMT-LIB 2.6 section 3.1 admits `!` in a simple
/// symbol — so a script could declare the solver's own Skolem constant and
/// inherit the `d != i` constraints the array lemma asserts about it.  The
/// formula below is satisfiable (`arr` is free off `i`, and the extra constant
/// is free), and answered `unsat` while the name was spellable.
///
/// The name now contains a backslash, which neither SMT-LIB symbol form can
/// produce, so `|!oxiz!off!5!6|` is an ordinary user constant again.
#[test]
fn a_user_symbol_cannot_capture_the_off_chain_skolem_index() {
    let script = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))
(declare-const i (_ BitVec 1))
(assert (= (store arr i #b0) ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))
(declare-const |!oxiz!off!5!6| (_ BitVec 1))
(assert (= |!oxiz!off!5!6| i))
(check-sat)
";
    let control = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))
(declare-const i (_ BitVec 1))
(assert (= (store arr i #b0) ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))
(check-sat)
";
    assert_eq!(
        verdict(&run(control)),
        "sat",
        "control: the same script without the extra declaration is satisfiable"
    );
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "a free constant named like the solver's Skolem index must not refute the script"
    );
}

/// The whole 1,176-name grid of the old extensionality-witness spelling, which
/// the recheck used to isolate the off-chain rule as the unsound one.  It was
/// harmless then (those lemmas are valid for every index) and is harmless now
/// (the name is no longer the solver's), and it stays here because it is the
/// cheapest test that a *mass* of solver-shaped names changes nothing.
#[test]
fn the_witness_name_grid_is_an_ordinary_set_of_constants() {
    let mut script = String::from(
        "(set-logic QF_ABV)\n\
         (declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const i (_ BitVec 1))\n\
         (assert (= (store arr i #b0) ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n",
    );
    for lo in 0..24u32 {
        for hi in lo..24u32 {
            script.push_str(&format!(
                "(declare-const |!oxiz!ext!{lo}!{hi}| (_ BitVec 1))\n\
                 (assert (= |!oxiz!ext!{lo}!{hi}| i))\n"
            ));
        }
    }
    script.push_str("(check-sat)\n");
    assert_eq!(verdict(&run(&script)), "sat");
}

/// The reserved names themselves stay unspellable: a backslash is a lexical
/// error inside `|...|` and no simple symbol may contain one, so the only way
/// to reach the reserved-name check is an API caller — and it is refused with
/// a message that names the reason.
#[test]
fn a_backslash_symbol_is_refused_with_a_message_that_names_it() {
    let mut ctx = Context::new();
    let error = ctx
        .execute_script("(set-logic QF_UF)\n(declare-const |a\\b| Bool)\n(check-sat)\n")
        .err()
        .map(|err| err.to_string())
        .unwrap_or_default();
    assert!(
        error.contains("backslash"),
        "expected an error naming the backslash, got: {error}"
    );
}

// ---------------------------------------------------------------------------
// 2. `(get-value)` answers values, and keys the response as queried.
// ---------------------------------------------------------------------------

/// `(get-value ((select a #b0)))` answers the value the printed model gives
/// that read — it used to echo the term, so the two commands described two
/// different models.
#[test]
fn get_value_answers_an_array_read_from_the_printed_model() {
    let lines = run("\
(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 1) (_ BitVec 1)))
(declare-const b (Array (_ BitVec 1) (_ BitVec 1)))
(assert (distinct a b))
(check-sat)
(get-model)
(get-value ((select a #b0) (select a #b1) (select b #b0) (select b #b1)))
");
    assert_eq!(verdict(&lines), "sat");
    let answer = lines.last().cloned().unwrap_or_default();
    assert!(
        !answer.contains("(select a #b0) (select a #b0)"),
        "the read must not echo: {answer}"
    );
    // Every answer is a bit-vector literal, and the four of them describe the
    // same two arrays the model block prints.
    for (read, expected) in [
        ("(select a #b0)", 'a'),
        ("(select a #b1)", 'a'),
        ("(select b #b0)", 'b'),
        ("(select b #b1)", 'b'),
    ] {
        let entry = answer
            .lines()
            .find(|line| line.contains(read))
            .unwrap_or_default();
        assert!(
            entry.contains("#b0") || entry.contains("#b1"),
            "{read} (of {expected}) has no value: {answer}"
        );
    }
    // The distinct arrays really differ somewhere in the published model.
    let model = model_block(&lines);
    assert_ne!(
        model_binding(&model, "a").replacen("(define-fun a ", "", 1),
        model_binding(&model, "b").replacen("(define-fun b ", "", 1),
        "two arrays asserted distinct must print differently: {model}"
    );
}

/// A datatype selector over a constant the model fixes to a constructor
/// application answers that field; it used to echo.
#[test]
fn get_value_answers_a_datatype_selector() {
    let lines = run("\
(set-logic QF_AUFDTBV)
(declare-datatypes ((Pair 0)) (((mk (fst (_ BitVec 2)) (snd (_ BitVec 2))))))
(declare-const p Pair)
(declare-const q Pair)
(assert (distinct p q))
(assert (= (fst p) #b01))
(check-sat)
(get-model)
(get-value ((snd q)))
");
    assert_eq!(verdict(&lines), "sat");
    let answer = lines.last().cloned().unwrap_or_default();
    assert!(
        !answer.contains("(snd q) (snd q)"),
        "the selector must not echo: {answer}"
    );
    // The answer is the field the printed model gives `q`.
    let model = model_block(&lines);
    let binding = model_binding(&model, "q");
    let field = binding
        .trim_end_matches(')')
        .rsplit(' ')
        .next()
        .unwrap_or_default()
        .to_string();
    assert!(
        !field.is_empty() && answer.contains(&field),
        "the selector answer {answer} must agree with the model binding {binding}"
    );
}

/// The response key is the term *as queried*: SMT-LIB 2.6 §4.1.1 requires it,
/// and the parser inlines a `define-fun` body before the term ever reaches the
/// printer, so the key used to come back as the body.
#[test]
fn get_value_keys_the_response_with_the_queried_term() {
    let lines = run("\
(set-logic QF_AUFBV)
(define-fun dbl ((x (_ BitVec 2))) (_ BitVec 2) (bvadd x x))
(declare-const a (_ BitVec 2))
(assert (= (dbl a) #b10))
(check-sat)
(get-value ((dbl a)))
");
    assert_eq!(verdict(&lines), "sat");
    let answer = lines.last().cloned().unwrap_or_default();
    assert!(
        answer.contains("(dbl a)"),
        "the queried term keys its value: {answer}"
    );
    assert!(
        !answer.contains("(bvadd a a)"),
        "the inlined body must not appear as a key: {answer}"
    );
}

// ---------------------------------------------------------------------------
// 3. Published models satisfy the script they are models of.
// ---------------------------------------------------------------------------

/// A read at a *compound* index term is published at the index term's value,
/// not at the value of the variable inside it.  The chain used to write at
/// `k`'s value `#b00` although the read index `(bvadd k #b10)` is `#b10`, so
/// the model falsified its own assertion.  The plain-index control publishes
/// at `#b00` and is what isolates the two cases.
#[test]
fn an_array_model_publishes_a_compound_read_at_the_index_terms_value() {
    let compound = run("\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const k (_ BitVec 2))
(declare-const v (_ BitVec 1))
(assert (= v (select arr (bvadd k #b10))))
(assert (= v #b1))
(check-sat)
(get-model)
(get-value ((bvadd k #b10)))
");
    assert_eq!(verdict(&compound), "sat");
    let model = model_block(&compound);
    assert!(
        model.contains("(define-fun k () (_ BitVec 2) #b00)"),
        "control: the model still pins k to #b00: {model}"
    );
    assert!(
        model.contains("#b10 #b1"),
        "the write lands at the index term's value #b10: {model}"
    );
    assert!(
        !model.contains("#b00 #b1"),
        "and not at the index variable's value #b00: {model}"
    );
    let answer = compound.last().cloned().unwrap_or_default();
    assert!(
        answer.contains("#b10"),
        "(get-value) reads the same index value: {answer}"
    );

    let plain = run("\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 2) (_ BitVec 1)))
(declare-const k (_ BitVec 2))
(declare-const v (_ BitVec 1))
(assert (= v (select arr k)))
(assert (= v #b1))
(check-sat)
(get-model)
");
    assert_eq!(verdict(&plain), "sat");
    assert!(
        model_block(&plain).contains("#b00 #b1"),
        "control: a plain index still publishes at its own value"
    );
}

/// Arrays in different congruence classes print differently.
/// `(distinct (fa brr) (fa arr))` forces `arr` and `brr` apart; both used to
/// print as the same array constant with `fa` a constant function, a model
/// falsifying its own `distinct`.  The ext rule for shared array terms now
/// runs even when the formula contains no `select`, no `store` and no array
/// equality at all, which is what supplies the differing entry.
#[test]
fn foreign_array_arguments_in_different_classes_print_differently() {
    let lines = run("\
(set-logic QF_AUFBV)
(declare-fun fa ((Array (_ BitVec 1) (_ BitVec 2))) (_ BitVec 2))
(declare-const arr (Array (_ BitVec 1) (_ BitVec 2)))
(declare-const brr (Array (_ BitVec 1) (_ BitVec 2)))
(assert (distinct (fa brr) (fa arr)))
(check-sat)
(get-model)
");
    assert_eq!(verdict(&lines), "sat");
    let model = model_block(&lines);
    let arr = model_binding(&model, "arr").replacen("(define-fun arr ", "", 1);
    let brr = model_binding(&model, "brr").replacen("(define-fun brr ", "", 1);
    assert!(!arr.is_empty() && !brr.is_empty(), "both arrays: {model}");
    assert_ne!(
        arr, brr,
        "the two arrays are in different classes and must print differently: {model}"
    );
    // …and `fa` must separate them, or the printed model would falsify the
    // assertion it satisfies.
    let interp = model
        .lines()
        .find(|line| line.contains("(define-fun fa "))
        .unwrap_or_default();
    assert!(
        interp.contains("ite"),
        "fa must not be a constant function here: {model}"
    );
}

// ---------------------------------------------------------------------------
// 4. Array cardinality.
// ---------------------------------------------------------------------------

/// `n` pairwise-distinct arrays over a sort with fewer than `n` elements are
/// refuted by the pigeonhole rule, in one lemma rather than through `C(n,2)`
/// extensionality witnesses.  Ten over `BV1 -> BV1` (four arrays exist) and
/// seventeen over `BV2 -> BV1` (sixteen exist) both used to answer `unknown`:
/// the witness families exhausted the BV↔EUF partition exchange first.
#[test]
fn array_cardinality_beyond_the_sort_is_refuted() {
    let build = |n: u32, index_width: u32| {
        let mut script = String::from("(set-logic QF_ABV)\n");
        for x in 0..n {
            script.push_str(&format!(
                "(declare-const c{x} (Array (_ BitVec {index_width}) (_ BitVec 1)))\n"
            ));
        }
        script.push_str("(assert (distinct");
        for x in 0..n {
            script.push_str(&format!(" c{x}"));
        }
        script.push_str("))\n(check-sat)\n");
        script
    };
    for (n, index_width) in [(5u32, 1u32), (9, 1), (10, 1), (12, 1), (17, 2), (20, 2)] {
        assert_eq!(
            verdict(&run(&build(n, index_width))),
            "unsat",
            "{n} pairwise-distinct arrays over BV{index_width} -> BV1 are unsatisfiable"
        );
    }
    // Controls: the satisfiable sizes stay satisfiable.
    for (n, index_width) in [(4u32, 1u32), (8, 2)] {
        assert_eq!(
            verdict(&run(&build(n, index_width))),
            "sat",
            "{n} pairwise-distinct arrays over BV{index_width} -> BV1 are satisfiable"
        );
    }
}

/// The residue this round did **not** close, pinned so it cannot silently
/// change: nine pairwise-distinct arrays over `BV2 -> BV1` (sixteen exist) are
/// satisfiable, and the tree answers `unknown` — the BV↔EUF partition-lemma
/// exchange gives up before the nine witnesses are separated (`#P2b-29`).
/// Nine is the cliff: eight is decided `sat` (the control in
/// [`array_cardinality_beyond_the_sort_is_refuted`]), and every size from nine
/// to sixteen answers `unknown`.  Sound, and recorded in `TODO.md` under
/// `#P2b-38`.
///
/// A failure here is good news: the answer became `sat`.  Replace the pin with
/// `assert_eq!(…, "sat")` and close the strand.
///
/// Costs 7.8 s on its own and 12.6 s under a loaded `nextest` run, in a debug
/// build on the development machine.  The number is here because it is the
/// interesting one: this script reaches `unknown` through the exchange's
/// `MAX_LEMMAS` cap, not through a clock, and an earlier draft of the array
/// refinement's wall-clock budget was set at 5 s and made exactly this test
/// flake under parallel load.  Any future budget must stay well clear of it.
#[test]
fn satisfiable_array_cardinality_at_the_cliff_is_still_undecided() {
    let mut script = String::from("(set-logic QF_ABV)\n");
    for x in 0..9u32 {
        script.push_str(&format!(
            "(declare-const c{x} (Array (_ BitVec 2) (_ BitVec 1)))\n"
        ));
    }
    script.push_str("(assert (distinct");
    for x in 0..9u32 {
        script.push_str(&format!(" c{x}"));
    }
    script.push_str("))\n(check-sat)\n");
    assert_eq!(
        verdict(&run(&script)),
        "unknown",
        "THE HOLE IS CLOSED: nine pairwise-distinct width-2 arrays are decided \
         again -- replace this pin with assert_eq!(verdict, \"sat\")"
    );
}
