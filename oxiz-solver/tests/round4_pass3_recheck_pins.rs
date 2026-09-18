//! Guards from the round-4 adversarial recheck, pass 3 — every pin inverted.
//!
//! This file was written by the pass-3 recheck as a set of **pins**: tests that
//! were green *because the tree was wrong*, each carrying a `THE HOLE IS
//! CLOSED` message so that the pass which fixed the defect would see the pin
//! turn red and know to invert it.  Every one of them has now been inverted:
//! the tests below assert the **correct** behaviour and turn red if the defect
//! returns.  Nothing here is a pin any more, so nothing here says `THE HOLE IS
//! CLOSED`.
//!
//! The defects, and where their fixes live:
//!
//! | was pinned | root fix |
//! |---|---|
//! | a `select` through an array-sorted `ite` is a free bit-vector | `solver/encode/bool_euf_encoding.rs::needs_ite_elimination` no longer excludes `Array` |
//! | an array `ite`'s model falsifies its own script | the same: the `ite` now has a defining variable to publish |
//! | `(as const)` with a *variable* default prints the sort default | `context/model_fmt/array_model.rs` renders the *evaluated* default |
//! | two arrays over an uninterpreted index sort print as one value | `context/model_fmt/array_model.rs::index_position_string` names positions with the `@uc_S_n` witnesses |
//! | a user symbol spelled `@uc_U_0` captures a model witness | `smtlib/parser/commands.rs::reject_reserved_symbol` refuses `@`- and `.`-leading symbols (SMT-LIB 2.6 §3.1) |
//! | two array constants beside two array `ite`s never answer | `solver/array_axioms/families.rs::build_const_array_witness_cell` is built one cell per refinement round |
//! | two wall-clock gates still reach a verdict | `solver/check_core.rs::REFINEMENT_WORK_CEILING_PROPAGATIONS` replaced `int_case_split::REFINEMENT_TIME_CEILING_MS` |
//!
//! The campaign that found them was 2,350 generated scripts scored against a
//! from-scratch total-table oracle plus a model replay that reads a published
//! `(model …)` the way any SMT-LIB consumer reads it.

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

/// The `(model …)` block, or the empty string.
fn model_block(lines: &[String]) -> String {
    lines
        .iter()
        .find(|line| line.starts_with("(model"))
        .cloned()
        .unwrap_or_default()
}

/// The `(get-value …)` response, or the empty string.
fn value_block(lines: &[String]) -> String {
    lines
        .iter()
        .find(|line| line.trim_start().starts_with("(("))
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

const ARR11: &str = "(Array (_ BitVec 1) (_ BitVec 1))";

// ---------------------------------------------------------------------------
// 1. GUARD — a `select` over an array-sorted `ite` is related to both branches.
// ---------------------------------------------------------------------------

/// `(select (ite p a b) i)` is `(select a i)` or `(select b i)`, so pinning
/// *both* branches' reads to `#b0` and demanding `#b1` of the read through the
/// `ite` is unsatisfiable.
///
/// This answered `sat` on this tree, on the 0.3.4 base `c4b04b7` and on
/// crates.io 0.3.3 alike — a wrong `sat` reachable from seven lines.  The root
/// cause was in the encoder rather than in the array theory:
/// `needs_ite_elimination` (`solver/encode/bool_euf_encoding.rs`) excluded
/// `SortKind::Array`, so an array-sorted `ite` was never named by a fresh
/// variable with the two defining implications, and `array_axioms`' walk only
/// noted the branches as *foreign* — no rule ever related the read through the
/// `ite` to the reads of its branches, and the read was a free bit-vector in
/// the circuit.
///
/// Ten of fifteen hand-built shapes in this family were a wrong `sat` on all
/// three trees, and one of 900 random scripts reached it without being aimed at
/// it.  GUARD (inverted pin).
#[test]
fn a_read_through_an_array_ite_is_refuted() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const a0 {ARR11})\n\
         (declare-const a1 {ARR11})\n\
         (declare-const p Bool)\n\
         (assert (= (select (ite p a0 a1) #b0) #b1))\n\
         (assert (= (select a0 #b0) #b0))\n\
         (assert (= (select a1 #b0) #b0))\n\
         (check-sat)\n"
    );
    assert_eq!(
        verdict(&run(&script)),
        "unsat",
        "the read through the array-sorted `ite` must be related to both \
         branches, and both are pinned to #b0"
    );
}

/// The same hole one `store` deeper, and the half the encoder's naming does not
/// reach on its own.
///
/// `needs_ite_elimination` names the array-sorted `ite` with a fresh variable
/// in the *encoded circuit*, but the array theory walks `Solver::assertions`,
/// which holds the term as written.  Read-over-write reduces
/// `select(store(ite(p,a0,a1), #b1, #b0), #b0)` to `select(ite(p,a0,a1), #b0)`
/// — a read of a term this module had no rule for — and the script answered
/// `sat` with the encoder half alone, printing `p = true` beside an `a0` that
/// reads `#b0` where the assertion demands `#b1`.
///
/// `array_axioms::families::build_array_ite_reads` closes it from the theory
/// side: `c => select(t,i) = select(then,i)` and `!c => select(t,i) =
/// select(else,i)`, at every index already read on the `ite` and at every
/// element of an index sort small enough to write out.  GUARD.
#[test]
fn a_read_through_an_array_ite_under_a_store_is_refuted() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const a0 {ARR11})\n\
         (declare-const a1 {ARR11})\n\
         (declare-const p Bool)\n\
         (assert (= (select (store (ite p a0 a1) #b1 #b0) #b0) #b1))\n\
         (assert (= (select a0 #b0) #b0))\n\
         (assert (= (select a1 #b0) #b0))\n\
         (check-sat)\n"
    );
    assert_eq!(
        verdict(&run(&script)),
        "unsat",
        "the `store` writes at #b1 and the read is at #b0, so read-over-write \
         leaves a read of the `ite`, and both of its branches are pinned to #b0"
    );
}

/// The same hole reached the way a generated corpus reaches it: an `ite` whose
/// two branches are the same array, under `distinct`, with the other operand's
/// `ite` collapsing to that array as well once `(= a1 a0)` is asserted.  Both
/// operands denote `a0`, so `distinct` cannot hold.  GUARD (inverted pin).
#[test]
fn an_array_ite_under_distinct_is_refuted() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a0 (Array (_ BitVec 1) (_ BitVec 2)))\n\
         (declare-const a1 (Array (_ BitVec 1) (_ BitVec 2)))\n\
         (declare-const v0 (_ BitVec 2))\n\
         (assert (distinct (ite (= v0 v0) a0 a0) (ite (= v0 #b01) a1 a0)))\n\
         (assert (not (distinct a1 a0)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "unsat",
        "both `ite` operands denote `a0`, so `distinct` cannot hold"
    );
}

/// The model half of the same defect, and the half no read-side lemma could
/// have closed: the solver answers `sat`, publishes a model, and the model must
/// *agree* with its own `(get-value)` response about the read through the
/// `ite`.
///
/// It did not: `p` printed `false`, so the read goes to `a1`, and the printed
/// `a1` answered `#b0` at index `#b0` where the assertion demands `#b1` — with
/// `((select (ite p a0 a1) #b0) #b1)` and `((select a1 #b0) #b0)` side by side
/// in one `(get-value)` response.  Naming the `ite` with a fresh array variable
/// is what gives the model builder something to publish for it.
///
/// GUARD (inverted pin): under the published `p`, the read through the `ite`
/// and the read of the branch it selects must answer the same value, and the
/// model's own binding for that branch must carry it.
#[test]
fn an_array_ite_publishes_a_model_that_agrees_with_its_own_script() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const a0 {ARR11})\n\
         (declare-const a1 {ARR11})\n\
         (declare-const p Bool)\n\
         (assert (= (select (ite p a0 a1) #b0) #b1))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((select (ite p a0 a1) #b0) (select a0 #b0) (select a1 #b0) p))\n"
    );
    let lines = run(&script);
    assert_eq!(verdict(&lines), "sat");
    let values = value_block(&lines);
    let model = model_block(&lines);
    // The assertion demands #b1 of the read through the `ite`; the response
    // must say so.
    assert!(
        values.contains("((select (ite p a0 a1) #b0) #b1)"),
        "the read through the `ite` must answer the value the script asserts. \
         Values were: {values}"
    );
    // …and so must the branch the published `p` selects.  Whichever way `p`
    // goes the guard has something to check, so the defect cannot hide behind
    // the branch it happens to pick: with the read a free bit-vector the two
    // answers came apart, which is precisely the falsifying model.
    let p_is_true = values.contains("(p true)");
    let p_is_false = values.contains("(p false)");
    assert!(
        p_is_true ^ p_is_false,
        "`p` is answered exactly once, either way: {values}"
    );
    let selected = if p_is_true { "a0" } else { "a1" };
    assert!(
        values.contains(&format!("((select {selected} #b0) #b1)")),
        "`p` is {p_is_true}, so the read through the `ite` is `{selected}`'s \
         read and the two must agree.  Values were: {values}; model was: \
         {model}"
    );
    assert!(
        !model_binding(&model, selected).is_empty(),
        "the model must bind `{selected}`: {model}"
    );
}

// ---------------------------------------------------------------------------
// 2. GUARD — an array constant with a *variable* default prints that default.
// ---------------------------------------------------------------------------

/// `(= a ((as const A) v))` with `(= v #b1)` says `a` reads back `#b1`
/// everywhere.  The published model printed `v = #b1` and `a = ((as const A)
/// #b0)` beside it — the sort's default rather than the variable's value — so
/// the model falsified its own first assertion.  Twenty of the campaign's 114
/// falsifying models were this family.
///
/// GUARD (inverted pin): `a` must read back `#b1`, and the model must not name
/// `#b0` as its value anywhere.
#[test]
fn an_array_constant_with_a_variable_default_prints_that_default() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const a {ARR11})\n\
         (declare-const v (_ BitVec 1))\n\
         (assert (= a ((as const {ARR11}) v)))\n\
         (assert (= v #b1))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((select a #b0) (select a #b1) v))\n"
    );
    let lines = run(&script);
    assert_eq!(verdict(&lines), "sat");
    let model = model_block(&lines);
    let values = value_block(&lines);
    assert!(
        model_binding(&model, "v").ends_with("#b1)"),
        "`v` is pinned to #b1: {model}"
    );
    assert!(
        values.contains("((select a #b0) #b1)") && values.contains("((select a #b1) #b1)"),
        "`a` is the constant `#b1` array, so both its reads answer #b1.  \
         Values were: {values}; model was: {model}"
    );
    // The model's own rendering has to say the same thing: nothing in `a`'s
    // binding may carry the sort default, because no index of `a` holds it.
    let binding = model_binding(&model, "a");
    let body = binding
        .rsplit_once(')')
        .map(|(head, _)| head)
        .unwrap_or(&binding);
    assert!(
        !body.contains(" #b0"),
        "`a` reads back #b1 at every index, so its printed value may not carry \
         #b0 at all: {binding}"
    );
}

// ---------------------------------------------------------------------------
// 3. GUARD — two arrays over an uninterpreted index sort print as two values.
// ---------------------------------------------------------------------------

/// `(distinct a b)` over `(Array U (_ BitVec 1))` is satisfiable — `U` is
/// uninterpreted, so it has as many elements as the model needs and the two
/// arrays can differ at one of them.  The verdict was right and the model was
/// not: both arrays printed as the *same* `((as const …) #b0)`, so the
/// published model falsified the only assertion in the script.  All 50 scripts
/// of the campaign's uninterpreted-index-sort family did this, on this tree and
/// on `c4b04b7` alike.
///
/// The printer could not name the position at which they differ: the
/// extensionality witness is an index of sort `U`, which has no literals.  It
/// names them now with the same `@uc_S_n` witnesses every other renderer in the
/// module uses for a class of an uninterpreted sort.
///
/// GUARD (inverted pin).
#[test]
fn two_arrays_over_an_uninterpreted_index_sort_print_as_two_values() {
    let script = "(set-logic QF_AUFBV)\n\
         (declare-sort U 0)\n\
         (declare-const a (Array U (_ BitVec 1)))\n\
         (declare-const b (Array U (_ BitVec 1)))\n\
         (assert (distinct a b))\n\
         (check-sat)\n\
         (get-model)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat");
    let model = model_block(&lines);
    let a = model_binding(&model, "a");
    let b = model_binding(&model, "b");
    assert!(!a.is_empty() && !b.is_empty(), "both arrays bound: {model}");
    assert_ne!(
        a.replace(" a ", " x "),
        b.replace(" b ", " x "),
        "the script asserts the two arrays differ, so the model may not print \
         them as the same value.  Model was: {model}"
    );
    assert!(
        a.contains("@uc_U_") || b.contains("@uc_U_"),
        "the position they differ at is an element of an uninterpreted sort and \
         is named by a witness: {model}"
    );
}

// ---------------------------------------------------------------------------
// 4. GUARD — a user symbol spelled like a model witness is refused.
// ---------------------------------------------------------------------------

/// The round's blocker 1 was closed "as a class" by routing every *minted*
/// symbol through `oxiz_core::smtlib::reserved_name`, whose `\oxiz.` prefix no
/// SMT-LIB 2.6 symbol form can spell.  The model printer's own witnesses were
/// not in that class: `@uc_<Sort>_<n>` is a perfectly spellable simple symbol,
/// and nothing rejected a user declaration of one.
///
/// Five lines were enough: `a` and the user's own `@uc_U_0` asserted distinct,
/// and the model printed `a = @uc_U_0` — which, read as SMT-LIB, is the
/// declared constant, so the model said the two were equal.  `(get-value)`
/// could be made to contradict itself the same way.
///
/// The fix is one level up from the `\oxiz.` prefix and costs a conforming
/// script nothing: SMT-LIB 2.6 §3.1 reserves symbols beginning with `@` and `.`
/// for solver use, and the parser now refuses them.  GUARD (inverted pin): the
/// capture no longer parses, in either reserved leading character.
#[test]
fn a_user_symbol_spelled_like_a_model_witness_is_refused() {
    for symbol in ["@uc_U_0", "@x", ".hidden"] {
        let script = format!(
            "(set-logic QF_UF)\n\
             (declare-sort U 0)\n\
             (declare-const a U)\n\
             (declare-const {symbol} U)\n\
             (assert (distinct a {symbol}))\n\
             (check-sat)\n\
             (get-model)\n"
        );
        let lines = run(&script);
        assert!(
            lines.iter().any(|line| line.starts_with("(error ")),
            "`{symbol}` is reserved to the solver by SMT-LIB 2.6 §3.1 and must \
             not be declarable, got {lines:?}"
        );
    }
    // The quoted spelling is the same symbol and must be refused with it.
    let quoted = "(set-logic QF_UF)\n\
         (declare-sort U 0)\n\
         (declare-const |@uc_U_0| U)\n\
         (check-sat)\n";
    let lines = run(quoted);
    assert!(
        lines.iter().any(|line| line.starts_with("(error ")),
        "the `|…|` spelling of a reserved symbol must be refused too, got \
         {lines:?}"
    );
    // And an ordinary symbol that merely *contains* `@` is still legal: the
    // rule is about the leading character, which is all §3.1 reserves.
    let legal = "(set-logic QF_UF)\n\
         (declare-const a@b Bool)\n\
         (assert a@b)\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(legal)),
        "sat",
        "`a@b` is a legal simple symbol"
    );
}

// ---------------------------------------------------------------------------
// 5. GUARD — two array constants beside two array `ite`s answer, with no clock.
// ---------------------------------------------------------------------------

/// The one *regression* the pass-3 campaign found: nine lines that `c4b04b7`
/// answers `sat` in 0.5 ms and crates.io 0.3.3 in 2.0 ms produced no verdict at
/// all on the tree as it stood — 400 s under `/usr/bin/time` with no answer,
/// and the deterministic refinement budget never fired because the loop accrued
/// no Boolean conflicts to count.  Only an explicit `:timeout` stopped it.
///
/// Two things fixed it, and this guard carries both: the array-constant witness
/// congruence is built one *cell* per refinement round instead of every
/// constant against every pair's witness (`build_const_array_witness_cell`),
/// and the refinement budget counts *lemma instances* as well as conflicts, so
/// a loop that only builds is bounded too
/// (`check_core::ARRAY_REFINEMENT_LEMMA_BUDGET`).
///
/// GUARD (inverted pin).  There is deliberately **no `:timeout`**: the point is
/// that the script answers without one.
#[test]
fn two_array_constants_beside_two_array_ites_answer() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const a0 {ARR11})\n\
         (declare-const a1 {ARR11})\n\
         (declare-const a2 {ARR11})\n\
         (declare-const v0 (_ BitVec 1))\n\
         (declare-const v1 (_ BitVec 1))\n\
         (assert (distinct ((as const {ARR11}) v0) (ite (= v1 #b0) a1 a0)))\n\
         (assert (= ((as const {ARR11}) #b1) (ite (= v1 #b0) a0 a2)))\n\
         (assert (= (select a1 #b0) #b1))\n\
         (check-sat)\n"
    );
    assert_eq!(
        verdict(&run(&script)),
        "sat",
        "this shape answers `sat` on the 0.3.4 base in 0.5 ms; it must answer \
         here without a `:timeout` to stop it"
    );
}

// ---------------------------------------------------------------------------
// 6. GUARD — no verdict is a function of the wall clock (decision (9)/(18)).
// ---------------------------------------------------------------------------

/// Decision (9) required the array refinement's wall-clock budget to be
/// replaced by a deterministic one, keeping an explicit user `:timeout` as the
/// only wall-clock limit.  The named budget was replaced, but the same function
/// still gated two *verdict-bearing* refinements on `check_start.elapsed()`
/// against `int_case_split::REFINEMENT_TIME_CEILING_MS` (120 s), armed whether
/// or not the caller set a `:timeout`:
///
/// * `case_split_affordable`: unaffordable marks the `Sat` unverified
///   (`note_unaffordable_case_split`), so the check answered `unknown` on a
///   slow machine and `sat` on a fast one;
/// * `blocking_affordable`: unaffordable stopped the model-blocking rounds and
///   the exit below it answered `unknown`.
///
/// Both now read `SatStats::propagations` against
/// `check_core::REFINEMENT_WORK_CEILING_PROPAGATIONS`, which the search
/// advances itself.
///
/// The pass-3 pin for this was `include_str!` + `.contains(…)`, which checks
/// two identifiers in one file and would stay green if a wall-clock budget
/// returned somewhere else.  Decision (18) asked for a behavioural guard
/// instead, and this is it: a corpus that reaches every one of those
/// refinements is decided twice, once on an idle process and once with six CPU
/// burners in flight, and the two runs must agree verdict for verdict.
///
/// It is discriminating, not merely necessary: dropping
/// `REFINEMENT_WORK_CEILING_PROPAGATIONS` to 1 in an isolated tree copy flips
/// members of this very corpus from `sat` to `unknown`, so a budget that
/// reaches these verdicts is one this guard sees.
#[test]
fn a_decided_corpus_is_verdict_identical_idle_and_loaded() {
    // Shapes chosen to reach the two repair refinements: a non-convex LIA goal
    // with shared terms (the case split), a UF/array goal whose first candidate
    // model is refuted (model blocking), and the array refinement itself.
    let corpus: Vec<String> = vec![
        "(set-logic QF_LIA)\n\
         (declare-const x Int)\n\
         (declare-const y Int)\n\
         (assert (and (<= 0 x) (<= x 3) (<= 0 y) (<= y 3)))\n\
         (assert (= (+ x y) 5))\n\
         (assert (distinct x y))\n\
         (check-sat)\n"
            .to_string(),
        "(set-logic QF_UFLIA)\n\
         (declare-fun f (Int) Int)\n\
         (declare-const x Int)\n\
         (assert (and (<= 1 x) (<= x 4)))\n\
         (assert (= (f x) 7))\n\
         (assert (distinct (f 1) (f 2)))\n\
         (check-sat)\n"
            .to_string(),
        format!(
            "(set-logic QF_ABV)\n\
             (declare-const arr {ARR11})\n\
             (declare-const brr {ARR11})\n\
             (assert (distinct arr brr))\n\
             (assert (= (select arr #b0) (select brr #b0)))\n\
             (assert (= (select arr #b1) (select brr #b1)))\n\
             (check-sat)\n"
        ),
        format!(
            "(set-logic QF_ABV)\n\
             (declare-const a0 {ARR11})\n\
             (declare-const a1 {ARR11})\n\
             (declare-const p Bool)\n\
             (assert (= (select (ite p a0 a1) #b0) #b1))\n\
             (assert (= (select a0 #b0) #b0))\n\
             (assert (= (select a1 #b0) #b0))\n\
             (check-sat)\n"
        ),
        format!(
            "(set-logic QF_ABV)\n\
             (declare-const arr {ARR11})\n\
             (declare-const i (_ BitVec 1))\n\
             (declare-const v (_ BitVec 1))\n\
             (assert (= (store arr i v) ((as const {ARR11}) #b0)))\n\
             (assert (= v #b1))\n\
             (check-sat)\n"
        ),
        "(set-logic QF_UF)\n\
         (declare-sort U 0)\n\
         (declare-fun g (U) U)\n\
         (declare-const p U)\n\
         (declare-const q U)\n\
         (assert (distinct p q))\n\
         (assert (= (g p) (g q)))\n\
         (check-sat)\n"
            .to_string(),
    ];

    let idle: Vec<String> = corpus.iter().map(|s| verdict(&run(s))).collect();
    assert!(
        idle.iter().all(|v| v == "sat" || v == "unsat"),
        "every member of this corpus is decided on an idle process: {idle:?}"
    );

    // Six CPU burners, the same six-way load the pass-3 recheck used when it
    // measured a wall-clock budget flipping a verdict.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let burners: Vec<std::thread::JoinHandle<u64>> = (0..6)
        .map(|_| {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut acc: u64 = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    for i in 0..100_000u64 {
                        acc = acc.wrapping_add(i).rotate_left(7) ^ i;
                    }
                }
                acc
            })
        })
        .collect();

    let loaded: Vec<String> = corpus.iter().map(|s| verdict(&run(s))).collect();

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for burner in burners {
        // The accumulator is returned only so the loop cannot be optimised
        // away; its value is not used.
        let _ = burner.join();
    }

    assert_eq!(
        idle, loaded,
        "a verdict may not be a function of how busy the machine is: an \
         explicit `:timeout` is the only wall clock this solver reads \
         (decision (9))"
    );
}

/// The other half of decision (18): the two repair gates are denominated in
/// work the search performs, and `(get-info :all-statistics)` publishes that
/// work, so the currency is observable from outside the crate.
///
/// A loaded run and an idle run of the same script must report the *same*
/// propagation count — the counter advances with the search, not with the
/// scheduler — which is what makes "same verdict under load" above a property
/// of the budget rather than a coincidence of timing.
#[test]
fn the_refinement_currency_is_published_and_load_independent() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const arr {ARR11})\n\
         (declare-const brr {ARR11})\n\
         (assert (distinct arr brr))\n\
         (assert (= (select arr #b0) (select brr #b0)))\n\
         (assert (= (select arr #b1) (select brr #b1)))\n\
         (check-sat)\n\
         (get-info :all-statistics)\n"
    );
    let statistics = |lines: &[String]| -> String {
        lines
            .iter()
            .find(|line| line.contains(":propagations"))
            .cloned()
            .unwrap_or_default()
    };
    let first = run(&script);
    let idle = statistics(&first);
    assert!(
        idle.contains(":propagations")
            && idle.contains(":array-refinement-rounds")
            && idle.contains(":array-lemma-instances"),
        "the deterministic currencies the refinement budget is denominated in \
         are published: {idle}"
    );

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let burners: Vec<std::thread::JoinHandle<u64>> = (0..6)
        .map(|_| {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut acc: u64 = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    for i in 0..100_000u64 {
                        acc = acc.wrapping_add(i).rotate_left(7) ^ i;
                    }
                }
                acc
            })
        })
        .collect();
    let loaded = statistics(&run(&script));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for burner in burners {
        let _ = burner.join();
    }

    assert_eq!(
        idle, loaded,
        "every counter the budget reads is a count of work performed, so a \
         loaded run reports exactly what an idle one does"
    );
}

// ---------------------------------------------------------------------------
// 7. GUARD — a user `:max-conflicts` is never extended (decision (20)).
// ---------------------------------------------------------------------------

/// The refinement round installs its own conflict ceiling when the first array
/// lemma is asserted, and it may not hand the search more budget than the
/// caller asked for.
///
/// It used to: the installed value was `min(ceiling, stats().conflicts + N)`
/// with `stats().conflicts` read *at that moment*, while the check's own entry
/// ceiling is `conflicts_so_far + N` with `conflicts_so_far` read at the top of
/// `check`.  Since the first solve may already have spent conflicts by then,
/// the refinement silently re-based the user's budget and granted more search
/// than `:max-conflicts N` allows.  It is now `min(ceiling, conflicts_so_far +
/// N)` with the entry value (decision (20)).
///
/// GUARD: the four-operand `distinct` over store chains costs 46 conflicts with
/// no budget; with `:max-conflicts 20` the whole check — array refinement
/// included — must stay inside that budget and answer `unknown`.  Measured
/// against the weaker behaviour rather than argued: with the refinement
/// granting itself a thousand extra conflicts in an isolated tree copy the same
/// script reports 36, so a budget this refinement extends is one this guard
/// sees.
#[test]
fn a_user_max_conflicts_is_not_extended_by_the_refinement() {
    let script = "(set-logic QF_AUFBV)\n\
         (set-option :max-conflicts 20)\n\
         (declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const brr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const i0 (_ BitVec 1))\n\
         (declare-const i1 (_ BitVec 1))\n\
         (declare-const i2 (_ BitVec 1))\n\
         (declare-const v0 (_ BitVec 1))\n\
         (declare-const v1 (_ BitVec 1))\n\
         (declare-const v2 (_ BitVec 1))\n\
         (assert (distinct \
           (store (store (store (store arr i0 v0) i2 v2) i2 (select arr #b1)) i2 (select arr i0)) \
           (store (store (store brr i2 v0) #b1 (select arr i1)) i0 v1) \
           (store (store (store arr i1 v0) #b0 v2) i0 v0) \
           (store (store (store brr i0 v0) i0 #b0) i1 v1)))\n\
         (check-sat)\n\
         (get-info :all-statistics)\n";
    let lines = run(script);
    let statistics = lines
        .iter()
        .find(|line| line.contains(":conflicts"))
        .cloned()
        .unwrap_or_default();
    let conflicts: u64 = statistics
        .split(":conflicts ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse().ok())
        .unwrap_or(u64::MAX);
    assert!(
        conflicts <= 20,
        "`:max-conflicts 20` is the budget of the whole check; the array \
         refinement may not re-base it (this script costs 46 conflicts \
         unbudgeted).  Statistics were: {statistics}"
    );
    assert_eq!(
        verdict(&lines),
        "unknown",
        "the budget really does bind here — a guard on a script the budget \
         never reaches would pin nothing"
    );
}

// ---------------------------------------------------------------------------
// 8. GUARDS — what this round closed, re-measured from the outside.
// ---------------------------------------------------------------------------

/// Decision (13)'s first fixture shape (`c20`, `upstream: U-Z25`): two arrays
/// over a one-bit index sort whose reads agree at both indices are the same
/// array, so `(distinct arr brr)` is unsatisfiable.  `c4b04b7` and crates.io
/// 0.3.3 both answer `sat`; this tree answers `unsat`.  GUARD.
#[test]
fn array_extensionality_under_distinct_is_refuted() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const arr {ARR11})\n\
         (declare-const brr {ARR11})\n\
         (assert (distinct arr brr))\n\
         (assert (= (select arr #b0) (select brr #b0)))\n\
         (assert (= (select arr #b1) (select brr #b1)))\n\
         (check-sat)\n"
    );
    assert_eq!(verdict(&run(&script)), "unsat");
}

/// Decision (13)'s second fixture shape (`c21`, `upstream: U-Z26`): a store
/// equated with an array constant pins the stored value to the constant's
/// default, so `(= v #b1)` beside a `#b0` constant is unsatisfiable.  `sat` on
/// both older trees, `unsat` here.  GUARD.
#[test]
fn a_store_equal_to_an_array_constant_pins_the_stored_value() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const arr {ARR11})\n\
         (declare-const i (_ BitVec 1))\n\
         (declare-const v (_ BitVec 1))\n\
         (assert (= (store arr i v) ((as const {ARR11}) #b0)))\n\
         (assert (= v #b1))\n\
         (check-sat)\n"
    );
    assert_eq!(verdict(&run(&script)), "unsat");
}

/// Decision (13)'s third fixture shape (`c22`, `upstream: U-Z27`): one variable
/// of indirection between two array constants with different defaults.  The
/// pairs are `{a, c1}` and `{a, c2}`, each instantiated at its *own*
/// extensionality witness, so nothing brought the two defaults into one
/// instance and the formula answered `sat` on this tree, on `c4b04b7` and on
/// crates.io 0.3.3 alike.  `unsat` here.  GUARD.
#[test]
fn an_array_constant_reached_through_a_variable_is_refuted() {
    let script = format!(
        "(set-logic QF_ABV)\n\
         (declare-const a {ARR11})\n\
         (assert (= a ((as const {ARR11}) #b1)))\n\
         (assert (= a ((as const {ARR11}) #b0)))\n\
         (check-sat)\n"
    );
    assert_eq!(verdict(&run(&script)), "unsat");
}

/// The reserved class holds against the two spellings SMT-LIB 2.6 section 3.1
/// admits: a quoted symbol carrying the prefix is refused by the parser, and a
/// bare one cannot be lexed at all.  GUARD for blocker 1's class-based fix.
#[test]
fn the_reserved_prefix_is_unspellable_in_both_symbol_forms() {
    let quoted = "(set-logic QF_UF)\n(declare-const |\\oxiz.numarg!1| Bool)\n(check-sat)\n";
    let bare = "(set-logic QF_UF)\n(declare-const \\oxiz.sk!0 Bool)\n(check-sat)\n";
    for script in [quoted, bare] {
        let lines = run(script);
        assert!(
            lines.iter().any(|line| line.starts_with("(error ")),
            "a script spelling the reserved prefix must be refused, got {lines:?}"
        );
    }
}
