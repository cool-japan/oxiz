//! Round-4 adversarial recheck pass 10 — what re-fix pass 10 left open, and
//! guards for the array-root re-spelling it landed.
//!
//! # How to read this file
//!
//! * A test whose doc carries **THE HOLE IS CLOSED** asserts what this tree
//!   answers **today**, which is the weaker answer. It is green now and must
//!   go **red** when the defect is fixed; the doc names the assertion to
//!   flip. Nothing here asserts a wrong `sat` or a wrong `unsat` — every hole
//!   pinned below is an *honest* `unknown`, or a counter that still grows,
//!   beside an oracle that establishes the truth **on this tree** so the pin
//!   never rests on an external solver.
//! * Every other test asserts the CORRECT answer and is an ordinary
//!   regression guard for `Solver::array_root_spelling` (`#P2b-59`), which
//!   re-spells an assertion's encoded root so that an array reachable both as
//!   `(ite c a b)` and as the proxy `eliminate_nonbool_ite` mints is one array
//!   term and not two.
//!
//! No test here installs a wall clock. The two budgeted families use the
//! **deterministic** `(set-option :max-bv-embedded-checks N)`, whose answer
//! and whose counters are a property of the tree and not of the machine.
//!
//! Every figure quoted for `c4b04b7` and for crates.io 0.3.3 was measured
//! with the round's release probes; the scripts and the raw output are under
//! `<scratchpad>/oxiz4/rk11/` with a `REBUILD.md`.

use oxiz_solver::Context;

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

fn assert_verdict(script: &str, expected: &str, why: &str) {
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        expected,
        "{why}\n--- script ---\n{script}--- response ---\n{}",
        lines.join("\n")
    );
}

/// `(get-info :all-statistics)`'s value for `key`, or 0 when the key is absent.
fn counter(lines: &[String], key: &str) -> u64 {
    lines
        .iter()
        .find(|line| line.contains(key))
        .and_then(|line| line.split(key).nth(1))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|token| token.parse().ok())
        .unwrap_or_default()
}

/// The two **quantifier-free** assertions of
/// `<scratchpad>/oxiz4/rk6/corpus/qmbqi120/q0033.smt2`, verbatim — the
/// minimal `#P2b-59` repro `<scratchpad>/oxiz4/rk9/min/q33_a01.smt2`. Two
/// width-8 arrays, `ite`-selected bases, one `store` each, one `(as const …)`.
/// `c4b04b7` answers `sat` in 0.1–6.4 ms and crates.io 0.3.3 in 6.8 ms.
const Q33_DECLS: &str = "(declare-const a0 (Array (_ BitVec 8) (_ BitVec 1)))\n\
     (declare-const a1 (Array (_ BitVec 8) (_ BitVec 1)))\n\
     (declare-const p Bool)\n\
     (declare-const q Bool)\n\
     (declare-const d (_ BitVec 1))\n";

const Q33_ASSERT_0: &str = "(assert (or (= (select (store (ite p a0 a1) #b10110111 #b1) #b11010110) \
     (select (ite p (store a1 #b10101011 #b1) (ite p a1 a1)) #b11000000)) \
     (= (bvxor (select (ite q (store a0 #b01110110 #b1) (store a0 #b11101110 #b0)) #b10111010) #b0) \
     (select ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0) #b00010001))))\n";

const Q33_ASSERT_1: &str = "(assert (= (select (ite q a1 ((as const (Array (_ BitVec 8) (_ BitVec 1))) d)) \
     #b00010111) (bvxor (select (ite q (ite p a1 a1) \
     ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0)) #b11110111) #b1)))\n";

/// Run `body` under a deterministic embedded-check budget and return
/// `(verdict, array-refinement rounds, array-lemma instances)`.
fn at_budget(body: &str, budget: u32) -> (String, u64, u64) {
    let script = format!(
        "(set-logic ALL)\n(set-option :max-bv-embedded-checks {budget})\n{body}(check-sat)\n\
         (get-info :all-statistics)\n"
    );
    let lines = run(&script);
    (
        verdict(&lines),
        counter(&lines, ":array-refinement-rounds "),
        counter(&lines, ":array-lemma-instances "),
    )
}

// ---------------------------------------------------------------------------
// 1. OPEN HOLES
// ---------------------------------------------------------------------------

/// **THE HOLE IS CLOSED** when the two round counts below are equal.
///
/// `#P2b-59`'s close-out, `TODO.md`'s decision (24a) mechanism (iv), the
/// `[0.3.4]` CHANGELOG bullet and the sibling test
/// `round4_pass9_recheck_pins::the_refinement_round_count_is_bounded_and_equal_inside_the_plateau`
/// (which carried the name `…_reaches_a_refinement_fixpoint` when this pin was
/// written) all stated that the lazy array refinement **reaches a fixpoint** on
/// this script, on the evidence of `:array-refinement-rounds` being 10 at
/// `:max-bv-embedded-checks 2000` **and** 10 at `5000`.
///
/// That equality is a **plateau inside the ramp**, not a fixpoint. The same
/// method one rung higher (release, `<scratchpad>/oxiz4/rk11/min/q33_b*.smt2`,
/// the counters byte-identical to the ones asserted here):
///
/// | budget | verdict | rounds / instances |
/// |---|---|---|
/// | 500 | `unknown` | 9 / 75 |
/// | 2,000 | `unknown` | 10 / 76 |
/// | 5,000 | `unknown` | 10 / 76 |
/// | **8,000** | `unknown` | **41 / 129** |
/// | 10,000 | `unknown` | 51 / 181 |
/// | 20,000 | `unknown` | 59 / 199 |
/// | 50,000 | **`sat`** | 66 / 208 |
///
/// So the refinement still consumes whatever budget it is given, up to the
/// 66 rounds and 30,402 embedded checks the unbudgeted run needs — it is the
/// *saturation* at 66 that makes the script decidable again, not a fixpoint at
/// 10. What re-fix pass 10 genuinely bought is the **verdict**: `sat` where
/// the pass-9 tree answered nothing in 400 s (measured, not quoted). That
/// half is guarded by `round4_pass9_recheck_pins::the_corpus_member_the_root_spelling_buys_back_is_decided`
/// and is not restated here.
///
/// 8,000 rather than 20,000 or 50,000 is the cheapest rung that breaks the
/// plateau (518 ms release against 4.6 s and 30.4 s), so the pin costs the
/// gate a second rather than a minute.
///
/// To close: make the two counts agree — then flip `assert!(rounds_high >
/// rounds_low)` to `assert_eq!`, and correct the three documents above.
#[test]
fn the_array_refinement_round_count_still_grows_with_the_budget() {
    let body = format!("{Q33_DECLS}{Q33_ASSERT_0}{Q33_ASSERT_1}");
    assert!(
        !body.contains("forall") && !body.contains("exists"),
        "this pin's whole point is that the script carries no binder"
    );
    let (v_low, rounds_low, inst_low) = at_budget(&body, 5_000);
    let (v_high, rounds_high, inst_high) = at_budget(&body, 8_000);
    assert_eq!(
        (v_low.as_str(), v_high.as_str()),
        ("unknown", "unknown"),
        "both budgets are deliberately below what the script consumes \
         (30,402 embedded checks); this pin is about the counters, not the \
         verdict",
    );
    assert!(
        rounds_low > 0 && rounds_low <= 12 && inst_low <= 90,
        "HOLE: the plateau the `fixpoint` claim rests on is 10 rounds / 76 \
         instances at 5,000 checks; this tree reports {rounds_low} / \
         {inst_low}. If the plateau moved, re-measure the whole ladder before \
         trusting either document.",
    );
    assert!(
        rounds_high > rounds_low && inst_high > inst_low,
        "THE HOLE IS CLOSED: the refinement reported {rounds_high} rounds / \
         {inst_high} instances at 8,000 embedded checks against \
         {rounds_low} / {inst_low} at 5,000, so it no longer consumes whatever \
         budget it is given, and a real fixpoint claim becomes assertable. \
         Invert this assertion, and re-state TODO.md (24a)(iv), `#P2b-59` \
         (e)/(g) and the CHANGELOG `[0.3.4]` Fixed bullet, which re-fix pass 11 \
         corrected to the measured ladder rather than to a fixpoint at 10.",
    );
}

/// **THE HOLE IS CLOSED** when this answers `sat`.
///
/// `#P2b-59`'s repro with its **two assertions in the other order** and
/// nothing else changed — as a set of lines the file is byte-identical to
/// `<scratchpad>/oxiz4/rk9/min/q33_a01.smt2`, so it is the same formula.
///
/// `c4b04b7` answers `sat` in 0.1 ms and crates.io 0.3.3 in 0.1 ms, exactly as
/// they do for the original order. Re-fix pass 10's fix decides the original
/// order (`sat`, 30.4 s release, 66 rounds / 30,402 checks) and does **not**
/// decide this one: no answer in 300 s (release, measured twice), `unknown` in
/// 55.3 s at `:max-bv-embedded-checks 50000` having spent all 50,002 checks on
/// 52 rounds / 171 instances. So `#P2b-59` is fixed for one spelling of its
/// own repro and not for the other, and no `TODO.md` entry names this script.
///
/// The claim is made on the **deterministic** counters rather than on a clock:
/// where the original order plateaus at 10 rounds at both 2,000 and 5,000
/// checks — which is the equality the sibling fixpoint pin asserts — this
/// order reports 25 and then 33. The test asserts both halves side by side, so
/// it is a statement about the **order** and not about the script.
///
/// To close: make this answer `sat` (the base answers it in a tenth of a
/// millisecond) and flip the two `unknown`s below.
#[test]
fn the_same_script_with_its_two_assertions_swapped_is_not_decided() {
    let swapped = format!("{Q33_DECLS}{Q33_ASSERT_1}{Q33_ASSERT_0}");
    let original = format!("{Q33_DECLS}{Q33_ASSERT_0}{Q33_ASSERT_1}");
    assert!(
        !swapped.contains("forall") && !swapped.contains("exists"),
        "this pin's whole point is that the script carries no binder"
    );
    let (v_low, rounds_low, _) = at_budget(&swapped, 2_000);
    let (v_high, rounds_high, _) = at_budget(&swapped, 5_000);
    assert_eq!(
        (v_low.as_str(), v_high.as_str()),
        ("unknown", "unknown"),
        "THE HOLE IS CLOSED if either of these is `sat`: the base decides this \
         formula in 0.1 ms and this tree decides the SAME formula in the other \
         assertion order",
    );
    let (_, orig_low, _) = at_budget(&original, 2_000);
    let (_, orig_high, _) = at_budget(&original, 5_000);
    assert_eq!(
        orig_low, orig_high,
        "the original order is the control: it plateaus at the same round \
         count at 2,000 and 5,000 checks ({orig_low} vs {orig_high}). If the \
         control moved, this pin no longer isolates the assertion order.",
    );
    assert!(
        rounds_high > rounds_low,
        "HOLE: in the swapped order the refinement is still climbing where the \
         original order plateaus — {rounds_low} rounds at 2,000 checks and \
         {rounds_high} at 5,000, against {orig_low} and {orig_high}. If the two \
         orders now agree, `#P2b-59`'s fix has stopped depending on the order \
         the assertions arrive in and this pin should be inverted.",
    );
}

/// `#P2b-58`'s third hole, **closed** by decision (36)'s model completion
/// (re-fix pass 11).
///
/// `#P2b-58` (decision (24a) mechanism (iii)), in the shape decision (36)
/// names as "the collision shape with the array name bound TWICE":
/// `<scratchpad>/oxiz4/rk8/atk/f5_binder_collide_index.smt2` with its vacuous
/// `(forall ((k …)))` nested one level deeper. `c4b04b7` answers `sat` in
/// 0.1 ms; this tree answered `unknown` until the completion landed and now
/// answers `sat` in 0.9 ms release.
///
/// The nesting is what this shape adds: a chain of consecutive `forall`s is
/// **one** multi-binder universal to
/// `solver::array_completion_certify::peel_universal`, so the doubly bound
/// name is renamed away with the singly bound one and neither is confused
/// with the declared constant `k`. A tree that peeled only the outermost
/// binder would leave a quantifier in the certificate's query and decline.
///
/// The truth rests on no external oracle: the second conjunct written out over
/// **all 128 points** of its `(_ BitVec 7)` index sort is `sat` on this tree
/// (14.8 ms release), and the first conjunct is `(= k k)` under two binders,
/// so the whole is satisfiable.
#[test]
fn a_satisfiable_script_with_the_index_name_bound_twice_is_decided() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (declare-const k (_ BitVec 7))\n\
         (assert (and (forall ((k (_ BitVec 7))) (forall ((k (_ BitVec 7))) (= k k)))\n\
                      (forall ((i (_ BitVec 7))) (= (select (store a i k) (_ bv1 7)) k))))\n\
         (check-sat)\n",
        "sat",
        "#P2b-58 (decision (36)): `c4b04b7` answers `sat` in 0.1 ms, the \
         ground expansion below is `sat` on this tree, and a certified \
         completion must now reach it",
    );
}

/// The in-tree oracle for the pin above: the same constraint over every point
/// of its index sort, with no binder left.
#[test]
fn the_doubly_bound_collision_script_is_satisfiable_over_its_whole_index_sort() {
    let mut script = String::from(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (declare-const k (_ BitVec 7))\n",
    );
    for i in 0..128u32 {
        script.push_str(&format!(
            "(assert (= (select (store a (_ bv{i} 7) k) (_ bv1 7)) k))\n"
        ));
    }
    script.push_str("(check-sat)\n");
    assert_verdict(
        &script,
        "sat",
        "the oracle: with the binder gone the constraint is plainly \
         satisfiable, so the `unknown` above is a solver limit and not the \
         formula",
    );
}

/// **Closed**, and the assumption it was pinned to break is discharged rather
/// than worked around (re-fix pass 11).
///
/// Decision (36)'s second named attack — "a body that constrains two arrays
/// through one binder" — at index width 7, one bit above `finite_expand`'s
/// 64-point budget. `c4b04b7` is `unknown` here too, so this is a verdict the
/// tree **gains** rather than one it lost; it was pinned because it refutes
/// the shape decision (36) described its certificate in ("one QF_BV query in
/// the index variable alone"): the body reads a *second* array whose default
/// the candidate model also leaves free.
///
/// The repair is to complete **every** array whose default is free at the same
/// time, and only then build the query. With `a` and `b` both interpreted the
/// negated body has the index variable as its only free symbol again, so it is
/// still one quantifier-free query — the assumption was about *how many arrays
/// a completion covers*, not about the certificate's shape.
/// `solver::array_completion_certify` searches the product of the per-array
/// default pools for exactly that reason, and the model it publishes here is
/// `a = ((as const …) #b1)` beside `b = ((as const …) #b0)`.
///
/// The truth rests on no external oracle: the body written out over all 128
/// points is `sat` on this tree (9.2 ms release), and the published model is
/// replayed against it below.
#[test]
fn a_binder_constraining_two_arrays_at_once_is_decided() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) (bvnot (select b i)))))\n\
         (assert (= (select a #b0000000) #b1))\n\
         (check-sat)\n(get-model)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "#P2b-58 (decision (36)): two arrays through one binder must be \
         completed together\n--- response ---\n{}",
        lines.join("\n")
    );
    // The model is *replayed*, not matched as text: a later pass that changes
    // the default search order may publish a different — and equally correct
    // — pair, and a string match would redden on a right answer. What must
    // hold is that the published interpretation satisfies the body at every
    // point of the index sort and the ground assertion beside it.
    let pins = model_equalities(&lines);
    assert!(
        !pins.is_empty(),
        "the `sat` published no model to replay:\n{}",
        lines.join("\n")
    );
    let mut replay = String::from(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n",
    );
    replay.push_str(&pins);
    for index in 0..128u32 {
        replay.push_str(&format!(
            "(assert (= (select a (_ bv{index} 7)) (bvnot (select b (_ bv{index} 7)))))\n"
        ));
    }
    replay.push_str("(assert (= (select a #b0000000) #b1))\n(check-sat)\n");
    assert_verdict(
        &replay,
        "sat",
        "the published completion must satisfy the body at EVERY point of the \
         index sort and the ground assertion beside it",
    );
}

/// Every `(define-fun n () S v)` of a published model as `(assert (= n v))`.
///
/// An entry whose value is the `?` placeholder is skipped: it is not a value,
/// and asserting it would be a parse error rather than a check.
fn model_equalities(lines: &[String]) -> String {
    let mut out = String::new();
    for line in lines {
        for raw in line.lines() {
            let trimmed = raw.trim();
            let Some(rest) = trimmed.strip_prefix("(define-fun ") else {
                continue;
            };
            let Some((name, rest)) = rest.split_once(" () ") else {
                continue;
            };
            // Exactly ONE trailing `)` — the `define-fun`'s own. Stripping
            // every trailing paren would eat the value's, which for an array
            // constant is `((as const …) #b1)`.
            let Some(body) = rest.trim_end().strip_suffix(')') else {
                continue;
            };
            let Some(value) = split_sort_and_value(body) else {
                continue;
            };
            if value == "?" {
                continue;
            }
            out.push_str(&format!("(assert (= {name} {value}))\n"));
        }
    }
    out
}

/// Split `"<sort> <value>"` into its value half, honouring nesting in the sort.
fn split_sort_and_value(body: &str) -> Option<&str> {
    let bytes = body.as_bytes();
    let mut depth = 0i32;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b' ' if depth == 0 && index > 0 => return Some(body[index + 1..].trim()),
            _ => {}
        }
        index += 1;
    }
    None
}

/// `rf9/atk/wu1.smt2` verbatim: the third script `#P2b-58`'s mechanism (iii)
/// names that had no in-tree pin at all until now.
///
/// A guarded universal (`(= p (forall …))` with `p` asserted) over a width-7
/// array, satisfiable by `a = ((as const …) #b1)`. `c4b04b7` answers `sat` in
/// 0.07 ms; this tree answered `unknown` until decision (36)'s completion
/// landed. It was carried only by the round's out-of-tree 102-script attack
/// battery, which nothing in the gate runs — so a regression here would have
/// been invisible to `cargo nextest`.
#[test]
fn the_guarded_universal_member_of_the_completion_family_is_decided() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (assert (= (select a #b0000011) (select a #b0111101)))\n\
         (assert (= (select (store a #b0100010 #b0) #b1011010) (select a #b0100010)))\n\
         (assert p)\n\
         (assert (= p (forall ((i!q (_ BitVec 7))) \
         (= (select (store a i!q #b1) #b1100010) (select a #b1101011)))))\n\
         (check-sat)\n",
        "sat",
        "#P2b-58 (decision (36)): `c4b04b7` answers `sat` in 0.07 ms and the \
         completion must certify it here too",
    );
}

/// The in-tree oracle for the pin above.
#[test]
fn two_arrays_through_one_binder_are_satisfiable_over_the_whole_index_sort() {
    let mut script = String::from(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n",
    );
    for i in 0..128u32 {
        script.push_str(&format!(
            "(assert (= (select a (_ bv{i} 7)) (bvnot (select b (_ bv{i} 7)))))\n"
        ));
    }
    script.push_str("(assert (= (select a #b0000000) #b1))\n(check-sat)\n");
    assert_verdict(
        &script,
        "sat",
        "the oracle: with the binder gone the two arrays are plainly \
         complementary and `a` can be `#b1` at zero",
    );
}

// ---------------------------------------------------------------------------
// 2. REGRESSION GUARDS for `Solver::array_root_spelling` (#P2b-59's fix)
//
//    The fix changes WHICH TERM the array collector sees for an `ite`-selected
//    array base: the `ite` itself rather than the proxy constant
//    `eliminate_nonbool_ite` mints. Every guard below is a shape where putting
//    the `ite` back could lose an array constraint and publish a wrong `sat`,
//    or lose the branch relation and publish a wrong `unsat`. Each is
//    unsatisfiable (or satisfiable) *by construction*, so no oracle is needed.
// ---------------------------------------------------------------------------

/// Six `ite`-selected array shapes whose refutation the re-spelling must not
/// lose. Each is `unsat` because the reading it denies is forced:
///
/// 1. a **three-deep** `ite` spine whose every branch is the same array;
/// 2. a three-deep spine over two arrays, both pinned at the read index;
/// 3. a `store` on an `ite` base read at an `ite` **index**;
/// 4. an **array of arrays** with an `ite` outer base;
/// 5. `(as const …)` under an `ite`, both branches the same constant array;
/// 6. `QF_AUFBV`: a `store` on an `ite` base at an **uninterpreted** index.
///
/// One test rather than six so the family cannot be closed one shape at a time
/// and still look finished. Shapes 2 and 5 are answered `sat` — wrongly — by
/// `c4b04b7` and by crates.io 0.3.3 respectively; this tree refutes all six.
#[test]
fn an_ite_selected_array_base_is_refuted_at_every_shape() {
    let cases: [(&str, &str); 6] = [
        (
            "three-deep spine, every branch the same array",
            "(declare-const a (Array (_ BitVec 8) (_ BitVec 1)))\n\
             (declare-const c1 Bool)(declare-const c2 Bool)(declare-const c3 Bool)\n\
             (declare-const i (_ BitVec 8))\n\
             (assert (not (= (select (ite c1 a (ite c2 a (ite c3 a a))) i) (select a i))))\n",
        ),
        (
            "three-deep spine over two arrays, both pinned at the read index",
            "(declare-const a (Array (_ BitVec 8) (_ BitVec 1)))\n\
             (declare-const b (Array (_ BitVec 8) (_ BitVec 1)))\n\
             (declare-const c1 Bool)(declare-const c2 Bool)\n\
             (declare-const i (_ BitVec 8))\n\
             (assert (= (select a i) #b1))\n\
             (assert (= (select b i) #b1))\n\
             (assert (not (= (select (ite c1 a (ite c2 b (ite c1 a b))) i) #b1)))\n",
        ),
        (
            "store on an `ite` base read at an `ite` index",
            "(declare-const a (Array (_ BitVec 8) (_ BitVec 1)))\n\
             (declare-const b (Array (_ BitVec 8) (_ BitVec 1)))\n\
             (declare-const c Bool)(declare-const d Bool)\n\
             (declare-const j (_ BitVec 8))\n(declare-const v (_ BitVec 1))\n\
             (assert (not (= (select (store (ite c a b) j v) (ite d j j)) v)))\n",
        ),
        (
            "array of arrays with an `ite` outer base",
            "(declare-const aa (Array (_ BitVec 8) (Array (_ BitVec 8) (_ BitVec 1))))\n\
             (declare-const c Bool)\n\
             (declare-const i (_ BitVec 8))(declare-const j (_ BitVec 8))\n\
             (assert (not (= (select (select (ite c aa aa) i) j) (select (select aa i) j))))\n",
        ),
        (
            "`(as const …)` under an `ite`, both branches the same constant array",
            "(declare-const c Bool)\n(declare-const i (_ BitVec 8))\n\
             (assert (not (= (select (ite c \
             ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b1) \
             ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b1)) i) #b1)))\n",
        ),
        (
            "QF_AUFBV: a store on an `ite` base at an uninterpreted index",
            "(declare-const a (Array (_ BitVec 8) (_ BitVec 1)))\n\
             (declare-const b (Array (_ BitVec 8) (_ BitVec 1)))\n\
             (declare-fun f ((_ BitVec 8)) (_ BitVec 8))\n\
             (declare-const c Bool)\n(declare-const x (_ BitVec 8))\n\
             (declare-const v (_ BitVec 1))\n\
             (assert (not (= (select (store (ite c a b) (f x) v) (f x)) v)))\n",
        ),
    ];
    for (what, body) in cases {
        assert_verdict(
            &format!("(set-logic ALL)\n{body}(check-sat)\n"),
            "unsat",
            &format!("the `ite`-selected base must stay refutable: {what}"),
        );
    }
}

/// The same family at index widths 2, 8, 16 and 64, on both sides of
/// `ARRAY_INDEX_ENUMERATION_LIMIT` — the boundary
/// `Solver::array_root_spelling` is narrowed to. Width 2 is *below* it (the
/// proxy deliberately survives) and 8, 16 and 64 are above it.
///
/// Both directions, so "answer `unsat` to everything" fails the test as
/// loudly as losing the refutation does: the `store` read back at its own
/// index is `unsat`, and the same `ite` base with the two branches disagreeing
/// at one index is `sat`.
#[test]
fn the_enumeration_limit_boundary_is_sound_in_both_directions() {
    for width in [2u32, 8, 16, 64] {
        assert_verdict(
            &format!(
                "(set-logic ALL)\n\
                 (declare-const a (Array (_ BitVec {width}) (_ BitVec 1)))\n\
                 (declare-const b (Array (_ BitVec {width}) (_ BitVec 1)))\n\
                 (declare-const c Bool)\n\
                 (declare-const i (_ BitVec {width}))\n\
                 (declare-const v (_ BitVec 1))\n\
                 (assert (not (= (select (store (ite c a b) i v) i) v)))\n\
                 (check-sat)\n"
            ),
            "unsat",
            &format!(
                "read-over-write at the write index through an `ite` base must \
                 be refuted at index width {width}"
            ),
        );
        assert_verdict(
            &format!(
                "(set-logic ALL)\n\
                 (declare-const a (Array (_ BitVec {width}) (_ BitVec 1)))\n\
                 (declare-const b (Array (_ BitVec {width}) (_ BitVec 1)))\n\
                 (declare-const c Bool)\n\
                 (declare-const i (_ BitVec {width}))\n\
                 (assert (= (select (ite c a b) i) #b1))\n\
                 (assert (distinct (select a i) (select b i)))\n\
                 (check-sat)\n"
            ),
            "sat",
            &format!(
                "the branches may still disagree away from the selected one at \
                 index width {width}"
            ),
        );
    }
}

/// `push`/`pop` around the array refinement: a lemma retracted by a `pop` is
/// re-derived after it, and the answer is the same the second time.
///
/// `Solver::ite_elim_aliases` carries no trail operation — the argument for
/// that is in its doc (the proxy's name encodes the `ite`'s own `TermId`, so a
/// re-mint re-learns the identical entry). This is the behavioural statement
/// of that argument: the verdict **sequence** is
/// `[sat, unsat, sat, unsat, sat]`, so the refutation survives being popped
/// and re-asserted, and the satisfiable outer scope is restored intact.
#[test]
fn a_popped_array_lemma_is_re_derived_after_the_pop() {
    let lines = run("(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const c Bool)\n\
         (declare-const i (_ BitVec 8))\n\
         (declare-const v (_ BitVec 1))\n\
         (assert (= (select (store (ite c a b) i v) i) v))\n\
         (check-sat)\n\
         (push 1)\n\
         (assert (not (= (select (store (ite c a b) i v) i) v)))\n\
         (check-sat)\n\
         (pop 1)\n\
         (check-sat)\n\
         (push 1)\n\
         (assert (not (= (select (store (ite c a b) i v) i) v)))\n\
         (check-sat)\n\
         (pop 1)\n\
         (check-sat)\n");
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "unsat", "sat", "unsat", "sat"],
        "the contradiction must be re-derived after the `pop` that retracted \
         it, and the outer scope must come back satisfiable\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 3. SOUNDNESS GUARDS for the array model completion (#P2b-58, decision (36))
//
//    Added by re-fix pass 11, beside the four pins it inverted above. The
//    completion searches a pool of constant-array defaults; what makes it
//    sound is not the search but the CERTIFICATE it has to pass — a
//    quantifier-free validity query per universal, plus one over the
//    assertions.
//
//    MEASURED, not assumed: the mutation "completion without its certificate"
//    (M11b in `TODO.md` `#P2b-58`) does NOT redden the two scripts below —
//    so whatever refutes them, it is not the completion. (Which path does is
//    NOT established here and is not claimed: the completion hook runs only
//    where the verdict would otherwise be `unknown` or a gate-downgraded
//    `sat`, and both answer `unsat` without it.) What M11b
//    reddens instead is the *model* half of
//    `round4_pass9_recheck_pins::a_satisfiable_width_seven_array_script_is_decided_by_a_certified_completion`
//    and of `a_binder_constraining_two_arrays_at_once_is_decided` above, which
//    then publish `sat` beside a model that falsifies their own formula
//    (`a = b = ((as const …) #b1)` for `∀i. a[i] = ¬b[i]`). That is the wrong
//    `sat` the certificate exists to prevent, and those two pins are its
//    mutation witnesses.
//
//    The two scripts below are kept for the other direction: they are the
//    shapes whose *truth* is `unsat` while the completion pool offers a
//    satisfying-looking default, so they redden if a later pass moves the
//    hook earlier — before the refinement has had its say — which is the
//    change that would make M11b able to publish a wrong verdict rather than
//    only a wrong model.
// ---------------------------------------------------------------------------

/// A completion that is wrong at a point the script pins must be refused, and
/// the verdict must be the `unsat` the formula actually has — never a `sat`.
///
/// Both shapes are ones whose default pool contains `#b1` — the value that
/// makes the quantifier true — while the formula is unsatisfiable:
///
/// 1. `a[#b0000001] = #b0` beside `∀i. a[i] = #b1` — the constant-`#b1`
///    completion satisfies the quantifier and contradicts the ground
///    assertion, so the *assertion* half of the certificate is what refuses
///    it;
/// 2. `∀i. (store a #b0000010 #b0)[i] = #b1` — the completion satisfies every
///    index the search looks at and is false at the store's own index, so the
///    *universal* half is what refuses it. Nothing outside the quantifier
///    contradicts anything here, which is why it is a separate case.
///
/// Both are unsatisfiable, and `c4b04b7` answers `unknown` to the second.
#[test]
fn a_completion_that_is_wrong_at_a_point_is_refuted_and_never_published() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (= (select a #b0000001) #b0))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n",
        "unsat",
        "the constant-`#b1` completion is in the search's pool and \
         contradicts the ground assertion: the certificate must refuse it",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const d (_ BitVec 1))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select (store a #b0000010 #b0) i) #b1)))\n\
         (check-sat)\n",
        "unsat",
        "the body is false at the store's own index, which only the \
         universal half of the certificate can see",
    );
}

/// The completion is **deterministic**: the same script answered twice gives a
/// byte-identical response, model included.
///
/// The search is a product over per-array default pools, and both the pool and
/// the array order are built from sorted term ids rather than from a hash
/// iteration order. Without that the published model could differ between two
/// runs of the same binary on the same script, which is the property
/// `rk11/atk/m5_model_determinism.smt2` was written to check.
#[test]
fn the_completed_model_is_the_same_on_two_runs() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (= (select a #b0000011) (select a #b0111101)))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select (store a i #b1) #b1100010) #b1)))\n\
         (check-sat)\n(get-model)\n";
    let first = run(script);
    let second = run(script);
    assert_eq!(verdict(&first), "sat", "response: {}", first.join("\n"));
    assert_eq!(
        first, second,
        "the completed model must not depend on a hash iteration order"
    );
}
