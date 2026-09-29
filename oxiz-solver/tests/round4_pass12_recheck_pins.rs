//! Round-4 adversarial recheck, pass 12 — the pins it leaves in the tree.
//!
//! * A test whose doc carries **THE HOLE IS CLOSED** pins an OPEN hole green:
//!   it asserts the answer this tree gives today, which is WRONG, and it fails
//!   with that phrase the moment the answer changes.  Whoever closes the hole
//!   inverts the test to assert the correct behaviour (its doc names it) and
//!   updates `TODO.md`.
//! * Every other test is a regression guard for behaviour this pass measured
//!   correct, asserting that behaviour directly.
//!
//! Every repro below is carried VERBATIM; the out-of-tree copies live under
//! `<scratchpad>/oxiz4/recheck12/atk/` and are not needed to run anything
//! here.  No test installs a wall clock: every assertion is on a verdict, a
//! verdict sequence or a replayed model (decision (16)).

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

/// Every `(define-fun n () S v)` of a published model as `(assert (= n v))`.
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

/// Every point of `(_ BitVec w)` as a `#b` literal.
fn points(width: u32) -> Vec<String> {
    (0u32..(1 << width))
        .map(|i| format!("#b{i:0width$b}", width = width as usize))
        .collect()
}

/// The published value of `name` in a `(get-model)` response.
fn published(lines: &[String], name: &str) -> Option<String> {
    let pins = model_equalities(lines);
    let prefix = format!("(assert (= {name} ");
    pins.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix("))"))
            .map(str::to_string)
    })
}

/// Pin a wrong `sat` green: panic with THE HOLE IS CLOSED on any other answer.
fn assert_still_wrong_sat(name: &str, script: &str, hole: &str) {
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "THE HOLE IS CLOSED ({hole}, `{name}`): this UNSATISFIABLE script no longer \
         answers the wrong `sat` it was pinned at. Invert this pin to assert \
         `unsat` (or at least never `sat`) and close {hole} in TODO.md.\n\
         --- script ---\n{script}--- response ---\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 1. OPEN HOLES — `#P2b-65`: a pigeonhole answered `sat`.
//
//    `∀i j. i ≠ j ⇒ a[i] ≠ a[j]` over an index sort with more points than the
//    element sort has values is UNSATISFIABLE (an injection that cannot
//    exist).  Spelled with the guard `(not (= i j))` — as a premise of `=>`,
//    swapped, under three negations, or inside an `and` guarding a three-way
//    `distinct` — and beside one ground read, it answers `sat` on this tree.
//    The cause recorded in `TODO.md` `#P2b-65` (`mbqi::sat_certify::eu_walk`
//    descends `Not` without polarity, so `premise_safe` accepts `i ≠ j` as the
//    monotone var-var guard) fits every spelling below: the `or`, `xor`,
//    `ite`, `not (and …)` and `distinct`-guard spellings of the same formula
//    are never `sat` (guarded in section 2).
//
//    Each script's truth is established IN THE TEST by an in-test oracle
//    rather than asserted: the same array read at one more point than the
//    element sort has values, pairwise distinct, is refuted by this tree —
//    and that quantifier-free formula is an instance of the universal.
// ---------------------------------------------------------------------------

/// `(declare-const a (Array <index> <element>))`, the universal `body`, and
/// the ground read `ground` — the shape every pigeonhole below shares.
fn pigeonhole(index: &str, element: &str, body: &str, ground: &str) -> String {
    format!(
        "(set-logic ALL)\n(declare-const a (Array {index} {element}))\n\
         (assert {body})\n(assert {ground})\n(check-sat)\n"
    )
}

/// The in-test oracle: `a` read at `points`, pairwise distinct, beside the
/// ground read — an instance of the universal — must be `unsat`.
fn pigeonhole_instance_is_refuted(index: &str, element: &str, ground: &str, at: &[&str]) {
    let reads: Vec<String> = at.iter().map(|p| format!("(select a {p})")).collect();
    let script = format!(
        "(set-logic ALL)\n(declare-const a (Array {index} {element}))\n\
         (assert (distinct {}))\n(assert {ground})\n(check-sat)\n",
        reads.join(" ")
    );
    assert_verdict(
        &script,
        "unsat",
        "the in-test oracle: more distinct reads than the element sort has values",
    );
}

const BV7: &str = "(_ BitVec 7)";

/// **THE HOLE IS CLOSED** when any script here stops answering `sat`.
///
/// `#P2b-65` at a bit-vector index sort, in six spellings.  Crates.io 0.3.3
/// answers `unknown` to the bit-vector-element members and the same wrong
/// `sat` to the `Bool`-element member; `c4b04b7`, HEAD `c702310` and this
/// tree answer `sat` to all six.  `pigeon2` is `corpus/named/pigeon2.smt2`
/// verbatim.  To close: fix `premise_safe`/`eu_walk` (decision (49)) and
/// invert every entry to assert `unsat` (or at least never `sat`).
#[test]
fn p2b65_bit_vector_index_pigeonholes_still_answer_a_wrong_sat() {
    let neg_impl = format!(
        "(forall ((i {BV7}) (j {BV7})) (=> (not (= i j)) (distinct (select a i) (select a j))))"
    );
    let swapped = format!(
        "(forall ((i {BV7}) (j {BV7})) (=> (not (= j i)) (not (= (select a j) (select a i)))))"
    );
    let triple_neg = format!(
        "(forall ((i {BV7}) (j {BV7})) (=> (not (not (not (= i j)))) \
         (distinct (select a i) (select a j))))"
    );
    let distinct3_neg = format!(
        "(forall ((i {BV7}) (j {BV7}) (k {BV7})) (=> (and (not (= i j)) (not (= j k)) \
         (not (= i k))) (distinct (select a i) (select a j) (select a k))))"
    );
    let bv1_ground = "(= (select a #b0000000) #b0)";
    let cases: [(&str, &str, &String, &str); 6] = [
        ("pigeon2", "(_ BitVec 1)", &neg_impl, bv1_ground),
        (
            "bv7_bv2_neg_impl_g",
            "(_ BitVec 2)",
            &neg_impl,
            "(= (select a #b0000000) #b00)",
        ),
        (
            "bv7_bv1_neg_impl_swapped_g",
            "(_ BitVec 1)",
            &swapped,
            bv1_ground,
        ),
        (
            "bv7_bv1_triple_neg_g",
            "(_ BitVec 1)",
            &triple_neg,
            bv1_ground,
        ),
        (
            "bv7_bv1_distinct3_neg_g",
            "(_ BitVec 1)",
            &distinct3_neg,
            bv1_ground,
        ),
        (
            "bv7_bool_neg_impl_g",
            "Bool",
            &neg_impl,
            "(select a #b0000000)",
        ),
    ];
    for (name, element, body, ground) in cases {
        assert_still_wrong_sat(name, &pigeonhole(BV7, element, body, ground), "#P2b-65");
    }
    let three = ["#b0000000", "#b0000001", "#b0000010"];
    let five = [
        "#b0000000",
        "#b0000001",
        "#b0000010",
        "#b0000011",
        "#b0000100",
    ];
    pigeonhole_instance_is_refuted(BV7, "(_ BitVec 1)", bv1_ground, &three);
    pigeonhole_instance_is_refuted(BV7, "Bool", "(select a #b0000000)", &three);
    pigeonhole_instance_is_refuted(BV7, "(_ BitVec 2)", "(= (select a #b0000000) #b00)", &five);
}

/// **THE HOLE IS CLOSED** when any script here stops answering `sat`.
///
/// `#P2b-65` over an **`Int`** index with a bit-vector element — a
/// REGRESSION against the released crate that `TODO.md` does not record:
/// crates.io 0.3.3 answers the correct `unsat` to all five, while `c4b04b7`,
/// HEAD `c702310` and this tree answer `sat` (measured 2026-09-29, release
/// probes).  `TODO.md` `#P2b-65`'s "crates.io 0.3.3: `unknown`" holds only at
/// a bit-vector index.  To close: as above, and invert every entry to assert
/// `unsat` — 0.3.3 already decides these.
#[test]
fn p2b65_int_index_pigeonholes_answer_a_wrong_sat_that_0_3_3_refutes() {
    let neg_impl =
        "(forall ((i Int) (j Int)) (=> (not (= i j)) (distinct (select a i) (select a j))))";
    let swapped =
        "(forall ((i Int) (j Int)) (=> (not (= j i)) (not (= (select a j) (select a i)))))";
    let triple_neg = "(forall ((i Int) (j Int)) (=> (not (not (not (= i j)))) \
         (distinct (select a i) (select a j))))";
    let distinct3_neg = "(forall ((i Int) (j Int) (k Int)) (=> (and (not (= i j)) \
         (not (= j k)) (not (= i k))) (distinct (select a i) (select a j) (select a k))))";
    let bv1_ground = "(= (select a 0) #b0)";
    let cases: [(&str, &str, &str, &str); 5] = [
        ("int_bv1_neg_impl_g", "(_ BitVec 1)", neg_impl, bv1_ground),
        (
            "int_bv2_neg_impl_g",
            "(_ BitVec 2)",
            neg_impl,
            "(= (select a 0) #b00)",
        ),
        (
            "int_bv1_neg_impl_swapped_g",
            "(_ BitVec 1)",
            swapped,
            bv1_ground,
        ),
        (
            "int_bv1_triple_neg_g",
            "(_ BitVec 1)",
            triple_neg,
            bv1_ground,
        ),
        (
            "int_bv1_distinct3_neg_g",
            "(_ BitVec 1)",
            distinct3_neg,
            bv1_ground,
        ),
    ];
    for (name, element, body, ground) in cases {
        assert_still_wrong_sat(name, &pigeonhole("Int", element, body, ground), "#P2b-65");
    }
    pigeonhole_instance_is_refuted("Int", "(_ BitVec 1)", bv1_ground, &["0", "1", "2"]);
    pigeonhole_instance_is_refuted(
        "Int",
        "(_ BitVec 2)",
        "(= (select a 0) #b00)",
        &["0", "1", "2", "3", "4"],
    );
}

/// **THE HOLE IS CLOSED** when either script stops answering `sat`.
///
/// `#P2b-65` over an `Int` index with a `Bool` element (the recheck's
/// `p8_int_g`): wrong `sat` on crates.io 0.3.3, `c4b04b7`, HEAD and this tree
/// alike, so pre-existing — but reachable, and the published model
/// `a = (store (store ((as const (Array Int Bool)) false) 0 true) 1 false)`
/// reads `false` at both `1` and `2`, falsifying the universal at `(1, 2)`.
#[test]
fn p2b65_int_index_bool_element_pigeonhole_still_answers_a_wrong_sat() {
    let neg_impl =
        "(forall ((i Int) (j Int)) (=> (not (= i j)) (distinct (select a i) (select a j))))";
    let triple_neg = "(forall ((i Int) (j Int)) (=> (not (not (not (= i j)))) \
         (distinct (select a i) (select a j))))";
    for (name, body) in [
        ("int_bool_neg_impl_g", neg_impl),
        ("int_bool_triple_neg_g", triple_neg),
    ] {
        assert_still_wrong_sat(
            name,
            &pigeonhole("Int", "Bool", body, "(select a 0)"),
            "#P2b-65",
        );
    }
    pigeonhole_instance_is_refuted("Int", "Bool", "(select a 0)", &["0", "1", "2"]);
}

// ---------------------------------------------------------------------------
// 2. GUARDS — the other spellings of the same pigeonhole are never `sat`.
//    (`unknown` is accepted: these are soundness guards, not verdict pins.)
// ---------------------------------------------------------------------------

/// The `or`, `xor`, `ite`, `not (and …)`, `distinct`-guard, injectivity and
/// nested-`forall` spellings of `#P2b-65`'s pigeonhole, at a bit-vector and at
/// an `Int` index, each beside its ground read.  Measured 2026-09-29: never
/// `sat` on 0.3.3, `c4b04b7`, HEAD or this tree (`unknown` at `(_ BitVec 7)`
/// except the nested `forall`, `unsat` here; `unsat` at `Int`).  A fix to
/// `premise_safe` must not open any of them.
#[test]
fn p2b65_the_other_guard_spellings_are_never_sat() {
    for (index, zero) in [(BV7, "#b0000000"), ("Int", "0")] {
        let bodies = [
            format!(
                "(forall ((i {index}) (j {index})) (or (= i j) (not (= (select a i) (select a j)))))"
            ),
            format!(
                "(forall ((i {index}) (j {index})) (xor (= i j) (distinct (select a i) (select a j))))"
            ),
            format!(
                "(forall ((i {index}) (j {index})) (ite (= i j) true (distinct (select a i) (select a j))))"
            ),
            format!(
                "(forall ((i {index}) (j {index})) (ite (not (= i j)) (distinct (select a i) (select a j)) true))"
            ),
            format!(
                "(forall ((i {index}) (j {index})) (not (and (not (= i j)) (= (select a i) (select a j)))))"
            ),
            format!(
                "(forall ((i {index}) (j {index})) (=> (distinct i j) (distinct (select a i) (select a j))))"
            ),
            format!(
                "(forall ((i {index}) (j {index})) (=> (= (select a i) (select a j)) (= i j)))"
            ),
            format!(
                "(forall ((i {index})) (forall ((j {index})) (=> (not (= i j)) (distinct (select a i) (select a j)))))"
            ),
        ];
        for body in &bodies {
            let script = pigeonhole(
                index,
                "(_ BitVec 1)",
                body,
                &format!("(= (select a {zero}) #b0)"),
            );
            let lines = run(&script);
            assert_ne!(
                verdict(&lines),
                "sat",
                "WRONG SAT: an unsatisfiable pigeonhole answered `sat`\n--- script ---\n{script}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. OPEN HOLES — `#P2b-61` / `#P2b-64`: datatype and declared index sorts.
// ---------------------------------------------------------------------------

/// **THE HOLE IS CLOSED** when either script stops answering `sat`.
///
/// `#P2b-61`, QUANTIFIER-FREE: `a = (store ((as const (Array Color Int)) 0)
/// red 1)` beside `a[green] = 1` is `unsat` (`red ≠ green`), and answers `sat`
/// on 0.3.3, `c4b04b7`, HEAD and this tree.  The quantified twin (`∀x. a[x] =
/// 1`) likewise.  The in-test oracle: with `(distinct red green)` asserted by
/// hand this tree answers `unsat`, so the one fact missing is constructor
/// distinctness reaching the array index reasoning.
#[test]
fn p2b61_enumeration_constructors_are_not_distinct_as_array_indices() {
    let prefix = "(set-logic ALL)\n(declare-datatypes ((Color 0)) (((red) (green) (blue))))\n\
         (declare-const a (Array Color Int))\n\
         (assert (= a (store ((as const (Array Color Int)) 0) red 1)))\n";
    assert_still_wrong_sat(
        "wdt_ground",
        &format!("{prefix}(assert (= (select a green) 1))\n(check-sat)\n"),
        "#P2b-61",
    );
    assert_still_wrong_sat(
        "wdt",
        &format!("{prefix}(assert (forall ((x Color)) (= (select a x) 1)))\n(check-sat)\n"),
        "#P2b-61",
    );
    assert_verdict(
        &format!(
            "{prefix}(assert (= (select a green) 1))\n(assert (distinct red green))\n(check-sat)\n"
        ),
        "unsat",
        "the in-test oracle: with distinctness spelled out the tree refutes it",
    );
}

/// **THE HOLE IS CLOSED** when either script stops answering `sat`.
///
/// `#P2b-64` (1), `corpus/named/dt_field.smt2` — and its QUANTIFIER-FREE core,
/// which the recheck found and `TODO.md` does not record: `a = (store ((as
/// const (Array L Int)) 0) nil 1)` beside `a[(cons 0 nil)] = 1` is `unsat`
/// (`nil ≠ cons 0 nil`) and answers `sat` on 0.3.3, `c4b04b7`, HEAD and this
/// tree.  So `#P2b-64` (1) is `#P2b-61`'s mechanism at a constructor with a
/// field, not a binder defect: the quantified `dt_field` needs no
/// `#P2b-60`-style instance to go wrong.  The in-test oracle names the same
/// index through a constant, `x = (cons 0 nil)`, and this tree refutes it.
#[test]
fn p2b64_a_constructor_with_a_field_is_not_distinct_as_an_array_index() {
    let prefix = "(set-logic ALL)\n(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
         (declare-const a (Array L Int))\n\
         (assert (= a (store ((as const (Array L Int)) 0) nil 1)))\n";
    assert_still_wrong_sat(
        "dt_field_ground",
        &format!("{prefix}(assert (= (select a (cons 0 nil)) 1))\n(check-sat)\n"),
        "#P2b-64 (1) / #P2b-61",
    );
    assert_still_wrong_sat(
        "dt_field",
        &format!("{prefix}(assert (forall ((x L)) (= (select a x) 1)))\n(check-sat)\n"),
        "#P2b-64 (1)",
    );
    assert_verdict(
        &format!(
            "{prefix}(declare-const x L)\n(assert (= x (cons 0 nil)))\n\
             (assert (= (select a x) 1))\n(check-sat)\n"
        ),
        "unsat",
        "the in-test oracle: the same index through a constant is refuted",
    );
}

/// **THE HOLE IS CLOSED** when the script stops answering `sat`.
///
/// `#P2b-64` (2), `corpus/named/usort_skolem.smt2` verbatim: the second
/// element of `U` exists only as the Skolem witness of `(exists ((x U))
/// (distinct x u))`, and `∀x. a[x] = 1` is false there, so the script is
/// `unsat`; it answers `sat` on 0.3.3, `c4b04b7`, HEAD and this tree.  The
/// in-test oracle spells the witness as a declared constant `w`, and this tree
/// refutes that.
#[test]
fn p2b64_a_skolem_witness_is_missing_from_the_index_universe() {
    let tail = "(declare-const a (Array U Int))\n\
         (assert (= a (store ((as const (Array U Int)) 0) u 1)))\n\
         (assert (forall ((x U)) (= (select a x) 1)))\n(check-sat)\n";
    assert_still_wrong_sat(
        "usort_skolem",
        &format!(
            "(set-logic ALL)\n(declare-sort U 0)\n(declare-const u U)\n\
             (assert (exists ((x U)) (distinct x u)))\n{tail}"
        ),
        "#P2b-64 (2)",
    );
    assert_verdict(
        &format!(
            "(set-logic ALL)\n(declare-sort U 0)\n(declare-const u U)\n(declare-const w U)\n\
             (assert (distinct w u))\n{tail}"
        ),
        "unsat",
        "the in-test oracle: the witness as a declared constant is refuted",
    );
}

// ---------------------------------------------------------------------------
// 4. OPEN HOLES — decisions (48) and (41).
// ---------------------------------------------------------------------------

/// **THE HOLE IS CLOSED** when `a` is no longer printed as the bare constant.
///
/// Decision (48): a `Sat` whose published model already satisfies every
/// assertion keeps that model; the completion replaces only a falsifying one.
/// Not implemented in this tree: `Solver::array_completion_at_exit` completes
/// every quantified array `Sat` unconditionally.  On
/// `bench/extended_theories/AUFLIRA/06_quantified_array_property.smt2` with
/// `(get-model)`, HEAD `c702310` (and `c4b04b7`) publish
/// `a = (store (store (store ((as const (Array Int Real)) 0.0) 0 0.0) 1 0.0) 42 0.0)`,
/// which already satisfies every assertion, and this tree replaces it with
/// `((as const (Array Int Real)) 0.0)` — a response change that is not
/// falsifying → certified.  Both models are correct, and that is replayed
/// here: the published `a` pinned beside a `k` with `a[k] < 0` is `unsat`.
/// To close: implement (48) and assert that the printed `a` is the candidate
/// model's store chain again.
#[test]
fn decision_48_a_correct_candidate_model_is_still_replaced() {
    let script = "(set-logic AUFLIRA)\n(set-option :produce-models true)\n\
         (declare-const a (Array Int Real))\n\
         (assert (forall ((i Int)) (>= (select a i) 0.0)))\n\
         (declare-const v0 Real)\n(declare-const v1 Real)\n(declare-const v42 Real)\n\
         (assert (= v0 (select a 0)))\n(assert (= v1 (select a 1)))\n(assert (= v42 (select a 42)))\n\
         (assert (>= v0 0.0))\n(assert (>= v1 0.0))\n(assert (>= v42 0.0))\n\
         (declare-const total Real)\n(assert (= total (+ v0 (+ v1 v42))))\n(assert (>= total 0.0))\n\
         (check-sat)\n(get-model)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let printed = published(&lines, "a").unwrap_or_default();
    assert_verdict(
        &format!(
            "(set-logic AUFLIRA)\n(declare-const a (Array Int Real))\n(assert (= a {printed}))\n\
             (declare-const k Int)\n(assert (< (select a k) 0.0))\n(check-sat)\n"
        ),
        "unsat",
        "whichever model is published must read `>= 0.0` at every index",
    );
    assert_eq!(
        printed,
        "((as const (Array Int Real)) 0.0)",
        "THE HOLE IS CLOSED (decision (48)): the correct candidate model is no \
         longer replaced by the completion. Invert this to assert the candidate's \
         store chain and record (48) closed.\n{}",
        lines.join("\n")
    );
}

/// **THE HOLE IS CLOSED** when the script is decided `sat`.
///
/// Decision (41)'s pinned completion samples every universal at the goal's
/// points plus ONE representative unnamed point — the smallest value of the
/// sort the points do not name.  Under a guard that representative can fall
/// outside the guarded region, and then no sampled instance constrains the
/// default: here `∀i ≥ 2. a[i] = 1` beside `a[0] = 0` (satisfiable, by
/// `(store ((as const …) #b1) #b0000000 #b0)`), the points are `#b0000000`
/// and `#b0000010`, the representative is `#b0000001` (below the guard), and
/// the fill query leaves the default free; the certificate refuses the `#b0`
/// it picks.  `unknown` on 0.3.3, `c4b04b7`, HEAD and this tree — not a lost
/// verdict, a completeness gap in (41).  To close: one representative per
/// region of every order a guard uses (as `mbqi::sat_certify::unnamed_region`
/// already does), then assert `sat` and replay the model at all 128 points.
/// A wrong `unsat` fails this pin too.
#[test]
fn decision_41_a_guard_region_the_representative_misses_stays_undecided() {
    let script = "(set-logic AUFBV)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (=> (bvuge i #b0000010) (= (select a i) #b1))))\n\
         (assert (= (select a #b0000000) #b0))\n(check-sat)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "unknown",
        "THE HOLE IS CLOSED (decision (41) representative): the script is now \
         decided. If `sat`, invert this pin to assert `sat` with a 128-point model \
         replay; an `unsat` is WRONG (the script is satisfiable).\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 4b. OPEN HOLE — a WRONG `unsat` in an incremental QF_AUFBV script.
// ---------------------------------------------------------------------------

/// **THE HOLE IS CLOSED** when the second `(check-sat)` stops answering
/// `unsat`.
///
/// Found by the recheck's own incremental QF_AUFBV fuzzer
/// (`recheck12/fuzz_abv.py`, seed 29092028, script 12503 — 2 wrong `unsat`
/// in 78,658 checks, and the same 2 on HEAD `c702310`), delta-debugged to
/// the eight lines below.  The first assertion makes `a0` the constant array
/// `#b0`; the premise of the second, `a0 = ((as const …) #b1)`, is then
/// false, so it holds; and `(store a0 i0 #b1)` differs from `a0` at `i0`
/// because `a0[i0] = #b0`.  The script is **satisfiable** (the in-test
/// oracle below enumerates all 8 interpretations of `i0` and the two cells of
/// `a0`).  Crates.io 0.3.3 answers `sat sat`; `c4b04b7` `sat unknown`; HEAD
/// `c702310` and this tree **`sat unsat`** — a wrong `unsat`, a regression
/// against the released crate, NOT introduced by re-fix pass 12 (HEAD has
/// it).  It needs the first `(check-sat)` (without it, or with a `push`
/// in its place, the tree answers `unknown`) and the `ite` over an array
/// `distinct` in the second assertion (spelled `(bvule i0 i0)` it is `sat`).
/// To close: find the incremental state the first check leaves behind, and
/// invert this pin to assert `sat sat` (or at least never `unsat`).
#[test]
fn an_incremental_array_script_answers_a_wrong_unsat() {
    let script = "(set-logic QF_AUFBV)\n\
         (declare-fun i0 () (_ BitVec 1))\n\
         (declare-fun a0 () (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= a0 ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n\
         (check-sat)\n\
         (assert (=> (= ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1) a0) \
         (bvule (ite (distinct a0 a0) #b1 i0) i0)))\n\
         (assert (distinct (store a0 i0 #b1) a0))\n\
         (check-sat)\n";
    // The in-test oracle: every interpretation, `a0` as its two cells.
    let mut satisfying = 0u32;
    for i0 in 0u8..2 {
        for cells in 0u8..4 {
            let cell = |index: u8| (cells >> index) & 1;
            let a0_is_const0 = cells == 0;
            let a0_is_const1 = cells == 3;
            // `(bvule x x)` holds for every `x`, so the implication's
            // consequent is true whatever `i0` is.
            let consequent = true;
            let guarded = !a0_is_const1 || consequent;
            let stored_differs = cell(i0) != 1;
            if a0_is_const0 && guarded && stored_differs {
                satisfying += 1;
            }
        }
    }
    assert!(
        satisfying > 0,
        "the oracle must find the model a0 = const #b0"
    );
    // Release answers `sat unsat`.  A build with debug assertions (the gate's
    // test profile) stops earlier, at `theory_manager/conflict_clause.rs`'s
    // `debug_assert!` "theory conflict over [..] produced an empty clause":
    // the array/EUF theory reports a conflict whose every reason is a
    // tautology or a level-0 fact — a spurious conflict — and release turns
    // the fallback clause into the wrong `unsat`.  Both shapes are the hole.
    let outcome = std::panic::catch_unwind(|| run(script));
    let still_open = match &outcome {
        Ok(lines) => verdicts(lines) == ["sat", "unsat"],
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            message.contains("produced an empty clause")
        }
    };
    assert!(
        still_open,
        "THE HOLE IS CLOSED: the second `(check-sat)` of this SATISFIABLE script \
         ({satisfying} of 8 interpretations satisfy it) no longer answers the wrong \
         `unsat` (release) nor trips the empty-clause debug assertion (debug). \
         Invert this pin to assert `sat sat` (never `unsat`).\n{:?}",
        outcome.as_ref().map(|lines| lines.join("\n"))
    );
}

// ---------------------------------------------------------------------------
// 5. REGRESSION GUARDS — decision (45): root-scoped bit-blasting.
//
//    Each verdict sequence is the brute-force truth (every assignment of the
//    script's bit-vector state enumerated, numpy, `recheck12/bf_bv.py`).
// ---------------------------------------------------------------------------

/// A circuit defined at push depth 2 (where `x = 5` and `y = 3` fix it),
/// popped, and asserted at the opposite polarity one level up; then a second
/// circuit defined at depth 1, popped, and negated at depth 0.  Truth:
/// `sat unsat sat unsat sat sat`.
#[test]
fn a_circuit_defined_two_scopes_deep_is_reused_at_the_opposite_polarity() {
    let lines = run("(set-logic QF_BV)\n\
         (declare-fun x () (_ BitVec 4))\n(declare-fun y () (_ BitVec 4))\n\
         (assert (bvult x #x9))\n(push 1)\n(assert (= y #x3))\n(push 1)\n(assert (= x #x5))\n\
         (check-sat)\n(assert (bvult (bvadd x y) #x8))\n(check-sat)\n(pop 1)\n\
         (assert (not (bvult (bvadd x y) #x8)))\n(check-sat)\n\
         (assert (bvult (bvmul x y) #x2))\n(check-sat)\n(pop 1)\n\
         (assert (not (bvult (bvmul x y) #x2)))\n(assert (= (bvadd x y) #x7))\n(check-sat)\n\
         (assert (bvult (bvadd x y) #x8))\n(check-sat)\n");
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "unsat", "sat", "unsat", "sat", "sat"],
        "{}",
        lines.join("\n")
    );
}

/// The adder `z = x + y` is first defined while `x` and `y` are fixed two
/// scopes deep (so every gate of it is unit under the scope's assignment the
/// moment it is installed), then negated one level up and re-asserted at
/// level 0 over a different `y`.  Truth: `sat sat unsat sat sat unsat`.
#[test]
fn a_definition_installed_under_a_fixed_assignment_survives_both_pops() {
    let lines = run("(set-logic QF_BV)\n\
         (declare-fun x () (_ BitVec 3))\n(declare-fun y () (_ BitVec 3))\n\
         (declare-fun z () (_ BitVec 3))\n\
         (push 1)\n(assert (= x #b111))\n(push 1)\n(assert (= y #b001))\n(check-sat)\n\
         (assert (= z (bvadd x y)))\n(check-sat)\n(assert (distinct z #b000))\n(check-sat)\n\
         (pop 1)\n(assert (distinct z (bvadd x y)))\n(check-sat)\n(pop 1)\n\
         (assert (= z (bvadd x y)))\n(assert (= z #b000))\n(assert (= y #b001))\n(check-sat)\n\
         (assert (distinct x #b111))\n(check-sat)\n");
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "sat", "unsat", "sat", "sat", "unsat"],
        "{}",
        lines.join("\n")
    );
}

/// An `ite` circuit over `bvadd`/`bvsub` defined two scopes deep and popped,
/// then asserted at the opposite polarity at level 0.  Truth: `sat sat unsat
/// unsat unsat`; crates.io 0.3.3 answers `sat sat sat unsat sat` (two wrong
/// `sat`s: with `¬p` and `a = b` the `ite` is `a - b = 0`).
#[test]
fn an_ite_circuit_popped_and_negated_is_refuted_where_0_3_3_was_not() {
    let lines = run("(set-logic QF_BV)\n\
         (declare-fun a () (_ BitVec 4))\n(declare-fun b () (_ BitVec 4))\n(declare-fun p () Bool)\n\
         (push 1)\n(push 1)\n(assert (= (ite p (bvadd a b) (bvsub a b)) #x0))\n(assert p)\n\
         (assert (= a #x3))\n(check-sat)\n(pop 2)\n\
         (assert (distinct (ite p (bvadd a b) (bvsub a b)) #x0))\n(assert (= a b))\n(check-sat)\n\
         (assert (not p))\n(check-sat)\n(push 1)\n(assert (= (bvadd a b) #x0))\n(check-sat)\n\
         (pop 1)\n(check-sat)\n");
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "sat", "unsat", "unsat", "unsat"],
        "{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 6. REGRESSION GUARDS — decision (40): the completion on the `Sat` path.
// ---------------------------------------------------------------------------

/// The certificate's own witness, which the evaluation pre-filter cannot
/// stand in for.
///
/// `∀i. a[i] = ((_ extract 0 0) i)` beside `∀i. a[i] = #b0` is UNSAT (at
/// `i = 1`).  The goal names no index literal, so the pre-filter samples the
/// single representative `#b0000000`, where the constant completion
/// `((as const …) #b0)` satisfies both bodies — only the certificate's
/// validity query over every point refuses it.  Measured in the isolated
/// mutation copy (`probe_iso`): with `certificate_passes` short-circuited to
/// `true` this script answers a **wrong `sat`** with a model false at
/// `#b0000001` — with the pre-filter left ON, and with both of re-fix pass
/// 12's hooks turned off as well (so it is the `Unknown`-exit completion of
/// pass 11 that publishes it).  The M11 guards
/// (`round4_pass10_recheck_pins::a_completion_that_is_wrong_at_a_point_is_refuted_and_never_published`,
/// `m4`, `m4b`) stay green under that mutation because `check_core` refutes
/// them first; this one does not.  The tree answers `unknown`, as `c4b04b7`
/// and crates.io 0.3.3 do.  The satisfiable half (`∀i. a[i] = ((_ extract 0
/// 0) i)` alone) must, if it is ever `sat`, publish the alternating array:
/// replayed at all 128 points.
#[test]
fn a_completion_only_the_certificate_can_refuse_is_never_published() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n";
    let alternating = "(assert (forall ((i (_ BitVec 7))) (= (select a i) ((_ extract 0 0) i))))\n";
    let unsat_script = format!(
        "(set-logic ALL)\n(set-option :produce-models true)\n{decls}{alternating}\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b0)))\n(check-sat)\n(get-model)\n"
    );
    let lines = run(&unsat_script);
    assert_ne!(
        verdict(&lines),
        "sat",
        "WRONG SAT: the script is unsatisfiable at i = #b0000001\n{}",
        lines.join("\n")
    );
    let lines = run(&format!(
        "(set-logic ALL)\n(set-option :produce-models true)\n{decls}{alternating}\
         (check-sat)\n(get-model)\n"
    ));
    if verdict(&lines) == "sat" {
        let pins = model_equalities(&lines);
        let mut replay = format!("(set-logic ALL)\n{decls}{pins}");
        for (index, point) in points(7).iter().enumerate() {
            replay.push_str(&format!(
                "(assert (= (select a {point}) #b{}))\n",
                index & 1
            ));
        }
        replay.push_str("(check-sat)\n");
        assert_verdict(
            &replay,
            "sat",
            "a published model must be the alternating array at every point",
        );
    }
}

/// Two universals over a width-8 index and a `Bool` element sort — `a` true
/// everywhere, `b` its negation — `sat`.  HEAD `c702310` published a model
/// the exact evaluation refutes; the published model is replayed here beside
/// both universals written out at all 256 points.
#[test]
fn a_bool_element_width_eight_pair_publishes_a_model_true_at_every_point() {
    let decls = "(declare-const a (Array (_ BitVec 8) Bool))\n\
         (declare-const b (Array (_ BitVec 8) Bool))\n";
    let lines = run(&format!(
        "(set-logic AUFBV)\n(set-option :produce-models true)\n{decls}\
         (assert (forall ((i (_ BitVec 8))) (select a i)))\n\
         (assert (forall ((i (_ BitVec 8))) (= (select b i) (not (select a i)))))\n\
         (check-sat)\n(get-model)\n"
    ));
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let pins = model_equalities(&lines);
    assert!(
        pins.contains("(= a ") && pins.contains("(= b "),
        "{}",
        lines.join("\n")
    );
    let mut replay = format!("(set-logic AUFBV)\n{decls}{pins}");
    for point in points(8) {
        replay.push_str(&format!(
            "(assert (select a {point}))\n(assert (= (select b {point}) (not (select a {point}))))\n"
        ));
    }
    replay.push_str("(check-sat)\n");
    assert_verdict(&replay, "sat", "the model must hold at all 256 points");
}

/// `(get-value)` at a stored and at an unstored point of a pinned completion,
/// and through a `store` over it, agrees with `(get-model)`.  `∀i ≠ 3. a[i]
/// = 1` beside `a[3] = 0`: HEAD `c702310` answered `unknown`; this tree `sat`
/// with `a = (store ((as const …) #b1) #b0000011 #b0)`.  The model is replayed
/// at all 128 points and every `(get-value)` entry is checked against it.
#[test]
fn get_value_at_stored_and_unstored_points_reads_the_pinned_completion() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n";
    let lines = run(&format!(
        "(set-logic AUFBV)\n(set-option :produce-models true)\n{decls}\
         (assert (forall ((i (_ BitVec 7))) (=> (distinct i #b0000011) (= (select a i) #b1))))\n\
         (assert (= (select a #b0000011) #b0))\n(check-sat)\n\
         (get-value ((select a #b0000011) (select a #b0000100) \
         (select (store a #b0000100 #b0) #b0000100) (select (store a #b0000100 #b0) #b0000101)))\n\
         (get-model)\n"
    ));
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let text = lines.join("\n");
    for expected in [
        "((select a #b0000011) #b0)",
        "((select a #b0000100) #b1)",
        "((select (store a #b0000100 #b0) #b0000100) #b0)",
        "((select (store a #b0000100 #b0) #b0000101) #b1)",
    ] {
        assert!(text.contains(expected), "missing `{expected}`\n{text}");
    }
    let pins = model_equalities(&lines);
    let mut replay = format!("(set-logic AUFBV)\n{decls}{pins}");
    for (index, point) in points(7).iter().enumerate() {
        let value = if index == 3 { "#b0" } else { "#b1" };
        replay.push_str(&format!("(assert (= (select a {point}) {value}))\n"));
    }
    replay.push_str("(check-sat)\n");
    assert_verdict(
        &replay,
        "sat",
        "the published `a` must be the pinned completion",
    );
}

/// A completed model does not leak across `push`/`pop`: after the scope that
/// held `∀i. a[i] = 1` is popped, `a[5] = 0` is `sat` and `(get-value)` reads
/// `#b0`; re-entering the universal beside it is `unsat`.
#[test]
fn a_completed_model_does_not_survive_the_pop_of_its_universal() {
    let lines = run("(set-logic AUFBV)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (push 1)\n(assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n(check-sat)\n\
         (get-value ((select a #b0000101)))\n(pop 1)\n\
         (assert (= (select a #b0000101) #b0))\n(check-sat)\n(get-value ((select a #b0000101)))\n\
         (push 1)\n(assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n(check-sat)\n\
         (pop 1)\n(check-sat)\n(get-value ((select a #b0000101)))\n");
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "sat", "unsat", "sat"],
        "{}",
        lines.join("\n")
    );
    let reads: Vec<&String> = lines
        .iter()
        .filter(|line| line.contains("(select a #b0000101)"))
        .collect();
    assert_eq!(reads.len(), 3, "{}", lines.join("\n"));
    assert!(
        reads[0].contains("#b0000101) #b1)")
            && reads[1].contains("#b0000101) #b0)")
            && reads[2].contains("#b0000101) #b0)"),
        "the first read is inside the universal's scope, the other two after its pop\n{}",
        lines.join("\n")
    );
}
