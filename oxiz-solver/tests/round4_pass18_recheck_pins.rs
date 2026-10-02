//! Adversarial recheck 18 (decision (89)) — `#P2b-90`'s family and `#P2b-89`'s
//! nested case attacked on shapes re-fix pass 18 did not write, and the hole the
//! recheck found in the lift's bound.
//!
//! * §1 HOLE — a tester over a datatype `ite` chain past
//!   `oxiz_core`'s `MAX_ITE_LIFT_NODES` (256): the lift leaves the tester as
//!   written and the encoded spelling alone decides it (`Solver::encoded_dt_scan`,
//!   the route `OXIZ_MUT18_NO_ITE_LIFT` isolates), which does not scale — on the
//!   release probe of re-fix pass 18's final tree (sha256 `82b744f7…`) the chain
//!   of 250 nodes is refuted in 0.47 s and the chain of 258 has no answer in
//!   130 s; with the lift switched off the encoded spelling alone refutes 120
//!   nodes in 0.42 s, 150 in 22.4 s and none of 180 or 250 in 130 s.  `c702310`
//!   has no answer within 60 s at 30, 120 and 300 nodes, re-fix pass 17 none at
//!   30 nodes and above (10 nodes: `unknown` in 16.4 s).  Never a wrong verdict:
//!   pinned under a deterministic budget, so no test waits on a clock.  Filed as
//!   `TODO.md` `#P2b-93` (open) by re-fix pass 19, which scoped `#P2b-90`'s
//!   closure to chains of at most 256 nodes.
//! * §2 REGRESSION GUARD — testers, selectors and datatype equalities over
//!   `ite`s in positions re-fix pass 18's own battery did not name: a
//!   `define-fun` whose body holds the `ite` or the tester, an n-ary `=`, a
//!   `let`, `check-sat-assuming`, an `ite` as an array index / a stored value /
//!   a selected element, an enumeration pigeonhole of `ite`s, a three-way
//!   `distinct`, a record field, a condition that compares two `ite`s, a
//!   universal and an existential over the list sort, a constructor read back
//!   through a selector and an uninterpreted function, an uninterpreted-sort
//!   `ite` under a function into the list, push / pop of one `ite` in both
//!   polarities.  Every verdict is z3 4.15.4's; re-fix pass 17 answered ten of
//!   these `unknown` and `a18b` (an existential over the list sort) `sat`,
//!   `c702310` also `a14c` `sat` — wrong.
//! * §3 REGRESSION GUARD — arrays of arrays (and functions into arrays, a
//!   datatype with an array field) over a datatype, an enumeration or an
//!   uninterpreted sort, indexed by a tree datatype, an enumeration, `Bool`, a
//!   bit-vector and an uninterpreted sort, under `ite`, `store`, `as const`,
//!   push / pop, a bounded universal and `get-value`: 28 shapes.  `m12` (a
//!   `Bool`-indexed pigeonhole) and `m22` (a universal over the outer index) were
//!   wrong `sat`s on `c702310`, re-fix pass 14 and re-fix pass 17.  Every printed
//!   model is judged by EXACT EVALUATION (`support/dt_eval.rs`); the shapes the
//!   honesty net turns `unknown` (z3: `sat`) are pinned as the property — never
//!   `unsat`, and a printed model holds.
//!
//! * §4 HOLE — `d00498` (fresh `gen_dt.py` seed 30102003): re-fix pass 17
//!   decides all three checks `sat` with models z3 confirms, this tree answers
//!   `unknown` (`incomplete`, not the net) — decision (88)'s class (B′), named
//!   under `TODO.md` decision (24a)'s (B′) beside `d00401` by re-fix pass 19.
//!
//! No test installs a wall clock (decision (16)).

use oxiz_solver::Context;

#[path = "support/dt_eval.rs"]
mod dt_eval;
use dt_eval::{ModelReading, judge};

/// The script's responses (an error folded into one line).
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// Every `sat` / `unsat` / `unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// §1. HOLE — a tester over a datatype `ite` chain past the lift's bound.
//
//     `q0 … qn` are lists all asserted `nil`, `b0 … b(n-1)` free Booleans, and the
//     chain `(ite b0 q0 (ite b1 q1 … qn))` holds `n` `ite` nodes; `((_ is cons)
//     chain)` is unsatisfiable (z3: `unsat` at both sizes).  Under one budget
//     (`:max-conflicts 200`, `:max-decisions 2000`) the 250-node chain is refuted
//     and the 258-node chain answers `unknown`: the lift's bound
//     (`oxiz-core/src/ast/manager/dt_ite_lift.rs`, `MAX_ITE_LIFT_NODES`) is a
//     cliff.  THE HOLE IS CLOSED when the 258-node chain is refuted too — invert
//     the second assertion then, and close `TODO.md` `#P2b-93`, this hole's item.
// ---------------------------------------------------------------------------

/// The tester-over-a-chain script with `n` `ite` nodes, under the budget.
fn chain(n: usize) -> String {
    let mut script = String::from(
        "(set-logic ALL)\n(set-option :max-conflicts 200)\n(set-option :max-decisions 2000)\n\
         (declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n",
    );
    for i in 0..=n {
        script.push_str(&format!(
            "(declare-const q{i} L)\n(declare-const b{i} Bool)\n(assert ((_ is nil) q{i}))\n"
        ));
    }
    let mut term = format!("q{n}");
    for i in (0..n).rev() {
        term = format!("(ite b{i} q{i} {term})");
    }
    script.push_str(&format!("(assert ((_ is cons) {term}))\n(check-sat)\n"));
    script
}

#[test]
fn a_tester_over_an_ite_chain_past_the_lift_bound_is_not_decided() {
    let inside = verdicts(&run(&chain(250)));
    assert_eq!(
        inside,
        vec!["unsat"],
        "the 250-node chain is refuted under the budget (z3: unsat)"
    );
    let past = verdicts(&run(&chain(258)));
    assert_ne!(
        past,
        vec!["sat"],
        "a WRONG sat on the 258-node chain (z3: unsat)"
    );
    if past == vec!["unsat"] {
        panic!(
            "THE HOLE IS CLOSED: the 258-node chain past MAX_ITE_LIFT_NODES is refuted under \
             the budget — invert this pin"
        );
    }
    assert_eq!(past, vec!["unknown"], "the hole as measured");
}

// ---------------------------------------------------------------------------
// §2. REGRESSION GUARD — `#P2b-90` on independent shapes (`$R/recheck18/atk/fam/`).
// ---------------------------------------------------------------------------

#[test]
fn testers_selectors_and_equalities_over_ites_stay_decided_on_independent_shapes() {
    let unsat: [(&str, &str); 26] = [
        ("a04", FAM_A04),
        ("a04b", FAM_A04B),
        ("a06", FAM_A06),
        ("a06b", FAM_A06B),
        ("a06c", FAM_A06C),
        ("a10", FAM_A10),
        ("a10b", FAM_A10B),
        ("a10c", FAM_A10C),
        ("a10d", FAM_A10D),
        ("a12", FAM_A12),
        ("a13", FAM_A13),
        ("a14", FAM_A14),
        ("a14b", FAM_A14B),
        ("a14c", FAM_A14C),
        ("a15", FAM_A15),
        ("a18", FAM_A18),
        ("a18b", FAM_A18B),
        ("a19", FAM_A19),
        ("a19b", FAM_A19B),
        ("a20", FAM_A20),
        ("a21", FAM_A21),
        ("b02", FAM_B02),
        ("b03", FAM_B03),
        ("b07", FAM_B07),
        ("a08", FAM_A08),
        ("b08", FAM_B08),
    ];
    for (name, script) in unsat {
        let lines = run(script);
        let got = verdicts(&lines);
        assert!(
            !got.is_empty() && got.iter().all(|v| v == "unsat"),
            "`{name}` (z3: unsat at every check)\n{}",
            lines.join("\n")
        );
    }
    // check-sat-assuming: verdicts only (z3: unsat, sat, unsat).
    for (name, script) in [("a07", FAM_A07), ("b06", FAM_B06)] {
        let lines = run(script);
        assert_eq!(
            verdicts(&lines),
            vec!["unsat", "sat", "unsat"],
            "`{name}`\n{}",
            lines.join("\n")
        );
    }
    // Satisfiable twins: every printed model holds by exact evaluation.
    for (name, script, checks) in [
        ("a08b", FAM_A08B, 2usize),
        ("a12b", FAM_A12B, 1),
        ("a13b", FAM_A13B, 1),
        ("a22", FAM_A22, 1),
    ] {
        let lines = run(script);
        let got = judge(script, &lines);
        assert_eq!(got.len(), checks, "`{name}`\n{}", lines.join("\n"));
        for (check, (verdict, reading)) in got.iter().enumerate() {
            assert_eq!(
                verdict,
                "sat",
                "`{name}` check {check} (z3: sat)\n{}",
                lines.join("\n")
            );
            assert_eq!(
                reading,
                &ModelReading::Holds,
                "`{name}` check {check}: the printed model holds\n{}",
                lines.join("\n")
            );
        }
    }
    // A tester over nested `ite`s under two nested universals: `unknown` today
    // (z3: unsat) — never `sat`.
    let lines = run(FAM_B04);
    assert_ne!(
        verdicts(&lines),
        vec!["sat"],
        "`b04`: a WRONG sat\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §3. REGRESSION GUARD — arrays of arrays over value sorts (`$R/recheck18/atk/nest/`).
// ---------------------------------------------------------------------------

#[test]
fn arrays_of_arrays_over_value_sorts_stay_decided_with_models_that_hold() {
    for (name, script, expected) in [
        ("m01", NEST_M01, vec!["sat"]),
        ("m02", NEST_M02, vec!["sat", "unsat"]),
        ("m03", NEST_M03, vec!["unsat"]),
        ("m04", NEST_M04, vec!["sat"]),
        ("m07", NEST_M07, vec!["sat"]),
        ("m08", NEST_M08, vec!["unsat"]),
        ("m09", NEST_M09, vec!["sat"]),
        ("m10", NEST_M10, vec!["sat"]),
        ("m11", NEST_M11, vec!["unsat"]),
        ("m12", NEST_M12, vec!["unsat"]),
        ("m13", NEST_M13, vec!["sat"]),
        ("m14", NEST_M14, vec!["sat"]),
        ("m15", NEST_M15, vec!["sat"]),
        ("m16", NEST_M16, vec!["unsat"]),
        ("m18", NEST_M18, vec!["unsat"]),
        ("m19", NEST_M19, vec!["unsat"]),
        ("m20", NEST_M20, vec!["unsat"]),
        ("m21", NEST_M21, vec!["sat", "unsat", "unsat"]),
        ("m22", NEST_M22, vec!["unsat"]),
        ("m23", NEST_M23, vec!["sat"]),
        ("m24", NEST_M24, vec!["unsat"]),
        ("m25", NEST_M25, vec!["unsat"]),
        ("m28", NEST_M28, vec!["sat"]),
    ] {
        let lines = run(script);
        let got = judge(script, &lines);
        assert_eq!(
            got.iter().map(|(v, _)| v.as_str()).collect::<Vec<_>>(),
            expected,
            "`{name}` (z3: {expected:?})\n{}",
            lines.join("\n")
        );
        for (check, (verdict, reading)) in got.iter().enumerate() {
            if verdict == "sat" {
                // `m23` holds a universal the evaluator leaves undecided.
                let fine = matches!(reading, ModelReading::Holds)
                    || (name == "m23" && matches!(reading, ModelReading::Undecided));
                assert!(
                    fine,
                    "`{name}` check {check}: {reading:?} — the printed model holds\n{}",
                    lines.join("\n")
                );
            }
        }
    }
}

/// The nested shapes the tree leaves `unknown` — `m05` (z3: unsat) and `m06`
/// (z3: sat) on every build, `m17`, `m26`, `m27` (z3: sat) where the honesty
/// net reads the candidate false (`:reason-unknown` "model check failed:
/// assertion 5 / 6 / 7 reads false"; under `OXIZ_MUT18_NET_PRINT` the model it
/// withholds is z3-falsifying; `c702310` and re-fix pass 14 printed a
/// falsifying model, re-fix pass 17 withheld or printed one) — never a wrong
/// verdict, and a printed model holds.
#[test]
fn nested_array_goals_left_unknown_are_never_wrong() {
    for (name, script, z3) in [
        ("m05", NEST_M05, "unsat"),
        ("m06", NEST_M06, "sat"),
        ("m17", NEST_M17, "sat"),
        ("m26", NEST_M26, "sat"),
        ("m27", NEST_M27, "sat"),
    ] {
        let lines = run(script);
        let got = judge(script, &lines);
        for (check, (verdict, reading)) in got.iter().enumerate() {
            assert!(
                verdict == "unknown" || verdict == z3,
                "`{name}` check {check}: a WRONG {verdict} (z3: {z3})\n{}",
                lines.join("\n")
            );
            if verdict == "sat" {
                assert_eq!(
                    reading,
                    &ModelReading::Holds,
                    "`{name}` check {check}: the printed model holds\n{}",
                    lines.join("\n")
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The scripts, verbatim from `$R/recheck18/atk/` (z3 4.15.4 verdicts above).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// §4. HOLE — a verdict re-fix pass 17 reached and this tree does not (decision (88)'s class (B'), a new
//     instance on this recheck's fresh `gen_dt.py` seed 30102003): `d00498`, verbatim.  z3 answers `sat`
//     at all three checks; re-fix pass 17 (`88bfac0a…`) answers `sat` at all three with models z3 confirms;
//     this tree answers `unknown` at all three with `:reason-unknown incomplete` — not the honesty net
//     (`OXIZ_MUT18_NET_PRINT` prints nothing) — in about 4 s.  On `probe_iso18`
//     `OXIZ_MUT15_NO_CTOR_FOLD` restores all three, `OXIZ_MUT18_NO_ITE_LIFT` + `OXIZ_MUT18_NO_ENCODED_SCAN`
//     together restore the second and third (as for `d00401`), no single pass-18 switch does.  THE HOLE
//     IS CLOSED when a check answers `sat` — every printed model must then hold.
// ---------------------------------------------------------------------------

const D00498: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun h (Int) L)
(declare-fun g (Int) C)
(declare-fun k (L) Int)
(declare-fun q (C) Int)
(declare-fun w (Int) U)
(declare-fun v (L) L)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const l1 L)
(declare-const l2 L)
(declare-const l3 L)
(declare-const c1 C)
(declare-const c2 C)
(declare-const p P)
(declare-const u1 U)
(declare-const u2 U)
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))
(assert (not (distinct l1 l1)))
(assert (= (cons y (cons 1 l2)) l3))
(assert (and ((_ is nil) (ite ((_ is cons) (h 0)) (tl (h 0)) nil)) (= p (mk (+ (k (h 2)) (- 1)) c1))))
(assert (or (= l3 (v (cons 0 (ite ((_ is cons) l1) (tl l1) nil)))) (distinct u1 u2)))
(check-sat)
(get-model)
(push 1)
(assert (not (distinct nil l2)))
(check-sat)
(get-model)
(check-sat)
(get-model)
"#;

#[test]
fn a_fresh_datatype_goal_pass_seventeen_decides_answers_unknown() {
    let lines = run(D00498);
    let got = judge(D00498, &lines);
    assert_eq!(got.len(), 3, "{}", lines.join("\n"));
    for (check, (verdict, reading)) in got.iter().enumerate() {
        assert_ne!(
            verdict,
            "unsat",
            "check {check}: a WRONG unsat (z3: sat)\n{}",
            lines.join("\n")
        );
        if verdict == "sat" {
            assert_eq!(
                reading,
                &ModelReading::Holds,
                "check {check}: the printed model holds\n{}",
                lines.join("\n")
            );
        }
    }
    if got.iter().any(|(verdict, _)| verdict == "sat") {
        panic!(
            "THE HOLE IS CLOSED (`d00498`): a check answers sat with a model that holds — invert \
             this pin\n{}",
            lines.join("\n")
        );
    }
}
const FAM_A04: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (= l1 l2))
(assert (forall ((n Int)) (=> (and (<= 0 n) (<= n 1)) (= (k (ite (= n 0) l1 l2)) n))))
(check-sat)
"#;
const FAM_A04B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (= (v nil) nil))
(assert (not c))
(assert (forall ((n Int)) (=> (and (<= 0 n) (<= n 1)) ((_ is cons) (v (ite (= n 5) l1 nil))))))
(check-sat)
"#;
const FAM_A06: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(define-fun pick ((b Bool) (s L) (t L)) L (ite b s t))
(assert (not c))
(assert ((_ is cons) (pick c l1 nil)))
(check-sat)
"#;
const FAM_A06B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(define-fun pick ((b Bool) (s L) (t L)) L (ite b s t))
(assert c)
(assert (= l1 (cons 4 nil)))
(assert (= (hd (pick c l1 nil)) 5))
(check-sat)
"#;
const FAM_A06C: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(define-fun pick ((b Bool) (s L) (t L)) L (ite b s t))
(assert (not c))
(assert (= (pick c l1 nil) (cons x l2)))
(check-sat)
"#;
const FAM_A08: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(push 1)
(assert (= (k (ite c l1 nil)) 3))
(assert c)
(assert (= (k l1) 4))
(check-sat)
(pop 1)
(assert (not c))
(assert (= (k (ite c l1 nil)) 3))
(assert (= (k nil) 4))
(check-sat)
"#;
const FAM_A10: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const arr (Array L Int))
(assert (not c))
(assert (= (select arr (ite c l1 nil)) 1))
(assert (= (select arr nil) 2))
(check-sat)
"#;
const FAM_A10B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const arr (Array L Int))
(assert (= (select arr (ite c l1 (cons 1 nil))) 1))
(assert (= (select arr (cons 1 nil)) 2))
(assert (= (select arr l1) 3))
(check-sat)
"#;
const FAM_A10C: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const arr (Array Int L))
(assert (= (select (store arr 0 (ite c l1 nil)) 0) (cons 1 nil)))
(assert (not c))
(check-sat)
"#;
const FAM_A10D: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const arr (Array Int L))
(assert ((_ is cons) (ite c (select arr 0) (select arr 1))))
(assert ((_ is nil) (select arr 0)))
(assert ((_ is nil) (select arr 1)))
(check-sat)
"#;
const FAM_A12: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const p Bool)
(declare-const q Bool)
(declare-const e1 C)
(assert (distinct (ite p red green) (ite q green blue) (ite c blue red) e1))
(check-sat)
"#;
const FAM_A13: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (distinct (ite c l1 nil) nil l1))
(check-sat)
"#;
const FAM_A14: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const r1 R)
(declare-const r2 R)
(assert ((_ is red) (pc r1)))
(assert ((_ is red) (pc r2)))
(assert ((_ is blue) (pc (ite c r1 r2))))
(check-sat)
"#;
const FAM_A14B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert ((_ is blue) (pc (ite c (mk 1 red) (mk x green)))))
(check-sat)
"#;
const FAM_A14C: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const r1 R)
(assert (= (ite c (mk 1 red) (mk x green)) (mk 1 blue)))
(check-sat)
"#;
const FAM_A15: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (not c))
(assert (= l2 nil))
(assert (= (ite (= (ite c l1 nil) l2) (cons 1 nil) nil) (cons 2 nil)))
(check-sat)
"#;
const FAM_A18: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (forall ((m L)) (=> ((_ is cons) m) ((_ is nil) (ite c m nil)))))
(assert c)
(assert ((_ is cons) l1))
(check-sat)
"#;
const FAM_A18B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (exists ((m L)) (and ((_ is cons) (ite c m nil)) (= (hd m) 7))))
(assert (not c))
(check-sat)
"#;
const FAM_A19: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (not c))
(assert ((_ is cons) (tl (cons 1 (ite c l1 nil)))))
(check-sat)
"#;
const FAM_A19B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (not c))
(assert (= (v nil) nil))
(assert ((_ is cons) (v (tl (cons 1 (ite c l1 nil))))))
(check-sat)
"#;
const FAM_A20: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const u1 U)
(declare-const u2 U)
(declare-fun wu (U) L)
(assert (distinct u1 u2))
(assert ((_ is nil) (wu u1)))
(assert ((_ is cons) (wu (ite c u1 u1))))
(check-sat)
"#;
const FAM_A21: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert ((_ is cons) (ite ((_ is cons) l1) (tl l1) l1)))
(assert (= l1 (cons 1 nil)))
(check-sat)
"#;
const FAM_A08B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(push 1)
(assert (= (k (ite c l1 nil)) 3))
(assert c)
(check-sat)
(get-model)
(pop 1)
(assert (not c))
(assert (= (k (ite c l1 nil)) 3))
(assert (= (k l1) 4))
(assert ((_ is cons) (v (ite c l1 nil))))
(check-sat)
(get-model)
"#;
const FAM_A12B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const p Bool)
(declare-const q Bool)
(declare-const e1 C)
(assert (distinct (ite p red green) (ite q green blue) e1))
(assert ((_ is red) (ite c e1 red)))
(assert (not c))
(assert ((_ is blue) e1))
(check-sat)
(get-model)
"#;
const FAM_A13B: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert (distinct (ite c l1 l2) nil l1))
(check-sat)
(get-model)
"#;
const FAM_A22: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(assert ((_ is cons) l1))
(assert (not c))
(check-sat)
(get-value (((_ is cons) (ite c nil l1)) (hd (ite c nil l1)) (ite c nil l1)))
(get-model)
"#;
const FAM_B02: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))
(define-fun tc ((x L)) Bool ((_ is cons) x))
(define-fun hx ((x L)) Int (hd x))
(declare-const l1 L)
(declare-const c Bool)
(assert (not c))
(assert (tc (ite c l1 nil)))
(check-sat)
"#;
const FAM_B03: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(assert (not c))
(assert (= l2 (ite c l1 nil) (cons 1 l1)))
(check-sat)
"#;
const FAM_B07: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))
(declare-const l1 L)
(declare-const c Bool)
(assert (not c))
(assert (let ((t (ite c l1 nil))) (and ((_ is cons) t) (= (hd t) 1))))
(check-sat)
"#;
const FAM_B08: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))
(declare-fun v (L) L)
(declare-const l1 L)
(declare-const c Bool)
(push 1)
(assert c)
(assert ((_ is cons) (v (ite c l1 nil))))
(assert ((_ is nil) (v l1)))
(check-sat)
(pop 1)
(assert (not c))
(assert ((_ is cons) (v (ite c l1 nil))))
(assert ((_ is nil) (v nil)))
(check-sat)
"#;
const FAM_A07: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (R 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun k (L) Int)
(declare-fun v (L) L)
(declare-fun h (Int) L)
(declare-const l1 L)
(declare-const l2 L)
(declare-const c Bool)
(declare-const d Bool)
(declare-const x Int)
(declare-const b0 Bool)
(assert (= b0 ((_ is cons) (ite c l1 nil))))
(assert ((_ is nil) l1))
(check-sat-assuming (b0))
(check-sat-assuming ((not b0)))
(get-model)
(check-sat-assuming (b0 c))
"#;
const FAM_B06: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))
(declare-const l1 L)
(declare-const c Bool)
(declare-const d Bool)
(assert (= d ((_ is cons) (ite c l1 nil))))
(assert ((_ is nil) l1))
(check-sat-assuming (d))
(check-sat-assuming (c))
(get-model)
(check-sat-assuming (c d))
"#;
const FAM_B04: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (E 0)) (((nil) (cons (hd Int) (tl L))) ((ea) (eb))))
(declare-const l1 L)
(assert ((_ is cons) l1))
(assert (forall ((e E)) (forall ((n Int)) (=> (and (<= 0 n) (<= n 1)) ((_ is nil) (ite (= e ea) (ite (= n 0) l1 nil) nil))))))
(check-sat)
"#;
const NEST_M01: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array T (Array Int U)))
(assert (distinct u1 u2))
(assert (= (select (select a leaf) x) u1))
(assert (= (select (select a (node 1 leaf leaf)) y) u2))
(assert (= x y))
(check-sat)
(get-model)
(get-value ((select (select a leaf) x) (select (select a (node 1 leaf leaf)) y)))
"#;
const NEST_M02: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array E (Array Int U)))
(declare-const e1 E)
(declare-const e2 E)
(assert (distinct u1 u2))
(assert (= (select (select a e1) 0) u1))
(assert (= (select (select a e2) 0) u2))
(check-sat)
(get-model)
(assert (= e1 e2))
(check-sat)
"#;
const NEST_M03: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(assert (distinct (select (select a x) 0) (select (select a y) 0) (select (select a z) 0) (select (select a w) 0)))
(check-sat)
"#;
const NEST_M04: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(assert (distinct (select (select a x) 0) (select (select a y) 0) (select (select a z) 0)))
(check-sat)
(get-model)
"#;
const NEST_M05: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(assert (= a ((as const (Array Int (Array Int E))) ((as const (Array Int E)) ea))))
(assert (= (select (select a x) y) eb))
(check-sat)
"#;
const NEST_M06: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(assert (= a (store ((as const (Array Int (Array Int E))) ((as const (Array Int E)) ea)) 2 ((as const (Array Int E)) ec))))
(assert (= (select (select a x) y) ec))
(assert (= (select (select a z) y) ea))
(check-sat)
(get-model)
"#;
const NEST_M07: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int U)))
(declare-const b (Array Int (Array Int U)))
(declare-const p Bool)
(assert (distinct u1 u2))
(assert (= (select (select (ite p a b) x) 0) u1))
(assert (= (select (select a x) 0) u2))
(assert (= (select (select b y) 0) u2))
(check-sat)
(get-model)
"#;
const NEST_M08: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int U)))
(declare-const b (Array Int (Array Int U)))
(declare-const p Bool)
(assert (distinct u1 u2))
(assert (= (select (select (ite p a b) x) 0) u1))
(assert (= (select (select a x) 0) u2))
(assert (= (select (select b x) 0) u2))
(check-sat)
"#;
const NEST_M09: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-fun fa (Int) (Array Int U))
(assert (distinct u1 u2 u3))
(assert (= (select (fa x) 0) u1))
(assert (= (select (fa y) 0) u2))
(assert (= (select (fa z) 1) u3))
(check-sat)
(get-model)
"#;
const NEST_M10: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int (Array Int E))))
(assert (= (select (select (select a x) 0) 0) ea))
(assert (= (select (select (select a y) 0) 0) eb))
(assert (= (select (select (select a z) 0) 0) ec))
(assert (distinct x y))
(check-sat)
(get-model)
"#;
const NEST_M11: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int U)))
(assert (distinct u1 u2))
(assert (= (select a 0) (select a 1)))
(assert (= (select (select a 0) x) u1))
(assert (= (select (select a 1) x) u2))
(check-sat)
"#;
const NEST_M12: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Bool (Array Int U)))
(declare-const p Bool)
(declare-const q Bool)
(assert (distinct u1 u2 u3))
(assert (= (select (select a p) 0) u1))
(assert (= (select (select a q) 0) u2))
(assert (= (select (select a (and p q)) 0) u3))
(check-sat)
"#;
const NEST_M13: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Bool (Array Int U)))
(declare-const p Bool)
(declare-const q Bool)
(assert (distinct u1 u2))
(assert (= (select (select a p) 0) u1))
(assert (= (select (select a q) 0) u2))
(check-sat)
(get-model)
"#;
const NEST_M14: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array U (Array U E)))
(assert (distinct (select (select a u1) u1) (select (select a u2) u2)))
(assert (= (select (select a u1) u2) eb))
(check-sat)
(get-model)
"#;
const NEST_M15: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int T)))
(assert ((_ is node) (select (select a x) 0)))
(assert ((_ is leaf) (select (select a y) 0)))
(assert (= (val (select (select a x) 0)) 3))
(check-sat)
(get-model)
"#;
const NEST_M16: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int T)))
(assert ((_ is node) (select (select a x) 0)))
(assert ((_ is leaf) (select (select a y) 0)))
(assert (= x y))
(check-sat)
"#;
const NEST_M17: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-fun h ((Array Int E)) Int)
(declare-const a (Array Int (Array Int E)))
(assert (distinct (h (select a x)) (h (select a y))))
(assert (= (select (select a x) 0) ea))
(check-sat)
(get-model)
"#;
const NEST_M18: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-fun h ((Array Int E)) Int)
(declare-const a (Array Int (Array Int E)))
(assert (distinct (h (select a x)) (h (select a y))))
(assert (= x y))
(check-sat)
"#;
const NEST_M19: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int U)))
(declare-const b (Array Int (Array Int U)))
(assert (distinct u1 u2))
(assert (= b (store a x (store (select a x) 0 u1))))
(assert (= (select (select b y) 0) u2))
(assert (= (select (select a y) 0) u1))
(check-sat)
(get-model)
"#;
const NEST_M20: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int U)))
(declare-const b (Array Int (Array Int U)))
(assert (distinct u1 u2))
(assert (= b (store a x (store (select a x) 0 u1))))
(assert (= (select (select b y) 0) u2))
(assert (= x y))
(check-sat)
"#;
const NEST_M21: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(push 1)
(assert (distinct (select (select a x) 0) (select (select a y) 0) (select (select a z) 0)))
(check-sat)
(get-model)
(pop 1)
(assert (= (select (select a x) 1) eb))
(assert (= (select (select a y) 1) ec))
(assert (= x y))
(check-sat)
(push 1)
(assert (distinct x y))
(check-sat)
(get-model)
(pop 1)
"#;
const NEST_M22: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(assert (forall ((k Int)) (=> (and (<= 0 k) (<= k 4)) (= (select (select a k) 0) ea))))
(assert (= (select (select a x) 0) eb))
(check-sat)
"#;
const NEST_M23: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(assert (forall ((k Int)) (=> (and (<= 1 k) (<= k 4)) (= (select (select a k) 0) ea))))
(assert (= (select (select a x) 0) eb))
(check-sat)
(get-model)
"#;
const NEST_M24: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array (_ BitVec 1) (Array (_ BitVec 1) E)))
(declare-const i (_ BitVec 1))
(declare-const j (_ BitVec 1))
(declare-const k (_ BitVec 1))
(assert (distinct (select (select a i) i) (select (select a j) j) (select (select a k) k)))
(assert (distinct (select (select a i) #b0) eb))
(check-sat)
(get-model)
"#;
const NEST_M25: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int E)))
(declare-const b (Array Int E))
(assert (= (select a x) b))
(assert (= (select a y) b))
(assert (distinct (select (select a x) z) (select (select a y) z)))
(check-sat)
"#;
const NEST_M26: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int U)))
(assert (distinct u1 u2))
(assert (= (select (select a (ite (= x 0) 1 2)) 0) u1))
(assert (= (select (select a (ite (= y 0) 2 1)) 0) u2))
(check-sat)
(get-model)
"#;
const NEST_M27: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-datatypes ((B 0)) (((mkb (arr (Array Int U)) (tg E)))))
(declare-const b1 B)
(declare-const b2 B)
(assert (distinct u1 u2))
(assert (= (select (arr b1) x) u1))
(assert (= (select (arr b2) y) u2))
(assert (= (tg b1) (tg b2)))
(check-sat)
(get-model)
"#;
const NEST_M28: &str = r#"(set-logic ALL)
(declare-sort U 0)
(declare-datatypes ((T 0) (E 0)) (((leaf) (node (val Int) (lft T) (rgt T))) ((ea) (eb) (ec))))
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const w Int)
(assert (<= 0 x 4))
(assert (<= 0 y 4))
(assert (<= 0 z 4))
(assert (<= 0 w 4))
(declare-const u1 U)
(declare-const u2 U)
(declare-const u3 U)
(declare-const a (Array Int (Array Int U)))
(assert (distinct u1 u2 u3))
(assert (= (select (select a x) y) u1))
(assert (= (select (select a y) x) u2))
(assert (= (select (select a z) z) u3))
(check-sat)
(get-model)
(get-value (x y z (select (select a x) y) (select (select a y) x) (select (select a z) z)))
"#;
