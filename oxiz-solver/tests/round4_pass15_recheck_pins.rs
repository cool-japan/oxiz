//! Round-4 adversarial recheck, pass 15 — the holes it found on re-fix pass
//! 15's tree, pinned GREEN, and the regression guards for what it measured
//! holding.
//!
//! Recheck 15 judged the tree with an independent solver (z3 4.15.4) over
//! three fresh seeded corpora of its own — mixed `Int` / `Real` arithmetic
//! (`gen_mix.py`, seed 30093153, 3,000 scripts), quantifier-free datatypes
//! with uninterpreted functions into datatype / enumeration / uninterpreted
//! sorts (`gen_dt.py`, seed 30093154, the first 600 scripts) and guarded
//! universals over `Int` / `Real` / bit-vectors (`gen_guard.py`, seed
//! 30093152, 2,000 scripts) — every verdict against z3's, every published
//! model replayed, and `c702310` (HEAD), re-fix pass 14's final binary, `c4b04b7`
//! and crates.io 0.3.3 run beside it.  The holes it pins:
//!
//! * §1 — a WRONG `sat`, quantifier-free, on every build (0.3.3 through the
//!   tree): a selector applied to a constructor literal, inside an argument
//!   of an uninterpreted function, is never folded for the congruence
//!   closure — `(distinct (f 1) (f (hd (cons 1 l1))))` answers `sat`.
//! * §2 — a WRONG `sat`, quantifier-free, on every build: a cycle through an
//!   uninterpreted application two constructors deep — `(= (h 1) (cons x
//!   (cons 2 (h 1))))` answers `sat`.  (One constructor deep the tree now
//!   answers `unsat`; HEAD and earlier builds answered `sat`.)
//! * §3 — quantifier-free models that FALSIFY their script, on every build:
//!   an uninterpreted function into a datatype or an enumeration applied at
//!   two arithmetic arguments the arithmetic valued alike prints one value
//!   for both (`uf_consistency`'s Ackermann round keys on scalar results
//!   only: `theory_class_value` answers nothing for a constructor value).
//! * §4 — a REGRESSION of re-fix pass 15 (`c702310` and pass 14 print a
//!   correct model): a record whose field is an arithmetic term over an
//!   uninterpreted application prints a model that falsifies it
//!   (`OXIZ_MUT15_NO_DT_REFINE` restores HEAD's model).
//! * §5 — a REGRESSION of re-fix pass 15 (verdict): a trivially
//!   unsatisfiable tester assertion after a `push`/`pop` answers `unknown`
//!   (`NO_DT_REFINE` / `NO_EQ_ATOM` restore `unsat`).
//! * §6 — verdicts `c702310` and pass 14 decided and the tree does not
//!   (`#P2b-79`'s integral models; `#P2b-75`'s eligibility-first
//!   certificate), down to the three-line `(forall ((i Int)) (distinct i
//!   j))`, `unsat` on every earlier build and `unknown` on the tree.
//! * §7 — a correct model withheld where HEAD and pass 14 print it: the
//!   certificate cannot fold a selector over a constructor literal.
//! * §8 — an `Int` constant printed with a decimal point, `(define-fun x0 ()
//!   Int 3.0)`, on every build (an ill-sorted model; `(get-value)` answers
//!   `3`).
//!
//! **Re-fix pass 16 closed §1–§5, §6 (a) and (c), §7, §8 and §9b at the root and
//! inverted their pins** (`#P2b-82` the selector / tester fold at construction,
//! `#P2b-83` the occurs check over congruence classes, `#P2b-84` class-keyed
//! functional collisions, `#P2b-85` a compound record field read from its
//! leaves, `#P2b-86` the separation repair kept inside the decided atoms,
//! `#P2b-87` the deferred integrality of a quantified goal, and the model
//! printer's two minors).  §6 (b), `g00280`, stays a HOLE, recorded by name
//! under decision (24a) as a cost of `#P2b-75`.
//!
//! A test that asserts the **wrong** answer the tree gives carries a `THE
//! HOLE IS CLOSED` note and fails when the defect is fixed: invert it then,
//! never relax it.  Every model is judged by REPLAY: the published
//! `define-fun` lines are copied into a closed quantifier-free script with
//! the assertion, which folds to a literal, so no replay trusts the solver's
//! reading of its own model.  No test installs a wall clock (decision (16));
//! every script decides in milliseconds in the release probe.
//!
//! **Profiles.**  Recheck 15 wrote §2, §3 and §9b with a branch for the dev
//! profile, where two of the solver's own debug nets fired on these shapes
//! (the table printer's two-entries-at-one-point assertion on §3's
//! non-function model, `model_builder::debug_verify_dt_model` on §2's cycle
//! and on the candidates the datatype refinement refutes).  Pass 16 removed
//! both causes — the candidate is a function, and the datatype net checks
//! only the model a `sat` publishes — so every pin asserts one answer in
//! both profiles.

use oxiz_solver::Context;

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

/// The last verdict, or `"none"`.
fn verdict(lines: &[String]) -> String {
    verdicts(lines)
        .last()
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// Every `(define-fun …)` line of the first published `(model …)`
/// response, verbatim, one per line; `None` when no model was published.
fn model_definitions(lines: &[String]) -> Option<String> {
    let model = lines
        .iter()
        .find(|line| line.trim_start().starts_with("(model"))?;
    let mut out = String::new();
    for raw in model.lines() {
        let trimmed = raw.trim();
        if trimmed.starts_with("(define-fun ") {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    Some(out)
}

/// Whether the response withheld the model (decision (54)(i)'s error).
fn withheld(lines: &[String]) -> bool {
    lines
        .iter()
        .any(|line| line.contains("model not certified"))
}

/// Replay a published model: `prelude` (sorts, datatypes), the model's
/// `define-fun` lines, and one closed claim.
fn replay(prelude: &str, definitions: &str, claim: &str) -> String {
    let script = format!("(set-logic ALL)\n{prelude}{definitions}(assert {claim})\n(check-sat)\n");
    verdict(&run(&script))
}

const LIST: &str = "(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n";

// ---------------------------------------------------------------------------
// §1. CLOSED (pass 16, `#P2b-82`; was a HOLE) — a WRONG `sat`, quantifier-free, on every build: a selector over
//     a constructor literal, inside an uninterpreted argument.
//
//     `(hd (cons 1 l1))` is `1` in every model, so `(f 1)` and `(f (hd (cons
//     1 l1)))` are one application and `distinct` is false: z3 answers
//     `unsat`.  The tree, `c702310`, pass 14, `c4b04b7` and 0.3.3 answer
//     `sat` (recheck15/atk/dt/sel1–sel3; found by `gen_dt.py` seed 30093154,
//     d00238).  The same selector compared with `1` directly is refuted on
//     every build (the oracle below), so the fold exists for the atom and not
//     for the congruence closure.  At an uninterpreted range (`w : Int -> U`)
//     and under the tester the generator writes, the same.
// ---------------------------------------------------------------------------

#[test]
fn a_selector_over_a_constructor_inside_an_application_is_refuted() {
    let prelude = format!(
        "(set-logic ALL)\n{LIST}(declare-sort U 0)\n\
         (declare-fun f (Int) Int)\n(declare-fun w (Int) U)\n(declare-const l1 L)\n"
    );
    let oracle = run(&format!(
        "{prelude}(assert (distinct 1 (hd (cons 1 l1))))\n(check-sat)\n"
    ));
    assert_eq!(
        verdict(&oracle),
        "unsat",
        "the selector over a constructor literal is folded in a plain atom"
    );
    for (name, body) in [
        (
            "sel2_int_range",
            "(assert (distinct (f 1) (f (hd (cons 1 l1)))))\n",
        ),
        (
            "sel1_uninterpreted_range",
            "(assert (distinct (w 1) (w (hd (cons 1 l1)))))\n",
        ),
        (
            "sel3_under_its_tester",
            "(assert (distinct (w 1) (w (ite ((_ is cons) (cons 1 l1)) (hd (cons 1 l1)) 0))))\n",
        ),
    ] {
        let lines = run(&format!("{prelude}{body}(check-sat)\n"));
        assert_eq!(
            verdict(&lines),
            "unsat",
            "`{name}`: the selector over a constructor literal is its field for the \
             congruence closure too (`#P2b-82`)\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §2. CLOSED (pass 16, `#P2b-83`; was a HOLE) — a WRONG `sat`, quantifier-free, on every build: a cycle through
//     an uninterpreted application two constructors deep.
//
//     `(h 1) = (cons x (cons 2 (h 1)))` makes `(h 1)` a proper sub-term of
//     itself; datatypes are well founded, so z3 answers `unsat`.  Every build
//     answers `sat` (recheck15/atk/dt/cyc4, cyc6; `gen_dt.py` d00315).  The
//     same cycle through a declared constant is refuted on every build, and
//     one constructor deep through the application the tree refutes it where
//     HEAD and earlier builds did not (`#P2b-76`'s refinement): those two are
//     the oracles.
// ---------------------------------------------------------------------------

#[test]
fn a_cycle_two_constructors_deep_through_an_application_is_refuted() {
    let prelude = format!(
        "(set-logic ALL)\n{LIST}(declare-fun h (Int) L)\n(declare-const x Int)\n(declare-const l L)\n"
    );
    let oracle = run(&format!(
        "{prelude}(assert (= l (cons 1 (cons 2 l))))\n(check-sat)\n"
    ));
    assert_eq!(
        verdict(&oracle),
        "unsat",
        "a cycle through a declared constant, two constructors deep, is refuted\n{}",
        oracle.join("\n")
    );
    for (name, body) in [
        (
            "cyc4_application_two_deep",
            "(assert (= (h 1) (cons x (cons 2 (h 1)))))\n",
        ),
        (
            "cyc6_through_a_constant",
            "(assert (= l (cons x (cons 2 (h 1)))))\n(assert (= (h 1) l))\n",
        ),
    ] {
        // Both profiles: the dev profile's datatype-model net now checks the
        // model a `sat` publishes only, and there is none.
        let lines = run(&format!("{prelude}{body}(check-sat)\n"));
        assert_eq!(
            verdict(&lines),
            "unsat",
            "`{name}`: the occurs check over the congruence classes refutes a cycle \
             through an application at every depth (`#P2b-83`)\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §3. CLOSED (pass 16, `#P2b-84`; was a HOLE) — quantifier-free models that FALSIFY their script, on every
//     build: a function into a datatype or an enumeration, applied at two
//     arithmetic arguments the arithmetic valued alike.
//
//     `(h x) = [1]` and `(h y) = [2]` with `x, y ∈ [0, 5]` is satisfiable
//     only with `x ≠ y`; the tree (and `c702310`, pass 14, `c4b04b7`) prints
//     `x = y = 0` and `h` constantly `[1]`, so `(h y)` reads `[1]`.  The
//     Ackermann round of `solver::uf_consistency` repairs exactly this for a
//     function whose results are scalars (`d07` of the recheck: `Int` range,
//     correct on the tree, falsifying on HEAD), but `theory_class_value`
//     answers `None` for a constructor value, so no lemma is built and the
//     candidate is printed.  README: "kept a function by an Ackermann lemma
//     at the candidate wherever the congruence closure left two values at
//     one point" — not at these ranges.
// ---------------------------------------------------------------------------

#[test]
fn a_function_into_a_datatype_is_kept_a_function_at_two_separated_points() {
    let bounds = "(declare-const x Int)\n(declare-const y Int)\n\
         (assert (>= x 0))\n(assert (<= x 5))\n(assert (>= y 0))\n(assert (<= y 5))\n";
    for (name, sorts, function, facts, claim) in [
        (
            "d01_list_range",
            LIST,
            "(declare-fun h (Int) L)\n",
            "(assert (= (h x) (cons 1 nil)))\n(assert (= (h y) (cons 2 nil)))\n",
            "(and (= (h x) (cons 1 nil)) (= (h y) (cons 2 nil)) (<= 0 x 5) (<= 0 y 5))",
        ),
        (
            "d10_enumeration_range",
            "(declare-datatypes ((C 0)) (((red) (green) (blue))))\n",
            "(declare-fun h (Int) C)\n",
            "(assert (= (h x) red))\n(assert (= (h y) green))\n",
            "(and (= (h x) red) (= (h y) green) (<= 0 x 5) (<= 0 y 5))",
        ),
    ] {
        let script =
            format!("(set-logic ALL)\n{sorts}{function}{bounds}{facts}(check-sat)\n(get-model)\n");
        // Both profiles: the table printer's two-entries assertion no longer
        // fires, because the candidate is a function.
        let lines = run(&script);
        assert_eq!(verdict(&lines), "sat", "`{name}`\n{}", lines.join("\n"));
        let definitions = model_definitions(&lines).unwrap_or_default();
        assert_eq!(
            replay(sorts, &definitions, claim),
            "sat",
            "`{name}`: the published model replays on its own assertions (`#P2b-84`)\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §4. CLOSED (pass 16, `#P2b-85`; was a HOLE) — a REGRESSION of re-fix pass 15: a record whose field is an
//     arithmetic term over an application prints a falsifying model.
//
//     `p = (mk (+ (q 3) 1) 5)`: `c702310` and pass 14 print `p = (mk 0 5)`
//     beside `q` constantly `-1`, which is a model; the tree prints `p = (mk 0
//     5)` beside `q` constantly `0`, so the field reads `1` (z3: falsifying;
//     `(get-value ((+ (q 3) 1) (px p)))` answers `1` and `0`).  The isolated
//     copy with `OXIZ_MUT15_NO_DT_REFINE` prints HEAD's model: the datatype
//     refinement round (`solver::dt_refinement`) re-solves and the candidate
//     it ends on no longer agrees with itself.  `gen_dt.py`'s corpus: 21 checks
//     in 15 of 600 scripts go from a model z3 confirms (HEAD) to a falsifying
//     one (tree), 60 the other way.
// ---------------------------------------------------------------------------

#[test]
fn a_record_field_over_an_application_prints_a_model_that_replays() {
    let sorts = "(declare-datatypes ((P 0)) (((mk (px Int) (py Int)))))\n";
    // Recheck 15's repro, and the same record with the application pinned
    // away from the one value (`-1`) at which a defaulted field happened to
    // be right: mutation `OXIZ_MUT16_NO_FIELD_FOLD` prints `(mk 0 5)` there
    // (or, with the quantifier-free datatype net, withholds it).
    for (name, extra, claim) in [
        ("recheck15", "", "(= p (mk (+ (q 3) 1) 5))"),
        (
            "q_pinned",
            "(assert (= (q 3) 4))\n",
            "(and (= p (mk (+ (q 3) 1) 5)) (= (q 3) 4))",
        ),
    ] {
        let lines = run(&format!(
            "(set-logic ALL)\n{sorts}(declare-fun q (Int) Int)\n(declare-const p P)\n\
             (assert (= p (mk (+ (q 3) 1) 5)))\n{extra}(check-sat)\n(get-model)\n"
        ));
        assert_eq!(verdict(&lines), "sat", "`{name}`\n{}", lines.join("\n"));
        assert!(!withheld(&lines), "`{name}`\n{}", lines.join("\n"));
        let definitions = model_definitions(&lines).unwrap_or_default();
        assert_eq!(
            replay(sorts, &definitions, claim),
            "sat",
            "`{name}`: the record's compound field is read from its leaves (`#P2b-85`)\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §5. CLOSED (pass 16, `#P2b-86` and `#P2b-82`; was a HOLE) — a REGRESSION of re-fix pass 15 (verdict): after a `push` / `pop`,
//     a trivially false tester assertion answers `unknown`.
//
//     `((_ is nil) (cons (k l2) l3))` is false; the third `check-sat` is
//     `unsat` on `c702310`, pass 14 and z3, `unknown` ("incomplete") on the
//     tree.  Without the `push` / `pop` the tree answers `unsat` (the
//     oracle).  `OXIZ_MUT15_NO_DT_REFINE` or `NO_EQ_ATOM` in the isolated copy
//     restore `unsat` (`gen_dt.py` d00274, reduced by line deletion).
// ---------------------------------------------------------------------------

const U03_HEAD: &str = "(set-logic ALL)\n\
     (declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
     (declare-sort U 0)\n\
     (declare-fun h (Int) L)\n(declare-fun g (Int) C)\n(declare-fun k (L) Int)\n(declare-fun v (L) L)\n\
     (declare-const x Int)\n(declare-const y Int)\n(declare-const z Int)\n\
     (declare-const l1 L)\n(declare-const l2 L)\n(declare-const l3 L)\n\
     (declare-const c1 C)\n(declare-const c2 C)\n(declare-const p P)\n(declare-const u1 U)\n(declare-const u2 U)\n\
     (assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
     (assert (distinct (g 2) c2))\n\
     (assert (= (k (v (cons 2 l1))) (+ (+ x x) x)))\n\
     (assert (not (= (h 2) l2)))\n\
     (assert (or (and (distinct (k (cons 0 (ite ((_ is cons) l2) (tl l2) nil))) 2) \
     (= l1 (ite ((_ is cons) (h 1)) (tl (h 1)) nil))) (or (= blue red) (= u1 u1))))\n";

const U03_TAIL: &str = "(assert ((_ is nil) (cons (k l2) l3)))\n\
     (assert ((_ is cons) (cons (+ 0 1) (cons 1 l1))))\n(check-sat)\n";

#[test]
fn a_false_tester_after_a_pop_is_refuted() {
    let oracle = run(&format!("{U03_HEAD}(assert (<= (px p) 1))\n{U03_TAIL}"));
    assert_eq!(
        verdicts(&oracle),
        vec!["unsat"],
        "without the scope the tree refutes it\n{}",
        oracle.join("\n")
    );
    let lines = run(&format!(
        "{U03_HEAD}(push 1)\n(assert (<= (px p) 1))\n(check-sat)\n(pop 1)\n(check-sat)\n{U03_TAIL}"
    ));
    // Two fixes carry it, each alone (mutation `OXIZ_MUT16_NO_SEP_CHECK`,
    // `NO_DT_FOLD`: either alone keeps `unsat`, both give `unknown`): the
    // second check's candidate is no longer refused over a separation the
    // search had excluded, so no blocking clause survives into the third
    // (`#P2b-86`), and the third's tester over a constructor folds to `false`
    // at construction (`#P2b-82`).
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "sat", "unsat"],
        "{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §6. (a) and (c) CLOSED (pass 16, `#P2b-87`), (b) a HOLE — verdicts `c702310` decides that the tree does not.
//
//     (a) `#P2b-79`'s integral models (`NO_MIXED_INT` restores `unsat`): a
//         guarded universal `∀z:Int. z < x2 ⇒ 2z > x2` (false at every `z` far
//         enough below `x2`) beside one ground disjunction over `to_real`
//         terms.  `c702310` and z3 answer `unsat`, the tree `unknown`
//         (`gen_mix.py` seed 30093153, m00128, reduced; the corpus loses 25
//         checks in 12 of its 324 quantified scripts this way, and gains 22
//         `unsat` and 31 `sat` the other way).
//     (b) a guarded universal of `gen_guard.py` seed 30093152 that
//         `c702310` AND pass 14 decide: g00280 (`Real`, `unsat`;
//         `NO_ELIG_FIRST` restores).
//     (c) `(forall ((i Int)) (distinct i j))` — see below.
// ---------------------------------------------------------------------------

#[test]
fn guarded_universals_head_refutes_are_refuted_again() {
    let m00128 = "(declare-const x0 Int)\n(declare-const x1 Int)\n(declare-const x2 Int)\n\
         (assert (and (>= x0 (- 6)) (<= x0 6)))\n(assert (and (>= x1 (- 6)) (<= x1 6)))\n\
         (assert (and (>= x2 (- 6)) (<= x2 6)))\n\
         (assert (forall ((z Int)) (=> (< z x2) (> (* 2 z) x2))))\n\
         (assert (or (not (>= (+ (* 2.0 (to_real x2)) (* 0.25 (to_real x2)) (* 0.25 (to_real x1))) \
         (+ (* 1.5 (to_real x0)) (* 0.5 (to_real (ite (<= (to_real x1) 2.25) x2 x2)))))) \
         (not (= (+ (* 0.5 (to_real (ite (< (to_real x0) 0.5) x0 x0))) (* 0.5 (to_real x0)) \
         (* 3.0 (to_real (ite (< x2 x2) x2 x1)))) (+ (* 0.5 (to_real x1)) (* (- 1.5) (to_real x1)) \
         (* 1.5 (to_real x2)))))))\n(check-sat)\n";
    let g00280 = "(set-logic ALL)\n(declare-const m Real)\n(declare-const k Real)\n\
         (declare-fun f (Real) Int)\n\
         (assert (forall ((x0 Real)) (=> (< x0 (+ m (- 1.0))) (distinct (f x0) 3))))\n\
         (assert (not (exists ((x1 Real)) (and (and (ite (= x1 32767.0) (< x1 65535.0) \
         (distinct (+ m 2.0) x1)) (or (< 32767.0 x1) (<= x1 (+ m (- 1.0))))) (not (= x1 32767.0))))))\n\
         (assert (<= m k))\n(check-sat)\n";
    // (c) the textbook unsatisfiable universal (`gen14.py` seed 30093155,
    //     s00907, reduced): `i := j` refutes it.  `c702310`, pass 14 and
    //     `c4b04b7` answer `unsat`; the tree `unknown` after 176 MBQI
    //     conflicts (`NO_MIXED_INT` restores `unsat`).  Spelled `(not (= i
    //     j))` it is `unsat` on the tree too.
    let forall_distinct = "(declare-const j Int)\n\
         (assert (forall ((i Int)) (distinct i j)))\n(check-sat)\n";
    let control =
        run("(declare-const j Int)\n(assert (forall ((i Int)) (not (= i j))))\n(check-sat)\n");
    assert_eq!(verdict(&control), "unsat", "{}", control.join("\n"));
    // (a) and (c): the MBQI loop reads the relaxation's values again and the
    // integrality is decided where it would conclude (`#P2b-87`,
    // `solver::integrality_exit`; mutation `OXIZ_MUT16_NO_DEFER_INT`).
    for (name, script) in [
        ("m00128_min", m00128),
        ("s00907_forall_distinct", forall_distinct),
    ] {
        let lines = run(script);
        assert_eq!(
            verdicts(&lines).first().map(String::as_str),
            Some("unsat"),
            "`{name}` (c702310 and z3: `unsat`)\n{}",
            lines.join("\n")
        );
        assert!(
            verdicts(&lines).iter().all(|v| v == "unsat"),
            "`{name}`\n{}",
            lines.join("\n")
        );
    }
    // (b) stays a HOLE, recorded by name under decision (24a) as a cost of
    // `#P2b-75`: `c702310` refuted g00280 through instance bodies the
    // sat-certificate interned and then declined (E-matching matches every
    // trigger against every term of the arena); the eligibility-first order
    // that stopped that leak (`OXIZ_MUT15_NO_ELIG_FIRST` restores `unsat`, and
    // `g14a/s00942` 0.1 s → 10.6 s with it) also removes the side channel.
    let lines = run(g00280);
    let got = verdict(&lines);
    assert_ne!(got, "sat", "`g00280`: a WRONG sat\n{}", lines.join("\n"));
    if got != "unknown" {
        panic!(
            "THE HOLE IS CLOSED (`g00280`): answered `{got}` (c702310 and z3: \
             `unsat`). Invert this pin to assert `unsat`.\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §7. CLOSED (pass 16, the certificate's fold; was a HOLE) — a correct model withheld where `c702310` and pass 14 printed one.
//
//     (Measured, not pinned: `gen_guard.py` seed 30093152 g01284 / g01497, where
//     the release tree's search ends on a candidate that falsifies the
//     universal and the net rightly withholds it while HEAD and pass 14 print
//     a model z3 confirms; the dev profile takes another trajectory there and
//     prints a correct model, so no profile-independent pin exists.)
// ---------------------------------------------------------------------------

/// A correct model withheld because the certificate cannot fold a selector
/// over a constructor literal (`(fst (mk k blue))`, `gen14.py` seed
/// 30093155, s01780, reduced): `c702310` and pass 14 print `k = 0`, `col`
/// constantly `0`, which z3 confirms; the tree answers `(error "model not
/// certified: … could not be certified")`.  On that corpus 7 checks in 5
/// scripts go from a model z3 confirms (HEAD) to withheld (3 against pass
/// 14, all s01780's).
#[test]
fn a_correct_model_over_a_selector_of_a_constructor_is_printed() {
    let script = "(set-logic ALL)\n\
         (declare-datatypes ((Color 0) (Pair 0)) (((red) (green) (blue)) ((mk (fst Int) (snd Color)))))\n\
         (declare-const k Int)\n(declare-fun col (Color) Int)\n\
         (assert (exists ((i Color)) (xor (< (fst (mk k blue)) (col blue)) (= red red))))\n\
         (check-sat)\n(get-model)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    assert!(
        !withheld(&lines),
        "`(fst (mk 0 blue))` folds where the printed model is evaluated, so the \
         certificate decides it (`TermManager::fold_constructor_accessors`)\n{}",
        lines.join("\n")
    );
    let definitions = model_definitions(&lines).unwrap_or_default();
    let sorts = "(declare-datatypes ((Color 0) (Pair 0)) (((red) (green) (blue)) ((mk (fst Int) (snd Color)))))\n";
    assert_eq!(
        replay(
            sorts,
            &definitions,
            "(not (< (fst (mk k blue)) (col blue)))"
        ),
        "sat",
        "the printed model replays\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §8. CLOSED (pass 16; was a HOLE) — an `Int` constant printed with a decimal point, on every build.
//
//     `(assert (= (to_real x0) 3.0))` prints `(define-fun x0 () Int 3.0)`: an
//     ill-sorted model (z3 rejects the replay), while `(get-value (x0))`
//     answers `3`.  0.3.3, `c4b04b7`, `c702310`, pass 14 and the tree alike;
//     it left 116 of the tree's models on `gen_mix.py` seed 30093153
//     unjudgeable until the spelling was rewritten (all 116 then replay).
// ---------------------------------------------------------------------------

#[test]
fn an_int_constant_equal_to_a_real_literal_prints_an_integer() {
    let lines = run("(set-logic ALL)\n(declare-const x0 Int)\n\
         (assert (= (to_real x0) 3.0))\n(check-sat)\n(get-model)\n(get-value (x0))\n");
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let joined = lines.join("\n");
    assert!(joined.contains("((x0 3))"), "{joined}");
    assert!(
        joined.contains("(define-fun x0 () Int 3)"),
        "the value is spelled at the sort of the constant it values\n{joined}"
    );
}

// ---------------------------------------------------------------------------
// §9. REGRESSION GUARDS — what recheck 15 measured holding on the tree and
//     failing on `c702310`.
// ---------------------------------------------------------------------------

/// `#P2b-79`: Nelson-Oppen runs on the relaxation before the integrality
/// branch-and-bound, so the leaf may equate two arguments the closure kept
/// apart; the Ackermann round catches it.  HEAD: `sat` with `x = y = 2` and
/// `f` constant (z3: `unsat`).  And an `Int`-sorted `ite`, an uninterpreted
/// `Int` application, an array read and a datatype `Int` field are integers.
#[test]
fn mixed_integer_goals_are_refuted_under_every_int_leaf() {
    for (name, body) in [
        (
            "m01_branch_after_combination",
            "(declare-const x Int)\n(declare-const y Int)\n(declare-fun f (Int) Int)\n\
             (assert (> (* 2 x) 3))\n(assert (< (* 2 x) 5))\n(assert (> (* 10 y) 19))\n\
             (assert (< (* 10 y) 21))\n(assert (distinct (f x) (f y)))\n",
        ),
        (
            "m02_disequality_after_branching",
            "(declare-const x Int)\n(declare-const y Int)\n\
             (assert (> (* 2 x) 3))\n(assert (< (* 2 x) 5))\n(assert (> (* 10 y) 19))\n\
             (assert (< (* 10 y) 21))\n(assert (distinct x y))\n",
        ),
        (
            "m09_int_ite",
            "(declare-const b Bool)\n(declare-const x Int)\n(declare-const y Int)\n\
             (assert (= (* 3 (ite b x y)) 1))\n",
        ),
        (
            "m10_array_read",
            "(declare-const a (Array Int Int))\n(assert (= (* 2 (select a 0)) 1))\n",
        ),
        (
            "m12_datatype_field",
            "(declare-datatypes ((P 0)) (((mk (fst Int) (snd Real)))))\n(declare-const p P)\n\
             (assert (= (* 2 (fst p)) 1))\n",
        ),
    ] {
        let lines = run(&format!("(set-logic ALL)\n{body}(check-sat)\n"));
        assert_eq!(verdict(&lines), "unsat", "`{name}`\n{}", lines.join("\n"));
    }
}

/// `#P2b-79` under `push` / `pop` and several checks: `sat`, `unsat`, `sat`
/// (z3 agrees; HEAD answered `sat sat sat`).
#[test]
fn mixed_integer_checks_across_scopes_follow_the_assertions() {
    let script = "(set-logic ALL)\n(declare-const x Int)\n(declare-const y Int)\n(declare-const r Real)\n\
         (assert (or (= (* 2 x) (+ y 1)) (= (* 3 x) (+ y 2))))\n\
         (assert (< (to_real y) r))\n(assert (< r 4.5))\n\
         (push 1)\n(assert (> (to_real y) 3.2))\n(check-sat)\n(pop 1)\n\
         (push 1)\n(assert (= y 4))\n(assert (> x 2))\n(check-sat)\n(pop 1)\n(check-sat)\n";
    let lines = run(script);
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "unsat", "sat"],
        "{}",
        lines.join("\n")
    );
}

/// `#P2b-76`'s refinement carries arithmetic: an equality the datatype lemma
/// derives between two fields reaches the arithmetic solver.  HEAD answers
/// `sat` to all three (z3: `unsat`), the release tree `unsat`.
///
/// §9b was a HOLE in the dev profile only: `debug_verify_dt_model` checked
/// the candidate the refinement round was about to refute and panicked
/// ("reconstructed datatype model falsifies an assertion").  CLOSED by re-fix
/// pass 16: the net runs on the model a `sat` publishes only
/// (`Solver::debug_verify_published_dt_model`), so both profiles answer `unsat`.
#[test]
fn a_record_field_equality_reaches_the_arithmetic() {
    let prelude = "(set-logic ALL)\n(declare-datatypes ((P 0)) (((mk (px Int) (py Int)))))\n\
         (declare-fun q (Int) Int)\n(declare-const p P)\n(declare-const n Int)\n";
    let cycle = format!(
        "(set-logic ALL)\n{LIST}(declare-fun h (Int) L)\n(declare-const x Int)\n\
         (assert (= (h 1) (cons x (h 1))))\n(check-sat)\n"
    );
    for (name, script) in [
        (
            "ari1",
            format!(
                "{prelude}(assert (= p (mk (+ (q 3) 1) 5)))\n(assert (= (px p) 0))\n(assert (= (q 3) 5))\n(check-sat)\n"
            ),
        ),
        (
            "ari4",
            format!(
                "{prelude}(assert (= p (mk (+ (q 3) 1) 5)))\n(assert (= p (mk (* 2 n) 5)))\n(assert (= (q 3) n))\n(assert (> n 3))\n(check-sat)\n"
            ),
        ),
        (
            "ari5",
            format!(
                "{prelude}(assert (= p (mk (+ (q 3) 1) 5)))\n(assert (< (px p) (q 3)))\n(check-sat)\n"
            ),
        ),
        ("cyc3_application_one_deep", cycle),
    ] {
        // Both profiles: the dev profile's datatype-model net checks the model
        // a `sat` publishes only, never a candidate the refinement refutes.
        let lines = run(&script);
        assert_eq!(verdict(&lines), "unsat", "`{name}`\n{}", lines.join("\n"));
    }
}

/// `#P2b-76`'s lemmas across `push` / `pop`: a refutation inside a scope,
/// the same assertions outside it, and a later equality.  HEAD answers `sat`
/// to every refutable check (z3: `unsat`).
#[test]
fn constructor_refutations_hold_across_scopes() {
    let prelude = format!(
        "(set-logic ALL)\n{LIST}(declare-const l1 L)\n(declare-const l3 L)\n\
         (declare-fun h (Int) L)\n(declare-const x Int)\n(declare-const y Int)\n"
    );
    let s04 = "(push 1)\n(assert (= l1 (cons 1 (cons 2 nil))))\n\
         (assert (= l3 (cons 1 (cons 2 (cons 3 nil)))))\n(assert (= l1 l3))\n(check-sat)\n(pop 1)\n\
         (assert (= l1 (cons 1 (cons 2 nil))))\n(assert (= l3 (cons 1 (cons 2 (cons 3 nil)))))\n\
         (check-sat)\n(assert (= l1 l3))\n(check-sat)\n";
    let lines = run(&format!("{prelude}{s04}"));
    assert_eq!(
        verdicts(&lines),
        vec!["unsat", "sat", "unsat"],
        "{}",
        lines.join("\n")
    );
    let s05 = "(push 1)\n(assert (= (h x) (cons 1 nil)))\n(assert (= (h y) (cons 1 (cons 2 nil))))\n\
         (assert (= x y))\n(check-sat)\n(pop 1)\n\
         (assert (= (h x) (cons 1 nil)))\n(assert (= (h y) (cons 1 (cons 2 nil))))\n(check-sat)\n\
         (assert (= x y))\n(check-sat)\n";
    let lines = run(&format!("{prelude}{s05}"));
    assert_eq!(
        verdicts(&lines),
        vec!["unsat", "sat", "unsat"],
        "{}",
        lines.join("\n")
    );
}

/// The Ackermann round at a scalar range (`uf_consistency`): `(h x) = 1`,
/// `(h y) = 2` over `x, y ∈ [0, 5]` prints `x ≠ y` and a table that replays;
/// HEAD printed `x = y = 0` beside `h` constantly `1`.
#[test]
fn a_function_into_the_integers_is_kept_a_function() {
    let lines = run(
        "(set-logic ALL)\n(declare-fun h (Int) Int)\n(declare-const x Int)\n\
         (declare-const y Int)\n(assert (>= x 0))\n(assert (<= x 5))\n(assert (>= y 0))\n\
         (assert (<= y 5))\n(assert (= (h x) 1))\n(assert (= (h y) 2))\n(check-sat)\n(get-model)\n",
    );
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let definitions = model_definitions(&lines).unwrap_or_default();
    assert_eq!(
        replay(
            "",
            &definitions,
            "(and (= (h x) 1) (= (h y) 2) (<= 0 x 5) (<= 0 y 5))"
        ),
        "sat",
        "{}",
        lines.join("\n")
    );
}
