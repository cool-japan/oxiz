//! Round-4 adversarial recheck 17 (decision (80): the commit criterion of
//! decision (78) measured) — pins for what the recheck found open on re-fix
//! pass 17's final tree.
//!
//! §1 and §2 quote corpus scripts of the recheck's own fresh seed of
//! `scripts/round4/gen_dt.py` (seed 30100192, 600 scripts) **verbatim**, or a
//! reduction of one; §3 and §4 are the recheck's status-by-measurement
//! battery of `TODO.md` `#P2b-89` (`$R/recheck17/atk/p89/`, `$R` the round's
//! scratch root).  z3 4.15.4 judged every verdict and every model the doc of a
//! test names.
//!
//! # How to read this file
//!
//! * §1, §2 and §3 were HOLES on re-fix pass 17's tree (each asserted the
//!   defect and failed with a closing message once it was gone).  Re-fix
//!   pass 18 closed all three (decisions (84), (85), (86); `TODO.md`
//!   `#P2b-90`, `#P2b-88`, `#P2b-89`) and inverted them: each now asserts the
//!   right answer and, for a model, its exact evaluation.  §4 is a regression
//!   guard and stays as written.
//! * A printed datatype model is judged here by EXACT EVALUATION (the
//!   evaluator in `support/dt_eval.rs`), not by a replay through this solver:
//!   §2's falsifying model escaped pass 17's printed-model net precisely
//!   because a fresh solver of that build answered `sat` on the closed
//!   (ground) assertion it falsifies, so a replay through the same solver
//!   would have confirmed it.  The evaluator reads the printed `define-fun`
//!   lines (constants and the `ite` tables of functions) and evaluates every
//!   assertion in scope over integers, Booleans, constructor values, arrays
//!   and `@uc_` witnesses (distinct witnesses are distinct elements, as the
//!   printed model says); a selector applied to the wrong constructor is
//!   unspecified and makes a term undecided, never true or false.
//! * On pass 17's tree the scripts of §1 and §2 PANICKED in the dev profile
//!   (the debug-only datatype-model net, `model_builder::debug_verify_dt_model`,
//!   found the reconstructed model false at the `sat` exit).  Re-fix pass 18
//!   removed that dev-only net: a false candidate is the honesty net's
//!   business at the `Context` layer now, in every profile (decision (85)),
//!   so the dev build and the release build answer alike; the panic check is
//!   kept in [`run`] so a regression to the old behaviour still reads as
//!   one.
//! * No test installs a wall clock (decision (16)); every script decides in
//!   well under a second in the release probe.

use oxiz_solver::Context;

/// The script's responses; a panic (the dev profile's datatype-model net,
/// `model_builder::debug_verify_dt_model`, fires on §1's and §2's models) is
/// the single line [`PANICKED`].
fn run(script: &str) -> Vec<String> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ctx = Context::new();
        ctx.execute_script(script)
    }));
    match outcome {
        Ok(Ok(lines)) => lines,
        Ok(Err(err)) => vec![format!("(error \"{err}\")")],
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|m| (*m).to_string()))
                .unwrap_or_default();
            vec![format!("{PANICKED}: {message}")]
        }
    }
}

/// What [`run`] answers for a script that panicked, before the panic's
/// message.
const PANICKED: &str = "PANICKED";

/// The panic of the dev profile's datatype-model net
/// (`model_builder::debug_verify_dt_model`) — and no other panic — on an
/// assertion whose printed form contains `naming` (empty: any assertion).
fn dt_model_net_panicked(lines: &[String], naming: &str) -> bool {
    lines.iter().any(|line| {
        line.starts_with(PANICKED)
            && line.contains("reconstructed datatype model falsifies an assertion")
            && line.contains(naming)
    })
}

/// Every `sat` / `unsat` / `unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

// The exact evaluator for the `gen_dt.py` fragment lives in
// `support/dt_eval.rs` (shared with `round4_pass18_fix_pins`).
#[path = "support/dt_eval.rs"]
mod dt_eval;
use dt_eval::{ModelReading, judge};

/// The datatypes every `gen_dt.py` script declares.
const DT_HEAD: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n";

/// The evaluator itself, on models whose reading is known: a list
/// equality, a table, an unspecified selector, a witness comparison.
#[test]
fn the_exact_evaluator_reads_the_gen_dt_fragment() {
    let script = format!(
        "{DT_HEAD}(declare-fun h (Int) L)\n(declare-const x Int)\n(declare-const l1 L)\n\
         (declare-const u1 U)\n(declare-const u2 U)\n\
         (assert (= (h x) (cons 1 nil)))\n(assert (distinct u1 u2))\n(check-sat)\n(get-model)\n\
         (assert (= l1 (cons x nil)))\n(check-sat)\n(get-model)\n\
         (assert (< 0 (hd (tl l1))))\n(check-sat)\n(get-model)\n"
    );
    let model = |l1: &str| {
        format!(
            "(model\n  (define-fun x () Int -2)\n  (define-fun l1 () L {l1})\n  \
             (define-fun u1 () U @uc_U_0)\n  (define-fun u2 () U @uc_U_1)\n  \
             (define-fun h ((x!0 Int)) L (ite (= x!0 -2) (cons 1 nil) nil))\n)"
        )
    };
    let lines: Vec<String> = vec![
        "sat".into(),
        model("nil"),
        "sat".into(),
        model("(cons (- 2) nil)"),
        "sat".into(),
        model("(cons -2 nil)"),
    ];
    let got = judge(&script, &lines);
    assert_eq!(
        got.iter().map(|(_, reading)| reading).collect::<Vec<_>>(),
        vec![
            &ModelReading::Holds,
            &ModelReading::Holds,
            &ModelReading::Undecided
        ],
        "{got:?}"
    );
    let wrong = vec![
        "sat".to_string(),
        model("nil").replace("(cons 1 nil) nil", "nil (cons 1 nil)"),
    ];
    assert_eq!(
        judge(&script, &wrong).first().map(|(_, r)| r),
        Some(&ModelReading::Falsifies(1))
    );
}

// ---------------------------------------------------------------------------
// §1. CLOSED by re-fix pass 18 (`TODO.md` `#P2b-90`, decision (84)); inverted.  Was a HOLE (BLOCKER:
//     a WRONG `sat`, every build — 0.3.3, `c4b04b7`, `c702310`, re-fix pass 14, re-fix pass 16 and pass
//     17's tree): a tester over a datatype-sorted `ite` is not tied to the `ite`'s value.
//
//     Found by this recheck's fresh `gen_dt.py` seed 30100192, `d00370` (verbatim below): `l2 = (cons
//     (px p) nil)`, the pushed `(= (ite ((_ is cons) l2) (tl l2) nil) l1)` makes `l1 = nil`, and then the
//     fourth assertion's `(ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) … 0)` is `0`, so `(< 0 …)`
//     is false.  z3: `unsat`; every build: `sat` (this tree withholds the model, "assertion 4 is false" —
//     its own net sees the model is false, but the verdict stands).  Smallest form (`M15`): `c` false and
//     `(< 0 (ite ((_ is cons) (ite c l1 nil)) 1 0))`.  Reduced from the corpus script (`M10`): `l1 =
//     nil` beside `(< 0 (ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) 1 0))` is `sat` on every
//     build; naming the inner `ite` with a constant (`(= p (ite …))`, then the tester over `p`) answers
//     `unsat`.  The GROUND formula `C318_B` — two testers / equalities over `ite` chains of list literals,
//     no declared symbol at all — is `sat` on every build too, and it is the closed assertion §2's net
//     confirms with a fresh solver.  The suspected mechanism is `TODO.md` `#P2b-88`'s: the datatype axioms
//     (`solver::dt_axioms`) read the terms as written while the encoder replaces every non-Bool `ite` by a
//     proxy (`encode::bool_euf_encoding`), so a tester applied to the written `ite` is never linked to the
//     proxy's class.
//
//     Re-fix pass 18 confirmed that mechanism (the datatype lemmas were encoded over the written `ite` while
//     the assertion's tester read the proxy: two unrelated atoms) and closed it twice over: a tester, a
//     selector or an equality over a datatype-sorted `ite` is lifted into the branches at term construction
//     and on substitution (`oxiz-core` `ast/manager/dt_ite_lift.rs`), and every datatype lemma is encoded
//     through the same `ite` elimination the assertions go through, so a lemma over an `ite` that is left
//     (under an uninterpreted or constructor argument) reads its proxy (`solver::dt_axioms::assert_dt_lemma`).
//     All four scripts and the control answer `unsat` in both profiles.
// ---------------------------------------------------------------------------

const D00370: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n\
(declare-fun h (Int) L)\n\
(declare-fun g (Int) C)\n\
(declare-fun k (L) Int)\n\
(declare-fun q (C) Int)\n\
(declare-fun w (Int) U)\n\
(declare-fun v (L) L)\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const l3 L)\n\
(declare-const c1 C)\n\
(declare-const c2 C)\n\
(declare-const p P)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
(assert (and (= l2 (cons (px p) (ite ((_ is cons) nil) (tl nil) nil))) (= (ite ((_ is cons) nil) (tl nil) nil) nil)))\n\
(assert (= (g (px p)) red))\n\
(assert (and (< 0 (ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) (hd (ite ((_ is cons) l1) (tl l1) nil)) 0)) (or ((_ is nil) (v l2)) (= blue blue))))\n\
(assert (= l1 (h (ite ((_ is cons) (cons y l3)) (hd (cons y l3)) 0))))\n\
(push 1)\n\
(assert (= (ite ((_ is cons) l2) (tl l2) nil) l1))\n\
(check-sat)\n\
(get-model)\n";

const M10: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(assert (= l1 nil))\n\
(assert (< 0 (ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) 1 0)))\n\
(check-sat)\n";

/// The smallest form found: a Boolean `c` asserted false, and a tester over
/// `(ite c l1 nil)` — which is `nil` — under an arithmetic `ite`.  z3: `unsat`;
/// 0.3.3, `c4b04b7`, `c702310`, pass 14, pass 16 and this tree: `sat`.
const M15: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const c Bool)\n\
(assert (not c))\n\
(assert (< 0 (ite ((_ is cons) (ite c l1 nil)) 1 0)))\n\
(check-sat)\n";

/// `M10` with the inner `ite` named by a constant: decided (`unsat`) today,
/// the in-test control.
const M10_NAMED: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const p L)\n\
(assert (= l1 nil))\n\
(assert (= p (ite ((_ is cons) l1) (tl l1) nil)))\n\
(assert (< 0 (ite ((_ is cons) p) 1 0)))\n\
(check-sat)\n";

const C318_B: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(assert (or ((_ is nil) (ite (= (+ 2 (- 1)) 2) (cons 1 nil) (ite (= (+ 2 (- 1)) 0) (cons -1 (cons -1 (cons -1 (cons 1 nil)))) (ite (= (+ 2 (- 1)) 1) (cons -4 nil) (ite (= (+ 2 (- 1)) -1) (cons 1 (cons -1 (cons -1 (cons 1 nil)))) (cons 1 nil)))))) (= (ite (= (+ 0 (- 1)) 2) (cons 1 nil) (ite (= (+ 0 (- 1)) 0) (cons -1 (cons -1 (cons -1 (cons 1 nil)))) (ite (= (+ 0 (- 1)) 1) (cons -4 nil) (ite (= (+ 0 (- 1)) -1) (cons 1 (cons -1 (cons -1 (cons 1 nil)))) (cons 1 nil))))) nil)))\n\
(check-sat)\n";

#[test]
fn a_tester_over_a_datatype_ite_is_refuted() {
    let control = verdicts(&run(M10_NAMED));
    assert_eq!(
        control,
        vec!["unsat"],
        "the named twin is refuted (z3: unsat)"
    );
    for (name, script) in [
        ("d00370", D00370),
        ("m10", M10),
        ("m15", M15),
        ("c318_b", C318_B),
    ] {
        let lines = run(script);
        assert_eq!(
            verdicts(&lines),
            vec!["unsat"],
            "`{name}` (a tester / equality over a datatype `ite`; z3: unsat)\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §2. CLOSED by re-fix pass 18 (decision (85), the honesty net); inverted.  Was a HOLE (BLOCKER: a
//     published model that FALSIFIES its own script, quantifier-free, datatype): the
//     re-fix pass 17 net (decision (79)(a)) decides the falsified assertion by value and then asks a fresh
//     solver to confirm the refutation — and the fresh solver, hit by §1, answers `sat` on the closed
//     (ground) assertion, so the falsifying model is printed.
//
//     This recheck's fresh `gen_dt.py` seed 30100192, `d00318` (verbatim): the first check is `unknown`,
//     the second `sat` with a model whose sixth assertion in scope — `(or (or ((_ is nil) (h (+ 2 (- 1))))
//     (= (h (+ 0 x)) l2)) (= p (mk (ite ((_ is cons) l3) (hd l3) 0) (pc p))))` — is false: `h` prints
//     non-`nil` lists at 1 and at `x = -1`, `l2 = nil`, and `p = (mk -3 green)` against `(mk 1 green)`.
//     z3 judges the model falsifying; 0.3.3 (by the pin re-check), `c4b04b7`, `c702310`, pass 14 and
//     pass 16 print a falsifying model there too.  The assertion closed over the printed model is
//     exactly the ground formula `C318_B`'s shape, which a fresh solver of this build answers `sat` (§1);
//     that the net's value reading decides it `false` and the confirmation is what lets the model
//     through was found by reading `Context::printed_datatype_model_refuted`, not by instrumentation.
//
//     In the dev profile the debug-only datatype-model net (`debug_verify_dt_model`) panics at the
//     `sat` exit, BEFORE the published-model net runs, so a fix that touches only the published-model
//     net leaves this test green in a dev build: closing it needs the release probe (and a refreshed
//     release model if the model moves).
//
//     Re-fix pass 18: an assertion the exact value reader reads FALSE makes the check `unknown` ("model
//     check failed: assertion N reads false under the candidate model"), whatever a fresh solver says of
//     the closed assertion, and `(get-model)` answers the certified-or-absent error; and `C318_B` itself
//     is refuted now (§1).  Measured on the release probe of the final tree (sha256 `82b744f7…`): both
//     checks — whose candidates the search finds `sat` and false at the sixth assertion (`#P2b-88`) —
//     answer `unknown` with that reason, in the dev profile too.  The model pass 17 printed is kept as
//     `D00318_PASS17_MODEL`, read false; a model an intermediate build of this pass printed at the second
//     check, which z3 confirms, is kept as `D00318_CORRECT_MODEL`, read true — the evaluator tells them
//     apart.
// ---------------------------------------------------------------------------

const D00318: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n\
(declare-fun h (Int) L)\n\
(declare-fun g (Int) C)\n\
(declare-fun k (L) Int)\n\
(declare-fun q (C) Int)\n\
(declare-fun w (Int) U)\n\
(declare-fun v (L) L)\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const l3 L)\n\
(declare-const c1 C)\n\
(declare-const c2 C)\n\
(declare-const p P)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
(assert (not (or (distinct (pc p) (pc p)) (= (k (cons x nil)) y))))\n\
(assert (<= x (+ (ite ((_ is cons) l2) (hd l2) 0) (- 1))))\n\
(assert (distinct nil l3))\n\
(assert (and (or (distinct l2 (cons (px p) l1)) (= (cons (k l1) (cons z (h z))) nil)) (distinct (cons y (cons x (cons x l3))) (h (ite ((_ is cons) (cons z l1)) (hd (cons z l1)) 0)))))\n\
(push 1)\n\
(assert (or (or ((_ is nil) (h (+ 2 (- 1)))) (= (h (+ 0 x)) l2)) (= p (mk (ite ((_ is cons) l3) (hd l3) 0) (pc p)))))\n\
(assert (distinct (w (px p)) u1))\n\
(check-sat)\n\
(get-model)\n\
(push 1)\n\
(assert (and (= p (mk (k l1) green)) (= l3 (h (ite ((_ is cons) (cons 2 l2)) (hd (cons 2 l2)) 0)))))\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn a_datatype_model_the_reader_shows_false_is_never_published() {
    let lines = run(D00318);
    assert!(
        !dt_model_net_panicked(&lines, ""),
        "the dev-only datatype-model net is gone (re-fix pass 18)\n{}",
        lines.join("\n")
    );
    let got = judge(D00318, &lines);
    assert_eq!(got.len(), 2, "{}", lines.join("\n"));
    // z3 answers `sat` at both checks: neither may be `unsat`.
    assert!(
        got.iter().all(|(verdict, _)| verdict != "unsat"),
        "z3: sat at both checks\n{}",
        lines.join("\n")
    );
    // Every printed model reads true; a `sat` never stands over a withheld
    // model, and an `unknown` names the assertion the reader read false.
    for (check, (verdict, reading)) in got.iter().enumerate() {
        if verdict == "sat" {
            assert_eq!(
                reading,
                &ModelReading::Holds,
                "check {check}: a published model satisfies every assertion in scope\n{}",
                lines.join("\n")
            );
        }
    }
    assert!(
        lines.iter().all(|line| !line.contains("is false\"")),
        "a withheld model behind a `sat`\n{}",
        lines.join("\n")
    );
}

/// The model pass 17's release build printed at `d00318`'s second check
/// (sha256 `88bfac0a…`), judged by the exact evaluator in every profile: its
/// sixth assertion in scope is false.
const D00318_PASS17_MODEL: &str = "(model\n\
  (define-fun x () Int -1)\n\
  (define-fun y () Int 2)\n\
  (define-fun z () Int 0)\n\
  (define-fun l1 () L (cons -2 nil))\n\
  (define-fun l2 () L nil)\n\
  (define-fun l3 () L (cons 1 nil))\n\
  (define-fun c1 () C red)\n\
  (define-fun c2 () C red)\n\
  (define-fun p () P (mk -3 green))\n\
  (define-fun u1 () U @uc_U_0)\n\
  (define-fun u2 () U @uc_U_1)\n\
  (define-fun h ((x!0 Int)) L (ite (= x!0 2) (cons 1 nil) (ite (= x!0 0) (cons -1 (cons -1 (cons -1 (cons 1 nil)))) (ite (= x!0 1) (cons -4 nil) (ite (= x!0 -1) (cons 1 (cons -1 (cons -1 (cons 1 nil)))) (cons 1 nil))))))\n\
  (define-fun g ((x!0 Int)) C red)\n\
  (define-fun k ((x!0 L)) Int (ite (= x!0 (cons -1 nil)) 3 (ite (= x!0 (cons -2 nil)) -3 3)))\n\
  (define-fun q ((x!0 C)) Int 0)\n\
  (define-fun w ((x!0 Int)) U @uc_U_2)\n\
  (define-fun v ((x!0 L)) L nil)\n\
)";

#[test]
fn the_pass17_release_model_of_d00318_falsifies_its_sixth_assertion() {
    let lines: Vec<String> = vec![
        "unknown".into(),
        "(error \"No model available\")".into(),
        "sat".into(),
        D00318_PASS17_MODEL.into(),
    ];
    let got = judge(D00318, &lines);
    assert_eq!(
        got.get(1)
            .map(|(verdict, reading)| (verdict.as_str(), reading)),
        Some(("sat", &ModelReading::Falsifies(6))),
        "{got:?}"
    );
}

/// A model of `d00318`'s second check that holds — printed by an intermediate
/// build of re-fix pass 18 (release probe sha256 `ef15ec28…`; the final tree
/// answers that check `unknown`, see above), judged by the exact evaluator in
/// every profile: every assertion in scope holds (z3 4.15.4 confirms it).
const D00318_CORRECT_MODEL: &str = "(model\n\
  (define-fun x () Int -4)\n\
  (define-fun y () Int -2)\n\
  (define-fun z () Int 0)\n\
  (define-fun l1 () L (cons 1 nil))\n\
  (define-fun l2 () L (cons -1 nil))\n\
  (define-fun l3 () L (cons -5 nil))\n\
  (define-fun c1 () C red)\n\
  (define-fun c2 () C red)\n\
  (define-fun p () P (mk -5 green))\n\
  (define-fun u1 () U @uc_U_0)\n\
  (define-fun u2 () U @uc_U_1)\n\
  (define-fun h ((x!0 Int)) L (ite (= x!0 0) (cons 0 nil) (ite (= x!0 2) (cons -5 nil) (ite (= x!0 -4) (cons -3 nil) (ite (= x!0 1) (cons -2 nil) (cons 0 nil))))))\n\
  (define-fun g ((x!0 Int)) C red)\n\
  (define-fun k ((x!0 L)) Int (ite (= x!0 (cons -4 nil)) 0 (ite (= x!0 (cons 1 nil)) -5 0)))\n\
  (define-fun q ((x!0 C)) Int 0)\n\
  (define-fun w ((x!0 Int)) U @uc_U_2)\n\
  (define-fun v ((x!0 L)) L nil)\n\
)";

#[test]
fn a_correct_model_of_d00318_reads_true() {
    let lines: Vec<String> = vec![
        "unknown".into(),
        "(error \"model not certified: model check failed: assertion 6 reads false under the candidate model\")".into(),
        "sat".into(),
        D00318_CORRECT_MODEL.into(),
    ];
    let got = judge(D00318, &lines);
    assert_eq!(
        got.iter()
            .map(|(verdict, reading)| (verdict.as_str(), reading))
            .collect::<Vec<_>>(),
        vec![
            ("unknown", &ModelReading::NoModel),
            ("sat", &ModelReading::Holds)
        ],
        "{got:?}"
    );
}

// ---------------------------------------------------------------------------
// §3. CLOSED by re-fix pass 18 (decision (86); `TODO.md` `#P2b-89` reopened for the nested case and
//     closed again); inverted.  Was a HOLE (a published model that FALSIFIES its own script, quantifier-
//     free, every build — 0.3.3, `c4b04b7`, `c702310`, re-fix pass 14, re-fix pass 16 and pass 17's tree): `TODO.md` `#P2b-89` one level down — an array of ARRAYS into an
//     uninterpreted sort, read at two indices the arithmetic left free.
//
//     `a : (Array Int (Array Int U))`, `x, y ∈ [0, 5]`, `a[x][0] = u1`, `a[y][0] = u2`, `u1 ≠ u2`: every
//     build measured (0.3.3, `c4b04b7`, `c702310`, pass 14, pass 16, this tree) prints `x = y = 0` and `a` with one entry `a[0][0] = @uc_U_0`, so `a[y][0] = u2 = @uc_U_1` is
//     false (z3: falsifying; the script is `sat`).  Re-fix pass 17's read-congruence key
//     (`Solver::build_index_congruence`, `ReadKey::Class`) covers an array whose ELEMENT sort is a
//     datatype or an uninterpreted sort; here the element sort is an array.  The printed-model net runs
//     (the sort contains `U`) but its value reader (`printed_eval::ground_values`) has no array values, so
//     the read of a read stays open and the model is published.  Found by this recheck's status-by-
//     measurement battery of `#P2b-89` (`$R/recheck17/atk/p89/`, 14 shapes): every other shape is decided
//     correctly (pinned below in §4).
//
//     Re-fix pass 18: a read whose element sort CONTAINS a datatype, an enumeration or an uninterpreted sort
//     — an array of arrays into one included — is compared by class, an array-valued read by its congruence
//     class (`Solver::theory_class_value`), so the violated lemma `x = y ⇒ a[x] = a[y]` is built and the
//     two inner reads then merge; and the exact value reader reads array values (nested ones included), so
//     the honesty net would read such a model false.  `b05` prints `x = 0`, `y = 1` and both entries.
// ---------------------------------------------------------------------------

const B05: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-const a (Array Int (Array Int U)))\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(assert (distinct u1 u2))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= (select (select a x) 0) u1))\n\
(assert (= (select (select a y) 0) u2))\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn an_array_of_arrays_into_an_uninterpreted_sort_keeps_both_entries() {
    let lines = run(B05);
    let got = judge(B05, &lines);
    assert_eq!(
        got,
        vec![("sat".to_string(), ModelReading::Holds)],
        "`b05` (z3: sat): the printed model reads true\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §4. REGRESSION GUARD for `TODO.md` `#P2b-89` (closed by re-fix pass 17): the array twin of `#P2b-84`
//     across the shapes this recheck measured — push / pop around the reads, three reads into an
//     enumeration, a store chain, a read used as an index of a datatype-indexed array, a pigeonhole of
//     five reads into three colours, a function and an array into one list sort, two arrays into an
//     uninterpreted sort, a bit-vector index, a record element, and a store over an uninterpreted array.
//     z3 4.15.4: every model `c702310` prints for these is falsifying; this tree's every model reads
//     true.  Each must stay `sat` with a model that evaluates true at every check.
// ---------------------------------------------------------------------------

const B01: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const a (Array Int L))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(push 1)\n\
(assert (= (select a x) (cons 1 nil)))\n\
(assert (= (select a y) (cons 2 nil)))\n\
(check-sat)\n\
(get-model)\n\
(pop 1)\n\
(assert (= (select a z) (cons 3 nil)))\n\
(assert (= (select a x) (cons 4 nil)))\n\
(check-sat)\n\
(get-model)\n\
(push 1)\n\
(assert (= (select a y) (cons 5 nil)))\n\
(check-sat)\n\
(get-model)\n\
(pop 1)\n";

const B02: &str = "(set-logic ALL)\n\
(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
(declare-const a (Array Int C))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= (select a x) red))\n\
(assert (= (select a y) green))\n\
(assert (= (select a z) blue))\n\
(check-sat)\n\
(get-model)\n";

const B03: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const b (Array Int L))\n\
(declare-const a (Array Int L))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= a (store b 3 (cons 5 nil))))\n\
(assert (= (select a x) (cons 1 nil)))\n\
(assert (= (select b y) (cons 2 nil)))\n\
(assert (= (select a z) (cons 5 nil)))\n\
(check-sat)\n\
(get-model)\n";

const B04: &str = "(set-logic ALL)\n\
(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
(declare-const a (Array Int C))\n\
(declare-const m (Array C Int))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= (select a x) red))\n\
(assert (= (select a y) green))\n\
(assert (= (select m (select a x)) 7))\n\
(assert (= (select m (select a y)) 8))\n\
(check-sat)\n\
(get-model)\n";

const B07: &str = "(set-logic ALL)\n\
(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
(declare-const a (Array Int C))\n\
(declare-const w Int)\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (<= 0 w 5))\n\
(assert (distinct (select a x) (select a y)))\n\
(assert (distinct (select a y) (select a z)))\n\
(assert (distinct (select a x) (select a z)))\n\
(assert (distinct (select a w) (select a x)))\n\
(assert (distinct (select a w) (select a y)))\n\
(check-sat)\n\
(get-model)\n";

const B08: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun f (Int) L)\n\
(declare-const a (Array Int L))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= (select a x) (f y)))\n\
(assert (= (f y) (cons 1 nil)))\n\
(assert (= (select a z) (cons 2 nil)))\n\
(assert (= (f z) (cons 3 nil)))\n\
(check-sat)\n\
(get-model)\n";

const B09: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-const a (Array Int U))\n\
(declare-const b (Array Int U))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= (select a x) (select b y)))\n\
(assert (distinct (select a y) (select b x)))\n\
(assert (distinct (select a x) (select a y)))\n\
(check-sat)\n\
(get-model)\n";

const B10: &str = "(set-logic ALL)\n\
(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
(declare-const a (Array (_ BitVec 2) C))\n\
(declare-const i (_ BitVec 2))\n\
(declare-const j (_ BitVec 2))\n\
(declare-const k (_ BitVec 2))\n\
(assert (= (select a i) red))\n\
(assert (= (select a j) green))\n\
(assert (= (select a k) blue))\n\
(check-sat)\n\
(get-model)\n";

const B11: &str = "(set-logic ALL)\n\
(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
(declare-datatypes ((P 0)) (((mk (px Int) (pc C)))))\n\
(declare-const a (Array Int P))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= (pc (select a x)) red))\n\
(assert (= (pc (select a y)) blue))\n\
(assert (= (px (select a x)) (px (select a y))))\n\
(check-sat)\n\
(get-model)\n";

const B13: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-const a (Array Int U))\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const u3 U)\n\
(assert (distinct u1 u2 u3))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(assert (= (store a 2 u3) a))\n\
(assert (= (select a x) u1))\n\
(assert (= (select a y) u2))\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn an_array_into_a_datatype_or_an_uninterpreted_sort_keeps_its_reads_apart() {
    for (name, script, checks) in [
        ("b01_pushpop", B01, 3),
        ("b02_three_reads_enum", B02, 1),
        ("b03_store_chain", B03, 1),
        ("b04_read_as_index", B04, 1),
        ("b07_pigeon_enum", B07, 1),
        ("b08_fun_and_array", B08, 1),
        ("b09_two_arrays_u", B09, 1),
        ("b10_bv_index_enum", B10, 1),
        ("b11_record_element", B11, 1),
        ("b13_u_store", B13, 1),
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
                "`{name}` check {check}: the printed model reads true at every assertion in scope\n{}",
                lines.join("\n")
            );
        }
    }
}
