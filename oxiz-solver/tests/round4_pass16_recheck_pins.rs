//! Round-4 adversarial recheck 16 (decision (75): a convergence pass) — pins
//! for what the recheck measured open on re-fix pass 16's final tree.
//!
//! Every script below is a corpus script of the round, or a line-deletion
//! reduction of one, quoted **verbatim**: recheck 15's `gen_dt.py` seed
//! 30093154 (`dt6`, its first 600 scripts), recheck 15's `gen14.py` seed
//! 30093155 (`g14b`), and this recheck's fresh `gen14.py` seed 30100164.  z3
//! 4.15.4 judged every verdict and every model the doc of a test names.
//!
//! # How to read this file
//!
//! * Re-fix pass 17 closed §1, §7 and §8 and inverted them (their names say
//!   what holds now); the other seven are holes as the recheck wrote them.
//! * A **HOLE** asserts what this tree answers today, together with every
//!   soundness guard that must hold whatever the tree answers (never the
//!   wrong verdict).  When the defect is fixed the assertion fails with **THE
//!   HOLE IS CLOSED** — invert the pin then (assert the right answer and, for
//!   a model, its replay), never delete it.
//! * Every model is judged by REPLAY: the published `define-fun` lines and one
//!   closed claim in a fresh script, as recheck 15's pins do.  A closed
//!   claim has no free symbol, so the replay decides it.
//! * No test installs a wall clock (decision (16)); every script decides in
//!   well under a second in the release probe.
//!
//! # A caveat every per-check comparison of the round inherits
//!
//! `Context::execute_script` parses the whole script before it runs a
//! command, so the terms of later commands are interned first: the answer to
//! one `check-sat` depends on the commands after it.  `gen_dt.py`'s `d00239`
//! prints a falsifying model at its first check in full and a correct one in
//! its one-check prefix; `d00552`'s first check is `unknown` in full and
//! `sat` in its prefix; `c702310` answers `gen14.py`'s `s00262` second check
//! `unsat` in full and `unknown` in its two-check prefix.  Deterministic, and
//! pre-existing; the pins below quote the text they measured, whole.

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

/// The `(get-model)` / `(get-value)` responses, in order: every line that
/// is not a verdict.
fn responses(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| !matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

/// The `define-fun` lines of one `(model …)` response, one per line.
fn definitions(model: &str) -> String {
    let mut out = String::new();
    for raw in model.lines() {
        let trimmed = raw.trim();
        if trimmed.starts_with("(define-fun ") {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    out
}

fn is_withheld(response: &str) -> bool {
    response.contains("model not certified")
}

/// Replay a published model: `prelude` (sorts, datatypes), the model's
/// `define-fun` lines, and one closed claim; the verdict.
fn replay(prelude: &str, definitions: &str, claim: &str) -> String {
    let script = format!("(set-logic ALL)\n{prelude}{definitions}(assert {claim})\n(check-sat)\n");
    verdicts(&run(&script))
        .last()
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// The sorts of every `gen_dt.py` script.
const DT_SORTS: &str = "(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) \
     ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n(declare-sort U 0)\n";

/// The declarations every `gen_dt.py` script carries after its sorts.
const DT_DECLS: &str = "(declare-fun h (Int) L)\n(declare-fun g (Int) C)\n(declare-fun k (L) Int)\n\
     (declare-fun q (C) Int)\n(declare-fun w (Int) U)\n(declare-fun v (L) L)\n\
     (declare-const x Int)\n(declare-const y Int)\n(declare-const z Int)\n\
     (declare-const l1 L)\n(declare-const l2 L)\n(declare-const l3 L)\n\
     (declare-const c1 C)\n(declare-const c2 C)\n(declare-const p P)\n\
     (declare-const u1 U)\n(declare-const u2 U)\n";

// ---------------------------------------------------------------------------
// §1. CLOSED (re-fix pass 17, decision (79)(a)) — was a REGRESSION of re-fix pass 16 (printed model,
//     quantifier-free, datatype): the first check published a model that falsifies its own second
//     assertion, where `c702310` and pass 14 print a model z3 confirms.
//
//     `gen_dt.py` seed 30093154 `d00239`, reduced by line deletion (the property: the tree's first model is
//     z3-falsifying).  The model prints `v` constantly `(cons 5 nil)` and `k` 5 at that point, so the left
//     side of `(= (cons (k (v nil)) l3) (h (+ 1 (- 1))))` is `(cons 5 l3)` while `h` 0 is a list headed by
//     2.  The decision (72)(a) net (`Context::withhold_a_falsifying_datatype_model`) withheld the second and
//     third checks' models and published the first.  Traced by re-fix pass 17: the exact evaluator has no
//     value for a datatype-sorted term, so the net decided only what the term builder folded while the
//     printed model was substituted in — at the first check `h`'s printed table closed the assertion to an
//     equality between an `ite` over `(+ 1 (- 1))` and a list (open: published), at the second and third to
//     two literal constructor applications (folded to `false`: withheld).  The net now decides every
//     comparison, `ite` and array read over values by value (`printed_eval::ground_values`) and withholds
//     the model at every check.  `TODO.md` `#P2b-88`'s churn decides which model the search finds; the
//     inverted pin asserts that none it prints is false.
// ---------------------------------------------------------------------------

const D00239_MIN: &str = "(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n\
(declare-fun h (Int) L)\n\
(declare-fun k (L) Int)\n\
(declare-fun q (C) Int)\n\
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
(assert (and (or (distinct l3 l2) (= l3 l2)) (and (= (cons (k (v nil)) l3) (h (+ 1 (- 1)))) (= (k l2) 1))))\n\
(assert (or (= l1 (h 1)) (distinct (k (cons 1 l1)) y)))\n\
(assert (distinct (h (k (h y))) (cons (q red) l3)))\n\
(check-sat)\n\
(get-model)\n\
(assert (= (h (px p)) l3))\n\
(check-sat)\n\
(get-model)\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn a_datatype_model_the_net_shows_false_is_withheld_at_every_check() {
    let lines = run(D00239_MIN);
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "sat", "sat"],
        "z3: sat at every check\n{}",
        lines.join("\n")
    );
    let prelude = "(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) \
         (blue)) ((mk (px Int) (pc C)))))\n(declare-sort U 0)\n";
    // Every assertion in scope at each check (the fifth from the second on).
    let first = "(and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4) (or (distinct l3 l2) (= l3 l2)) \
                 (= (cons (k (v nil)) l3) (h (+ 1 (- 1)))) (= (k l2) 1) (or (= l1 (h 1)) (distinct (k \
                 (cons 1 l1)) y)) (distinct (h (k (h y))) (cons (q red) l3)))";
    let later = format!("(and {first} (= (h (px p)) l3))");
    let printed = responses(&lines);
    assert_eq!(printed.len(), 3, "{}", lines.join("\n"));
    for (check, response) in printed.iter().enumerate() {
        if is_withheld(response) {
            continue;
        }
        let claim = if check == 0 {
            first.to_string()
        } else {
            later.clone()
        };
        let got = replay(
            prelude,
            // The uninterpreted-sort witnesses (`@uc_U_n`) are not symbols the
            // replay can read, and no assertion mentions `u1` / `u2`.
            &definitions(response)
                .lines()
                .filter(|line| !line.contains("@uc_"))
                .map(|line| format!("{line}\n"))
                .collect::<String>(),
            &claim,
        );
        assert_eq!(
            got,
            "sat",
            "check {check}: a printed model must satisfy every assertion (or be withheld)\n{}",
            lines.join("\n")
        );
    }
    // The first check's model is the one the net let through before; it is
    // withheld now, with the assertion it falsifies named.
    assert!(
        printed
            .first()
            .is_some_and(|response| is_withheld(response)),
        "the first model is withheld\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §2. HOLE — `#P2b-88`'s churn on a five-assertion goal: both models `c702310` and pass 14 print correctly
//     (z3) are withheld.
//
//     `gen_dt.py` seed 30093154 `d00507`, verbatim.  `c702310` and pass 14 print `k` constantly 1, `v`
//     constantly `(cons 0 nil)`, `q` constantly 0, `l3 = nil`, `p = (mk 0 red)`, `z = 0`; the tree withholds
//     both checks with `(error "model not certified: assertion 4 is false")`.  The oracle below replays
//     `c702310`'s model on the script's assertions: a model exists, and the one the search found is not the
//     one the printed model rebuilt (`TODO.md` `#P2b-88`).
// ---------------------------------------------------------------------------

#[test]
fn a_datatype_model_c702310_prints_is_withheld_at_both_checks() {
    let script = format!(
        "(set-logic ALL)\n{DT_SORTS}{DT_DECLS}\
         (assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
         (assert (not (distinct (+ 0 1) (k l1))))\n\
         (assert (or (not (= (v l2) (v l2))) (= (px p) z)))\n\
         (push 1)\n\
         (assert (= (v nil) (cons (q blue) l3)))\n\
         (check-sat)\n(get-model)\n(check-sat)\n(get-model)\n"
    );
    // The oracle: `c702310`'s model replays on every assertion.
    let head_model = "(define-fun x () Int 0)\n(define-fun y () Int 0)\n(define-fun z () Int 0)\n\
         (define-fun l1 () L nil)\n(define-fun l2 () L nil)\n(define-fun l3 () L nil)\n\
         (define-fun p () P (mk 0 red))\n(define-fun k ((x!0 L)) Int 1)\n\
         (define-fun q ((x!0 C)) Int 0)\n(define-fun v ((x!0 L)) L (cons 0 nil))\n";
    let claim = "(and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4) (not (distinct (+ 0 1) (k l1))) \
                 (or (not (= (v l2) (v l2))) (= (px p) z)) (= (v nil) (cons (q blue) l3)))";
    assert_eq!(
        replay(DT_SORTS, head_model, claim),
        "sat",
        "the oracle: c702310's model satisfies every assertion"
    );
    let lines = run(&script);
    assert_eq!(verdicts(&lines), vec!["sat", "sat"], "{}", lines.join("\n"));
    let printed = responses(&lines);
    if !printed.iter().all(|response| is_withheld(response)) {
        panic!(
            "THE HOLE IS CLOSED (`d00507`, `#P2b-88`): a model is printed again — invert this pin \
             and replay it on `{claim}`\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §3. HOLE — a REGRESSION of re-fix pass 16 (verdict, against pass 14 and pass 15): a satisfiable
//     quantifier-free datatype goal answers `unknown`.
//
//     `gen_dt.py` seed 30093154 `d00389`, its first check (the one-check prefix the round's `camp14.py`
//     writes; the full script answers the same).  z3, pass 14 and pass 15 answer `sat` (pass 14 in 127
//     conflicts), the tree `unknown` ("incomplete") after 405 conflicts; `c702310` `unknown` too.  In
//     recheck 16's isolated copy `OXIZ_MUT16_NO_DT_FOLD` (the selector / tester fold of `#P2b-82`) answers
//     `sat` again; no other switch does.  Not named in `TODO.md` decision (24a).
// ---------------------------------------------------------------------------

#[test]
fn a_satisfiable_datatype_goal_pass_fourteen_decides_answers_unknown() {
    let script = format!(
        "(set-logic ALL)\n{DT_SORTS}{DT_DECLS}\
         (assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
         (assert (distinct (+ (k l3) x) (k (ite ((_ is cons) l1) (tl l1) nil))))\n\
         (assert (or (= l1 l1) (= nil l1)))\n\
         (push 1)\n\
         (assert (and (< 0 1) (= p (mk (q c1) (pc p)))))\n\
         (assert (and (and ((_ is nil) (h 0)) (= l3 l1)) (or (< z (px p)) (= l2 l1))))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = run(&script);
    let got = verdicts(&lines).first().cloned().unwrap_or_default();
    assert_ne!(
        got,
        "unsat",
        "a WRONG unsat (z3: sat)\n{}",
        lines.join("\n")
    );
    if got != "unknown" {
        panic!(
            "THE HOLE IS CLOSED (`d00389`): answered `{got}` (pass 14 and z3: sat); invert this \
             pin and replay the model\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §4. HOLE — a REGRESSION of re-fix pass 16 (printed model, quantified): the third check's model, which
//     `c702310` and pass 15 print and z3 confirms, is withheld.
//
//     `gen14.py` seed 30093155 `s02298`, verbatim.  In recheck 16's isolated copy the model is printed
//     again with `OXIZ_MUT16_NO_DEFER_INT` (the deferred integrality of `#P2b-87`), with the take-back alone
//     switched off (`integrality_at_quantified_exit` answering `Integral`), and with pass 15's
//     `NO_MIXED_INT` or `NO_TIGHTEN`: the trajectory `#P2b-87`'s take-back gives this goal ends on a
//     candidate the certificate refuses.  Not named in `TODO.md` decision (24a).
// ---------------------------------------------------------------------------

#[test]
fn a_quantified_model_pass_fifteen_printed_is_withheld_after_the_integrality_take_back() {
    let script = "(set-logic ALL)\n(set-option :produce-models true)\n\
         (declare-const k Int)\n(declare-const j Int)\n(declare-fun f (Int) Int)\n(declare-fun g (Int) Int)\n\
         (declare-fun p (Int) Bool)\n(declare-const a (Array Int Int))\n(declare-const b (Array Int Int))\n\
         (assert (forall ((q Int)) (p (select a k))))\n(check-sat)\n(get-model)\n(get-value (j j))\n\
         (push 1)\n(assert (and (p 0) (distinct j k)))\n\
         (assert (=> (xor (distinct (select a (select a j)) (g (select a j))) (< j j)) (or (distinct k j) (p k))))\n\
         (assert (forall ((xx Int)) (=> (>= xx 0) (not (p k)))))\n(check-sat)\n(get-model)\n(get-value (3 k))\n\
         (push 1)\n\
         (assert (ite (distinct j j) (and (not (distinct (ite (distinct k k) k j) k)) (<= (select (store b 1 j) k) 1)) (not (not (= j (- 1))))))\n\
         (assert (forall ((xx Int)) (=> (< (select a xx) k) (=> (< 3 (f 0)) (p k)))))\n\
         (assert (exists ((q Int)) (= (select b j) j)))\n\
         (assert (exists ((xx Int)) (ite (ite (< (select a (ite (<= k 0) (- 1) j)) xx) (p xx) (not (p 2))) (and (= (select b k) j) (distinct 3 (- 1))) (ite (<= (f (+ (- 1) (- 1))) 7) (= (f 0) 5) (not (< (ite (= j xx) xx 5) (select b 5)))))))\n\
         (check-sat)\n(get-model)\n(get-value ((ite (= k 5) (g j) (- j 10)) j))\n";
    let lines = run(script);
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "sat", "sat"],
        "z3: sat at every check\n{}",
        lines.join("\n")
    );
    let models: Vec<String> = responses(&lines)
        .into_iter()
        .filter(|response| response.starts_with("(model") || is_withheld(response))
        .collect();
    let third = models.get(2).cloned().unwrap_or_default();
    if !is_withheld(&third) {
        panic!(
            "THE HOLE IS CLOSED (`s02298`): the third check prints a model again — invert this pin \
             and judge it\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §5. HOLE — a verdict `c702310` and pass 14 reach (`sat`, z3 `sat`) that the tree does not: a guarded
//     universal beside a bit-vector array read.
//
//     `gen14.py` seed 30093155 `s01487`, verbatim.  The tree answers `unknown` with 0 outer conflicts and 4
//     embedded bit-vector checks; `c702310` answers `sat` (its model is z3-falsifying), pass 14 `sat` with
//     the model withheld, pass 15 `unknown`.  In recheck 16's isolated copy pass 15's
//     `OXIZ_MUT15_NO_NEIGHBOURS` (`#P2b-75`'s guard neighbours) answers `sat`; no other switch does.  A
//     trajectory cost of `#P2b-75`'s soundness fix, not named in `TODO.md` decision (24a).
// ---------------------------------------------------------------------------

#[test]
fn a_guarded_universal_beside_a_bit_vector_read_answers_unknown() {
    let script = "(set-logic ALL)\n(set-option :produce-models true)\n\
         (declare-const k Int)\n(declare-const j Int)\n(declare-const m Int)\n(declare-fun f (Int) Int)\n\
         (declare-fun g (Int) Int)\n(declare-fun p (Int) Bool)\n(declare-const a (Array Int Int))\n\
         (declare-const x (_ BitVec 7))\n(declare-const y (_ BitVec 7))\n\
         (declare-fun h ((_ BitVec 7)) (_ BitVec 7))\n(declare-const c (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (assert (not (distinct j m)))\n\
         (assert (forall ((xx Int)) (=> (< xx 0) (and (= (ite (< k j) y x) (h x)) (<= (g 10) 3)))))\n\
         (assert (xor (= (- 1) k) (and (= (h (ite (<= j 10) (_ bv119 7) (_ bv101 7))) (select c (select c y))) (= m k))))\n\
         (check-sat)\n(get-model)\n(get-value (m (select a (select a 1)) y))\n";
    let lines = run(script);
    let got = verdicts(&lines).first().cloned().unwrap_or_default();
    assert_ne!(
        got,
        "unsat",
        "a WRONG unsat (z3: sat)\n{}",
        lines.join("\n")
    );
    if got != "unknown" {
        panic!(
            "THE HOLE IS CLOSED (`s01487`): answered `{got}` (c702310, pass 14 and z3: sat); invert \
             this pin\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §6. HOLE — `#P2b-81` (open, no pin before this one): a quantifier-free model that falsifies its own
//     script, two arrays and one select equality.
//
//     This recheck's fresh `gen14.py` seed 30100164 `s00635`, verbatim.  The tree and `c702310` print
//     `m = 0`, `a = (store (store ((as const (Array Int Int)) 0) 7 0) 0 1)`, `b = (store ((as const (Array
//     Int Int)) 0) 0 1)`: `a` and `b` are the same array, so `(not (= b a))` is false, and `(select b m)` is 1
//     where `(select a 7)` is 0 — the third assertion is false (z3: falsifying).  `c4b04b7` prints a
//     model with `b` apart from `a`.
// ---------------------------------------------------------------------------

#[test]
fn two_arrays_the_third_assertion_keeps_apart_print_as_one() {
    let script = "(set-logic ALL)\n(set-option :produce-models true)\n\
         (declare-const k Int)\n(declare-const j Int)\n(declare-const m Int)\n\
         (declare-fun f (Int) Int)\n(declare-fun g (Int) Int)\n\
         (declare-const a (Array Int Int))\n(declare-const b (Array Int Int))\n\
         (assert (<= m 5))\n\
         (assert (xor (and (not (<= j (- 3))) (not (= (- k 5) (- 1)))) (not (<= (+ j 2) 3))))\n\
         (assert (or (< j m) (or (not (= b a)) (= (select b m) (select a 7)))))\n\
         (check-sat)\n(get-model)\n\
         (get-value ((ite (= (select b k) 1) (ite (distinct k k) 7 k) (f j)) (ite (= j 1) (ite (= k k) m m) k)))\n";
    let lines = run(script);
    assert_eq!(verdicts(&lines), vec!["sat"], "{}", lines.join("\n"));
    let model = responses(&lines).first().cloned().unwrap_or_default();
    if is_withheld(&model) {
        panic!(
            "THE HOLE IS CLOSED (`s00635`, `#P2b-81`): the model is withheld now; invert this pin\n{}",
            lines.join("\n")
        );
    }
    let claim = "(and (<= m 5) (xor (and (not (<= j (- 3))) (not (= (- k 5) (- 1)))) (not (<= (+ j 2) \
                 3))) (or (< j m) (or (not (= b a)) (= (select b m) (select a 7)))))";
    let got = replay("", &definitions(&model), claim);
    if got == "sat" {
        panic!(
            "THE HOLE IS CLOSED (`s00635`, `#P2b-81`): the model replays now; invert this pin\n{}",
            lines.join("\n")
        );
    }
    assert_eq!(
        got,
        "unsat",
        "the model falsifies its script (HOLE)\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §7. CLOSED (re-fix pass 17, decision (79)(e)) — recheck 15's minor 11, half open until then: on a
//     quantified goal, `(get-value)` of a read of a constant array whose default the script spells as a
//     negated numeral answered a non-value.
//
//     Every build (`c702310`, pass 14, pass 16) answered `(select ((as const (Array Int Int)) (- 3)) (f 0))`
//     for `(select ((as const (Array Int Int)) (- 3)) (f j))`, while the same read of a default `3` answered
//     `3` and a quantifier-free goal answered `-3` (z3: `(- 3)`).  `recheck14`'s `g14a` echoed six such
//     reads (`s00428`, `s00457`, `s00526`, `s00986`, `s02141`, `s02551`; `s01467`'s echo is an array
//     EQUALITY against such a constant, which still echoes — `TODO.md` `#P2b-81`).  The exact evaluator read
//     an array constant's default only when it was a literal (`model_eval::open::store_chain`); a negated
//     numeral is one now, and the read answers `-3`.
// ---------------------------------------------------------------------------

#[test]
fn a_negated_numeral_default_read_on_a_quantified_goal_answers_its_value() {
    let script = "(set-logic ALL)\n(declare-fun f (Int) Int)\n(declare-const j Int)\n\
         (assert (forall ((x Int)) (>= (f x) 0)))\n(check-sat)\n\
         (get-value ((select ((as const (Array Int Int)) (- 3)) (f j)) (select ((as const (Array Int Int)) 3) (f j))))\n";
    let lines = run(script);
    assert_eq!(verdicts(&lines), vec!["sat"], "{}", lines.join("\n"));
    let answer = responses(&lines).join("\n");
    assert!(
        answer.contains("((select ((as const (Array Int Int)) 3) (f j)) 3)"),
        "the positive default folds\n{answer}"
    );
    assert!(
        answer.contains("((select ((as const (Array Int Int)) (- 3)) (f j)) -3)"),
        "the negated default's read answers its value\n{answer}"
    );
}

// ---------------------------------------------------------------------------
// §8. CLOSED (re-fix pass 17, decision (79)(b), `TODO.md` `#P2b-89`) — a quantifier-free model that
//     FALSIFIED its script on every build (`c702310`, pass 14, pass 16): an ARRAY whose element sort is a
//     datatype, an enumeration or an uninterpreted sort, read at two indices the arithmetic left free,
//     printed one entry at one index.
//
//     Recheck 15's `atk/dt/d05.smt2`, verbatim (the list range), its enumeration twin and its
//     uninterpreted-sort twin (recheck 16's `atk/arr/a_enum`, `a_usort`).  `x, y ∈ [0, 5]`, `a[x] = [1]`,
//     `a[y] = [2]`: every earlier build printed `x = y = 0` beside one entry.  The array refinement's
//     read-congruence family (`Solver::build_index_congruence`) compared two reads by their `EvalVal`,
//     which a datatype or uninterpreted read has none of, so the violated lemma `x = y ⇒ a[x] = a[y]` was
//     never built; it now compares such reads by the value their class prints, keyed as `#P2b-84` keys a
//     function's results.  The uninterpreted twin needed the printer too: a read of an uninterpreted sort
//     has no model entry, so its array printed as one constant; it prints its class witnesses now
//     (`array_model`'s renderer and reader alike).  The `Int`-element twin was correct throughout.
// ---------------------------------------------------------------------------

#[test]
fn an_array_into_a_datatype_read_at_two_free_indices_keeps_both_entries() {
    let bounds = "(declare-const x Int)\n(declare-const y Int)\n\
         (assert (>= x 0))\n(assert (<= x 5))\n(assert (>= y 0))\n(assert (<= y 5))\n";
    for (name, sorts, array, facts, claim) in [
        (
            "d05_list_element",
            "(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n",
            "(declare-const a (Array Int L))\n",
            "(assert (= (select a x) (cons 1 nil)))\n(assert (= (select a y) (cons 2 nil)))\n",
            "(and (= (select a x) (cons 1 nil)) (= (select a y) (cons 2 nil)) (<= 0 x 5) (<= 0 y 5))",
        ),
        (
            "enumeration_element",
            "(declare-datatypes ((C 0)) (((red) (green) (blue))))\n",
            "(declare-const a (Array Int C))\n",
            "(assert (= (select a x) red))\n(assert (= (select a y) green))\n",
            "(and (= (select a x) red) (= (select a y) green) (<= 0 x 5) (<= 0 y 5))",
        ),
        (
            "uninterpreted_element",
            "(declare-sort U 0)\n",
            "(declare-const a (Array Int U))\n(declare-const u1 U)\n(declare-const u2 U)\n\
             (assert (distinct u1 u2))\n",
            "(assert (= (select a x) u1))\n(assert (= (select a y) u2))\n",
            "(and (distinct u1 u2) (= (select a x) u1) (= (select a y) u2) (<= 0 x 5) (<= 0 y 5))",
        ),
    ] {
        let script =
            format!("(set-logic ALL)\n{sorts}{array}{bounds}{facts}(check-sat)\n(get-model)\n");
        let lines = run(&script);
        assert_eq!(
            verdicts(&lines),
            vec!["sat"],
            "`{name}`\n{}",
            lines.join("\n")
        );
        let model = responses(&lines).first().cloned().unwrap_or_default();
        assert!(
            model.starts_with("(model"),
            "`{name}`: a model is printed\n{}",
            lines.join("\n")
        );
        // An `@uc_U_n` witness is not a symbol the replay can read: each one
        // becomes a constant of `U`, the witnesses pairwise distinct (what
        // the printed model says).
        let mut witnesses: Vec<String> = Vec::new();
        for word in model.split(|c: char| c.is_whitespace() || c == '(' || c == ')') {
            if let Some(rest) = word.strip_prefix("@uc_")
                && !witnesses.iter().any(|w| w == rest)
            {
                witnesses.push(rest.to_string());
            }
        }
        let mut prelude = sorts.to_string();
        for witness in &witnesses {
            prelude.push_str(&format!("(declare-const uc_{witness} U)\n"));
        }
        if witnesses.len() > 1 {
            let names: Vec<String> = witnesses.iter().map(|w| format!("uc_{w}")).collect();
            prelude.push_str(&format!("(assert (distinct {}))\n", names.join(" ")));
        }
        let got = replay(&prelude, &definitions(&model).replace("@uc_", "uc_"), claim);
        assert_eq!(
            got,
            "sat",
            "`{name}`: the printed model satisfies its script\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §9. HOLE — a quantifier-free mixed `Int` / `Real` goal `c702310` and pass 14 answer `sat` (z3 `sat`)
//     answers `unknown` at its second check: a cost of `#P2b-79`'s integrality (pass 15), not named in
//     `TODO.md` decision (24a).
//
//     This recheck's fresh `gen_mix.py` seed 30100161 `m00666`, verbatim (no logic set, no quantifier).
//     Pass 15's tree answers the same; in recheck 16's isolated copy pass 15's `OXIZ_MUT15_NO_MIXED_INT`,
//     `NO_TIGHTEN`, `NO_MIXED_EQ` or `NO_UFC` answers `sat` again.  (`c702310`'s models are z3-falsifying:
//     the verdict was right and the model was not.)
// ---------------------------------------------------------------------------

#[test]
fn a_mixed_integer_goal_c702310_decides_answers_unknown_at_its_second_check() {
    let script = "(declare-const x0 Int)\n(declare-const x1 Int)\n(declare-const x2 Int)\n(declare-const x3 Int)\n\
         (declare-fun f (Int) Int)\n\
         (assert (and (>= x0 (- 6)) (<= x0 6)))\n(assert (and (>= x1 (- 6)) (<= x1 6)))\n\
         (assert (and (>= x2 (- 6)) (<= x2 6)))\n(assert (and (>= x3 (- 6)) (<= x3 6)))\n\
         (assert (or (or (= (+ (* 2.0 (to_real (f -1))) (* 0.25 (to_real x3)) (* 3.0 (to_real (f x0)))) 1.5) (< (+ (* (- 1) x0) (* 1 x0) (* 1 x0)) (+ (* 2 (f x1)) (* 3 x1) (* (- 2) (f x0))))) (<= (+ (* 3 x3) (* 2 x2) (* 2 (f x1))) (* (- 2) x0))))\n\
         (assert (or (distinct (+ (* 1 x0) (* 1 x0)) (+ (* (- 2) x2) (* (- 1) x1) (* 1 x3))) (or (< (+ (* (- 1) (f x3)) (* 3 x1) (* 1 (f -2))) (- 1)) (= (+ (* 0.25 (to_real x0)) (* 2.0 (to_real x2))) (+ (* 3.0 (to_real x0)) (* (- 1.5) (to_real x0)) (* (- 1.5) (to_real (f x2))))))))\n\
         (assert (and (< (+ (* 0.5 (to_real x1)) (* 0.5 (to_real x0))) 0.0) (> (+ (* 2.0 (to_real (f x0))) (* (- 1.5) (to_real x0)) (* 0.5 (to_real x3))) (* 2.0 (to_real (ite (distinct (to_real x3) 2.25) x2 x0))))))\n\
         (push 1)\n(assert (> (+ (* 2 x2) (* 1 x0) (* 1 x0)) 2))\n(check-sat)\n(get-model)\n\
         (push 1)\n\
         (assert (and (and (> (* 3 x2) 0) (< (+ (* 2.0 (to_real (ite (= x2 x0) x1 x3))) (* 3.0 (to_real x2))) 0.0)) (< (+ (* (- 2) (f -2)) (* 3 (ite (< (to_real x1) (- 1.5)) x1 x0))) 5)))\n\
         (check-sat)\n(get-model)\n";
    let lines = run(script);
    let got = verdicts(&lines);
    assert_eq!(
        got.first().map(String::as_str),
        Some("sat"),
        "{}",
        lines.join("\n")
    );
    let second = got.get(1).cloned().unwrap_or_default();
    assert_ne!(
        second,
        "unsat",
        "a WRONG unsat (z3: sat)\n{}",
        lines.join("\n")
    );
    if second != "unknown" {
        panic!(
            "THE HOLE IS CLOSED (`m00666`): the second check answered `{second}` (c702310 and z3: \
             sat); invert this pin and replay the model\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §10. HOLE — a quantified array goal pass 14 refutes answers `unknown`: a cost of `#P2b-75`'s
//      eligibility-first certifier (pass 15), not named in `TODO.md` decision (24a).
//
//      This recheck's fresh `fuzz_qc.py` seed 30100165 script 982, verbatim.  `a1` is `true` at every
//      point and `false` at every point but `k0`, over 128 points: `unsat` (pass 14 `unsat`; z3 `unsat`).
//      `c702310`, pass 15 and the tree answer `unknown`; in recheck 16's isolated copy pass 15's
//      `OXIZ_MUT15_NO_ELIG_FIRST` answers `unsat` again.  The same seed's script 2751 loses its second
//      check (`sat` → `unknown`, pass 14 `sat`) with no switch restoring it.
// ---------------------------------------------------------------------------

#[test]
fn a_bool_array_true_everywhere_and_false_but_at_one_point_answers_unknown() {
    let script = "(set-logic ALL)\n\
         (declare-const a0 (Array (_ BitVec 7) Bool))\n(declare-const a1 (Array (_ BitVec 7) Bool))\n\
         (declare-const a2 (Array (_ BitVec 7) Bool))\n(declare-const a3 (Array (_ BitVec 7) Bool))\n\
         (declare-const k0 (_ BitVec 7))\n(declare-const k1 (_ BitVec 7))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a1 i) true)))\n\
         (assert (forall ((i (_ BitVec 7))) (=> (distinct i k0) (= (select a1 i) false))))\n\
         (assert (= (select a1 k1) true))\n\
         (assert (forall ((i (_ BitVec 7))) (=> (bvuge i (_ bv86 7)) (= (select a0 i) (select a3 i)))))\n\
         (check-sat)\n(get-model)\n";
    let lines = run(script);
    let got = verdicts(&lines).first().cloned().unwrap_or_default();
    assert_ne!(got, "sat", "a WRONG sat (z3: unsat)\n{}", lines.join("\n"));
    if got != "unknown" {
        panic!(
            "THE HOLE IS CLOSED (`qc 30100165/982`): answered `{got}` (pass 14 and z3: unsat); \
             invert this pin\n{}",
            lines.join("\n")
        );
    }
}
