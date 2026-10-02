//! Round-4 re-fix pass 17 (decision (79)) — pins for what it fixed beside
//! adversarial recheck 16's own pins (`round4_pass16_recheck_pins.rs`, §1, §7
//! and §8 inverted there).
//!
//! * Decision (79)(a): the quantifier-free net withholds every printed model
//!   it can show false, at EVERY check, and runs wherever a declared symbol's
//!   sort CONTAINS a datatype, an enumeration or an uninterpreted sort — a
//!   function's range included.  Recheck 16's fresh `gen_dt.py` seed
//!   30100162 `d00290` printed a model falsifying its second assertion at its
//!   third check, where `c702310` and pass 14 print one z3 confirms.
//!
//! Every printed model is judged by EXACT EVALUATION on every assertion in
//! scope (`support/dt_eval.rs`, re-fix pass 18; it was a replay through this
//! solver, which can answer `unknown` on a correct model's closed script).
//! No test installs a wall clock (decision (16)).

use oxiz_solver::Context;

// The exact evaluator of the round-4 datatype pins (re-fix pass 18).
#[path = "support/dt_eval.rs"]
mod dt_eval;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

/// Recheck 16's fresh `gen_dt.py` seed 30100162 `d00290`, verbatim.
const D00290: &str = "(set-logic ALL)\n\
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
(assert (or (or (= (w (px p)) u2) (= (ite ((_ is cons) (h 2)) (tl (h 2)) nil) l3)) (or (= p (mk (ite ((_ is cons) (h y)) (hd (h y)) 0) (g 2))) (= (w z) u2))))\n\
(assert (or (and (distinct (+ (q c2) x) y) (distinct (cons 1 nil) (cons x nil))) (distinct (h (+ 0 x)) l1)))\n\
(assert (not (not (< (k l3) z))))\n\
(assert (or (<= (+ (k nil) x) (k l1)) (= (g (k (v l2))) c1)))\n\
(push 1)\n\
(assert (= (g z) blue))\n\
(check-sat)\n\
(get-model)\n\
(pop 1)\n\
(check-sat)\n\
(get-model)\n\
(push 1)\n\
(assert (distinct (w (ite ((_ is cons) (cons 1 nil)) (hd (cons 1 nil)) 0)) u1))\n\
(check-sat)\n\
(get-model)\n";

/// `d00290`: `unknown`, `unknown`, then `sat`; pass 16 printed a model whose
/// second assertion is false at the third check (`c702310`, pass 14 and pass
/// 15 printed one z3 confirms).  Whatever model the search finds, a printed
/// one satisfies every assertion in scope, and a model the net shows false is
/// withheld with the assertion it falsifies named.
#[test]
fn a_falsifying_datatype_model_at_a_third_check_is_withheld() {
    let lines = run(D00290);
    let got = verdicts(&lines);
    assert_eq!(got.len(), 3, "{}", lines.join("\n"));
    assert!(
        got.iter().all(|verdict| verdict != "unsat"),
        "z3: sat at every check\n{}",
        lines.join("\n")
    );
    // Re-fix pass 18: every printed model is judged by EXACT EVALUATION
    // (`support/dt_eval.rs`), not by a replay through this solver — the
    // replay of a correct model of the first check answered `unknown`
    // (`:reason-unknown incomplete`) where z3 confirms it, so the replay could
    // not tell a false model from an incompleteness.
    let judged = dt_eval::judge(D00290, &lines);
    assert_eq!(judged.len(), 3, "{}", lines.join("\n"));
    for (check, (verdict, reading)) in judged.iter().enumerate() {
        if verdict != "sat" {
            continue;
        }
        assert!(
            matches!(
                reading,
                dt_eval::ModelReading::Holds | dt_eval::ModelReading::Withheld
            ),
            "check {check}: a printed model satisfies every assertion in scope ({reading:?})\n{}",
            lines.join("\n")
        );
    }
}
