//! Re-fix pass 18 (decisions (84), (85), (86); `TODO.md` `#P2b-90`, `#P2b-88`,
//! `#P2b-89`) — the families this pass closed, pinned by construction.
//!
//! * §1, `#P2b-90` (SOUNDNESS, every build since 0.3.3): a tester, a selector
//!   or a datatype equality applied to a datatype-sorted `ite` was not tied to
//!   the `ite`'s value — the datatype lemmas were encoded over the `ite` as
//!   written while the assertion read the encoder's proxy for it.  Every
//!   position the recheck named (a tester under an arithmetic `ite`, nested
//!   `ite`s, an `ite` under an uninterpreted or a constructor argument, both
//!   sides of an equality, push / pop, a quantifier body, an enumeration and
//!   an uninterpreted sort) is decided; z3 4.15.4 gives every verdict below.
//! * §2, `#P2b-89` one level down (decision (86)): an array of arrays — and a
//!   function into an array — over a datatype, an enumeration or an
//!   uninterpreted sort prints a model that holds; `n11`, the same shape over
//!   `Int`, is `#P2b-81`'s and is pinned as the hole it is.
//! * §3, the honesty net (decision (85)): a quantifier-free `sat` whose
//!   candidate model the exact value reader reads false answers `unknown`,
//!   names the assertion, and publishes no model.
//!
//! Every printed model is judged by EXACT EVALUATION (`support/dt_eval.rs`),
//! never by a replay through this solver.  No test installs a wall clock
//! (decision (16)).

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
// The scripts (re-fix pass 18's `$R/fix18/atk/fam/` and `$R/fix18/atk/nest/`).
// ---------------------------------------------------------------------------

const F01: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert c)\n\
(assert (= l1 (cons 1 (ite c l1 nil))))\n\
(check-sat)\n";

const F02: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (= l1 (cons 1 nil)))\n\
(assert (distinct (k (ite c l1 nil)) (k nil)))\n\
(check-sat)\n";

const F03: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (= l1 (cons 1 nil)))\n\
(assert (= (hd (ite c nil l1)) 2))\n\
(check-sat)\n";

const F04: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert d)\n\
(assert (= l1 (cons 1 nil)))\n\
(assert (= (ite c l1 nil) (ite d l1 l2)))\n\
(assert (= l2 nil))\n\
(check-sat)\n";

const F05: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (= (k l1) 3))\n\
(assert (= l1 nil))\n\
(assert (= 3 (k (tl (cons 2 (ite c (cons 5 nil) l1))))))\n\
(assert ((_ is cons) (tl (cons 2 (ite c (cons 5 nil) l1)))))\n\
(check-sat)\n";

const F06: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (= l1 (ite c (cons 1 l1) (cons 2 l1))))\n\
(check-sat)\n";

const F07: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(push 1)\n\
(assert c)\n\
(assert ((_ is nil) (ite c (cons 1 l1) l2)))\n\
(check-sat)\n\
(pop 1)\n\
(assert (not c))\n\
(assert ((_ is nil) (ite c (cons 1 l1) l2)))\n\
(assert ((_ is cons) l2))\n\
(check-sat)\n";

const F08: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun k (L) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (forall ((x Int)) (=> (> x 0) ((_ is cons) (ite c l1 (cons x nil))))))\n\
(assert ((_ is nil) (ite c l1 (cons 1 nil))))\n\
(check-sat)\n";

const F09: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (= c1 red))\n\
(assert (< 0 (ite ((_ is blue) (ite c blue c1)) 1 0)))\n\
(check-sat)\n";

const F10: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert c)\n\
(assert (not d))\n\
(assert (= (ite c (ite d red green) blue) c1))\n\
(assert ((_ is red) c1))\n\
(check-sat)\n";

const F11: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (distinct u1 u2))\n\
(assert (= (w u2) 3))\n\
(assert (distinct (w (ite c u1 u2)) 3))\n\
(check-sat)\n";

const F12: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (= l1 (cons 1 (cons 2 nil))))\n\
(assert ((_ is nil) (tl (ite c nil l1))))\n\
(check-sat)\n";

const F13: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert c)\n\
(assert d)\n\
(assert (= l1 nil))\n\
(assert (= (ite c (cons 1 l1) l2) (ite d (cons 2 l1) l2)))\n\
(check-sat)\n";

const F14: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert (not c))\n\
(assert (= l2 (cons 3 (ite c l1 nil))))\n\
(assert ((_ is cons) (tl l2)))\n\
(check-sat)\n";

const F15: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(assert c)\n\
(assert (= l1 (cons 1 nil)))\n\
(assert (< 0 (ite ((_ is cons) (ite c l1 nil)) (hd (ite c l1 nil)) 0)))\n\
(check-sat)\n\
(get-model)\n";

const F16: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-sort U 0)\n\
(declare-fun k (L) Int)\n\
(declare-fun g (C) Int)\n\
(declare-fun w (U) Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const c1 C)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-const c Bool)\n\
(declare-const d Bool)\n\
(push 1)\n\
(assert c)\n\
(assert ((_ is green) (ite c (ite d red green) c1)))\n\
(assert d)\n\
(check-sat)\n\
(pop 1)\n\
(assert (not c))\n\
(assert ((_ is green) (ite c red c1)))\n\
(check-sat)\n\
(get-model)\n";

const N01: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int C)))\n\
(assert (= (select (select a x) 0) red))\n\
(assert (= (select (select a y) 0) green))\n\
(check-sat)\n\
(get-model)\n";

const N02: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int L)))\n\
(assert (= (select (select a x) 1) (cons 1 nil)))\n\
(assert (= (select (select a y) 1) nil))\n\
(check-sat)\n\
(get-model)\n";

const N03: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int U)))\n\
(declare-const u1 U) (declare-const u2 U) (declare-const u3 U)\n\
(assert (distinct u1 u2 u3))\n\
(assert (= (select (select a x) 0) u1))\n\
(assert (= (select (select a y) 0) u2))\n\
(assert (= (select (select a z) 0) u3))\n\
(check-sat)\n\
(get-model)\n";

const N04: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int U)))\n\
(declare-const u1 U) (declare-const u2 U)\n\
(assert (distinct u1 u2))\n\
(assert (= (select (select a 0) x) u1))\n\
(assert (= (select (select a 0) y) u2))\n\
(check-sat)\n\
(get-model)\n";

const N05: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int (Array Int C))))\n\
(assert (= (select (select (select a x) 0) 0) red))\n\
(assert (= (select (select (select a y) 0) 0) blue))\n\
(check-sat)\n\
(get-model)\n";

const N06: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array U (Array Int C)))\n\
(declare-const u1 U) (declare-const u2 U)\n\
(assert (= (select (select a u1) 0) red))\n\
(assert (= (select (select a u2) 0) blue))\n\
(check-sat)\n\
(get-model)\n";

const N07: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int C)))\n\
(declare-const b (Array Int (Array Int C)))\n\
(assert (= b (store a 3 (store (select a 3) 0 blue))))\n\
(assert (= (select (select b x) 0) red))\n\
(assert (= (select (select a y) 0) green))\n\
(assert (= x 3))\n\
(check-sat)\n\
(get-model)\n";

const N08: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int C)))\n\
(declare-const r (Array Int C))\n\
(assert (= (select a x) r))\n\
(assert (= (select r 0) red))\n\
(assert (= (select (select a y) 0) green))\n\
(check-sat)\n\
(get-model)\n";

const N09: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int U)))\n\
(declare-const u1 U) (declare-const u2 U)\n\
(assert (distinct u1 u2))\n\
(assert (= x y))\n\
(assert (= (select (select a x) 0) u1))\n\
(assert (= (select (select a y) 0) u2))\n\
(check-sat)\n\
(get-model)\n";

const N10: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int C)))\n\
(push 1)\n\
(assert (= (select (select a x) 0) red))\n\
(assert (= (select (select a y) 0) green))\n\
(check-sat)\n\
(get-model)\n\
(pop 1)\n\
(assert (= (select (select a x) 2) blue))\n\
(assert (= (select (select a z) 2) red))\n\
(check-sat)\n\
(get-model)\n";

const N11: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int Int)))\n\
(assert (= (select (select a x) 0) 1))\n\
(assert (= (select (select a y) 0) 2))\n\
(check-sat)\n\
(get-model)\n";

const N12: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array C L)))\n\
(assert (= (select (select a x) red) (cons 1 nil)))\n\
(assert (= (select (select a y) red) (cons 2 nil)))\n\
(check-sat)\n\
(get-model)\n";

const N13: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int U)))\n\
(declare-const u1 U) (declare-const u2 U)\n\
(assert (distinct u1 u2))\n\
(assert (= (select (select a x) 0) u1))\n\
(assert (= (select (select a y) 0) u2))\n\
(check-sat)\n\
(get-value ((select (select a x) 0) (select (select a y) 0) x y))\n\
(check-sat)\n\
(get-model)\n";

const N14: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array Int (Array Int C)))\n\
(declare-const w Int) (assert (<= 0 w 5))\n\
(assert (distinct (select (select a x) 0) (select (select a y) 0) (select (select a z) 0)))\n\
(assert (distinct (select (select a w) 0) (select (select a x) 0)))\n\
(assert (distinct (select (select a w) 0) (select (select a y) 0)))\n\
(check-sat)\n\
(get-model)\n";

const N15: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-const a (Array (_ BitVec 2) (Array Int C)))\n\
(declare-const i (_ BitVec 2)) (declare-const j (_ BitVec 2)) (declare-const k (_ BitVec 2))\n\
(assert (= (select (select a i) 0) red))\n\
(assert (= (select (select a j) 0) green))\n\
(assert (= (select (select a k) 0) blue))\n\
(check-sat)\n\
(get-model)\n";

const N16: &str = "(set-logic ALL)\n\
(declare-sort U 0)\n\
(declare-datatypes ((L 0) (C 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue))))\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(assert (<= 0 x 5))\n\
(assert (<= 0 y 5))\n\
(assert (<= 0 z 5))\n\
(declare-fun f (Int) (Array Int C))\n\
(assert (= (select (f x) 0) red))\n\
(assert (= (select (f y) 0) green))\n\
(check-sat)\n\
(get-model)\n";

// ---------------------------------------------------------------------------
// §1. `#P2b-90`: testers, selectors and datatype equalities over datatype `ite`s.
//
//     `F01` a cycle through an `ite` under a constructor argument; `F02` an `ite` under an uninterpreted
//     argument; `F03` a selector over an `ite`; `F04` an equality of two `ite`s; `F05` a tester under an
//     uninterpreted argument's selector chain; `F06` a size cycle through both branches; `F07` push / pop;
//     `F08` a tester over an `ite` in a quantifier body and its ground instance; `F09` an enumeration
//     tester under an arithmetic `ite`; `F10` an enumeration equality over a nested `ite`; `F11` an
//     uninterpreted-sort `ite` under a function; `F12` a tester over a selector over an `ite`; `F13` both
//     sides `ite`s of constructor applications; `F14` an `ite` inside a constructor argument read back by
//     a selector; `F15` (sat) the control; `F16` an enumeration `ite` around push / pop.  On pass 17's
//     tree `F03`, `F05`, `F07`, `F08` answered `unknown`; each answer below is z3's.
// ---------------------------------------------------------------------------

#[test]
fn testers_selectors_and_equalities_over_datatype_ites_are_decided() {
    for (name, script, expected) in [
        ("f01", F01, vec!["unsat"]),
        ("f02", F02, vec!["unsat"]),
        ("f03", F03, vec!["unsat"]),
        ("f04", F04, vec!["unsat"]),
        ("f05", F05, vec!["unsat"]),
        ("f06", F06, vec!["unsat"]),
        ("f07", F07, vec!["unsat", "unsat"]),
        ("f08", F08, vec!["unsat"]),
        ("f09", F09, vec!["unsat"]),
        ("f10", F10, vec!["unsat"]),
        ("f11", F11, vec!["unsat"]),
        ("f12", F12, vec!["unsat"]),
        ("f13", F13, vec!["unsat"]),
        ("f14", F14, vec!["unsat"]),
        ("f15", F15, vec!["sat"]),
        ("f16", F16, vec!["unsat", "sat"]),
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
                assert_eq!(
                    reading,
                    &ModelReading::Holds,
                    "`{name}` check {check}: the printed model reads true\n{}",
                    lines.join("\n")
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// §2. Arrays of arrays (and a function into an array) over a datatype, an enumeration or an
//     uninterpreted sort (decision (86)).  `N01` enumeration, `N02` list, `N03` three reads into `U`,
//     `N04` the free index inside, `N05` three levels, `N06` a `U` index, `N07` (unsat) a store of a
//     store, `N08` a read equated with an array constant, `N09` (unsat) the reads at one index, `N10`
//     push / pop, `N12` a datatype element under an enumeration index, `N13` `(get-value)` of the reads,
//     `N14` a pigeonhole over three reads, `N15` a bit-vector outer index, `N16` a function into
//     `(Array Int C)`.  On pass 17's tree `N01`-`N03`, `N05`, `N08`, `N10`, `N12`, `N13`, `N16` printed a
//     model that falsifies the script (z3), `N14` withheld it.
// ---------------------------------------------------------------------------

#[test]
fn arrays_of_arrays_over_value_sorts_print_models_that_hold() {
    for (name, script, expected) in [
        ("n01", N01, vec!["sat"]),
        ("n02", N02, vec!["sat"]),
        ("n03", N03, vec!["sat"]),
        ("n04", N04, vec!["sat"]),
        ("n05", N05, vec!["sat"]),
        ("n06", N06, vec!["sat"]),
        ("n07", N07, vec!["unsat"]),
        ("n08", N08, vec!["sat"]),
        ("n09", N09, vec!["unsat"]),
        ("n10", N10, vec!["sat", "sat"]),
        ("n12", N12, vec!["sat"]),
        ("n13", N13, vec!["sat", "sat"]),
        ("n14", N14, vec!["sat"]),
        ("n15", N15, vec!["sat"]),
        ("n16", N16, vec!["sat"]),
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
                assert!(
                    matches!(reading, ModelReading::Holds | ModelReading::NoModel),
                    "`{name}` check {check}: {reading:?}, the printed model reads true\n{}",
                    lines.join("\n")
                );
            }
        }
    }
}

/// HOLE (`TODO.md` `#P2b-81`, every build; outside decision (86), whose key
/// covers element sorts that contain a datatype, an enumeration or an
/// uninterpreted sort): an array of arrays of `Int` read at two free indices
/// prints `x = y = 0` beside one entry, so `a[y][0] = 2` is false (z3:
/// falsifying; the script is `sat`).  A quantifier-free model over integers
/// and integer arrays is not checked (README).  When it is fixed, invert this
/// pin to assert the model reads true.
#[test]
fn an_int_only_array_of_arrays_still_prints_one_entry() {
    let lines = run(N11);
    let got = judge(N11, &lines);
    assert_eq!(
        got.iter().map(|(v, _)| v.as_str()).collect::<Vec<_>>(),
        vec!["sat"],
        "{}",
        lines.join("\n")
    );
    assert!(
        matches!(got.first(), Some((_, ModelReading::Falsifies(_)))),
        "the Int-only twin of `b05` now prints a model that holds — invert this pin\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §3. The honesty net (decision (85)): a quantifier-free `sat` whose candidate model the exact value
//     reader reads false answers `unknown`, names the assertion in `(get-info :reason-unknown)` and in
//     `(get-model)`'s error, and publishes no model — the verdict is only as good as the model behind it.
//
//     `D00085` and `D00087` are adversarial recheck 17's fresh `gen_dt.py` seed 30100192 scripts, verbatim,
//     with `(get-info :reason-unknown)` appended; pass 17 printed a model z3 confirms for both.  On the final
//     tree `D00085`'s candidate is false (its eighth assertion, a `distinct` of two `ite`s whose proxies the
//     search kept apart and the model builder rebuilt alike — `#P2b-88`), so it answers `unknown` with the net's
//     reason (decision (24a)'s (A′) leg names the check); `D00087`'s search finds a model that holds (an
//     intermediate build of this pass answered it `unknown` — the candidate moves with the trajectory).  The
//     pin asserts the PROPERTY, not the trajectory: a `sat` prints a model that holds; an `unknown` says why,
//     and the reason is the net's; never `unsat` (z3: `sat`).
// ---------------------------------------------------------------------------

const D00085: &str = "(set-logic ALL)\n\
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
(assert (not (= red green)))\n\
(assert (or (< 2 (k nil)) (< x (q (g y)))))\n\
(assert ((_ is cons) (ite ((_ is cons) l2) (tl l2) nil)))\n\
(assert (distinct l2 l1))\n\
(assert (= l2 (h (ite ((_ is cons) l2) (hd l2) 0))))\n\
(push 1)\n\
(assert (not (and (= (g x) green) (= p (mk 0 (pc p))))))\n\
(assert (and ((_ is nil) nil) (distinct (ite ((_ is cons) l1) (tl l1) nil) (ite ((_ is cons) (ite ((_ is cons) (cons x l3)) (tl (cons x l3)) nil)) (tl (ite ((_ is cons) (cons x l3)) (tl (cons x l3)) nil)) nil))))\n\
(check-sat)\n\
(get-model)\n\
(get-info :reason-unknown)\n";

const D00087: &str = "(set-logic ALL)\n\
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
(assert (= (h (k (h y))) (h y)))\n\
(assert (not ((_ is cons) (h z))))\n\
(assert ((_ is cons) (v (ite ((_ is cons) l3) (tl l3) nil))))\n\
(assert (= (+ (px p) (- 1)) y))\n\
(check-sat)\n\
(get-model)\n\
(get-info :reason-unknown)\n";

#[test]
fn a_sat_over_a_model_the_reader_shows_false_answers_unknown() {
    for (name, script) in [("d00085", D00085), ("d00087", D00087)] {
        let lines = run(script);
        let got = judge(script, &lines);
        assert_eq!(got.len(), 1, "`{name}`\n{}", lines.join("\n"));
        let (verdict, reading) = &got[0];
        match verdict.as_str() {
            "sat" => assert_eq!(
                reading,
                &ModelReading::Holds,
                "`{name}`: a published model holds\n{}",
                lines.join("\n")
            ),
            "unknown" => {
                assert!(
                    lines.iter().any(|line| line.starts_with(
                        "(error \"model not certified: model check failed: assertion "
                    ) && line
                        .ends_with(" reads false under the candidate model\")")),
                    "`{name}`: (get-model) names the assertion the reader read false\n{}",
                    lines.join("\n")
                );
                assert!(
                    lines.iter().any(|line| line
                        .starts_with("(:reason-unknown \"model check failed: assertion ")),
                    "`{name}`: :reason-unknown carries the net's reason\n{}",
                    lines.join("\n")
                );
            }
            other => panic!("`{name}`: {other} (z3: sat)\n{}", lines.join("\n")),
        }
    }
}

// ---------------------------------------------------------------------------
// §4. `#P2b-90`'s quantified half (found by this pass's `gen_dtite.py --v1` seed 30101802, `t00088`): a datatype
//     term only a quantifier body names had no datatype axiom — the axioms' scan of the assertions stops at
//     the binder, and neither the quantifier's ground expansion nor an MBQI instance was scanned — so a
//     tester over it was a free Boolean.  `Q2`: `(= l1 l2)`, `((_ is cons) l1)`, `((_ is cons) (tl l1))` and
//     `(forall ((n Int)) (=> (and (<= 0 n) (<= n 2)) ((_ is nil) (tl l2))))` answered `sat` on 0.3.3,
//     `c4b04b7`, `c702310` and every round-4 pass through 17 (z3: `unsat`); `Q1` the same through an `ite`,
//     `Q5` under an unbounded guard (MBQI), `T00088` the corpus script verbatim (two checks).  `Q3` (no guard)
//     and `Q7` answered `unknown`, `Q6` `unsat`.  Fixed at the root: every quantified assertion's encoding and
//     every quantifier instance that holds a datatype term is a root of the datatype axioms
//     (`Solver::register_ground_dt_root`), axiomatised before each MBQI round.  Mutation
//     `OXIZ_MUT18_NO_DT_ROOTS` of the isolated copy: `Q1`, `Q2`, `Q5`, `T00088` answer `sat` again.
// ---------------------------------------------------------------------------

const Q1: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(assert (= l1 l2))\n\
(assert ((_ is cons) l1))\n\
(assert ((_ is cons) (tl l1)))\n\
(assert (forall ((n Int)) (=> (and (<= 0 n) (<= n 2)) ((_ is nil) (ite ((_ is cons) l2) (tl l2) nil)))))\n\
(check-sat)\n";

const Q2: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(assert (= l1 l2))\n\
(assert ((_ is cons) l1))\n\
(assert ((_ is cons) (tl l1)))\n\
(assert (forall ((n Int)) (=> (and (<= 0 n) (<= n 2)) ((_ is nil) (tl l2)))))\n\
(check-sat)\n";

const Q5: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(assert (= l1 l2))\n\
(assert ((_ is cons) l1))\n\
(assert ((_ is cons) (tl l1)))\n\
(assert (forall ((n Int)) (=> (> n 0) ((_ is nil) (tl l2)))))\n\
(check-sat)\n";

const Q6: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(assert (= l1 l2))\n\
(assert ((_ is cons) l1))\n\
(assert ((_ is cons) (tl l1)))\n\
(assert (forall ((n Int)) (=> (> n 0) (= (tl l2) (cons n nil)))))\n\
(check-sat)\n";

const Q7: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(assert (= l1 l2))\n\
(assert ((_ is cons) l1))\n\
(assert ((_ is cons) (tl l1)))\n\
(assert (forall ((n Int)) (or (< n 0) ((_ is nil) (tl (tl l2))))))\n\
(assert ((_ is cons) (tl (tl l1))))\n\
(check-sat)\n";

const T00088: &str = "(set-logic ALL)\n\
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
(declare-const a Bool)\n\
(declare-const b Bool)\n\
(declare-const e Bool)\n\
(assert (and (<= (- 3) x 3) (<= (- 3) y 3) (<= (- 3) z 3)))\n\
(assert (distinct (ite ((_ is cons) l3) (tl l3) nil) l3))\n\
(assert ((_ is cons) (ite (= u1 u1) (ite ((_ is cons) l1) (tl l1) nil) (ite ((_ is cons) l1) (v nil) l2))))\n\
(assert (and (= (ite (<= z (- 1)) (ite ((_ is cons) nil) (tl nil) nil) (ite ((_ is cons) nil) (cons x l1) l2)) (ite ((_ is cons) (cons x l1)) (tl (cons x l1)) nil)) (distinct (k (ite ((_ is cons) (ite ((_ is cons) nil) (tl nil) nil)) (tl (ite ((_ is cons) nil) (tl nil) nil)) nil)) (+ x 1))))\n\
(assert (forall ((n Int)) (=> (and (<= 0 n) (<= n 2)) ((_ is nil) (ite ((_ is cons) l2) (tl l2) nil)))))\n\
(check-sat)\n\
(get-model)\n\
(push 1)\n\
(assert (and ((_ is cons) (ite (= l1 l2) l2 l1)) ((_ is blue) blue)))\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn a_datatype_term_only_a_quantifier_body_names_is_axiomatised() {
    for (name, script, expected) in [
        ("q1", Q1, vec!["unsat"]),
        ("q2", Q2, vec!["unsat"]),
        ("q5", Q5, vec!["unsat"]),
        ("q6", Q6, vec!["unsat"]),
        ("q7", Q7, vec!["unsat"]),
        ("t00088", T00088, vec!["unsat", "unsat"]),
    ] {
        let lines = run(script);
        assert_eq!(
            verdicts(&lines),
            expected,
            "`{name}` (z3: {expected:?})\n{}",
            lines.join("\n")
        );
    }
}
