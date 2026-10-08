//! Round-4 re-fix pass 16 — pins for what it fixed beside recheck 15's own
//! pins (`round4_pass15_recheck_pins.rs`, inverted where closed).
//!
//! * `#P2b-84` at the two ranges recheck 15 named and did not pin: an
//!   uninterpreted sort (`d08`) and a datatype argument (`d11`, `h : L → L`).
//! * `#P2b-81`'s named mechanism: an array constant whose default a script
//!   spells as a negated numeral, `(- 3)`, printed as the constant `0`.
//!
//! Every model is judged by REPLAY on its own assertions (the published
//! `define-fun` lines in a closed script), as recheck 15's pins do.  No test
//! installs a wall clock (decision (16)).

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// Every `(define-fun …)` line of the first `(model …)` response.
fn model_definitions(lines: &[String]) -> String {
    let mut out = String::new();
    if let Some(model) = lines
        .iter()
        .find(|line| line.trim_start().starts_with("(model"))
    {
        for raw in model.lines() {
            let trimmed = raw.trim();
            if trimmed.starts_with("(define-fun ") {
                out.push_str(trimmed);
                out.push('\n');
            }
        }
    }
    out
}

fn replay(prelude: &str, definitions: &str, claim: &str) -> String {
    verdict(&run(&format!(
        "(set-logic ALL)\n{prelude}{definitions}(assert {claim})\n(check-sat)\n"
    )))
}

/// `(h x) = u1`, `(h y) = u2` with `u1 ≠ u2` over `x, y ∈ [0, 5]`: every
/// build printed `x = y = 0` beside `h` constantly `@uc_U_0` (an uninterpreted
/// range compared by congruence class, `#P2b-84`); and `(h x) = [1]`, `(h y) =
/// [2]` with `h : L → L` printed one value at two equal arguments.
#[test]
fn a_function_into_or_over_a_declared_sort_is_kept_a_function() {
    let script = "(set-logic ALL)\n(declare-sort U 0)\n(declare-const u1 U)\n(declare-const u2 U)\n\
         (declare-fun h (Int) U)\n(declare-const x Int)\n(declare-const y Int)\n\
         (assert (>= x 0))\n(assert (<= x 5))\n(assert (>= y 0))\n(assert (<= y 5))\n\
         (assert (distinct u1 u2))\n(assert (= (h x) u1))\n(assert (= (h y) u2))\n\
         (check-sat)\n(get-model)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let joined = lines.join("\n");
    assert!(
        !joined.contains("(define-fun x () Int 0)") || !joined.contains("(define-fun y () Int 0)"),
        "`h`'s two applications sit at one point\n{joined}"
    );

    let list = "(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n";
    let lines = run(&format!(
        "(set-logic ALL)\n{list}(declare-fun h (L) L)\n(declare-const x L)\n(declare-const y L)\n\
         (assert (= (h x) (cons 1 nil)))\n(assert (= (h y) (cons 2 nil)))\n(check-sat)\n(get-model)\n"
    ));
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    assert_eq!(
        replay(
            list,
            &model_definitions(&lines),
            "(and (= (h x) (cons 1 nil)) (= (h y) (cons 2 nil)))"
        ),
        "sat",
        "{}",
        lines.join("\n")
    );
}

/// `(= ((as const (Array Int Int)) (- 3)) a)` printed `a` as the constant `0`
/// beside a `(get-value)` answering `-3`, on every build.
#[test]
fn an_array_default_spelled_as_a_negated_numeral_is_printed() {
    let lines = run(
        "(set-logic ALL)\n(declare-const a (Array Int Int))\n(declare-const k Int)\n\
         (assert (= ((as const (Array Int Int)) (- 3)) a))\n(check-sat)\n(get-model)\n\
         (get-value ((select a k)))\n",
    );
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let joined = lines.join("\n");
    assert!(joined.contains("((select a k) -3)"), "{joined}");
    assert!(
        !joined.contains("((as const (Array Int Int)) 0)"),
        "the default is read from the negated numeral\n{joined}"
    );
    assert_eq!(
        replay(
            "",
            &model_definitions(&lines),
            "(= ((as const (Array Int Int)) (- 3)) a)"
        ),
        "sat",
        "{joined}"
    );
}
