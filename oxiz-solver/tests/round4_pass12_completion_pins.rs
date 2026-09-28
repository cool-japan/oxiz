//! Round-4 re-fix pass 12 — regression guards for the array model completion
//! at every `Sat` exit (decision (40), `#P2b-51`) and for its default **plus
//! pinned points** (decision (41)).
//!
//! The holes these decisions closed are pinned, inverted, where they were
//! first pinned (`round4_pass6_recheck_pins`, `round4_pass11_recheck_pins`).
//! This file guards the properties the change must keep:
//!
//! * on a `Sat` no honesty gate takes away, the **verdict** is never touched,
//!   and a model is replaced only by one that satisfies every assertion — the
//!   published model is *replayed*, never string-matched;
//! * `(get-value)` answers from the interpretation `(get-model)` prints, also
//!   when that interpretation replaced the candidate model's;
//! * a pinned completion the script contradicts is refused (the verdict is the
//!   `unsat` the formula has), survives no `pop`, and is deterministic;
//! * where the completion cannot certify anything, the original verdict and a
//!   correct model are what the user gets.
//!
//! No test here installs a wall clock (decision (16)): everything asserted is
//! a verdict or a replayed model.

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

/// `(get-value)` reads through the certified interpretation that replaced the
/// candidate model on the `Sat` path, and agrees with `(get-model)`.
///
/// The candidate model of this script carries the read `(select a #b0000010)
/// ↦ true` and nothing else, so before decision (40) the array printed as
/// `(store ((as const …) false) #b0000010 true)`.  The certified
/// interpretation is `((as const …) true)`: a read at the pinned index, a read
/// at an index the script never names, and the array itself must all answer
/// from it — an entry left over from the replaced interpretation, or a read
/// answered from the congruence class, would describe a second model.
#[test]
fn get_value_answers_from_a_model_the_sat_path_completion_installed() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) Bool))\n\
         (declare-const p Bool)\n\
         (assert (or p (select a #b0000010)))\n\
         (assert (forall ((i (_ BitVec 7))) (select a i)))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((select a #b0000010) (select a #b1010101) a))\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let printed = published(&lines, "a").unwrap_or_default();
    let text = lines.join("\n");
    assert!(
        printed.contains("as const"),
        "`a` must be published as an array value\n{text}"
    );
    assert!(
        text.contains("((select a #b0000010) true)")
            && text.contains("((select a #b1010101) true)"),
        "both reads must answer `true`, the value the certified interpretation \
         gives every index\n{text}"
    );
    assert!(
        text.contains(&format!("(a {printed})")),
        "`(get-value (a))` must print the interpretation `(get-model)` printed \
         ({printed})\n{text}"
    );
}

/// `(get-value)` of a `store` over a completed array, and of an array `ite`
/// that selects one, reads the **installed** base: the certified `a` is the
/// constant `#b1`, so the store at `#b0000011` reads `#b1` everywhere else.
/// Before this pass's repair the chain printed the candidate model's base (the
/// sort default `#b0`) beside a `(get-model)` printing `#b1` — two models.
#[test]
fn get_value_of_a_store_over_a_completed_array_reads_the_installed_base() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (not p))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n\
         (get-value ((store a #b0000011 #b0) (ite p a (store a #b0000001 #b0)) \
         (select (store a #b0000011 #b0) #b0000100)))\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let text = lines.join("\n");
    let constant = "((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1)";
    assert!(
        text.contains(&format!(
            "((store a #b0000011 #b0) (store {constant} #b0000011 #b0))"
        )) && text.contains(&format!(
            "((ite p a (store a #b0000001 #b0)) (store {constant} #b0000001 #b0))"
        )) && text.contains("((select (store a #b0000011 #b0) #b0000100) #b1)"),
        "a store over the completed `a` must be printed over the installed \
         constant `#b1`\n{text}"
    );
}

/// The `Sat`-path completion (decision (40)) on an **`Int`** index sort, the
/// case only that hook reaches (the saturation-time one is for finite index
/// sorts).  `∀i. a[i] = 5` is `sat`, and HEAD `c702310` published
/// `a = (store … (store ((as const …) 0) -2 5) … 5 5)`, false at `i = 6`.
///
/// Over `Int` the model cannot be replayed point by point, so it is replayed
/// against the universal's own negation instead: the published model pinned,
/// beside a fresh `k` with `a[k] ≠ 5`, must be `unsat` — which it is exactly
/// when the published `a` reads `5` everywhere.  With the hook mutated away
/// (isolated copy) the model is HEAD's and this replay answers `sat`.
#[test]
fn the_sat_path_completion_repairs_an_int_indexed_model() {
    let decls = "(declare-const a (Array Int Int))\n";
    let lines = run(&format!(
        "(set-logic ALL)\n(set-option :produce-models true)\n{decls}\
         (assert (forall ((i Int)) (= (select a i) 5)))\n(check-sat)\n(get-model)\n"
    ));
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let pins = model_equalities(&lines);
    assert!(pins.contains("(= a "), "{}", lines.join("\n"));
    assert_verdict(
        &format!(
            "(set-logic ALL)\n{decls}{pins}(declare-const k Int)\n\
             (assert (distinct (select a k) 5))\n(check-sat)\n"
        ),
        "unsat",
        "the published `a` must read `5` at every index, so no `k` may escape",
    );
}

/// A universal at **negative** polarity is never "completed" into a model that
/// makes it true.
///
/// `(not (forall ((i …)) (= (select a i) #b1)))` beside `a[0] = #b1` is
/// satisfiable, and the constant array `#b1` — which the pool offers, and
/// which certifies the universal itself — is exactly the interpretation that
/// falsifies the assertion.  The certificate's assertion half must refuse it,
/// and whatever model is published must satisfy the negated universal: some
/// point reads other than `#b1`.  Replayed over all 128 points.
#[test]
fn a_negated_universal_is_never_completed_into_a_falsifying_model() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n";
    let ground = "(assert (= (select a #b0000000) #b1))\n";
    let script = format!(
        "(set-logic ALL)\n(set-option :produce-models true)\n{decls}\
         (assert (not (forall ((i (_ BitVec 7))) (= (select a i) #b1))))\n{ground}\
         (check-sat)\n(get-model)\n"
    );
    let lines = run(&script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let pins = model_equalities(&lines);
    assert!(pins.contains("(= a "), "{}", lines.join("\n"));
    let mut disjuncts = String::new();
    for point in points(7) {
        disjuncts.push_str(&format!(" (distinct (select a {point}) #b1)"));
    }
    let replay =
        format!("(set-logic ALL)\n{decls}{pins}{ground}(assert (or{disjuncts}))\n(check-sat)\n");
    assert_verdict(
        &replay,
        "sat",
        "the published model must falsify the universal at some point, as \
         the negated assertion demands",
    );
}

/// A pinned completion the script contradicts is refused, at a plain point and
/// at a stored one; the verdict is the `unsat` the formula has.
///
/// Both default-plus-pins interpretations the pinned search can propose here
/// satisfy the universal and contradict something else: a ground read in the
/// first script, and the `store` the universal reads through in the second.
#[test]
fn a_pinned_completion_the_script_contradicts_is_refused() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (= (select a #b0000000) #b1))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select a i) (ite (= i #b0000000) #b0 #b1))))\n\
         (check-sat)\n",
        "unsat",
        "the universal pins `a[0] = #b0` and the ground read says `#b1`",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select (store a #b0000011 #b0) i) (ite (= i #b0000000) #b0 #b1))))\n\
         (check-sat)\n",
        "unsat",
        "at `i = #b0000011` the store reads `#b0` and the body demands `#b1`",
    );
}

/// `push`/`pop` around a pinned completion: a ground assertion added after a
/// certified `sat` takes it away, and nothing certified inside the scope
/// survives the `pop`.
#[test]
fn a_push_pop_scope_around_a_pinned_completion_keeps_every_verdict() {
    let lines = run("(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (push 1)\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select a i) (ite (= i #b0000000) #b0 #b1))))\n\
         (check-sat)\n\
         (assert (= (select a #b0000000) #b1))\n\
         (check-sat)\n\
         (pop 1)\n\
         (assert (= (select a #b0000000) #b1))\n\
         (check-sat)\n");
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "unsat", "sat"],
        "{}",
        lines.join("\n")
    );
}

/// The pinned completion is byte-identical across two fresh `Context`s: the
/// points are ordered by value, not by term id or hash order.
#[test]
fn the_pinned_completion_is_the_same_on_two_runs() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (= (select b #b1000001) #b0))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select a i) (ite (= i #b0000101) #b0 (ite (= i #b0000000) #b0 #b1)))))\n\
         (check-sat)\n(get-model)\n";
    let first = run(script);
    let second = run(script);
    assert_eq!(verdict(&first), "sat", "{}", first.join("\n"));
    assert_eq!(first, second, "the pinned completion must be deterministic");
}

/// Where no default-plus-pins interpretation exists, the completion declines,
/// and the verdict and the model `check_core` produced are what is published.
///
/// `a[i] = ((_ extract 0 0) i)` over `(_ BitVec 6)` makes `a` alternate — no
/// constant and no finite set of points the goal names will do — and the
/// 64-point finite expansion decides it `sat` with an exact model.  The model
/// is replayed at every point.
#[test]
fn a_sat_the_completion_cannot_certify_keeps_its_verdict_and_a_correct_model() {
    let decls = "(declare-const a (Array (_ BitVec 6) (_ BitVec 1)))\n";
    let script = format!(
        "(set-logic ALL)\n(set-option :produce-models true)\n{decls}\
         (assert (forall ((i (_ BitVec 6))) (= (select a i) ((_ extract 0 0) i))))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = run(&script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let pins = model_equalities(&lines);
    let mut replay = format!("(set-logic ALL)\n{decls}{pins}");
    for (index, point) in points(6).iter().enumerate() {
        replay.push_str(&format!(
            "(assert (= (select a {point}) #b{}))\n",
            index & 1
        ));
    }
    replay.push_str("(check-sat)\n");
    assert_verdict(
        &replay,
        "sat",
        "the published model must be the alternating array at every point",
    );
}

/// `#P2b-62`: a quantifier body's read over its **bound** variable is never
/// laid down as an array entry.
///
/// `bench/z3_parity/benchmarks/AUFLIA/array_extensionality.smt2` verbatim,
/// with `(get-model)`.  The body term `(select a i)` and the bound `i` carried
/// model entries (0 and 0), and the renderer printed them as `a[0] = 0` —
/// shadowing the real `(select a 0) = 10`, so HEAD `c702310` published a model
/// falsifying the script's own GROUND assertion (`c4b04b7` printed `0 10`).
/// The replay pins the published arrays back beside every ground assertion.
#[test]
fn a_quantifier_bodys_read_over_its_bound_variable_is_never_a_model_entry() {
    let decls = "(declare-const a (Array Int Int))\n(declare-const b (Array Int Int))\n";
    let ground = "(assert (= (select a 0) 10))\n(assert (= (select a 1) 20))\n\
         (assert (= (select b 0) 10))\n(assert (= (select b 1) 20))\n";
    let lines = run(&format!(
        "(set-logic AUFLIA)\n(set-option :produce-models true)\n{decls}\
         (assert (forall ((i Int)) (= (select a i) (select b i))))\n(assert (= a b))\n{ground}\
         (check-sat)\n(get-model)\n"
    ));
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let pins = model_equalities(&lines);
    assert!(
        pins.contains("(= a ") && pins.contains("(= b "),
        "{}",
        lines.join("\n")
    );
    assert_verdict(
        &format!("(set-logic AUFLIA)\n{decls}{pins}{ground}(assert (= a b))\n(check-sat)\n"),
        "sat",
        "the published arrays must satisfy every ground assertion of the script",
    );
}

// ---------------------------------------------------------------------------
// OPEN HOLES — `#P2b-63`: where the completion DECLINES, a `sat` still
// publishes the candidate model, and that model can falsify the quantifier.
// ---------------------------------------------------------------------------

/// **THE HOLE IS CLOSED** when any replay below answers `sat`.
///
/// `#P2b-51`'s mechanism — the completion was never consulted on a `Sat` — is
/// closed (decision (40)).  What is left is its reach: three shapes the
/// completion declines by rule (`TODO.md` `#P2b-58` (f)), each `sat` and each
/// still published with a model false at most points of the universal
/// `∀i. a[i] = #b1` (or `a0[i]`), established here by replaying the model into
/// the quantifier written out over all 128 points:
///
/// * an uninterpreted function anywhere in the goal (`f`, unrelated to `a`);
/// * more than three array-sorted free variables (four here);
/// * an `exists` beside the universal.
///
/// Byte-identical verdicts on HEAD `c702310` (`sat`, falsifying) — not a
/// regression.  To close: interpret the declined symbol from the candidate
/// model instead of declining (a function's entries and default; the arrays
/// the universal does not read at their candidate values; the existential's
/// Skolem witness), certify, and flip the `unsat`s below to `sat`.
#[test]
fn a_declined_completion_still_publishes_a_falsifying_model() {
    let arrays = |names: &[&str]| -> String {
        names
            .iter()
            .map(|n| format!("(declare-const {n} (Array (_ BitVec 7) (_ BitVec 1)))\n"))
            .collect()
    };
    let cases: [(&str, String, String, &str); 3] = [
        (
            "an uninterpreted function",
            format!(
                "{}(declare-fun f ((_ BitVec 7)) (_ BitVec 1))\n\
                 (assert (= (f #b0000000) #b0))\n",
                arrays(&["a"])
            ),
            "(forall ((i (_ BitVec 7))) (= (select a i) #b1))".to_string(),
            "a",
        ),
        (
            "four arrays",
            format!(
                "{}(assert (= (select a1 #b0000001) (select a2 #b0000010)))\n\
                 (assert (= (select a3 #b0000011) #b0))\n",
                arrays(&["a0", "a1", "a2", "a3"])
            ),
            "(forall ((i (_ BitVec 7))) (= (select a0 i) #b1))".to_string(),
            "a0",
        ),
        (
            "an exists beside the universal",
            format!(
                "{}(assert (exists ((j (_ BitVec 7))) (= (select a j) #b1)))\n",
                arrays(&["a"])
            ),
            "(forall ((i (_ BitVec 7))) (= (select a i) #b1))".to_string(),
            "a",
        ),
    ];
    for (what, prefix, universal, array) in cases {
        let lines = run(&format!(
            "(set-logic ALL)\n(set-option :produce-models true)\n{prefix}(assert {universal})\n\
             (check-sat)\n(get-model)\n"
        ));
        assert_eq!(verdict(&lines), "sat", "({what})\n{}", lines.join("\n"));
        let pins = model_equalities(&lines)
            .lines()
            .filter(|line| line.starts_with(&format!("(assert (= {array} ")))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !pins.is_empty(),
            "({what}) `{array}` must be published\n{}",
            lines.join("\n")
        );
        let mut replay = format!("(set-logic ALL)\n{}{pins}\n", arrays(&[array]));
        for point in points(7) {
            replay.push_str(&format!("(assert (= (select {array} {point}) #b1))\n"));
        }
        replay.push_str("(check-sat)\n");
        assert_verdict(
            &replay,
            "unsat",
            &format!(
                "THE HOLE IS CLOSED ({what}): the published `{array}` now holds at \
                 all 128 points. Flip this `unsat` to `sat` and close `#P2b-63`"
            ),
        );
    }
}
