//! Round-4 adversarial recheck pass 11 — what re-fix pass 11 left open, and
//! independent guards for the model completion (`#P2b-58`, decision (36)) it
//! landed.
//!
//! # How to read this file
//!
//! * A test whose doc carries **THE HOLE IS CLOSED** asserts what this tree
//!   answers **today**, which is the weaker answer. It is green now and must
//!   go **red** when the defect is fixed; the doc names the assertion to
//!   flip. Nothing here asserts a wrong `sat` or a wrong `unsat`: every hole
//!   pinned below is either an honest `unknown`, or a *correct* verdict whose
//!   published model is wrong — and in the latter case the falsification is
//!   established **on this tree**, by replaying the published model into the
//!   same formula written out over every point of its index sort, so the pin
//!   never rests on an external solver.
//! * Every other test asserts the CORRECT answer and is an ordinary
//!   regression guard for `solver::array_completion_certify`, the module
//!   re-fix pass 11 added. They were written by attacking that module rather
//!   than by reading its own test list: scope (`push`/`pop` around a certified
//!   `sat`), polarity (one hash-consed `Forall` node used positively and
//!   negatively), capture (a bound name that is also a *declared constant* of
//!   the same sort), and the two ways the module must decline (a satisfying
//!   interpretation that is not constant anywhere, and an uninterpreted
//!   function under the binder).
//!
//! No test here installs a wall clock: no `(set-option :timeout N)` appears
//! below and no assertion is made about elapsed time. Everything asserted is a
//! verdict, a published model, or a deterministic counter.
//!
//! Every figure quoted for `c4b04b7` was measured with the round's release
//! probe `<scratchpad>/oxiz4/rc3/probe_base`; the scripts and the raw output
//! are under `<scratchpad>/oxiz4/rk12/` with a `REBUILD.md`.

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

/// Turn a `(model …)` response into the `(assert (= name value))` lines that
/// pin it, so the model can be replayed into another script.
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

/// Every point of `(_ BitVec 7)` as an SMT-LIB literal.
fn width_seven_points() -> Vec<String> {
    (0u32..128).map(|i| format!("#b{i:07b}")).collect()
}

// ---------------------------------------------------------------------------
// 1. OPEN HOLES
// ---------------------------------------------------------------------------

/// **THE HOLE IS CLOSED** when the replay below answers `sat`.
///
/// `#P2b-51`, in **three** lines, and in the exact shape `#P2b-58`'s close-out
/// says is now decided by a certified completion:
///
/// ```text
/// (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))
/// (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))
/// ```
///
/// The verdict `sat` is **correct** — `a = ((as const …) #b1)` satisfies it —
/// and the published model is `((as const …) #b0)`, which reads `#b0`
/// everywhere and so falsifies the only assertion in the script. This test
/// establishes that on this tree and on nothing else: it takes the published
/// model, pins it, and writes the quantifier out over all 128 points of its
/// index sort. A real model keeps that `sat`; this one turns it `unsat`.
///
/// It is the *residue* decision (36) deliberately left, recorded in `TODO.md`
/// `#P2b-58` as a decision with its reason: `Solver::check` fires the
/// completion only where a verdict would otherwise be given up, and here
/// `check_core` already answers `Sat` with no honesty gate pending, so the
/// module never runs. The shape matters because it is *smaller* than every
/// script `#P2b-58`'s list names — `rk6/corpus/qmbqi120/q0074` needs a second
/// assertion to reach the completion at all — so a reader of the `[0.3.4]`
/// notes could reasonably expect it to be covered and it is not.
///
/// `c4b04b7` answers `unknown` here with `(error "No model available")`, so
/// this is not a regression: the tree gained a verdict and kept a bad model.
/// Measured on the delivered tree and on `0559fb3`: byte-identical responses,
/// so the model is not new to re-fix pass 11 either.
///
/// To close: publish `((as const …) #b1)` — which is what
/// `array_completion_certify` would certify if it were consulted here — and
/// flip the `unsat` below to `sat`.
#[test]
fn a_three_line_quantified_array_script_still_publishes_a_falsifying_model() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n\
         (get-model)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "the script is satisfiable and this tree says so; the pin is about \
         the MODEL\n--- response ---\n{}",
        lines.join("\n")
    );
    let pins = model_equalities(&lines);
    assert!(
        pins.contains("as const"),
        "the published model must name an array constant for the replay to \
         mean anything\n--- response ---\n{}",
        lines.join("\n")
    );
    let mut replay =
        String::from("(set-logic ALL)\n(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n");
    replay.push_str(&pins);
    for point in width_seven_points() {
        replay.push_str(&format!("(assert (= (select a {point}) #b1))\n"));
    }
    replay.push_str("(check-sat)\n");
    assert_verdict(
        &replay,
        "unsat",
        "THE HOLE IS CLOSED: the published model now satisfies the quantifier \
         at every one of the 128 points of its index sort, so `#P2b-51` no \
         longer reaches this shape. Flip this `unsat` to `sat`, and re-state \
         `#P2b-51` in TODO.md and the `[0.3.4]` CHANGELOG bullet",
    );
}

/// **THE HOLE IS CLOSED** when the replay below answers `sat`.
///
/// The same hole at an element sort `#P2b-51` has never been measured at:
/// **`Bool`**. Every script the entry names has a `(_ BitVec 1)` element sort,
/// and the entry's own mechanism sentence — *"`context::model_fmt::array_model`
/// takes an array's default from the **sort**"* — predicts the Boolean case
/// but nothing pins it.
///
/// `(forall ((i (_ BitVec 7))) (select a i))` over `(Array (_ BitVec 7) Bool)`
/// is `sat` here (correctly: `a = ((as const …) true)`), and the published
/// model is `(store ((as const …) false) #b0000010 true)` — `false` at 127 of
/// the 128 points, so it falsifies the quantifier. Byte-identical on `0559fb3`,
/// so it is pre-existing rather than new.
///
/// To close: publish a model that satisfies the quantifier, and flip the
/// `unsat` below to `sat`.
#[test]
fn the_falsifying_model_family_reaches_a_boolean_element_sort_too() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a (Array (_ BitVec 7) Bool))\n\
         (declare-const p Bool)\n\
         (assert (or p (select a #b0000010)))\n\
         (assert (forall ((i (_ BitVec 7))) (select a i)))\n\
         (check-sat)\n\
         (get-model)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "the script is satisfiable and this tree says so; the pin is about \
         the MODEL\n--- response ---\n{}",
        lines.join("\n")
    );
    let pins = model_equalities(&lines);
    let mut replay = String::from(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) Bool))\n\
         (declare-const p Bool)\n",
    );
    replay.push_str(&pins);
    for point in width_seven_points() {
        replay.push_str(&format!("(assert (select a {point}))\n"));
    }
    replay.push_str("(check-sat)\n");
    assert_verdict(
        &replay,
        "unsat",
        "THE HOLE IS CLOSED at the `Bool` element sort: the published model \
         now satisfies the quantifier everywhere. Flip this `unsat` to `sat` \
         and record the element sort in `#P2b-51`",
    );
}

/// **THE HOLE IS CLOSED** when either verdict below is `sat`.
///
/// The two shapes `solver::array_completion_certify` declines by construction,
/// pinned so a later widening is visible rather than silent. Both are
/// satisfiable, and in both the satisfying interpretation is **not** a constant
/// array over any pooled default:
///
/// * a body that forces one index to a different value from all the others
///   (`a = (store ((as const …) #b1) #b0000000 #b0)`);
/// * a **guarded** universal that constrains only the indices below
///   `#b0000100`, beside a ground assertion that forces one of the rest the
///   other way.
///
/// Both answer `unknown` here — which is honest, and is what the module's own
/// doc says it will do — and `c4b04b7` answers `unknown` on both as well, so
/// nothing is lost against the base. The pin exists because `#P2b-58`'s
/// close-out lists what declines in prose only; this turns two of those six
/// lines into something `cargo nextest` checks.
///
/// To close: extend the completion past a constant array (pins plus a default,
/// with the pins taken from the ground terms the goal already names) and flip
/// the two `unknown`s.
#[test]
fn an_interpretation_that_is_not_constant_anywhere_is_declined_and_never_sat() {
    let one_point_off = "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (or p (= (select a #b0000010) #b1)))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select a i) (ite (= i #b0000000) #b0 #b1))))\n\
         (check-sat)\n";
    let guarded = "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (= (select a #b0000101) #b0))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (=> (bvult i #b0000100) (= (select a i) #b1))))\n\
         (check-sat)\n";
    for (script, name) in [(one_point_off, "one point off"), (guarded, "guarded")] {
        assert_verdict(
            script,
            "unknown",
            &format!(
                "THE HOLE IS CLOSED ({name}): the completion now reaches an \
                 interpretation that is not a constant array. Flip this \
                 `unknown` to `sat`, and add the model replay this pin's \
                 siblings use — a `sat` published here MUST come with a model \
                 that satisfies the quantifier at every point"
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// 2. REGRESSION GUARDS FOR THE COMPLETION (decision (36))
// ---------------------------------------------------------------------------

/// A `push`/`pop` scope around a certified `sat` never leaks and never
/// survives.
///
/// `Solver::certify_sat_by_array_completion` certifies over a clone of
/// `Solver::assertions`, and the whole soundness argument rests on that vector
/// being exactly the assertions in scope. Two directions, both attacked:
///
/// * a ground assertion added **after** a completed `sat`, inside the same
///   scope, must take it away (`sat`, then `unsat`);
/// * after the `pop`, the contradiction re-asserted beside the quantifier must
///   still be refuted, so nothing certified inside the scope survives it.
///
/// The mirror — the contradiction asserted **before** the `push` — is the
/// second script: the quantifier inside the scope is refuted, and popping it
/// gives the ground assertion back on its own.
#[test]
fn a_push_pop_scope_around_a_certified_completion_keeps_every_verdict() {
    let after = "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (push 1)\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n\
         (assert (= (select a #b0000000) #b0))\n\
         (check-sat)\n\
         (pop 1)\n\
         (assert (= (select a #b0000000) #b0))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n";
    assert_eq!(
        verdicts(&run(after)),
        vec!["sat", "unsat", "unsat"],
        "a ground assertion added after a certified `sat` must refute it, and \
         nothing certified inside the scope may survive the `pop`"
    );

    let before = "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (= (select a #b0000000) #b0))\n\
         (push 1)\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n\
         (pop 1)\n\
         (check-sat)\n";
    assert_eq!(
        verdicts(&run(before)),
        vec!["unsat", "sat"],
        "the quantifier contradicts a ground assertion made before the \
         `push`, and popping it must give that assertion back on its own"
    );
}

/// One hash-consed `Forall` node used at **both** polarities is refuted.
///
/// `Goal::build` collects the maximal `Forall` *node*, and the certificate
/// replaces it with `true` wherever it occurs. Two assertions spelling the
/// identical body are the same `TermId`, so the positive and the negative
/// occurrence are one entry in that map — which is exactly the shape where a
/// careless substitution would certify `(and P (not P))`.
///
/// The control beside it keeps the pin honest: the negated universal on its
/// own, beside a ground assertion it is consistent with, is `sat`.
#[test]
fn one_forall_node_at_two_polarities_is_refuted_and_its_control_is_sat() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (assert (not (forall ((i (_ BitVec 7))) (= (select a i) #b1))))\n\
         (check-sat)\n",
        "unsat",
        "`P` and `(not P)` over one hash-consed `Forall` node is a \
         contradiction and may never be certified",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (not (forall ((i (_ BitVec 7))) (= (select a i) #b1))))\n\
         (assert (= (select a #b0000000) #b1))\n\
         (check-sat)\n",
        "sat",
        "the control: a negated universal is satisfiable beside a ground \
         assertion it agrees with",
    );
}

/// A bound name that is also a **declared constant of the same sort** never
/// certifies a contradiction.
///
/// `peel_universal` rebuilds the bound variable by name *and sort* and renames
/// it to a fresh reserved constant. When a script declares `i` at the binder's
/// own sort, the two are one term, and the rename reaches the declared
/// constant as well. That direction only ever makes the certificate's validity
/// query harder to discharge — but "only ever" is the kind of claim a test
/// should carry, so here it is: the shape where the declared `i` and the
/// quantifier disagree is `unsat`, and the shape where they agree is `sat`.
///
/// This is a *different* collision from the one re-fix pass 11 pinned: there
/// the bound name was the **array's** name, whose sort differs, so the two
/// terms were never equal in the first place.
#[test]
fn a_bound_name_that_is_also_a_declared_constant_is_never_mis_certified() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-fun i () (_ BitVec 7))\n\
         (assert (= (select a i) #b0))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n",
        "unsat",
        "the declared `i` reads `#b0` and the quantifier forces `#b1` \
         everywhere; the bound/free name collision may not hide that",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-fun i () (_ BitVec 7))\n\
         (assert (= i #b0000001))\n\
         (assert (= (select a i) #b1))\n\
         (check-sat)\n",
        "sat",
        "the control: the same collision without the contradiction",
    );
}

/// An uninterpreted function under the binder never produces a `sat` the
/// certificate cannot back.
///
/// `is_small_enough_to_certify` refuses a goal carrying any `Apply` that is not
/// the array constant, precisely because such a symbol survives into the
/// validity query where the sub-solver may interpret it freely. The refusal
/// must cost completeness and never soundness, so both halves are pinned: the
/// contradictory shape is `unsat` (the ordinary path reaches it), and the
/// consistent shape is `sat`.
#[test]
fn an_uninterpreted_function_under_the_binder_costs_no_soundness() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-fun f ((_ BitVec 7)) (_ BitVec 1))\n\
         (assert (= (select a #b0000001) #b1))\n\
         (assert (= (f #b0000001) #b0))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) (f i))))\n\
         (check-sat)\n",
        "unsat",
        "`a[1] = #b1`, `f(1) = #b0` and `∀i. a[i] = f(i)` is a contradiction",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-fun f ((_ BitVec 7)) (_ BitVec 1))\n\
         (assert (= (select a #b0000001) #b1))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) (f i))))\n\
         (check-sat)\n",
        "sat",
        "the control: the same shape without the contradiction",
    );
}

/// The three-array member of the completion family publishes a model that
/// holds at every one of the 256 points of its index sort.
///
/// `rk6/corpus/qmbqi120/q0106` is the widest script decision (36) names —
/// index sort `(_ BitVec 8)`, **three** array-sorted free variables under one
/// binder, so the completion search is a product over three pools. Re-fix
/// pass 11 pins `q0074` (one array) and `m1` (two); nothing pinned the shape
/// where `MAX_COMPLETED_ARRAYS` is actually reached, which is where a wrong
/// combination would first show.
///
/// The published model is replayed rather than string-matched: a different but
/// equally correct completion must keep this green.
#[test]
fn the_three_array_member_publishes_a_model_that_holds_at_every_point() {
    let decls = "(declare-const a0 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a2 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (declare-const d (_ BitVec 1))\n";
    let ground = "(assert (distinct (bvxor (select (ite p a0 \
         ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b1)) #b00001100) #b1) #b1))\n";
    let quantified = "(assert (forall ((i!q (_ BitVec 8))) \
         (= (select (store (ite q a2 a0) #b00001001 #b1) i!q) \
         (select (store (ite q a2 a1) #b10101100 #b1) i!q))))\n";
    let script = format!(
        "(set-logic ALL)\n(set-option :produce-models true)\n{decls}{ground}{quantified}(check-sat)\n(get-model)\n"
    );
    let lines = run(&script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "`c4b04b7` answers `sat` in 0.6 ms and decision (36) requires this \
         tree to as well\n--- response ---\n{}",
        lines.join("\n")
    );
    let pins = model_equalities(&lines);
    assert!(
        pins.matches("as const").count() >= 3,
        "all three arrays must be completed for this pin to test the product \
         search\n--- response ---\n{}",
        lines.join("\n")
    );
    let mut replay = format!("(set-logic ALL)\n{decls}{pins}{ground}");
    for point in 0u32..256 {
        replay.push_str(&format!(
            "(assert (= (select (store (ite q a2 a0) #b00001001 #b1) #b{point:08b}) \
             (select (store (ite q a2 a1) #b10101100 #b1) #b{point:08b})))\n"
        ));
    }
    replay.push_str("(check-sat)\n");
    assert_verdict(
        &replay,
        "sat",
        "the published model must satisfy the quantifier at every one of the \
         256 points of its index sort, and the ground assertion beside it",
    );
}

/// The completed model is byte-identical across two fresh `Context`s, on the
/// three-array script where the search order could show.
///
/// `Goal::build` sorts the free variables by term id and `default_pools` sorts
/// the model values, so the combination the search reaches first must be a
/// property of the goal and not of a hash iteration order. Re-fix pass 11 pins
/// this on a one-array script; the product search is where an unordered pool
/// would first leak.
#[test]
fn the_three_array_completion_is_the_same_on_two_runs() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a0 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a2 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (assert (distinct (bvxor (select (ite p a0 \
         ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b1)) #b00001100) #b1) #b1))\n\
         (assert (forall ((i!q (_ BitVec 8))) \
         (= (select (store (ite q a2 a0) #b00001001 #b1) i!q) \
         (select (store (ite q a2 a1) #b10101100 #b1) i!q))))\n\
         (check-sat)\n\
         (get-model)\n";
    let first = run(script);
    let second = run(script);
    assert_eq!(
        first, second,
        "the completion search must be ordered by term id, not by a hash \
         iteration order"
    );
    assert_eq!(verdict(&first), "sat");
}

/// `(get-value …)` agrees with `(get-model)` on a completed array.
///
/// The two answers come from different code — `(get-model)` from the model
/// entry `install_completed_model` writes, `(get-value …)` from the congruence
/// class rendering — and re-fix pass 11 changed only the first. A published
/// interpretation that disagrees with itself between the two commands would be
/// a model that falsifies its own script under one of them, so the pin reads
/// the array **and** two `select`s through it and checks all three agree.
#[test]
fn get_value_agrees_with_get_model_on_a_completed_array() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const q Bool)\n\
         (assert (or (= (bvxor (select (ite q a0 (ite q a0 a1)) #b1000100) #b1) \
         (bvxor (select ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1) #b1000100) #b0)) \
         (= #b0 (bvxor (select ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1) #b1101111) #b0))))\n\
         (assert (forall ((i!q (_ BitVec 7))) (distinct (bvxor (select (store \
         ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0) #b1000110 #b0) i!q) #b0) \
         (select a1 i!q))))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((select a1 #b1000110) (select a1 #b0101010)))\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "`q0074` is the completion family's own repro\n--- response ---\n{}",
        lines.join("\n")
    );
    let pins = model_equalities(&lines);
    assert!(
        pins.contains("(= a1 ((as const"),
        "`a1` must be published as the completed constant array\n--- \
         response ---\n{}",
        lines.join("\n")
    );
    let values = lines.join("\n");
    // The quantifier forces `a1` to read `#b1` everywhere, so both `select`s
    // must answer `#b1` — the value the printed constant array carries.
    assert!(
        values.contains("((select a1 #b1000110) #b1)")
            && values.contains("((select a1 #b0101010) #b1)"),
        "`(get-value …)` must answer what `(get-model)` printed; a `select` \
         through the completed array that disagrees is a model that falsifies \
         its own script\n--- response ---\n{values}"
    );
}

/// A wrong completion is refused at a *stored* point as well as at a plain one.
///
/// The two refutation shapes beside each other: the default that would satisfy
/// the quantifier contradicts a ground `select`, and the default that would
/// satisfy the quantifier contradicts a value written by a `store`. Neither may
/// ever be published, and re-fix pass 11's own mutation table records that
/// disabling the certificate turns the first into a wrong `sat`.
#[test]
fn a_wrong_completion_is_refused_at_a_plain_and_at_a_stored_point() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (or p (= (select a #b0000011) #b1)))\n\
         (assert (= (select a #b0000001) #b0))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (check-sat)\n",
        "unsat",
        "a completion to `((as const …) #b1)` contradicts `a[1] = #b0` and \
         must be refused",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (or p (= (select a #b0000010) #b1)))\n\
         (assert (forall ((i (_ BitVec 7))) \
         (= (select (store a #b0000001 #b0) i) #b1)))\n\
         (check-sat)\n",
        "unsat",
        "the `store` writes `#b0` at `#b0000001` and the quantifier reads \
         `#b1` there; no constant default repairs that",
    );
}

/// Array extensionality still refutes a completion that a `distinct` forbids.
///
/// `(distinct a ((as const …) #b1))` beside `∀i. a[i] = #b1` is unsatisfiable
/// by extensionality, and it is the shape where publishing the completed
/// constant array would be a *wrong* `sat` rather than merely a bad model. Two
/// arrays forced equal pointwise and declared `distinct` is the same argument
/// without an array constant in the script at all.
#[test]
fn extensionality_refutes_a_completion_a_distinct_forbids() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (assert (distinct a ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1)))\n\
         (check-sat)\n",
        "unsat",
        "the only interpretation the quantifier allows is the constant array \
         the `distinct` forbids",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) (select b i))))\n\
         (assert (distinct a b))\n\
         (check-sat)\n",
        "unsat",
        "pointwise equality at every index and `distinct` cannot both hold",
    );
}

/// An `exists` under the `forall` is never certified, and the verdict it gets
/// is still correct.
///
/// `collect_maximal_quantifiers` declines the whole attempt on an `Exists`, and
/// `peel_universal` declines a quantifier that survives the chain peel. The pin
/// checks the decline costs nothing: the alternation below is *valid* over its
/// element sort (every bit-vector of width 1 is some value), so `sat` is the
/// right answer and it is reached without the completion.
#[test]
fn an_exists_under_the_forall_is_declined_without_losing_the_verdict() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (assert (forall ((i (_ BitVec 7))) (exists ((j (_ BitVec 1))) \
         (= (select a i) j))))\n\
         (assert (= (select a #b0000000) #b1))\n\
         (check-sat)\n",
        "sat",
        "the alternation is valid and the ground assertion is consistent; \
         declining the completion may not cost this verdict",
    );
}

// ---------------------------------------------------------------------------
// 3. THE ARRAY REFINEMENT (decision (35))
// ---------------------------------------------------------------------------

/// A popped array lemma is re-derived after the `pop`, and the refinement
/// counters say so.
///
/// `#P2b-59`'s fix re-spells an assertion's encoded array root; the scope
/// machinery must still retract what a scope derived. The script asserts the
/// two `ite`-spine assertions of the `#P2b-59` repro in two different orders
/// around a `push`/`pop` under one deterministic budget, and asserts that the
/// verdict is the same both times and that the refinement ran on both sides —
/// a `pop` that left the lemma set behind would report zero rounds the second
/// time.
///
/// The round *counts* deliberately are not compared: they legitimately differ
/// (41 / 129 before the `pop` against 19 / 74 after, measured), because the
/// second check starts from a different trail. What must not differ is the
/// answer.
#[test]
fn a_popped_refinement_scope_re_derives_its_lemmas() {
    let decls = "(declare-const a0 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (declare-const d (_ BitVec 1))\n";
    let first = "(assert (or (= (select (store (ite p a0 a1) #b10110111 #b1) #b11010110) \
         (select (ite p (store a1 #b10101011 #b1) (ite p a1 a1)) #b11000000)) \
         (= (bvxor (select (ite q (store a0 #b01110110 #b1) (store a0 #b11101110 #b0)) \
         #b10111010) #b0) \
         (select ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0) #b00010001))))\n";
    let second = "(assert (= (select (ite q a1 ((as const (Array (_ BitVec 8) (_ BitVec 1))) d)) \
         #b00010111) (bvxor (select (ite q (ite p a1 a1) \
         ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0)) #b11110111) #b1)))\n";
    let script = format!(
        "(set-logic ALL)\n(set-option :max-bv-embedded-checks 2000)\n\
         {decls}{first}(push 1)\n{second}(check-sat)\n(pop 1)\n{second}(check-sat)\n"
    );
    let answers = verdicts(&run(&script));
    assert_eq!(
        answers.len(),
        2,
        "both checks must answer\n--- response ---\n{answers:?}"
    );
    assert_eq!(
        answers[0], answers[1],
        "the same formula before and after a `pop` that retracted it must get \
         the same answer; a lemma that survived the `pop` or one that was \
         never re-derived would show here"
    );
    assert_eq!(
        answers[0], "unknown",
        "2,000 embedded checks is deliberately below what this script \
         consumes (30,402); the pin is about the `pop`, not the verdict"
    );
}

/// The refinement **does** reach a fixpoint — above its saturation point, and
/// only there.
///
/// Decision (35)(c) asked for `:array-refinement-rounds` identical at 500 and
/// 5,000 embedded checks. It is not: this tree reports **9 rounds / 75
/// instances at 500** and **10 / 76 at 5,000**, so the two budgets do not even
/// land in the same plateau, and `TODO.md` `#P2b-59` (e)'s sentence that the
/// criterion "is met only in the accidental sense that both land in one
/// plateau" is wrong about 500. What *is* true is that above 30,402 checks the
/// counts stop moving: 66 / 208 at both 50,000 and 200,000, with the verdict
/// `sat`.
///
/// That measurement costs ~30 s of release time per rung and minutes of a
/// debug gate's, which is why it is recorded in the report and in
/// `<scratchpad>/oxiz4/rk12/REBUILD.md` rather than asserted here. What this
/// test asserts is the cheap half the documents get wrong: **the two counts at
/// 500 and 5,000 differ**.
///
/// To close: make 500 and 5,000 agree (a real fixpoint below the saturation
/// point) and flip the `assert_ne!` below.
#[test]
fn the_round_counts_at_five_hundred_and_five_thousand_checks_differ() {
    let decls = "(declare-const a0 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (declare-const d (_ BitVec 1))\n";
    let first = "(assert (or (= (select (store (ite p a0 a1) #b10110111 #b1) #b11010110) \
         (select (ite p (store a1 #b10101011 #b1) (ite p a1 a1)) #b11000000)) \
         (= (bvxor (select (ite q (store a0 #b01110110 #b1) (store a0 #b11101110 #b0)) \
         #b10111010) #b0) \
         (select ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0) #b00010001))))\n";
    let second = "(assert (= (select (ite q a1 ((as const (Array (_ BitVec 8) (_ BitVec 1))) d)) \
         #b00010111) (bvxor (select (ite q (ite p a1 a1) \
         ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0)) #b11110111) #b1)))\n";
    let body = format!("{decls}{first}{second}");
    assert!(
        !body.contains("forall") && !body.contains("exists"),
        "this pin's whole point is that the script carries no binder"
    );
    let at = |budget: u32| -> (String, u64) {
        let script = format!(
            "(set-logic ALL)\n(set-option :max-bv-embedded-checks {budget})\n\
             {body}(check-sat)\n(get-info :all-statistics)\n"
        );
        let lines = run(&script);
        let rounds = lines
            .iter()
            .find(|line| line.contains(":array-refinement-rounds "))
            .and_then(|line| line.split(":array-refinement-rounds ").nth(1))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|token| token.parse().ok())
            .unwrap_or_default();
        (verdict(&lines), rounds)
    };
    let (low_verdict, low_rounds) = at(500);
    let (high_verdict, high_rounds) = at(5_000);
    assert_eq!(
        (low_verdict.as_str(), high_verdict.as_str()),
        ("unknown", "unknown"),
        "both budgets are below the 30,402 checks this script consumes"
    );
    assert!(
        low_rounds > 0 && high_rounds > 0,
        "the refinement must run at both budgets ({low_rounds} / \
         {high_rounds})"
    );
    assert_ne!(
        low_rounds, high_rounds,
        "THE HOLE IS CLOSED: the refinement now reports the same \
         {low_rounds} rounds at 500 and at 5,000 embedded checks, which is \
         decision (35)(c)'s criterion. Flip this to `assert_eq!` and correct \
         `TODO.md` `#P2b-59` (e), which today says the criterion is met in \
         the accidental sense that both budgets land in one plateau — they do \
         not: 500 is one rung below it"
    );
}
