//! Round-4 adversarial recheck, pass 13 — the pins it leaves in the tree,
//! inverted by re-fix pass 14 where the holes closed.
//!
//! Two kinds of test live here.
//!
//! * **Inverted HOLE pins** (sections 1–6 and 13).  Recheck 13 left them green
//!   as holes; re-fix pass 14 (decisions (54)–(56), (60)) closed each one at
//!   the root, and each now asserts the correct behaviour: an `exists` with no
//!   universal beside it publishes its Skolem witness, an `exists` under a
//!   negation is completed as the universal it is, four arrays under
//!   independent binders are completed per array, an `Int` / `Real` index
//!   completes with a pooled default over the candidate's points, a
//!   datatype-indexed array prints its entries, `#P2b-70`'s five order-guard
//!   spellings are refuted, and numeric constants the arithmetic solver never
//!   constrained print distinct values.  Every model is replayed.  Section 14
//!   pins the net under all of it (decision (54)(i)): a model no certificate
//!   covers is withheld, never printed.
//! * **Regression guards** (sections 7–12) assert the correct behaviour of
//!   what this recheck attacked and found holding.
//!
//! Every repro is carried VERBATIM (the out-of-tree copies are under
//! `<scratchpad>/oxiz4/recheck13/atk/`, not needed to run anything here).
//! Every model is judged by REPLAY: the published values are pinned back into
//! a quantifier-free script that asks whether the quantifier holds at every
//! point of its index sort (or, over `Int` / `Real`, at the named points and
//! at one point no term names), so no assertion trusts the solver's own
//! reading of its model.  No test installs a wall clock (decision (16)).

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// Every `sat`/`unsat`/`unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

/// The last `sat`/`unsat`/`unknown` line, or `"none"`.
fn verdict(lines: &[String]) -> String {
    verdicts(lines)
        .last()
        .cloned()
        .unwrap_or_else(|| "none".to_string())
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

/// Replay a published model: `decls`, the model's values pinned, and
/// `claim`, one quantifier-free assertion; the verdict of that script.
fn replay(decls: &str, lines: &[String], claim: &str) -> String {
    let pins = model_equalities(lines);
    assert!(
        !pins.is_empty(),
        "the replay needs a published model (in a HOLE pin: THE HOLE IS CLOSED \
         or the model is now withheld — re-derive the pin)\n--- response ---\n{}",
        lines.join("\n")
    );
    let script = format!("(set-logic ALL)\n{decls}{pins}(assert {claim})\n(check-sat)\n");
    verdict(&run(&script))
}

/// `(and body(p₀) … body(pₙ))` over every point of `(_ BitVec width)`.
fn at_every_point(width: u32, body: impl Fn(&str) -> String) -> String {
    let parts: Vec<String> = points(width).iter().map(|p| body(p)).collect();
    format!("(and {})", parts.join(" "))
}

/// `(or body(p₀) … body(pₙ))` over every point of `(_ BitVec width)`.
fn at_some_point(width: u32, body: impl Fn(&str) -> String) -> String {
    let parts: Vec<String> = points(width).iter().map(|p| body(p)).collect();
    format!("(or {})", parts.join(" "))
}

/// Run `script` (which ends in `(check-sat)` + `(get-model)`), require `sat`,
/// and return the response.
fn sat_with_model(name: &str, script: &str) -> Vec<String> {
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "`{name}` is satisfiable and this tree decides it (in a HOLE pin: THE \
         HOLE IS CLOSED or the model is now withheld — re-derive the pin)\n\
         --- script ---\n{script}--- response ---\n{}",
        lines.join("\n")
    );
    lines
}

// ---------------------------------------------------------------------------
// 1. INVERTED (was HOLE) — an `exists` with no universal beside it publishes
//    its Skolem witness.
//
//    `(assert (exists ((i (_ BitVec 7))) (= (select a i) #b10)))` alone
//    answered the correct `sat` and published `a = ((as const …) #b00)`: no
//    point held `#b10`.  The `exists` is Skolemised to `a[sk] = #b10`, and the
//    Skolem constant `sk` occurs nowhere a theory values it, so the read had
//    no position and the renderer dropped it.  Re-fix pass 14 (`#P2b-71`'s
//    fresh values, `context::model_fmt::published`) gives the Skolem a value
//    of its own and the witness is printed.  The same over `Int`.
//    recheck13/atk/g01_exists_only.smt2, g02.
// ---------------------------------------------------------------------------

#[test]
fn an_exists_only_array_script_publishes_its_skolem_witness() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 2)))\n";
    let script = format!(
        "(set-logic ALL)\n{decls}\
         (assert (exists ((i (_ BitVec 7))) (= (select a i) #b10)))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("g01_exists_only", &script);
    let holds = at_some_point(7, |p| format!("(= (select a {p}) #b10)"));
    assert_eq!(
        replay(decls, &lines, &holds),
        "sat",
        "the published model has a point where `a[i] = #b10`\n--- response ---\n{}",
        lines.join("\n")
    );

    let decls_int = "(declare-const a (Array Int Int))\n";
    let script_int = format!(
        "(set-logic ALL)\n{decls_int}(assert (exists ((i Int)) (= (select a i) 5)))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines_int = sat_with_model("g02_exists_only_int", &script_int);
    let pinned = model_equalities(&lines_int);
    let witness = stored_point_with_value(&pinned, " 5)")
        .unwrap_or_else(|| panic!("the published `a` stores a `5`\n{}", lines_int.join("\n")));
    assert_eq!(
        replay(
            decls_int,
            &lines_int,
            &format!("(= (select a {witness}) 5)")
        ),
        "sat",
        "the published `a` reads `5` at its witness `{witness}`\n{}",
        lines_int.join("\n")
    );
}

/// The index of a `store` whose value ends with `value_suffix` in a model's
/// pinned equalities (`… IDX VALUE)`), for a replay at that point.
fn stored_point_with_value(pinned: &str, value_suffix: &str) -> Option<String> {
    let position = pinned.find(value_suffix)?;
    let head = pinned.get(..position)?;
    let index = head.rsplit(' ').next()?;
    Some(index.to_string())
}

// ---------------------------------------------------------------------------
// 2. INVERTED (was HOLE) — an `exists` under a negation is completed as the
//    universal it is.
//
//    `∀i. a[i] = #b1` beside `¬∃j. a[j] = #b0`, `(=> (∃j. a[j] = #b0) false)`
//    and `¬∃j. ¬a[j]` over `(Array (_ BitVec 8) Bool)` were declined by the
//    completion ("an `exists` not at positive polarity") and published models
//    false at every point the search had not pinned.  Re-fix pass 14
//    (`array_completion_certify::polarity`) certifies a negated `exists` false
//    through its dual `∀j. ¬φ`, and all three publish the constant array.
//    recheck13/atk/c04, d02, d05.
// ---------------------------------------------------------------------------

#[test]
fn a_negated_exists_is_completed_as_the_universal_it_is() {
    let decls7 = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n";
    let all_one = at_every_point(7, |p| format!("(= (select a {p}) #b1)"));
    for (name, assertions) in [
        (
            "c04_exists_negated",
            "(assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
             (assert (not (exists ((j (_ BitVec 7))) (= (select a j) #b0))))\n",
        ),
        (
            "d02_exists_under_implication",
            "(assert (=> (exists ((j (_ BitVec 7))) (= (select a j) #b0)) false))\n",
        ),
    ] {
        let script = format!("(set-logic ALL)\n{decls7}{assertions}(check-sat)\n(get-model)\n");
        let lines = sat_with_model(name, &script);
        assert_eq!(
            replay(decls7, &lines, &all_one),
            "sat",
            "`{name}`: the published `a` is `#b1` at all 128 points\n--- response ---\n{}",
            lines.join("\n")
        );
    }

    let decls8 = "(declare-const a (Array (_ BitVec 8) Bool))\n";
    let script = format!(
        "(set-logic ALL)\n{decls8}\
         (assert (not (exists ((j (_ BitVec 8))) (not (select a j)))))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("d05_not_exists_distinct", &script);
    let all_true = at_every_point(8, |p| format!("(select a {p})"));
    assert_eq!(
        replay(decls8, &lines, &all_true),
        "sat",
        "`d05`: the published Bool array is `true` at all 256 points\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 3. INVERTED (was HOLE) — four arrays, each read under its own binder, are
//    completed per array.
//
//    `Goal::build` declined the goal because it held four arrays, more than
//    `MAX_COMPLETED_ARRAYS` = 3, and the `sat` kept a candidate in which every
//    array was `#b0` off the points the search named.  The bound is on a
//    product search, and the product is only needed over arrays one assertion
//    mentions together: re-fix pass 14 (`array_completion_certify::groups`)
//    searches each independent group on its own and certifies the union over
//    every assertion.  recheck13/atk/d03_four_simple.smt2.
// ---------------------------------------------------------------------------

#[test]
fn four_arrays_under_independent_binders_are_completed_per_array() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
                 (declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n\
                 (declare-const c (Array (_ BitVec 7) (_ BitVec 1)))\n\
                 (declare-const d (Array (_ BitVec 7) (_ BitVec 1)))\n";
    let script = format!(
        "(set-logic ALL)\n{decls}\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select b i) #b1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select c i) #b1)))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select d i) #b1)))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("d03_four_simple", &script);
    let holds = at_every_point(7, |p| {
        format!(
            "(= (select a {p}) #b1) (= (select b {p}) #b1) (= (select c {p}) #b1) \
             (= (select d {p}) #b1)"
        )
    });
    assert_eq!(
        replay(decls, &lines, &holds),
        "sat",
        "all four published arrays are `#b1` at every point\n--- response ---\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 4. INVERTED (was HOLE) — over an `Int` or `Real` index, a universal no
//    constant array satisfies is completed with the candidate's points over a
//    pooled default.
//
//    `∀i:Int. a[i] ≥ 3` beside `a[5] = 4`, `a[6] = 3`; `∀i:Int. a[i] = b[i] + 1`
//    beside `b[3] = 7`; `∀x:Real. a[x] > 0` beside `a[0.5] = 2`, `a[1.5] = 1`:
//    the completion tried constants only over an infinite index and published
//    each candidate's reads over the sort default `0`.  Re-fix pass 14
//    (`array_completion_certify::infinite`, `split`) keeps the candidate's
//    points, draws the default from the script's constants and their `± 1`,
//    and certifies the universal pointwise (the named points exactly, the
//    unnamed region through the default).  Each `Int` model is replayed at
//    the named points and at `1000003`, a point no term names; the `Real`
//    model at its named points and by its published default.
//    recheck13/atk/d01, d06, d08.
// ---------------------------------------------------------------------------

#[test]
fn an_int_or_real_index_completes_with_a_pooled_default() {
    let cases: [(&str, &str, &str, &str); 2] = [
        (
            "d01_int_nonconst",
            "(declare-const a (Array Int Int))\n",
            "(assert (forall ((i Int)) (>= (select a i) 3)))\n\
             (assert (= (select a 5) 4))\n(assert (= (select a 6) 3))\n",
            "(and (>= (select a 1000003) 3) (= (select a 5) 4) (= (select a 6) 3))",
        ),
        (
            "d08_int_two_arrays",
            "(declare-const a (Array Int Int))\n(declare-const b (Array Int Int))\n",
            "(assert (forall ((i Int)) (= (select a i) (+ (select b i) 1))))\n\
             (assert (= (select b 3) 7))\n",
            "(and (= (select a 1000003) (+ (select b 1000003) 1)) \
             (= (select a 3) (+ (select b 3) 1)) (= (select b 3) 7))",
        ),
    ];
    for (name, decls, assertions, claim) in cases {
        let script = format!("(set-logic ALL)\n{decls}{assertions}(check-sat)\n(get-model)\n");
        let lines = sat_with_model(name, &script);
        assert!(
            !model_equalities(&lines).contains("1000003"),
            "the replay point must be one no term names"
        );
        assert_eq!(
            replay(decls, &lines, claim),
            "sat",
            "`{name}`: the published model holds the universal at a point no \
             term names and every ground assertion\n--- response ---\n{}",
            lines.join("\n")
        );
    }

    let decls = "(declare-const a (Array Real Int))\n";
    let script = format!(
        "(set-logic ALL)\n{decls}(assert (forall ((x Real)) (> (select a x) 0)))\n\
         (assert (= (select a 0.5) 2))\n(assert (= (select a 1.5) 1))\n(check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("d06_real_index", &script);
    // A quantifier-free replay of a `Real`-indexed store chain answers
    // `unknown` on every build (`#P2b-73`), so the published chain is read
    // directly: its default and every stored value hold `a[x] > 0`, and it
    // reads `2` at `1/2` and `1` at `3/2`.
    let pinned = model_equalities(&lines);
    let value = pinned
        .split("(assert (= a ")
        .nth(1)
        .and_then(|rest| rest.rsplit_once("))"))
        .map(|(value, _)| value.to_string())
        .unwrap_or_else(|| panic!("a published `a`\n{}", lines.join("\n")));
    let (default, entries) = real_int_chain(&value)
        .unwrap_or_else(|| panic!("a store chain over a constant array: {value}"));
    assert!(default > 0, "the default holds `a[x] > 0`: {value}");
    assert!(
        entries.iter().all(|&(_, stored)| stored > 0),
        "every stored value holds `a[x] > 0`: {value}"
    );
    let read = |at: (i64, i64)| -> i64 {
        entries
            .iter()
            .find(|&&(index, _)| index == at)
            .map_or(default, |&(_, stored)| stored)
    };
    assert_eq!(read((1, 2)), 2, "a[1/2] = 2: {value}");
    assert_eq!(read((3, 2)), 1, "a[3/2] = 1: {value}");
}

/// One stored entry of a `(Array Real Int)` value: the index as a
/// `(numerator, denominator)` pair, and the stored value.
type RealIntEntry = ((i64, i64), i64);

/// `(default, entries)` of a printed `(Array Real Int)` value — a `store`
/// chain over `((as const (Array Real Int)) d)`, outermost store first — or
/// `None`.
fn real_int_chain(text: &str) -> Option<(i64, Vec<RealIntEntry>)> {
    let tokens: Vec<String> = text
        .replace('(', " ( ")
        .replace(')', " ) ")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    let (tree, _) = parse_sexp(&tokens, 0)?;
    let mut entries: Vec<RealIntEntry> = Vec::new();
    let mut current = &tree;
    loop {
        match current {
            Sexp::List(items) if items.first() == Some(&Sexp::Atom("store".to_string())) => {
                let index = real_value(items.get(2)?)?;
                let value = int_value(items.get(3)?)?;
                if !entries.iter().any(|&(named, _)| named == index) {
                    entries.push((index, value));
                }
                current = items.get(1)?;
            }
            Sexp::List(items) if items.len() == 2 => {
                return Some((int_value(items.get(1)?)?, entries));
            }
            _ => return None,
        }
    }
}

#[derive(PartialEq, Debug)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

fn parse_sexp(tokens: &[String], at: usize) -> Option<(Sexp, usize)> {
    let token = tokens.get(at)?;
    if token != "(" {
        return Some((Sexp::Atom(token.clone()), at + 1));
    }
    let mut items: Vec<Sexp> = Vec::new();
    let mut next = at + 1;
    while tokens.get(next)? != ")" {
        let (item, after) = parse_sexp(tokens, next)?;
        items.push(item);
        next = after;
    }
    Some((Sexp::List(items), next + 1))
}

fn int_value(node: &Sexp) -> Option<i64> {
    match node {
        Sexp::Atom(text) => text.parse().ok(),
        Sexp::List(items) if items.first() == Some(&Sexp::Atom("-".to_string())) => {
            int_value(items.get(1)?).map(|value| -value)
        }
        _ => None,
    }
}

fn real_value(node: &Sexp) -> Option<(i64, i64)> {
    match node {
        Sexp::Atom(text) => {
            let whole = text.strip_suffix(".0")?;
            Some((whole.parse().ok()?, 1))
        }
        Sexp::List(items) if items.first() == Some(&Sexp::Atom("/".to_string())) => {
            Some((int_value(items.get(1)?)?, int_value(items.get(2)?)?))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 5. INVERTED (was HOLE) — QUANTIFIER-FREE: a datatype-indexed array's
//    `(get-model)` prints its entries (`#P2b-72`).
//
//    `(assert (= (select a red) 1))` over `(Array C Int)` printed
//    `a = ((as const (Array C Int)) 0)`: a constructor-valued index was
//    neither a literal nor an uninterpreted-sort class, so
//    `index_position_string` named no position and the entry was dropped.  It
//    now spells the constructor value.  recheck13/atk/e08, e09.
// ---------------------------------------------------------------------------

#[test]
fn a_datatype_indexed_array_model_prints_its_entries() {
    let decls = "(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
                 (declare-const a (Array C Int))\n";
    let script = format!(
        "(set-logic QF_ALL)\n{decls}(assert (= (select a red) 1))\n\
         (check-sat)\n(get-model)\n(get-value ((select a red) (select a green)))\n"
    );
    let lines = sat_with_model("e08_dt_model_qf", &script);
    let joined = lines.join("\n");
    assert!(
        joined.contains("((select a red) 1)") && joined.contains("((select a green) 0)"),
        "(get-value) reads the printed interpretation at both points\n{joined}"
    );
    assert_eq!(
        replay(decls, &lines, "(= (select a red) 1)"),
        "sat",
        "the published model satisfies `(= (select a red) 1)`\n{joined}"
    );

    let decls2 = "(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
                  (declare-const a (Array C Int))\n(declare-const b (Array C Int))\n";
    let script2 = format!(
        "(set-logic QF_ALL)\n{decls2}(assert (= b (store a green 5)))\n\
         (assert (= (select a red) 1))\n(check-sat)\n(get-model)\n"
    );
    let lines2 = sat_with_model("e09_dt_model_store", &script2);
    assert_eq!(
        replay(
            decls2,
            &lines2,
            "(and (= b (store a green 5)) (= (select a red) 1))"
        ),
        "sat",
        "`e09`: the published `a` / `b` satisfy both assertions\n{}",
        lines2.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 6. INVERTED (was HOLE) — `#P2b-70`: a bit-vector order guard over an array
//    pinned by a ground equality is refuted in all five spellings.
//
//    HEAD `c702310` refuted them and re-fix pass 12's tree answered `unknown`:
//    the counterexample search instantiated the bound variable with the
//    values the candidate model gave its sort (`#b0000000`, `#b0000101`), both
//    outside the guard, so no round produced the refuting instance.  Re-fix
//    pass 14 (`mbqi::counterexample::guard_points`) tries the points the
//    body's guards name (each literal and its `± 1`) where the model's values
//    find no counterexample.  VERBATIM from `$R/corpus/named/wsigned.smt2` and
//    `$R/fix13b/g_{uge,ugt5,ule,nult}.smt2`; the in-test oracle is the guard's
//    own witness: `a` is `#b0` at `#b1111111` (not `#b0000101`).
// ---------------------------------------------------------------------------

#[test]
fn p2b70_a_bit_vector_order_guard_over_a_pinned_array_is_refuted() {
    let pinned = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
                  (assert (= a (store ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0) \
                  #b0000101 #b1)))\n";
    let oracle =
        format!("(set-logic ALL)\n{pinned}(assert (= (select a #b1111111) #b1))\n(check-sat)\n");
    assert_eq!(
        verdict(&run(&oracle)),
        "unsat",
        "in-test oracle: the pinned array is #b0 at #b1111111"
    );
    for (name, guard) in [
        ("wsigned", "(bvslt i #b0000000)"),
        ("g_uge", "(bvuge i #b1000000)"),
        ("g_ugt5", "(bvugt i #b0000101)"),
        ("g_ule", "(bvule #b1000000 i)"),
        ("g_nult", "(not (bvult i #b1000000))"),
    ] {
        let script = format!(
            "(set-logic ALL)\n{pinned}\
             (assert (forall ((i (_ BitVec 7))) (=> {guard} (= (select a i) #b1))))\n\
             (check-sat)\n"
        );
        assert_eq!(
            verdict(&run(&script)),
            "unsat",
            "`{name}`: the order-guarded universal is false at #b1111111"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. GUARD — decision (40): a completion whose default the ground select
//    contradicts is pinned at that point, and the model holds everywhere.
//    HEAD, `c4b04b7` and 0.3.3 answer `unknown`.  recheck13/atk/a01.
// ---------------------------------------------------------------------------

#[test]
fn a_default_the_ground_select_contradicts_is_pinned_at_that_point() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
                 (declare-const k (_ BitVec 7))\n";
    let script = format!(
        "(set-logic ALL)\n{decls}\
         (assert (forall ((i (_ BitVec 7))) (=> (distinct i k) (= (select a i) #b1))))\n\
         (assert (= (select a k) #b0))\n(check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("a01_ground_contradicts_default", &script);
    let holds = at_every_point(7, |p| {
        format!("(=> (distinct {p} k) (= (select a {p}) #b1))")
    });
    let claim = format!("(and {holds} (= (select a k) #b0))");
    assert_eq!(
        replay(decls, &lines, &claim),
        "sat",
        "the published model satisfies both assertions at all 128 points\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 8. GUARD — decision (40): two `check-sat`s, only the second quantified; the
//    completion under a guard `i > 3`; `(get-value)` at stored and unstored
//    points after a guarded completion with a 2-bit element.
//    recheck13/atk/a03, a07.
// ---------------------------------------------------------------------------

#[test]
fn a_quantified_second_check_and_a_guarded_completion_replay_everywhere() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
                 (declare-const x (_ BitVec 7))\n";
    let script = format!(
        "(set-logic ALL)\n{decls}(assert (= (select a x) #b0))\n(check-sat)\n\
         (assert (forall ((i (_ BitVec 7))) (=> (bvugt i #b0000011) (= (select a i) #b1))))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = run(&script);
    assert_eq!(verdicts(&lines), ["sat", "sat"], "{}", lines.join("\n"));
    let holds = at_every_point(7, |p| {
        format!("(=> (bvugt {p} #b0000011) (= (select a {p}) #b1))")
    });
    let claim = format!("(and {holds} (= (select a x) #b0))");
    assert_eq!(replay(decls, &lines, &claim), "sat", "{}", lines.join("\n"));

    let decls2 = "(declare-const a (Array (_ BitVec 7) (_ BitVec 2)))\n";
    let script2 = format!(
        "(set-logic ALL)\n{decls2}\
         (assert (forall ((i (_ BitVec 7))) (=> (bvuge i #b0010000) (= (select a i) #b10))))\n\
         (assert (= (select a #b0000011) #b01))\n(check-sat)\n\
         (get-value ((select a #b0000011) (select a #b0010000) (select a #b1111111)))\n\
         (get-model)\n"
    );
    let lines2 = sat_with_model("a07_getvalue_points", &script2);
    let joined = lines2.join("\n");
    for expected in [
        "((select a #b0000011) #b01)",
        "((select a #b0010000) #b10)",
        "((select a #b1111111) #b10)",
    ] {
        assert!(joined.contains(expected), "missing {expected}\n{joined}");
    }
    let holds2 = at_every_point(7, |p| {
        format!("(=> (bvuge {p} #b0010000) (= (select a {p}) #b10))")
    });
    let claim2 = format!("(and {holds2} (= (select a #b0000011) #b01))");
    assert_eq!(replay(decls2, &lines2, &claim2), "sat", "{joined}");
}

// ---------------------------------------------------------------------------
// 9. GUARD — decision (41): pins from two arrays through one binder; a
//    ground-select pin the guarded body contradicts is refuted; a pin the
//    body contradicts through a symbolic guard bound is refuted.
//    recheck13/atk/b01, b02, b07.
// ---------------------------------------------------------------------------

#[test]
fn pins_from_two_arrays_through_one_binder_and_contradicted_pins() {
    let decls = "(declare-const a (Array (_ BitVec 7) (_ BitVec 2)))\n\
                 (declare-const b (Array (_ BitVec 7) (_ BitVec 2)))\n";
    let script = format!(
        "(set-logic ALL)\n{decls}\
         (assert (forall ((i (_ BitVec 7))) (= (select a i) (bvnot (select b i)))))\n\
         (assert (= (select a #b0000011) #b01))\n(assert (= (select b #b0000101) #b00))\n\
         (assert (= (select a #b1000000) #b00))\n(check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("b02_two_arrays_one_binder", &script);
    let holds = at_every_point(7, |p| format!("(= (select a {p}) (bvnot (select b {p})))"));
    let claim = format!(
        "(and {holds} (= (select a #b0000011) #b01) (= (select b #b0000101) #b00) \
         (= (select a #b1000000) #b00))"
    );
    assert_eq!(replay(decls, &lines, &claim), "sat", "{}", lines.join("\n"));

    for (name, body) in [
        (
            "b01_pin_contradicted",
            "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n\
             (assert (forall ((i (_ BitVec 7))) (=> (bvult i #b0010000) (= (select a i) #b1))))\n\
             (assert (= (select a #b0000011) #b0))\n",
        ),
        (
            "b07_symbolic_guard_bound",
            "(declare-const a (Array (_ BitVec 7) (_ BitVec 2)))\n\
             (declare-const k (_ BitVec 7))\n\
             (assert (forall ((i (_ BitVec 7))) (=> (bvuge i k) (= (select a i) #b10))))\n\
             (assert (= k #b0000100))\n(assert (= (select a #b1111111) #b01))\n",
        ),
    ] {
        let lines = run(&format!("(set-logic ALL)\n{body}(check-sat)\n"));
        assert_eq!(verdict(&lines), "unsat", "`{name}` is unsatisfiable");
    }
}

// ---------------------------------------------------------------------------
// 10. GUARD — `#P2b-65` beyond the recheck-12 156: eleven further `Int`-index
//     guard spellings (a `not` inside an `and`, strict orders, `+ 1`, unary
//     minus, a Boolean `=` against `false`, an `ite` guard, an inner `=>`)
//     over a bit-vector and a `Bool` element — all unsatisfiable pigeonholes,
//     all refuted on this tree; and `(=> (<= i j) (=> (<= j i) …))`, which is
//     valid (it only constrains `i = j`) and must stay `sat`.
// ---------------------------------------------------------------------------

#[test]
fn eleven_more_int_index_pigeonhole_spellings_are_refuted() {
    let bodies = [
        "(=> (and (<= i j) (not (= i j))) (distinct (select a i) (select a j)))",
        "(=> (not (>= i j)) (distinct (select a i) (select a j)))",
        "(=> (not (<= j i)) (distinct (select a i) (select a j)))",
        "(=> (or (< i j) (< j i)) (distinct (select a i) (select a j)))",
        "(=> (<= (+ i 1) j) (distinct (select a i) (select a j)))",
        "(=> (<= (- j) (- i)) (=> (not (= i j)) (distinct (select a i) (select a j))))",
        "(=> (<= i j) (=> (not (= i j)) (distinct (select a i) (select a j))))",
        "(=> (= (= i j) false) (distinct (select a i) (select a j)))",
        "(=> (ite (= i j) false true) (distinct (select a i) (select a j)))",
        "(=> (=> (= i j) false) (distinct (select a i) (select a j)))",
        "(=> (not (and (>= i j) (<= i j))) (distinct (select a i) (select a j)))",
    ];
    for (element, ground) in [
        ("(_ BitVec 1)", "(= (select a 0) #b0)"),
        ("Bool", "(not (select a 0))"),
    ] {
        for body in bodies {
            let script = format!(
                "(set-logic ALL)\n(declare-const a (Array Int {element}))\n\
                 (assert (forall ((i Int) (j Int)) {body}))\n(assert {ground})\n(check-sat)\n"
            );
            let lines = run(&script);
            assert_eq!(
                verdict(&lines),
                "unsat",
                "an Int-index pigeonhole is unsatisfiable\n{script}{}",
                lines.join("\n")
            );
        }
        let valid = format!(
            "(set-logic ALL)\n(declare-const a (Array Int {element}))\n\
             (assert (forall ((i Int) (j Int)) (=> (not (not (<= i j))) \
             (=> (not (not (<= j i))) (= (select a i) (select a j))))))\n\
             (assert {ground})\n(check-sat)\n"
        );
        assert_eq!(verdict(&run(&valid)), "sat", "{valid}");
    }
}

// ---------------------------------------------------------------------------
// 11. GUARD — `#P2b-61` / `#P2b-64`: one constructor with different fields, and
//     a nested application, are distinct as array indices (quantifier-free).
//     0.3.3 and `c4b04b7` answer a wrong `sat` to both; HEAD and this tree
//     `unsat`.  recheck13/atk/e01, e02.
// ---------------------------------------------------------------------------

#[test]
fn one_constructor_with_different_fields_is_distinct_as_an_array_index() {
    let list = "(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
                (declare-const a (Array L Int))\n\
                (assert (= a (store ((as const (Array L Int)) 0) (cons 0 nil) 1)))\n";
    for other in ["(cons 1 nil)", "(cons 0 (cons 0 nil))"] {
        let script =
            format!("(set-logic ALL)\n{list}(assert (= (select a {other}) 1))\n(check-sat)\n");
        assert_eq!(verdict(&run(&script)), "unsat", "{script}");
    }
}

// ---------------------------------------------------------------------------
// 12. GUARD — decision (45): circuits defined while operands are fixed two
//     scopes deep, re-used after the pops at the opposite polarity, and
//     arrays interleaved with bit-vector atoms across three levels; every
//     answer checked by hand (brute force at widths 3-4).
//     recheck13/atk/h01, h02.
// ---------------------------------------------------------------------------

#[test]
fn root_defined_circuits_keep_every_verdict_across_three_levels() {
    let script = "(set-logic QF_BV)\n(declare-const x (_ BitVec 4))\n\
         (declare-const y (_ BitVec 4))\n(declare-const z (_ BitVec 4))\n\
         (push 1)\n(push 1)\n(assert (= x #x5))\n(assert (bvult z #x3))\n(check-sat)\n\
         (assert (= y (bvadd x #x1)))\n(assert (= z (bvmul y #x3)))\n(check-sat)\n\
         (pop 1)\n(assert (distinct y (bvadd x #x1)))\n(assert (= x #x5))\n\
         (assert (= z (bvmul y #x3)))\n(check-sat)\n(pop 1)\n\
         (assert (= y (bvadd x #x1)))\n(assert (= y #x0))\n(check-sat)\n\
         (assert (= x #x5))\n(check-sat)\n(push 1)\n(check-sat)\n(pop 1)\n(check-sat)\n";
    assert_eq!(
        verdicts(&run(script)),
        ["sat", "sat", "sat", "sat", "unsat", "unsat", "unsat"],
        "h01"
    );
    let arrays = "(set-logic QF_AUFBV)\n\
         (declare-const a (Array (_ BitVec 3) (_ BitVec 3)))\n\
         (declare-const i (_ BitVec 3))\n(declare-const v (_ BitVec 3))\n\
         (push 1)\n(assert (= (select a i) (bvadd v #b001)))\n(push 1)\n\
         (assert (= v #b111))\n(assert (= i #b010))\n(check-sat)\n(push 1)\n\
         (assert (distinct (select (store a i v) #b010) #b111))\n(check-sat)\n(pop 1)\n\
         (assert (= (select (store a #b011 v) i) #b000))\n(check-sat)\n(pop 1)\n\
         (assert (= (select a i) #b000))\n(check-sat)\n(pop 1)\n\
         (assert (= (select a i) (bvadd v #b001)))\n\
         (assert (not (= (select a i) (bvadd v #b001))))\n(check-sat)\n";
    assert_eq!(
        verdicts(&run(arrays)),
        ["sat", "unsat", "sat", "sat", "unsat"],
        "h02"
    );
}

// ---------------------------------------------------------------------------
// 13. INVERTED (was HOLE) — QUANTIFIER-FREE: numeric constants the arithmetic
//     solver never constrained print distinct values (`#P2b-71`).
//
//     `(= (f k) 5)` beside `(= (f j) 6)` over `Int` printed `k = 0`, `j = 0`
//     and `f` the constant `5`; the array spelling printed `k = j = 0` and
//     `a = (store ((as const (Array Int Int)) 0) 0 5)`, and
//     `(get-value ((= k j)))` answered `true`.  The model builder filled both
//     with the sort default because no theory valued them, and two classes the
//     congruence closure kept apart printed one value.  Re-fix pass 14 records
//     such an entry as a default (`Model::set_default`) and the printer gives
//     each class no theory valued a fresh value (`context::model_fmt::
//     published`), so distinct classes print distinct values.
//     recheck13/atk/g08, g09, g11, g13.
// ---------------------------------------------------------------------------

#[test]
fn numeric_constants_seen_only_as_arguments_get_distinct_values() {
    let decls = "(declare-const a (Array Int Int))\n(declare-const k Int)\n\
                 (declare-const j Int)\n";
    let script = format!(
        "(set-logic QF_ALIA)\n{decls}(assert (= (select a k) 5))\n\
         (assert (= (select a j) 6))\n(check-sat)\n(get-model)\n\
         (get-value (k j (select a k) (select a j) (= k j)))\n"
    );
    let lines = sat_with_model("g08_qf_unvalued_index_int", &script);
    assert_eq!(
        replay(decls, &lines, "(and (= (select a k) 5) (= (select a j) 6))"),
        "sat",
        "the published `k`, `j` and `a` satisfy both reads\n--- response ---\n{}",
        lines.join("\n")
    );
    assert!(
        lines.iter().any(|line| line.contains("((= k j) false)")),
        "`(get-value)` reads the same distinct values `(get-model)` prints\n{}",
        lines.join("\n")
    );

    for (name, sort, five, six) in [
        ("g11_qf_uf_two_points", "Int", "5", "6"),
        ("g13_real", "Real", "5.0", "6.0"),
    ] {
        let uf_decls = format!("(declare-const k {sort})\n(declare-const j {sort})\n");
        let uf_script = format!(
            "(set-logic ALL)\n(declare-fun f ({sort}) {sort})\n{uf_decls}\
             (assert (= (f k) {five}))\n(assert (= (f j) {six}))\n(check-sat)\n(get-model)\n"
        );
        let uf_lines = sat_with_model(name, &uf_script);
        assert_eq!(
            replay(&uf_decls, &uf_lines, "(distinct k j)"),
            "sat",
            "`{name}`: the published `k` and `j` are distinct, as `(f k) = {five}`, \
             `(f j) = {six}` requires\n--- response ---\n{}",
            uf_lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// 14. GUARD — decision (54)(i): a published model is certified or absent,
//     never falsifying.
//
//     Where the completion declines, the candidate is rendered as
//     `(get-model)` would print it, parsed back and certified against every
//     assertion; one that does not certify is withheld.  The fuzzed script
//     below (fuzz_qc seed 29093021, script 2315) links four arrays through
//     one chain of assertions — a group larger than the completion's search
//     bound — and its candidate prints `a1` false at points `¬∃i. a1[i] =
//     false` forbids: `(get-model)` answers the certificate's error, and so
//     does `(get-value)` over a term that reads an array of that model, while
//     a term that reads none still answers.  Beside it, three models the
//     certificate reads correctly although no completion produced them: an
//     uninterpreted-sort index (`@uc_U_n` witnesses, recheck13/atk/s03), a
//     datatype index, and the group-wise completion of `¬∃i. a0[i] = #b0`
//     beside two `exists` over `a1` (fuzz_qc seed 29093020, script 548).
// ---------------------------------------------------------------------------

#[test]
fn a_model_the_certificate_refuses_is_withheld_and_never_printed_falsifying() {
    let script = "(set-logic ALL)\n\
         (declare-const a0 (Array (_ BitVec 7) Bool))\n\
         (declare-const a1 (Array (_ BitVec 7) Bool))\n\
         (declare-const a2 (Array (_ BitVec 7) Bool))\n\
         (declare-const a3 (Array (_ BitVec 7) Bool))\n\
         (declare-const a4 (Array (_ BitVec 7) Bool))\n\
         (declare-const k (_ BitVec 7))\n\
         (assert (= (select a3 (_ bv37 7)) (select a1 (_ bv44 7))))\n\
         (assert (forall ((i (_ BitVec 7))) (= (select a0 i) (select a2 i))))\n\
         (assert (not (exists ((i (_ BitVec 7))) (= (select a1 i) false))))\n\
         (assert (= a2 (store a3 (_ bv15 7) false)))\n\
         (assert (= k #b0000011))\n\
         (check-sat)\n(get-model)\n(get-value ((select a1 #b0000001)))\n(get-value (k))\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let joined = lines.join("\n");
    let model = lines
        .iter()
        .find(|line| line.contains("model not certified") || line.starts_with("(model"))
        .cloned()
        .unwrap_or_default();
    if model.starts_with("(model") {
        // Never a falsifying model: whatever is printed holds `¬∃i. a1[i] =
        // false` at every point.  Printed at all, the completeness residue
        // this pin records (a group of more than three linked arrays) is gone.
        let all_true = at_every_point(7, |p| format!("(select a1 {p})"));
        let decls = "(declare-const a1 (Array (_ BitVec 7) Bool))\n";
        assert_eq!(replay(decls, &lines, &all_true), "sat", "{joined}");
        panic!(
            "THE HOLE IS CLOSED (completeness, TODO `#P2b-51` residue): a group \
             of four linked arrays is now completed and its model certified. \
             Keep the replay above, drop this panic and the residue line\n{joined}"
        );
    }
    assert!(
        model.starts_with("(error \"model not certified: "),
        "a model the certificate refuses answers the certificate's error\n{joined}"
    );
    let errors = lines
        .iter()
        .filter(|line| line.starts_with("(error \"model not certified: "))
        .count();
    assert_eq!(
        errors, 2,
        "`(get-value)` over an array of the withheld model answers the same error\n{joined}"
    );
    assert!(
        joined.contains("((k #b0000011))"),
        "a term that reads no array of the withheld model still answers\n{joined}"
    );
    // The error is one line, deterministic, and names an assertion.
    assert!(model.contains("assertion "), "{model}");
    assert_eq!(
        run(script),
        lines,
        "the withheld answer is the same on two runs"
    );
}

#[test]
fn models_the_certificate_reads_through_witnesses_constructors_and_groups() {
    // An uninterpreted index sort: the printed `@uc_U_n` witnesses are read
    // back as distinct elements (recheck13/atk/s03_declared_sort_sat).
    let usort = "(set-logic ALL)\n(declare-sort U 0)\n(declare-const u U)\n\
         (declare-const a (Array U Int))\n\
         (assert (= a (store ((as const (Array U Int)) 1) u 1)))\n\
         (assert (forall ((x U)) (= (select a x) 1)))\n(check-sat)\n(get-model)\n";
    let lines = run(usort);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    assert!(
        lines.iter().any(|line| line.starts_with("(model")),
        "the witness-bearing model is certified and printed\n{}",
        lines.join("\n")
    );

    // A datatype index: constructors are read back as constructor terms.
    let decls = "(declare-datatypes ((C 0)) (((red) (green) (blue))))\n\
                 (declare-const a (Array C Int))\n(declare-const c C)\n";
    let dt = format!(
        "(set-logic ALL)\n{decls}(assert (forall ((x C)) (>= (select a x) 0)))\n\
         (assert (= (select a red) 1))\n(assert (distinct c red))\n(check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("datatype_index_guard", &dt);
    assert_eq!(
        replay(
            decls,
            &lines,
            "(and (>= (select a red) 0) (>= (select a green) 0) (>= (select a blue) 0) \
             (= (select a red) 1) (distinct c red))"
        ),
        "sat",
        "the published model holds at every constructor\n{}",
        lines.join("\n")
    );

    // Two groups needing different treatment: a constant completion for `a0`,
    // the printed Skolem witnesses kept for `a1` (fuzz_qc 29093020 / 548).
    let decls = "(declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\n\
                 (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n";
    let groups = format!(
        "(set-logic ALL)\n{decls}\
         (assert (exists ((i (_ BitVec 7))) (= (select a1 i) #b1)))\n\
         (assert (exists ((i (_ BitVec 7))) (= (select a1 i) #b0)))\n\
         (assert (not (exists ((i (_ BitVec 7))) (= (select a0 i) #b0))))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = sat_with_model("qc_548_groups", &groups);
    let a0_ones = at_every_point(7, |p| format!("(= (select a0 {p}) #b1)"));
    let a1_one = at_some_point(7, |p| format!("(= (select a1 {p}) #b1)"));
    let a1_zero = at_some_point(7, |p| format!("(= (select a1 {p}) #b0)"));
    assert_eq!(
        replay(
            decls,
            &lines,
            &format!("(and {a0_ones} {a1_one} {a1_zero})")
        ),
        "sat",
        "the group-wise completion holds every assertion\n{}",
        lines.join("\n")
    );
}
