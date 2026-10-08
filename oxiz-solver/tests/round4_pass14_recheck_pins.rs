//! Round-4 adversarial recheck, pass 14 — the pins it left in the tree, as
//! re-fix pass 15 inverted them.
//!
//! Recheck 14 judged the tree with an independent solver (z3 4.15.4) over a
//! seeded mixed-theory corpus (`gen14.py`, seed 30093001, 3,000 scripts:
//! `QF_UFLIA` / `QF_UFLRA` / `QF_AUFLIA` with nested applications, quantified
//! UF and arrays over `Int` / bit-vectors, datatype indices, push/pop), every
//! verdict against z3's and every published model replayed.  It left five
//! holes pinned GREEN; re-fix pass 15 closed each at its root and inverted
//! its pin here:
//!
//! * §1 — a WRONG `sat` on every earlier build (0.3.3, `c4b04b7`,
//!   `c702310`, pass 14): a universal whose guard is a strict order
//!   comparison or a negated equality, `(forall ((q Int)) (=> (> q 7)
//!   false))` (`#P2b-75`).  Every spelling now answers `unsat`.
//! * §1b — a WRONG `sat` on every earlier build, quantifier-free: two list
//!   values that differ two constructors deep were equated (`#P2b-76`).
//!   Both spellings now answer `unsat`.
//! * §2 — a `sat` whose published model FALSIFIED its own quantified
//!   assertion outside the net's old `has_array_ops` scope (decision (67)).
//!   Each model is now certified, and replays.
//! * §3 — quantifier-free models that FALSIFIED their script: nested
//!   application classes printed one value (`#P2b-71`'s residue, decision
//!   (68)), a fresh value dropped a table entry (pass 14's regression
//!   `r03`, decision (64)), and two list values sharing a prefix printed as
//!   one (pass 13's regression `d03`, decision (64)).  Each now replays.
//! * §4 — a correct model withheld by the net (decision (68): the
//!   certificate reads an uninterpreted function from its printed table).
//!   It is published again, and replays.
//!
//! §5 holds the regression guards for the controls the recheck measured
//! holding (one re-derived, see there).
//!
//! Every model is judged by REPLAY: the published `define-fun` lines are
//! copied verbatim into a quantifier-free script together with a closed
//! instance of the assertion, so no assertion trusts the solver's own reading
//! of its model.  No test installs a wall clock (decision (16)); every script
//! here decides in milliseconds.

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

/// Every `(define-fun …)` line of the first published `(model …)` response,
/// verbatim, one per line; `None` when no model was published.
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

/// Replay a published model: its `define-fun` lines (plus `prelude`, for
/// sorts and datatypes the definitions mention) and one closed claim.
fn replay(prelude: &str, definitions: &str, claim: &str) -> String {
    let script = format!("(set-logic ALL)\n{prelude}{definitions}(assert {claim})\n(check-sat)\n");
    verdict(&run(&script))
}

/// The script `decls` + `body` answered with `(check-sat)` and `(get-model)`.
fn solve(decls: &str, body: &str) -> Vec<String> {
    run(&format!(
        "(set-logic ALL)\n{decls}{body}(check-sat)\n(get-model)\n"
    ))
}

// ---------------------------------------------------------------------------
// §1. INVERTED (re-fix pass 15, `#P2b-75`) — a strict-order or
//     negated-equality guard over a binder is refuted.
//
//     `(forall ((q Int)) (=> (> q 7) false))` is unsatisfiable (`q = 8`), and
//     0.3.3, `c4b04b7`, `c702310` and pass 14 all answered `sat`.  The root
//     (traced, not read): `mbqi::sat_certify` instantiated a guarded
//     universal over the ground side `t` of each guard `x ⊕ t` alone, and
//     for `x > t`, `x < t` and `¬(x = t)` the guard is FALSE at `t` itself,
//     so every instance was vacuous and the saturation test certified the
//     universal.  The projection that extends a model of the instances to the
//     whole domain needs a point on both sides of every boundary: the
//     relevant set now carries `t ± 1` as terms (`sat_certify::guard_neighbours`
//     — `(+ m 1)` for a declared `m`, which stays one point however the model
//     moves `m`), and a sort a guard compares at gets the unnamed-region
//     representatives an array index sort gets (`sat_certify::unnamed_region`:
//     `r ± 1` over `Int` and bit-vectors, midpoints over `Real`, every ground
//     term of a declared sort, a value no named term denotes over a datatype
//     with a field — `sat_certify::datatype_points`).  `(f 2)`, spelled only
//     inside the body, names its argument too.  Each spelling is backed by a
//     closed instance that is `unsat` on its own; z3 answers `unsat` to every
//     script here.
// ---------------------------------------------------------------------------

/// `(name, script, closed instance that is unsat on its own)`.
const STRICT_GUARD_WRONG_SATS: &[(&str, &str, &str)] = &[
    (
        "w01_forall_guard_false",
        "(assert (forall ((q Int)) (=> (> q 7) false)))\n",
        "(=> (> 8 7) false)",
    ),
    (
        "w302_negated_equality_guard",
        "(assert (forall ((x Int)) (=> (not (= x 3)) false)))\n",
        "(=> (not (= 4 3)) false)",
    ),
    (
        "w303_below_ten_forces_m",
        "(declare-const m Int)\n\
         (assert (forall ((x Int)) (=> (< x 10) (= m 1))))\n(assert (= m 2))\n",
        "(and (=> (< 9 10) (= m 1)) (= m 2))",
    ),
    (
        "w308_above_a_constant",
        "(declare-const m Int)\n(assert (forall ((x Int)) (=> (> x m) false)))\n",
        "(=> (> (+ m 1) m) false)",
    ),
    (
        "w309_two_guarded_universals_over_f",
        "(declare-fun f (Int) Int)\n\
         (assert (forall ((x Int)) (=> (> x 7) (> (f x) (f 7)))))\n\
         (assert (forall ((x Int)) (=> (> x 7) (< (f x) 0))))\n\
         (assert (= (f 7) 0))\n",
        "(and (=> (> 8 7) (> (f 8) (f 7))) (=> (> 8 7) (< (f 8) 0)) (= (f 7) 0))",
    ),
    (
        "w203_f_below_itself",
        "(declare-fun f (Int) Int)\n\
         (assert (forall ((q Int)) (=> (> q 1) (< (f 2) (f q)))))\n",
        "(=> (> 2 1) (< (f 2) (f 2)))",
    ),
];

#[test]
fn strict_order_guards_over_an_int_binder_are_refuted() {
    for &(name, script, instance) in STRICT_GUARD_WRONG_SATS {
        let decls: String = script
            .lines()
            .filter(|line| line.starts_with("(declare"))
            .map(|line| format!("{line}\n"))
            .collect();
        let oracle = run(&format!(
            "(set-logic ALL)\n{decls}(assert {instance})\n(check-sat)\n"
        ));
        assert_eq!(
            verdict(&oracle),
            "unsat",
            "`{name}`: the closed instance `{instance}` of the universal must be \
             unsat on its own — it is what shows the script unsatisfiable"
        );
        let lines = run(&format!("(set-logic ALL)\n{script}(check-sat)\n"));
        assert_eq!(
            verdict(&lines),
            "unsat",
            "`{name}` is unsatisfiable (`#P2b-75`): a strict or negated guard must \
             get an instance on the far side of its boundary\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §1b. INVERTED (re-fix pass 15, `#P2b-76`) — two list values that differ
//      two constructors deep are distinct.
//
//      `(= (cons 1 (cons 2 nil)) (cons 1 (cons 2 (cons 3 nil))))` is false
//      (injectivity twice, then `nil ≠ cons`) and every earlier build answered
//      `sat`; one level shallower pass 13's constructor fold answered `unsat`.
//      Two layers now: `TermManager::mk_eq` decides an equality of two
//      constructor applications at every depth (`oxiz_core`'s `dt_eq`), and a
//      congruence class that merged two applications through an opaque term
//      is checked at the candidate model (`solver::dt_refinement`: a lemma
//      from the closure's own explanation — distinct constructors refute it,
//      the same constructor equates the fields).
// ---------------------------------------------------------------------------

#[test]
fn list_values_differing_two_constructors_deep_are_distinct() {
    let prelude = "(set-logic ALL)\n\
         (declare-datatypes ((IntList 0)) (((nil) (cons (head Int) (tail IntList)))))\n";
    let oracle = run(&format!(
        "{prelude}(assert (= (cons 2 nil) (cons 2 (cons 3 nil))))\n(check-sat)\n"
    ));
    assert_eq!(
        verdict(&oracle),
        "unsat",
        "one level of injectivity plus constructor distinctness is refuted"
    );
    for (name, body) in [
        (
            "d04_two_levels_deep",
            "(assert (= (cons 1 (cons 2 nil)) (cons 1 (cons 2 (cons 3 nil)))))\n",
        ),
        (
            "d05_through_a_constant",
            "(declare-const l3 IntList)\n(assert (= l3 (cons 1 (cons 2 nil))))\n\
             (assert (= l3 (cons 1 (cons 2 (cons 3 nil)))))\n",
        ),
    ] {
        let lines = run(&format!("{prelude}{body}(check-sat)\n"));
        assert_eq!(
            verdict(&lines),
            "unsat",
            "`{name}`: the two lists differ in their third cell (`#P2b-76`)\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §2. INVERTED (re-fix pass 15, decision (67)) — a quantified goal of any
//     theory publishes a certified model or none.
//
//     Re-fix pass 14's net certified a model only where the solver's
//     `has_array_ops` flag was set, and read the solver's assertions (after
//     Skolemisation and the rest).  A quantified goal with no array (a scalar,
//     a UF) and an array goal whose only reads sit under a binder published a
//     falsifying model on `c4b04b7`, HEAD `c702310` and pass 14.  The net's
//     scope is now what was ASSERTED: any quantifier in an asserted term puts
//     the `sat` under the printed-model certificate, uninterpreted functions
//     read from their printed tables (`context::model_fmt::printed_check`).
//     And the strict-guard repair of §1 gives these goals the instance at the
//     far side of their guard, so the search's own model is right for `q01`
//     and `n02`; `u04`'s printed table is certified once its else value is the
//     one the universal needs (`published::repair_else_values`).  Each is
//     published, and replays.
// ---------------------------------------------------------------------------

/// `(name, declarations, assertions, closed claim every model must satisfy)`.
const OUT_OF_SCOPE_FALSIFYING: &[(&str, &str, &str, &str)] = &[
    (
        "q01_scalar_guarded_equality",
        "(declare-const j Int)\n(declare-const m Int)\n",
        "(assert (= j 10))\n(assert (forall ((q Int)) (=> (< q 2) (= m j))))\n",
        "(=> (< 0 2) (= m j))",
    ),
    (
        "u04_uf_guarded_constant",
        "(declare-fun f (Int) Int)\n(declare-const k Int)\n",
        "(assert (forall ((x Int)) (=> (> x 3) (= (f x) 7))))\n\
         (assert (= (f k) 2))\n(assert (> k 0))\n",
        "(and (= (f 4) 7) (= (f 5) 7))",
    ),
    (
        "n02_array_read_only_under_a_binder",
        "(declare-const j Int)\n(declare-const b (Array Int Int))\n",
        "(assert (forall ((xx Int)) (=> (> xx 10) (= (select b j) 2))))\n",
        "(= (select b j) 2)",
    ),
];

#[test]
fn quantified_goals_outside_the_old_net_publish_a_certified_model() {
    for &(name, decls, body, claim) in OUT_OF_SCOPE_FALSIFYING {
        let lines = solve(decls, body);
        assert_eq!(
            verdict(&lines),
            "sat",
            "`{name}` is satisfiable\n{}",
            lines.join("\n")
        );
        assert!(
            !withheld(&lines),
            "`{name}`: its model is certified, and published (decision (67); the \
             certificate reads every function from its printed table)\n{}",
            lines.join("\n")
        );
        let definitions = model_definitions(&lines)
            .unwrap_or_else(|| panic!("`{name}`: a model\n{}", lines.join("\n")));
        assert_eq!(
            replay("", &definitions, claim),
            "sat",
            "`{name}`: the published model satisfies the closed instance `{claim}` \
             of its own universal\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §3. INVERTED (re-fix pass 15, decisions (64), (68)) — quantifier-free
//     models replay on their own script.
//
//     (a) `#P2b-71`'s residue, pre-existing on `c4b04b7` and HEAD: the
//         classes of `(g k)` and `(g j)` are kept apart by the search but
//         printed one value.  Every congruence class with no theory-chosen
//         value — application and `select` members included — now gets a
//         fresh value distinct from every value in use, and every function
//         table is built from those class values; `(get-value ((= (g k) (g
//         j))))` answers `false`.
//     (b) Pass 14's regression `r03` (HEAD printed a correct model): the
//         fresh value of `k` re-keyed `f`'s table and `(f j)`'s class had no
//         value of its own, so its entry at `1` was lost.  It gets one now;
//         and a quantifier-free `sat` whose fresh values would make the
//         printed model falsify an assertion prints the solver's own model
//         instead (`context::model_fmt::printed_check`).
//     (c) Pass 13's regression `d03`: two list values sharing a prefix printed
//         as one.  Bisected in an isolated copy to pass 13's constructor fold
//         in `mk_eq` (switched off, the model is right again): the fold took
//         the atom `(= nil (cons 3 nil))` out of the selector-congruence lemma
//         that had kept `l1 ≠ l3`, nothing else could refute `l1 = l3`
//         (§1b's gap), the search set it true, and the model builder printed
//         one value for the class.  Closed by §1b's two layers.
// ---------------------------------------------------------------------------

/// `(name, declarations, assertions, claim = the assertions as one term)`.
const QUANTIFIER_FREE_FALSIFYING: &[(&str, &str, &str, &str)] = &[
    (
        "m06_nested_uf_classes_collapse",
        "(declare-fun f (Int) Int)\n(declare-fun g (Int) Int)\n\
         (declare-const k Int)\n(declare-const j Int)\n",
        "(assert (= (f (g k)) 1))\n(assert (= (f (g j)) 2))\n",
        "(and (= (f (g k)) 1) (= (f (g j)) 2))",
    ),
    (
        "m09_nested_uf_classes_collapse_real",
        "(declare-fun f (Real) Real)\n(declare-fun g (Real) Real)\n\
         (declare-const k Real)\n(declare-const j Real)\n",
        "(assert (= (f (g k)) 1.0))\n(assert (= (f (g j)) 2.0))\n",
        "(and (= (f (g k)) 1.0) (= (f (g j)) 2.0))",
    ),
    (
        "r03_minted_value_drops_a_table_entry",
        "(declare-fun f (Int) Int)\n(declare-const k Int)\n(declare-const j Int)\n",
        "(assert (distinct (f k) 1))\n(assert (> j 0))\n(assert (<= (f (f j)) 0))\n",
        "(and (distinct (f k) 1) (> j 0) (<= (f (f j)) 0))",
    ),
];

#[test]
fn quantifier_free_models_replay_on_their_script() {
    for &(name, decls, body, claim) in QUANTIFIER_FREE_FALSIFYING {
        let lines = solve(decls, body);
        assert_eq!(
            verdict(&lines),
            "sat",
            "`{name}` is satisfiable\n{}",
            lines.join("\n")
        );
        assert!(!withheld(&lines), "`{name}`\n{}", lines.join("\n"));
        let definitions = model_definitions(&lines)
            .unwrap_or_else(|| panic!("`{name}`: a model\n{}", lines.join("\n")));
        assert_eq!(
            replay("", &definitions, claim),
            "sat",
            "`{name}`: the published model satisfies every assertion\n{}",
            lines.join("\n")
        );
    }
}

#[test]
fn two_list_values_sharing_a_prefix_print_two_values() {
    let prelude = "(declare-datatypes ((IntList 0)) (((nil) (cons (head Int) (tail IntList)))))\n";
    let decls = format!("{prelude}(declare-const l1 IntList)\n(declare-const l3 IntList)\n");
    let body = "(assert (= l1 (cons 1 (cons 2 nil))))\n\
         (assert (= l3 (cons 1 (cons 2 (cons 3 nil)))))\n";
    let lines = solve(&decls, body);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    assert!(!withheld(&lines), "{}", lines.join("\n"));
    let definitions =
        model_definitions(&lines).unwrap_or_else(|| panic!("a model\n{}", lines.join("\n")));
    // Replayed through testers and selectors — and, now that `#P2b-76` is
    // closed, through the list equalities themselves as well.
    let claim = "(and ((_ is cons) l3) (= (head l3) 1) \
         ((_ is cons) (tail l3)) (= (head (tail l3)) 2) \
         ((_ is cons) (tail (tail l3))) (= (head (tail (tail l3))) 3) \
         ((_ is nil) (tail (tail (tail l3)))) \
         ((_ is cons) l1) (= (head l1) 1) ((_ is cons) (tail l1)) (= (head (tail l1)) 2) \
         ((_ is nil) (tail (tail l1))))";
    assert_eq!(
        replay(prelude, &definitions, claim),
        "sat",
        "the published `l1` and `l3` hold both assertions, read cell by cell\n{}",
        lines.join("\n")
    );
    assert_eq!(
        replay(
            prelude,
            &definitions,
            "(and (= l1 (cons 1 (cons 2 nil))) (= l3 (cons 1 (cons 2 (cons 3 nil)))))"
        ),
        "sat",
        "and as the asserted equalities\n{}",
        lines.join("\n")
    );
}

#[test]
fn get_value_of_a_comparison_over_separated_classes_is_false() {
    let lines = run(
        "(set-logic ALL)\n(declare-fun f (Int) Int)\n(declare-fun g (Int) Int)\n\
         (declare-const k Int)\n(declare-const j Int)\n\
         (assert (= (f (g k)) 1))\n(assert (= (f (g j)) 2))\n(check-sat)\n\
         (get-value ((= (g k) (g j))))\n",
    );
    assert_eq!(verdict(&lines), "sat");
    let answer = lines.last().cloned().unwrap_or_default();
    assert_eq!(
        answer, "(((= (g k) (g j)) false))",
        "`(= (g k) (g j))` is `false`, the only value consistent with `(f (g k)) = 1` \
         and `(f (g j)) = 2` (decision (68)); HEAD `c702310` answered `true`, pass \
         14 echoed the term"
    );
}

// ---------------------------------------------------------------------------
// §4. INVERTED (re-fix pass 15, decision (68)) — a correct model the net
//     withheld is published.
//
//     `f` constantly `7` with `d = (store … 0 1)` satisfies every assertion,
//     and HEAD printed that model; pass 14 answered `(error "model not
//     certified: an uninterpreted function the certificate cannot
//     interpret")`, because `(f (f 7))` has an application as its argument.
//     The certificate now reads every uninterpreted function from its printed
//     `define-fun` table, bottom-up and under binders
//     (`context::model_fmt::printed_check`): what it certifies is exactly the
//     printed model.
// ---------------------------------------------------------------------------

#[test]
fn a_correct_model_is_published_by_the_net() {
    let decls = "(declare-fun f (Int) Int)\n(declare-const d (Array Int Int))\n";
    let body = "(assert (= (select d 0) 1))\n(assert (= (f (f 7)) 7))\n\
         (assert (forall ((x Int)) (= (select d x) (select d x))))\n";
    let lines = solve(decls, body);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    assert!(
        !withheld(&lines),
        "the certificate reads `f` from its printed table\n{}",
        lines.join("\n")
    );
    let definitions = model_definitions(&lines).unwrap_or_default();
    assert_eq!(
        replay("", &definitions, "(and (= (select d 0) 1) (= (f (f 7)) 7))"),
        "sat",
        "the published model satisfies the assertions\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §5. REGRESSION GUARDS — the controls the recheck measured holding (the n05
//     replay claim re-derived by re-fix pass 15, see there).
// ---------------------------------------------------------------------------

#[test]
fn guards_whose_boundary_lies_inside_the_guard_are_refuted() {
    for (name, script) in [
        (
            "w301_distinct_guard",
            "(assert (forall ((x Int)) (=> (distinct x 3) false)))\n",
        ),
        (
            "w304_non_strict_guard",
            "(declare-const m Int)\n\
             (assert (forall ((x Int)) (=> (>= x 10) (= m 1))))\n(assert (= m 2))\n",
        ),
        (
            "w307_disjunctive_bound",
            "(declare-const m Int)\n\
             (assert (forall ((x Int)) (or (<= x 7) (= m 1))))\n(assert (= m 2))\n",
        ),
        (
            "v04_ground_point_inside_a_strict_guard",
            "(declare-fun f (Int) Int)\n\
             (assert (forall ((i Int)) (=> (< i 5) (= (f i) 0))))\n(assert (= (f 4) 1))\n",
        ),
    ] {
        let lines = run(&format!("(set-logic ALL)\n{script}(check-sat)\n"));
        assert_eq!(verdict(&lines), "unsat", "`{name}` is unsat");
    }
}

#[test]
fn a_ground_array_read_brings_the_goal_into_the_net_and_the_model_replays() {
    // n05: n02 with one ground read of `b`.  RE-DERIVED by re-fix pass 15:
    // the recheck's claim also required `(select b 11)` to be `2`, which the
    // script does not say — the universal reads `b` at `j`, never at its bound
    // variable, so its instance at `xx = 11` is `(=> (> 11 10) (= (select b
    // j) 2))`.  The `2` at `11` was pass 14's completion choosing the constant
    // default `2` for `b`; with the strict-guard repair (§1) the search's own
    // model sets `b[j] = 2` at `j = 0` and keeps the default `0`, which z3
    // replays as a model.  The claim is now the closed instances the script
    // actually has: the ground read and the universal at two points past its
    // guard.
    let decls = "(declare-const j Int)\n(declare-const b (Array Int Int))\n";
    let body = "(assert (= (select b 5) 5))\n\
         (assert (forall ((xx Int)) (=> (> xx 10) (= (select b j) 2))))\n";
    let lines = solve(decls, body);
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let definitions = model_definitions(&lines)
        .unwrap_or_else(|| panic!("a certified model is published\n{}", lines.join("\n")));
    assert_eq!(
        replay(
            "",
            &definitions,
            "(and (= (select b 5) 5) (=> (> 11 10) (= (select b j) 2)) \
             (=> (> 1000003 10) (= (select b j) 2)))"
        ),
        "sat",
        "the certified model satisfies the ground read and the universal's instances\n{}",
        lines.join("\n")
    );
}
