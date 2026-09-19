//! Round-4 adversarial recheck, **pass 5** — what the round's array work still
//! does not reach, and regression guards for what it does.
//!
//! # The hole this file pins
//!
//! Every array rule in this solver is collected over the **ground** fragment.
//! `solver/array_axioms.rs::ground_children` stops at `Forall`/`Exists`/`Let`/
//! `Match` on purpose, and its doc says why: "a `select` under a `forall` …
//! may mention a bound variable, and a ground lemma over a bound variable is
//! an instance of nothing.  Quantified array reasoning is MBQI's job".
//!
//! MBQI does not do that job.  An instance is ground by construction, but it
//! reaches the SAT solver through `Solver::encode` (`check_core.rs`, the
//! `self.encode(inst.result, manager)` call sites) — not through
//! `Solver::assert`, which is the only place `collect_array_structure` and
//! `eliminate_nonbool_ite` ever run.  So an array term that is first ground
//! *after* instantiation receives neither its read-over-write lemmas, nor its
//! constant-array congruence, nor an `ite` naming: the read is a free value of
//! the element sort and the search satisfies the instance by inventing one.
//!
//! The result is a wrong `sat` from three lines, and the solver contradicts
//! itself without any oracle being asked:
//!
//! ```smt2
//! (declare-const d (_ BitVec 1))
//! (assert (forall ((i (_ BitVec 1)))
//!   (distinct #b0 (select ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0) i))))
//! (check-sat)   ; sat, and (get-value) answers #b1 for a read of the
//!               ; constant-#b0 array
//! ```
//!
//! Measured on a 400-script generated corpus whose scripts pair a quantified
//! assertion with its own ground expansion over the finite index sort (the two
//! are the same formula, because `forall` over `(_ BitVec w)` *is* the
//! conjunction over its `2^w` elements): **29 wrong `sat` and 130 published
//! models that falsify their own script** on the quantified form against
//! **0 wrong `sat`, 0 wrong `unsat`, 0 falsifying models** on the ground form.
//! A second 400-script corpus generated with no array-sorted `ite` at all
//! still gives 20 wrong `sat` and 67 falsifying models, so this is not the
//! `ite` family of `#P2b-41`: it is every array rule at once.
//!
//! crates.io `0.3.3` answers `unknown` on the four-line script above; this
//! tree and the 0.3.4 base `c4b04b7` both answer `sat`.
//!
//! # Pins versus guards
//!
//! * A **pin** is green because the tree is still wrong.  Each carries a
//!   `THE HOLE IS CLOSED` message, so the pass that fixes the defect sees it
//!   turn red and knows to invert it into a guard.
//! * A **guard** asserts the correct behaviour of something the round closed,
//!   and turns red if it comes back.
//!
//! No test here sets a wall-clock `(set-option :timeout)`, and none asserts a
//! verdict behind one (decision (16)).  Every script is decided in
//! milliseconds by the deterministic budgets.

use oxiz_solver::Context;

/// Run a script and return the response lines, folding an `Err` into a single
/// `(error …)` line the way the conformance runner does.
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

/// The whole response, joined, for an assertion message.
fn joined(lines: &[String]) -> String {
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// 1. PINS — an array read that is first ground inside a quantifier instance
//    is a free value of the element sort.
// ---------------------------------------------------------------------------

/// `(store a #b0 #b1)` read at the *other* index under a `forall`.
///
/// `a[#b1] = #b0` is asserted at the top level, and the quantified assertion
/// says every entry of `(store a #b0 #b1)` is `#b1` — at `i = #b1` that is
/// `a[#b1] = #b1`.  The script is unsatisfiable.  The ground twin below, the
/// same formula with the quantifier expanded over the two elements of
/// `(_ BitVec 1)`, is correctly `unsat`, which is what localises the defect to
/// the instantiation path rather than to the array rules themselves.
#[test]
fn a_store_read_inside_a_quantifier_is_a_free_value() {
    let script = "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= (select a #b1) #b0))\n\
         (assert (forall ((i (_ BitVec 1))) (= (select (store a #b0 #b1) i) #b1)))\n\
         (check-sat)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "THE HOLE IS CLOSED: a `store` read inside a quantifier instance is \
         constrained now — invert this pin into a guard that requires `unsat`, \
         and see the module doc for the other three.  Response:\n{}",
        joined(&lines)
    );

    // And the `sat` is a soundness claim, not an incompleteness one: the
    // solver serves a model and a `(get-value)` for it, and the two lines it
    // prints cannot both be true — a `store` at `#b0` read at `#b1` is the
    // base array's entry at `#b1`.
    let with_model = run(&format!(
        "{script_head}(get-model)\n\
         (get-value ((select a #b1) (select (store a #b0 #b1) #b1)))\n",
        script_head = script
    ));
    let text = joined(&with_model);
    assert!(
        text.contains("((select a #b1) #b0)")
            && text.contains("((select (store a #b0 #b1) #b1) #b1)"),
        "the pin's point is that the answer contradicts the answer.  \
         Response:\n{text}"
    );
}

/// The same script with the quantifier expanded by hand over the two elements
/// of the index sort: the *identical* formula, decided correctly.
///
/// This is the control that makes the pin above a statement about the
/// instantiation path and not about `store`.
#[test]
fn the_ground_expansion_of_that_store_read_is_refuted() {
    let script = "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= (select a #b1) #b0))\n\
         (assert (and (= (select (store a #b0 #b1) #b0) #b1) \
                      (= (select (store a #b0 #b1) #b1) #b1)))\n\
         (check-sat)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "unsat",
        "the ground fragment must stay right — this is the control the \
         quantified pins are measured against.  Response:\n{}",
        joined(&lines)
    );
}

/// A `select` of `((as const …) #b0)` under a `forall`, and the solver's own
/// `(get-value)` in the same response says the constant-`#b0` array reads
/// `#b1`.
///
/// Four lines, no oracle, no second solver: the answer contradicts the answer.
#[test]
fn a_constant_array_read_inside_a_quantifier_contradicts_get_value() {
    let sort = "(Array (_ BitVec 1) (_ BitVec 1))";
    let script = format!(
        "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const d (_ BitVec 1))\n\
         (assert (forall ((i (_ BitVec 1))) \
                  (distinct #b0 (select ((as const {sort}) #b0) i))))\n\
         (check-sat)\n\
         (get-value (d (select ((as const {sort}) #b0) d)))\n"
    );
    let lines = run(&script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "THE HOLE IS CLOSED: the constant-array read under a quantifier is \
         constrained now — invert this pin.  Response:\n{}",
        joined(&lines)
    );
    let values = joined(&lines);
    assert!(
        values.contains("((select ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0) d) #b1)"),
        "the pin's point is the self-contradiction: `(get-value)` must still \
         answer `#b1` for a read of the constant-`#b0` array while the check \
         answers `sat`.  Response:\n{values}"
    );
}

/// The ground twin of the constant-array read: correctly `unsat`.
#[test]
fn the_ground_constant_array_read_is_refuted() {
    let script = "(set-logic ALL)\n\
         (declare-const d (_ BitVec 1))\n\
         (assert (distinct #b0 (select ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0) d)))\n\
         (check-sat)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "unsat",
        "the ground constant-array read must stay refuted.  Response:\n{}",
        joined(&lines)
    );
}

/// Decision (14) named an array-sorted `ite` with a fresh variable and two
/// defining implications, and `needs_ite_elimination` no longer excludes
/// `Array`.  That pass runs in `Solver::assert` only, so it never sees a
/// quantifier body: the `ite` read is a free value again as soon as the same
/// shape is written under a `forall`.
///
/// The published model says `p = false` and gives `a1` a `#b0` entry at `#b0`,
/// while the assertion it is a model of demands `#b1` there.
#[test]
fn a_read_through_an_array_ite_inside_a_quantifier_is_a_free_value() {
    let script = "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a0 (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (= (select a0 #b0) #b0))\n\
         (assert (= (select a1 #b0) #b0))\n\
         (assert (forall ((i (_ BitVec 1))) (= (select (ite p a0 a1) i) #b1)))\n\
         (check-sat)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "THE HOLE IS CLOSED: decision (14)'s `ite` naming reaches quantifier \
         instances now — invert this pin.  Response:\n{}",
        joined(&lines)
    );
}

/// An `Int`-indexed array under a `forall`, so the class cannot be read as a
/// bit-vector artefact: the index sort is infinite and the same read is free.
#[test]
fn an_int_indexed_array_read_inside_a_quantifier_is_a_free_value() {
    let script = "(set-logic ALL)\n\
         (declare-const a (Array Int Int))\n\
         (assert (= (select a 1) 0))\n\
         (assert (forall ((i Int)) (= (select (store a 0 5) i) 5)))\n\
         (check-sat)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "THE HOLE IS CLOSED: the `Int`-indexed quantified array read is \
         constrained now — invert this pin.  Response:\n{}",
        joined(&lines)
    );
}

// ---------------------------------------------------------------------------
// 2. GUARDS — decision (14), the ground half, over the shapes the in-tree
//    generator still does not emit.
// ---------------------------------------------------------------------------

/// Fourteen ground spellings of a read through an array-sorted `ite`, every
/// one of them unsatisfiable by construction, every one of them `sat` on
/// crates.io `0.3.3`.
///
/// `let`, `define-fun`, a `:named` annotation, a nested `ite`, a `store` over
/// an `ite`, an `ite` of `store`s, an array-sorted `ite` inside an `n`-ary
/// `distinct`, under a `(as const)` equality, at an uninterpreted index sort,
/// at `Int` index sort, through arrays of arrays, under a uninterpreted
/// function of array sort, and across a `push`/`pop` pair.
#[test]
fn ground_reads_through_an_array_ite_are_refuted_in_every_spelling() {
    let prelude = "(set-logic ALL)\n\
         (declare-const a0 (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (= (select a0 #b0) #b0))\n\
         (assert (= (select a1 #b0) #b0))\n";
    let bodies: [(&str, &str); 11] = [
        (
            "let",
            "(assert (let ((t (ite p a0 a1))) (= (select t #b0) #b1)))",
        ),
        (
            "define-fun",
            "(define-fun t () (Array (_ BitVec 1) (_ BitVec 1)) (ite p a0 a1))\n\
             (assert (= (select t #b0) #b1))",
        ),
        (
            "named",
            "(assert (! (= (select (ite p a0 a1) #b0) #b1) :named lbl))",
        ),
        (
            "nested-ite",
            "(declare-const q Bool)\n\
             (assert (= (select (ite p a0 (ite q a1 a0)) #b0) #b1))",
        ),
        (
            "store-over-ite",
            "(assert (= (select (store (ite p a0 a1) #b1 #b0) #b0) #b1))",
        ),
        (
            "ite-of-stores",
            "(assert (= (select (ite p (store a0 #b1 #b0) (store a1 #b1 #b1)) #b0) #b1))",
        ),
        ("distinct", "(assert (distinct a0 a1 (ite p a0 a1)))"),
        (
            "const-default",
            "(declare-const d (_ BitVec 1))\n\
             (assert (= (ite p a0 a1) ((as const (Array (_ BitVec 1) (_ BitVec 1))) d)))\n\
             (assert (distinct (select a0 #b0) d))\n\
             (assert (distinct (select a1 #b0) d))",
        ),
        (
            "uf-of-array",
            "(declare-fun h ((Array (_ BitVec 1) (_ BitVec 1))) (_ BitVec 1))\n\
             (assert (distinct (h (ite p a0 a1)) (h a0)))\n\
             (assert (distinct (h (ite p a0 a1)) (h a1)))",
        ),
        (
            "store-equality",
            "(assert (= (ite p a0 a1) (store a0 #b0 #b1)))",
        ),
        (
            "push-pop",
            "(push 1)\n(assert (= (select (ite p a0 a1) #b0) #b1))\n(check-sat)\n(pop 1)\n\
             (assert (= (select (ite p a0 a1) #b0) #b1))",
        ),
    ];
    for (tag, body) in bodies {
        let script = format!("{prelude}{body}\n(check-sat)\n");
        let lines = run(&script);
        assert_eq!(
            verdict(&lines),
            "unsat",
            "[{tag}] a ground read through an array-sorted `ite` must be \
             refuted (decision (14)).  Response:\n{}",
            joined(&lines)
        );
    }

    // The three spellings that need their own prelude.
    let standalone: [(&str, &str); 3] = [
        (
            "uninterpreted-index",
            "(set-logic ALL)\n\
             (declare-sort U 0)\n\
             (declare-const b0 (Array U (_ BitVec 1)))\n\
             (declare-const b1 (Array U (_ BitVec 1)))\n\
             (declare-const u U)\n\
             (declare-const p Bool)\n\
             (assert (= (select b0 u) #b0))\n\
             (assert (= (select b1 u) #b0))\n\
             (assert (= (select (ite p b0 b1) u) #b1))\n",
        ),
        (
            "array-of-array",
            "(set-logic ALL)\n\
             (declare-const a0 (Array (_ BitVec 1) (Array (_ BitVec 1) (_ BitVec 1))))\n\
             (declare-const a1 (Array (_ BitVec 1) (Array (_ BitVec 1) (_ BitVec 1))))\n\
             (declare-const p Bool)\n\
             (assert (= (select (select a0 #b0) #b0) #b0))\n\
             (assert (= (select (select a1 #b0) #b0) #b0))\n\
             (assert (= (select (select (ite p a0 a1) #b0) #b0) #b1))\n",
        ),
        (
            "int-index",
            "(set-logic ALL)\n\
             (declare-const c0 (Array Int Int))\n\
             (declare-const c1 (Array Int Int))\n\
             (declare-const p Bool)\n\
             (assert (= (select c0 0) 0))\n\
             (assert (= (select c1 0) 0))\n\
             (assert (= (select (ite p c0 c1) 0) 1))\n",
        ),
    ];
    for (tag, head) in standalone {
        let script = format!("{head}(check-sat)\n");
        let lines = run(&script);
        assert_eq!(
            verdict(&lines),
            "unsat",
            "[{tag}] a ground read through an array-sorted `ite` must be \
             refuted (decision (14)).  Response:\n{}",
            joined(&lines)
        );
    }
}

// ---------------------------------------------------------------------------
// 3. GUARD — decision (19): the `@`/`.` class is refused at every binder.
// ---------------------------------------------------------------------------

/// SMT-LIB 2.6 section 3.1 reserves symbols beginning with `@` or `.` for the
/// solver, and this solver prints `@uc_S_n` witnesses for the elements of an
/// uninterpreted sort.  Decision (19) refuses the class at the parser so a
/// script cannot declare a name its own model will be read through.
///
/// A *class* refusal is only worth the name if it covers every binding form,
/// so this guard walks all of them — including the ones the re-fix pass's own
/// mutation table never touched (`declare-sort`, `define-sort`, a
/// `define-fun`'s **parameter**, a datatype's sort, constructor and selector
/// names, a `let` binding, a quantifier binder, `define-fun-rec`) and the
/// quoted `|@uc_U_0|` spelling, which the simple-symbol lexer would otherwise
/// let straight through.
#[test]
fn every_binder_refuses_a_reserved_leading_character() {
    let refused: [(&str, &str); 16] = [
        (
            "declare-const",
            "(declare-const @uc_U_0 Int)\n(assert (= @uc_U_0 1))",
        ),
        (
            "declare-fun",
            "(declare-fun @g (Int) Int)\n(assert (= (@g 1) 2))",
        ),
        ("declare-sort", "(declare-sort @S 0)\n(declare-const a @S)"),
        (
            "define-sort",
            "(define-sort @S () Int)\n(declare-const a @S)",
        ),
        (
            "define-fun-name",
            "(define-fun @f () Int 1)\n(assert (= @f 1))",
        ),
        (
            "define-fun-param",
            "(define-fun f ((@x Int)) Int @x)\n(assert (= (f 1) 1))",
        ),
        (
            "define-fun-rec",
            "(define-fun-rec @fr ((n Int)) Int (ite (<= n 0) 0 n))",
        ),
        (
            "datatype-sort",
            "(declare-datatypes ((@D 0)) (((c (s Int)))))\n(declare-const d @D)",
        ),
        (
            "datatype-constructor",
            "(declare-datatypes ((D 0)) (((@c (sel Int)))))\n(declare-const d D)",
        ),
        (
            "datatype-selector",
            "(declare-datatypes ((D 0)) (((c (@sel Int)))))\n(declare-const d D)",
        ),
        (
            "let-binding",
            "(declare-const y Int)\n(assert (let ((@v 1)) (= y @v)))",
        ),
        (
            "quantifier-binder",
            "(declare-fun q (Int) Bool)\n(assert (forall ((@x Int)) (q @x)))",
        ),
        (
            "named-annotation",
            "(declare-const y Bool)\n(assert (! y :named @lbl))",
        ),
        (
            "quoted-at",
            "(declare-const |@uc_U_0| Int)\n(assert (= |@uc_U_0| 1))",
        ),
        (
            "leading-dot",
            "(declare-const .foo Int)\n(assert (= .foo 1))",
        ),
        (
            "quoted-dot",
            "(declare-const |.hidden| Int)\n(assert (= |.hidden| 1))",
        ),
    ];
    for (tag, body) in refused {
        let script = format!("(set-logic ALL)\n{body}\n(check-sat)\n");
        let lines = run(&script);
        assert!(
            lines.iter().any(|line| line.starts_with("(error ")),
            "[{tag}] a symbol beginning with `@` or `.` must be refused \
             (decision (19), SMT-LIB 2.6 section 3.1).  Response:\n{}",
            joined(&lines)
        );
    }

    // Two controls: the reserved characters are reserved only in *leading*
    // position, and a conforming script that uses them elsewhere must still
    // run.  A refusal that swallowed these would be a new defect.
    for (tag, body) in [
        (
            "at-in-the-middle",
            "(declare-const a@b Int)\n(assert (= a@b 1))",
        ),
        (
            "dot-in-the-middle",
            "(declare-const a.b Int)\n(assert (= a.b 1))",
        ),
    ] {
        let script = format!("(set-logic ALL)\n{body}\n(check-sat)\n(get-model)\n");
        let lines = run(&script);
        assert_eq!(
            verdict(&lines),
            "sat",
            "[{tag}] `@` and `.` are reserved in leading position only.  \
             Response:\n{}",
            joined(&lines)
        );
    }
}

// ---------------------------------------------------------------------------
// 4. GUARD — decision (20): a user `:max-conflicts` is a budget for the whole
//    check, not for every refinement round.
// ---------------------------------------------------------------------------

/// The array refinement re-installs the SAT engine's conflict ceiling on every
/// round.  Re-reading the live conflict count there rebased the ceiling on
/// what the previous solve had already spent and silently granted the search
/// more than the user asked for (`R3-8`); the fix installs
/// `min(refinement ceiling, conflicts_at_entry + N)`.
///
/// Measured behaviourally rather than from the source: nine pairwise-distinct
/// arrays over `(Array (_ BitVec 3) (_ BitVec 1))` is a script the refinement
/// loop works at, and the `:conflicts` statistic it reports must never exceed
/// the `:max-conflicts` it was given, at any budget.
#[test]
fn a_user_max_conflicts_bounds_the_whole_check() {
    let mut any_budget_bound = false;
    for budget in [50_u64, 100, 200, 2_000] {
        let mut script = format!("(set-logic QF_AUFBV)\n(set-option :max-conflicts {budget})\n");
        for k in 0..9 {
            script.push_str(&format!(
                "(declare-const a{k} (Array (_ BitVec 3) (_ BitVec 1)))\n"
            ));
        }
        script.push_str("(assert (distinct a0 a1 a2 a3 a4 a5 a6 a7 a8))\n");
        script.push_str("(check-sat)\n(get-info :all-statistics)\n");
        let lines = run(&script);
        let stats = lines
            .iter()
            .find(|line| line.contains(":conflicts"))
            .cloned()
            .unwrap_or_default();
        let conflicts = stats
            .split(":conflicts ")
            .nth(1)
            .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|digits| digits.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("no `:conflicts` in statistics: {stats}"));
        assert!(
            conflicts <= budget,
            "`:max-conflicts {budget}` was exceeded: {conflicts} conflicts \
             (decision (20)).  Statistics: {stats}"
        );
        if verdict(&lines) == "unknown" {
            any_budget_bound = true;
        }
    }
    // Without this the guard could pass vacuously on a tree where the budget
    // never bites at any of the four values above.
    assert!(
        any_budget_bound,
        "no budget in the ladder actually stopped the check, so the bound \
         above was never tested — raise the array count or lower the budgets"
    );
}

// ---------------------------------------------------------------------------
// 5. GUARD — decision (17): the three families of falsifying model are gone,
//    and each script's own `(get-value)` agrees with the model it published.
// ---------------------------------------------------------------------------

/// (A) an array over an **uninterpreted** index sort publishes one entry per
/// witness element rather than collapsing to a single constant array;
/// (B) an array-sorted `ite` has a value to publish because decision (14) gave
/// it a defining variable; and (C) an `(as const)` whose default is a
/// *variable* renders the **evaluated** default, not the sort default.
#[test]
fn the_three_published_model_families_agree_with_their_own_scripts() {
    // (A)
    let lines = run("(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-sort U 0)\n\
         (declare-const a (Array U (_ BitVec 1)))\n\
         (declare-const i U)\n\
         (declare-const j U)\n\
         (assert (distinct i j))\n\
         (assert (= (select a i) #b0))\n\
         (assert (= (select a j) #b1))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((select a i) (select a j)))\n");
    assert_eq!(verdict(&lines), "sat", "{}", joined(&lines));
    let text = joined(&lines);
    assert!(
        text.contains("@uc_U_0") && text.contains("@uc_U_1"),
        "(A) the array printer must name the witness elements of the \
         uninterpreted index sort, or two distinct indices collapse into one \
         constant array.  Response:\n{text}"
    );
    assert!(
        text.contains("((select a i) #b0)") && text.contains("((select a j) #b1)"),
        "(A) the published reads must be the ones the script asserts.  \
         Response:\n{text}"
    );

    // (B)
    let lines = run("(set-logic QF_ABV)\n\
         (set-option :produce-models true)\n\
         (declare-const a0 (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (= (select a0 #b0) #b0))\n\
         (assert (= (select a1 #b0) #b1))\n\
         (assert (= (select (ite p a0 a1) #b0) #b1))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value (p (select (ite p a0 a1) #b0)))\n");
    assert_eq!(verdict(&lines), "sat", "{}", joined(&lines));
    let text = joined(&lines);
    assert!(
        text.contains("(p false)") && text.contains("((select (ite p a0 a1) #b0) #b1)"),
        "(B) the only model of this script has `p = false` and reads `a1`; a \
         response that pairs `p = true` with a `#b1` read would falsify its \
         own script.  Response:\n{text}"
    );

    // (C)
    let lines = run("(set-logic QF_ABV)\n\
         (set-option :produce-models true)\n\
         (declare-const d (_ BitVec 2))\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 2)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 2))) d)))\n\
         (assert (= d #b10))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((select a #b0) (select a #b1)))\n");
    assert_eq!(verdict(&lines), "sat", "{}", joined(&lines));
    let text = joined(&lines);
    assert!(
        text.contains("((as const (Array (_ BitVec 1) (_ BitVec 2))) #b10)"),
        "(C) an `(as const)` class value must render its *evaluated* default \
         (`#b10` here), not the sort default and not the variable.  \
         Response:\n{text}"
    );
    assert!(
        text.contains("((select a #b0) #b10)") && text.contains("((select a #b1) #b10)"),
        "(C) every read of the constant array must be the evaluated default.  \
         Response:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// 6. PIN — the campaign behind the module doc, carried into the tree so the
//    pass that closes the hole can measure the same thing.
// ---------------------------------------------------------------------------

/// A 64-bit xorshift, so the corpus below is a *fixed* corpus: same scripts,
/// same counts, on every machine and every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A generated pair: one quantified script and the same formula with the
/// quantifier expanded over the finite index sort.
struct Pair {
    quantified: String,
    ground: String,
}

fn bv_literal(value: u64, width: u32) -> String {
    let mut out = String::from("#b");
    for bit in (0..width).rev() {
        out.push(if (value >> bit) & 1 == 1 { '1' } else { '0' });
    }
    out
}

fn array_term(rng: &mut Rng, sort: &str, index_width: u32, elem_width: u32, depth: u32) -> String {
    let pick = rng.below(100);
    if depth >= 2 || pick < 40 {
        return format!("a{}", rng.below(2));
    }
    if pick < 60 {
        return format!(
            "(store {} {} {})",
            array_term(rng, sort, index_width, elem_width, depth + 1),
            bv_literal(rng.below(1 << index_width), index_width),
            bv_literal(rng.below(1 << elem_width), elem_width)
        );
    }
    if pick < 75 {
        let default = if rng.below(2) == 0 {
            bv_literal(rng.below(1 << elem_width), elem_width)
        } else {
            "d".to_string()
        };
        return format!("((as const {sort}) {default})");
    }
    format!(
        "(ite {} {} {})",
        if rng.below(2) == 0 { "p" } else { "q" },
        array_term(rng, sort, index_width, elem_width, depth + 1),
        array_term(rng, sort, index_width, elem_width, depth + 1)
    )
}

fn elem_term(
    rng: &mut Rng,
    sort: &str,
    index_width: u32,
    elem_width: u32,
    bound: &str,
    depth: u32,
) -> String {
    let pick = rng.below(100);
    if depth >= 1 || pick < 55 {
        let index = if rng.below(2) == 0 {
            bound.to_string()
        } else {
            bv_literal(rng.below(1 << index_width), index_width)
        };
        return format!(
            "(select {} {index})",
            array_term(rng, sort, index_width, elem_width, 0)
        );
    }
    if pick < 75 {
        return bv_literal(rng.below(1 << elem_width), elem_width);
    }
    format!(
        "(bvxor {} {})",
        elem_term(rng, sort, index_width, elem_width, bound, depth + 1),
        bv_literal(rng.below(1 << elem_width), elem_width)
    )
}

fn atom(rng: &mut Rng, sort: &str, index_width: u32, elem_width: u32, bound: &str) -> String {
    let left = elem_term(rng, sort, index_width, elem_width, bound, 0);
    let right = elem_term(rng, sort, index_width, elem_width, bound, 0);
    match rng.below(4) {
        0 => format!("(distinct {left} {right})"),
        1 => format!("(not (= {left} {right}))"),
        _ => format!("(= {left} {right})"),
    }
}

/// One generated pair.  The index sort is `(_ BitVec index_width)`, a finite
/// domain, so `(forall ((i …)) body)` **is** the conjunction of `body` over
/// its `2 ^ index_width` elements and `(exists …)` is the disjunction: the two
/// scripts are the same formula, written two ways.
fn generate_pair(rng: &mut Rng) -> Pair {
    let index_width = if rng.below(3) == 0 { 2 } else { 1 };
    let elem_width = if rng.below(3) == 0 { 2 } else { 1 };
    let sort = format!("(Array (_ BitVec {index_width}) (_ BitVec {elem_width}))");
    let mut header = String::from("(set-logic ALL)\n");
    for k in 0..2 {
        header.push_str(&format!("(declare-const a{k} {sort})\n"));
    }
    header.push_str("(declare-const p Bool)\n(declare-const q Bool)\n");
    header.push_str(&format!("(declare-const d (_ BitVec {elem_width}))\n"));
    for _ in 0..=rng.below(3) {
        let index = bv_literal(rng.below(1 << index_width), index_width);
        let ground = atom(rng, &sort, index_width, elem_width, &index);
        header.push_str(&format!("(assert {ground})\n"));
    }
    let universal = rng.below(10) < 7;
    let body = atom(rng, &sort, index_width, elem_width, "i!q");
    let quantifier = if universal { "forall" } else { "exists" };
    let quantified = format!(
        "{header}(assert ({quantifier} ((i!q (_ BitVec {index_width}))) {body}))\n(check-sat)\n"
    );
    let mut expansion = String::new();
    for value in 0..(1u64 << index_width) {
        expansion.push(' ');
        expansion.push_str(&body.replace("i!q", &bv_literal(value, index_width)));
    }
    let joiner = if universal { "and" } else { "or" };
    let ground = format!("{header}(assert ({joiner}{expansion}))\n(check-sat)\n");
    Pair { quantified, ground }
}

/// **PIN.**  The quantified form of a formula and its own ground expansion get
/// different answers, and the quantified one is the wrong one.
///
/// This is the campaign of the module doc, shrunk to 300 fixed scripts so it
/// costs milliseconds and carried here so the pass that closes the hole can
/// re-run the same measurement: it must end with *zero* disagreements, and
/// then this pin turns red and becomes that guard.
///
/// The `unsat`-side count is asserted to be zero today and must stay zero: a
/// quantified `unsat` where the ground expansion is `sat` would be a wrong
/// `unsat`, a strictly worse defect than the one pinned here, and nothing in
/// this round has produced one.
#[test]
fn quantified_array_scripts_disagree_with_their_own_ground_expansions() {
    let mut rng = Rng(0x0519_2026_0919_0001);
    let (mut wrong_sat, mut wrong_unsat, mut agree, mut other) = (0u32, 0u32, 0u32, 0u32);
    let mut first = String::new();
    for _ in 0..300 {
        let pair = generate_pair(&mut rng);
        let quantified = verdict(&run(&pair.quantified));
        let ground = verdict(&run(&pair.ground));
        match (quantified.as_str(), ground.as_str()) {
            ("sat", "unsat") => {
                wrong_sat += 1;
                if first.is_empty() {
                    first = pair.quantified.clone();
                }
            }
            ("unsat", "sat") => wrong_unsat += 1,
            (a, b) if a == b => agree += 1,
            _ => other += 1,
        }
    }
    eprintln!(
        "[quantified-vs-ground] wrong_sat {wrong_sat} wrong_unsat {wrong_unsat} \
         agree {agree} other {other}"
    );
    assert_eq!(
        wrong_unsat, 0,
        "a quantified `unsat` against a ground `sat` would be a wrong `unsat`; \
         this class has only ever produced wrong `sat`"
    );
    assert!(
        wrong_sat > 0,
        "THE HOLE IS CLOSED: every quantified script now agrees with its own \
         ground expansion ({agree} agree, {other} otherwise) — invert this pin \
         into a guard that requires `wrong_sat == 0` and invert the four pins \
         above with it"
    );
}
