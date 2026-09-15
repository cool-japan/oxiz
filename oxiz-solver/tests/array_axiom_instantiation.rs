//! Regression tests for lazy array-axiom instantiation in the CDCL(T) loop.
//!
//! These pin the behaviour of the read-over-write / extensionality /
//! select-congruence lemmas added by `Solver::instantiate_array_axioms`.  Every
//! case here is one the syntactic array pre-checks (`check_array.rs`) cannot
//! decide on their own: without in-loop axiom instantiation the solver would
//! either return a spurious `Sat` (an unsound result) or an over-cautious
//! `Unknown`.

use oxiz_solver::{Context, SolverResult};

/// Run an SMT-LIB script and return the verdict of the final `check-sat`.
fn run_script(script: &str) -> SolverResult {
    let mut ctx = Context::new();
    let outputs = ctx.execute_script(script).unwrap_or_default();
    for tok in outputs.iter().rev() {
        match tok.trim() {
            "sat" => return SolverResult::Sat,
            "unsat" => return SolverResult::Unsat,
            "unknown" => return SolverResult::Unknown,
            _ => {}
        }
    }
    SolverResult::Unknown
}

/// Read-over-write case 2 (different index) drives a conflict: with `i != j`,
/// `select(store(a,i,v),j)` must equal `select(a,j)`, so `5` and `7` clash.
///
/// The syntactic pre-check only handles the *same*-index read-over-write; the
/// disequality-driven case is decided solely by the in-loop RoW-2 axiom.
#[test]
fn row2_different_index_is_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const i Int)
(declare-const j Int)
(declare-const v Int)
(assert (not (= i j)))
(assert (= (select (store a i v) j) 5))
(assert (= (select a j) 7))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// The same RoW-2 shape but consistent (`select(a,j) = 5`) must stay `Sat`,
/// proving the axiom does not over-constrain.
#[test]
fn row2_different_index_consistent_is_sat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const i Int)
(declare-const j Int)
(declare-const v Int)
(assert (not (= i j)))
(assert (= (select (store a i v) j) 5))
(assert (= (select a j) 5))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Two nested stores at three pairwise-distinct indices: reading at `k` skips
/// both writes and must reduce to `select(a,k)`.  Requires two rounds of RoW-2
/// (outer store, then inner store), i.e. genuine saturation over the lemma the
/// first round introduces.
#[test]
fn nested_store_read_skips_both_writes_is_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const i Int)
(declare-const j Int)
(declare-const k Int)
(declare-const v1 Int)
(declare-const v2 Int)
(assert (not (= i k)))
(assert (not (= j k)))
(assert (= (select (store (store a i v1) j v2) k) 5))
(assert (= (select a k) 9))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Nested-store read where the outer write *does* land on the query index:
/// `select(store(store(a,i,v1),j,v2), j) = v2` regardless of `i`.  Asserting a
/// different value for `v2` is unsat via same-index RoW-1 at the outer store.
#[test]
fn nested_store_read_hits_outer_write_is_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const i Int)
(declare-const j Int)
(assert (= (select (store (store a i 1) j 2) j) 3))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Extensionality witness: `a != b` while every *named* read agrees.  A witness
/// index (distinct from `0` and `1`) can differ, so the formula is satisfiable —
/// the extensionality lemma must introduce such a witness rather than force a
/// spurious `Unsat`, and the solver must not report `Unknown`.
#[test]
fn extensionality_witness_is_sat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(assert (not (= a b)))
(assert (= (select a 0) (select b 0)))
(assert (= (select a 1) (select b 1)))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Select congruence over an asserted array equality: `a = b` forces
/// `select(a,0) = select(b,0)`, so pinning them to `5` and `7` is unsat.
#[test]
fn select_congruence_over_equality_is_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(assert (= a b))
(assert (= (select a 0) 5))
(assert (= (select b 0) 7))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Store equality via extensionality on a disequality that is genuinely
/// contradictory: `a != store(a,i,v)` yet `select(a,i) = v` (so the store is a
/// no-op and the arrays are actually equal at every index) — the witness index
/// must collapse onto `i`, making the disequality unsatisfiable.
#[test]
fn store_noop_disequality_is_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const i Int)
(declare-const v Int)
(assert (= (select a i) v))
(assert (not (= a (store a i v))))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// A plain disequality of two unconstrained arrays with no shared reads is
/// trivially satisfiable and must remain `Sat` (extensionality adds a witness
/// but never over-constrains).
#[test]
fn unconstrained_array_disequality_is_sat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(assert (not (= a b)))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Read-over-write with a store to a base array that itself is aliased through
/// an equality: `B = store(a,i,v)` and reading `B` at a different index must go
/// to `a`.  Exercises the alias-guarded RoW instantiation.
#[test]
fn aliased_store_read_over_write_is_unsat() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(declare-const i Int)
(declare-const j Int)
(declare-const v Int)
(assert (= b (store a i v)))
(assert (not (= i j)))
(assert (= (select b j) 5))
(assert (= (select a j) 7))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Issue #22 (reduced): an equality between two *independent* read-over-write
/// reads whose indices are arithmetic compounds.
///
/// Both base arrays are unconstrained, so a model exists trivially (choose the
/// arrays so the two reads agree).  Nothing here entails a conflict, yet the
/// index terms (`+`, `div`) are not syntactically comparable — the case that
/// stresses the RoW instantiation's "index relationship unknown" path.  The
/// axiom layer must leave such a read unresolved (`eval_read` → `None`) rather
/// than committing to a value and manufacturing a disequality conflict.
#[test]
fn test_issue_22_read_over_write_arith_index() {
    let script = r#"
(set-logic QF_AUFLIA)
(declare-const a0 (Array Int Int))
(declare-const a1 (Array Int Int))
(declare-const i0 Int)
(declare-const i1 Int)
(declare-const i2 Int)
(assert (= (select (store a1 1 2) (+ 2 i1))
           (select (store a0 i0 (div i1 8)) (div i2 10))))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Issue #22 (verbatim reproducer): read-over-write with `div`, `mod`, `abs`,
/// `ite` and `+` in the index/value positions, wrapped in `(not (distinct …))`.
///
/// `z3` answers `sat`; this pins that oxiz agrees.  The Euclidean constants
/// involved — `(div 7 7) = 1`, `(mod (- 3) (- 5)) = 2`, `(abs 7) = 7` — are
/// covered independently by `oxiz-core/tests/audit_div_semantics.rs`.
#[test]
fn test_issue_22_full_reproducer() {
    let script = r#"
(set-logic QF_AUFLIA)
(declare-const a0 (Array Int Int))
(declare-const a1 (Array Int Int))
(declare-const i0 Int)
(declare-const i1 Int)
(declare-const i2 Int)
(assert (not (distinct
  (select (store a1 (div 7 7) (mod (- 3) (- 5))) (+ 2 i1))
  (select (store a0 (ite (<= (mod (abs 7) 2) (- 3)) (- 9) i0) (div i1 8)) (div i2 10)))))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Sat);
}

// ──────────────────────────────────────────────────────────────────
// Polarity boundary: facts must be unconditionally asserted
// ──────────────────────────────────────────────────────────────────

/// `(not (and A B))` is `(or (not A) (not B))`, so neither conjunct is
/// entailed.  The array pre-check's collector tracked polarity but passed it
/// straight through its `And` arm, so a single `(not (and …))` handed every
/// conjunct to the definite-conflict maps at negative polarity.
///
/// Here `(= (select (store a 3 5) 3) 5)` is a *true* read-over-write, so
/// recording it as a negated select assertion made the axiom evaluation agree
/// with the "negated" value and fire a conflict.  The formula is satisfiable
/// with `p = false` (`z3` answers `sat`).
#[test]
fn test_array_bool_eq_polarity_boundary() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const p Bool)
(assert (not (and (= (select (store a 3 5) 3) 5) p)))
(check-sat)
"#;
    assert_ne!(
        run_script(script),
        SolverResult::Unsat,
        "conjuncts of a negated And are disjunctive; they must not be collected \
         as unconditional read-over-write facts"
    );

    // Double negation reaches the store=store extensionality collector, which
    // had the same pass-through: the inner disequality flips back to positive
    // polarity and was recorded as an asserted `(= (store a 0 1) (store b 0 2))`.
    // Satisfiable with `p = false` — the inner disequality is in fact valid, so
    // the assertion reduces to `(not p)`.
    let double_negation = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(declare-const p Bool)
(assert (not (and (not (= (store a 0 1) (store b 0 2))) p)))
(check-sat)
"#;
    assert_ne!(
        run_script(double_negation),
        SolverResult::Unsat,
        "a doubly-negated equality under a negated And is not asserted; the \
         store extensionality check must not fire on it"
    );
}

/// Control: the same store=store equality asserted *unconditionally* really is
/// unsatisfiable, so the fix above must not have disabled the check.
#[test]
fn test_array_store_extensionality_still_fires_when_asserted() {
    let script = r#"
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(assert (= (store a 0 1) (store b 0 2)))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

// ---------------------------------------------------------------------------
// `#P2b-33`: a read that occurs ONLY as an argument of an uninterpreted
// application.
//
// `Solver::has_array_ops` is the guard on the whole refinement loop, and it was
// raised by `track_theory_vars`, a walk that deliberately does not descend into
// an application's arguments (nor into `distinct` / `=>` / `xor` operands).  A
// formula whose only read sits there therefore left the flag `false`: no
// read-over-write instance was ever built, the read stayed an unconstrained
// leaf, and `(distinct (f (select (store arr i v) i)) (f v))` — unsatisfiable in
// QF_AUF, QF_AUFBV and QF_AUFLIA alike — answered `sat`.  It answered `sat` on
// crates.io 0.3.3 and on every tree before this fix.
//
// The guard is now computed by the instantiator's own exhaustive walk
// (`Solver::mark_array_ops`).  The `Int` rows need the second half of the fix as
// well: `purify_numeric_uf_args` rewrites a numeric literal under an application
// into a fresh proxy variable *throughout that assertion*, so the literal inside
// the `store` becomes a proxy too and the instantiator — which walks the
// pre-purification assertions — reasons about a different `store` term.  EUF
// congruence over `store` (`TheoryManager::STORE_FUNC_ID`) is what joins them.
// ---------------------------------------------------------------------------

/// The QF_AUF row: pure EUF over an uninterpreted sort, no circuit and no
/// tableau anywhere — only the array lemma and congruence closure can refute it.
#[test]
fn p2b33_read_over_write_under_an_application_is_refuted_over_an_uninterpreted_sort() {
    let script = r#"
(set-logic QF_AUF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const arr (Array U U))
(declare-const i U)
(declare-const v U)
(assert (distinct (f (select (store arr i v) i)) (f v)))
(check-sat)
"#;
    assert_eq!(
        run_script(script),
        SolverResult::Unsat,
        "the pre-#P2b-33 `sat` is back"
    );
}

/// The RoW-2 sibling: with `i != j` the write is invisible at `j`, so the two
/// applications are congruent.
#[test]
fn p2b33_row2_under_an_application_is_refuted() {
    let script = r#"
(set-logic QF_AUF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const arr (Array U U))
(declare-const i U)
(declare-const j U)
(declare-const v U)
(assert (distinct i j))
(assert (distinct (f (select (store arr i v) j)) (f (select arr j))))
(check-sat)
"#;
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// Every wrapper the recheck tried around the same instance: a binary function,
/// a nested application, `not (= ..)` instead of `distinct`, and the value
/// reached through a third constant.
#[test]
fn p2b33_the_application_wrappers_are_all_refuted() {
    let prelude = r#"
(set-logic QF_AUF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U U) U)
(declare-const arr (Array U U))
(declare-const i U)
(declare-const v U)
(declare-const w U)
"#;
    for (name, body) in [
        (
            "binary function",
            "(assert (distinct (g (select (store arr i v) i) w) (g v w)))",
        ),
        (
            "nested application",
            "(assert (distinct (f (f (select (store arr i v) i))) (f (f v))))",
        ),
        (
            "not-equals",
            "(assert (not (= (f (select (store arr i v) i)) (f v))))",
        ),
        (
            "through a third constant",
            "(assert (= (f (select (store arr i v) i)) w))\n(assert (distinct (f v) w))",
        ),
    ] {
        let script = format!("{prelude}{body}\n(check-sat)\n");
        assert_eq!(
            run_script(&script),
            SolverResult::Unsat,
            "{name}: the read-over-write must reach the application"
        );
    }
}

/// The bit-vector rows, with the stored value a literal, a variable, and a
/// variable pinned to a literal by two unsigned bounds.
#[test]
fn p2b33_read_over_write_under_an_application_is_refuted_over_bit_vectors() {
    let prelude = r#"
(set-logic QF_AUFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(declare-const v (_ BitVec 8))
"#;
    for (name, body) in [
        (
            "literal value",
            "(assert (distinct (f (select (store arr i #x05) i)) (f #x05)))",
        ),
        (
            "variable value",
            "(assert (distinct (f (select (store arr i v) i)) (f v)))",
        ),
        (
            "value bounded to a literal",
            "(assert (bvule v #x05))\n(assert (bvuge v #x05))\n\
             (assert (distinct (f (select (store arr i v) i)) (f #x05)))",
        ),
        (
            "read under an operator inside the application",
            "(assert (distinct (f (bvadd (select (store arr i #x05) i) #x01)) (f #x06)))",
        ),
    ] {
        let script = format!("{prelude}{body}\n(check-sat)\n");
        assert_eq!(
            run_script(&script),
            SolverResult::Unsat,
            "{name}: the read-over-write must reach the application"
        );
    }
}

/// The `Int` rows, which additionally need EUF congruence over `store`: numeric
/// purification replaces the literal `5` under `g` with a proxy variable
/// everywhere in the assertion, including inside the `store`.
#[test]
fn p2b33_read_over_write_under_an_application_is_refuted_over_integers() {
    let prelude = r#"
(set-logic QF_AUFLIA)
(declare-fun g (Int) Int)
(declare-const arr (Array Int Int))
(declare-const i Int)
(declare-const v Int)
"#;
    for (name, body) in [
        (
            "literal value",
            "(assert (distinct (g (select (store arr i 5) i)) (g 5)))",
        ),
        (
            "value through a variable",
            "(assert (= v 5))\n(assert (distinct (g (select (store arr i 5) i)) (g v)))",
        ),
        (
            "instance also asserted by hand",
            "(assert (= (select (store arr i 5) i) 5))\n\
             (assert (distinct (g (select (store arr i 5) i)) (g 5)))",
        ),
    ] {
        let script = format!("{prelude}{body}\n(check-sat)\n");
        assert_eq!(
            run_script(&script),
            SolverResult::Unsat,
            "{name}: the read-over-write must reach the application"
        );
    }
}

/// Two writes that agree argument-wise are the same array: EUF congruence over
/// `store` (`#P2b-33`), which is what carries the `Int` rows above.
#[test]
fn p2b33_congruent_stores_are_one_array() {
    let script = r#"
(set-logic QF_AUFLIA)
(declare-const arr (Array Int Int))
(declare-const i Int)
(declare-const v Int)
(declare-const w Int)
(assert (= v w))
(assert (distinct (store arr i v) (store arr i w)))
(check-sat)
"#;
    assert_eq!(
        run_script(script),
        SolverResult::Unsat,
        "store is a function: equal arguments, equal array"
    );
}

/// Controls that must stay satisfiable, so the fix decides rather than
/// over-constrains: the same shapes with the write at a *different* index, and
/// a write whose value genuinely differs.
#[test]
fn p2b33_satisfiable_controls_stay_sat() {
    for (name, script) in [
        (
            "the application sees a different index",
            r#"
(set-logic QF_AUF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const arr (Array U U))
(declare-const i U)
(declare-const j U)
(declare-const v U)
(assert (distinct i j))
(assert (distinct (f (select (store arr i v) j)) (f v)))
(check-sat)
"#,
        ),
        (
            "different stored values",
            r#"
(set-logic QF_AUFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (distinct (f (select (store arr i #x05) i)) (f #x06)))
(check-sat)
"#,
        ),
        (
            "stores that differ in their value",
            r#"
(set-logic QF_AUFLIA)
(declare-const arr (Array Int Int))
(declare-const i Int)
(declare-const v Int)
(declare-const w Int)
(assert (distinct v w))
(assert (distinct (store arr i v) (store arr i w)))
(check-sat)
"#,
        ),
    ] {
        assert_eq!(
            run_script(script),
            SolverResult::Sat,
            "{name}: the fix must not over-constrain"
        );
    }
}

/// The boundary the defect never crossed, kept as a control: the same read as a
/// direct operand of the atom, or with the instance asserted at user level, was
/// decided before the fix and must stay decided.
#[test]
fn p2b33_atom_operand_controls_are_unchanged() {
    for script in [
        r#"
(set-logic QF_AUF)
(declare-sort U 0)
(declare-const arr (Array U U))
(declare-const i U)
(declare-const v U)
(assert (distinct (select (store arr i v) i) v))
(check-sat)
"#,
        r#"
(set-logic QF_AUF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const arr (Array U U))
(declare-const i U)
(declare-const v U)
(assert (distinct (f (select (store arr i v) i)) (f v)))
(assert (= (select (store arr i v) i) v))
(check-sat)
"#,
    ] {
        assert_eq!(run_script(script), SolverResult::Unsat);
    }
}

// ---------------------------------------------------------------------------
// `#P2b-36` — the SMT-LIB array constant `((as const (Array D R)) d)`.
//
// It has no term kind of its own: the parser turns the qualified identifier
// into an ordinary uninterpreted `Apply`, so a read of one was an opaque free
// leaf and `(= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00)
// #x00) #x05)` answered `sat` — on 0.3.3 and on every tree before this one.
// `build_const_array_reads` supplies the missing axiom: an array constant
// reads back its default at every index.
// ---------------------------------------------------------------------------

/// The bare read, with a literal index and with a variable one.
#[test]
fn p2b36_a_read_of_an_array_constant_is_its_default() {
    for script in [
        r#"
(set-logic QF_ABV)
(assert (= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00) #x00) #x05))
(check-sat)
"#,
        r#"
(set-logic QF_ABV)
(declare-const i (_ BitVec 8))
(assert (= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00) i) #x05))
(check-sat)
"#,
        r#"
(set-logic QF_ALIA)
(declare-const i Int)
(assert (= (select ((as const (Array Int Int)) 0) i) 5))
(check-sat)
"#,
    ] {
        assert_eq!(
            run_script(script),
            SolverResult::Unsat,
            "an array constant reads back its default:\n{script}"
        );
    }
}

/// The read under an operator and under an uninterpreted function — the two
/// positions `#P2b-32` and `#P2b-33` had to reach separately, here decided by
/// the same axiom because the instantiator's walk already collects them.
#[test]
fn p2b36_a_read_of_an_array_constant_is_decided_under_a_wrapper() {
    for script in [
        r#"
(set-logic QF_ABV)
(declare-const i (_ BitVec 8))
(assert (= (bvadd (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x02) i) #x01) #x06))
(check-sat)
"#,
        r#"
(set-logic QF_AUFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(assert (distinct (f (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x03) #x01)) (f #x03)))
(check-sat)
"#,
    ] {
        assert_eq!(
            run_script(script),
            SolverResult::Unsat,
            "the array-constant read is decided under a wrapper too:\n{script}"
        );
    }
}

/// Composition with the two families around it: a read over a `store` chain
/// whose base is an array constant is reduced by RoW-2 to a read *of* the
/// constant, and `arr = ((as const …) d)` is carried across by select
/// congruence.  Neither needs an axiom of its own, but both need more than one
/// refinement round, so they pin the saturation as much as the axiom.
#[test]
fn p2b36_the_array_constant_axiom_composes_with_row_and_congruence() {
    for script in [
        // RoW-2 down to the constant base.
        r#"
(set-logic QF_ABV)
(declare-const i (_ BitVec 8))
(declare-const j (_ BitVec 8))
(assert (distinct i j))
(assert (= (select (store ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00) i #x07) j) #x05))
(check-sat)
"#,
        // Select congruence across an asserted array equality.
        r#"
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (= arr ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00)))
(assert (= (select arr i) #x05))
(check-sat)
"#,
    ] {
        assert_eq!(
            run_script(script),
            SolverResult::Unsat,
            "the array-constant axiom must compose:\n{script}"
        );
    }
}

/// Controls that must stay satisfiable: a read that agrees with the default,
/// and a write that overrides it at the index being read.
#[test]
fn p2b36_satisfiable_array_constant_controls_stay_sat() {
    for script in [
        r#"
(set-logic QF_ABV)
(assert (= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00) #x00) #x00))
(check-sat)
"#,
        r#"
(set-logic QF_ABV)
(declare-const i (_ BitVec 8))
(assert (= (select (store ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00) i #x05) i) #x05))
(check-sat)
"#,
    ] {
        assert_eq!(
            run_script(script),
            SolverResult::Sat,
            "the axiom must decide, not over-constrain:\n{script}"
        );
    }
}

/// The ambiguity the recognition is conservative about: `|(as const)|` is a
/// legal quoted SMT-LIB symbol whose interned name is exactly the one the
/// parser gives an array constant, so a script that *declares* it switches the
/// axiom off rather than read a user's function as an array constant.
///
/// Without that guard this formula — satisfiable, because `f` is uninterpreted
/// and its result is an unconstrained array — answers `unsat`.
#[test]
fn p2b36_a_declared_as_const_symbol_is_not_an_array_constant() {
    let script = r#"
(set-logic QF_AUFBV)
(declare-fun |(as const)| ((_ BitVec 8)) (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (select (|(as const)| #x00) #x00) #x05))
(check-sat)
"#;
    assert_eq!(
        run_script(script),
        SolverResult::Sat,
        "a declared `|(as const)|` is an uninterpreted function, not an array constant"
    );
}
