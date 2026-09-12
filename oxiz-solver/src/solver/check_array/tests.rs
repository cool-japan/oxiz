//! Tests for the array-theory constraint collector.

use super::Solver;
use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use smallvec::smallvec;

/// The three fact sets these tests inspect.
type ArrayFacts = (
    Vec<(TermId, TermId, TermId, TermId)>,
    Vec<(TermId, TermId)>,
    Vec<(TermId, TermId)>,
);

/// Run the array collector over `assertions`, returning the
/// store-select-same-index facts, the positive select assertions and the
/// negated ones.
fn collect(manager: &TermManager, assertions: Vec<TermId>) -> ArrayFacts {
    let mut solver = Solver::new();
    solver.assertions = assertions;
    let mut select_values = FxHashMap::default();
    let mut store_select_same_index = Vec::new();
    let mut array_equalities = Vec::new();
    let mut select_assertions = Vec::new();
    let mut negated_select_assertions = Vec::new();
    let mut read_conflicts = Vec::new();
    solver.collect_array_constraints(
        manager,
        &mut select_values,
        &mut store_select_same_index,
        &mut array_equalities,
        &mut select_assertions,
        &mut negated_select_assertions,
        &mut read_conflicts,
    );
    (
        store_select_same_index,
        select_assertions,
        negated_select_assertions,
    )
}

/// A two-conjunct `And` the builder cannot flatten into its parent — see
/// `check_dt.rs`'s twin for why `mk_and` will not do.
fn nested_and(manager: &mut TermManager, first: TermId, second: TermId) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    manager.intern_term(TermKind::And(smallvec![first, second]), bool_sort)
}

/// An `And` whose two conjuncts are the *same* term — one level of the
/// doubling DAG.  Interned directly so no builder can deduplicate it away.
fn doubling_and(manager: &mut TermManager, child: TermId) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    manager.intern_term(TermKind::And(smallvec![child, child]), bool_sort)
}

/// `(= (select (store a 3 5) 3) 5)` and a Boolean filler, in a scratch
/// manager.
struct Fixture {
    manager: TermManager,
    read_over_write: TermId,
    base: TermId,
    index: TermId,
    stored: TermId,
    result: TermId,
    filler: TermId,
}

impl Fixture {
    fn new() -> Self {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let base = manager.mk_var("a", array_sort);
        let index = manager.mk_int(3);
        let stored = manager.mk_int(5);
        let result = manager.mk_int(5);
        let store = manager.mk_store(base, index, stored);
        let select = manager.mk_select(store, index);
        let read_over_write = manager.mk_eq(select, result);
        let filler = manager.mk_var("p", manager.sorts.bool_sort);
        Self {
            manager,
            read_over_write,
            base,
            index,
            stored,
            result,
            filler,
        }
    }
}

/// A directly asserted read-over-write equality is collected.
#[test]
fn unconditional_read_over_write_is_collected() {
    let f = Fixture::new();
    let (store_select, selects, negated) = collect(&f.manager, vec![f.read_over_write]);
    assert_eq!(store_select, vec![(f.base, f.index, f.stored, f.result)]);
    assert_eq!(selects.len(), 1);
    assert!(negated.is_empty());
}

/// The two de Morgan holes this release closed, restated as tests.
///
/// `(not (and (= (select (store a 3 5) 3) 5) p))` is `(or (not ..) (not p))`
/// — satisfiable with `p = false` — so neither conjunct is entailed, and
/// `(or (= ..) p)` is satisfiable with `p` alone.  Harvesting the equality
/// from either shape refuted a satisfiable formula.
#[test]
fn de_morgan_boundaries_yield_nothing() {
    let mut f = Fixture::new();
    let conjunction = f.manager.mk_and([f.read_over_write, f.filler]);
    let negated_and = f.manager.mk_not(conjunction);
    let (store_select, selects, negated) = collect(&f.manager, vec![negated_and]);
    assert!(store_select.is_empty());
    assert!(selects.is_empty());
    assert!(negated.is_empty());

    let disjunction = f.manager.mk_or([f.read_over_write, f.filler]);
    let (store_select, selects, negated) = collect(&f.manager, vec![disjunction]);
    assert!(store_select.is_empty());
    assert!(selects.is_empty());
    assert!(negated.is_empty());
}

/// A `Not` over the equality alone *is* asserted — negatively — so it feeds
/// the negated-select list rather than the positive one.
#[test]
fn negation_routes_the_fact_to_the_negative_list() {
    let mut f = Fixture::new();
    let negated_eq = f.manager.mk_not(f.read_over_write);
    let (store_select, selects, negated) = collect(&f.manager, vec![negated_eq]);
    assert!(store_select.is_empty());
    assert!(selects.is_empty());
    assert_eq!(negated.len(), 1);
    assert_eq!(negated[0].1, f.result);
}

/// A nested Boolean equality is a polarity boundary: `(= p (= (select ..) 6))`
/// is satisfied with both sides false, so the inner read-over-write is not
/// an asserted fact.  This is what the `collect_facts` flag enforces.
#[test]
fn equality_operands_are_not_asserted() {
    let mut f = Fixture::new();
    let outer = f.manager.mk_eq(f.filler, f.read_over_write);
    let (store_select, selects, negated) = collect(&f.manager, vec![outer]);
    assert!(store_select.is_empty());
    assert!(selects.is_empty());
    assert!(negated.is_empty());
}

/// A deeply nested conjunction is walked on the heap, and the
/// read-over-write equality at the bottom is still collected exactly once.
#[test]
fn deeply_nested_conjunction_walks_on_a_worker_stack() {
    // Stack and depth scale together (1 MiB/200k -> 128 KiB/25k): the
    // ~5 B-per-frame threshold is the pin, so never raise one alone.
    const DEPTH: usize = 25_000;

    let collected = std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let mut f = Fixture::new();
            let mut chain = f.read_over_write;
            for _ in 0..DEPTH {
                chain = nested_and(&mut f.manager, chain, f.filler);
            }
            let (store_select, selects, negated) = collect(&f.manager, vec![chain]);
            (
                store_select,
                selects.len(),
                negated.len(),
                (f.base, f.index, f.stored, f.result),
            )
        })
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not abort");

    assert_eq!(collected.0, vec![collected.3]);
    assert_eq!(collected.1, 1);
    assert_eq!(collected.2, 0);
}

/// A shared sub-DAG is walked once per `(term, polarity)`, not once per path.
///
/// Sixty doubling levels are 2⁶⁰ paths to the equality at the bottom, so
/// finishing at all is the proof that re-visits are pruned — and the fact is
/// collected exactly once, not once per path.
#[test]
fn a_doubling_dag_is_walked_in_linear_time() {
    const LEVELS: usize = 60;
    let mut f = Fixture::new();
    let mut chain = f.read_over_write;
    for _ in 0..LEVELS {
        chain = doubling_and(&mut f.manager, chain);
    }
    let (store_select, selects, negated) = collect(&f.manager, vec![chain]);
    assert_eq!(store_select, vec![(f.base, f.index, f.stored, f.result)]);
    assert_eq!(selects.len(), 1);
    assert!(negated.is_empty());
}

/// The revisit key must include the polarity: the same equality reachable
/// both positively and negatively yields BOTH facts.  A `TermId`-only key
/// would drop whichever polarity is reached second.
#[test]
fn both_polarities_of_a_shared_equality_are_collected() {
    let mut f = Fixture::new();
    let negated_eq = f.manager.mk_not(f.read_over_write);
    let both = nested_and(&mut f.manager, f.read_over_write, negated_eq);
    let (store_select, selects, negated) = collect(&f.manager, vec![both]);
    assert_eq!(store_select, vec![(f.base, f.index, f.stored, f.result)]);
    assert_eq!(selects.len(), 1);
    assert_eq!(negated.len(), 1);
    assert_eq!(negated[0].1, f.result);
}

/// The store=store equality scanner prunes re-visits the same way, and the
/// end-to-end array check both terminates on the doubling DAG and still
/// refutes the store–store extensionality conflict at its bottom.
#[test]
fn store_equality_scan_survives_a_doubling_dag() {
    const LEVELS: usize = 60;
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let array_sort = manager.sorts.array(int_sort, int_sort);
    let base_a = manager.mk_var("a", array_sort);
    let base_b = manager.mk_var("b", array_sort);
    let index = manager.mk_int(0);
    let one = manager.mk_int(1);
    let two = manager.mk_int(2);
    let store_a = manager.mk_store(base_a, index, one);
    let store_b = manager.mk_store(base_b, index, two);
    let eq = manager.mk_eq(store_a, store_b);
    let mut chain = eq;
    for _ in 0..LEVELS {
        chain = doubling_and(&mut manager, chain);
    }

    let mut solver = Solver::new();
    solver.assertions = vec![chain];
    assert_eq!(
        solver.collect_positive_array_term_equalities(&manager),
        vec![(store_a, store_b)]
    );
    // Extensionality forces select(store_a, 0) = select(store_b, 0), i.e.
    // 1 = 2 — the whole check refutes it, through every walker at once.
    assert!(solver.check_array_constraints(&manager));
}

// ── Cross-theory soundness regressions ────────────────────────────────────
//
// `check_cross_theory_conflict` is the only component in the solver that closes
// the EUF↔bit-vector gap for two array reads at congruent indices: congruence
// merges `select(a, x)` with `select(a, #x05)` once `x = #x05`, but the
// bit-vector solver receives the two reads as independent fresh variables and no
// Nelson–Oppen bridge carries the equality across.  Nor does any honesty gate
// cover the shape — `array_atoms_need_theory` needs a positive `store = store`
// equality, `arith_atoms_need_theory` skips non-Int atoms, and the
// model-verification gate — which now does read bit-vector values
// (`EvalVal::Bv`, `solver/model_eval_bv.rs`) — only inspects a finished
// candidate model, while this check runs during the search and is what
// produces the `Unsat` these tests expect.
//
// So every operator that `evaluate_bv_expr` could not fold made this formula
// come back `sat`, and each of the five below did until the evaluator delegated
// to `bv_fold` and gained an arm for it.  These are soundness regressions, not
// precision ones.

/// Run a script and return the last `sat`/`unsat`/`unknown` it printed.
fn script_result(script: &str) -> crate::SolverResult {
    let mut context = crate::Context::new();
    let outputs = context.execute_script(script).unwrap_or_default();
    for token in outputs.iter().rev() {
        match token.trim() {
            "sat" => return crate::SolverResult::Sat,
            "unsat" => return crate::SolverResult::Unsat,
            "unknown" => return crate::SolverResult::Unknown,
            _ => {}
        }
    }
    crate::SolverResult::Unknown
}

/// `x = 5` forces the two reads to be the same array element, so a value
/// expression that folds to anything other than `#x10` is a contradiction.
///
/// Each operator below sits in the value position.  With `bvadd` this was always
/// refuted; with the other four it was reported `sat` — verified by disabling
/// just those arms and re-running, which turns every row after the first back
/// into `Sat`.
#[test]
fn cross_theory_conflict_is_refuted_for_every_value_operator() {
    let head = concat!(
        "(set-logic QF_ABV)\n",
        "(declare-const x (_ BitVec 8))\n",
        "(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))\n",
    );
    let tail = "(assert (= x #x05))\n(assert (= (select a #x05) #x10))\n(check-sat)\n";
    // Each value expression folds, under `x = 5`, to something that is not 16.
    let values = [
        "(bvadd x #x01)",
        "(bvashr x #x01)",
        "(bvsdiv x #x02)",
        "(bvsrem x #x02)",
        "(concat (_ bv0 4) ((_ extract 3 0) x))",
        "(ite (= x #x05) #x06 #x07)",
        "(ite (bvult x #x0a) #x06 #x07)",
        "(bvnot (bvshl x #x01))",
    ];
    for value in values {
        let script = format!("{head}(assert (= (select a x) {value}))\n{tail}");
        assert_eq!(
            script_result(&script),
            crate::SolverResult::Unsat,
            "value expression {value} must be refuted"
        );
    }
}

/// The same shape with a value that genuinely *is* `#x10` stays satisfiable, so
/// the check above is not simply answering `unsat` for everything.
#[test]
fn a_consistent_cross_theory_read_stays_satisfiable() {
    let script = concat!(
        "(set-logic QF_ABV)\n",
        "(declare-const x (_ BitVec 8))\n",
        "(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))\n",
        "(assert (= (select a x) (bvadd x #x0b)))\n",
        "(assert (= x #x05))\n",
        "(assert (= (select a #x05) #x10))\n",
        "(check-sat)\n",
    );
    assert_eq!(script_result(script), crate::SolverResult::Sat);
}

/// The cyclic-alias input that once hung `check` forever, through the public
/// `Context` API this time (the unit-level twin lives in `eval_int`'s tests).
///
/// `(= b (store a 0 (select b 0)))` makes the alias map rewrite
/// `(select b 0)` to itself; the recursive evaluator followed that rewrite
/// without end, so `(check-sat)` never returned — on two well-sorted
/// assertions.  The iterative evaluator carries the set of reads already
/// rewritten along the chain and declines the repeat.
///
/// The worker is joined through a channel with a deadline because the failure
/// mode under guard is precisely "never returns" — a plain `join` would hang
/// the whole suite with it.  The formula is satisfiable (`b = a` is a model),
/// so a sound answer is `sat` or `unknown`; `unsat` would be a soundness
/// regression, and a timeout is the original bug back again.
#[test]
fn a_cyclic_alias_returns_through_the_public_api() {
    let script = concat!(
        "(set-logic QF_ALIA)\n",
        "(declare-const a (Array Int Int))\n",
        "(declare-const b (Array Int Int))\n",
        "(assert (= b (store a 0 (select b 0))))\n",
        "(check-sat)\n",
    );

    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("cyclic-alias-check".into())
        .spawn(move || {
            let _ = sender.send(script_result(script));
        })
        .expect("spawn worker thread");

    let result = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("check-sat on a cyclic array alias must return, not hang");
    assert_ne!(
        result,
        crate::SolverResult::Unsat,
        "the cyclic alias is satisfiable (b = a)"
    );
}
