//! Deep-nesting and shared-DAG regression tests for the iterative EUF
//! interning walks in the parent module.
//!
//! Kept in its own file so `theory_manager.rs` stays under the workspace
//! 2000-line limit.

use super::*;
use oxiz_core::ast::TermManager;

/// Nesting depth that would overflow the native stack under the previous
/// recursive walk; the assertion is that the call **returns**.
///
/// This depth and [`SMALL_STACK`] were scaled down together by a factor of 8
/// (from 60 000 on 1 MiB).  What the test pins is the ~17 bytes of stack
/// available per level — far under any native frame — not the absolute
/// depth, and the smaller pair costs a fraction of the memory the interner
/// has to keep live.  Never raise one without the other.
const DEEP: usize = 7_500;

/// Worker stack for the deep-nesting test; see [`DEEP`].
const SMALL_STACK: usize = 1 << 17;

/// Scratch state a bare [`TheoryManager`] borrows.
#[derive(Default)]
struct Scratch {
    euf: EufSolver,
    arith: ArithSolver,
    bv: BvSolver,
    bv_terms: FxHashSet<TermId>,
    var_to_constraint: FxHashMap<Var, Constraint>,
    var_to_parsed_arith: FxHashMap<Var, ParsedArithConstraint>,
    term_to_var: FxHashMap<TermId, Var>,
    var_to_term: Vec<TermId>,
    derived_reasons: DerivedReasons,
    statistics: Statistics,
    quantifier_uf_funcs: FxHashSet<oxiz_core::interner::Spur>,
}

impl Scratch {
    fn manager<'a>(&'a mut self, tm: &'a TermManager) -> TheoryManager<'a> {
        TheoryManager::new(
            tm,
            &mut self.euf,
            &mut self.arith,
            &mut self.bv,
            &self.bv_terms,
            &self.var_to_constraint,
            &self.var_to_parsed_arith,
            &self.term_to_var,
            &self.var_to_term,
            &mut self.derived_reasons,
            TheoryMode::Lazy,
            &mut self.statistics,
            0,
            0,
            false,
            false,
            &self.quantifier_uf_funcs,
            None,
        )
    }
}

/// `f(f(f(... x)))`, `depth` applications deep.
fn deep_apply_chain(tm: &mut TermManager, depth: usize) -> TermId {
    let int_sort = tm.sorts.int_sort;
    let mut current = tm.mk_var("x", int_sort);
    for _ in 0..depth {
        current = tm.mk_apply("f", vec![current], int_sort);
    }
    current
}

#[test]
fn s8_intern_term_for_congruence_deep_apply_returns() {
    let handle = std::thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(|| {
            let mut tm = TermManager::new();
            let deep = deep_apply_chain(&mut tm, DEEP);
            let mut scratch = Scratch::default();
            let mut tmgr = scratch.manager(&tm);
            tmgr.intern_term_for_congruence(deep, &tm);
        })
        .expect("spawn deep-nesting worker");
    assert!(
        handle.join().is_ok(),
        "interning a {DEEP}-deep application must return, not overflow"
    );
}

/// A doubling DAG (`f(t, t)` 55 times): `euf.term_to_node` memoises, so
/// this completes in linear time instead of expanding exponentially.
#[test]
fn s8_intern_term_for_congruence_shared_dag_completes() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let mut current = tm.mk_var("x", int_sort);
    for _ in 0..55 {
        current = tm.mk_apply("g", vec![current, current], int_sort);
    }
    let mut scratch = Scratch::default();
    let mut tmgr = scratch.manager(&tm);
    let node = tmgr.intern_term_for_congruence(current, &tm);
    // Re-interning the same term is the memo hit, not a second walk.
    assert_eq!(tmgr.intern_term_for_congruence(current, &tm), node);
}

/// Semantic pins: identical terms share a node, distinct arguments do not,
/// and `select` operands are interned as a binary application so
/// congruence can fire on them.
#[test]
fn s8_intern_term_for_congruence_node_identity_preserved() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let a = tm.mk_var("a", int_sort);
    let b = tm.mk_var("b", int_sort);
    let fa = tm.mk_apply("f", vec![a], int_sort);
    let fb = tm.mk_apply("f", vec![b], int_sort);
    let ffa = tm.mk_apply("f", vec![fa], int_sort);

    let mut scratch = Scratch::default();
    let mut tmgr = scratch.manager(&tm);
    let n_fa = tmgr.intern_term_for_congruence(fa, &tm);
    let n_fb = tmgr.intern_term_for_congruence(fb, &tm);
    let n_ffa = tmgr.intern_term_for_congruence(ffa, &tm);

    assert_ne!(n_fa, n_fb, "f(a) and f(b) are distinct EUF nodes");
    assert_ne!(n_ffa, n_fa, "f(f(a)) is distinct from f(a)");
    assert_eq!(
        tmgr.intern_term_for_congruence(fa, &tm),
        n_fa,
        "re-interning must be idempotent"
    );
    // The post-order of the iterative walk interns every operand before
    // the application that owns it, so the leaves have nodes too.
    assert!(tmgr.euf.term_to_node(a).is_some());
    assert!(tmgr.euf.term_to_node(b).is_some());
}
