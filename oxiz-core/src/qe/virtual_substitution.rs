//! Loos-Weispfenning style virtual substitution for linear arithmetic.

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;

/// Formula identifier in the term-based QE pipeline.
pub type Formula = TermId;

/// Variable identifier in the term-based QE pipeline.
pub type VariableId = TermId;

/// Eliminate one existentially-quantified linear arithmetic variable using
/// a small virtual-substitution test set.
pub fn eliminate_quantifier_vs(
    var: VariableId,
    formula: &Formula,
    manager: &mut TermManager,
) -> Formula {
    let mut lower_bounds = Vec::new();
    let mut equalities = Vec::new();
    collect_candidates(*formula, var, manager, &mut lower_bounds, &mut equalities);

    let mut witnesses = Vec::new();
    witnesses.push(negative_infinity(var, manager));
    witnesses.extend(
        lower_bounds
            .into_iter()
            .map(|bound| epsilon_shift(bound, var, manager)),
    );
    witnesses.extend(equalities);

    if witnesses.is_empty() {
        return *formula;
    }

    let mut disjuncts = Vec::with_capacity(witnesses.len());
    for witness in witnesses {
        let mut subst = FxHashMap::default();
        subst.insert(var, witness);
        let substituted = manager.substitute(*formula, &subst);
        disjuncts.push(simplify_formula(substituted, manager));
    }

    simplify_formula(manager.mk_or(disjuncts), manager)
}

/// Collect the virtual-substitution test-set candidates for `var`.
///
/// Traversal is iterative (an explicit stack plus a visited set, so a deep
/// or heavily shared formula can neither overflow nor be re-expanded), and
/// it descends through *every* term kind rather than only `and`/`or`. The
/// previous catch-all silently stopped at negations, implications, `ite`
/// and every other connective, so bounds on `var` hidden below them were
/// dropped from the test set and the elimination lost those cases. Every
/// collected candidate `t` only ever contributes the disjunct `φ[var := t]`,
/// which is implied by `∃var. φ`, so a larger test set is always sound and
/// strictly more complete.
fn collect_candidates(
    formula: TermId,
    var: VariableId,
    manager: &TermManager,
    lower_bounds: &mut Vec<TermId>,
    equalities: &mut Vec<TermId>,
) {
    let mut stack = vec![formula];
    let mut visited = FxHashSet::default();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(term) = manager.get(current) else {
            continue;
        };

        match &term.kind {
            TermKind::Gt(lhs, rhs) | TermKind::Ge(lhs, rhs) => {
                if *lhs == var {
                    lower_bounds.push(*rhs);
                } else if *rhs == var {
                    equalities.push(*lhs);
                }
            }
            TermKind::Lt(lhs, rhs) | TermKind::Le(lhs, rhs) if *rhs == var => {
                lower_bounds.push(*lhs);
            }
            TermKind::Eq(lhs, rhs) => {
                if *lhs == var {
                    equalities.push(*rhs);
                } else if *rhs == var {
                    equalities.push(*lhs);
                }
            }
            _ => {}
        }

        stack.extend(crate::ast::traversal::get_children(&term.kind));
    }
}

fn negative_infinity(var: VariableId, manager: &mut TermManager) -> TermId {
    let sort = manager
        .get(var)
        .map_or(manager.sorts.int_sort, |term| term.sort);
    if sort == manager.sorts.real_sort {
        manager.mk_real(Rational64::new(-1_000_000, 1))
    } else {
        manager.mk_int(-1_000_000)
    }
}

fn epsilon_shift(bound: TermId, var: VariableId, manager: &mut TermManager) -> TermId {
    let sort = manager
        .get(var)
        .map_or(manager.sorts.int_sort, |term| term.sort);
    if sort == manager.sorts.real_sort {
        let epsilon = manager.mk_real(Rational64::new(1, 2));
        manager.mk_add([bound, epsilon])
    } else {
        let one = manager.mk_int(1);
        manager.mk_add([bound, one])
    }
}

/// Work item of the iterative bottom-up simplifier.
enum SimpStep {
    /// Schedule a term's operands (or record it as unchanged).
    Enter(TermId),
    /// Rebuild a term from its already-simplified operands.
    Build(TermId),
}

/// Bottom-up rebuild-and-simplify over the formula structure.
///
/// Uses an explicit stack and a memo table: the recursive form overflowed on
/// deep formulas (its frames additionally held `manager.simplify`'s own
/// state) and re-walked shared subterms. The memo is keyed on `TermId`
/// alone, which is sound because no binder is entered and the rebuild is a
/// deterministic function of the term.
fn simplify_formula(term: TermId, manager: &mut TermManager) -> TermId {
    let mut memo: FxHashMap<TermId, TermId> = FxHashMap::default();
    let mut stack = vec![SimpStep::Enter(term)];

    while let Some(step) = stack.pop() {
        match step {
            SimpStep::Enter(id) => {
                if memo.contains_key(&id) {
                    continue;
                }
                let Some(node) = manager.get(id).cloned() else {
                    memo.insert(id, id);
                    continue;
                };
                match &node.kind {
                    TermKind::And(args) | TermKind::Or(args) | TermKind::Add(args) => {
                        stack.push(SimpStep::Build(id));
                        for &arg in args.iter() {
                            stack.push(SimpStep::Enter(arg));
                        }
                    }
                    TermKind::Not(arg) => {
                        stack.push(SimpStep::Build(id));
                        stack.push(SimpStep::Enter(*arg));
                    }
                    TermKind::Eq(lhs, rhs)
                    | TermKind::Lt(lhs, rhs)
                    | TermKind::Le(lhs, rhs)
                    | TermKind::Gt(lhs, rhs)
                    | TermKind::Ge(lhs, rhs) => {
                        stack.push(SimpStep::Build(id));
                        stack.push(SimpStep::Enter(*lhs));
                        stack.push(SimpStep::Enter(*rhs));
                    }
                    // No rewrite is defined for any other kind: the term is
                    // returned unchanged, which is a correct answer rather
                    // than a default.
                    _ => {
                        memo.insert(id, id);
                    }
                }
            }
            SimpStep::Build(id) => {
                let Some(node) = manager.get(id).cloned() else {
                    memo.insert(id, id);
                    continue;
                };
                let mapped = |child: TermId, memo: &FxHashMap<TermId, TermId>| -> TermId {
                    memo.get(&child).copied().unwrap_or(child)
                };
                let rebuilt = match &node.kind {
                    TermKind::And(args) => {
                        let simplified: Vec<_> = args.iter().map(|&a| mapped(a, &memo)).collect();
                        manager.mk_and(simplified)
                    }
                    TermKind::Or(args) => {
                        let simplified: Vec<_> = args.iter().map(|&a| mapped(a, &memo)).collect();
                        manager.mk_or(simplified)
                    }
                    TermKind::Add(args) => {
                        let simplified: Vec<_> = args.iter().map(|&a| mapped(a, &memo)).collect();
                        manager.mk_add(simplified)
                    }
                    TermKind::Not(arg) => {
                        let arg = mapped(*arg, &memo);
                        manager.mk_not(arg)
                    }
                    TermKind::Eq(lhs, rhs) => {
                        let (lhs, rhs) = (mapped(*lhs, &memo), mapped(*rhs, &memo));
                        manager.mk_eq(lhs, rhs)
                    }
                    TermKind::Lt(lhs, rhs) => {
                        let (lhs, rhs) = (mapped(*lhs, &memo), mapped(*rhs, &memo));
                        manager.mk_lt(lhs, rhs)
                    }
                    TermKind::Le(lhs, rhs) => {
                        let (lhs, rhs) = (mapped(*lhs, &memo), mapped(*rhs, &memo));
                        manager.mk_le(lhs, rhs)
                    }
                    TermKind::Gt(lhs, rhs) => {
                        let (lhs, rhs) = (mapped(*lhs, &memo), mapped(*rhs, &memo));
                        manager.mk_gt(lhs, rhs)
                    }
                    TermKind::Ge(lhs, rhs) => {
                        let (lhs, rhs) = (mapped(*lhs, &memo), mapped(*rhs, &memo));
                        manager.mk_ge(lhs, rhs)
                    }
                    _ => id,
                };
                let simplified = manager.simplify(rebuilt);
                memo.insert(id, simplified);
            }
        }
    }

    memo.get(&term).copied().unwrap_or(term)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_candidates_descends_below_negation() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let bound = manager.mk_int(3);
        let atom = manager.mk_gt(x, bound);
        let formula = manager.mk_not(atom);

        let mut lower_bounds = Vec::new();
        let mut equalities = Vec::new();
        collect_candidates(formula, x, &manager, &mut lower_bounds, &mut equalities);

        // The bound below the negation must not be dropped.
        assert_eq!(lower_bounds, vec![bound]);
        assert!(equalities.is_empty());
    }

    #[test]
    fn test_collect_candidates_conjunction_pin() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let lo = manager.mk_int(1);
        let eq_rhs = manager.mk_int(7);
        let gt = manager.mk_gt(x, lo);
        let eq = manager.mk_eq(x, eq_rhs);
        let formula = manager.mk_and([gt, eq]);

        let mut lower_bounds = Vec::new();
        let mut equalities = Vec::new();
        collect_candidates(formula, x, &manager, &mut lower_bounds, &mut equalities);

        assert_eq!(lower_bounds, vec![lo]);
        assert_eq!(equalities, vec![eq_rhs]);
    }

    #[test]
    fn test_collect_candidates_shared_dag_is_fast() {
        // 55 levels of a two-strand DAG: 2^55 nodes if shared sub-terms were
        // re-expanded.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let one = manager.mk_int(1);
        let mut a = manager.mk_gt(x, zero);
        let mut b = manager.mk_gt(x, one);
        for _ in 0..55 {
            let next_a = manager.mk_implies(a, b);
            let next_b = manager.mk_implies(b, a);
            a = next_a;
            b = next_b;
        }

        let mut lower_bounds = Vec::new();
        let mut equalities = Vec::new();
        collect_candidates(a, x, &manager, &mut lower_bounds, &mut equalities);
        lower_bounds.sort_by_key(|t| t.0);
        let mut expected = vec![zero, one];
        expected.sort_by_key(|t| t.0);
        assert_eq!(lower_bounds, expected);
    }

    #[test]
    fn test_collect_candidates_deep_nesting_does_not_overflow() {
        // 128 KiB / 6_250 levels: the house ratio for a genuinely nested
        // `Not` chain (see oxiz-spacer/src/walk.rs's `DEEP_STACK`/`DEEP_DEPTH`).
        // `mk_not` folds double negation (`not(not(p)) == p`), so a loop of
        // `mk_not` calls oscillates between depth 0 and 1 and never actually
        // nests -- intern the `Not` nodes directly so the chain really is
        // this deep.
        const DEEP_STACK: usize = 1 << 17;
        const DEEP_DEPTH: usize = 6_250;

        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let bool_sort = manager.sorts.bool_sort;
                let x = manager.mk_var("x", int_sort);
                let zero = manager.mk_int(0);
                let mut formula = manager.mk_gt(x, zero);
                for _ in 0..DEEP_DEPTH {
                    formula = manager.intern_term(TermKind::Not(formula), bool_sort);
                }

                let mut lower_bounds = Vec::new();
                let mut equalities = Vec::new();
                collect_candidates(formula, x, &manager, &mut lower_bounds, &mut equalities);
                lower_bounds.len()
            })
            .expect("thread spawn should succeed");

        assert_eq!(handle.join().expect("deep walk must not overflow"), 1);
    }

    #[test]
    fn test_simplify_formula_deep_nesting_does_not_overflow() {
        // Rescaled to the house 128 KiB / 6_250 ratio (see
        // oxiz-spacer/src/walk.rs's `DEEP_STACK`/`DEEP_DEPTH`); this used to
        // be a 1 MiB / 20_000 STACK-1MIB exception, but `mk_not` folds
        // double negation (`not(not(p)) == p`), so the old loop of
        // `mk_not` calls oscillated between depth 0 and 1 and never
        // actually nested -- intern the `Not` nodes directly so the chain
        // really is this deep.
        const DEEP_STACK: usize = 1 << 17;
        const DEEP_DEPTH: usize = 6_250;

        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let bool_sort = manager.sorts.bool_sort;
                let x = manager.mk_var("x", int_sort);
                let zero = manager.mk_int(0);
                let mut formula = manager.mk_gt(x, zero);
                for _ in 0..DEEP_DEPTH {
                    formula = manager.intern_term(TermKind::Not(formula), bool_sort);
                }
                simplify_formula(formula, &mut manager)
            })
            .expect("thread spawn should succeed");

        // Returning at all is the proof that the walk is iterative.
        let _ = handle.join().expect("deep simplify must not overflow");
    }
}
