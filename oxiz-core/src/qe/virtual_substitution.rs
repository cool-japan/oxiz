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
    witnesses.extend(lower_bounds.into_iter().map(|bound| epsilon_shift(bound, var, manager)));
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

fn collect_candidates(
    formula: TermId,
    var: VariableId,
    manager: &TermManager,
    lower_bounds: &mut Vec<TermId>,
    equalities: &mut Vec<TermId>,
) {
    let Some(term) = manager.get(formula) else {
        return;
    };

    match &term.kind {
        TermKind::And(args) | TermKind::Or(args) => {
            for &arg in args {
                collect_candidates(arg, var, manager, lower_bounds, equalities);
            }
        }
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
}

fn negative_infinity(var: VariableId, manager: &mut TermManager) -> TermId {
    let sort = manager.get(var).map_or(manager.sorts.int_sort, |term| term.sort);
    if sort == manager.sorts.real_sort {
        manager.mk_real(Rational64::new(-1_000_000, 1))
    } else {
        manager.mk_int(-1_000_000)
    }
}

fn epsilon_shift(bound: TermId, var: VariableId, manager: &mut TermManager) -> TermId {
    let sort = manager.get(var).map_or(manager.sorts.int_sort, |term| term.sort);
    if sort == manager.sorts.real_sort {
        let epsilon = manager.mk_real(Rational64::new(1, 2));
        manager.mk_add([bound, epsilon])
    } else {
        let one = manager.mk_int(1);
        manager.mk_add([bound, one])
    }
}

fn simplify_formula(term: TermId, manager: &mut TermManager) -> TermId {
    let Some(node) = manager.get(term).cloned() else {
        return term;
    };

    match node.kind {
        TermKind::And(args) => {
            let simplified: Vec<_> = args
                .into_iter()
                .map(|arg| simplify_formula(arg, manager))
                .collect();
            manager.mk_and(simplified)
        }
        TermKind::Or(args) => {
            let simplified: Vec<_> = args
                .into_iter()
                .map(|arg| simplify_formula(arg, manager))
                .collect();
            manager.mk_or(simplified)
        }
        TermKind::Not(arg) => {
            let arg = simplify_formula(arg, manager);
            manager.mk_not(arg)
        }
        TermKind::Eq(lhs, rhs) => {
            let lhs = simplify_formula(lhs, manager);
            let rhs = simplify_formula(rhs, manager);
            manager.mk_eq(lhs, rhs)
        }
        TermKind::Lt(lhs, rhs) => {
            let lhs = simplify_formula(lhs, manager);
            let rhs = simplify_formula(rhs, manager);
            manager.mk_lt(lhs, rhs)
        }
        TermKind::Le(lhs, rhs) => {
            let lhs = simplify_formula(lhs, manager);
            let rhs = simplify_formula(rhs, manager);
            manager.mk_le(lhs, rhs)
        }
        TermKind::Gt(lhs, rhs) => {
            let lhs = simplify_formula(lhs, manager);
            let rhs = simplify_formula(rhs, manager);
            manager.mk_gt(lhs, rhs)
        }
        TermKind::Ge(lhs, rhs) => {
            let lhs = simplify_formula(lhs, manager);
            let rhs = simplify_formula(rhs, manager);
            manager.mk_ge(lhs, rhs)
        }
        TermKind::Add(args) => {
            let simplified: Vec<_> = args
                .into_iter()
                .map(|arg| simplify_formula(arg, manager))
                .collect();
            manager.mk_add(simplified)
        }
        _ => term,
    }
}
