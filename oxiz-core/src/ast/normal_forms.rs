//! Normal form conversions for boolean formulas
//!
//! This module provides utilities for converting boolean formulas to various
//! normal forms such as CNF (Conjunctive Normal Form), DNF (Disjunctive Normal Form),
//! and NNF (Negation Normal Form). Also includes Skolemization for quantifier elimination.

use super::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Convert a boolean formula to Conjunctive Normal Form (CNF)
///
/// CNF is a conjunction of clauses, where each clause is a disjunction of literals.
/// For example: (a ∨ b) ∧ (¬c ∨ d) ∧ e
///
/// # Algorithm
/// 1. Eliminate implications: (a → b) becomes (¬a ∨ b)
/// 2. Push negations inward (De Morgan's laws)
/// 3. Distribute OR over AND
pub fn to_cnf(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut cache = rustc_hash::FxHashMap::default();
    to_cnf_cached(term_id, manager, &mut cache)
}

fn to_cnf_cached(
    term_id: TermId,
    manager: &mut TermManager,
    cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    if let Some(&result) = cache.get(&term_id) {
        return result;
    }

    let result = match manager.get(term_id).map(|t| t.kind.clone()) {
        None
        | Some(
            TermKind::True
            | TermKind::False
            | TermKind::Var(_)
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. },
        ) => term_id,

        // Eliminate implication: (a → b) = (¬a ∨ b)
        Some(TermKind::Implies(lhs, rhs)) => {
            let not_lhs = manager.mk_not(lhs);
            let not_lhs_cnf = to_cnf_cached(not_lhs, manager, cache);
            let rhs_cnf = to_cnf_cached(rhs, manager, cache);
            distribute_or_over_and(not_lhs_cnf, rhs_cnf, manager, cache)
        }

        // Push negation inward
        Some(TermKind::Not(arg)) => to_cnf_not(arg, manager, cache),

        // Convert children and combine
        Some(TermKind::And(args)) => {
            let cnf_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| to_cnf_cached(a, manager, cache))
                .collect();
            manager.mk_and(cnf_args)
        }

        Some(TermKind::Or(args)) => {
            let cnf_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| to_cnf_cached(a, manager, cache))
                .collect();
            // Distribute OR over AND
            distribute_or_over_and_multi(cnf_args, manager, cache)
        }

        // For other terms, return as-is
        Some(_) => term_id,
    };

    cache.insert(term_id, result);
    result
}

/// Convert ¬term to CNF
fn to_cnf_not(
    term_id: TermId,
    manager: &mut TermManager,
    cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    match manager.get(term_id).map(|t| t.kind.clone()) {
        Some(TermKind::True) => manager.mk_false(),
        Some(TermKind::False) => manager.mk_true(),

        // Double negation: ¬¬a = a
        Some(TermKind::Not(inner)) => to_cnf_cached(inner, manager, cache),

        // De Morgan: ¬(a ∧ b) = ¬a ∨ ¬b
        Some(TermKind::And(args)) => {
            let negated_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| {
                    let not_a = manager.mk_not(a);
                    to_cnf_cached(not_a, manager, cache)
                })
                .collect();
            distribute_or_over_and_multi(negated_args, manager, cache)
        }

        // De Morgan: ¬(a ∨ b) = ¬a ∧ ¬b
        Some(TermKind::Or(args)) => {
            let negated_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| {
                    let not_a = manager.mk_not(a);
                    to_cnf_cached(not_a, manager, cache)
                })
                .collect();
            manager.mk_and(negated_args)
        }

        // ¬(a → b) = ¬(¬a ∨ b) = a ∧ ¬b
        Some(TermKind::Implies(lhs, rhs)) => {
            let lhs_cnf = to_cnf_cached(lhs, manager, cache);
            let not_rhs = manager.mk_not(rhs);
            let not_rhs_cnf = to_cnf_cached(not_rhs, manager, cache);
            manager.mk_and([lhs_cnf, not_rhs_cnf])
        }

        // For other terms, just negate
        _ => manager.mk_not(term_id),
    }
}

/// Distribute OR over AND: (a ∨ b) where a or b might be conjunctions
///
/// If a = (a1 ∧ a2), then (a ∨ b) = (a1 ∨ b) ∧ (a2 ∨ b)
fn distribute_or_over_and(
    lhs: TermId,
    rhs: TermId,
    manager: &mut TermManager,
    _cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    let lhs_kind = manager.get(lhs).map(|t| t.kind.clone());
    let rhs_kind = manager.get(rhs).map(|t| t.kind.clone());

    match (lhs_kind, rhs_kind) {
        // (a1 ∧ a2 ∧ ...) ∨ b = (a1 ∨ b) ∧ (a2 ∨ b) ∧ ...
        (Some(TermKind::And(lhs_args)), _) => {
            let clauses: SmallVec<[TermId; 4]> = lhs_args
                .iter()
                .map(|&a| distribute_or_over_and(a, rhs, manager, _cache))
                .collect();
            manager.mk_and(clauses)
        }

        // a ∨ (b1 ∧ b2 ∧ ...) = (a ∨ b1) ∧ (a ∨ b2) ∧ ...
        (_, Some(TermKind::And(rhs_args))) => {
            let clauses: SmallVec<[TermId; 4]> = rhs_args
                .iter()
                .map(|&b| distribute_or_over_and(lhs, b, manager, _cache))
                .collect();
            manager.mk_and(clauses)
        }

        // a ∨ b (no distribution needed)
        _ => manager.mk_or([lhs, rhs]),
    }
}

/// Distribute OR over AND for multiple disjuncts
fn distribute_or_over_and_multi(
    args: SmallVec<[TermId; 4]>,
    manager: &mut TermManager,
    cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    if args.is_empty() {
        return manager.mk_false();
    }

    if args.len() == 1 {
        return args[0];
    }

    let mut result = args[0];
    for &arg in &args[1..] {
        result = distribute_or_over_and(result, arg, manager, cache);
    }
    result
}

/// Convert a boolean formula to Disjunctive Normal Form (DNF)
///
/// DNF is a disjunction of conjunctions, where each conjunction is a conjunction of literals.
/// For example: (a ∧ b) ∨ (¬c ∧ d) ∨ e
///
/// # Algorithm
/// 1. Eliminate implications
/// 2. Push negations inward
/// 3. Distribute AND over OR
pub fn to_dnf(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut cache = rustc_hash::FxHashMap::default();
    to_dnf_cached(term_id, manager, &mut cache)
}

fn to_dnf_cached(
    term_id: TermId,
    manager: &mut TermManager,
    cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    if let Some(&result) = cache.get(&term_id) {
        return result;
    }

    let result = match manager.get(term_id).map(|t| t.kind.clone()) {
        None
        | Some(
            TermKind::True
            | TermKind::False
            | TermKind::Var(_)
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. },
        ) => term_id,

        // Eliminate implication: (a → b) = (¬a ∨ b)
        Some(TermKind::Implies(lhs, rhs)) => {
            let not_lhs = manager.mk_not(lhs);
            let not_lhs_dnf = to_dnf_cached(not_lhs, manager, cache);
            let rhs_dnf = to_dnf_cached(rhs, manager, cache);
            manager.mk_or([not_lhs_dnf, rhs_dnf])
        }

        // Push negation inward
        Some(TermKind::Not(arg)) => to_dnf_not(arg, manager, cache),

        // Convert children and combine
        Some(TermKind::Or(args)) => {
            let dnf_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| to_dnf_cached(a, manager, cache))
                .collect();
            manager.mk_or(dnf_args)
        }

        Some(TermKind::And(args)) => {
            let dnf_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| to_dnf_cached(a, manager, cache))
                .collect();
            // Distribute AND over OR
            distribute_and_over_or_multi(dnf_args, manager, cache)
        }

        // For other terms, return as-is
        Some(_) => term_id,
    };

    cache.insert(term_id, result);
    result
}

/// Convert ¬term to DNF
fn to_dnf_not(
    term_id: TermId,
    manager: &mut TermManager,
    cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    match manager.get(term_id).map(|t| t.kind.clone()) {
        Some(TermKind::True) => manager.mk_false(),
        Some(TermKind::False) => manager.mk_true(),

        // Double negation: ¬¬a = a
        Some(TermKind::Not(inner)) => to_dnf_cached(inner, manager, cache),

        // De Morgan: ¬(a ∧ b) = ¬a ∨ ¬b
        Some(TermKind::And(args)) => {
            let negated_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| {
                    let not_a = manager.mk_not(a);
                    to_dnf_cached(not_a, manager, cache)
                })
                .collect();
            manager.mk_or(negated_args)
        }

        // De Morgan: ¬(a ∨ b) = ¬a ∧ ¬b
        Some(TermKind::Or(args)) => {
            let negated_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| {
                    let not_a = manager.mk_not(a);
                    to_dnf_cached(not_a, manager, cache)
                })
                .collect();
            distribute_and_over_or_multi(negated_args, manager, cache)
        }

        // ¬(a → b) = a ∧ ¬b
        Some(TermKind::Implies(lhs, rhs)) => {
            let lhs_dnf = to_dnf_cached(lhs, manager, cache);
            let not_rhs = manager.mk_not(rhs);
            let not_rhs_dnf = to_dnf_cached(not_rhs, manager, cache);
            distribute_and_over_or_multi(
                SmallVec::from_vec(vec![lhs_dnf, not_rhs_dnf]),
                manager,
                cache,
            )
        }

        // For other terms, just negate
        _ => manager.mk_not(term_id),
    }
}

/// Distribute AND over OR: (a ∧ b) where a or b might be disjunctions
///
/// If a = (a1 ∨ a2), then (a ∧ b) = (a1 ∧ b) ∨ (a2 ∧ b)
fn distribute_and_over_or(
    lhs: TermId,
    rhs: TermId,
    manager: &mut TermManager,
    _cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    let lhs_kind = manager.get(lhs).map(|t| t.kind.clone());
    let rhs_kind = manager.get(rhs).map(|t| t.kind.clone());

    match (lhs_kind, rhs_kind) {
        // (a1 ∨ a2 ∨ ...) ∧ b = (a1 ∧ b) ∨ (a2 ∧ b) ∨ ...
        (Some(TermKind::Or(lhs_args)), _) => {
            let terms: SmallVec<[TermId; 4]> = lhs_args
                .iter()
                .map(|&a| distribute_and_over_or(a, rhs, manager, _cache))
                .collect();
            manager.mk_or(terms)
        }

        // a ∧ (b1 ∨ b2 ∨ ...) = (a ∧ b1) ∨ (a ∧ b2) ∨ ...
        (_, Some(TermKind::Or(rhs_args))) => {
            let terms: SmallVec<[TermId; 4]> = rhs_args
                .iter()
                .map(|&b| distribute_and_over_or(lhs, b, manager, _cache))
                .collect();
            manager.mk_or(terms)
        }

        // a ∧ b (no distribution needed)
        _ => manager.mk_and([lhs, rhs]),
    }
}

/// Distribute AND over OR for multiple conjuncts
fn distribute_and_over_or_multi(
    args: SmallVec<[TermId; 4]>,
    manager: &mut TermManager,
    cache: &mut rustc_hash::FxHashMap<TermId, TermId>,
) -> TermId {
    if args.is_empty() {
        return manager.mk_true();
    }

    if args.len() == 1 {
        return args[0];
    }

    let mut result = args[0];
    for &arg in &args[1..] {
        result = distribute_and_over_or(result, arg, manager, cache);
    }
    result
}

/// Check if a term is in CNF form
#[must_use]
pub fn is_cnf(term_id: TermId, manager: &TermManager) -> bool {
    match manager.get(term_id).map(|t| &t.kind) {
        None | Some(TermKind::True | TermKind::False | TermKind::Var(_)) => true,

        Some(TermKind::Not(arg)) => {
            // Negation of a literal is OK
            matches!(manager.get(*arg).map(|t| &t.kind), Some(TermKind::Var(_)))
        }

        Some(TermKind::And(args)) => {
            // AND of clauses - each clause must be a disjunction of literals
            args.iter().all(|&a| is_clause(a, manager))
        }

        _ => is_clause(term_id, manager),
    }
}

/// Check if a term is a clause (disjunction of literals)
fn is_clause(term_id: TermId, manager: &TermManager) -> bool {
    match manager.get(term_id).map(|t| &t.kind) {
        None | Some(TermKind::True | TermKind::False | TermKind::Var(_)) => true,

        Some(TermKind::Not(arg)) => {
            // Negation of a literal
            matches!(manager.get(*arg).map(|t| &t.kind), Some(TermKind::Var(_)))
        }

        Some(TermKind::Or(args)) => {
            // OR of literals
            args.iter().all(|&a| is_literal(a, manager))
        }

        _ => false,
    }
}

/// Check if a term is a literal (variable or negated variable)
fn is_literal(term_id: TermId, manager: &TermManager) -> bool {
    match manager.get(term_id).map(|t| &t.kind) {
        Some(TermKind::Var(_)) | Some(TermKind::True) | Some(TermKind::False) => true,

        Some(TermKind::Not(arg)) => {
            matches!(manager.get(*arg).map(|t| &t.kind), Some(TermKind::Var(_)))
        }

        _ => false,
    }
}

/// Check if a term is in DNF form
#[must_use]
pub fn is_dnf(term_id: TermId, manager: &TermManager) -> bool {
    match manager.get(term_id).map(|t| &t.kind) {
        None | Some(TermKind::True | TermKind::False | TermKind::Var(_)) => true,

        Some(TermKind::Not(arg)) => {
            // Negation of a literal is OK
            matches!(manager.get(*arg).map(|t| &t.kind), Some(TermKind::Var(_)))
        }

        Some(TermKind::Or(args)) => {
            // OR of terms - each term must be a conjunction of literals
            args.iter().all(|&a| is_term_conjunction(a, manager))
        }

        _ => is_term_conjunction(term_id, manager),
    }
}

/// Check if a term is a conjunction of literals
fn is_term_conjunction(term_id: TermId, manager: &TermManager) -> bool {
    match manager.get(term_id).map(|t| &t.kind) {
        None | Some(TermKind::True | TermKind::False | TermKind::Var(_)) => true,

        Some(TermKind::Not(arg)) => {
            matches!(manager.get(*arg).map(|t| &t.kind), Some(TermKind::Var(_)))
        }

        Some(TermKind::And(args)) => args.iter().all(|&a| is_literal(a, manager)),

        _ => false,
    }
}

/// Extract clauses from a CNF formula
///
/// Returns a vector of clauses, where each clause is a vector of literals
#[must_use]
pub fn extract_cnf_clauses(term_id: TermId, manager: &TermManager) -> Vec<Vec<TermId>> {
    let mut clauses = Vec::new();

    match manager.get(term_id).map(|t| &t.kind) {
        Some(TermKind::And(args)) => {
            for &arg in args {
                clauses.push(extract_clause_literals(arg, manager));
            }
        }
        _ => {
            clauses.push(extract_clause_literals(term_id, manager));
        }
    }

    clauses
}

/// Extract literals from a clause (disjunction)
fn extract_clause_literals(term_id: TermId, manager: &TermManager) -> Vec<TermId> {
    match manager.get(term_id).map(|t| &t.kind) {
        Some(TermKind::Or(args)) => args.iter().copied().collect(),
        _ => vec![term_id],
    }
}

/// Simplify a boolean formula by eliminating redundant terms
pub fn simplify_boolean(term_id: TermId, manager: &mut TermManager) -> TermId {
    match manager.get(term_id).map(|t| t.kind.clone()) {
        None
        | Some(
            TermKind::True
            | TermKind::False
            | TermKind::Var(_)
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. },
        ) => term_id,

        Some(TermKind::Not(arg)) => {
            let simplified = simplify_boolean(arg, manager);
            manager.mk_not(simplified)
        }

        Some(TermKind::And(args)) => {
            let simplified: SmallVec<[TermId; 4]> =
                args.iter().map(|&a| simplify_boolean(a, manager)).collect();
            // Remove duplicates
            let unique: FxHashSet<TermId> = simplified.iter().copied().collect();
            manager.mk_and(unique)
        }

        Some(TermKind::Or(args)) => {
            let simplified: SmallVec<[TermId; 4]> =
                args.iter().map(|&a| simplify_boolean(a, manager)).collect();
            // Remove duplicates
            let unique: FxHashSet<TermId> = simplified.iter().copied().collect();
            manager.mk_or(unique)
        }

        _ => term_id,
    }
}

/// Convert a boolean formula to Negation Normal Form (NNF)
///
/// NNF is a formula where negations only appear directly on variables.
/// This is achieved by:
/// 1. Eliminating implications: (a → b) becomes (¬a ∨ b)
/// 2. Pushing negations inward using De Morgan's laws
///
/// NNF is simpler than CNF/DNF as it doesn't require distribution.
pub fn to_nnf(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut cache = FxHashMap::default();
    to_nnf_cached(term_id, manager, &mut cache, false)
}

fn to_nnf_cached(
    term_id: TermId,
    manager: &mut TermManager,
    cache: &mut FxHashMap<(TermId, bool), TermId>,
    negate: bool,
) -> TermId {
    if let Some(&result) = cache.get(&(term_id, negate)) {
        return result;
    }

    let result = match manager.get(term_id).map(|t| t.kind.clone()) {
        None => term_id,

        Some(TermKind::True) => {
            if negate {
                manager.mk_false()
            } else {
                manager.mk_true()
            }
        }

        Some(TermKind::False) => {
            if negate {
                manager.mk_true()
            } else {
                manager.mk_false()
            }
        }

        Some(TermKind::Var(_)) => {
            if negate {
                manager.mk_not(term_id)
            } else {
                term_id
            }
        }

        // Double negation
        Some(TermKind::Not(arg)) => to_nnf_cached(arg, manager, cache, !negate),

        // De Morgan's laws when negating
        Some(TermKind::And(args)) => {
            let nnf_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| to_nnf_cached(a, manager, cache, negate))
                .collect();

            if negate {
                // ¬(a ∧ b) = ¬a ∨ ¬b
                manager.mk_or(nnf_args)
            } else {
                manager.mk_and(nnf_args)
            }
        }

        Some(TermKind::Or(args)) => {
            let nnf_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| to_nnf_cached(a, manager, cache, negate))
                .collect();

            if negate {
                // ¬(a ∨ b) = ¬a ∧ ¬b
                manager.mk_and(nnf_args)
            } else {
                manager.mk_or(nnf_args)
            }
        }

        // Eliminate implication: (a → b) = (¬a ∨ b)
        Some(TermKind::Implies(lhs, rhs)) => {
            let lhs_nnf = to_nnf_cached(lhs, manager, cache, !negate);
            let rhs_nnf = to_nnf_cached(rhs, manager, cache, negate);

            if negate {
                // ¬(a → b) = a ∧ ¬b
                manager.mk_and([lhs_nnf, rhs_nnf])
            } else {
                // (a → b) = ¬a ∨ b
                manager.mk_or([lhs_nnf, rhs_nnf])
            }
        }

        // XOR
        Some(TermKind::Xor(lhs, rhs)) => {
            // a ⊕ b = (a ∨ b) ∧ (¬a ∨ ¬b)
            let lhs_nnf = to_nnf_cached(lhs, manager, cache, false);
            let rhs_nnf = to_nnf_cached(rhs, manager, cache, false);
            let not_lhs_nnf = to_nnf_cached(lhs, manager, cache, true);
            let not_rhs_nnf = to_nnf_cached(rhs, manager, cache, true);

            let clause1 = manager.mk_or([lhs_nnf, rhs_nnf]);
            let clause2 = manager.mk_or([not_lhs_nnf, not_rhs_nnf]);

            let result = manager.mk_and([clause1, clause2]);

            if negate {
                to_nnf_cached(result, manager, cache, true)
            } else {
                result
            }
        }

        // Quantifiers
        Some(TermKind::Forall {
            vars,
            body,
            patterns,
        }) => {
            let body_nnf = to_nnf_cached(body, manager, cache, negate);

            // Resolve strings first to avoid borrowing issues
            let var_names: Vec<_> = vars
                .iter()
                .map(|(s, sort)| (manager.resolve_str(*s).to_string(), *sort))
                .collect();

            if negate {
                // ¬∀x. P(x) = ∃x. ¬P(x)
                manager.mk_exists_with_patterns(
                    var_names.iter().map(|(s, sort)| (s.as_str(), *sort)),
                    body_nnf,
                    patterns,
                )
            } else {
                manager.mk_forall_with_patterns(
                    var_names.iter().map(|(s, sort)| (s.as_str(), *sort)),
                    body_nnf,
                    patterns,
                )
            }
        }

        Some(TermKind::Exists {
            vars,
            body,
            patterns,
        }) => {
            let body_nnf = to_nnf_cached(body, manager, cache, negate);

            // Resolve strings first to avoid borrowing issues
            let var_names: Vec<_> = vars
                .iter()
                .map(|(s, sort)| (manager.resolve_str(*s).to_string(), *sort))
                .collect();

            if negate {
                // ¬∃x. P(x) = ∀x. ¬P(x)
                manager.mk_forall_with_patterns(
                    var_names.iter().map(|(s, sort)| (s.as_str(), *sort)),
                    body_nnf,
                    patterns,
                )
            } else {
                manager.mk_exists_with_patterns(
                    var_names.iter().map(|(s, sort)| (s.as_str(), *sort)),
                    body_nnf,
                    patterns,
                )
            }
        }

        // For other terms, just apply negation if needed
        _ => {
            if negate {
                manager.mk_not(term_id)
            } else {
                term_id
            }
        }
    };

    cache.insert((term_id, negate), result);
    result
}

/// Check if a term is in NNF (Negation Normal Form)
#[must_use]
pub fn is_nnf(term_id: TermId, manager: &TermManager) -> bool {
    is_nnf_impl(term_id, manager, &mut FxHashSet::default())
}

fn is_nnf_impl(term_id: TermId, manager: &TermManager, visited: &mut FxHashSet<TermId>) -> bool {
    if !visited.insert(term_id) {
        return true;
    }

    match manager.get(term_id).map(|t| &t.kind) {
        None | Some(TermKind::True | TermKind::False | TermKind::Var(_)) => true,

        // Negation is only allowed on variables
        Some(TermKind::Not(arg)) => {
            matches!(manager.get(*arg).map(|t| &t.kind), Some(TermKind::Var(_)))
        }

        // Implications not allowed in NNF
        Some(TermKind::Implies(_, _)) => false,

        Some(TermKind::And(args) | TermKind::Or(args)) => {
            args.iter().all(|&a| is_nnf_impl(a, manager, visited))
        }

        Some(TermKind::Forall { body, .. } | TermKind::Exists { body, .. }) => {
            is_nnf_impl(*body, manager, visited)
        }

        // Other terms are considered okay
        _ => true,
    }
}

/// Skolemize a formula by eliminating existential quantifiers
///
/// Skolemization replaces existentially quantified variables with fresh
/// function symbols (Skolem functions). For example:
/// - ∃x. P(x) becomes P(sk_0)
/// - ∀y. ∃x. P(x, y) becomes ∀y. P(f_sk_0(y), y)
///
/// The formula should be in NNF for best results.
pub fn skolemize(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut counter = 0;
    let universal_vars = Vec::new();
    skolemize_impl(term_id, manager, &universal_vars, &mut counter)
}

fn skolemize_impl(
    term_id: TermId,
    manager: &mut TermManager,
    universal_vars: &[lasso::Spur],
    counter: &mut usize,
) -> TermId {
    match manager.get(term_id).map(|t| t.kind.clone()) {
        None
        | Some(
            TermKind::True
            | TermKind::False
            | TermKind::Var(_)
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. },
        ) => term_id,

        Some(TermKind::Not(arg)) => {
            let sk_arg = skolemize_impl(arg, manager, universal_vars, counter);
            manager.mk_not(sk_arg)
        }

        Some(TermKind::And(args)) => {
            let sk_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| skolemize_impl(a, manager, universal_vars, counter))
                .collect();
            manager.mk_and(sk_args)
        }

        Some(TermKind::Or(args)) => {
            let sk_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| skolemize_impl(a, manager, universal_vars, counter))
                .collect();
            manager.mk_or(sk_args)
        }

        Some(TermKind::Implies(lhs, rhs)) => {
            let sk_lhs = skolemize_impl(lhs, manager, universal_vars, counter);
            let sk_rhs = skolemize_impl(rhs, manager, universal_vars, counter);
            manager.mk_implies(sk_lhs, sk_rhs)
        }

        Some(TermKind::Forall { vars, body, .. }) => {
            // Add universal vars to context
            let mut new_universal_vars = universal_vars.to_vec();
            new_universal_vars.extend(vars.iter().map(|(s, _)| *s));

            let sk_body = skolemize_impl(body, manager, &new_universal_vars, counter);

            // Recreate forall with skolemized body
            // Resolve strings first to avoid borrowing issues
            let var_names: Vec<_> = vars
                .iter()
                .map(|(s, sort)| (manager.resolve_str(*s).to_string(), *sort))
                .collect();
            manager.mk_forall(
                var_names.iter().map(|(s, sort)| (s.as_str(), *sort)),
                sk_body,
            )
        }

        Some(TermKind::Exists { vars, body, .. }) => {
            // Create Skolem substitution for existential vars
            let mut subst = FxHashMap::default();

            for (var_name, var_sort) in &vars {
                let skolem_name = format!("sk_{}", counter);
                *counter += 1;

                // Resolve string first to avoid borrowing issues
                let var_name_str = manager.resolve_str(*var_name).to_string();

                // Get the term ID for this variable
                let var_id = manager.mk_var(&var_name_str, *var_sort);

                // Create Skolem function/constant
                let skolem_term = if universal_vars.is_empty() {
                    // No universal vars: create Skolem constant
                    manager.mk_var(&skolem_name, *var_sort)
                } else {
                    // Has universal vars: create Skolem function
                    // Resolve universal var names first
                    let univ_var_names: Vec<_> = universal_vars
                        .iter()
                        .map(|s| manager.resolve_str(*s).to_string())
                        .collect();

                    let univ_var_ids: SmallVec<[TermId; 4]> = univ_var_names
                        .iter()
                        .map(|s| manager.mk_var(s, manager.sorts.bool_sort))
                        .collect();

                    manager.mk_apply(&skolem_name, univ_var_ids, *var_sort)
                };

                subst.insert(var_id, skolem_term);
            }

            // Apply substitution to body
            let substituted = manager.substitute(body, &subst);

            // Recursively skolemize the result
            skolemize_impl(substituted, manager, universal_vars, counter)
        }

        // For other terms, recurse on children
        _ => term_id,
    }
}

/// Eliminate universal quantifiers by replacing them with fresh variables
///
/// This is useful for converting formulas to quantifier-free form when
/// combined with Skolemization. The formula should have existential
/// quantifiers eliminated first (via Skolemization).
pub fn eliminate_universal_quantifiers(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut counter = 0;
    eliminate_universal_impl(term_id, manager, &mut counter)
}

fn eliminate_universal_impl(
    term_id: TermId,
    manager: &mut TermManager,
    counter: &mut usize,
) -> TermId {
    match manager.get(term_id).map(|t| t.kind.clone()) {
        None
        | Some(
            TermKind::True
            | TermKind::False
            | TermKind::Var(_)
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. },
        ) => term_id,

        Some(TermKind::Not(arg)) => {
            let new_arg = eliminate_universal_impl(arg, manager, counter);
            manager.mk_not(new_arg)
        }

        Some(TermKind::And(args)) => {
            let new_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| eliminate_universal_impl(a, manager, counter))
                .collect();
            manager.mk_and(new_args)
        }

        Some(TermKind::Or(args)) => {
            let new_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&a| eliminate_universal_impl(a, manager, counter))
                .collect();
            manager.mk_or(new_args)
        }

        Some(TermKind::Forall { vars, body, .. }) => {
            // Replace quantified vars with fresh constants
            let mut subst = FxHashMap::default();

            for (var_name, var_sort) in &vars {
                let fresh_name = format!("u_{}", counter);
                *counter += 1;

                // Resolve string first to avoid borrowing issues
                let var_name_str = manager.resolve_str(*var_name).to_string();

                let var_id = manager.mk_var(&var_name_str, *var_sort);
                let fresh_var = manager.mk_var(&fresh_name, *var_sort);

                subst.insert(var_id, fresh_var);
            }

            let substituted = manager.substitute(body, &subst);
            eliminate_universal_impl(substituted, manager, counter)
        }

        _ => term_id,
    }
}
