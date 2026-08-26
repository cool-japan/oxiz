//! Tests for [`super::to_cnf_tseitin`].
//!
//! Three things are pinned here: the *shape* of the clauses each connective
//! is given (so a future edit to a definition is a visible diff, not a
//! silent change of meaning), the *size* guarantee that distinguishes this
//! pass from [`crate::ast::normal_forms::to_cnf`], and the *contract* --
//! equisatisfiability, checked by brute force over the truth table for small
//! formulas.

use super::*;
use crate::ast::normal_forms::{extract_cnf_clauses, to_cnf};

// ===========================================================================
// Helpers
// ===========================================================================

/// Render a literal as `name` / `!name` for readable clause assertions.
fn literal_name(manager: &TermManager, id: TermId) -> String {
    match manager.get(id).map(|term| &term.kind) {
        Some(TermKind::Var(name)) => manager.resolve_str(*name).to_string(),
        Some(TermKind::Not(inner)) => format!("!{}", literal_name(manager, *inner)),
        Some(TermKind::True) => "true".to_string(),
        Some(TermKind::False) => "false".to_string(),
        _ => format!("<term {}>", id.raw()),
    }
}

/// The CNF's clauses as sorted literal-name lists, in conjunct order.
fn clause_names(manager: &TermManager, cnf: TermId) -> Vec<Vec<String>> {
    extract_cnf_clauses(cnf, manager)
        .into_iter()
        .map(|clause| {
            let mut names: Vec<String> = clause
                .into_iter()
                .map(|lit| literal_name(manager, lit))
                .collect();
            names.sort();
            names
        })
        .collect()
}

/// Evaluate a boolean formula under `assignment`.
///
/// Iterative (the formulas here are small, but the walk is the same shape as
/// the encoder's and there is no reason to reintroduce recursion in a test
/// helper). Returns `None` for anything outside the boolean fragment these
/// tests use, so an accidental non-boolean term fails loudly instead of
/// defaulting.
fn eval(manager: &TermManager, root: TermId, assignment: &FxHashMap<TermId, bool>) -> Option<bool> {
    let mut memo: FxHashMap<TermId, bool> = FxHashMap::default();
    let mut work: Vec<(TermId, bool)> = vec![(root, false)];

    while let Some((id, children_done)) = work.pop() {
        if memo.contains_key(&id) {
            continue;
        }
        let kind = manager.get(id)?.kind.clone();

        if !children_done {
            let operands: Vec<TermId> = match &kind {
                TermKind::True | TermKind::False | TermKind::Var(_) => Vec::new(),
                TermKind::Not(arg) => vec![*arg],
                TermKind::And(args) | TermKind::Or(args) => args.iter().copied().collect(),
                TermKind::Implies(lhs, rhs) | TermKind::Xor(lhs, rhs) | TermKind::Eq(lhs, rhs) => {
                    vec![*lhs, *rhs]
                }
                TermKind::Ite(cond, then_branch, else_branch) => {
                    vec![*cond, *then_branch, *else_branch]
                }
                _ => return None,
            };
            work.push((id, true));
            for operand in operands.into_iter().rev() {
                work.push((operand, false));
            }
            continue;
        }

        let value = match &kind {
            TermKind::True => true,
            TermKind::False => false,
            TermKind::Var(_) => *assignment.get(&id)?,
            TermKind::Not(arg) => !memo.get(arg)?,
            TermKind::And(args) => args
                .iter()
                .try_fold(true, |acc, arg| memo.get(arg).map(|value| acc && *value))?,
            TermKind::Or(args) => args
                .iter()
                .try_fold(false, |acc, arg| memo.get(arg).map(|value| acc || *value))?,
            TermKind::Implies(lhs, rhs) => !*memo.get(lhs)? || *memo.get(rhs)?,
            TermKind::Xor(lhs, rhs) => memo.get(lhs)? ^ memo.get(rhs)?,
            TermKind::Eq(lhs, rhs) => memo.get(lhs)? == memo.get(rhs)?,
            TermKind::Ite(cond, then_branch, else_branch) => {
                if *memo.get(cond)? {
                    *memo.get(then_branch)?
                } else {
                    *memo.get(else_branch)?
                }
            }
            _ => return None,
        };
        memo.insert(id, value);
    }

    memo.get(&root).copied()
}

/// The boolean variables occurring in `term`, in a stable order.
fn bool_vars(manager: &TermManager, term: TermId) -> Vec<TermId> {
    let bool_sort = manager.sorts.bool_sort;
    let mut vars: Vec<TermId> = manager
        .free_vars_including_patterns(term)
        .into_iter()
        .filter(|&id| {
            manager
                .get(id)
                .is_some_and(|t| t.sort == bool_sort && matches!(t.kind, TermKind::Var(_)))
        })
        .collect();
    vars.sort_by_key(|id| id.raw());
    vars.dedup();
    vars
}

/// Enumerate every assignment of `vars`, extending `base`.
fn assignments(vars: &[TermId], base: &FxHashMap<TermId, bool>) -> Vec<FxHashMap<TermId, bool>> {
    let mut out = Vec::with_capacity(1usize << vars.len());
    for mask in 0..(1usize << vars.len()) {
        let mut assignment = base.clone();
        for (bit, &var) in vars.iter().enumerate() {
            assignment.insert(var, mask & (1 << bit) != 0);
        }
        out.push(assignment);
    }
    out
}

/// Brute-force the equisatisfiability contract for a small formula: for
/// every assignment of the *original* variables, the formula holds exactly
/// when the CNF can be extended to a model by choosing values for the
/// introduced `tseitin!*` variables.
fn assert_equisatisfiable(manager: &mut TermManager, formula: TermId) {
    let cnf = to_cnf_tseitin(formula, manager);

    let original_vars = bool_vars(manager, formula);
    let cnf_vars = bool_vars(manager, cnf);
    let fresh_vars: Vec<TermId> = cnf_vars
        .iter()
        .copied()
        .filter(|id| !original_vars.contains(id))
        .collect();

    assert!(
        original_vars.len() <= 4 && fresh_vars.len() <= 8,
        "brute force expects a small formula, got {} + {} variables",
        original_vars.len(),
        fresh_vars.len()
    );

    let empty = FxHashMap::default();
    for original in assignments(&original_vars, &empty) {
        let expected =
            eval(manager, formula, &original).expect("formula is in the boolean fragment");
        let extendable = assignments(&fresh_vars, &original)
            .into_iter()
            .any(|full| eval(manager, cnf, &full) == Some(true));
        assert_eq!(
            expected, extendable,
            "equisatisfiability broken for one assignment of the original variables"
        );
    }
}

/// `n` fresh boolean variables named `p0..p{n-1}`.
fn bool_var_row(manager: &mut TermManager, count: usize) -> Vec<TermId> {
    let bool_sort = manager.sorts.bool_sort;
    (0..count)
        .map(|index| manager.mk_var(&format!("p{index}"), bool_sort))
        .collect()
}

// ===========================================================================
// Clause-shape pins, one per connective
// ===========================================================================

#[test]
fn and_definition_clause_shape() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 2);
    let formula = manager.mk_and(vars);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    assert_eq!(
        clause_names(&manager, cnf),
        vec![
            vec!["tseitin!0".to_string()],
            vec!["!tseitin!0".to_string(), "p0".to_string()],
            vec!["!tseitin!0".to_string(), "p1".to_string()],
            vec![
                "!p0".to_string(),
                "!p1".to_string(),
                "tseitin!0".to_string()
            ],
        ]
    );
}

#[test]
fn or_definition_clause_shape() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 2);
    let formula = manager.mk_or(vars);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    assert_eq!(
        clause_names(&manager, cnf),
        vec![
            vec!["tseitin!0".to_string()],
            vec!["!p0".to_string(), "tseitin!0".to_string()],
            vec!["!p1".to_string(), "tseitin!0".to_string()],
            vec!["!tseitin!0".to_string(), "p0".to_string(), "p1".to_string()],
        ]
    );
}

/// `Not` costs no definitional variable at all: the negation of a literal is
/// already a literal.
#[test]
fn not_needs_no_definitional_variable() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 1);
    let formula = manager.mk_not(vars[0]);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    assert_eq!(clause_names(&manager, cnf), vec![vec!["!p0".to_string()]]);
    assert_eq!(cnf, formula);
}

#[test]
fn implies_definition_clause_shape() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 2);
    let formula = manager.mk_implies(vars[0], vars[1]);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    assert_eq!(
        clause_names(&manager, cnf),
        vec![
            vec!["tseitin!0".to_string()],
            vec![
                "!p0".to_string(),
                "!tseitin!0".to_string(),
                "p1".to_string()
            ],
            vec!["p0".to_string(), "tseitin!0".to_string()],
            vec!["!p1".to_string(), "tseitin!0".to_string()],
        ]
    );
}

#[test]
fn xor_definition_clause_shape() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 2);
    let formula = manager.mk_xor(vars[0], vars[1]);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    assert_eq!(
        clause_names(&manager, cnf),
        vec![
            vec!["tseitin!0".to_string()],
            vec!["!tseitin!0".to_string(), "p0".to_string(), "p1".to_string()],
            vec![
                "!p0".to_string(),
                "!p1".to_string(),
                "!tseitin!0".to_string()
            ],
            vec!["!p0".to_string(), "p1".to_string(), "tseitin!0".to_string()],
            vec!["!p1".to_string(), "p0".to_string(), "tseitin!0".to_string()],
        ]
    );
}

/// `Eq` between two booleans is `Iff` and gets a definition; `Eq` over any
/// other sort stays an atom.
#[test]
fn iff_definition_clause_shape() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 2);
    let formula = manager.mk_eq(vars[0], vars[1]);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    assert_eq!(
        clause_names(&manager, cnf),
        vec![
            vec!["tseitin!0".to_string()],
            vec![
                "!p0".to_string(),
                "!tseitin!0".to_string(),
                "p1".to_string()
            ],
            vec![
                "!p1".to_string(),
                "!tseitin!0".to_string(),
                "p0".to_string()
            ],
            vec!["p0".to_string(), "p1".to_string(), "tseitin!0".to_string()],
            vec![
                "!p0".to_string(),
                "!p1".to_string(),
                "tseitin!0".to_string()
            ],
        ]
    );
}

#[test]
fn ite_definition_clause_shape() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 3);
    let formula = manager.mk_ite(vars[0], vars[1], vars[2]);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    assert_eq!(
        clause_names(&manager, cnf),
        vec![
            vec!["tseitin!0".to_string()],
            vec![
                "!p0".to_string(),
                "!tseitin!0".to_string(),
                "p1".to_string()
            ],
            vec!["!tseitin!0".to_string(), "p0".to_string(), "p2".to_string()],
            vec![
                "!p0".to_string(),
                "!p1".to_string(),
                "tseitin!0".to_string()
            ],
            vec!["!p2".to_string(), "p0".to_string(), "tseitin!0".to_string()],
        ]
    );
}

/// A boolean-sorted term that is not a connective (here an uninterpreted
/// predicate application) is copied through as an atom.
#[test]
fn non_connective_boolean_terms_stay_atoms() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let predicate = manager.mk_apply("q", vec![x], bool_sort);
    let int_eq = manager.mk_eq(x, y);
    let formula = manager.mk_and([predicate, int_eq]);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    let clauses = extract_cnf_clauses(cnf, &manager);

    // One unit clause for the root plus three definitional clauses for the
    // binary `And`; both operands appear verbatim.
    assert_eq!(clauses.len(), 4);
    assert!(clauses.iter().flatten().any(|&lit| lit == predicate));
    assert!(clauses.iter().flatten().any(|&lit| lit == int_eq));
}

// ===========================================================================
// Size: linear where `to_cnf` is exponential
// ===========================================================================

/// Thirty nested `Iff`s: four clauses and one variable per level, and it has
/// to be instant.
///
/// An *equivalent* CNF of this formula has 2^29 clauses -- the size is
/// forced by the contract, not by the algorithm -- which is exactly why
/// [`to_cnf_tseitin`] exists alongside [`to_cnf`].
#[test]
fn thirty_nested_iffs_stay_linear() {
    const LEVELS: usize = 30;

    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, LEVELS + 1);

    let mut formula = vars[0];
    for &var in vars.iter().skip(1) {
        formula = manager.mk_eq(var, formula);
    }

    let terms_before = manager.len();
    let started = oxiz_time::Instant::now();
    let cnf = to_cnf_tseitin(formula, &mut manager);
    let elapsed = started.elapsed();

    let clauses = extract_cnf_clauses(cnf, &manager);
    // 1 unit clause for the root + 4 per `Iff` level.
    assert_eq!(clauses.len(), 1 + 4 * LEVELS);
    // Every clause is short: definitional clauses never grow with depth.
    assert!(clauses.iter().all(|clause| clause.len() <= 3));
    // Term growth is linear in the number of levels, not exponential.
    assert!(
        manager.len() - terms_before < 40 * LEVELS,
        "term count grew by {} for {LEVELS} levels",
        manager.len() - terms_before
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "conversion took {elapsed:?}"
    );
}

/// The size contrast against [`to_cnf`] on a shape `to_cnf` *does* expand:
/// a disjunction of ten conjunctions distributes into 2^10 clauses, while
/// the definitional form needs a constant number per operand.
#[test]
fn distribution_blows_up_where_definitions_do_not() {
    const PAIRS: usize = 10;

    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 2 * PAIRS);
    let conjunctions: Vec<TermId> = (0..PAIRS)
        .map(|index| manager.mk_and([vars[2 * index], vars[2 * index + 1]]))
        .collect();
    let formula = manager.mk_or(conjunctions);

    let distributed = to_cnf(formula, &mut manager);
    let definitional = to_cnf_tseitin(formula, &mut manager);

    let distributed_clauses = extract_cnf_clauses(distributed, &manager);
    let definitional_clauses = extract_cnf_clauses(definitional, &manager);

    assert_eq!(distributed_clauses.len(), 1 << PAIRS);
    // 1 root + 3 per inner `And` + (PAIRS + 1) for the outer `Or`.
    assert_eq!(definitional_clauses.len(), 1 + 3 * PAIRS + PAIRS + 1);
}

/// Shared subformulas are named once, so a DAG does not re-expand along
/// every path.
#[test]
fn shared_subformulas_are_named_once() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 4);
    let shared = manager.mk_xor(vars[0], vars[1]);
    let left = manager.mk_or([shared, vars[2]]);
    let right = manager.mk_or([shared, vars[3]]);
    let formula = manager.mk_and([left, right]);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    let original: Vec<TermId> = bool_vars(&manager, formula);
    let introduced: Vec<TermId> = bool_vars(&manager, cnf)
        .into_iter()
        .filter(|id| !original.contains(id))
        .collect();

    // One variable per *distinct* compound subformula: the shared `Xor`,
    // the two `Or`s, the `And`. Re-expanding the shared node along both
    // paths would make it five.
    assert_eq!(introduced.len(), 4);

    let clauses = extract_cnf_clauses(cnf, &manager);
    // root(1) + Xor(4) + Or(3) + Or(3) + And(3).
    assert_eq!(clauses.len(), 14);
}

// ===========================================================================
// Contract: equisatisfiability
// ===========================================================================

#[test]
fn equisatisfiable_on_nested_iff() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 3);
    let inner = manager.mk_eq(vars[1], vars[2]);
    let formula = manager.mk_eq(vars[0], inner);
    assert_equisatisfiable(&mut manager, formula);
}

#[test]
fn equisatisfiable_on_mixed_connectives() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 3);
    let conjunction = manager.mk_and([vars[0], vars[1]]);
    let negated = manager.mk_not(vars[2]);
    let disjunction = manager.mk_or([vars[1], negated]);
    let formula = manager.mk_implies(conjunction, disjunction);
    assert_equisatisfiable(&mut manager, formula);
}

#[test]
fn equisatisfiable_on_xor_and_ite() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 3);
    let negated = manager.mk_not(vars[2]);
    let branch = manager.mk_ite(vars[0], vars[1], negated);
    let formula = manager.mk_xor(branch, vars[1]);
    assert_equisatisfiable(&mut manager, formula);
}

/// An unsatisfiable input must stay unsatisfiable: no assignment of the
/// original variables extends to a model of the CNF.
#[test]
fn equisatisfiable_on_a_contradiction() {
    let mut manager = TermManager::new();
    let vars = bool_var_row(&mut manager, 1);
    let negated = manager.mk_not(vars[0]);
    let formula = manager.mk_and([vars[0], negated]);
    assert_equisatisfiable(&mut manager, formula);
}

// ===========================================================================
// Fresh variables and traversal
// ===========================================================================

/// A user variable already named `tseitin!0` must not be captured: minting
/// checks the manager rather than assuming the prefix is unused.
#[test]
fn minting_skips_names_already_in_use() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let squatter = manager.mk_var("tseitin!0", bool_sort);
    let vars = bool_var_row(&mut manager, 2);
    let formula = manager.mk_and(vars);

    let cnf = to_cnf_tseitin(formula, &mut manager);
    let introduced: Vec<TermId> = bool_vars(&manager, cnf)
        .into_iter()
        .filter(|id| !bool_vars(&manager, formula).contains(id))
        .collect();

    assert_eq!(introduced.len(), 1);
    assert_ne!(introduced[0], squatter);
    assert_eq!(literal_name(&manager, introduced[0]), "tseitin!1");
}

/// The traversal is a heap walk: a 6 250-level implication chain converts
/// on a 128 KiB stack.
#[test]
fn deep_formula_converts_on_a_small_stack() {
    const LEVELS: usize = 6_250;
    const STACK_SIZE: usize = 1 << 17;

    let worker = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut manager = TermManager::new();
            let bool_sort = manager.sorts.bool_sort;
            let mut formula = manager.mk_var("p0", bool_sort);
            for level in 1..=LEVELS {
                let var = manager.mk_var(&format!("p{level}"), bool_sort);
                formula = manager.mk_implies(var, formula);
            }

            let cnf = to_cnf_tseitin(formula, &mut manager);
            let clauses = extract_cnf_clauses(cnf, &manager);
            assert_eq!(clauses.len(), 1 + 3 * LEVELS);
        })
        .expect("spawning the 128 KiB worker thread");

    worker
        .join()
        .expect("the deep conversion must not overflow");
}
