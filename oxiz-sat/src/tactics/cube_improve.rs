use crate::cube::{CubeConfig, CubeGenerator};
use crate::literal::Var;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::error::Result;
use oxiz_core::tactic::{Goal, TacticResult};
use std::collections::HashMap;

/// Generates cube-based sub-goals guided by variable occurrence activity.
pub struct CubeImproveTactic<'a> {
    manager: &'a mut TermManager,
    config: CubeConfig,
}

impl<'a> CubeImproveTactic<'a> {
    /// Create a new cube-improve tactic.
    pub fn new(manager: &'a mut TermManager) -> Self {
        let config = CubeConfig {
            vsids_guided: true,
            min_cube_size: 1,
            ..CubeConfig::default()
        };
        Self { manager, config }
    }

    /// Apply the tactic and split the goal into cube-constrained sub-goals.
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let vars = collect_boolean_vars(self.manager, &goal.assertions);
        if vars.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let activity = collect_activity(self.manager, &goal.assertions, &vars);
        let generator = CubeGenerator::new(vars.len(), self.config.clone());
        let cubes = generator.generate_vsids_guided(&activity);
        if cubes.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let subgoals = cubes
            .into_iter()
            .map(|cube| {
                let mut assertions = goal.assertions.clone();
                for lit in cube.literals {
                    let term = vars[lit.var().index()];
                    assertions.push(if lit.is_pos() {
                        term
                    } else {
                        self.manager.mk_not(term)
                    });
                }

                Goal {
                    assertions,
                    precision: goal.precision,
                }
            })
            .collect();

        Ok(TacticResult::SubGoals(subgoals))
    }
}

/// Push the sub-terms of `term` onto `stack` in reverse order, so that a
/// LIFO `Vec` work-stack pops them left-to-right — reproducing exactly the
/// pre-order, left-to-right traversal of the former recursive walks.
///
/// Leaves and term kinds this tactic does not descend into (numerals,
/// quantifiers, bit-vector operations, ...) push nothing, which is the same
/// no-op the recursive `_ => {}` arms performed.
fn push_sub_terms(manager: &TermManager, term: TermId, stack: &mut Vec<TermId>) {
    let Some(node) = manager.get(term) else {
        return;
    };
    match &node.kind {
        TermKind::Not(inner) => stack.push(*inner),
        TermKind::And(args)
        | TermKind::Or(args)
        | TermKind::Distinct(args)
        | TermKind::Add(args)
        | TermKind::Mul(args) => stack.extend(args.iter().rev().copied()),
        TermKind::Implies(lhs, rhs)
        | TermKind::Eq(lhs, rhs)
        | TermKind::Lt(lhs, rhs)
        | TermKind::Le(lhs, rhs)
        | TermKind::Gt(lhs, rhs)
        | TermKind::Ge(lhs, rhs)
        | TermKind::Sub(lhs, rhs) => {
            stack.push(*rhs);
            stack.push(*lhs);
        }
        TermKind::Ite(cond, then_branch, else_branch) => {
            stack.push(*else_branch);
            stack.push(*then_branch);
            stack.push(*cond);
        }
        _ => {}
    }
}

fn collect_boolean_vars(manager: &TermManager, assertions: &[TermId]) -> Vec<TermId> {
    let mut ordered = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut visited = std::collections::HashSet::new();
    for &assertion in assertions {
        collect_boolean_vars_in_term(manager, assertion, &mut seen, &mut visited, &mut ordered);
    }
    ordered
}

/// Collect the boolean variables reachable from `term`, in first-occurrence
/// pre-order.
///
/// Uses an explicit heap work-stack rather than recursion: assertion terms
/// come from user input, so nesting depth is attacker-controlled and this
/// function returns `()` — there is no channel through which a depth cap
/// could report that it gave up, so a cap could only silently drop
/// variables and corrupt the cube split.
///
/// `visited` is a genuine traversal-pruning set (the pre-existing `seen`
/// set only ever recorded `Var` nodes, so shared sub-terms of the
/// hash-consed DAG were re-expanded exponentially). Pruning cannot change
/// the result: a second visit to an already-fully-explored node can only
/// reach variables that are already in `seen`, and `seen` suppresses them.
fn collect_boolean_vars_in_term(
    manager: &TermManager,
    term: TermId,
    seen: &mut std::collections::HashSet<TermId>,
    visited: &mut std::collections::HashSet<TermId>,
    ordered: &mut Vec<TermId>,
) {
    let mut stack = vec![term];
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        if let Some(TermKind::Var(_)) = manager.get(current).map(|node| &node.kind)
            && seen.insert(current)
        {
            ordered.push(current);
        }
        push_sub_terms(manager, current, &mut stack);
    }
}

fn collect_activity(
    manager: &TermManager,
    assertions: &[TermId],
    vars: &[TermId],
) -> HashMap<Var, f64> {
    let var_to_index: std::collections::HashMap<TermId, usize> = vars
        .iter()
        .enumerate()
        .map(|(idx, &var)| (var, idx))
        .collect();
    let mut activity = HashMap::new();

    for &assertion in assertions {
        bump_activity(manager, assertion, &var_to_index, &mut activity);
    }

    activity
}

/// Bump the activity score of every variable occurring in `term`.
///
/// Like [`collect_boolean_vars_in_term`], this runs on an explicit heap
/// work-stack: assertion nesting is attacker-controlled and the function
/// returns `()`, so a depth cap could only silently skew the heuristic.
///
/// Each *distinct* sub-term of the hash-consed DAG is visited once per
/// assertion. The recursive form instead walked the term's tree unfolding,
/// so a variable under a shared node was counted once per path reaching it
/// — exponentially many times for a DAG with sharing. Counting DAG
/// occurrences is both the intended "how many places does this variable
/// appear" measure and the only one computable in polynomial time.
fn bump_activity(
    manager: &TermManager,
    term: TermId,
    var_to_index: &std::collections::HashMap<TermId, usize>,
    activity: &mut HashMap<Var, f64>,
) {
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![term];
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        if let Some(TermKind::Var(_)) = manager.get(current).map(|node| &node.kind)
            && let Some(&idx) = var_to_index.get(&current)
        {
            let entry = activity.entry(Var::new(idx as u32)).or_insert(0.0);
            *entry += 1.0;
        }
        push_sub_terms(manager, current, &mut stack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::ast::TermManager;
    use oxiz_core::tactic::Goal;

    fn make_manager_with_bool_vars(count: usize) -> (TermManager, Vec<TermId>) {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let vars: Vec<TermId> = (0..count)
            .map(|i| manager.mk_var(&format!("b{i}"), bool_sort))
            .collect();
        (manager, vars)
    }

    /// A term nested 12_500 levels deep, walked on a 128 KiB stack. The
    /// assertion is that the walk *returns* — a stack overflow aborts the
    /// whole process, so returning at all proves it no longer recurses.
    ///
    /// `Sub` is used for the spine because `mk_and`/`mk_or` flatten nested
    /// conjunctions/disjunctions and `mk_not` collapses double negation, so
    /// none of them can build a deep spine through the public builder.
    #[test]
    fn test_collect_boolean_vars_deep_term_does_not_overflow() {
        let worker = std::thread::Builder::new().stack_size(1 << 17).spawn(|| {
            let (mut manager, vars) = make_manager_with_bool_vars(2);
            let mut term = vars[0];
            for _ in 0..12_500 {
                term = manager.mk_sub(term, vars[1]);
            }
            let collected = collect_boolean_vars(&manager, &[term]);
            let activity = collect_activity(&manager, &[term], &collected);
            (collected, activity.len())
        });
        let (collected, activity_len) = match worker.map(std::thread::JoinHandle::join) {
            Ok(Ok(result)) => result,
            _ => panic!("deep-term worker thread did not complete"),
        };
        // Pre-order, left-to-right: the innermost left operand is `b0`,
        // and `b1` is first seen as the right operand of the outermost
        // `Sub`... which is visited after the whole left spine.
        assert_eq!(collected.len(), 2);
        assert_eq!(activity_len, 2);
    }

    /// A doubling DAG `t_k = (- t_{k-1} t_{k-1})`: 60 levels, 61 distinct
    /// hash-consed nodes, 2^60 tree-unfoldings. Without a traversal-pruning
    /// visited set both walks re-expand every shared node and never finish.
    #[test]
    fn test_shared_dag_is_not_re_expanded() {
        let (mut manager, vars) = make_manager_with_bool_vars(1);
        let mut term = vars[0];
        for _ in 0..60 {
            term = manager.mk_sub(term, term);
        }
        let collected = collect_boolean_vars(&manager, &[term]);
        assert_eq!(collected, vec![vars[0]]);
        let activity = collect_activity(&manager, &[term], &collected);
        assert_eq!(activity.len(), 1);
    }

    /// Semantic pin: variables are reported in first-occurrence pre-order,
    /// left-to-right, and each is reported once.
    #[test]
    fn test_collect_boolean_vars_order_preserved() {
        let (mut manager, vars) = make_manager_with_bool_vars(3);
        // (or b2 (and b0 b1) b0)
        let inner = manager.mk_and(vec![vars[0], vars[1]]);
        let term = manager.mk_or(vec![vars[2], inner, vars[0]]);
        let collected = collect_boolean_vars(&manager, &[term]);
        assert_eq!(collected, vec![vars[2], vars[0], vars[1]]);
    }

    /// Applying CubeImproveTactic to a goal with Boolean vars should produce
    /// at least 2 sub-goals, each containing more assertions than the original.
    #[test]
    fn test_cube_improve_tactic_emits_subgoals_per_cube() {
        let (mut manager, vars) = make_manager_with_bool_vars(4);
        // Build a goal with all four vars as direct assertions
        let goal = Goal::new(vars.clone());
        let original_len = goal.assertions.len();

        let mut tactic = CubeImproveTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("tactic application failed");

        match result {
            TacticResult::SubGoals(subgoals) => {
                assert!(
                    subgoals.len() >= 2,
                    "expected at least 2 sub-goals, got {}",
                    subgoals.len()
                );
                for (i, sg) in subgoals.iter().enumerate() {
                    assert!(
                        sg.assertions.len() > original_len,
                        "subgoal {} has {} assertions, expected more than original {}",
                        i,
                        sg.assertions.len(),
                        original_len
                    );
                }
            }
            other => panic!("expected SubGoals, got {other:?}"),
        }
    }

    /// The precision field of the original goal should be propagated to all sub-goals.
    #[test]
    fn test_cube_improve_precision_preserved() {
        use oxiz_core::tactic::Precision;

        let (mut manager, vars) = make_manager_with_bool_vars(4);
        let goal = Goal {
            assertions: vars.clone(),
            precision: Precision::Over,
        };

        let mut tactic = CubeImproveTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("tactic application failed");

        match result {
            TacticResult::SubGoals(subgoals) => {
                assert!(!subgoals.is_empty());
                for (i, sg) in subgoals.iter().enumerate() {
                    assert_eq!(
                        sg.precision,
                        Precision::Over,
                        "subgoal {} precision should be Over, got {:?}",
                        i,
                        sg.precision
                    );
                }
            }
            other => panic!("expected SubGoals, got {other:?}"),
        }
    }
}
