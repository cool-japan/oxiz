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

fn collect_boolean_vars(manager: &TermManager, assertions: &[TermId]) -> Vec<TermId> {
    let mut ordered = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &assertion in assertions {
        collect_boolean_vars_in_term(manager, assertion, &mut seen, &mut ordered);
    }
    ordered
}

fn collect_boolean_vars_in_term(
    manager: &TermManager,
    term: TermId,
    seen: &mut std::collections::HashSet<TermId>,
    ordered: &mut Vec<TermId>,
) {
    match manager.get(term).map(|node| &node.kind) {
        Some(TermKind::Var(_)) if seen.insert(term) => {
            ordered.push(term);
        }
        Some(TermKind::Not(inner)) => collect_boolean_vars_in_term(manager, *inner, seen, ordered),
        Some(TermKind::And(args)) | Some(TermKind::Or(args)) | Some(TermKind::Distinct(args)) => {
            for &arg in args {
                collect_boolean_vars_in_term(manager, arg, seen, ordered);
            }
        }
        Some(TermKind::Implies(lhs, rhs))
        | Some(TermKind::Eq(lhs, rhs))
        | Some(TermKind::Lt(lhs, rhs))
        | Some(TermKind::Le(lhs, rhs))
        | Some(TermKind::Gt(lhs, rhs))
        | Some(TermKind::Ge(lhs, rhs))
        | Some(TermKind::Sub(lhs, rhs)) => {
            collect_boolean_vars_in_term(manager, *lhs, seen, ordered);
            collect_boolean_vars_in_term(manager, *rhs, seen, ordered);
        }
        Some(TermKind::Ite(cond, then_branch, else_branch)) => {
            collect_boolean_vars_in_term(manager, *cond, seen, ordered);
            collect_boolean_vars_in_term(manager, *then_branch, seen, ordered);
            collect_boolean_vars_in_term(manager, *else_branch, seen, ordered);
        }
        Some(TermKind::Add(args)) | Some(TermKind::Mul(args)) => {
            for &arg in args {
                collect_boolean_vars_in_term(manager, arg, seen, ordered);
            }
        }
        _ => {}
    }
}

fn collect_activity(
    manager: &TermManager,
    assertions: &[TermId],
    vars: &[TermId],
) -> HashMap<Var, f64> {
    let var_to_index: std::collections::HashMap<TermId, usize> =
        vars.iter().enumerate().map(|(idx, &var)| (var, idx)).collect();
    let mut activity = HashMap::new();

    for &assertion in assertions {
        bump_activity(manager, assertion, &var_to_index, &mut activity);
    }

    activity
}

fn bump_activity(
    manager: &TermManager,
    term: TermId,
    var_to_index: &std::collections::HashMap<TermId, usize>,
    activity: &mut HashMap<Var, f64>,
) {
    match manager.get(term).map(|node| &node.kind) {
        Some(TermKind::Var(_)) => {
            if let Some(&idx) = var_to_index.get(&term) {
                let entry = activity.entry(Var::new(idx as u32)).or_insert(0.0);
                *entry += 1.0;
            }
        }
        Some(TermKind::Not(inner)) => bump_activity(manager, *inner, var_to_index, activity),
        Some(TermKind::And(args)) | Some(TermKind::Or(args)) | Some(TermKind::Distinct(args)) => {
            for &arg in args {
                bump_activity(manager, arg, var_to_index, activity);
            }
        }
        Some(TermKind::Implies(lhs, rhs))
        | Some(TermKind::Eq(lhs, rhs))
        | Some(TermKind::Lt(lhs, rhs))
        | Some(TermKind::Le(lhs, rhs))
        | Some(TermKind::Gt(lhs, rhs))
        | Some(TermKind::Ge(lhs, rhs))
        | Some(TermKind::Sub(lhs, rhs)) => {
            bump_activity(manager, *lhs, var_to_index, activity);
            bump_activity(manager, *rhs, var_to_index, activity);
        }
        Some(TermKind::Ite(cond, then_branch, else_branch)) => {
            bump_activity(manager, *cond, var_to_index, activity);
            bump_activity(manager, *then_branch, var_to_index, activity);
            bump_activity(manager, *else_branch, var_to_index, activity);
        }
        Some(TermKind::Add(args)) | Some(TermKind::Mul(args)) => {
            for &arg in args {
                bump_activity(manager, arg, var_to_index, activity);
            }
        }
        _ => {}
    }
}
