use crate::literal::{Lit, Var};
use crate::{AutomorphismDetector, SymmetryBreaker, SymmetryBreakingMethod};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::error::Result;
use oxiz_core::tactic::{Goal, TacticResult};
use smallvec::SmallVec;

/// Adds lex-leader symmetry-breaking constraints to a Boolean goal.
pub struct SymmetryBreakTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> SymmetryBreakTactic<'a> {
    /// Create a new symmetry-breaking tactic.
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply symmetry breaking to the current goal.
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let extraction = extract_boolean_clauses(self.manager, &goal.assertions);
        let Some((clauses, term_vars)) = extraction else {
            return Ok(TacticResult::NotApplicable);
        };

        if term_vars.len() < 2 || clauses.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut detector = AutomorphismDetector::new(term_vars.len());
        for clause in clauses {
            detector.add_clause(clause);
        }

        let group = detector.detect_symmetries();
        if group.generators().is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut breaker = SymmetryBreaker::new(group, SymmetryBreakingMethod::Lex);
        breaker.generate_predicates();
        if breaker.get_clauses().is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut assertions = goal.assertions.clone();
        for clause in breaker.get_clauses() {
            assertions.push(clause_to_term(self.manager, clause, &term_vars));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions,
            precision: goal.precision,
        }]))
    }
}

fn extract_boolean_clauses(
    manager: &TermManager,
    assertions: &[TermId],
) -> Option<(Vec<Vec<Lit>>, Vec<TermId>)> {
    let mut ordered_vars = Vec::new();
    let mut var_to_index = std::collections::HashMap::new();
    let mut clauses = Vec::new();

    for &assertion in assertions {
        let mut clause_terms = Vec::new();
        if !collect_clause_literals(manager, assertion, &mut clause_terms) {
            return None;
        }

        let mut clause = Vec::new();
        for (var_term, sign) in clause_terms {
            let idx = if let Some(&idx) = var_to_index.get(&var_term) {
                idx
            } else {
                let idx = ordered_vars.len();
                ordered_vars.push(var_term);
                var_to_index.insert(var_term, idx);
                idx
            };
            let var = Var::new(idx as u32);
            clause.push(if sign { Lit::pos(var) } else { Lit::neg(var) });
        }
        clauses.push(clause);
    }

    Some((clauses, ordered_vars))
}

fn collect_clause_literals(
    manager: &TermManager,
    term: TermId,
    out: &mut Vec<(TermId, bool)>,
) -> bool {
    match manager.get(term).map(|node| &node.kind) {
        Some(TermKind::Var(_)) => {
            out.push((term, true));
            true
        }
        Some(TermKind::Not(inner)) => match manager.get(*inner).map(|node| &node.kind) {
            Some(TermKind::Var(_)) => {
                out.push((*inner, false));
                true
            }
            _ => false,
        },
        Some(TermKind::Or(args)) => args
            .iter()
            .copied()
            .all(|arg| collect_clause_literals(manager, arg, out)),
        _ => false,
    }
}

fn clause_to_term(manager: &mut TermManager, clause: &[Lit], vars: &[TermId]) -> TermId {
    let literals: SmallVec<[TermId; 4]> = clause
        .iter()
        .map(|lit| {
            let base = vars[lit.var().index()];
            if lit.is_pos() {
                base
            } else {
                manager.mk_not(base)
            }
        })
        .collect();

    manager.mk_or(literals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetry_break_adds_clauses() {
        let mut manager = TermManager::new();
        let a = manager.mk_var("a", manager.sorts.bool_sort);
        let b = manager.mk_var("b", manager.sorts.bool_sort);
        let clause = manager.mk_or([a, b]);

        let goal = Goal::new(vec![clause]);
        let mut tactic = SymmetryBreakTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("test operation should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert!(goals[0].assertions.len() > goal.assertions.len());
            }
            other => panic!("expected symmetry-breaking subgoal, got {other:?}"),
        }
    }
}
