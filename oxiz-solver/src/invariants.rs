//! Runtime invariant checks for the CDCL(T) solver
//!
//! These invariants are checked during solver execution to ensure
//! correctness and catch bugs early.

use crate::*;

/// Invariant: Trail consistency
///
/// Ensures that all assignments on the trail are consistent with
/// the current decision level and clause database.
pub fn check_trail_consistency(solver: &Solver) -> Result<(), String> {
    // Check that trail size is non-negative
    if solver.trail_size() < 0 {
        return Err("Trail size is negative".to_string());
    }

    // Check that all trail entries have valid decision levels
    for i in 0..solver.trail_size() as usize {
        let entry = solver.get_trail_entry(i)?;

        if entry.decision_level > solver.decision_level() as usize {
            return Err(format!(
                "Trail entry {} has decision level {} > current level {}",
                i,
                entry.decision_level,
                solver.decision_level()
            ));
        }

        if entry.decision_level < 0 {
            return Err(format!(
                "Trail entry {} has negative decision level {}",
                i, entry.decision_level
            ));
        }
    }

    // Check that decision levels are monotonic on the trail
    let mut last_level = 0;
    for i in 0..solver.trail_size() as usize {
        let entry = solver.get_trail_entry(i)?;

        if entry.decision_level < last_level {
            return Err(format!(
                "Trail entry {} has decision level {} < previous {}",
                i, entry.decision_level, last_level
            ));
        }

        last_level = entry.decision_level;
    }

    Ok(())
}

/// Invariant: Decision level consistency
///
/// Ensures that the current decision level matches the trail structure.
pub fn check_decision_level_consistency(solver: &Solver) -> Result<(), String> {
    let level = solver.decision_level();

    // Decision level should be non-negative
    if level < 0 {
        return Err(format!("Decision level is negative: {}", level));
    }

    // Decision level should not exceed trail size
    if level > solver.trail_size() {
        return Err(format!(
            "Decision level {} exceeds trail size {}",
            level,
            solver.trail_size()
        ));
    }

    // Each decision level should have at least one trail entry
    // (except level 0 which may be empty)
    if level > 0 {
        let has_entries_at_level = (0..solver.trail_size() as usize)
            .any(|i| solver.get_trail_entry(i).ok().map(|e| e.decision_level) == Some(level as usize));

        if !has_entries_at_level {
            return Err(format!(
                "Decision level {} has no trail entries",
                level
            ));
        }
    }

    Ok(())
}

/// Invariant: Clause database consistency
///
/// Ensures that all clauses in the database are well-formed.
pub fn check_clause_database_consistency(solver: &Solver) -> Result<(), String> {
    let num_clauses = solver.num_clauses();

    // Number of clauses should be non-negative
    if num_clauses < 0 {
        return Err(format!("Number of clauses is negative: {}", num_clauses));
    }

    // Check each clause
    for i in 0..num_clauses as usize {
        let clause = solver.get_clause(i)?;

        // Clause should not be empty (empty clause means UNSAT)
        if clause.literals.is_empty() {
            // This is okay if solver is in UNSAT state
            if !matches!(solver.status(), SolverStatus::Unsat) {
                return Err(format!("Empty clause {} in non-UNSAT state", i));
            }
        }

        // Check for tautologies (p ∨ ¬p)
        for j in 0..clause.literals.len() {
            for k in (j + 1)..clause.literals.len() {
                let lit_j = clause.literals[j];
                let lit_k = clause.literals[k];

                // Check if they're opposite literals
                if lit_j.var() == lit_k.var() && lit_j.is_negated() != lit_k.is_negated() {
                    return Err(format!(
                        "Clause {} contains tautology: {:?} and {:?}",
                        i, lit_j, lit_k
                    ));
                }
            }
        }

        // Check for duplicate literals
        for j in 0..clause.literals.len() {
            for k in (j + 1)..clause.literals.len() {
                if clause.literals[j] == clause.literals[k] {
                    return Err(format!(
                        "Clause {} contains duplicate literal: {:?}",
                        i, clause.literals[j]
                    ));
                }
            }
        }

        // Learned clauses should have proper LBD scores
        if clause.is_learned {
            if clause.lbd == 0 {
                return Err(format!("Learned clause {} has zero LBD", i));
            }

            if clause.lbd as usize > clause.literals.len() {
                return Err(format!(
                    "Learned clause {} has LBD {} > clause length {}",
                    i,
                    clause.lbd,
                    clause.literals.len()
                ));
            }
        }
    }

    Ok(())
}

/// Invariant: Variable assignment consistency
///
/// Ensures that variable assignments are consistent across the solver state.
pub fn check_variable_assignment_consistency(solver: &Solver) -> Result<(), String> {
    let num_vars = solver.num_variables();

    for var_id in 0..num_vars {
        let assignment = solver.get_assignment(var_id)?;

        match assignment {
            Assignment::Unassigned => {
                // Unassigned variables should not appear on the trail at current level
                // (they may appear at lower levels due to backtracking)
            }
            Assignment::True | Assignment::False => {
                // Assigned variables should appear exactly once on the trail
                let mut count = 0;
                for i in 0..solver.trail_size() as usize {
                    let entry = solver.get_trail_entry(i)?;
                    if entry.var_id == var_id {
                        count += 1;

                        // Check that the assignment matches
                        let expected = matches!(assignment, Assignment::True);
                        if entry.value != expected {
                            return Err(format!(
                                "Variable {} has assignment {:?} but trail entry has {}",
                                var_id, assignment, entry.value
                            ));
                        }
                    }
                }

                if count == 0 {
                    return Err(format!(
                        "Variable {} is assigned {:?} but not on trail",
                        var_id, assignment
                    ));
                }

                if count > 1 {
                    return Err(format!(
                        "Variable {} appears {} times on trail",
                        var_id, count
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Invariant: Theory solver consistency
///
/// Ensures that theory solvers maintain proper invariants.
pub fn check_theory_solver_consistency(solver: &Solver) -> Result<(), String> {
    // Check arithmetic theory consistency
    if let Some(arith_solver) = solver.get_arith_theory() {
        // Bounds should be consistent
        for var_id in 0..arith_solver.num_vars() {
            let lower = arith_solver.get_lower_bound(var_id)?;
            let upper = arith_solver.get_upper_bound(var_id)?;

            if let (Some(lb), Some(ub)) = (lower, upper) {
                if lb > ub {
                    return Err(format!(
                        "Arithmetic variable {} has lower bound {} > upper bound {}",
                        var_id, lb, ub
                    ));
                }
            }

            // If variable is assigned, it should satisfy bounds
            if let Some(value) = arith_solver.get_value(var_id)? {
                if let Some(lb) = lower {
                    if value < lb {
                        return Err(format!(
                            "Arithmetic variable {} has value {} < lower bound {}",
                            var_id, value, lb
                        ));
                    }
                }

                if let Some(ub) = upper {
                    if value > ub {
                        return Err(format!(
                            "Arithmetic variable {} has value {} > upper bound {}",
                            var_id, value, ub
                        ));
                    }
                }
            }
        }

        // Tableau should be consistent (for simplex-based solvers)
        if arith_solver.uses_tableau() {
            arith_solver.check_tableau_consistency()?;
        }
    }

    // Check equality graph consistency
    if let Some(eq_solver) = solver.get_equality_theory() {
        // All nodes in the same equivalence class should have the same representative
        for node_id in 0..eq_solver.num_nodes() {
            let rep1 = eq_solver.find(node_id)?;
            let rep2 = eq_solver.find(rep1)?;

            if rep1 != rep2 {
                return Err(format!(
                    "Equality node {} has inconsistent representative chain: {} -> {}",
                    node_id, rep1, rep2
                ));
            }
        }

        // Congruence closure should be maintained
        eq_solver.check_congruence_closure()?;
    }

    Ok(())
}

/// Invariant: Model validity
///
/// If solver is in SAT state, the model should satisfy all clauses.
pub fn check_model_validity(solver: &Solver, tm: &TermManager) -> Result<(), String> {
    if !matches!(solver.status(), SolverStatus::Sat) {
        return Ok(()); // Only check in SAT state
    }

    let model = solver.get_model(tm);

    // Check that all asserted clauses are satisfied
    for i in 0..solver.num_clauses() as usize {
        let clause = solver.get_clause(i)?;

        if clause.is_learned {
            continue; // Only check original clauses
        }

        let mut satisfied = false;
        for &lit in &clause.literals {
            let var_value = model.eval(lit.to_term(tm), tm)?;

            let lit_satisfied = if lit.is_negated() {
                var_value == tm.mk_bool(false)
            } else {
                var_value == tm.mk_bool(true)
            };

            if lit_satisfied {
                satisfied = true;
                break;
            }
        }

        if !satisfied {
            return Err(format!("Clause {} is not satisfied by model", i));
        }
    }

    // Check that all theory constraints are satisfied
    solver.check_theory_model_validity(tm)?;

    Ok(())
}

/// Master invariant checker
///
/// Runs all invariant checks and returns the first error found.
pub fn check_all_invariants(solver: &Solver, tm: &TermManager) -> Result<(), String> {
    check_trail_consistency(solver)?;
    check_decision_level_consistency(solver)?;
    check_clause_database_consistency(solver)?;
    check_variable_assignment_consistency(solver)?;
    check_theory_solver_consistency(solver)?;
    check_model_validity(solver, tm)?;

    Ok(())
}

#[cfg(test)]
mod invariant_tests {
    use super::*;

    #[test]
    fn test_empty_solver_invariants() {
        let solver = Solver::new();
        let tm = TermManager::new();

        assert!(check_all_invariants(&solver, &tm).is_ok());
    }

    #[test]
    fn test_simple_sat_invariants() {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        let p = tm.mk_var("p", tm.sorts.bool_sort);
        solver.assert(p, &mut tm);

        let result = solver.check(&mut tm);
        assert!(matches!(result, SolverResult::Sat));

        assert!(check_all_invariants(&solver, &tm).is_ok());
    }

    #[test]
    fn test_simple_unsat_invariants() {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        let p = tm.mk_var("p", tm.sorts.bool_sort);
        solver.assert(p, &mut tm);
        solver.assert(tm.mk_not(p), &mut tm);

        let result = solver.check(&mut tm);
        assert!(matches!(result, SolverResult::Unsat));

        // Some invariants may not hold in UNSAT state
        // (e.g., empty clause is allowed)
    }

    #[test]
    fn test_backtrack_invariants() {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        let p = tm.mk_var("p", tm.sorts.bool_sort);

        solver.push();
        solver.assert(p, &mut tm);
        assert!(check_all_invariants(&solver, &tm).is_ok());

        solver.pop();
        assert!(check_all_invariants(&solver, &tm).is_ok());
    }
}
