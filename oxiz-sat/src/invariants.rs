//! Runtime invariant checks for CDCL SAT solver
//!
//! Ensures correctness of the SAT solver implementation

use crate::*;

/// Check clause database integrity
pub fn check_clause_database(solver: &CDCLSolver) -> Result<(), String> {
    let num_clauses = solver.num_clauses();

    for i in 0..num_clauses {
        let clause = solver.get_clause(i)?;

        // No duplicate literals
        for j in 0..clause.len() {
            for k in (j+1)..clause.len() {
                if clause[j] == clause[k] {
                    return Err(format!("Duplicate literal in clause {}", i));
                }
            }
        }

        // No tautologies (p ∨ ¬p)
        for j in 0..clause.len() {
            for k in (j+1)..clause.len() {
                if clause[j].var() == clause[k].var() &&
                   clause[j].is_pos() != clause[k].is_pos() {
                    return Err(format!("Tautology in clause {}", i));
                }
            }
        }

        // All literals reference valid variables
        for &lit in clause.iter() {
            if lit.var() >= solver.num_vars() {
                return Err(format!("Invalid variable in clause {}", i));
            }
        }
    }

    Ok(())
}

/// Check assignment consistency
pub fn check_assignment_consistency(solver: &CDCLSolver) -> Result<(), String> {
    for var in 0..solver.num_vars() {
        let assignment = solver.get_assignment(var)?;

        // If assigned, must be on trail
        if assignment != LBool::Undef {
            let trail = solver.get_trail();
            let on_trail = trail.iter().any(|&lit| lit.var() == var);

            if !on_trail {
                return Err(format!("Variable {} assigned but not on trail", var));
            }
        }
    }

    Ok(())
}

/// Check watched literals scheme
pub fn check_watched_literals(solver: &CDCLSolver) -> Result<(), String> {
    for i in 0..solver.num_clauses() {
        let clause = solver.get_clause(i)?;

        if clause.len() < 2 {
            continue; // Unit or empty clauses don't need watches
        }

        let watches = solver.get_watches(i)?;

        // Should have exactly 2 watched literals
        if watches.len() != 2 {
            return Err(format!("Clause {} has {} watches, expected 2", i, watches.len()));
        }

        // Watched literals must be in the clause
        for &watch_idx in &watches {
            if watch_idx >= clause.len() {
                return Err(format!("Invalid watch index in clause {}", i));
            }
        }

        // If clause is not satisfied, at least one watch should not be false
        let all_false = clause.iter().all(|&lit| solver.get_assignment(lit.var()).ok() == Some(lit.neg().sign()));

        if !all_false {
            let watch0_false = solver.get_assignment(clause[watches[0]].var()).ok() == Some(clause[watches[0]].neg().sign());
            let watch1_false = solver.get_assignment(clause[watches[1]].var()).ok() == Some(clause[watches[1]].neg().sign());

            if watch0_false && watch1_false {
                // Both watches false but clause not all false - invariant violation
                return Err(format!("Both watches false in non-falsified clause {}", i));
            }
        }
    }

    Ok(())
}

/// Check implication graph acyclicity
pub fn check_implication_graph_acyclic(solver: &CDCLSolver) -> Result<(), String> {
    let trail = solver.get_trail();

    // Use DFS to detect cycles
    let mut visited = vec![false; solver.num_vars()];
    let mut rec_stack = vec![false; solver.num_vars()];

    for &lit in trail.iter() {
        let var = lit.var();

        if !visited[var] {
            if has_cycle_dfs(solver, var, &mut visited, &mut rec_stack)? {
                return Err("Cycle detected in implication graph".to_string());
            }
        }
    }

    Ok(())
}

fn has_cycle_dfs(
    solver: &CDCLSolver,
    var: usize,
    visited: &mut [bool],
    rec_stack: &mut [bool],
) -> Result<bool, String> {
    visited[var] = true;
    rec_stack[var] = true;

    // Get reason clause
    if let Some(reason) = solver.get_reason(var)? {
        let clause = solver.get_clause(reason)?;

        for &lit in clause.iter() {
            let next_var = lit.var();

            if next_var == var {
                continue; // Skip self
            }

            if !visited[next_var] {
                if has_cycle_dfs(solver, next_var, visited, rec_stack)? {
                    return Ok(true);
                }
            } else if rec_stack[next_var] {
                return Ok(true); // Back edge found
            }
        }
    }

    rec_stack[var] = false;
    Ok(false)
}

/// Check decision level consistency
pub fn check_decision_levels(solver: &CDCLSolver) -> Result<(), String> {
    let trail = solver.get_trail();
    let mut last_level = 0;

    for (i, &lit) in trail.iter().enumerate() {
        let level = solver.get_level(lit.var())?;

        // Decision levels should be monotonic on trail
        if level < last_level {
            return Err(format!(
                "Trail position {} has level {} < previous {}",
                i, level, last_level
            ));
        }

        last_level = level;

        // Level should not exceed current decision level
        if level > solver.decision_level() {
            return Err(format!(
                "Variable {} has level {} > current {}",
                lit.var(),
                level,
                solver.decision_level()
            ));
        }
    }

    Ok(())
}

/// Check learned clause quality (LBD)
pub fn check_learned_clause_quality(solver: &CDCLSolver) -> Result<(), String> {
    for i in 0..solver.num_clauses() {
        let clause = solver.get_clause(i)?;

        if !solver.is_learned(i)? {
            continue;
        }

        let lbd = solver.get_lbd(i)?;

        // LBD should be positive
        if lbd == 0 {
            return Err(format!("Learned clause {} has LBD 0", i));
        }

        // LBD should not exceed clause length
        if lbd > clause.len() {
            return Err(format!(
                "Learned clause {} has LBD {} > length {}",
                i,
                lbd,
                clause.len()
            ));
        }

        // Compute actual LBD and check it matches
        let mut levels = std::collections::HashSet::new();
        for &lit in clause.iter() {
            let level = solver.get_level(lit.var())?;
            levels.insert(level);
        }

        if levels.len() != lbd {
            return Err(format!(
                "Learned clause {} has stored LBD {} but computed {}",
                i,
                lbd,
                levels.len()
            ));
        }
    }

    Ok(())
}

/// Check conflict analysis correctness
pub fn check_conflict_analysis(solver: &CDCLSolver) -> Result<(), String> {
    if let Some(conflict_clause) = solver.get_last_conflict()? {
        let clause = solver.get_clause(conflict_clause)?;

        // Conflict clause should be falsified
        for &lit in clause.iter() {
            let assignment = solver.get_assignment(lit.var())?;

            if assignment != lit.neg().sign() {
                return Err("Conflict clause not falsified".to_string());
            }
        }

        // All literals should be assigned
        for &lit in clause.iter() {
            if solver.get_assignment(lit.var())? == LBool::Undef {
                return Err("Conflict clause has unassigned literal".to_string());
            }
        }
    }

    Ok(())
}

/// Check unit propagation completeness
pub fn check_unit_propagation_complete(solver: &CDCLSolver) -> Result<(), String> {
    // Check that there are no unprocessed unit clauses
    for i in 0..solver.num_clauses() {
        let clause = solver.get_clause(i)?;

        if clause.is_empty() {
            continue;
        }

        // Count unassigned and satisfied literals
        let mut unassigned = 0;
        let mut satisfied = false;

        for &lit in clause.iter() {
            let assignment = solver.get_assignment(lit.var())?;

            if assignment == LBool::Undef {
                unassigned += 1;
            } else if assignment == lit.sign() {
                satisfied = true;
                break;
            }
        }

        // If clause is unit (exactly one unassigned, rest false), it should be propagated
        if !satisfied && unassigned == 1 && solver.propagation_queue_empty()? {
            return Err(format!("Unit clause {} not propagated", i));
        }
    }

    Ok(())
}

/// Check restart consistency
pub fn check_restart_consistency(solver: &CDCLSolver) -> Result<(), String> {
    // After restart, decision level should be 0
    if solver.last_restart()? {
        if solver.decision_level() != 0 {
            return Err("Decision level not 0 after restart".to_string());
        }

        // Trail should only contain propagated literals
        let trail = solver.get_trail();
        for &lit in trail.iter() {
            let level = solver.get_level(lit.var())?;

            if level != 0 {
                return Err("Non-level-0 literal on trail after restart".to_string());
            }
        }
    }

    Ok(())
}

/// Check clause deletion safety
pub fn check_clause_deletion_safety(solver: &CDCLSolver) -> Result<(), String> {
    // Deleted clauses should not be reason clauses
    for deleted_idx in solver.get_deleted_clauses()? {
        for var in 0..solver.num_vars() {
            if let Some(reason) = solver.get_reason(var)? {
                if reason == deleted_idx {
                    return Err(format!(
                        "Deleted clause {} is reason for variable {}",
                        deleted_idx, var
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Master invariant check for SAT solver
pub fn check_all_sat_invariants(solver: &CDCLSolver) -> Result<(), String> {
    check_clause_database(solver)?;
    check_assignment_consistency(solver)?;
    check_watched_literals(solver)?;
    check_implication_graph_acyclic(solver)?;
    check_decision_levels(solver)?;
    check_learned_clause_quality(solver)?;
    check_conflict_analysis(solver)?;
    check_unit_propagation_complete(solver)?;
    check_restart_consistency(solver)?;
    check_clause_deletion_safety(solver)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_solver_invariants() {
        let solver = CDCLSolver::new();
        assert!(check_all_sat_invariants(&solver).is_ok());
    }

    #[test]
    fn test_simple_sat_invariants() {
        let mut solver = CDCLSolver::new();
        solver.add_variable();

        solver.add_clause(vec![Lit::new(0, true)]);

        let result = solver.solve();
        assert_eq!(result, SatResult::Sat);

        assert!(check_all_sat_invariants(&solver).is_ok());
    }

    #[test]
    fn test_simple_unsat_invariants() {
        let mut solver = CDCLSolver::new();
        solver.add_variable();

        solver.add_clause(vec![Lit::new(0, true)]);
        solver.add_clause(vec![Lit::new(0, false)]);

        let result = solver.solve();
        assert_eq!(result, SatResult::Unsat);
    }
}
