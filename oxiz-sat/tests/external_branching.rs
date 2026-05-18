//! Integration tests for the external BranchingHeuristic hook.

use std::sync::{Arc, Mutex};

use oxiz_sat::{
    BoxedBranchingHeuristic, BranchingHeuristic, Lit, Solver, SolverConfig, SolverResult, Var,
};

// ---------------------------------------------------------------------------
// Test heuristics
// ---------------------------------------------------------------------------

/// A heuristic that always picks the first candidate and counts calls.
struct CountingHeuristic {
    call_count: usize,
}

impl CountingHeuristic {
    fn new() -> Self {
        Self { call_count: 0 }
    }
}

impl BranchingHeuristic for CountingHeuristic {
    fn select(&mut self, candidates: &[Var], _scores: &[f64]) -> Option<Var> {
        if candidates.is_empty() {
            return None;
        }
        self.call_count += 1;
        Some(candidates[0])
    }
}

/// A heuristic that always returns None (defers to built-in).
struct DeferringHeuristic;

impl BranchingHeuristic for DeferringHeuristic {
    fn select(&mut self, _candidates: &[Var], _scores: &[f64]) -> Option<Var> {
        None
    }
}

/// A heuristic that counts on_conflict_var invocations.
/// Does not override select (uses default no-op for conflict calls
/// in structs that do override — but here we track via a shared counter).
struct ConflictCountingHeuristic {
    conflict_hook_count: Arc<Mutex<usize>>,
}

impl BranchingHeuristic for ConflictCountingHeuristic {
    fn select(&mut self, _candidates: &[Var], _scores: &[f64]) -> Option<Var> {
        None // always defer so built-in VSIDS drives the solve
    }

    fn on_conflict_var(&mut self, _var: Var, _level: u32) {
        *self
            .conflict_hook_count
            .lock()
            .unwrap_or_else(|e| e.into_inner()) += 1;
    }
}

/// A heuristic that picks the candidate with the highest VSIDS score.
struct HighestScoreHeuristic;

impl BranchingHeuristic for HighestScoreHeuristic {
    fn select(&mut self, candidates: &[Var], scores: &[f64]) -> Option<Var> {
        candidates
            .iter()
            .zip(scores.iter())
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(&v, _)| v)
    }
}

// ---------------------------------------------------------------------------
// Helper to build a tiny 2-variable SAT formula: (a OR b) AND (NOT a OR b)
// satisfiable with b=true, a=anything
// ---------------------------------------------------------------------------
fn build_simple_sat(solver: &mut Solver) -> (Var, Var) {
    let a = solver.new_var();
    let b = solver.new_var();
    solver.add_clause([Lit::pos(a), Lit::pos(b)]);
    solver.add_clause([Lit::neg(a), Lit::pos(b)]);
    (a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_external_branching_default_none() {
    let cfg = SolverConfig::default();
    assert!(
        cfg.external_branching.is_none(),
        "default config should have no external heuristic"
    );
}

#[test]
fn test_external_branching_field_accepts_arc_mutex() {
    // Verify that a BoxedBranchingHeuristic can be stored and retrieved from SolverConfig.
    let heuristic: BoxedBranchingHeuristic = Arc::new(Mutex::new(CountingHeuristic::new()));
    let config = SolverConfig {
        external_branching: Some(heuristic),
        ..SolverConfig::default()
    };
    assert!(config.external_branching.is_some());
}

#[test]
fn test_external_branching_called_during_solve() {
    // Verify the heuristic is actually invoked when the solver makes decisions.
    let heuristic = Arc::new(Mutex::new(CountingHeuristic::new()));
    let config = SolverConfig {
        external_branching: Some(heuristic.clone()),
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);
    build_simple_sat(&mut solver);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat, "formula should be satisfiable");

    // The heuristic must have been called at least once (one decision needed).
    let count = heuristic
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .call_count;
    assert!(
        count > 0,
        "external heuristic should have been called at least once; got call_count={count}"
    );
}

#[test]
fn test_external_branching_deferring_still_solves() {
    // When the heuristic always defers, the solver falls through to built-in VSIDS.
    let config = SolverConfig {
        external_branching: Some(Arc::new(Mutex::new(DeferringHeuristic))),
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);
    build_simple_sat(&mut solver);

    let result = solver.solve();
    assert_eq!(
        result,
        SolverResult::Sat,
        "formula should still be solvable with deferring heuristic"
    );
}

#[test]
fn test_external_branching_unsat_formula() {
    // External heuristic should not prevent UNSAT detection.
    let config = SolverConfig {
        external_branching: Some(Arc::new(Mutex::new(CountingHeuristic::new()))),
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);
    let a = solver.new_var();
    solver.add_clause([Lit::pos(a)]);
    solver.add_clause([Lit::neg(a)]);

    let result = solver.solve();
    assert_eq!(
        result,
        SolverResult::Unsat,
        "contradictory formula must be UNSAT"
    );
}

#[test]
fn test_external_branching_highest_score_heuristic_solves() {
    // A non-trivial heuristic (pick highest-score candidate) should still yield SAT.
    let config = SolverConfig {
        external_branching: Some(Arc::new(Mutex::new(HighestScoreHeuristic))),
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);
    build_simple_sat(&mut solver);

    let result = solver.solve();
    assert_eq!(
        result,
        SolverResult::Sat,
        "formula should be SAT with HighestScoreHeuristic"
    );
}

#[test]
fn test_external_branching_scores_parallel_to_candidates() {
    // Verify that candidates and scores vectors are the same length when received.
    struct LengthCheckHeuristic {
        lengths_matched: bool,
    }

    impl BranchingHeuristic for LengthCheckHeuristic {
        fn select(&mut self, candidates: &[Var], scores: &[f64]) -> Option<Var> {
            if candidates.len() != scores.len() {
                self.lengths_matched = false;
            }
            None // defer; we only check structure
        }
    }

    let heuristic = Arc::new(Mutex::new(LengthCheckHeuristic {
        lengths_matched: true,
    }));
    let config = SolverConfig {
        external_branching: Some(heuristic.clone()),
        ..SolverConfig::default()
    };

    let mut solver = Solver::with_config(config);
    // Use a slightly larger formula so the heuristic is called with non-trivial candidates.
    let vars: Vec<Var> = (0..5).map(|_| solver.new_var()).collect();
    // Satisfiable formula: big OR of all variables
    solver.add_clause(vars.iter().map(|&v| Lit::pos(v)));

    let _ = solver.solve();

    assert!(
        heuristic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .lengths_matched,
        "candidates and scores must always be the same length"
    );
}

#[test]
fn test_branching_heuristic_trait_object_is_send_sync() {
    // Compile-time check: Arc<Mutex<dyn BranchingHeuristic>> must be Send + Sync.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<Mutex<CountingHeuristic>>>();
    assert_send_sync::<Arc<Mutex<DeferringHeuristic>>>();
    // Also verify the type alias resolves to the same bounds.
    assert_send_sync::<BoxedBranchingHeuristic>();
}

#[test]
fn test_external_branching_config_clone() {
    // SolverConfig derives Clone; make sure the new field doesn't break that.
    let heuristic: BoxedBranchingHeuristic = Arc::new(Mutex::new(DeferringHeuristic));
    let config = SolverConfig {
        external_branching: Some(heuristic),
        ..SolverConfig::default()
    };

    // Clone must succeed and the clone must point to the same Arc.
    let config2 = config.clone();
    assert!(config2.external_branching.is_some());
}

#[test]
fn test_external_branching_receives_conflict_calls() {
    // Wire a ConflictCountingHeuristic as external_branching.
    // Solve PHP(3,2) — 3 pigeons, 2 holes — which is UNSAT and requires real CDCL
    // search with genuine conflict analysis (not resolvable by unit propagation alone).
    // Assert that on_conflict_var was invoked at least once.
    let conflict_hook_count = Arc::new(Mutex::new(0usize));

    let heuristic = Arc::new(Mutex::new(ConflictCountingHeuristic {
        conflict_hook_count: Arc::clone(&conflict_hook_count),
    }));

    let config = SolverConfig {
        external_branching: Some(heuristic),
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);

    // PHP(3,2): variables p_ij = pigeon i in hole j
    // p11=0, p12=1, p21=2, p22=3, p31=4, p32=5
    for _ in 0..6 {
        solver.new_var();
    }

    // Each pigeon must be in at least one hole
    solver.add_clause_dimacs(&[1, 2]); // p1 in h1 or h2
    solver.add_clause_dimacs(&[3, 4]); // p2 in h1 or h2
    solver.add_clause_dimacs(&[5, 6]); // p3 in h1 or h2

    // At most one pigeon per hole (pairwise exclusion)
    solver.add_clause_dimacs(&[-1, -3]); // not (p1h1 and p2h1)
    solver.add_clause_dimacs(&[-1, -5]); // not (p1h1 and p3h1)
    solver.add_clause_dimacs(&[-3, -5]); // not (p2h1 and p3h1)
    solver.add_clause_dimacs(&[-2, -4]); // not (p1h2 and p2h2)
    solver.add_clause_dimacs(&[-2, -6]); // not (p1h2 and p3h2)
    solver.add_clause_dimacs(&[-4, -6]); // not (p2h2 and p3h2)

    let result = solver.solve();
    assert_eq!(result, SolverResult::Unsat, "PHP(3,2) must be UNSAT");

    let hook_calls = *conflict_hook_count
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    assert!(
        hook_calls > 0,
        "on_conflict_var must have been called at least once during conflict analysis; got {hook_calls}"
    );
}
