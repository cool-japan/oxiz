//! Integration tests: MLBranchingHeuristic wired into oxiz-sat Solver.

use std::sync::{Arc, Mutex};

use oxiz_ml::branching::{MLBranchingHeuristic, MLEnhancedVSIDS};
use oxiz_sat::{BranchingHeuristic, Lit, Solver, SolverConfig, SolverResult, Var};

#[test]
fn test_adapter_constructs_with_default_vsids() {
    let _adapter = MLBranchingHeuristic::new(MLEnhancedVSIDS::default_config());
}

#[test]
fn test_adapter_returns_chosen_var() {
    let mut vsids = MLEnhancedVSIDS::default_config();
    vsids.bump_activity(1);
    vsids.bump_activity(1);
    let mut adapter = MLBranchingHeuristic::new(vsids);
    let candidates = vec![Var::new(0), Var::new(1), Var::new(2)];
    let scores = vec![0.0_f64; 3];
    let chosen = adapter.select(&candidates, &scores);
    assert!(chosen.is_some());
    assert_eq!(chosen.unwrap(), Var::new(1));
}

#[test]
fn test_adapter_empty_candidates_returns_none() {
    let mut adapter = MLBranchingHeuristic::new(MLEnhancedVSIDS::default_config());
    assert!(adapter.select(&[], &[]).is_none());
}

#[test]
fn test_adapter_min_confidence_gate() {
    // Default decisions from MLEnhancedVSIDS have confidence 0.5; gate at 0.99 → None.
    let vsids = MLEnhancedVSIDS::default_config();
    let mut adapter = MLBranchingHeuristic::new(vsids).with_min_confidence(0.99);
    let candidates = vec![Var::new(0), Var::new(1)];
    let scores = vec![0.0_f64; 2];
    assert!(adapter.select(&candidates, &scores).is_none());
}

/// A counting heuristic wrapper for tests 5 and 6.
struct CountingHeuristic {
    inner: Option<Box<dyn BranchingHeuristic>>,
    call_count: Arc<Mutex<usize>>,
    always_none: bool,
}

impl BranchingHeuristic for CountingHeuristic {
    fn select(&mut self, candidates: &[Var], scores: &[f64]) -> Option<Var> {
        *self.call_count.lock().unwrap() += 1;
        if self.always_none {
            return None;
        }
        self.inner.as_mut().and_then(|h| h.select(candidates, scores))
    }
}

#[test]
fn test_solver_routes_through_adapter() {
    let call_count = Arc::new(Mutex::new(0usize));

    let adapter = MLBranchingHeuristic::new(MLEnhancedVSIDS::default_config());
    let counting = CountingHeuristic {
        inner: Some(Box::new(adapter)),
        call_count: Arc::clone(&call_count),
        always_none: false,
    };

    let config = SolverConfig {
        external_branching: Some(Arc::new(Mutex::new(counting))),
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);

    // (a ∨ b) ∧ (¬a ∨ b): b must be true
    let a = solver.new_var();
    let b = solver.new_var();
    solver.add_clause([Lit::pos(a), Lit::pos(b)]);
    solver.add_clause([Lit::neg(a), Lit::pos(b)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
    assert!(*call_count.lock().unwrap() > 0);
}

#[test]
fn test_solver_falls_back_when_adapter_returns_none() {
    let call_count = Arc::new(Mutex::new(0usize));

    let counting = CountingHeuristic {
        inner: None,
        call_count: Arc::clone(&call_count),
        always_none: true,
    };

    let config = SolverConfig {
        external_branching: Some(Arc::new(Mutex::new(counting))),
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);

    // Same CNF: (a ∨ b) ∧ (¬a ∨ b)
    let a = solver.new_var();
    let b = solver.new_var();
    solver.add_clause([Lit::pos(a), Lit::pos(b)]);
    solver.add_clause([Lit::neg(a), Lit::pos(b)]);

    // The adapter always returns None, so the built-in VSIDS takes over.
    let result = solver.solve();
    assert!(matches!(result, SolverResult::Sat | SolverResult::Unsat));
    assert!(*call_count.lock().unwrap() > 0);
}
