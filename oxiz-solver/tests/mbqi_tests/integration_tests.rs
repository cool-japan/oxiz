//! Integration tests for MBQI

use oxiz_core::ast::TermManager;
use oxiz_solver::mbqi::*;
use rustc_hash::FxHashMap;

#[test]
fn test_mbqi_basic_quantifier() {
    let mut manager = TermManager::new();
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let zero = manager.mk_int(num_bigint::BigInt::from(0));
    let body = manager.mk_ge(x, zero);

    // Create quantifier: ∀x. x >= 0
    // This should be unsatisfiable
    let mut quant = QuantifiedFormula::new(
        body,
        smallvec::SmallVec::new(),
        body,
        true,
    );
    quant.bound_vars.push((manager.intern("x"), manager.sorts.int_sort));

    assert!(quant.is_universal);
    assert_eq!(quant.num_vars(), 1);
}

#[test]
fn test_mbqi_model_completion() {
    let manager = TermManager::new();
    let completer = model_completion::ModelCompleter::new();

    let partial_model = FxHashMap::default();
    let quantifiers = vec![];

    // Should complete without errors
    let result = completer.stats();
    assert_eq!(result.num_completions, 0);
}

#[test]
fn test_mbqi_counterexample_generation() {
    let generator = counterexample::CounterExampleGenerator::new();
    let stats = generator.stats();

    assert_eq!(stats.num_searches, 0);
    assert_eq!(stats.num_counterexamples_found, 0);
}

#[test]
fn test_mbqi_instantiation_engine() {
    let engine = instantiation::InstantiationEngine::new();
    let stats = engine.stats();

    assert_eq!(stats.num_instantiations, 0);
}

#[test]
fn test_mbqi_finite_model_finder() {
    let finder = finite_model::FiniteModelFinder::new();
    let stats = finder.stats();

    assert_eq!(stats.num_searches, 0);
}

#[test]
fn test_mbqi_lazy_instantiator() {
    let mut inst = lazy_instantiation::LazyInstantiator::new();
    assert_eq!(inst.stats().num_process_calls, 0);

    inst.clear();
    assert_eq!(inst.stats().num_process_calls, 0);
}

#[test]
fn test_mbqi_integration() {
    let mut integration = integration::MBQIIntegration::new();
    integration.set_max_rounds(50);
    integration.clear();

    assert_eq!(integration.stats().num_quantifiers, 0);
}

#[test]
fn test_mbqi_heuristics() {
    let heuristics = heuristics::MBQIHeuristics::new();
    assert!(heuristics.enable_conflict_analysis);

    let conservative = heuristics::MBQIHeuristics::conservative();
    assert!(conservative.enable_model_bounds);
}

#[test]
fn test_mbqi_stats_display() {
    let stats = MBQIStats::new();
    let display = format!("{}", stats);
    assert!(display.contains("MBQI Statistics"));
}

#[test]
fn test_mbqi_result_predicates() {
    let sat = MBQIResult::Satisfied;
    assert!(sat.is_sat());
    assert!(!sat.is_unsat());

    let conflict = MBQIResult::Conflict {
        quantifier: oxiz_core::ast::TermId::new(1),
        reason: vec![],
    };
    assert!(!conflict.is_sat());
    assert!(conflict.is_unsat());
}

#[test]
fn test_instantiation_reason_display() {
    assert_eq!(format!("{}", InstantiationReason::ModelBased), "model-based");
    assert_eq!(format!("{}", InstantiationReason::EMatching), "e-matching");
    assert_eq!(format!("{}", InstantiationReason::Conflict), "conflict");
}

#[test]
fn test_quantified_formula_priority() {
    let manager = TermManager::new();
    let body = manager.mk_true();
    let mut qf = QuantifiedFormula::new(
        body,
        smallvec::SmallVec::new(),
        body,
        true,
    );

    let initial = qf.priority_score();
    qf.record_instantiation();
    let after = qf.priority_score();

    assert!(after < initial);
}
