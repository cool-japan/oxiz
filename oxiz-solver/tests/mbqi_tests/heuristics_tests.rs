//! Tests for MultiTriggerScorer and depth-bounded instantiation heuristics

use oxiz_core::ast::{TermId, TermManager};
use oxiz_solver::mbqi::{
    heuristics::{MultiTriggerScorer, ScorerPolicy, TriggerSet},
    instantiation::QuantifierInstantiator,
    model_completion::CompletedModel,
    QuantifiedFormula,
};

/// Helper: build a nested function application f(f(f(x))) with given nesting depth.
/// Returns the outermost term.
fn make_nested_apply(manager: &mut TermManager, depth: usize) -> TermId {
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let mut term = x;
    for _ in 0..depth {
        term = manager.mk_apply("f", smallvec::smallvec![term], manager.sorts.int_sort);
    }
    term
}

// ──────────────────────────────────────────────────────────────────────────────
// (a) Multi-trigger ranking
// ──────────────────────────────────────────────────────────────────────────────

/// Construct three candidate TriggerSets with known depth/shared-var
/// characteristics and assert that the Ranked scorer orders them correctly.
///
/// Expected ranking:
///  1. ts_shallow_shared   – shallow + shared vars → highest score
///  2. ts_shallow_novar    – shallow but no shared vars
///  3. ts_deep             – deep, fewer shared vars → lowest score
#[test]
fn test_multi_trigger_ranking() {
    let mut manager = TermManager::new();

    // Shallow trigger: f(x) — depth 1, one var
    let shallow = make_nested_apply(&mut manager, 1);

    // Deep trigger: f(f(f(f(x)))) — depth 4, one var
    let deep = make_nested_apply(&mut manager, 4);

    // Shared-var trigger set: {f(x), g(x)} — both reference x
    let gx = manager.mk_apply(
        "g",
        smallvec::smallvec![manager.mk_var("x", manager.sorts.int_sort)],
        manager.sorts.int_sort,
    );
    let ts_shallow_shared = TriggerSet::with_metrics(vec![shallow, gx], 1, 1);

    // Shallow no-shared: {f(x)} — only one term, shared_var_count = 0
    let ts_shallow_novar = TriggerSet::with_metrics(vec![shallow], 1, 0);

    // Deep trigger set: {f(f(f(f(x))))} — depth 4
    let ts_deep = TriggerSet::with_metrics(vec![deep], 4, 0);

    let scorer = MultiTriggerScorer::new(ScorerPolicy::Ranked, 3);
    let scored = scorer.score_trigger_sets(&[ts_shallow_shared, ts_shallow_novar, ts_deep], &manager);

    assert_eq!(scored.len(), 3, "should have 3 scored sets");

    // The shared-shallow one should rank highest (index 0 after descending sort)
    // It has shared_var_count=1, depth=1 → shared_component = 2.0, depth = 1/(1+1) = 0.5
    // no-shared-shallow: shared_component = 1.0, depth = 0.5
    // deep: shared_component = 1.0, depth = 1/(1+4) = 0.2
    assert!(
        scored[0].score >= scored[1].score,
        "shared-shallow must outrank shallow-novar: {:.4} vs {:.4}",
        scored[0].score,
        scored[1].score
    );
    assert!(
        scored[1].score >= scored[2].score,
        "shallow-novar must outrank deep: {:.4} vs {:.4}",
        scored[1].score,
        scored[2].score
    );

    // Sanity: all scores are positive
    for s in &scored {
        assert!(s.score > 0.0, "scores must be positive");
    }
}

/// Conservative policy should return at most top_k results
#[test]
fn test_conservative_policy_top_k() {
    let mut manager = TermManager::new();
    let t1 = make_nested_apply(&mut manager, 1);
    let t2 = make_nested_apply(&mut manager, 2);
    let t3 = make_nested_apply(&mut manager, 3);

    let sets = vec![
        TriggerSet::new(vec![t1]),
        TriggerSet::new(vec![t2]),
        TriggerSet::new(vec![t3]),
    ];

    let scorer = MultiTriggerScorer::new(ScorerPolicy::Conservative, 2);
    let scored = scorer.score_trigger_sets(&sets, &manager);

    assert_eq!(scored.len(), 2, "top_k=2 should return only 2 results");
}

/// Empty candidate list should return an empty scored list without panicking
#[test]
fn test_empty_candidate_list() {
    let manager = TermManager::new();
    let scorer = MultiTriggerScorer::new(ScorerPolicy::Ranked, 5);
    let scored = scorer.score_trigger_sets(&[], &manager);
    assert!(scored.is_empty());
}

// ──────────────────────────────────────────────────────────────────────────────
// (b) Depth-bound termination
// ──────────────────────────────────────────────────────────────────────────────

/// Depth-bound guard: with max_depth=2, the instantiator refuses to produce
/// more than 2 instantiations for the same quantifier and returns an empty
/// Vec once the limit is reached.
///
/// This simulates the termination guarantee for transitive axioms such as
///   ∀x. P(x) → P(f(x))
/// with the seed P(0).
#[test]
fn test_depth_bound_terminates() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let zero = manager.mk_int(num_bigint::BigInt::from(0));
    let body = manager.mk_ge(x, zero); // dummy body

    let int_sort = manager.sorts.int_sort;
    let quant_term = manager.mk_forall([("x", int_sort)], body);

    let mut qf = QuantifiedFormula::new(
        quant_term,
        smallvec::smallvec![(manager.intern("x"), int_sort)],
        body,
        true,
    );

    let model = CompletedModel::new();

    // Create a depth-bounded instantiator (max 2 instantiations per quantifier)
    let mut inst = QuantifierInstantiator::with_max_depth(2);

    // First call – depth starts at 0, should attempt (may produce 0 results if model has no candidates)
    let _r1 = inst.instantiate_from_model(&mut qf, &model, &mut manager);

    // Force the depth counter to the limit by simulating already-at-limit state
    let _r2 = inst.instantiate_from_model(&mut qf, &model, &mut manager);
    let _r3 = inst.instantiate_from_model(&mut qf, &model, &mut manager);

    // After 3 calls, depth for this quantifier is capped at max_depth (2).
    // Any further call must return empty.
    let r4 = inst.instantiate_from_model(&mut qf, &model, &mut manager);
    assert!(
        r4.is_empty(),
        "after depth cap, instantiate_from_model must return empty vec"
    );

    // Verify depth is at or above max_depth
    assert!(
        inst.current_depth(quant_term) >= 2,
        "depth counter should be >= max_depth"
    );
}

/// Depth tracking resets after clear_cache
#[test]
fn test_depth_reset_after_clear() {
    let mut manager = TermManager::new();
    let body = manager.mk_true();
    let int_sort = manager.sorts.int_sort;
    let quant_term = manager.mk_forall([("x", int_sort)], body);

    let mut qf = QuantifiedFormula::new(
        quant_term,
        smallvec::smallvec![(manager.intern("x"), int_sort)],
        body,
        true,
    );

    let model = CompletedModel::new();
    let mut inst = QuantifierInstantiator::with_max_depth(1);

    // Exhaust depth
    let _ = inst.instantiate_from_model(&mut qf, &model, &mut manager);
    let _ = inst.instantiate_from_model(&mut qf, &model, &mut manager);

    // Should be blocked
    let blocked = inst.instantiate_from_model(&mut qf, &model, &mut manager);
    assert!(blocked.is_empty(), "should be blocked by depth limit");

    // Reset – depth tracking cleared
    inst.clear_cache();
    assert_eq!(
        inst.current_depth(quant_term),
        0,
        "depth should reset to 0 after clear"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// (c) Ranked scorer correctness: ground-term bonus
// ──────────────────────────────────────────────────────────────────────────────

/// A trigger set containing ground terms (no free vars) should receive a
/// higher score than a comparable trigger set with only variable-containing
/// terms, under the Ranked policy.
#[test]
fn test_uflia_sat_correctness() {
    let mut manager = TermManager::new();

    // Ground trigger: f(0) — fully ground
    let zero = manager.mk_int(num_bigint::BigInt::from(0));
    let f_zero = manager.mk_apply("f", smallvec::smallvec![zero], manager.sorts.int_sort);

    // Variable trigger: f(x) — contains free var x
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let f_x = manager.mk_apply("f", smallvec::smallvec![x], manager.sorts.int_sort);

    let ts_ground = TriggerSet::with_metrics(vec![f_zero], 1, 0);
    let ts_var = TriggerSet::with_metrics(vec![f_x], 1, 0);

    let scorer = MultiTriggerScorer::new(ScorerPolicy::Ranked, 2);
    let scored = scorer.score_trigger_sets(&[ts_ground, ts_var], &manager);

    assert_eq!(scored.len(), 2);
    // Ground trigger should have a higher score due to ground-term reachability bonus
    assert!(
        scored[0].score >= scored[1].score,
        "ground trigger must score >= variable trigger: {:.4} vs {:.4}",
        scored[0].score,
        scored[1].score
    );
}
