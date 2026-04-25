use oxiz_solver::mbqi::{
    MultiTriggerScorer, QuantifiedFormula, QuantifierInstantiator, ScorerPolicy, TriggerSet,
};
use oxiz_solver::{Context, SolverResult};

fn make_nested_apply(
    manager: &mut oxiz_core::ast::TermManager,
    var_name: &str,
    func_name: &str,
    depth: usize,
) -> oxiz_core::ast::TermId {
    let mut term = manager.mk_var(var_name, manager.sorts.int_sort);
    for _ in 0..depth {
        term = manager.mk_apply(func_name, [term], manager.sorts.int_sort);
    }
    term
}

#[test]
fn test_multi_trigger_ranking() {
    let mut manager = oxiz_core::ast::TermManager::new();

    let shallow_shared_left = make_nested_apply(&mut manager, "x", "f", 1);
    let x_var = manager.mk_var("x", manager.sorts.int_sort);
    let shallow_shared_right = manager.mk_apply("g", [x_var], manager.sorts.int_sort);
    let shallow_non_ground = make_nested_apply(&mut manager, "y", "h", 1);
    let zero = manager.mk_int(0);
    let inner_ground = manager.mk_apply("inner", [zero], manager.sorts.int_sort);
    let mid_ground = manager.mk_apply("mid", [inner_ground], manager.sorts.int_sort);
    let deep_ground = manager.mk_apply("d", [mid_ground], manager.sorts.int_sort);

    let candidates = vec![
        TriggerSet::new(vec![deep_ground]),
        TriggerSet::new(vec![shallow_non_ground]),
        TriggerSet::new(vec![shallow_shared_left, shallow_shared_right]),
    ];

    let scorer = MultiTriggerScorer::new(ScorerPolicy::Ranked, 3);
    let scored = scorer.score_trigger_sets(&candidates, &manager);

    assert_eq!(scored.len(), 3);
    assert_eq!(
        scored[0].triggers.terms,
        vec![shallow_shared_left, shallow_shared_right]
    );
    assert_eq!(scored[1].triggers.terms, vec![shallow_non_ground]);
    assert_eq!(scored[2].triggers.terms, vec![deep_ground]);
    assert!(scored[0].score > scored[1].score);
    assert!(scored[1].score > scored[2].score);
}

#[test]
fn test_depth_bound_terminates() {
    let mut manager = oxiz_core::ast::TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let zero = manager.mk_int(0);
    let body = manager.mk_ge(x, zero);
    let quant_term = manager.mk_forall([("x", int_sort)], body);
    let quantifier = QuantifiedFormula::new(
        quant_term,
        smallvec::smallvec![(manager.intern_str("x"), int_sort)],
        body,
        true,
    );

    let mut instantiator = QuantifierInstantiator::with_max_depth(2);
    let model = oxiz_solver::mbqi::model_completion::CompletedModel::new();

    instantiator.increment_depth(quant_term);
    instantiator.increment_depth(quant_term);

    let blocked =
        instantiator.instantiate_from_conflict(&quantifier, &[zero], &model, &mut manager);

    assert!(
        blocked.is_empty(),
        "instantiation must stop once max_depth=2 is reached"
    );
    assert_eq!(instantiator.current_depth(quant_term), 2);
}

#[test]
fn test_uflia_sat_correctness() {
    let mut ctx = Context::new();
    ctx.set_logic("UFLIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let a = ctx.declare_const("a", int_sort);
    ctx.declare_fun("f", vec![int_sort], int_sort);

    let zero = ctx.terms.mk_int(0);
    let a_ge_zero = ctx.terms.mk_ge(a, zero);
    ctx.assert(a_ge_zero);

    let x = ctx.terms.mk_var("x", int_sort);
    let f_x = ctx.terms.mk_apply("f", [x], int_sort);
    let body = ctx.terms.mk_eq(f_x, f_x);
    let quantifier = ctx.terms.mk_forall([("x", int_sort)], body);
    ctx.assert(quantifier);

    let result = ctx.check_sat();
    assert_eq!(
        result,
        SolverResult::Sat,
        "simple UFLIA formula should be SAT"
    );
}
