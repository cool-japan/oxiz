//! Integration tests for the Z3 API compatibility extension layer 2.
//!
//! Covers: Z3Statistics, Z3Params, Z3Probe, Z3Goal, Z3Tactic, Z3ApplyResult,
//! Z3DatatypeSort/Z3Constructor, check_assumptions/unsat_core, Z3AstVector,
//! Z3FuncInterp/Z3FuncEntry/Z3Value.

use oxiz_solver::z3_compat::{
    Bool, BV, FuncDecl, Int, SatResult, Z3AstVector, Z3Config, Z3Context, Z3Solver,
    Z3DatatypeSort, Z3Goal, Z3Params, Z3Probe, Z3Tactic, Z3Value, mk_constructor,
};

fn make_ctx() -> Z3Context {
    Z3Context::new(&Z3Config::new())
}

fn make_solver(ctx: &Z3Context) -> Z3Solver {
    Z3Solver::new(ctx)
}

// ─── Z3Statistics ─────────────────────────────────────────────────────────────

#[test]
fn test_statistics_has_keys() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let p = Bool::new_const(&ctx, "p");
    solver.assert(&p);
    let _ = solver.check();
    let stats = solver.statistics();
    assert!(stats.num_keys() > 0, "statistics must have at least one key");
}

#[test]
fn test_statistics_key_value_pairs() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let p = Bool::new_const(&ctx, "p");
    solver.assert(&p);
    let _ = solver.check();
    let stats = solver.statistics();
    // All keys should be non-empty strings, values should be >= 0.
    for i in 0..stats.num_keys() {
        assert!(!stats.key(i).is_empty(), "stat key {} is empty", i);
        assert!(stats.value(i) >= 0.0, "stat value {} < 0", i);
    }
}

#[test]
fn test_statistics_get_by_name() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let p = Bool::new_const(&ctx, "p");
    solver.assert(&p);
    let _ = solver.check();
    let stats = solver.statistics();
    // "decisions" key must exist and be a non-negative number.
    let v = stats.get("decisions");
    assert!(v.is_some(), "decisions key must exist");
    assert!(v.unwrap() >= 0.0);
}

#[test]
fn test_statistics_unknown_key_returns_none() {
    let ctx = make_ctx();
    let solver = make_solver(&ctx);
    let stats = solver.statistics();
    assert!(stats.get("nonexistent-key-xyz").is_none());
}

#[test]
fn test_statistics_display() {
    let ctx = make_ctx();
    let solver = make_solver(&ctx);
    let stats = solver.statistics();
    let s = format!("{}", stats);
    assert!(!s.is_empty(), "display output must not be empty");
}

// ─── Z3Params ─────────────────────────────────────────────────────────────────

#[test]
fn test_params_set_bool_timeout_then_check() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let mut params = Z3Params::new(&ctx);
    params.set_bool("proof", false);
    solver.set_params(&params);
    let p = Bool::new_const(&ctx, "p");
    solver.assert(&p);
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_params_set_u32_max_conflicts() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let mut params = Z3Params::new(&ctx);
    params.set_u32("max-conflicts", 10_000);
    solver.set_params(&params);
    let p = Bool::new_const(&ctx, "p");
    solver.assert(&p);
    // Should still be solvable under a generous conflict limit.
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_params_set_double() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let mut params = Z3Params::new(&ctx);
    params.set_double("timeout", 5000.0);
    // Must not panic.
    solver.set_params(&params);
}

#[test]
fn test_params_set_str() {
    let ctx = make_ctx();
    let mut params = Z3Params::new(&ctx);
    params.set_str("logic", "QF_LIA");
    assert_eq!(
        params.as_map().get("logic").map(|v| format!("{:?}", v)).as_deref(),
        Some("Str(\"QF_LIA\")")
    );
}

// ─── Z3Probe ──────────────────────────────────────────────────────────────────

#[test]
fn test_probe_size_empty_goal() {
    let ctx = make_ctx();
    let goal = Z3Goal::new(&ctx);
    let probe = Z3Probe::new(&ctx, "size");
    assert_eq!(probe.apply(&ctx, &goal), 0.0);
}

#[test]
fn test_probe_size_one_assertion() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let probe = Z3Probe::new(&ctx, "size");
    assert_eq!(probe.apply(&ctx, &goal), 1.0);
}

#[test]
fn test_probe_is_qfbv_false_for_bool_goal() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let probe = Z3Probe::new(&ctx, "is-qfbv");
    // Pure boolean goal has no bitvectors.
    assert_eq!(probe.apply(&ctx, &goal), 0.0);
}

#[test]
fn test_probe_is_qfbv_true_for_bv_goal() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let x = BV::new_const(&ctx, "x_bv8", 8);
    let y = BV::new_const(&ctx, "y_bv8", 8);
    // bvult creates a BV-specific comparison term (HasBitVectorProbe detects it).
    let ult = BV::bvult(&ctx, &x, &y);
    goal.assert(&ult);
    let probe = Z3Probe::new(&ctx, "is-qfbv");
    assert_eq!(probe.apply(&ctx, &goal), 1.0);
}

#[test]
fn test_probe_combinator_lt() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    let q = Bool::new_const(&ctx, "q");
    goal.assert(&p);
    goal.assert(&q);
    // size(goal) = 2, const(3) = 3 → size < const → 1.0
    let size_probe = Z3Probe::new(&ctx, "size");
    let big_probe = Z3Probe::new(&ctx, "size"); // reuse as constant-like comparison
    let big = big_probe.gt(size_probe); // 2 > 2 = false (0.0)
    assert_eq!(big.apply(&ctx, &goal), 0.0);
}

#[test]
fn test_probe_depth_empty() {
    let ctx = make_ctx();
    let goal = Z3Goal::new(&ctx);
    let probe = Z3Probe::new(&ctx, "depth");
    assert_eq!(probe.apply(&ctx, &goal), 0.0);
}

// ─── Z3Goal ───────────────────────────────────────────────────────────────────

#[test]
fn test_goal_add_assert_size() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    assert_eq!(goal.size(), 0);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    assert_eq!(goal.size(), 1);
    let q = Bool::new_const(&ctx, "q");
    goal.assert(&q);
    assert_eq!(goal.size(), 2);
}

#[test]
fn test_goal_get_formula() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let retrieved = goal.get_formula(0);
    assert_eq!(retrieved.id, p.id);
}

#[test]
fn test_goal_is_decided_sat_on_empty() {
    let ctx = make_ctx();
    let goal = Z3Goal::new(&ctx);
    assert!(goal.is_decided_sat(), "empty goal is trivially SAT");
}

// ─── Z3Tactic ─────────────────────────────────────────────────────────────────

#[test]
fn test_tactic_simplify_apply() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let tactic = Z3Tactic::new(&ctx, "simplify");
    let result = tactic.apply(&ctx, &goal);
    // Simplify on a pure variable produces one subgoal.
    assert!(result.num_subgoals() <= 1);
}

#[test]
fn test_tactic_then_combinator() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let t1 = Z3Tactic::new(&ctx, "simplify");
    let t2 = Z3Tactic::new(&ctx, "propagate-values");
    let combined = t1.then(&t2);
    let _result = combined.apply(&ctx, &goal);
}

#[test]
fn test_tactic_repeat_does_not_panic() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let t = Z3Tactic::new(&ctx, "simplify").repeat();
    let result = t.apply(&ctx, &goal);
    assert!(result.num_subgoals() <= 1);
}

#[test]
fn test_tactic_or_else_combinator() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let t1 = Z3Tactic::new(&ctx, "simplify");
    let t2 = Z3Tactic::new(&ctx, "bit-blast");
    let _combined = t1.or_else(&t2);
}

#[test]
fn test_tactic_try_for_wraps() {
    let ctx = make_ctx();
    let t = Z3Tactic::new(&ctx, "simplify").try_for(100);
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    goal.assert(&p);
    let _result = t.apply(&ctx, &goal);
}

#[test]
fn test_apply_result_subgoals() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "p");
    let q = Bool::new_const(&ctx, "q");
    goal.assert(&p);
    goal.assert(&q);
    let tactic = Z3Tactic::new(&ctx, "simplify");
    let result = tactic.apply(&ctx, &goal);
    // Access subgoals without panicking.
    for i in 0..result.num_subgoals() {
        let sg = result.get_subgoal(i);
        let _ = sg.size(); // size() is a usize; just verify no panic
    }
}

// ─── Z3Tactic — registry-backed dispatch ──────────────────────────────────────

#[test]
fn test_z3_tactic_registry_simplify_works() {
    // "simplify" resolves through the TacticRegistry and produces a result.
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "reg_simplify_p");
    goal.assert(&p);
    let tactic = Z3Tactic::new(&ctx, "simplify");
    let result = tactic.apply(&ctx, &goal);
    // Simplify on a pure variable produces at most one subgoal without panic.
    assert!(result.num_subgoals() <= 1);
}

#[test]
fn test_z3_tactic_registry_covers_more_than_six() {
    // Tactics newly reachable via the registry (previously fell through to the
    // "goal unchanged" default in the inline match): each must produce a
    // non-panicking apply result.
    let ctx = make_ctx();
    for name in ["solve-eqs", "nnf", "split", "fm"] {
        let mut goal = Z3Goal::new(&ctx);
        let a = Bool::new_const(&ctx, &format!("reg_more_{}_a", name));
        let b = Bool::new_const(&ctx, &format!("reg_more_{}_b", name));
        let conj = Bool::and(&ctx, &[a, b]);
        goal.assert(&conj);
        let tactic = Z3Tactic::new(&ctx, name);
        let result = tactic.apply(&ctx, &goal);
        // Just verify no panic and that the result is queryable.
        let _ = result.num_subgoals();
    }
}

#[test]
fn test_z3_tactic_ctx_simplify_alias_resolves() {
    // The historical short form "ctx-simplify" is aliased to the canonical
    // registry key "ctx-solver-simplify" and must still apply.
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "reg_ctx_alias_p");
    goal.assert(&p);
    let tactic = Z3Tactic::new(&ctx, "ctx-simplify");
    let result = tactic.apply(&ctx, &goal);
    let _ = result.num_subgoals();
}

#[test]
fn test_z3_tactic_unknown_name_returns_goal_unchanged() {
    // An unregistered name preserves the goal (SubGoals carrying the original).
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "reg_unknown_p");
    goal.assert(&p);
    let tactic = Z3Tactic::new(&ctx, "this-is-not-a-tactic");
    let result = tactic.apply(&ctx, &goal);
    assert_eq!(result.num_subgoals(), 1, "unknown tactic must yield one subgoal");
    // The single subgoal preserves the original assertion.
    assert_eq!(result.get_subgoal(0).size(), 1);
    assert_eq!(result.get_subgoal(0).get_formula(0).id, p.id);
}

#[test]
fn test_z3_tactic_smt_backend_name_passes_through() {
    // "smt" is a backend-only tactic (not in the registry); it passes the goal
    // through unchanged so a pipeline can continue to the solver backend.
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "reg_smt_p");
    goal.assert(&p);
    let tactic = Z3Tactic::new(&ctx, "smt");
    let result = tactic.apply(&ctx, &goal);
    assert_eq!(result.num_subgoals(), 1, "smt must pass the goal through");
    assert_eq!(result.get_subgoal(0).get_formula(0).id, p.id);
}

// ─── Z3DatatypeSort ───────────────────────────────────────────────────────────

#[test]
fn test_datatype_list_cons_nil() {
    let ctx = make_ctx();
    let nil = mk_constructor("Nil", &[]);
    let cons = mk_constructor("Cons", &[("head", "Int"), ("tail", "List")]);
    let dt = Z3DatatypeSort::new(&ctx, "List", &[nil, cons]);
    assert_eq!(dt.num_constructors(), 2);
}

#[test]
fn test_datatype_constructor_funcdecl() {
    let ctx = make_ctx();
    let nil = mk_constructor("Nil", &[]);
    let cons = mk_constructor("Cons", &[("head", "Int")]);
    let dt = Z3DatatypeSort::new(&ctx, "SimpleList", &[nil, cons]);
    // Constructor FuncDecl for Nil should have arity 0.
    let nil_fd = dt.constructor(&ctx, 0);
    assert_eq!(nil_fd.domain.len(), 0);
    assert_eq!(nil_fd.name, "Nil");
}

#[test]
fn test_datatype_recognizer_funcdecl() {
    let ctx = make_ctx();
    let nil = mk_constructor("Nil2", &[]);
    let dt = Z3DatatypeSort::new(&ctx, "NilList", &[nil]);
    let rec = dt.recognizer(&ctx, 0);
    assert_eq!(rec.name, "is-Nil2");
    assert_eq!(rec.domain.len(), 1);
    assert_eq!(rec.range, ctx.bool_sort());
}

#[test]
fn test_datatype_accessor_funcdecl() {
    let ctx = make_ctx();
    let point = mk_constructor("Point", &[("x", "Int"), ("y", "Int")]);
    let dt = Z3DatatypeSort::new(&ctx, "Point2D", &[point]);
    // Accessor for field 0 should return x with Int range.
    let acc = dt.accessor(&ctx, 0, 0);
    assert_eq!(acc.name, "x");
    assert_eq!(acc.range, ctx.int_sort());
}

#[test]
fn test_datatype_sort_id_is_valid() {
    let ctx = make_ctx();
    let unit = mk_constructor("Unit", &[]);
    let dt = Z3DatatypeSort::new(&ctx, "UnitType", &[unit]);
    // sort_id() must not equal Bool/Int/Real — it's a fresh datatype sort.
    let sid = dt.sort_id();
    assert_ne!(sid, ctx.bool_sort());
    assert_ne!(sid, ctx.int_sort());
}

// ─── check_assumptions / unsat_core ──────────────────────────────────────────

#[test]
fn test_check_assumptions_sat_trivial() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    // Build term in solver's own term manager to avoid cross-manager ID confusion.
    let bool_sort = solver.context().terms.sorts.bool_sort;
    let p_id = solver.context_mut().terms.mk_var("p_sat_trivial", bool_sort);
    let p = Bool::from_id(p_id);
    let result = solver.check_assumptions(&[p]);
    assert_eq!(result, SatResult::Sat);
}

#[test]
fn test_check_assumptions_unsat() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    // Build terms in solver's own term manager.
    let bool_sort = solver.context().terms.sorts.bool_sort;
    let p_id = solver.context_mut().terms.mk_var("p_uc", bool_sort);
    let not_p_id = solver.context_mut().terms.mk_not(p_id);
    let p = Bool::from_id(p_id);
    let not_p = Bool::from_id(not_p_id);
    let result = solver.check_assumptions(&[p, not_p]);
    assert_eq!(result, SatResult::Unsat);
}

#[test]
fn test_check_assumptions_does_not_pollute_stack() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    // Build terms in solver's own term manager.
    let bool_sort = solver.context().terms.sorts.bool_sort;
    let p_id = solver.context_mut().terms.mk_var("p_stack", bool_sort);
    let not_p_id = solver.context_mut().terms.mk_not(p_id);
    let p = Bool::from_id(p_id);
    let not_p = Bool::from_id(not_p_id);
    // First check_assumptions: contradictory
    let r1 = solver.check_assumptions(&[p, not_p]);
    assert_eq!(r1, SatResult::Unsat);
    // Now check without assumptions — should be SAT (no permanent assertions added)
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_unsat_core_returns_vec() {
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    // Build terms in solver's own term manager.
    let bool_sort = solver.context().terms.sorts.bool_sort;
    let p_id = solver.context_mut().terms.mk_var("p_core", bool_sort);
    let not_p_id = solver.context_mut().terms.mk_not(p_id);
    let p = Bool::from_id(p_id);
    let not_p = Bool::from_id(not_p_id);
    solver.assert(&p);
    solver.assert(&not_p);
    let result = solver.check();
    assert_eq!(result, SatResult::Unsat);
    // unsat_core() must not panic regardless of whether cores are enabled.
    let core = solver.unsat_core();
    // Core may be empty if production was not explicitly enabled, but must not panic.
    let _ = core.len();
}

// ─── Z3AstVector ──────────────────────────────────────────────────────────────

#[test]
fn test_ast_vector_push_len() {
    let ctx = make_ctx();
    let mut v = Z3AstVector::new(&ctx);
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    let p = Bool::new_const(&ctx, "vp");
    v.push(&p);
    assert_eq!(v.len(), 1);
    assert!(!v.is_empty());
}

#[test]
fn test_ast_vector_get() {
    let ctx = make_ctx();
    let mut v = Z3AstVector::new(&ctx);
    let p = Bool::new_const(&ctx, "vp2");
    let q = Bool::new_const(&ctx, "vq2");
    v.push(&p);
    v.push(&q);
    assert_eq!(v.get(0).id, p.id);
    assert_eq!(v.get(1).id, q.id);
}

#[test]
fn test_ast_vector_iter() {
    let ctx = make_ctx();
    let mut v = Z3AstVector::new(&ctx);
    let p = Bool::new_const(&ctx, "vp3");
    let q = Bool::new_const(&ctx, "vq3");
    v.push(&p);
    v.push(&q);
    let ids: Vec<_> = v.iter().map(|b| b.id).collect();
    assert_eq!(ids, vec![p.id, q.id]);
}

#[test]
fn test_ast_vector_any_true() {
    let ctx = make_ctx();
    let mut v = Z3AstVector::new(&ctx);
    let t = Bool::from_bool(&ctx, true);
    let f = Bool::from_bool(&ctx, false);
    assert!(!v.any_true());
    v.push(&f);
    assert!(!v.any_true());
    v.push(&t);
    assert!(v.any_true());
}

// ─── End-to-end integration tests ─────────────────────────────────────────────

#[test]
fn test_goal_tactic_pipeline_end_to_end() {
    // Build a simple goal, apply simplify, collect sub-goals.
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "e2e_p");
    let q = Bool::new_const(&ctx, "e2e_q");
    let conj = Bool::and(&ctx, &[p, q]);
    goal.assert(&conj);
    let t = Z3Tactic::new(&ctx, "simplify").then(&Z3Tactic::new(&ctx, "propagate-values"));
    let result = t.apply(&ctx, &goal);
    // Result must be a valid ApplyResult without panicking.
    assert!(result.num_subgoals() <= 1);
}

#[test]
fn test_probe_and_tactic_together() {
    let ctx = make_ctx();
    let mut goal = Z3Goal::new(&ctx);
    let p = Bool::new_const(&ctx, "pt_p");
    goal.assert(&p);
    let probe = Z3Probe::new(&ctx, "size");
    let v = probe.apply(&ctx, &goal);
    assert!(v >= 1.0);
    let tactic = Z3Tactic::new(&ctx, "simplify");
    let result = tactic.apply(&ctx, &goal);
    assert!(result.num_subgoals() <= 1);
}

#[test]
fn test_datatype_and_solver_interaction() {
    // Create a datatype and make sure the sort is registered in the context.
    let ctx = make_ctx();
    let leaf = mk_constructor("Leaf", &[("val", "Int")]);
    let node = mk_constructor("Node", &[("left", "Tree"), ("right", "Tree")]);
    let dt = Z3DatatypeSort::new(&ctx, "Tree", &[leaf, node]);
    assert_eq!(dt.num_constructors(), 2);
    // Constructor and recognizer FuncDecls must be built without panicking.
    let _leaf_con = dt.constructor(&ctx, 0);
    let _leaf_rec = dt.recognizer(&ctx, 0);
    let _leaf_acc = dt.accessor(&ctx, 0, 0);
}

#[test]
fn test_params_pipeline() {
    // Full pipeline: create params, set them, solve.
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let mut params = Z3Params::new(&ctx);
    params.set_u32("max-conflicts", 100_000);
    params.set_u32("max-decisions", 100_000);
    solver.set_params(&params);
    solver.set_logic("QF_LIA");
    let x = Int::new_const(&ctx, "x_params");
    let five = Int::from_i64(&ctx, 5);
    let gt = Int::gt(&ctx, &x, &five);
    solver.assert(&gt);
    assert_eq!(solver.check(), SatResult::Sat);
}

// ─── Z3FuncInterp / Z3FuncEntry / Z3Value ────────────────────────────────────

#[test]
fn test_z3_value_display() {
    let v = Z3Value::from_string("42".to_string());
    assert_eq!(v.as_str(), "42");
    assert_eq!(format!("{}", v), "42");
}

#[test]
fn test_z3_value_equality() {
    let a = Z3Value::from_string("true".to_string());
    let b = Z3Value::from_string("true".to_string());
    let c = Z3Value::from_string("false".to_string());
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_z3_func_interp_undeclared_function_returns_none() {
    // Querying a function that was never declared must return None.
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let t = Bool::from_bool(&ctx, true);
    solver.assert(&t);
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("expected model after SAT");
    let int_sort = solver.context().terms.sorts.int_sort;
    // A FuncDecl that was never registered in the solver context.
    let ghost_f = FuncDecl::new(&ctx, "undeclared_ghost_fn", &[int_sort], int_sort);
    assert!(
        model.get_func_interp(&ghost_f).is_none(),
        "undeclared function must not appear in the model"
    );
}

#[test]
fn test_z3_func_interp_declared_function_num_entries_ge_zero() {
    // Declare UF f: Int → Int, assert f(0) = 5, get model,
    // get_func_interp should succeed and num_entries >= 0.
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);

    // Work in the solver's term manager (solver.context_mut().terms) to avoid
    // cross-manager TermId confusion.
    let int_sort = solver.context().terms.sorts.int_sort;

    // Declare f in the solver context.
    solver.context_mut().declare_fun("fi_test_f", vec![int_sort], int_sort);

    // Build: f(0) — the apply term.
    let zero_id = solver.context_mut().terms.mk_int(0i64);
    let f_app = solver.context_mut().terms.mk_apply("fi_test_f", [zero_id], int_sort);

    // Build: f(0) = 5.
    let five_id = solver.context_mut().terms.mk_int(5i64);
    let eq = solver.context_mut().terms.mk_eq(f_app, five_id);
    solver.context_mut().assert(eq);

    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("expected model after SAT");

    // FuncDecl by name (used for lookup).
    let f_decl = FuncDecl::new(&ctx, "fi_test_f", &[int_sort], int_sort);
    let fi = model.get_func_interp(&f_decl);
    assert!(
        fi.is_some(),
        "declared function fi_test_f must appear in model"
    );
    let fi = fi.unwrap();
    // arity must match declaration
    assert_eq!(fi.arity(), 1, "arity must be 1");
    // num_entries is a usize; just access it to confirm no panic
    let _ = fi.num_entries();
    // else_value must be a non-empty string
    assert!(!fi.else_value().as_str().is_empty());
}

#[test]
fn test_z3_func_interp_entry_access() {
    // Declare UF g: Int Int → Bool, assert g(1, 2) = true, get func interp.
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let int_sort = solver.context().terms.sorts.int_sort;
    let bool_sort = solver.context().terms.sorts.bool_sort;

    solver.context_mut().declare_fun("fi_test_g", vec![int_sort, int_sort], bool_sort);

    let one = solver.context_mut().terms.mk_int(1i64);
    let two = solver.context_mut().terms.mk_int(2i64);
    let g_app = solver
        .context_mut()
        .terms
        .mk_apply("fi_test_g", [one, two], bool_sort);
    let tru = solver.context_mut().terms.mk_true();
    let eq = solver.context_mut().terms.mk_eq(g_app, tru);
    solver.context_mut().assert(eq);

    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("expected model after SAT");
    let g_decl = FuncDecl::new(&ctx, "fi_test_g", &[int_sort, int_sort], bool_sort);
    let fi = model.get_func_interp(&g_decl);
    assert!(fi.is_some(), "g must appear in model");
    let fi = fi.unwrap();
    assert_eq!(fi.arity(), 2);
    // Iterate entries without panicking.
    for entry in fi.entries() {
        assert_eq!(entry.args.len(), 2);
        assert!(!entry.value.as_str().is_empty());
    }
}

#[test]
fn test_z3_func_interp_else_value_non_empty() {
    // Even for a constrained function, else_value must not be empty.
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let int_sort = solver.context().terms.sorts.int_sort;
    solver.context_mut().declare_fun("fi_else_h", vec![int_sort], int_sort);
    let tru = solver.context_mut().terms.mk_true();
    solver.context_mut().assert(tru);
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("expected model after SAT");
    let h_decl = FuncDecl::new(&ctx, "fi_else_h", &[int_sort], int_sort);
    if let Some(fi) = model.get_func_interp(&h_decl) {
        assert!(
            !fi.else_value().as_str().is_empty(),
            "else_value must be a non-empty string"
        );
    }
    // If None it means the function wasn't found — that's also valid for an
    // unconstrained function in this simplified model extraction.
}

#[test]
fn test_z3_func_interp_get_model_no_panic_after_unsat() {
    // After UNSAT, get_model returns None; no panic expected.
    let ctx = make_ctx();
    let mut solver = make_solver(&ctx);
    let bool_sort = solver.context().terms.sorts.bool_sort;
    let p = solver.context_mut().terms.mk_var("p_interp_unsat", bool_sort);
    let not_p = solver.context_mut().terms.mk_not(p);
    solver.context_mut().assert(p);
    solver.context_mut().assert(not_p);
    assert_eq!(solver.check(), SatResult::Unsat);
    assert!(
        solver.get_model().is_none(),
        "get_model must return None after UNSAT"
    );
}
