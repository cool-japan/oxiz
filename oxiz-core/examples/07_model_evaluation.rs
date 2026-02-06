//! # Model Evaluation Example
//!
//! This example demonstrates model construction and formula evaluation.
//! It covers:
//! - Creating models (variable assignments)
//! - Evaluating formulas under a model
//! - Model completion for partial assignments
//! - Different value types
//!
//! ## Models in SMT
//! A model is a satisfying assignment that makes all asserted formulas true.
//! Models map variables to concrete values.
//!
//! ## Complexity
//! - Evaluation: O(n) where n is formula size
//! - Completion: O(n * m) where m is number of variables
//!
//! ## See Also
//! - [`Model`](oxiz_core::model::Model) for model representation
//! - [`ModelEvaluator`](oxiz_core::model::ModelEvaluator) for evaluation

use num_rational::Rational64;
use oxiz_core::ast::TermManager;
use oxiz_core::model::{EvalResult, Model, ModelEvaluator, Value};

fn main() {
    println!("=== OxiZ Core: Model Evaluation ===\n");

    let mut tm = TermManager::new();

    // ===== Example 1: Boolean Model =====
    println!("--- Example 1: Boolean Model Evaluation ---");
    let p = tm.mk_var("p", tm.sorts.bool_sort);
    let q = tm.mk_var("q", tm.sorts.bool_sort);
    let r = tm.mk_var("r", tm.sorts.bool_sort);

    // Create a model: p=true, q=false, r=true
    let mut model = Model::new();
    model.assign(p, Value::Bool(true));
    model.assign(q, Value::Bool(false));
    model.assign(r, Value::Bool(true));

    println!("Model:");
    println!("  p = true");
    println!("  q = false");
    println!("  r = true");

    // Evaluate formulas
    let formula1 = tm.mk_and(vec![p, q]); // p AND q
    let formula2 = tm.mk_or(vec![p, r]); // p OR r
    let formula3 = tm.mk_implies(q, r); // q => r

    let mut evaluator = ModelEvaluator::new(&model);
    println!("\nEvaluations:");
    print_eval("p AND q", &evaluator.eval(formula1, &tm));
    print_eval("p OR r", &evaluator.eval(formula2, &tm));
    print_eval("q => r", &evaluator.eval(formula3, &tm));

    // ===== Example 2: Integer Model =====
    println!("\n--- Example 2: Integer Model Evaluation ---");
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);
    let z = tm.mk_var("z", tm.sorts.int_sort);

    // Model: x=5, y=10, z=-3
    let mut int_model = Model::new();
    int_model.assign(x, Value::Int(5));
    int_model.assign(y, Value::Int(10));
    int_model.assign(z, Value::Int(-3));

    println!("Model:");
    println!("  x = 5");
    println!("  y = 10");
    println!("  z = -3");

    // Evaluate arithmetic expressions
    let x_plus_y = tm.mk_add(vec![x, y]);
    let y_minus_z = tm.mk_sub(y, z);
    let two = tm.mk_int(2);
    let x_times_2 = tm.mk_mul(vec![x, two]);

    let mut int_evaluator = ModelEvaluator::new(&int_model);
    println!("\nEvaluations:");
    print_eval("x + y", &int_evaluator.eval(x_plus_y, &tm));
    print_eval("y - z", &int_evaluator.eval(y_minus_z, &tm));
    print_eval("x * 2", &int_evaluator.eval(x_times_2, &tm));

    // Evaluate comparisons
    let five = tm.mk_int(5);
    let zero_cmp = tm.mk_int(0);
    let x_eq_5 = tm.mk_eq(x, five);
    let y_gt_x = tm.mk_gt(y, x);
    let z_lt_0 = tm.mk_lt(z, zero_cmp);

    println!("\nComparisons:");
    print_eval("x = 5 ?", &int_evaluator.eval(x_eq_5, &tm));
    print_eval("y > x ?", &int_evaluator.eval(y_gt_x, &tm));
    print_eval("z < 0 ?", &int_evaluator.eval(z_lt_0, &tm));

    // ===== Example 3: Value Types =====
    println!("\n--- Example 3: Value Types ---");

    println!("Boolean value:");
    let v_bool = Value::Bool(true);
    println!("  Value::Bool(true): {}", v_bool);
    println!("  is_bool(): {}", v_bool.is_bool());
    println!("  as_bool(): {:?}", v_bool.as_bool());

    println!("\nInteger value:");
    let v_int = Value::Int(42);
    println!("  Value::Int(42): {}", v_int);
    println!("  is_int(): {}", v_int.is_int());
    println!("  as_int(): {:?}", v_int.as_int());

    println!("\nRational value:");
    let v_rat = Value::Rational(Rational64::new(1, 3));
    println!("  Value::Rational(1/3): {}", v_rat);
    println!("  is_rational(): {}", v_rat.is_rational());
    println!("  as_rational(): {:?}", v_rat.as_rational());

    println!("\nBitvector value:");
    let v_bv = Value::BitVec(8, 255);
    println!("  Value::BitVec(8, 255): {}", v_bv);
    println!("  is_bitvec(): {}", v_bv.is_bitvec());
    println!("  as_bitvec(): {:?}", v_bv.as_bitvec());

    println!("\nString value:");
    let v_str = Value::String("hello".to_string());
    println!("  Value::String(\"hello\"): {}", v_str);
    println!("  as_string(): {:?}", v_str.as_string());

    println!("\nUndefined value:");
    let v_undef = Value::Undefined;
    println!("  Value::Undefined: {}", v_undef);
    println!("  is_undefined(): {}", v_undef.is_undefined());

    // ===== Example 4: Bitvector Evaluation =====
    println!("\n--- Example 4: Bitvector Evaluation ---");

    let bv8_sort = tm.sorts.bitvec(8);
    let a = tm.mk_var("a", bv8_sort);
    let b = tm.mk_var("b", bv8_sort);

    let mut bv_model = Model::new();
    bv_model.assign(a, Value::BitVec(8, 0xF0)); // 11110000
    bv_model.assign(b, Value::BitVec(8, 0x0F)); // 00001111

    println!("Model: a=0xF0, b=0x0F");

    let mut bv_eval = ModelEvaluator::new(&bv_model);

    // a AND b
    let a_and_b = tm.mk_bv_and(a, b);
    print_eval("a AND b (BV)", &bv_eval.eval(a_and_b, &tm));

    // a OR b
    let a_or_b = tm.mk_bv_or(a, b);
    print_eval("a OR b (BV)", &bv_eval.eval(a_or_b, &tm));

    // a XOR b
    let a_xor_b = tm.mk_bv_xor(a, b);
    print_eval("a XOR b (BV)", &bv_eval.eval(a_xor_b, &tm));

    // ===== Example 5: ITE Evaluation =====
    println!("\n--- Example 5: ITE Evaluation ---");

    // ite(p, x, y) with p=true, x=5, y=10 => 5
    let mut ite_model = Model::new();
    ite_model.assign(p, Value::Bool(true));
    ite_model.assign(q, Value::Bool(false));
    ite_model.assign(x, Value::Int(5));
    ite_model.assign(y, Value::Int(10));

    let ite_expr = tm.mk_ite(p, x, y);
    let ite_expr2 = tm.mk_ite(q, x, y);

    let mut ite_eval = ModelEvaluator::new(&ite_model);
    print_eval("ite(p, x, y) with p=true", &ite_eval.eval(ite_expr, &tm));
    print_eval("ite(q, x, y) with q=false", &ite_eval.eval(ite_expr2, &tm));

    // ===== Example 6: Undefined Variables =====
    println!("\n--- Example 6: Undefined Variables ---");

    let undefined_var = tm.mk_var("undefined_z", tm.sorts.int_sort);
    let undef_result = ite_eval.eval(undefined_var, &tm);
    print_eval("undefined_z", &undef_result);

    // ===== Example 7: Model Operations =====
    println!("\n--- Example 7: Model Operations ---");

    println!("Model has {} assignments", model.len());
    println!("Has assignment for p? {}", model.has(p));

    let unknown = tm.mk_var("unknown", tm.sorts.bool_sort);
    println!("Has assignment for unknown? {}", model.has(unknown));

    // Get value
    if let Some(val) = model.get(p) {
        println!("Value of p: {}", val);
    }

    // Remove assignment
    let mut model_copy = model.clone();
    let removed = model_copy.remove(p);
    println!("\nRemoved p: {:?}", removed);
    println!("Model size after removal: {}", model_copy.len());

    // ===== Example 8: Model Merge =====
    println!("\n--- Example 8: Model Merge ---");

    let mut model_a = Model::new();
    let mut model_b = Model::new();

    model_a.assign(x, Value::Int(1));
    model_a.assign(y, Value::Int(2));

    model_b.assign(y, Value::Int(100)); // Will not override
    model_b.assign(z, Value::Int(3));

    println!("Model A: x=1, y=2");
    println!("Model B: y=100, z=3");

    model_a.merge(&model_b);
    println!("\nAfter merge (A.merge(B)):");
    println!("  x = {:?}", model_a.get(x).map(|v| v.to_string()));
    println!(
        "  y = {:?} (A's value preserved)",
        model_a.get(y).map(|v| v.to_string())
    );
    println!("  z = {:?}", model_a.get(z).map(|v| v.to_string()));

    // ===== Example 9: Evaluator Caching =====
    println!("\n--- Example 9: Evaluator Caching ---");

    let mut cached_eval = ModelEvaluator::new(&int_model);
    let _ = cached_eval.eval(x_plus_y, &tm);
    let _ = cached_eval.eval(x_plus_y, &tm);
    let _ = cached_eval.eval(x_plus_y, &tm);

    println!("Cache size after evaluations: {}", cached_eval.cache_size());

    cached_eval.clear_cache();
    println!("Cache size after clear: {}", cached_eval.cache_size());

    // Without cache
    let mut nocache_eval = ModelEvaluator::without_cache(&int_model);
    let _ = nocache_eval.eval(x_plus_y, &tm);
    println!(
        "\nEvaluator without cache: cache_size = {}",
        nocache_eval.cache_size()
    );

    // ===== Example 10: EvalResult API =====
    println!("\n--- Example 10: EvalResult API ---");

    let ok_result = EvalResult::Ok(Value::Int(42));
    let undef_result = EvalResult::Undefined(x);
    let err_result = EvalResult::Error("Type mismatch".to_string());

    println!("EvalResult::Ok(42):");
    println!("  is_ok(): {}", ok_result.is_ok());
    println!("  value(): {:?}", ok_result.value());

    println!("\nEvalResult::Undefined:");
    println!("  is_ok(): {}", undef_result.is_ok());
    println!("  value(): {:?}", undef_result.value());

    println!("\nEvalResult::Error:");
    println!("  is_ok(): {}", err_result.is_ok());
    println!("  value(): {:?}", err_result.value());

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Models assign concrete values to variables");
    println!("  2. Evaluation computes formula truth value under model");
    println!("  3. Various value types: Bool, Int, Rational, BitVec, etc.");
    println!("  4. Caching improves repeated evaluation performance");
    println!("  5. Model operations: merge, remove, get, has");
}

fn print_eval(expr: &str, result: &EvalResult) {
    match result {
        EvalResult::Ok(value) => {
            println!("  {} = {}", expr, value);
        }
        EvalResult::Undefined(term) => {
            println!("  {} = undefined (term {:?})", expr, term);
        }
        EvalResult::Error(msg) => {
            println!("  {} = ERROR: {}", expr, msg);
        }
    }
}
