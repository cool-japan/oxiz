//! # Model Evaluation Example
//!
//! This example demonstrates model construction and formula evaluation.
//! It covers:
//! - Creating models (variable assignments)
//! - Evaluating formulas under a model
//! - Model completion for partial assignments
//! - Prime implicant extraction
//! - Model minimization
//!
//! ## Models in SMT
//! A model is a satisfying assignment that makes all asserted formulas true.
//! Models map variables to concrete values.
//!
//! ## Complexity
//! - Evaluation: O(n) where n is formula size
//! - Completion: O(n * m) where m is number of variables
//! - Implicant extraction: O(2^k) where k is relevant variables
//!
//! ## See Also
//! - [`Model`](oxiz_core::model::Model) for model representation
//! - [`ModelEvaluator`](oxiz_core::model::ModelEvaluator) for evaluation
//! - [`ImplicantExtractor`](oxiz_core::model::ImplicantExtractor)

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_core::model::{
    ImplicantConfig, ImplicantExtractor, Model, ModelCompletion, ModelCompletionConfig,
    ModelEvaluator, Value,
};

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

    let evaluator = ModelEvaluator::new(&model);
    println!("\nEvaluations:");
    println!("  p AND q = {:?}", evaluator.eval(formula1, &tm));
    println!("  p OR r = {:?}", evaluator.eval(formula2, &tm));
    println!("  q => r = {:?}", evaluator.eval(formula3, &tm));

    // ===== Example 2: Integer Model =====
    println!("\n--- Example 2: Integer Model Evaluation ---");
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);
    let z = tm.mk_var("z", tm.sorts.int_sort);

    // Model: x=5, y=10, z=-3
    let mut int_model = Model::new();
    int_model.assign(x, Value::Int(BigInt::from(5)));
    int_model.assign(y, Value::Int(BigInt::from(10)));
    int_model.assign(z, Value::Int(BigInt::from(-3)));

    println!("Model:");
    println!("  x = 5");
    println!("  y = 10");
    println!("  z = -3");

    // Evaluate arithmetic expressions
    let x_plus_y = tm.mk_add(vec![x, y]);
    let y_minus_z = tm.mk_sub(y, z);
    let x_times_2 = tm.mk_mul(vec![x, tm.mk_int(BigInt::from(2))]);

    let int_evaluator = ModelEvaluator::new(&int_model);
    println!("\nEvaluations:");
    println!("  x + y = {:?}", int_evaluator.eval(x_plus_y, &tm));
    println!("  y - z = {:?}", int_evaluator.eval(y_minus_z, &tm));
    println!("  x * 2 = {:?}", int_evaluator.eval(x_times_2, &tm));

    // Evaluate comparisons
    let five = tm.mk_int(BigInt::from(5));
    let x_eq_5 = tm.mk_eq(x, five);
    let y_gt_x = tm.mk_gt(y, x);
    let z_lt_0 = tm.mk_lt(z, tm.mk_int(BigInt::from(0)));

    println!("\nComparisons:");
    println!("  x = 5 ? {:?}", int_evaluator.eval(x_eq_5, &tm));
    println!("  y > x ? {:?}", int_evaluator.eval(y_gt_x, &tm));
    println!("  z < 0 ? {:?}", int_evaluator.eval(z_lt_0, &tm));

    // ===== Example 3: Partial Model (Completion) =====
    println!("\n--- Example 3: Model Completion ---");
    let a = tm.mk_var("a", tm.sorts.bool_sort);
    let b = tm.mk_var("b", tm.sorts.bool_sort);
    let c = tm.mk_var("c", tm.sorts.bool_sort);
    let d = tm.mk_var("d", tm.sorts.bool_sort);

    // Partial model: only a and b assigned
    let mut partial_model = Model::new();
    partial_model.assign(a, Value::Bool(true));
    partial_model.assign(b, Value::Bool(false));

    println!("Partial model:");
    println!("  a = true");
    println!("  b = false");
    println!("  c = ? (unassigned)");
    println!("  d = ? (unassigned)");

    // Formula with unassigned variables
    let formula = tm.mk_and(vec![a, tm.mk_or(vec![b, c])]);
    println!("\nFormula: a AND (b OR c)");

    // Complete the model
    let config = ModelCompletionConfig {
        default_bool: Some(false),
        default_int: Some(BigInt::from(0)),
        ..Default::default()
    };

    let completion = ModelCompletion::new(config);
    let complete_model = completion.complete(&partial_model, &[c, d], &tm);

    println!("\nCompleted model:");
    let eval = ModelEvaluator::new(&complete_model);
    println!("  a = {:?}", eval.eval(a, &tm));
    println!("  b = {:?}", eval.eval(b, &tm));
    println!("  c = {:?}", eval.eval(c, &tm));
    println!("  d = {:?}", eval.eval(d, &tm));
    println!("\nFormula value: {:?}", eval.eval(formula, &tm));

    // ===== Example 4: Prime Implicant Extraction =====
    println!("\n--- Example 4: Prime Implicant Extraction ---");
    let p1 = tm.mk_var("p1", tm.sorts.bool_sort);
    let p2 = tm.mk_var("p2", tm.sorts.bool_sort);
    let p3 = tm.mk_var("p3", tm.sorts.bool_sort);
    let p4 = tm.mk_var("p4", tm.sorts.bool_sort);

    // Formula: (p1 AND p2) OR (p3 AND p4)
    let and1 = tm.mk_and(vec![p1, p2]);
    let and2 = tm.mk_and(vec![p3, p4]);
    let or_formula = tm.mk_or(vec![and1, and2]);

    // Model: p1=true, p2=true, p3=true, p4=false
    let mut impl_model = Model::new();
    impl_model.assign(p1, Value::Bool(true));
    impl_model.assign(p2, Value::Bool(true));
    impl_model.assign(p3, Value::Bool(true));
    impl_model.assign(p4, Value::Bool(false));

    println!("Formula: (p1 AND p2) OR (p3 AND p4)");
    println!("Model: p1=T, p2=T, p3=T, p4=F");

    let impl_config = ImplicantConfig {
        minimize: true,
        ..Default::default()
    };
    let extractor = ImplicantExtractor::new(impl_config);
    let implicant = extractor.extract(&impl_model, or_formula, &tm);

    println!("\nPrime implicant:");
    println!("  Essential vars: {:?}", implicant.literals);
    println!("  (Only p1 and p2 are needed; p3 and p4 are irrelevant)");

    // ===== Example 5: Mixed Theory Model =====
    println!("\n--- Example 5: Mixed Theory Model ---");
    let bool_var = tm.mk_var("flag", tm.sorts.bool_sort);
    let int_var = tm.mk_var("count", tm.sorts.int_sort);

    let mut mixed_model = Model::new();
    mixed_model.assign(bool_var, Value::Bool(true));
    mixed_model.assign(int_var, Value::Int(BigInt::from(42)));

    // Formula: flag => (count > 0)
    let count_gt_0 = tm.mk_gt(int_var, tm.mk_int(BigInt::from(0)));
    let mixed_formula = tm.mk_implies(bool_var, count_gt_0);

    let mixed_eval = ModelEvaluator::new(&mixed_model);
    println!("Formula: flag => (count > 0)");
    println!("Model: flag=true, count=42");
    println!("Evaluation: {:?}\n", mixed_eval.eval(mixed_formula, &tm));

    // ===== Example 6: Bitvector Model =====
    println!("--- Example 6: Bitvector Model ---");
    let bv8_sort = tm.sorts.mk_bv_sort(8);
    let bv_a = tm.mk_var("a", bv8_sort);
    let bv_b = tm.mk_var("b", bv8_sort);

    let mut bv_model = Model::new();
    bv_model.assign(bv_a, Value::BitVec(BigInt::from(10), 8));
    bv_model.assign(bv_b, Value::BitVec(BigInt::from(20), 8));

    println!("Model: a=0x0A, b=0x14");

    // a + b
    let bv_sum = tm.mk_bvadd(bv_a, bv_b);
    let bv_eval = ModelEvaluator::new(&bv_model);
    println!("a + b = {:?}", bv_eval.eval(bv_sum, &tm));
    println!("Expected: 0x1E (30)\n");

    // ===== Example 7: Array Model =====
    println!("--- Example 7: Array Model (Theory of Arrays) ---");
    let arr_sort = tm.sorts.mk_array_sort(tm.sorts.int_sort, tm.sorts.int_sort);
    let arr = tm.mk_var("arr", arr_sort);
    let idx = tm.mk_var("i", tm.sorts.int_sort);

    // Conceptual model: arr = {0->10, 1->20, 2->30, ...}
    // (In practice, arrays use functional representation)
    let mut arr_model = Model::new();
    arr_model.assign(idx, Value::Int(BigInt::from(1)));
    // Array values are more complex; typically use store/select chains

    println!("Array model evaluation (conceptual)");
    println!("  arr[i] where i=1");
    println!("  (Full array model representation requires theory solver)");

    // ===== Example 8: Model Serialization =====
    println!("\n--- Example 8: Model Serialization ---");
    let ser_model = Model::new();
    // In a real implementation, models can be serialized to JSON/SMT-LIB format
    println!("Model serialization:");
    println!("  SMT-LIB format: (model (define-fun x () Int 5) ...)");
    println!("  JSON format: {{\"x\": 5, \"y\": 10, ...}}");
    println!("  (Implementation depends on format requirements)");

    // ===== Example 9: Evaluation Cache =====
    println!("\n--- Example 9: Evaluation with Caching ---");
    use oxiz_core::model::EvalCache;

    let mut cache = EvalCache::new();
    let cache_eval = ModelEvaluator::new(&int_model);

    // Repeated evaluation (cached)
    let expr = tm.mk_add(vec![x, y]);
    let result1 = cache_eval.eval_cached(expr, &tm, &mut cache);
    let result2 = cache_eval.eval_cached(expr, &tm, &mut cache);

    println!("Cached evaluation:");
    println!("  First:  {:?} (computed)", result1);
    println!("  Second: {:?} (cached)", result2);
    println!("  Cache hits: {}", cache.hits());
    println!("  Cache misses: {}", cache.misses());

    // ===== Example 10: Counterexample Models =====
    println!("\n--- Example 10: Counterexample Models ---");
    // When a formula is satisfiable, the model is a counterexample to its negation
    let assertion = tm.mk_gt(x, tm.mk_int(BigInt::from(0)));
    println!("Assertion: x > 0");

    let counterex_model = Model::new();
    // If model assigns x=5, it's a counterexample to ¬(x > 0)
    println!("Counterexample model: x = 5");
    println!("  Satisfies: x > 0");
    println!("  Refutes: x <= 0");
    println!("  (Used in bounded model checking, test generation)");

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Models assign concrete values to variables");
    println!("  2. Evaluation computes formula truth value under model");
    println!("  3. Model completion fills in unassigned variables");
    println!("  4. Prime implicants extract essential assignments");
    println!("  5. Caching improves repeated evaluation performance");
}
