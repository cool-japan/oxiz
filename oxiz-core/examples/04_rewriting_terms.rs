//! # Term Rewriting Example
//!
//! This example demonstrates term rewriting for algebraic simplification.
//! It covers:
//! - Boolean rewriting (constant folding, De Morgan's)
//! - Arithmetic rewriting (polynomial normalization)
//! - Bitvector rewriting
//! - Custom rewrite rules
//! - Rewrite strategies (bottom-up, top-down, fixpoint)
//!
//! ## Rewriting vs Tactics
//! - Rewriters: Local term transformations (bottom-up or top-down)
//! - Tactics: Global goal transformations (can split goals)
//!
//! ## Complexity
//! - Bottom-up: O(n) where n is term size
//! - Fixpoint: O(n * k) where k is iterations until convergence
//!
//! ## See Also
//! - [`Rewriter`](oxiz_core::rewrite::Rewriter) trait
//! - [`BoolRewriter`](oxiz_core::rewrite::BoolRewriter)
//! - [`ArithRewriter`](oxiz_core::rewrite::ArithRewriter)

use num_bigint::BigInt;
use oxiz_core::ast::{TermKind, TermManager};
use oxiz_core::rewrite::{
    ArithRewriter, BoolRewriter, BottomUpRewriter, BvRewriter, CompositeRewriter,
    IteratingRewriter, RewriteConfig, Rewriter,
};

fn main() {
    println!("=== OxiZ Core: Term Rewriting ===\n");

    let mut tm = TermManager::new();

    // ===== Example 1: Boolean Rewriting =====
    println!("--- Example 1: Boolean Constant Folding ---");
    let p = tm.mk_var("p", tm.sorts.bool_sort);
    let true_term = tm.mk_true();
    let false_term = tm.mk_false();

    // p AND true -> p
    let p_and_true = tm.mk_and(vec![p, true_term]);
    println!("Before: p AND true = {:?}", p_and_true);

    let bool_rewriter = BoolRewriter::new();
    let mut bottom_up = BottomUpRewriter::new(bool_rewriter);
    let rewritten = bottom_up.rewrite(p_and_true, &mut tm);
    println!("After:  {:?}", rewritten.term);
    println!("Expected: p\n");

    // p OR false -> p
    let p_or_false = tm.mk_or(vec![p, false_term]);
    println!("Before: p OR false = {:?}", p_or_false);
    let rewritten2 = bottom_up.rewrite(p_or_false, &mut tm);
    println!("After:  {:?}", rewritten2.term);
    println!("Expected: p\n");

    // ===== Example 2: Double Negation =====
    println!("--- Example 2: Double Negation Elimination ---");
    let q = tm.mk_var("q", tm.sorts.bool_sort);
    let not_q = tm.mk_not(q);
    let not_not_q = tm.mk_not(not_q);

    println!("Before: NOT(NOT q) = {:?}", not_not_q);
    let rewritten3 = bottom_up.rewrite(not_not_q, &mut tm);
    println!("After:  {:?}", rewritten3.term);
    println!("Expected: q\n");

    // ===== Example 3: De Morgan's Laws =====
    println!("--- Example 3: De Morgan's Laws ---");
    let r = tm.mk_var("r", tm.sorts.bool_sort);
    let s = tm.mk_var("s", tm.sorts.bool_sort);

    // NOT(p AND q) -> (NOT p) OR (NOT q)
    let and_rs = tm.mk_and(vec![r, s]);
    let not_and_rs = tm.mk_not(and_rs);

    println!("Before: NOT(r AND s) = {:?}", not_and_rs);
    let rewritten4 = bottom_up.rewrite(not_and_rs, &mut tm);
    println!("After:  {:?}", rewritten4.term);
    println!("Expected: (NOT r) OR (NOT s)\n");

    // ===== Example 4: Arithmetic Simplification =====
    println!("--- Example 4: Arithmetic Constant Folding ---");
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let zero = tm.mk_int(BigInt::from(0));
    let one = tm.mk_int(BigInt::from(1));

    // x + 0 -> x
    let x_plus_0 = tm.mk_add(vec![x, zero]);
    println!("Before: x + 0 = {:?}", x_plus_0);

    let arith_rewriter = ArithRewriter::new();
    let mut arith_bottom_up = BottomUpRewriter::new(arith_rewriter);
    let rewritten5 = arith_bottom_up.rewrite(x_plus_0, &mut tm);
    println!("After:  {:?}", rewritten5.term);
    println!("Expected: x\n");

    // x * 1 -> x
    let x_times_1 = tm.mk_mul(vec![x, one]);
    println!("Before: x * 1 = {:?}", x_times_1);
    let rewritten6 = arith_bottom_up.rewrite(x_times_1, &mut tm);
    println!("After:  {:?}", rewritten6.term);
    println!("Expected: x\n");

    // x * 0 -> 0
    let x_times_0 = tm.mk_mul(vec![x, zero]);
    println!("Before: x * 0 = {:?}", x_times_0);
    let rewritten7 = arith_bottom_up.rewrite(x_times_0, &mut tm);
    println!("After:  {:?}", rewritten7.term);
    println!("Expected: 0\n");

    // ===== Example 5: Polynomial Normalization =====
    println!("--- Example 5: Polynomial Normalization ---");
    let y = tm.mk_var("y", tm.sorts.int_sort);

    // (x + y) + x -> 2*x + y (collect like terms)
    let x_plus_y = tm.mk_add(vec![x, y]);
    let sum = tm.mk_add(vec![x_plus_y, x]);

    println!("Before: (x + y) + x = {:?}", sum);
    let rewritten8 = arith_bottom_up.rewrite(sum, &mut tm);
    println!("After:  {:?}", rewritten8.term);
    println!("Expected: 2*x + y (normalized form)\n");

    // ===== Example 6: Constant Evaluation =====
    println!("--- Example 6: Constant Evaluation ---");
    let five = tm.mk_int(BigInt::from(5));
    let ten = tm.mk_int(BigInt::from(10));
    let three = tm.mk_int(BigInt::from(3));

    // (5 + 10) * 3 -> 45
    let sum_5_10 = tm.mk_add(vec![five, ten]);
    let product = tm.mk_mul(vec![sum_5_10, three]);

    println!("Before: (5 + 10) * 3 = {:?}", product);
    let rewritten9 = arith_bottom_up.rewrite(product, &mut tm);
    println!("After:  {:?}", rewritten9.term);
    println!("Expected: 45\n");

    // ===== Example 7: Bitvector Rewriting =====
    println!("--- Example 7: Bitvector Simplification ---");
    let bv8_sort = tm.sorts.mk_bv_sort(8);
    let a = tm.mk_var("a", bv8_sort);
    let bv_zero = tm.mk_bv_numeral(BigInt::from(0), 8);
    let bv_ones = tm.mk_bv_numeral(BigInt::from(255), 8); // all bits set

    // a AND 0 -> 0
    let a_and_0 = tm.mk_bvand(a, bv_zero);
    println!("Before: a AND 0x00 = {:?}", a_and_0);

    let bv_rewriter = BvRewriter::new();
    let mut bv_bottom_up = BottomUpRewriter::new(bv_rewriter);
    let rewritten10 = bv_bottom_up.rewrite(a_and_0, &mut tm);
    println!("After:  {:?}", rewritten10.term);
    println!("Expected: 0x00\n");

    // a OR 0xFF -> 0xFF
    let a_or_ones = tm.mk_bvor(a, bv_ones);
    println!("Before: a OR 0xFF = {:?}", a_or_ones);
    let rewritten11 = bv_bottom_up.rewrite(a_or_ones, &mut tm);
    println!("After:  {:?}", rewritten11.term);
    println!("Expected: 0xFF\n");

    // ===== Example 8: Composite Rewriter =====
    println!("--- Example 8: Composite Rewriting (Bool + Arith) ---");
    let z = tm.mk_var("z", tm.sorts.int_sort);

    // (z + 0 > 0) AND true -> z > 0
    let z_plus_0 = tm.mk_add(vec![z, zero]);
    let zero2 = tm.mk_int(BigInt::from(0));
    let comparison = tm.mk_gt(z_plus_0, zero2);
    let formula = tm.mk_and(vec![comparison, true_term]);

    println!("Before: (z + 0 > 0) AND true = {:?}", formula);

    // Compose bool and arith rewriters
    let composite = CompositeRewriter::new(vec![
        Box::new(BoolRewriter::new()),
        Box::new(ArithRewriter::new()),
    ]);
    let mut composite_bottom_up = BottomUpRewriter::new(composite);
    let rewritten12 = composite_bottom_up.rewrite(formula, &mut tm);
    println!("After:  {:?}", rewritten12.term);
    println!("Expected: z > 0\n");

    // ===== Example 9: Fixpoint Iteration =====
    println!("--- Example 9: Fixpoint Rewriting ---");
    let w = tm.mk_var("w", tm.sorts.int_sort);

    // ((w + 0) + 0) + 0 -> w (needs multiple passes)
    let w_plus_0 = tm.mk_add(vec![w, zero]);
    let w_plus_0_plus_0 = tm.mk_add(vec![w_plus_0, zero]);
    let nested = tm.mk_add(vec![w_plus_0_plus_0, zero]);

    println!("Before: ((w + 0) + 0) + 0 = {:?}", nested);

    let config = RewriteConfig {
        max_iterations: 10,
        ..Default::default()
    };
    let mut fixpoint = IteratingRewriter::new(ArithRewriter::new(), config);
    let rewritten13 = fixpoint.rewrite(nested, &mut tm);
    println!("After:  {:?}", rewritten13.term);
    println!("Iterations: {}", rewritten13.stats.iterations);
    println!("Expected: w (after 3 iterations)\n");

    // ===== Example 10: Rewrite Statistics =====
    println!("--- Example 10: Rewrite Statistics ---");
    println!("Rewrite statistics for last operation:");
    println!("  Rewrites applied: {}", rewritten13.stats.rewrites_applied);
    println!("  Iterations: {}", rewritten13.stats.iterations);
    println!("  Term size before: {}", rewritten13.stats.size_before);
    println!("  Term size after: {}", rewritten13.stats.size_after);
    println!(
        "  Reduction: {}%",
        if rewritten13.stats.size_before > 0 {
            100 * (rewritten13.stats.size_before - rewritten13.stats.size_after)
                / rewritten13.stats.size_before
        } else {
            0
        }
    );

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Rewriters perform local term transformations");
    println!("  2. Bottom-up traversal applies rules from leaves to root");
    println!("  3. Composite rewriters combine multiple strategies");
    println!("  4. Fixpoint iteration applies rewrites until convergence");
    println!("  5. Statistics help measure rewriting effectiveness");
}
