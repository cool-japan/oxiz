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

use oxiz_core::ast::TermManager;
use oxiz_core::rewrite::{
    ArithRewriter, BoolRewriter, BottomUpRewriter, BvRewriter, CompositeRewriter,
    IteratingRewriter, RewriteConfig, RewriteContext, RewriteResult, Rewriter,
};

fn main() {
    println!("=== OxiZ Core: Term Rewriting ===\n");

    let mut tm = TermManager::new();
    let mut ctx = RewriteContext::new();

    // ===== Example 1: Boolean Rewriting =====
    println!("--- Example 1: Boolean Constant Folding ---");
    let p = tm.mk_var("p", tm.sorts.bool_sort);
    let true_term = tm.mk_true();
    let false_term = tm.mk_false();

    // p AND true -> p
    let p_and_true = tm.mk_and(vec![p, true_term]);
    println!("Before: p AND true = {:?}", p_and_true);

    let mut bool_rewriter = BoolRewriter::new();
    let rewritten = bool_rewriter.rewrite(p_and_true, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten.term());
    println!("Was rewritten: {}", rewritten.was_rewritten());
    println!("Expected: p\n");

    // p OR false -> p
    let p_or_false = tm.mk_or(vec![p, false_term]);
    println!("Before: p OR false = {:?}", p_or_false);
    let rewritten2 = bool_rewriter.rewrite(p_or_false, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten2.term());
    println!("Was rewritten: {}", rewritten2.was_rewritten());
    println!("Expected: p\n");

    // ===== Example 2: Double Negation =====
    println!("--- Example 2: Double Negation Elimination ---");
    let q = tm.mk_var("q", tm.sorts.bool_sort);
    let not_q = tm.mk_not(q);
    let not_not_q = tm.mk_not(not_q);

    println!("Before: NOT(NOT q) = {:?}", not_not_q);
    let rewritten3 = bool_rewriter.rewrite(not_not_q, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten3.term());
    println!("Was rewritten: {}", rewritten3.was_rewritten());
    println!("Expected: q\n");

    // ===== Example 3: De Morgan's Laws =====
    println!("--- Example 3: De Morgan's Laws ---");
    let r = tm.mk_var("r", tm.sorts.bool_sort);
    let s = tm.mk_var("s", tm.sorts.bool_sort);

    // NOT(p AND q) -> (NOT p) OR (NOT q)
    let and_rs = tm.mk_and(vec![r, s]);
    let not_and_rs = tm.mk_not(and_rs);

    println!("Before: NOT(r AND s) = {:?}", not_and_rs);
    let rewritten4 = bool_rewriter.rewrite(not_and_rs, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten4.term());
    println!("Was rewritten: {}", rewritten4.was_rewritten());
    println!("Expected: (NOT r) OR (NOT s)\n");

    // ===== Example 4: Arithmetic Simplification =====
    println!("--- Example 4: Arithmetic Constant Folding ---");
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let zero = tm.mk_int(0);
    let one = tm.mk_int(1);

    // x + 0 -> x
    let x_plus_0 = tm.mk_add(vec![x, zero]);
    println!("Before: x + 0 = {:?}", x_plus_0);

    let mut arith_rewriter = ArithRewriter::new();
    let rewritten5 = arith_rewriter.rewrite(x_plus_0, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten5.term());
    println!("Was rewritten: {}", rewritten5.was_rewritten());
    println!("Expected: x\n");

    // x * 1 -> x
    let x_times_1 = tm.mk_mul(vec![x, one]);
    println!("Before: x * 1 = {:?}", x_times_1);
    let rewritten6 = arith_rewriter.rewrite(x_times_1, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten6.term());
    println!("Was rewritten: {}", rewritten6.was_rewritten());
    println!("Expected: x\n");

    // x * 0 -> 0
    let x_times_0 = tm.mk_mul(vec![x, zero]);
    println!("Before: x * 0 = {:?}", x_times_0);
    let rewritten7 = arith_rewriter.rewrite(x_times_0, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten7.term());
    println!("Was rewritten: {}", rewritten7.was_rewritten());
    println!("Expected: 0\n");

    // ===== Example 5: Polynomial Normalization =====
    println!("--- Example 5: Polynomial Normalization ---");
    let y = tm.mk_var("y", tm.sorts.int_sort);

    // (x + y) + x -> 2*x + y (collect like terms)
    let x_plus_y = tm.mk_add(vec![x, y]);
    let sum = tm.mk_add(vec![x_plus_y, x]);

    println!("Before: (x + y) + x = {:?}", sum);
    let rewritten8 = arith_rewriter.rewrite(sum, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten8.term());
    println!("Was rewritten: {}", rewritten8.was_rewritten());
    println!("Expected: 2*x + y (normalized form)\n");

    // ===== Example 6: Constant Evaluation =====
    println!("--- Example 6: Constant Evaluation ---");
    let five = tm.mk_int(5);
    let ten = tm.mk_int(10);
    let three = tm.mk_int(3);

    // (5 + 10) * 3 -> 45
    let sum_5_10 = tm.mk_add(vec![five, ten]);
    let product = tm.mk_mul(vec![sum_5_10, three]);

    println!("Before: (5 + 10) * 3 = {:?}", product);
    let rewritten9 = arith_rewriter.rewrite(product, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten9.term());
    println!("Was rewritten: {}", rewritten9.was_rewritten());
    println!("Expected: 45\n");

    // ===== Example 7: Bitvector Rewriting =====
    println!("--- Example 7: Bitvector Simplification ---");
    let bv8_sort = tm.sorts.bitvec(8);
    let a = tm.mk_var("a", bv8_sort);
    let bv_zero = tm.mk_bitvec(0u64, 8);
    let bv_ones = tm.mk_bitvec(255u64, 8); // all bits set

    // a AND 0 -> 0
    let a_and_0 = tm.mk_bv_and(a, bv_zero);
    println!("Before: a AND 0x00 = {:?}", a_and_0);

    let mut bv_rewriter = BvRewriter::new();
    let rewritten10 = bv_rewriter.rewrite(a_and_0, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten10.term());
    println!("Was rewritten: {}", rewritten10.was_rewritten());
    println!("Expected: 0x00\n");

    // a OR 0xFF -> 0xFF
    let a_or_ones = tm.mk_bv_or(a, bv_ones);
    println!("Before: a OR 0xFF = {:?}", a_or_ones);
    let rewritten11 = bv_rewriter.rewrite(a_or_ones, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten11.term());
    println!("Was rewritten: {}", rewritten11.was_rewritten());
    println!("Expected: 0xFF\n");

    // ===== Example 8: Composite Rewriter =====
    println!("--- Example 8: Composite Rewriting (Bool + Arith) ---");
    let z = tm.mk_var("z", tm.sorts.int_sort);

    // (z + 0 > 0) AND true -> z > 0
    let z_plus_0 = tm.mk_add(vec![z, zero]);
    let zero2 = tm.mk_int(0);
    let comparison = tm.mk_gt(z_plus_0, zero2);
    let formula = tm.mk_and(vec![comparison, true_term]);

    println!("Before: (z + 0 > 0) AND true = {:?}", formula);

    // Compose bool and arith rewriters
    let mut composite = CompositeRewriter::new("bool+arith");
    composite.add(BoolRewriter::new());
    composite.add(ArithRewriter::new());

    let mut composite_bottom_up = BottomUpRewriter::new(composite);
    let rewritten12 = composite_bottom_up.rewrite(formula, &mut ctx, &mut tm);
    println!("After:  {:?}", rewritten12.term());
    println!("Was rewritten: {}", rewritten12.was_rewritten());
    println!("Expected: z > 0\n");

    // ===== Example 9: Fixpoint Iteration =====
    println!("--- Example 9: Fixpoint Rewriting ---");
    let w = tm.mk_var("w", tm.sorts.int_sort);

    // ((w + 0) + 0) + 0 -> w (needs multiple passes)
    let w_plus_0 = tm.mk_add(vec![w, zero]);
    let w_plus_0_plus_0 = tm.mk_add(vec![w_plus_0, zero]);
    let nested = tm.mk_add(vec![w_plus_0_plus_0, zero]);

    println!("Before: ((w + 0) + 0) + 0 = {:?}", nested);

    let mut fixpoint = IteratingRewriter::new(ArithRewriter::new(), 10);
    let mut fresh_ctx = RewriteContext::new();
    let rewritten13 = fixpoint.rewrite(nested, &mut fresh_ctx, &mut tm);
    println!("After:  {:?}", rewritten13.term());
    println!("Was rewritten: {}", rewritten13.was_rewritten());
    println!("Iterations: {}", fresh_ctx.stats().iterations);
    println!("Expected: w (after iterations)\n");

    // ===== Example 10: Rewrite Statistics =====
    println!("--- Example 10: Rewrite Statistics ---");
    println!("Rewrite statistics from context:");
    println!("  Terms visited: {}", ctx.stats().terms_visited);
    println!("  Rewrites applied: {}", ctx.stats().rewrites_applied);
    println!("  Cache hits: {}", ctx.stats().cache_hits);
    println!("  Cache misses: {}", ctx.stats().cache_misses);
    println!(
        "  Cache hit rate: {:.1}%",
        ctx.stats().cache_hit_rate() * 100.0
    );

    // ===== Example 11: Rewrite Configuration =====
    println!("\n--- Example 11: Rewrite Configuration ---");
    let config = RewriteConfig {
        max_iterations: 100,
        enable_cache: true,
        max_cache_size: 10_000,
        aggressive: false,
        sort_args: true,
        flatten: true,
        ..Default::default()
    };

    println!("RewriteConfig:");
    println!("  Strategy: {:?}", config.strategy);
    println!("  Max iterations: {}", config.max_iterations);
    println!("  Enable cache: {}", config.enable_cache);
    println!("  Max cache size: {}", config.max_cache_size);
    println!("  Aggressive: {}", config.aggressive);
    println!("  Sort args: {}", config.sort_args);
    println!("  Flatten: {}", config.flatten);

    let _configured_ctx = RewriteContext::with_config(config);
    println!("\nContext created with custom config");

    // ===== Example 12: RewriteResult API =====
    println!("\n--- Example 12: RewriteResult API ---");
    let term_a = tm.mk_var("a", tm.sorts.int_sort);
    let term_b = tm.mk_var("b", tm.sorts.int_sort);

    let unchanged: RewriteResult = RewriteResult::Unchanged(term_a);
    let rewritten_result: RewriteResult = RewriteResult::Rewritten(term_b);

    println!("RewriteResult::Unchanged:");
    println!("  term(): {:?}", unchanged.term());
    println!("  was_rewritten(): {}", unchanged.was_rewritten());

    println!("\nRewriteResult::Rewritten:");
    println!("  term(): {:?}", rewritten_result.term());
    println!("  was_rewritten(): {}", rewritten_result.was_rewritten());

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Rewriters perform local term transformations");
    println!("  2. Bottom-up traversal applies rules from leaves to root");
    println!("  3. Composite rewriters combine multiple strategies");
    println!("  4. Fixpoint iteration applies rewrites until convergence");
    println!("  5. Statistics help measure rewriting effectiveness");
    println!("  6. RewriteContext handles caching and depth tracking");
}
