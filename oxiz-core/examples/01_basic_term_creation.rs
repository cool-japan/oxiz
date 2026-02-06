//! # Basic Term Creation Example
//!
//! This example demonstrates how to create and manipulate terms using the OxiZ term manager.
//! It covers:
//! - Creating boolean, integer, and bitvector terms
//! - Building compound expressions (AND, OR, arithmetic)
//! - Type checking and sort system
//! - Variable declaration
//!
//! ## Complexity
//! - Time: O(1) per term creation (hash consing)
//! - Space: O(n) where n is the number of unique terms
//!
//! ## See Also
//! - [`TermManager`](oxiz_core::ast::TermManager) for the main API
//! - [`TermKind`](oxiz_core::ast::TermKind) for term types

use oxiz_core::ast::{TermKind, TermManager};

fn main() {
    println!("=== OxiZ Core: Basic Term Creation ===\n");

    // Create a term manager - the central data structure for all terms
    let mut tm = TermManager::new();
    println!("Created term manager with built-in sorts:");
    println!("  - Bool: {:?}", tm.sorts.bool_sort);
    println!("  - Int: {:?}", tm.sorts.int_sort);
    println!("  - Real: {:?}\n", tm.sorts.real_sort);

    // ===== Boolean Terms =====
    println!("--- Boolean Terms ---");
    let p = tm.mk_var("p", tm.sorts.bool_sort);
    let q = tm.mk_var("q", tm.sorts.bool_sort);
    let r = tm.mk_var("r", tm.sorts.bool_sort);

    println!("Created variables: p={:?}, q={:?}, r={:?}", p, q, r);

    // Boolean operations
    let and_pq = tm.mk_and(vec![p, q]);
    let or_pq = tm.mk_or(vec![p, q]);
    let not_p = tm.mk_not(p);
    let implies_pq = tm.mk_implies(p, q);
    let eq_pq = tm.mk_eq(p, q); // Equivalence (iff)

    println!("Boolean operations:");
    println!("  p AND q: {:?}", and_pq);
    println!("  p OR q: {:?}", or_pq);
    println!("  NOT p: {:?}", not_p);
    println!("  p => q: {:?}", implies_pq);
    println!("  p <=> q: {:?}\n", eq_pq);

    // Complex boolean expression: (p AND q) OR (NOT r)
    let not_r = tm.mk_not(r);
    let complex_bool = tm.mk_or(vec![and_pq, not_r]);
    println!("Complex: (p AND q) OR (NOT r) = {:?}\n", complex_bool);

    // ===== Integer Arithmetic =====
    println!("--- Integer Arithmetic ---");
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);
    let z = tm.mk_var("z", tm.sorts.int_sort);

    println!("Integer variables: x={:?}, y={:?}, z={:?}", x, y, z);

    // Integer constants
    let zero = tm.mk_int(0);
    let five = tm.mk_int(5);
    let ten = tm.mk_int(10);
    let neg_three = tm.mk_int(-3);

    println!(
        "Constants: 0={:?}, 5={:?}, 10={:?}, -3={:?}",
        zero, five, ten, neg_three
    );

    // Arithmetic operations
    let x_plus_y = tm.mk_add(vec![x, y]);
    let x_minus_y = tm.mk_sub(x, y);
    let x_times_5 = tm.mk_mul(vec![x, five]);
    let two = tm.mk_int(2);
    let three = tm.mk_int(3);
    let x_div_2 = tm.mk_div(x, two);
    let x_mod_3 = tm.mk_mod(x, three);

    println!("Arithmetic:");
    println!("  x + y: {:?}", x_plus_y);
    println!("  x - y: {:?}", x_minus_y);
    println!("  x * 5: {:?}", x_times_5);
    println!("  x / 2: {:?}", x_div_2);
    println!("  x mod 3: {:?}\n", x_mod_3);

    // Comparisons
    let x_eq_5 = tm.mk_eq(x, five);
    let x_lt_10 = tm.mk_lt(x, ten);
    let x_le_10 = tm.mk_le(x, ten);
    let x_gt_0 = tm.mk_gt(x, zero);
    let x_ge_0 = tm.mk_ge(x, zero);

    println!("Comparisons:");
    println!("  x = 5: {:?}", x_eq_5);
    println!("  x < 10: {:?}", x_lt_10);
    println!("  x <= 10: {:?}", x_le_10);
    println!("  x > 0: {:?}", x_gt_0);
    println!("  x >= 0: {:?}\n", x_ge_0);

    // Complex arithmetic constraint: (x + y >= 10) AND (x - y <= 5)
    let constraint1 = tm.mk_ge(x_plus_y, ten);
    let constraint2 = tm.mk_le(x_minus_y, five);
    let complex_arith = tm.mk_and(vec![constraint1, constraint2]);
    println!(
        "Complex: (x + y >= 10) AND (x - y <= 5) = {:?}\n",
        complex_arith
    );

    // ===== Bitvector Terms =====
    println!("--- Bitvector Terms ---");
    let bv8_sort = tm.sorts.bitvec(8); // 8-bit bitvector
    let bv32_sort = tm.sorts.bitvec(32); // 32-bit bitvector

    let bv_a = tm.mk_var("a", bv8_sort);
    let bv_b = tm.mk_var("b", bv8_sort);
    let bv_c = tm.mk_var("c", bv32_sort);

    println!(
        "Bitvector variables: a={:?} (8-bit), b={:?} (8-bit), c={:?} (32-bit)",
        bv_a, bv_b, bv_c
    );

    // Bitvector constants
    let bv_const = tm.mk_bitvec(42u64, 8);
    println!("Bitvector constant: 42={:?} (8-bit)\n", bv_const);

    // Bitvector operations
    let bv_and = tm.mk_bv_and(bv_a, bv_b);
    let bv_or = tm.mk_bv_or(bv_a, bv_b);
    let bv_xor = tm.mk_bv_xor(bv_a, bv_b);
    let bv_add = tm.mk_bv_add(bv_a, bv_b);

    println!("Bitvector operations:");
    println!("  a AND b: {:?}", bv_and);
    println!("  a OR b: {:?}", bv_or);
    println!("  a XOR b: {:?}", bv_xor);
    println!("  a + b: {:?}\n", bv_add);

    // ===== Hash Consing Demonstration =====
    println!("--- Hash Consing (Term Sharing) ---");
    let p1 = tm.mk_var("shared", tm.sorts.bool_sort);
    let p2 = tm.mk_var("shared", tm.sorts.bool_sort);

    println!("Creating variable 'shared' twice:");
    println!("  First:  {:?}", p1);
    println!("  Second: {:?}", p2);
    println!("  Are they identical? {}", p1 == p2);
    println!("  (Hash consing ensures structural sharing)\n");

    // ===== Term Inspection =====
    println!("--- Term Inspection ---");
    if let Some(term) = tm.get(and_pq) {
        println!("Inspecting term {:?}:", and_pq);
        println!("  Kind: {:?}", term.kind);
        println!("  Sort: {:?}", term.sort);
        if let TermKind::And(children) = &term.kind {
            println!("  Children: {:?}", children);
        }
    }

    // ===== ITE (If-Then-Else) =====
    println!("\n--- ITE (If-Then-Else) ---");
    let condition = p;
    let then_val = tm.mk_int(100);
    let else_val = tm.mk_int(0);
    let ite = tm.mk_ite(condition, then_val, else_val);

    println!("ite(p, 100, 0) = {:?}", ite);

    // ===== Distinct =====
    println!("\n--- Distinct ---");
    let distinct = tm.mk_distinct(vec![x, y, z]);
    println!("distinct(x, y, z) = {:?}", distinct);

    // ===== Statistics =====
    println!("\n--- Term Manager Statistics ---");
    println!("Total terms created: {}", tm.term_count());
    println!("  (Due to hash consing, identical terms are shared)");

    println!("\n=== Example Complete ===");
}
