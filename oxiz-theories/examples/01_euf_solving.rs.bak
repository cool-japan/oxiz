//! # Equality with Uninterpreted Functions (EUF) Example
//!
//! This example demonstrates the EUF theory solver.
//! It covers:
//! - Congruence closure algorithm
//! - Equality propagation
//! - Disequality constraints
//! - Function applications
//! - Transitive equality reasoning
//!
//! ## EUF Theory
//! The theory of equality with uninterpreted functions reasons about
//! equality between terms with uninterpreted function symbols.
//!
//! ## Core Axioms
//! 1. Reflexivity: x = x
//! 2. Symmetry: x = y → y = x
//! 3. Transitivity: x = y ∧ y = z → x = z
//! 4. Congruence: x = y → f(x) = f(y)
//!
//! ## Complexity
//! - Congruence closure: O(n log n) (nearly linear)
//! - Query: O(α(n)) (inverse Ackermann, effectively O(1))
//!
//! ## See Also
//! - [`EUFSolver`](oxiz_theories::euf::EUFSolver)
//! - [`CongruenceClosure`](oxiz_theories::euf::CongruenceClosure)

use oxiz_core::ast::TermManager;
use oxiz_solver::{Solver, SolverResult};

fn main() {
    println!("=== EUF Theory Solver Examples ===\n");

    // ===== Example 1: Basic Equality =====
    println!("--- Example 1: Basic Equality Reasoning ---");

    let mut solver = Solver::new();
    solver.set_logic("QF_UF");
    let mut tm = TermManager::new();

    // Variables
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);
    let z = tm.mk_var("z", tm.sorts.int_sort);

    // Assertions:
    // x = y
    // y = z
    // Query: x = z? (should be true by transitivity)

    solver.assert(tm.mk_eq(x, y), &mut tm);
    solver.assert(tm.mk_eq(y, z), &mut tm);

    println!("Assertions: x = y, y = z");

    // Check if x = z is implied
    solver.push();
    solver.assert(tm.mk_not(tm.mk_eq(x, z)), &mut tm);
    println!("Query: Is x = z implied?");

    match solver.check(&mut tm) {
        SolverResult::Unsat => {
            println!("Result: YES (x = z is implied by transitivity)\n");
        }
        SolverResult::Sat => {
            println!("Result: NO (unexpected)\n");
        }
        SolverResult::Unknown => {
            println!("Result: UNKNOWN\n");
        }
    }
    solver.pop();

    // ===== Example 2: Congruence =====
    println!("--- Example 2: Congruence Reasoning ---");

    let mut solver2 = Solver::new();
    solver2.set_logic("QF_UF");
    let mut tm2 = TermManager::new();

    // Uninterpreted function f
    let f = tm2.mk_func("f", vec![tm2.sorts.int_sort], tm2.sorts.int_sort);

    let a = tm2.mk_var("a", tm2.sorts.int_sort);
    let b = tm2.mk_var("b", tm2.sorts.int_sort);

    // f(a) and f(b)
    let fa = tm2.mk_app(f, vec![a]);
    let fb = tm2.mk_app(f, vec![b]);

    // Assertions:
    // a = b
    // f(a) ≠ f(b) (should be unsatisfiable by congruence)

    println!("Assertions: a = b, f(a) ≠ f(b)");

    solver2.assert(tm2.mk_eq(a, b), &mut tm2);
    solver2.assert(tm2.mk_not(tm2.mk_eq(fa, fb)), &mut tm2);

    match solver2.check(&mut tm2) {
        SolverResult::Unsat => {
            println!("Result: UNSAT (congruence axiom: a = b → f(a) = f(b))\n");
        }
        _ => println!("Result: SAT or UNKNOWN (unexpected)\n"),
    }

    // ===== Example 3: Multi-Argument Functions =====
    println!("--- Example 3: Multi-Argument Function Congruence ---");

    let mut solver3 = Solver::new();
    solver3.set_logic("QF_UF");
    let mut tm3 = TermManager::new();

    // Binary function g
    let g = tm3.mk_func(
        "g",
        vec![tm3.sorts.int_sort, tm3.sorts.int_sort],
        tm3.sorts.int_sort,
    );

    let x3 = tm3.mk_var("x", tm3.sorts.int_sort);
    let y3 = tm3.mk_var("y", tm3.sorts.int_sort);
    let u = tm3.mk_var("u", tm3.sorts.int_sort);
    let v = tm3.mk_var("v", tm3.sorts.int_sort);

    let gxy = tm3.mk_app(g, vec![x3, y3]);
    let guv = tm3.mk_app(g, vec![u, v]);

    // Assertions:
    // x = u
    // y = v
    // g(x, y) ≠ g(u, v) (unsatisfiable)

    println!("Assertions: x = u, y = v, g(x, y) ≠ g(u, v)");

    solver3.assert(tm3.mk_eq(x3, u), &mut tm3);
    solver3.assert(tm3.mk_eq(y3, v), &mut tm3);
    solver3.assert(tm3.mk_not(tm3.mk_eq(gxy, guv)), &mut tm3);

    match solver3.check(&mut tm3) {
        SolverResult::Unsat => {
            println!("Result: UNSAT (congruence for multi-argument functions)\n");
        }
        _ => println!("Result: SAT or UNKNOWN (unexpected)\n"),
    }

    // ===== Example 4: Satisfiable EUF Formula =====
    println!("--- Example 4: Satisfiable EUF Formula ---");

    let mut solver4 = Solver::new();
    solver4.set_logic("QF_UF");
    let mut tm4 = TermManager::new();

    let f4 = tm4.mk_func("f", vec![tm4.sorts.int_sort], tm4.sorts.int_sort);
    let a4 = tm4.mk_var("a", tm4.sorts.int_sort);
    let b4 = tm4.mk_var("b", tm4.sorts.int_sort);
    let c4 = tm4.mk_var("c", tm4.sorts.int_sort);

    let fa4 = tm4.mk_app(f4, vec![a4]);
    let fb4 = tm4.mk_app(f4, vec![b4]);
    let fc4 = tm4.mk_app(f4, vec![c4]);

    // Assertions:
    // f(a) = f(b)
    // f(b) = f(c)
    // a ≠ b (satisfiable: f is not injective)

    println!("Assertions: f(a) = f(b), f(b) = f(c), a ≠ b");

    solver4.assert(tm4.mk_eq(fa4, fb4), &mut tm4);
    solver4.assert(tm4.mk_eq(fb4, fc4), &mut tm4);
    solver4.assert(tm4.mk_not(tm4.mk_eq(a4, b4)), &mut tm4);

    match solver4.check(&mut tm4) {
        SolverResult::Sat => {
            println!("Result: SAT");
            if let Some(model) = solver4.get_model(&tm4) {
                println!("Model (conceptual):");
                println!("  a, b, c can be distinct");
                println!("  f maps all three to the same value");
                println!("  Example: a=1, b=2, c=3, f(1)=f(2)=f(3)=0\n");
            }
        }
        _ => println!("Result: UNSAT or UNKNOWN (unexpected)\n"),
    }

    // ===== Example 5: Nested Functions =====
    println!("--- Example 5: Nested Function Applications ---");

    let mut solver5 = Solver::new();
    solver5.set_logic("QF_UF");
    let mut tm5 = TermManager::new();

    let h = tm5.mk_func("h", vec![tm5.sorts.int_sort], tm5.sorts.int_sort);
    let p = tm5.mk_var("p", tm5.sorts.int_sort);
    let q = tm5.mk_var("q", tm5.sorts.int_sort);

    // h(h(p)) and h(h(q))
    let hp = tm5.mk_app(h, vec![p]);
    let hhp = tm5.mk_app(h, vec![hp]);
    let hq = tm5.mk_app(h, vec![q]);
    let hhq = tm5.mk_app(h, vec![hq]);

    // Assertions:
    // p = q
    // h(h(p)) ≠ h(h(q)) (unsatisfiable)

    println!("Assertions: p = q, h(h(p)) ≠ h(h(q))");

    solver5.assert(tm5.mk_eq(p, q), &mut tm5);
    solver5.assert(tm5.mk_not(tm5.mk_eq(hhp, hhq)), &mut tm5);

    match solver5.check(&mut tm5) {
        SolverResult::Unsat => {
            println!("Result: UNSAT (congruence applies to nested applications)\n");
        }
        _ => println!("Result: SAT or UNKNOWN (unexpected)\n"),
    }

    // ===== Example 6: Equivalence Classes =====
    println!("--- Example 6: Equivalence Classes ---");

    let mut solver6 = Solver::new();
    solver6.set_logic("QF_UF");
    let mut tm6 = TermManager::new();

    let w = tm6.mk_var("w", tm6.sorts.int_sort);
    let x6 = tm6.mk_var("x", tm6.sorts.int_sort);
    let y6 = tm6.mk_var("y", tm6.sorts.int_sort);
    let z6 = tm6.mk_var("z", tm6.sorts.int_sort);

    // Build equivalence classes: {w, x} and {y, z}
    solver6.assert(tm6.mk_eq(w, x6), &mut tm6);
    solver6.assert(tm6.mk_eq(y6, z6), &mut tm6);

    println!("Equivalence classes: {w, x}, {y, z}");

    // Query: w = y?
    solver6.push();
    solver6.assert(tm6.mk_eq(w, y6), &mut tm6);
    println!("After asserting w = y:");

    match solver6.check(&mut tm6) {
        SolverResult::Sat => {
            println!("Result: SAT (can merge classes)");
            println!("New class: {w, x, y, z}\n");
        }
        _ => println!("Result: UNSAT or UNKNOWN\n"),
    }
    solver6.pop();

    // ===== Example 7: Chain of Equalities =====
    println!("--- Example 7: Long Equality Chain ---");

    let mut solver7 = Solver::new();
    solver7.set_logic("QF_UF");
    let mut tm7 = TermManager::new();

    let vars: Vec<_> = (0..10)
        .map(|i| tm7.mk_var(&format!("v{}", i), tm7.sorts.int_sort))
        .collect();

    // Chain: v0 = v1 = v2 = ... = v9
    for i in 0..9 {
        solver7.assert(tm7.mk_eq(vars[i], vars[i + 1]), &mut tm7);
    }

    println!("Chain: v0 = v1 = v2 = ... = v9");

    // Query: v0 = v9?
    solver7.push();
    solver7.assert(tm7.mk_not(tm7.mk_eq(vars[0], vars[9])), &mut tm7);

    match solver7.check(&mut tm7) {
        SolverResult::Unsat => {
            println!("Result: v0 = v9 (by transitivity through chain)\n");
        }
        _ => println!("Result: SAT or UNKNOWN (unexpected)\n"),
    }
    solver7.pop();

    // ===== Example 8: Disjoint Classes =====
    println!("--- Example 8: Disjoint Equivalence Classes ---");

    let mut solver8 = Solver::new();
    solver8.set_logic("QF_UF");
    let mut tm8 = TermManager::new();

    let a8 = tm8.mk_var("a", tm8.sorts.int_sort);
    let b8 = tm8.mk_var("b", tm8.sorts.int_sort);
    let c8 = tm8.mk_var("c", tm8.sorts.int_sort);
    let d8 = tm8.mk_var("d", tm8.sorts.int_sort);

    // Two disjoint classes
    solver8.assert(tm8.mk_eq(a8, b8), &mut tm8);
    solver8.assert(tm8.mk_eq(c8, d8), &mut tm8);
    solver8.assert(tm8.mk_not(tm8.mk_eq(a8, c8)), &mut tm8);

    println!("Classes: {a, b}, {c, d}, a ≠ c");

    match solver8.check(&mut tm8) {
        SolverResult::Sat => {
            println!("Result: SAT (classes remain disjoint)\n");
        }
        _ => println!("Result: UNSAT or UNKNOWN\n"),
    }

    // ===== Example 9: Complex Formula =====
    println!("--- Example 9: Complex EUF Formula ---");

    println!("Formula:");
    println!("  f(f(f(a))) = a");
    println!("  f(f(f(f(f(a))))) = a");
    println!("  a ≠ f(a)");
    println!();
    println!("Analysis:");
    println!("  Let f(a) = b, f(b) = c, f(c) = a (3-cycle)");
    println!("  Then f(f(f(a))) = a ✓");
    println!("  And f^5(a) = f^2(a) = c ≠ a ✗");
    println!("Result: UNSAT\n");

    // ===== Example 10: EUF with Constants =====
    println!("--- Example 10: EUF with Named Constants ---");

    let mut solver10 = Solver::new();
    solver10.set_logic("QF_UF");
    let mut tm10 = TermManager::new();

    // Constants
    let zero = tm10.mk_const("zero", tm10.sorts.int_sort);
    let one = tm10.mk_const("one", tm10.sorts.int_sort);

    let f10 = tm10.mk_func("f", vec![tm10.sorts.int_sort], tm10.sorts.int_sort);
    let f_zero = tm10.mk_app(f10, vec![zero]);
    let f_one = tm10.mk_app(f10, vec![one]);

    // Assertions:
    // f(zero) = one
    // f(one) = zero
    // zero ≠ one

    solver10.assert(tm10.mk_eq(f_zero, one), &mut tm10);
    solver10.assert(tm10.mk_eq(f_one, zero), &mut tm10);
    solver10.assert(tm10.mk_not(tm10.mk_eq(zero, one)), &mut tm10);

    println!("Assertions: f(zero) = one, f(one) = zero, zero ≠ one");

    match solver10.check(&mut tm10) {
        SolverResult::Sat => {
            println!("Result: SAT (f is a swap function)\n");
        }
        _ => println!("Result: UNSAT or UNKNOWN (unexpected)\n"),
    }

    println!("=== Examples Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. EUF reasons about equality via congruence closure");
    println!("  2. Transitivity: x=y ∧ y=z → x=z");
    println!("  3. Congruence: x=y → f(x)=f(y)");
    println!("  4. Functions are uninterpreted (no axioms)");
    println!("  5. Complexity: nearly linear (union-find)");
    println!("  6. EUF forms the basis for many theory combinations");
}
