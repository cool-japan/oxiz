//! SMT-LIB2 script execution example
//!
//! This example demonstrates executing SMT-LIB2 scripts using the Context API.
//! SMT-LIB2 is the standard input language for SMT solvers.

use oxiz_solver::Context;

fn main() {
    println!("=== SMT-LIB2 Script Execution Example ===\n");

    // Example 1: Simple satisfiability check
    println!("Example 1: Simple SAT check\n");
    {
        let script = r#"
(set-logic QF_UF)
(declare-const p Bool)
(declare-const q Bool)
(assert (and p q))
(check-sat)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 2: Integer arithmetic
    println!("Example 2: Integer arithmetic\n");
    {
        let script = r#"
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (>= x 5))
(assert (<= x 10))
(assert (= y (* 2 x)))
(check-sat)
(get-model)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 3: Push/pop with incremental solving
    println!("Example 3: Incremental solving with push/pop\n");
    {
        let script = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 0))
(check-sat)
(push 1)
(assert (<= x 10))
(check-sat)
(push 1)
(assert (= x 5))
(check-sat)
(pop 1)
(check-sat)
(pop 1)
(check-sat)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 4: Unsatisfiable formula with unsat core
    println!("Example 4: Unsatisfiable formula\n");
    {
        let script = r#"
(set-logic QF_LIA)
(set-option :produce-unsat-cores true)
(declare-const x Int)
(assert (! (> x 10) :named a1))
(assert (! (< x 5) :named a2))
(check-sat)
(get-unsat-core)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 5: Get value of terms
    println!("Example 5: Get value of terms\n");
    {
        let script = r#"
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (= x 42))
(assert (= y (+ x 8)))
(check-sat)
(get-value (x y (+ x y)))
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 6: Boolean logic with let bindings
    println!("Example 6: Boolean logic with let bindings\n");
    {
        let script = r#"
(set-logic QF_UF)
(declare-const p Bool)
(declare-const q Bool)
(assert (let ((r (and p q))) (or r (not p))))
(check-sat)
(get-model)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 7: Distinct constraint
    println!("Example 7: Distinct constraint\n");
    {
        let script = r#"
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(assert (distinct x y z))
(assert (>= x 0))
(assert (<= x 2))
(assert (>= y 0))
(assert (<= y 2))
(assert (>= z 0))
(assert (<= z 2))
(check-sat)
(get-model)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 8: Real arithmetic
    println!("Example 8: Real arithmetic\n");
    {
        let script = r#"
(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)
(assert (>= x 0.5))
(assert (<= x 1.5))
(assert (= y (+ x 0.5)))
(check-sat)
(get-value (x y))
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 9: Formula simplification
    println!("Example 9: Formula simplification\n");
    {
        let script = r#"
(set-logic QF_UF)
(declare-const p Bool)
(simplify (and p true))
(simplify (or p false))
(simplify (not (not p)))
(simplify (and true true))
(simplify (or false false))
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 10: Get assertions
    println!("Example 10: Get assertions\n");
    {
        let script = r#"
(set-logic QF_LIA)
(set-option :produce-assertions true)
(declare-const x Int)
(declare-const y Int)
(assert (>= x 0))
(assert (<= y 10))
(assert (= (+ x y) 15))
(get-assertions)
(check-sat)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!("\n{}\n", "=".repeat(60));

    // Example 11: Named assertions and check-sat-assuming
    println!("Example 11: Named assertions\n");
    {
        let script = r#"
(set-logic QF_LIA)
(declare-const x Int)
(assert (! (>= x 0) :named pos))
(assert (! (<= x 100) :named bounded))
(assert (! (= x 50) :named exact))
(check-sat)
"#;

        let mut ctx = Context::new();
        println!("Script:");
        println!("{}", script);
        println!("Output:");
        let _ = ctx.execute_script(script);
    }

    println!();
}
