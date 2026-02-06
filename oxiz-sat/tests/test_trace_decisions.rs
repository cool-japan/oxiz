//! Trace decisions during incremental solving

use oxiz_sat::{Lit, Solver, SolverResult};

/// Compare two scenarios: fresh vs incremental
#[test]
fn test_compare_scenarios() {
    println!("=== FRESH SOLVER ===");
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();

        sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
        sat.add_clause([Lit::pos(x)]);

        let r = sat.solve();
        println!("Fresh (x=T only): {:?}", r);
        if r == SolverResult::Sat {
            let m = sat.model();
            println!(
                "  x={}, a={}, b={}",
                m[x.index()].is_true(),
                m[a.index()].is_true(),
                m[b.index()].is_true()
            );
        }
    }

    println!("\n=== INCREMENTAL SOLVER ===");
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();

        sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);

        let r1 = sat.solve();
        println!("Incremental 1 (clause only): {:?}", r1);
        if r1 == SolverResult::Sat {
            let m = sat.model();
            println!(
                "  x={}, a={}, b={}",
                m[x.index()].is_true(),
                m[a.index()].is_true(),
                m[b.index()].is_true()
            );
        }

        sat.add_clause([Lit::pos(x)]);

        let r2 = sat.solve();
        println!("Incremental 2 (after x=T): {:?}", r2);
        if r2 == SolverResult::Sat {
            let m = sat.model();
            let xv = m[x.index()].is_true();
            let av = m[a.index()].is_true();
            let bv = m[b.index()].is_true();
            println!("  x={}, a={}, b={}", xv, av, bv);

            // Check clause satisfaction
            let satisfied = !xv || av || bv;
            println!(
                "  Clause (~x|a|b) = !{} || {} || {} = {}",
                xv, av, bv, satisfied
            );
            assert!(satisfied, "Clause must be satisfied!");
        }
    }
}

/// What if we don't solve in between?
#[test]
fn test_no_intermediate_solve() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    // Don't solve yet!
    sat.add_clause([Lit::pos(x)]);

    let r = sat.solve();
    println!("No intermediate solve: {:?}", r);
    if r == SolverResult::Sat {
        let m = sat.model();
        println!(
            "  x={}, a={}, b={}",
            m[x.index()].is_true(),
            m[a.index()].is_true(),
            m[b.index()].is_true()
        );

        let xv = m[x.index()].is_true();
        let av = m[a.index()].is_true();
        let bv = m[b.index()].is_true();
        let satisfied = !xv || av || bv;
        assert!(satisfied, "Clause must be satisfied!");
    }
}

/// Test with push/pop instead of raw incremental
#[test]
fn test_with_push_pop() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);

    // Use push/pop
    sat.push();

    let r1 = sat.solve();
    println!("With push/pop solve 1: {:?}", r1);
    if r1 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "  x={}, a={}, b={}",
            m[x.index()].is_true(),
            m[a.index()].is_true(),
            m[b.index()].is_true()
        );
    }

    sat.pop();
    sat.push();
    sat.add_clause([Lit::pos(x)]);

    let r2 = sat.solve();
    println!("With push/pop solve 2: {:?}", r2);
    if r2 == SolverResult::Sat {
        let m = sat.model();
        let xv = m[x.index()].is_true();
        let av = m[a.index()].is_true();
        let bv = m[b.index()].is_true();
        println!("  x={}, a={}, b={}", xv, av, bv);

        let satisfied = !xv || av || bv;
        assert!(satisfied, "Clause must be satisfied!");
    }
}
