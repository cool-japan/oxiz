//! Debug propagation test

use oxiz_sat::{Lit, Solver, SolverResult, Var};

fn encode_xor(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

fn encode_and(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::neg(out), Lit::pos(a)]);
    sat.add_clause([Lit::neg(out), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::neg(b)]);
}

fn encode_or(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::pos(out), Lit::neg(a)]);
    sat.add_clause([Lit::pos(out), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
}

fn encode_full_adder(sat: &mut Solver, sum: Var, cout: Var, a: Var, b: Var, cin: Var) {
    let xor_ab = sat.new_var();
    encode_xor(sat, xor_ab, a, b);
    encode_xor(sat, sum, xor_ab, cin);

    let and_ab = sat.new_var();
    encode_and(sat, and_ab, a, b);

    let and_cin_xor = sat.new_var();
    encode_and(sat, and_cin_xor, cin, xor_ab);

    encode_or(sat, cout, and_ab, and_cin_xor);
}

/// Debug version with output
#[test]
fn test_3bit_with_solve_between_debug() {
    use oxiz_sat::SolverConfig;

    // Disable hyper-binary resolution to isolate the issue
    let config = SolverConfig {
        enable_lazy_hyper_binary: false,
        ..Default::default()
    };
    let mut sat = Solver::with_config(config);

    let width = 3;

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]);

    for i in 0..width {
        let cout = sat.new_var();
        encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carry);
        carry = cout;
    }

    // Add sum constraints ONE AT A TIME with solve() between
    println!("Adding sum[0]=1...");
    sat.add_clause([Lit::pos(sum[0])]);
    let r1 = sat.solve();
    println!("Result 1: {:?}", r1);
    println!(
        "Stats: conflicts={}, learned={}",
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );
    sat.debug_print_learned_clauses();
    assert_eq!(r1, SolverResult::Sat);

    println!("\nAdding sum[1]=0...");
    sat.add_clause([Lit::neg(sum[1])]);
    let r2 = sat.solve();
    println!("Result 2: {:?}", r2);
    println!(
        "Stats: conflicts={}, learned={}",
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );
    sat.debug_print_learned_clauses();
    assert_eq!(r2, SolverResult::Sat);

    println!("\nAdding sum[2]=1...");
    sat.add_clause([Lit::pos(sum[2])]);
    let r3 = sat.solve();
    println!("Result 3: {:?}", r3);
    println!(
        "Stats: conflicts={}, learned={}",
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );
    sat.debug_print_learned_clauses();
    assert_eq!(r3, SolverResult::Sat);

    println!("\nAdding a[0]=0...");
    sat.add_clause([Lit::neg(a[0])]);
    println!("Trail after add: {:?}", sat.trail().assignments());

    let r4 = sat.solve();
    println!("Result 4: {:?}", r4);
    println!(
        "Stats: conflicts={}, learned={}",
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );
    sat.debug_print_learned_clauses();

    if r4 == SolverResult::Unsat {
        println!("UNSAT - this should be SAT!");

        // Fresh solver comparison
        println!("\n=== Fresh solver ===");
        let mut fresh = Solver::new();
        let width = 3;

        let a: Vec<Var> = (0..width).map(|_| fresh.new_var()).collect();
        let b: Vec<Var> = (0..width).map(|_| fresh.new_var()).collect();
        let sum: Vec<Var> = (0..width).map(|_| fresh.new_var()).collect();

        let mut carry = fresh.new_var();
        fresh.add_clause([Lit::neg(carry)]);

        for i in 0..width {
            let cout = fresh.new_var();
            encode_full_adder(&mut fresh, sum[i], cout, a[i], b[i], carry);
            carry = cout;
        }

        fresh.add_clause([Lit::pos(sum[0])]);
        fresh.add_clause([Lit::neg(sum[1])]);
        fresh.add_clause([Lit::pos(sum[2])]);
        fresh.add_clause([Lit::neg(a[0])]);

        let rf = fresh.solve();
        println!("Fresh result: {:?}", rf);
    }

    assert_eq!(r4, SolverResult::Sat, "Should be SAT");
}
