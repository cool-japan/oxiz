//! Trace the bug step by step

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

/// Trace 3-bit case
#[test]
fn test_trace_3bit() {
    let mut sat = Solver::new();
    let width = 3;

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    println!("Variables:");
    println!(
        "  a = {:?}",
        a.iter().map(|v| v.index()).collect::<Vec<_>>()
    );
    println!(
        "  b = {:?}",
        b.iter().map(|v| v.index()).collect::<Vec<_>>()
    );
    println!(
        "  sum = {:?}",
        sum.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]);

    for i in 0..width {
        let cout = sat.new_var();
        encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carry);
        carry = cout;
    }

    // Add sum constraints
    sat.add_clause([Lit::pos(sum[0])]); // sum[0] = 1
    sat.add_clause([Lit::neg(sum[1])]); // sum[1] = 0
    sat.add_clause([Lit::pos(sum[2])]); // sum[2] = 1
    // sum = 101 = 5

    println!("\nFirst solve...");
    let result = sat.solve();
    println!("Result: {:?}", result);

    if result == SolverResult::Sat {
        let model = sat.model();
        println!("\nFirst model:");
        for i in 0..width {
            println!("  a[{}] = {}", i, model[a[i].index()].is_true());
            println!("  b[{}] = {}", i, model[b[i].index()].is_true());
            println!("  sum[{}] = {}", i, model[sum[i].index()].is_true());
        }
        let a_val: u64 = a
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        let b_val: u64 = b
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        let sum_val: u64 = sum
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        println!(
            "  a = {}, b = {}, sum = {}, a+b = {}",
            a_val,
            b_val,
            sum_val,
            (a_val + b_val) % 8
        );
    }

    // Now add a[0] = 0
    println!("\nAdding a[0] = 0...");
    sat.add_clause([Lit::neg(a[0])]);

    println!("\nSecond solve...");
    let result2 = sat.solve();
    println!("Result: {:?}", result2);

    if result2 == SolverResult::Sat {
        let model = sat.model();
        println!("\nSecond model:");
        for i in 0..width {
            println!("  a[{}] = {}", i, model[a[i].index()].is_true());
            println!("  b[{}] = {}", i, model[b[i].index()].is_true());
            println!("  sum[{}] = {}", i, model[sum[i].index()].is_true());
        }
        let a_val: u64 = a
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        let b_val: u64 = b
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        let sum_val: u64 = sum
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        println!(
            "  a = {}, b = {}, sum = {}, a+b = {}",
            a_val,
            b_val,
            sum_val,
            (a_val + b_val) % 8
        );

        // The issue: a=0, b=1 should give a+b=1, but sum=5
        // Verify each bit:
        for i in 0..width {
            let expected_sum_bit = ((a_val + b_val) >> i) & 1;
            let actual_sum_bit = model[sum[i].index()].is_true() as u64;
            println!(
                "  Bit {}: expected={}, actual={}, match={}",
                i,
                expected_sum_bit,
                actual_sum_bit,
                expected_sum_bit == actual_sum_bit
            );
        }
    }

    // Verify the model
    if result2 == SolverResult::Sat {
        let model = sat.model();
        let a_val: u64 = a
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        let b_val: u64 = b
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        let sum_val: u64 = sum
            .iter()
            .enumerate()
            .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
            .sum();
        assert_eq!(
            (a_val + b_val) % 8,
            sum_val,
            "Model does not satisfy addition constraint!"
        );
    }
}
