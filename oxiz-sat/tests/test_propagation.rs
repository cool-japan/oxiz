//! Test propagation timing

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

/// Try: solve() between adding constraints to force propagation
#[test]
fn test_3bit_with_solve_between() {
    let mut sat = Solver::new();
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
    assert_eq!(sat.solve(), SolverResult::Sat);

    println!("Adding sum[1]=0...");
    sat.add_clause([Lit::neg(sum[1])]);
    assert_eq!(sat.solve(), SolverResult::Sat);

    println!("Adding sum[2]=1...");
    sat.add_clause([Lit::pos(sum[2])]);
    assert_eq!(sat.solve(), SolverResult::Sat);

    println!("Adding a[0]=0...");
    sat.add_clause([Lit::neg(a[0])]);

    match sat.solve() {
        SolverResult::Sat => {
            let a_val: u64 = a
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            let b_val: u64 = b
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            let sum_val: u64 = sum
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            println!("SAT: a={}, b={}, sum={}", a_val, b_val, sum_val);
            assert_eq!(a_val % 2, 0);
            assert_eq!((a_val + b_val) % 8, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// Try: all sum constraints at once, then solve, then add extra constraint
#[test]
fn test_3bit_sum_first_then_extra() {
    let mut sat = Solver::new();
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

    // Add all sum constraints at once
    sat.add_clause([Lit::pos(sum[0])]); // 1
    sat.add_clause([Lit::neg(sum[1])]); // 0
    sat.add_clause([Lit::pos(sum[2])]); // 1
    // sum = 101 = 5

    println!("After adder + sum=5, checking...");
    let result = sat.solve();
    println!("First solve: {:?}", result);
    assert_eq!(result, SolverResult::Sat);

    // Now add extra constraint
    println!("Adding a[0]=0...");
    sat.add_clause([Lit::neg(a[0])]);

    match sat.solve() {
        SolverResult::Sat => {
            let a_val: u64 = a
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            let b_val: u64 = b
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            let sum_val: u64 = sum
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            println!("SAT: a={}, b={}, sum={}", a_val, b_val, sum_val);
            assert_eq!(a_val % 2, 0);
            assert_eq!((a_val + b_val) % 8, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT (add extra constraint after solve)"),
        _ => panic!("Unknown"),
    }
}

/// Original failing test for comparison
#[test]
fn test_3bit_original() {
    let mut sat = Solver::new();
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

    // Add all constraints at once without intermediate solve()
    sat.add_clause([Lit::pos(sum[0])]);
    sat.add_clause([Lit::neg(sum[1])]);
    sat.add_clause([Lit::pos(sum[2])]);
    sat.add_clause([Lit::neg(a[0])]);

    // This is the failing case
    match sat.solve() {
        SolverResult::Sat => {
            let a_val: u64 = a
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            let b_val: u64 = b
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            let sum_val: u64 = sum
                .iter()
                .enumerate()
                .map(|(i, &v)| (sat.model()[v.index()].is_true() as u64) << i)
                .sum();
            println!("SAT: a={}, b={}, sum={}", a_val, b_val, sum_val);
            assert_eq!(a_val % 2, 0);
            assert_eq!((a_val + b_val) % 8, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT (original failing case)"),
        _ => panic!("Unknown"),
    }
}
