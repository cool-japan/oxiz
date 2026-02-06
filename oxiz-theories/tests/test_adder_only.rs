//! Test just the adder encoding

use oxiz_sat::{Lit, Solver, SolverResult, Var};

/// Encode XOR: out = a XOR b
fn encode_xor(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Encode AND: out = a AND b
fn encode_and(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::neg(out), Lit::pos(a)]);
    sat.add_clause([Lit::neg(out), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::neg(b)]);
}

/// Encode OR: out = a OR b
fn encode_or(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::pos(out), Lit::neg(a)]);
    sat.add_clause([Lit::pos(out), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
}

/// Encode full adder
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

/// Test: 4-bit adder alone
/// a + b = sum, constrain sum = 10, find a, b
#[test]
fn test_adder_4bit() {
    let mut sat = Solver::new();
    let width = 4;

    // Create variables
    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    println!(
        "a = vars {:?}",
        a.iter().map(|v| v.index()).collect::<Vec<_>>()
    );
    println!(
        "b = vars {:?}",
        b.iter().map(|v| v.index()).collect::<Vec<_>>()
    );
    println!(
        "sum = vars {:?}",
        sum.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    // Encode ripple-carry adder
    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]); // Initial carry = 0
    println!("initial carry = var {}", carry.index());

    for i in 0..width {
        let cout = sat.new_var();
        println!(
            "Encoding bit {}: sum[{}]=var {}, cout=var {}, a=var {}, b=var {}, cin=var {}",
            i,
            i,
            sum[i].index(),
            cout.index(),
            a[i].index(),
            b[i].index(),
            carry.index()
        );
        encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carry);
        carry = cout;
    }

    // Constrain sum = 10 (1010)
    println!("Constraining sum = 10...");
    sat.add_clause([Lit::neg(sum[0])]); // 0
    sat.add_clause([Lit::pos(sum[1])]); // 1
    sat.add_clause([Lit::neg(sum[2])]); // 0
    sat.add_clause([Lit::pos(sum[3])]); // 1

    println!("Solving...");
    match sat.solve() {
        SolverResult::Sat => {
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

            println!("SAT:");
            println!("  a = {} (binary: {:04b})", a_val, a_val);
            println!("  b = {} (binary: {:04b})", b_val, b_val);
            println!("  sum = {} (binary: {:04b})", sum_val, sum_val);
            println!("  a + b = {} (mod 16)", (a_val + b_val) % 16);

            // Verify
            assert_eq!(sum_val, 10, "sum should be 10");
            assert_eq!((a_val + b_val) % 16, sum_val, "a + b should equal sum");
        }
        SolverResult::Unsat => {
            println!("UNSAT");
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}

/// Test: fixed a and b, check sum
#[test]
fn test_adder_fixed_inputs() {
    let mut sat = Solver::new();
    let width = 4;

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    // Fix a = 3 (0011)
    sat.add_clause([Lit::pos(a[0])]); // 1
    sat.add_clause([Lit::pos(a[1])]); // 1
    sat.add_clause([Lit::neg(a[2])]); // 0
    sat.add_clause([Lit::neg(a[3])]); // 0

    // Fix b = 7 (0111)
    sat.add_clause([Lit::pos(b[0])]); // 1
    sat.add_clause([Lit::pos(b[1])]); // 1
    sat.add_clause([Lit::pos(b[2])]); // 1
    sat.add_clause([Lit::neg(b[3])]); // 0

    // Encode adder
    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]);

    for i in 0..width {
        let cout = sat.new_var();
        encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carry);
        carry = cout;
    }

    // 3 + 7 = 10

    match sat.solve() {
        SolverResult::Sat => {
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

            println!("FIXED INPUTS:");
            println!("  a = {} (expected 3)", a_val);
            println!("  b = {} (expected 7)", b_val);
            println!("  sum = {} (expected 10)", sum_val);

            assert_eq!(a_val, 3);
            assert_eq!(b_val, 7);
            assert_eq!(sum_val, 10, "3 + 7 should be 10");
        }
        SolverResult::Unsat => {
            println!("UNSAT");
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}

/// Test: 2-bit adder
#[test]
fn test_adder_2bit() {
    let mut sat = Solver::new();
    let width = 2;

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    // Encode adder
    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]);

    for i in 0..width {
        let cout = sat.new_var();
        encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carry);
        carry = cout;
    }

    // Constrain sum = 2 (10)
    sat.add_clause([Lit::neg(sum[0])]); // 0
    sat.add_clause([Lit::pos(sum[1])]); // 1

    match sat.solve() {
        SolverResult::Sat => {
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

            println!("2-BIT:");
            println!("  a = {}, b = {}, sum = {}", a_val, b_val, sum_val);
            println!("  a + b = {} (mod 4)", (a_val + b_val) % 4);

            assert_eq!(sum_val, 2);
            assert_eq!((a_val + b_val) % 4, sum_val, "a + b should equal sum");
        }
        SolverResult::Unsat => {
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}
