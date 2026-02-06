//! Test adder + ULT together at SAT level

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

/// Encode ULT helper: out = ~a & b
fn encode_and_not_a(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a)]);
    sat.add_clause([Lit::neg(out), Lit::pos(b)]);
    sat.add_clause([Lit::pos(a), Lit::neg(b), Lit::pos(out)]);
}

/// Encode XNOR: out = (a <=> b)
fn encode_xnor(sat: &mut Solver, out: Var, a: Var, b: Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::neg(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::pos(b)]);
}

/// Encode ULT: result = (a < b)
fn encode_ult(sat: &mut Solver, result: Var, a_bits: &[Var], b_bits: &[Var]) {
    let width = a_bits.len();
    if width == 0 {
        sat.add_clause([Lit::neg(result)]);
        return;
    }

    // lt_0 = ~a[0] & b[0]
    let mut lt_prev = sat.new_var();
    encode_and_not_a(sat, lt_prev, a_bits[0], b_bits[0]);

    for i in 1..width {
        let ai = a_bits[i];
        let bi = b_bits[i];

        // lt_at_i = ~ai & bi
        let lt_at_i = sat.new_var();
        encode_and_not_a(sat, lt_at_i, ai, bi);

        // eq_i = (ai <=> bi)
        let eq_i = sat.new_var();
        encode_xnor(sat, eq_i, ai, bi);

        // carry_prev = eq_i & lt_prev
        let carry_prev = sat.new_var();
        encode_and(sat, carry_prev, eq_i, lt_prev);

        // lt_next = lt_at_i | carry_prev
        let lt_next = sat.new_var();
        encode_or(sat, lt_next, lt_at_i, carry_prev);

        lt_prev = lt_next;
    }

    // result <=> lt_prev
    sat.add_clause([Lit::neg(result), Lit::pos(lt_prev)]);
    sat.add_clause([Lit::pos(result), Lit::neg(lt_prev)]);
}

/// Test: adder + ult at 4 bits
/// a + b = sum, sum = 10, a < b
#[test]
fn test_adder_ult_4bit() {
    let mut sat = Solver::new();
    let width = 4;

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

    // Encode adder: sum = a + b
    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]);

    for i in 0..width {
        let cout = sat.new_var();
        encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carry);
        carry = cout;
    }
    println!("After adder encoding");

    // Constrain sum = 10 (1010)
    sat.add_clause([Lit::neg(sum[0])]); // 0
    sat.add_clause([Lit::pos(sum[1])]); // 1
    sat.add_clause([Lit::neg(sum[2])]); // 0
    sat.add_clause([Lit::pos(sum[3])]); // 1
    println!("After sum = 10 constraint");

    // Encode ULT: a < b
    let ult_result = sat.new_var();
    encode_ult(&mut sat, ult_result, &a, &b);
    sat.add_clause([Lit::pos(ult_result)]);
    println!("After ULT encoding");

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
            let ult_val = model[ult_result.index()].is_true();

            println!("SAT:");
            println!("  a = {} (binary: {:04b})", a_val, a_val);
            println!("  b = {} (binary: {:04b})", b_val, b_val);
            println!("  sum = {} (binary: {:04b})", sum_val, sum_val);
            println!("  ult_result = {}", ult_val);
            println!("  a + b = {} (mod 16)", (a_val + b_val) % 16);
            println!("  a < b = {}", a_val < b_val);

            assert_eq!(sum_val, 10, "sum should be 10");
            assert!(a_val < b_val, "a {} should be < b {}", a_val, b_val);
            assert_eq!((a_val + b_val) % 16, sum_val, "a + b should equal sum");
        }
        SolverResult::Unsat => {
            println!("UNSAT");
            panic!("Should be SAT - valid solutions: a=0,b=10; a=1,b=9; etc.");
        }
        _ => panic!("Unknown"),
    }
}

/// Test: adder + ult at 2 bits
#[test]
fn test_adder_ult_2bit() {
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

    // Encode ULT: a < b
    let ult_result = sat.new_var();
    encode_ult(&mut sat, ult_result, &a, &b);
    sat.add_clause([Lit::pos(ult_result)]);

    // Solutions: a + b = 2 with a < b
    // a=0, b=2 (0 < 2, 0+2=2) ✓

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

            println!("2-BIT ADDER+ULT:");
            println!("  a = {}, b = {}, sum = {}", a_val, b_val, sum_val);
            println!("  a < b = {}", a_val < b_val);
            println!("  a + b = {} (mod 4)", (a_val + b_val) % 4);

            assert_eq!(sum_val, 2);
            assert!(a_val < b_val);
            assert_eq!((a_val + b_val) % 4, sum_val);
        }
        SolverResult::Unsat => {
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}

/// Test: adder + ult at 3 bits
#[test]
fn test_adder_ult_3bit() {
    let mut sat = Solver::new();
    let width = 3;

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

    // Constrain sum = 5 (101)
    sat.add_clause([Lit::pos(sum[0])]); // 1
    sat.add_clause([Lit::neg(sum[1])]); // 0
    sat.add_clause([Lit::pos(sum[2])]); // 1

    // Encode ULT: a < b
    let ult_result = sat.new_var();
    encode_ult(&mut sat, ult_result, &a, &b);
    sat.add_clause([Lit::pos(ult_result)]);

    // Solutions: a + b = 5 with a < b
    // a=0, b=5; a=1, b=4; a=2, b=3

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

            println!("3-BIT ADDER+ULT:");
            println!("  a = {}, b = {}, sum = {}", a_val, b_val, sum_val);
            println!("  a < b = {}", a_val < b_val);
            println!("  a + b = {} (mod 8)", (a_val + b_val) % 8);

            assert_eq!(sum_val, 5);
            assert!(a_val < b_val);
            assert_eq!((a_val + b_val) % 8, sum_val);
        }
        SolverResult::Unsat => {
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}
