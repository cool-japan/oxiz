//! Test adder with extra constraints

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

/// Test: adder + simple extra constraint (a[0] = 0)
#[test]
fn test_adder_with_simple_constraint() {
    let mut sat = Solver::new();
    let width = 4;

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

    // Constrain sum = 10
    sat.add_clause([Lit::neg(sum[0])]);
    sat.add_clause([Lit::pos(sum[1])]);
    sat.add_clause([Lit::neg(sum[2])]);
    sat.add_clause([Lit::pos(sum[3])]);

    // Extra constraint: a[0] = 0
    sat.add_clause([Lit::neg(a[0])]);

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

            println!("a[0]=0: a={}, b={}, sum={}", a_val, b_val, sum_val);
            assert!(a_val.is_multiple_of(2), "a should be even");
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// Test: adder + multiple extra constraints
#[test]
fn test_adder_with_multiple_constraints() {
    let mut sat = Solver::new();
    let width = 4;

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

    // Constrain sum = 10
    sat.add_clause([Lit::neg(sum[0])]);
    sat.add_clause([Lit::pos(sum[1])]);
    sat.add_clause([Lit::neg(sum[2])]);
    sat.add_clause([Lit::pos(sum[3])]);

    // Multiple extra constraints
    sat.add_clause([Lit::neg(a[0])]); // a[0] = 0
    sat.add_clause([Lit::neg(a[2])]); // a[2] = 0
    sat.add_clause([Lit::pos(b[1])]); // b[1] = 1

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

            println!("multi: a={}, b={}, sum={}", a_val, b_val, sum_val);
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// Test: adder + nested XOR constraint (like ULT uses)
#[test]
fn test_adder_with_xor_constraint() {
    let mut sat = Solver::new();
    let width = 4;

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

    // Constrain sum = 11 (binary: 1011)
    // NOTE: sum must be odd for a[0] XOR b[0] = 1 to be satisfiable
    // (even sum requires both operands to have same parity)
    sat.add_clause([Lit::pos(sum[0])]); // 1
    sat.add_clause([Lit::pos(sum[1])]); // 1
    sat.add_clause([Lit::neg(sum[2])]); // 0
    sat.add_clause([Lit::pos(sum[3])]); // 1

    // Add XOR constraint on a[0] and b[0] like ULT might create
    let xor_a0_b0 = sat.new_var();
    encode_xor(&mut sat, xor_a0_b0, a[0], b[0]);
    // Force xor_a0_b0 = true (a[0] != b[0])
    sat.add_clause([Lit::pos(xor_a0_b0)]);

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
            let xor_val = model[xor_a0_b0.index()].is_true();

            println!(
                "xor: a={}, b={}, sum={}, xor_a0_b0={}",
                a_val, b_val, sum_val, xor_val
            );
            assert!((a_val % 2) != (b_val % 2), "a[0] != b[0]");
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// Test: adder + AND constraint like ULT uses
#[test]
fn test_adder_with_and_constraint() {
    let mut sat = Solver::new();
    let width = 4;

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

    // Constrain sum = 11 (binary: 1011)
    // NOTE: sum must be odd for a[0]=0, b[0]=1 to be satisfiable
    // (a even + b odd = odd sum)
    sat.add_clause([Lit::pos(sum[0])]); // 1
    sat.add_clause([Lit::pos(sum[1])]); // 1
    sat.add_clause([Lit::neg(sum[2])]); // 0
    sat.add_clause([Lit::pos(sum[3])]); // 1

    // Add AND constraint: (~a[0] AND b[0]) - like first step of ULT
    // This encodes: a[0]=0 AND b[0]=1, i.e., a[0] < b[0]
    let and_nota_b = sat.new_var();
    // ~and_nota_b | ~a[0]
    sat.add_clause([Lit::neg(and_nota_b), Lit::neg(a[0])]);
    // ~and_nota_b | b[0]
    sat.add_clause([Lit::neg(and_nota_b), Lit::pos(b[0])]);
    // a[0] | ~b[0] | and_nota_b
    sat.add_clause([Lit::pos(a[0]), Lit::neg(b[0]), Lit::pos(and_nota_b)]);

    // Force and_nota_b = true (a[0] < b[0], which means a[0]=0, b[0]=1)
    sat.add_clause([Lit::pos(and_nota_b)]);

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

            println!(
                "and: a={} (a[0]={}), b={} (b[0]={}), sum={}",
                a_val,
                a_val % 2,
                b_val,
                b_val % 2,
                sum_val
            );
            assert_eq!(a_val % 2, 0, "a[0] should be 0");
            assert_eq!(b_val % 2, 1, "b[0] should be 1");
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}
