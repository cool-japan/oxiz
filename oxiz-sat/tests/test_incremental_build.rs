//! Build up the encoding incrementally to find where it breaks

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

/// Build incrementally: encode 1 bit, constrain, add unit, solve
#[test]
fn test_incremental_1bit() {
    let mut sat = Solver::new();

    let a = sat.new_var();
    let b = sat.new_var();
    let sum = sat.new_var();

    let carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]); // carry_in = 0

    let cout = sat.new_var();
    encode_full_adder(&mut sat, sum, cout, a, b, carry);

    // Constrain sum = 1
    sat.add_clause([Lit::pos(sum)]);

    println!("1-bit: After adder + sum=1, adding a=0...");
    sat.add_clause([Lit::neg(a)]);

    match sat.solve() {
        SolverResult::Sat => {
            let model = sat.model();
            let a_val = model[a.index()].is_true();
            let b_val = model[b.index()].is_true();
            let sum_val = model[sum.index()].is_true();
            println!("SAT: a={}, b={}, sum={}", a_val, b_val, sum_val);
            assert!(!a_val);
            assert!(sum_val);
            // a=0, sum=1 => b must be 1 (0+1=1)
            assert!(b_val, "b should be 1");
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// 2 bits
#[test]
fn test_incremental_2bit() {
    let mut sat = Solver::new();
    let width = 2;

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

    // Constrain sum = 2 (10 binary)
    sat.add_clause([Lit::neg(sum[0])]);
    sat.add_clause([Lit::pos(sum[1])]);

    println!("2-bit: After adder + sum=2, adding a[0]=0...");
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
            assert_eq!((a_val + b_val) % 4, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// 3 bits
#[test]
fn test_incremental_3bit() {
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

    // Constrain sum = 5 (101 binary)
    sat.add_clause([Lit::pos(sum[0])]);
    sat.add_clause([Lit::neg(sum[1])]);
    sat.add_clause([Lit::pos(sum[2])]);

    println!("3-bit: After adder + sum=5, adding a[0]=0...");
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

/// 4 bits
#[test]
fn test_incremental_4bit() {
    let mut sat = Solver::new();
    let width = 4;

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

    // Constrain sum = 10 (1010 binary)
    sat.add_clause([Lit::neg(sum[0])]);
    sat.add_clause([Lit::pos(sum[1])]);
    sat.add_clause([Lit::neg(sum[2])]);
    sat.add_clause([Lit::pos(sum[3])]);

    println!("4-bit: After adder + sum=10, adding a[0]=0...");
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
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// 4 bits - unit constraint BEFORE sum constraint
#[test]
fn test_4bit_unit_before_sum() {
    let mut sat = Solver::new();
    let width = 4;

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

    // Add unit constraint BEFORE sum constraint
    println!("4-bit: After adder, adding a[0]=0 BEFORE sum constraints...");
    sat.add_clause([Lit::neg(a[0])]);

    // Then constrain sum = 10
    sat.add_clause([Lit::neg(sum[0])]);
    sat.add_clause([Lit::pos(sum[1])]);
    sat.add_clause([Lit::neg(sum[2])]);
    sat.add_clause([Lit::pos(sum[3])]);

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
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

/// 4 bits - all constraints at once
#[test]
fn test_4bit_all_at_once() {
    let mut sat = Solver::new();
    let width = 4;

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    // Add all constraints first
    sat.add_clause([Lit::neg(a[0])]); // a[0] = 0
    sat.add_clause([Lit::neg(sum[0])]); // sum = 10
    sat.add_clause([Lit::pos(sum[1])]);
    sat.add_clause([Lit::neg(sum[2])]);
    sat.add_clause([Lit::pos(sum[3])]);

    // Then encode adder
    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]);

    for i in 0..width {
        let cout = sat.new_var();
        encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carry);
        carry = cout;
    }

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
            println!("all-at-once SAT: a={}, b={}, sum={}", a_val, b_val, sum_val);
            assert_eq!(a_val % 2, 0);
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}
