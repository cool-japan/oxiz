//! Verify SAT solver correctness

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

/// Encode full adder - returns the intermediate vars for verification
fn encode_full_adder(
    sat: &mut Solver,
    sum: Var,
    cout: Var,
    a: Var,
    b: Var,
    cin: Var,
) -> (Var, Var, Var) {
    let xor_ab = sat.new_var();
    encode_xor(sat, xor_ab, a, b);
    encode_xor(sat, sum, xor_ab, cin);

    let and_ab = sat.new_var();
    encode_and(sat, and_ab, a, b);

    let and_cin_xor = sat.new_var();
    encode_and(sat, and_cin_xor, cin, xor_ab);

    encode_or(sat, cout, and_ab, and_cin_xor);

    (xor_ab, and_ab, and_cin_xor)
}

/// Parameters for full adder verification
struct FullAdderParams {
    sum: Var,
    cout: Var,
    a: Var,
    b: Var,
    cin: Var,
    xor_ab: Var,
    and_ab: Var,
    and_cin_xor: Var,
}

/// Verify full adder values
fn verify_full_adder(model: &[oxiz_sat::LBool], params: &FullAdderParams) -> bool {
    let FullAdderParams {
        sum,
        cout,
        a,
        b,
        cin,
        xor_ab,
        and_ab,
        and_cin_xor,
    } = *params;
    let av = model[a.index()].is_true();
    let bv = model[b.index()].is_true();
    let cinv = model[cin.index()].is_true();
    let sumv = model[sum.index()].is_true();
    let coutv = model[cout.index()].is_true();
    let xor_abv = model[xor_ab.index()].is_true();
    let and_abv = model[and_ab.index()].is_true();
    let and_cin_xorv = model[and_cin_xor.index()].is_true();

    // Verify each encoding
    let expected_xor_ab = av ^ bv;
    let expected_sum = xor_abv ^ cinv;
    let expected_and_ab = av && bv;
    let expected_and_cin_xor = cinv && xor_abv;
    let expected_cout = and_abv || and_cin_xorv;

    let ok = xor_abv == expected_xor_ab
        && sumv == expected_sum
        && and_abv == expected_and_ab
        && and_cin_xorv == expected_and_cin_xor
        && coutv == expected_cout;

    if !ok {
        println!("  a={}, b={}, cin={}", av, bv, cinv);
        println!("  xor_ab={} (expected {})", xor_abv, expected_xor_ab);
        println!("  sum={} (expected {})", sumv, expected_sum);
        println!("  and_ab={} (expected {})", and_abv, expected_and_ab);
        println!(
            "  and_cin_xor={} (expected {})",
            and_cin_xorv, expected_and_cin_xor
        );
        println!("  cout={} (expected {})", coutv, expected_cout);
    }

    ok
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

    let mut lt_prev = sat.new_var();
    encode_and_not_a(sat, lt_prev, a_bits[0], b_bits[0]);

    for i in 1..width {
        let ai = a_bits[i];
        let bi = b_bits[i];

        let lt_at_i = sat.new_var();
        encode_and_not_a(sat, lt_at_i, ai, bi);

        let eq_i = sat.new_var();
        encode_xnor(sat, eq_i, ai, bi);

        let carry_prev = sat.new_var();
        encode_and(sat, carry_prev, eq_i, lt_prev);

        let lt_next = sat.new_var();
        encode_or(sat, lt_next, lt_at_i, carry_prev);

        lt_prev = lt_next;
    }

    sat.add_clause([Lit::neg(result), Lit::pos(lt_prev)]);
    sat.add_clause([Lit::pos(result), Lit::neg(lt_prev)]);
}

#[test]
fn test_4bit_with_verification() {
    let mut sat = Solver::new();
    let width = 4usize;

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    // Track all intermediate variables for verification
    let mut carries: Vec<Var> = Vec::new();
    let mut xor_abs: Vec<Var> = Vec::new();
    let mut and_abs: Vec<Var> = Vec::new();
    let mut and_cin_xors: Vec<Var> = Vec::new();

    // Encode adder
    let initial_carry = sat.new_var();
    sat.add_clause([Lit::neg(initial_carry)]);
    carries.push(initial_carry);

    for i in 0..width {
        let cout = sat.new_var();
        let (xor_ab, and_ab, and_cin_xor) =
            encode_full_adder(&mut sat, sum[i], cout, a[i], b[i], carries[i]);
        carries.push(cout);
        xor_abs.push(xor_ab);
        and_abs.push(and_ab);
        and_cin_xors.push(and_cin_xor);
    }

    // Constrain sum = 10 (1010)
    sat.add_clause([Lit::neg(sum[0])]); // 0
    sat.add_clause([Lit::pos(sum[1])]); // 1
    sat.add_clause([Lit::neg(sum[2])]); // 0
    sat.add_clause([Lit::pos(sum[3])]); // 1

    // Encode ULT
    let ult_result = sat.new_var();
    encode_ult(&mut sat, ult_result, &a, &b);
    sat.add_clause([Lit::pos(ult_result)]);

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

            println!("SAT result: a={}, b={}, sum={}", a_val, b_val, sum_val);

            // Verify each full adder
            println!("\nVerifying full adders:");
            let mut all_ok = true;
            for i in 0..width {
                println!("Bit {}:", i);
                let params = FullAdderParams {
                    sum: sum[i],
                    cout: carries[i + 1],
                    a: a[i],
                    b: b[i],
                    cin: carries[i],
                    xor_ab: xor_abs[i],
                    and_ab: and_abs[i],
                    and_cin_xor: and_cin_xors[i],
                };
                let ok = verify_full_adder(model, &params);
                if !ok {
                    all_ok = false;
                    println!("  VERIFICATION FAILED at bit {}", i);
                } else {
                    println!("  OK");
                }
            }

            if !all_ok {
                panic!("Model verification failed!");
            }

            assert_eq!((a_val + b_val) % 16, sum_val, "a + b should equal sum");
        }
        SolverResult::Unsat => {
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}

/// Test: solve adder first, then add ULT
#[test]
fn test_4bit_solve_first() {
    let mut sat = Solver::new();
    let width = 4usize;

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

    // First solve without ULT
    println!("First solve (no ULT)...");
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
                "First solve: a={}, b={}, sum={}, a+b={}",
                a_val,
                b_val,
                sum_val,
                (a_val + b_val) % 16
            );
            assert_eq!((a_val + b_val) % 16, sum_val);
        }
        _ => panic!("First solve should be SAT"),
    }

    // Now add ULT and solve again
    println!("\nAdding ULT and solving again...");
    let ult_result = sat.new_var();
    encode_ult(&mut sat, ult_result, &a, &b);
    sat.add_clause([Lit::pos(ult_result)]);

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
                "Second solve: a={}, b={}, sum={}, a+b={}",
                a_val,
                b_val,
                sum_val,
                (a_val + b_val) % 16
            );

            assert!(a_val < b_val, "a {} should be < b {}", a_val, b_val);
            assert_eq!((a_val + b_val) % 16, sum_val, "a + b should equal sum");
        }
        SolverResult::Unsat => {
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}
