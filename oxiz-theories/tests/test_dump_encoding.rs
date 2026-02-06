//! Dump encoding details to understand the issue

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

/// Encode ULT: result = (a < b)
fn encode_ult(sat: &mut Solver, result: Var, a_bits: &[Var], b_bits: &[Var]) {
    let width = a_bits.len();
    if width == 0 {
        sat.add_clause([Lit::neg(result)]);
        return;
    }

    // lt_0 = ~a[0] & b[0]
    let mut lt_prev = sat.new_var();
    // ~lt | ~a
    sat.add_clause([Lit::neg(lt_prev), Lit::neg(a_bits[0])]);
    // ~lt | b
    sat.add_clause([Lit::neg(lt_prev), Lit::pos(b_bits[0])]);
    // a | ~b | lt
    sat.add_clause([Lit::pos(a_bits[0]), Lit::neg(b_bits[0]), Lit::pos(lt_prev)]);

    for i in 1..width {
        let ai = a_bits[i];
        let bi = b_bits[i];

        // lt_at_i = ~ai & bi
        let lt_at_i = sat.new_var();
        sat.add_clause([Lit::neg(lt_at_i), Lit::neg(ai)]);
        sat.add_clause([Lit::neg(lt_at_i), Lit::pos(bi)]);
        sat.add_clause([Lit::pos(ai), Lit::neg(bi), Lit::pos(lt_at_i)]);

        // eq_i = (ai <=> bi) = XNOR
        let eq_i = sat.new_var();
        sat.add_clause([Lit::neg(eq_i), Lit::neg(ai), Lit::pos(bi)]);
        sat.add_clause([Lit::neg(eq_i), Lit::pos(ai), Lit::neg(bi)]);
        sat.add_clause([Lit::pos(eq_i), Lit::neg(ai), Lit::neg(bi)]);
        sat.add_clause([Lit::pos(eq_i), Lit::pos(ai), Lit::pos(bi)]);

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

/// 4-bit test at SAT level
#[test]
fn test_sat_level_4bit() {
    let mut sat = Solver::new();
    let width = 4;

    println!("Creating variables...");

    // a (bits 0-3)
    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    println!(
        "a = vars {:?}",
        a.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    // b (bits 4-7)
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    println!(
        "b = vars {:?}",
        b.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    // sum (bits 8-11)
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    println!(
        "sum = vars {:?}",
        sum.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    // target = 10 (1010)
    println!("Constraining sum = 10 (1010)...");
    sat.add_clause([Lit::neg(sum[0])]); // 0
    sat.add_clause([Lit::pos(sum[1])]); // 1
    sat.add_clause([Lit::neg(sum[2])]); // 0
    sat.add_clause([Lit::pos(sum[3])]); // 1

    // Encode addition: sum_computed = a + b
    println!("Encoding addition...");
    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]); // Initial carry = 0

    let mut sum_computed: Vec<Var> = Vec::new();
    for i in 0..width {
        let s = sat.new_var();
        let cout = sat.new_var();
        encode_full_adder(&mut sat, s, cout, a[i], b[i], carry);
        sum_computed.push(s);
        carry = cout;
    }
    println!(
        "sum_computed = vars {:?}",
        sum_computed.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    // Wait! I think I see the bug - the BvSolver creates SEPARATE sum bits,
    // but the encode_adder writes to different variables than the ones in sum!
    // Let me verify by constraining sum_computed = sum (which BvSolver should do)

    println!("Constraining sum_computed == sum...");
    for i in 0..width {
        // sum_computed[i] <=> sum[i]
        sat.add_clause([Lit::neg(sum_computed[i]), Lit::pos(sum[i])]);
        sat.add_clause([Lit::pos(sum_computed[i]), Lit::neg(sum[i])]);
    }

    // Encode ULT: a < b
    println!("Encoding a < b...");
    let ult_result = sat.new_var();
    encode_ult(&mut sat, ult_result, &a, &b);
    sat.add_clause([Lit::pos(ult_result)]); // Assert a < b

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
            let sum_computed_val: u64 = sum_computed
                .iter()
                .enumerate()
                .map(|(i, &v)| (model[v.index()].is_true() as u64) << i)
                .sum();

            println!("SAT:");
            println!("  a = {} (binary: {:04b})", a_val, a_val);
            println!("  b = {} (binary: {:04b})", b_val, b_val);
            println!("  sum = {} (binary: {:04b})", sum_val, sum_val);
            println!(
                "  sum_computed = {} (binary: {:04b})",
                sum_computed_val, sum_computed_val
            );
            println!("  a + b = {} (mod 16)", (a_val + b_val) % 16);

            assert!(a_val < b_val, "a {} should be < b {}", a_val, b_val);
            assert_eq!(sum_val, 10, "sum should be 10");
            assert_eq!(sum_computed_val, sum_val, "sum_computed should equal sum");
            assert_eq!((a_val + b_val) % 16, sum_val, "a + b should equal sum");
        }
        SolverResult::Unsat => {
            println!("UNSAT");
            panic!("Should be SAT");
        }
        _ => panic!("Unknown"),
    }
}

/// Simple test without the sum_computed == sum constraint
/// This mimics what BvSolver is probably doing incorrectly
#[test]
fn test_sat_level_4bit_wrong() {
    let mut sat = Solver::new();
    let width = 4;

    println!("Creating variables...");

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    // target = 10 (1010)
    sat.add_clause([Lit::neg(sum[0])]); // 0
    sat.add_clause([Lit::pos(sum[1])]); // 1
    sat.add_clause([Lit::neg(sum[2])]); // 0
    sat.add_clause([Lit::pos(sum[3])]); // 1

    // Encode addition but DON'T connect output to sum
    let mut carry = sat.new_var();
    sat.add_clause([Lit::neg(carry)]);

    let mut sum_computed: Vec<Var> = Vec::new();
    for i in 0..width {
        let s = sat.new_var();
        let cout = sat.new_var();
        encode_full_adder(&mut sat, s, cout, a[i], b[i], carry);
        sum_computed.push(s);
        carry = cout;
    }

    // NO constraint connecting sum_computed to sum!
    // This is what might be happening in BvSolver

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

            println!("WRONG TEST - SAT:");
            println!("  a = {} (binary: {:04b})", a_val, a_val);
            println!("  b = {} (binary: {:04b})", b_val, b_val);
            println!("  sum = {} (should be 10)", sum_val);
            println!("  a + b = {} (probably != sum)", (a_val + b_val) % 16);

            // This SHOULD pass because sum is disconnected from a, b
            assert_eq!(sum_val, 10, "sum constrained to 10");
            // But this will fail:
            // assert_eq!((a_val + b_val) % 16, sum_val, "a + b != sum because disconnected");
        }
        SolverResult::Unsat => {
            println!("UNSAT - unexpected");
        }
        _ => panic!("Unknown"),
    }
}
