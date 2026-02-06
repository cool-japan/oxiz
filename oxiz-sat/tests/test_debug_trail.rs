//! Debug the trail state

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

/// Simpler test: 2 full adders (2-bit)
#[test]
fn test_2bit_detailed() {
    let mut sat = Solver::new();
    let width = 2;

    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();

    println!(
        "Main vars: a={:?}, b={:?}, sum={:?}",
        a.iter().map(|v| v.index()).collect::<Vec<_>>(),
        b.iter().map(|v| v.index()).collect::<Vec<_>>(),
        sum.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    let carry0 = sat.new_var();
    println!("carry0 = {}", carry0.index());
    sat.add_clause([Lit::neg(carry0)]);

    let carry1 = sat.new_var();
    let (xor01, and01, and_c0_xor01) =
        encode_full_adder(&mut sat, sum[0], carry1, a[0], b[0], carry0);
    println!(
        "FA0: sum[0]={}, carry1={}, xor01={}, and01={}, and_c0_xor01={}",
        sum[0].index(),
        carry1.index(),
        xor01.index(),
        and01.index(),
        and_c0_xor01.index()
    );

    let carry2 = sat.new_var();
    let (xor12, and12, and_c1_xor12) =
        encode_full_adder(&mut sat, sum[1], carry2, a[1], b[1], carry1);
    println!(
        "FA1: sum[1]={}, carry2={}, xor12={}, and12={}, and_c1_xor12={}",
        sum[1].index(),
        carry2.index(),
        xor12.index(),
        and12.index(),
        and_c1_xor12.index()
    );

    // Constrain sum = 2 (binary 10)
    println!("\nAdding sum[0]=0, sum[1]=1 (sum=2)...");
    sat.add_clause([Lit::neg(sum[0])]); // sum[0] = 0
    sat.add_clause([Lit::pos(sum[1])]); // sum[1] = 1

    println!("Total vars: {}", sat.num_vars());

    println!("\nFirst solve...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);

    if r1 == SolverResult::Sat {
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
            (a_val + b_val) % 4
        );

        // Verify intermediate variables
        println!("Intermediate:");
        println!(
            "  carry0={}, carry1={}, carry2={}",
            model[carry0.index()].is_true(),
            model[carry1.index()].is_true(),
            model[carry2.index()].is_true()
        );
        println!(
            "  xor01={}, and01={}, and_c0_xor01={}",
            model[xor01.index()].is_true(),
            model[and01.index()].is_true(),
            model[and_c0_xor01.index()].is_true()
        );
        println!(
            "  xor12={}, and12={}, and_c1_xor12={}",
            model[xor12.index()].is_true(),
            model[and12.index()].is_true(),
            model[and_c1_xor12.index()].is_true()
        );
    }

    // Now add a[0] = 0
    println!("\nAdding a[0] = 0...");
    sat.add_clause([Lit::neg(a[0])]);

    println!("\nSecond solve...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);

    if r2 == SolverResult::Sat {
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
            (a_val + b_val) % 4
        );

        // Verify intermediate variables
        println!("Intermediate:");
        println!(
            "  carry0={}, carry1={}, carry2={}",
            model[carry0.index()].is_true(),
            model[carry1.index()].is_true(),
            model[carry2.index()].is_true()
        );
        println!(
            "  xor01={}, and01={}, and_c0_xor01={}",
            model[xor01.index()].is_true(),
            model[and01.index()].is_true(),
            model[and_c0_xor01.index()].is_true()
        );
        println!(
            "  xor12={}, and12={}, and_c1_xor12={}",
            model[xor12.index()].is_true(),
            model[and12.index()].is_true(),
            model[and_c1_xor12.index()].is_true()
        );

        // Verify XOR at bit 0: sum[0] = xor01 XOR carry0
        // xor01 = a[0] XOR b[0]
        let a0 = model[a[0].index()].is_true();
        let b0 = model[b[0].index()].is_true();
        let xor01_expected = a0 ^ b0;
        let xor01_actual = model[xor01.index()].is_true();
        let c0 = model[carry0.index()].is_true();
        let sum0_expected = xor01_actual ^ c0;
        let sum0_actual = model[sum[0].index()].is_true();

        println!("\nBit 0 verification:");
        println!(
            "  a[0]={}, b[0]={}, xor01: expected={}, actual={}, match={}",
            a0,
            b0,
            xor01_expected,
            xor01_actual,
            xor01_expected == xor01_actual
        );
        println!(
            "  xor01={}, carry0={}, sum[0]: expected={}, actual={}, match={}",
            xor01_actual,
            c0,
            sum0_expected,
            sum0_actual,
            sum0_expected == sum0_actual
        );

        // Carry out: carry1 = (a[0] AND b[0]) OR (carry0 AND xor01)
        let and01_expected = a0 && b0;
        let and01_actual = model[and01.index()].is_true();
        let and_c0_xor01_expected = c0 && xor01_actual;
        let and_c0_xor01_actual = model[and_c0_xor01.index()].is_true();
        let carry1_expected = and01_actual || and_c0_xor01_actual;
        let carry1_actual = model[carry1.index()].is_true();

        println!(
            "  and01: expected={}, actual={}, match={}",
            and01_expected,
            and01_actual,
            and01_expected == and01_actual
        );
        println!(
            "  and_c0_xor01: expected={}, actual={}, match={}",
            and_c0_xor01_expected,
            and_c0_xor01_actual,
            and_c0_xor01_expected == and_c0_xor01_actual
        );
        println!(
            "  carry1: expected={}, actual={}, match={}",
            carry1_expected,
            carry1_actual,
            carry1_expected == carry1_actual
        );

        assert_eq!((a_val + b_val) % 4, sum_val, "a + b should equal sum");
    }
}
