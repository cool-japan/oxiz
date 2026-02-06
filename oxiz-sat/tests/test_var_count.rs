//! Count variables to check for overflow/aliasing

use oxiz_sat::{Solver, Var};

#[test]
fn test_var_count() {
    let mut sat = Solver::new();

    println!("Creating 100 variables...");
    let vars: Vec<Var> = (0..100).map(|_| sat.new_var()).collect();

    for (i, v) in vars.iter().enumerate() {
        println!("var[{}] = Var({})", i, v.index());
        assert_eq!(v.index(), i, "Variable index mismatch!");
    }

    println!("All variables have sequential indices.");
}

#[test]
fn test_4bit_var_allocation() {
    let mut sat = Solver::new();
    let width = 4;

    println!("Creating bitvectors...");
    let a: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    println!(
        "a[0..4] = {:?}",
        a.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    let b: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    println!(
        "b[4..8] = {:?}",
        b.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    let sum: Vec<Var> = (0..width).map(|_| sat.new_var()).collect();
    println!(
        "sum[8..12] = {:?}",
        sum.iter().map(|v| v.index()).collect::<Vec<_>>()
    );

    println!("Creating adder intermediate vars...");
    let initial_carry = sat.new_var();
    println!("initial_carry = {}", initial_carry.index());

    // Each full adder creates: xor_ab, and_ab, and_cin_xor, cout
    // That's 4 vars per bit, but cout is reused as next carry
    // Actually let me trace more carefully

    let mut var_count = 13; // 4 + 4 + 4 + 1 (a, b, sum, initial_carry)

    for i in 0..width {
        let cout = sat.new_var();
        let xor_ab = sat.new_var();
        let and_ab = sat.new_var();
        let and_cin_xor = sat.new_var();
        println!(
            "  bit {}: cout={}, xor_ab={}, and_ab={}, and_cin_xor={}",
            i,
            cout.index(),
            xor_ab.index(),
            and_ab.index(),
            and_cin_xor.index()
        );
        var_count += 4;
    }

    println!("After adder: {} vars created", var_count);

    // ULT encoding
    println!("Creating ULT intermediate vars...");
    let lt_0 = sat.new_var(); // ~a[0] & b[0]
    println!("  lt_0 = {}", lt_0.index());

    for i in 1..width {
        let lt_at_i = sat.new_var();
        let eq_i = sat.new_var();
        let carry_prev = sat.new_var();
        let lt_next = sat.new_var();
        println!(
            "  bit {}: lt_at_i={}, eq_i={}, carry_prev={}, lt_next={}",
            i,
            lt_at_i.index(),
            eq_i.index(),
            carry_prev.index(),
            lt_next.index()
        );
    }

    let ult_result = sat.new_var();
    println!("ult_result = {}", ult_result.index());

    println!("\nTotal vars used by test: {}", sat.num_vars());
}
