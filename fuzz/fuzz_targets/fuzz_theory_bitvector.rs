//! Fuzz target for bitvector theory
//!
//! Tests bitvector operations with random constraints

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use oxiz::{Solver, TermManager};

#[derive(Debug, Arbitrary)]
enum BvOp {
    And,
    Or,
    Xor,
    Not,
    Add,
    Sub,
    Mul,
    UDiv,
    URem,
    Shl,
    LShr,
    AShr,
}

#[derive(Debug, Arbitrary)]
struct BvConstraint {
    var_idx: u8,
    op: BvOp,
    other_var_idx: u8,
    value: u32,
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    let bitwidth: u8 = match unstructured.arbitrary() {
        Ok(n) => match n % 4 {
            0 => 8,
            1 => 16,
            2 => 32,
            _ => 64,
        },
        Err(_) => return,
    };

    let num_vars: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 8) + 1,
        Err(_) => return,
    };

    let num_constraints: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 15) + 1,
        Err(_) => return,
    };

    let mut solver = Solver::new();
    let mut tm = TermManager::new();

    // Create bitvector sort
    let bv_sort = tm.mk_bv_sort(bitwidth as u32);

    // Create variables
    let mut vars = Vec::new();
    for i in 0..num_vars {
        let var = tm.mk_var(&format!("bv{}", i), bv_sort);
        vars.push(var);
    }

    // Generate and assert constraints
    for _ in 0..num_constraints {
        let constraint: Result<BvConstraint, _> = unstructured.arbitrary();
        let constraint = match constraint {
            Ok(c) => c,
            Err(_) => break,
        };

        let v1 = vars[(constraint.var_idx as usize) % vars.len()];
        let v2 = vars[(constraint.other_var_idx as usize) % vars.len()];

        // Build bitvector expression
        let bv_expr = match constraint.op {
            BvOp::And => tm.mk_bvand(v1, v2),
            BvOp::Or => tm.mk_bvor(v1, v2),
            BvOp::Xor => tm.mk_bvxor(v1, v2),
            BvOp::Not => tm.mk_bvnot(v1),
            BvOp::Add => tm.mk_bvadd(v1, v2),
            BvOp::Sub => tm.mk_bvsub(v1, v2),
            BvOp::Mul => tm.mk_bvmul(v1, v2),
            BvOp::UDiv => tm.mk_bvudiv(v1, v2),
            BvOp::URem => tm.mk_bvurem(v1, v2),
            BvOp::Shl => tm.mk_bvshl(v1, v2),
            BvOp::LShr => tm.mk_bvlshr(v1, v2),
            BvOp::AShr => tm.mk_bvashr(v1, v2),
        };

        // Create equality constraint
        let const_bv = tm.mk_bv_const(constraint.value, bitwidth as u32);
        let eq = tm.mk_eq(bv_expr, const_bv);

        solver.assert(eq, &mut tm);
    }

    // Run solver
    let _ = solver.check(&mut tm);
});
