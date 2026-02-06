//! Fuzz target for arithmetic theory solver
//!
//! Tests arithmetic operations with random constraints

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;
use oxiz::core::smtlib::*;
use oxiz::{Solver, TermManager};

#[derive(Debug, Arbitrary)]
enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
}

#[derive(Debug, Arbitrary)]
enum ArithRelation {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Arbitrary)]
struct ArithConstraint {
    var_idx: u8,
    op: ArithOp,
    other_var_idx: u8,
    relation: ArithRelation,
    value: i16,
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    let num_vars: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 10) + 1,
        Err(_) => return,
    };

    let num_constraints: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 20) + 1,
        Err(_) => return,
    };

    let mut solver = Solver::new();
    let mut tm = TermManager::new();

    // Create variables
    let mut vars = Vec::new();
    for i in 0..num_vars {
        let var = tm.mk_var(&format!("x{}", i), tm.sorts.int_sort);
        vars.push(var);
    }

    // Generate and assert constraints
    for _ in 0..num_constraints {
        let constraint: Result<ArithConstraint, _> = unstructured.arbitrary();
        let constraint = match constraint {
            Ok(c) => c,
            Err(_) => break,
        };

        let v1 = vars[(constraint.var_idx as usize) % vars.len()];
        let v2 = vars[(constraint.other_var_idx as usize) % vars.len()];
        let const_val = tm.mk_int(BigInt::from(constraint.value));

        // Build arithmetic expression
        let arith_expr = match constraint.op {
            ArithOp::Add => tm.mk_add(vec![v1, v2]),
            ArithOp::Sub => tm.mk_sub(v1, v2),
            ArithOp::Mul => tm.mk_mul(vec![v1, v2]),
            ArithOp::Div => {
                if constraint.value != 0 {
                    tm.mk_div(v1, const_val)
                } else {
                    v1
                }
            }
            ArithOp::Mod => {
                if constraint.value != 0 {
                    tm.mk_mod(v1, const_val)
                } else {
                    v1
                }
            }
            ArithOp::Neg => tm.mk_neg(v1),
        };

        // Build relational constraint
        let rel_expr = match constraint.relation {
            ArithRelation::Eq => tm.mk_eq(arith_expr, const_val),
            ArithRelation::Ne => tm.mk_not(tm.mk_eq(arith_expr, const_val)),
            ArithRelation::Lt => tm.mk_lt(arith_expr, const_val),
            ArithRelation::Le => tm.mk_le(arith_expr, const_val),
            ArithRelation::Gt => tm.mk_gt(arith_expr, const_val),
            ArithRelation::Ge => tm.mk_ge(arith_expr, const_val),
        };

        solver.assert(rel_expr, &mut tm);
    }

    // Run solver - we don't care about result, just that it doesn't crash
    let _ = solver.check(&mut tm);
});
