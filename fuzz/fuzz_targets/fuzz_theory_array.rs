//! Fuzz target for array theory
//!
//! Tests array select/store operations

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;
use oxiz::{Solver, TermManager};

#[derive(Debug, Arbitrary)]
enum ArrayOp {
    Select,
    Store,
    Equals,
}

#[derive(Debug, Arbitrary)]
struct ArrayConstraint {
    array_idx: u8,
    op: ArrayOp,
    index_value: i16,
    element_value: i16,
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    let num_arrays: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 5) + 1,
        Err(_) => return,
    };

    let num_ops: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 15) + 1,
        Err(_) => return,
    };

    let mut solver = Solver::new();
    let mut tm = TermManager::new();

    // Create array sort: Array[Int, Int]
    let array_sort = tm.mk_array_sort(tm.sorts.int_sort, tm.sorts.int_sort);

    // Create array variables
    let mut arrays = Vec::new();
    for i in 0..num_arrays {
        let arr = tm.mk_var(&format!("arr{}", i), array_sort);
        arrays.push(arr);
    }

    // Generate and assert array operations
    for _ in 0..num_ops {
        let constraint: Result<ArrayConstraint, _> = unstructured.arbitrary();
        let constraint = match constraint {
            Ok(c) => c,
            Err(_) => break,
        };

        let arr = arrays[(constraint.array_idx as usize) % arrays.len()];
        let idx = tm.mk_int(BigInt::from(constraint.index_value));
        let elem = tm.mk_int(BigInt::from(constraint.element_value));

        match constraint.op {
            ArrayOp::Select => {
                // select(arr, idx) = elem
                let select = tm.mk_select(arr, idx);
                let eq = tm.mk_eq(select, elem);
                solver.assert(eq, &mut tm);
            }
            ArrayOp::Store => {
                // store(arr, idx, elem)
                let stored = tm.mk_store(arr, idx, elem);

                // Assert something about the stored array
                let idx2 = tm.mk_int(BigInt::from(constraint.index_value + 1));
                let select = tm.mk_select(stored, idx2);
                let ge = tm.mk_ge(select, tm.mk_int(BigInt::from(-100)));
                solver.assert(ge, &mut tm);
            }
            ArrayOp::Equals => {
                // arr = arr (tautology, but exercises extensionality)
                let eq = tm.mk_eq(arr, arr);
                solver.assert(eq, &mut tm);
            }
        }
    }

    // Run solver
    let _ = solver.check(&mut tm);
});
