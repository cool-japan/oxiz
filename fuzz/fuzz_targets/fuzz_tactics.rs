//! Fuzz target for tactic application
//!
//! Tests various tactics on random formulas

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;
use oxiz::{Solver, TermManager};
use oxiz::core::tactic::*;

#[derive(Debug, Arbitrary)]
enum TacticType {
    Simplify,
    Propagate,
    SolveEqs,
    Eliminate,
    Split,
    CtxSimplify,
}

#[derive(Debug, Arbitrary)]
struct TacticApplication {
    tactic: TacticType,
    formula_type: u8,
    value: i16,
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    let num_applications: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 10) + 1,
        Err(_) => return,
    };

    let mut tm = TermManager::new();

    // Create some variables
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);
    let p = tm.mk_var("p", tm.sorts.bool_sort);

    // Apply tactics
    for _ in 0..num_applications {
        let app: Result<TacticApplication, _> = unstructured.arbitrary();
        let app = match app {
            Ok(a) => a,
            Err(_) => break,
        };

        // Create a formula
        let c = tm.mk_int(BigInt::from(app.value));
        let formula = match app.formula_type % 6 {
            0 => tm.mk_eq(x, c),
            1 => tm.mk_le(x, c),
            2 => {
                let sum = tm.mk_add(vec![x, y]);
                tm.mk_eq(sum, c)
            }
            3 => {
                let eq = tm.mk_eq(x, c);
                tm.mk_and(vec![p, eq])
            }
            4 => {
                let eq = tm.mk_eq(x, c);
                tm.mk_or(vec![p, eq])
            }
            _ => tm.mk_bool(true),
        };

        // Apply tactic
        let _result = match app.tactic {
            TacticType::Simplify => simplify::apply_simplify(&mut tm, formula),
            TacticType::Propagate => propagate::apply_propagate(&mut tm, formula),
            TacticType::SolveEqs => solve_eqs::apply_solve_eqs(&mut tm, formula),
            TacticType::Eliminate => eliminate::apply_eliminate(&mut tm, formula),
            TacticType::Split => {
                let branches = split::apply_split(&mut tm, formula);
                if !branches.is_empty() {
                    branches[0]
                } else {
                    formula
                }
            }
            TacticType::CtxSimplify => ctx_simplify::apply_ctx_simplify(&mut tm, formula),
        };
    }
});
