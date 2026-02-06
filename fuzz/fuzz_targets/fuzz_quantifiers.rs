//! Fuzz target for quantifier instantiation
//!
//! Tests quantified formulas and instantiation

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;
use oxiz::{Solver, TermManager};

#[derive(Debug, Arbitrary)]
enum Quantifier {
    Forall,
    Exists,
}

#[derive(Debug, Arbitrary)]
struct QuantifiedFormula {
    quantifier: Quantifier,
    num_vars: u8,
    body_type: u8,
    value: i16,
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    let num_formulas: u8 = match unstructured.arbitrary() {
        Ok(n) => (n % 8) + 1,
        Err(_) => return,
    };

    let mut solver = Solver::new();
    let mut tm = TermManager::new();

    // Generate quantified formulas
    for _ in 0..num_formulas {
        let formula: Result<QuantifiedFormula, _> = unstructured.arbitrary();
        let formula = match formula {
            Ok(f) => f,
            Err(_) => break,
        };

        let num_quant_vars = (formula.num_vars % 3) + 1;

        // Create quantified variables
        let mut quant_vars = Vec::new();
        for i in 0..num_quant_vars {
            let v = tm.mk_var(&format!("q{}", i), tm.sorts.int_sort);
            quant_vars.push(v);
        }

        // Create body based on body_type
        let body = match formula.body_type % 4 {
            0 => {
                // q0 >= value
                let v = quant_vars[0];
                let c = tm.mk_int(BigInt::from(formula.value));
                tm.mk_ge(v, c)
            }
            1 => {
                // q0 = value
                let v = quant_vars[0];
                let c = tm.mk_int(BigInt::from(formula.value));
                tm.mk_eq(v, c)
            }
            2 => {
                // q0 + q1 > value (if we have 2 vars)
                if quant_vars.len() >= 2 {
                    let sum = tm.mk_add(vec![quant_vars[0], quant_vars[1]]);
                    let c = tm.mk_int(BigInt::from(formula.value));
                    tm.mk_gt(sum, c)
                } else {
                    tm.mk_bool(true)
                }
            }
            _ => {
                // q0 = q0 (tautology)
                tm.mk_eq(quant_vars[0], quant_vars[0])
            }
        };

        // Create quantified formula
        let quantified = match formula.quantifier {
            Quantifier::Forall => tm.mk_forall(quant_vars, body),
            Quantifier::Exists => tm.mk_exists(quant_vars, body),
        };

        solver.assert(quantified, &mut tm);
    }

    // Run solver with timeout (quantifiers can be expensive)
    solver.set_timeout(std::time::Duration::from_millis(100));
    let _ = solver.check(&mut tm);
});
