//! Fuzz target for term construction
//!
//! This fuzzer tests term construction with random operations to ensure
//! the TermManager handles all cases without crashing.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;
use num_rational::Rational64;
use oxiz_core::ast::TermManager;
use oxiz_core::TermId;

/// Represents a random term operation
#[derive(Debug, Arbitrary)]
#[allow(clippy::enum_variant_names)]
enum TermOp {
    // Boolean operations
    MkTrue,
    MkFalse,
    MkBoolVar { name_idx: u8 },
    MkNot { arg_idx: u8 },
    MkAnd { arg_indices: Vec<u8> },
    MkOr { arg_indices: Vec<u8> },
    MkImplies { lhs_idx: u8, rhs_idx: u8 },
    MkXor { lhs_idx: u8, rhs_idx: u8 },
    MkIte { cond_idx: u8, then_idx: u8, else_idx: u8 },
    MkEq { lhs_idx: u8, rhs_idx: u8 },
    MkDistinct { arg_indices: Vec<u8> },

    // Integer operations
    MkInt { value: i64 },
    MkIntVar { name_idx: u8 },
    MkAdd { arg_indices: Vec<u8> },
    MkSub { lhs_idx: u8, rhs_idx: u8 },
    MkMul { arg_indices: Vec<u8> },
    MkDiv { lhs_idx: u8, rhs_idx: u8 },
    MkMod { lhs_idx: u8, rhs_idx: u8 },
    MkNeg { arg_idx: u8 },
    MkLt { lhs_idx: u8, rhs_idx: u8 },
    MkLe { lhs_idx: u8, rhs_idx: u8 },
    MkGt { lhs_idx: u8, rhs_idx: u8 },
    MkGe { lhs_idx: u8, rhs_idx: u8 },

    // Real operations
    MkReal { numer: i32, denom: i32 },
    MkRealVar { name_idx: u8 },

    // Array operations
    MkSelect { array_idx: u8, index_idx: u8 },
    MkStore { array_idx: u8, index_idx: u8, value_idx: u8 },

    // String operations
    MkString { value_idx: u8 },
    MkStrLen { arg_idx: u8 },
    MkStrConcat { s1_idx: u8, s2_idx: u8 },
    MkStrContains { s_idx: u8, sub_idx: u8 },

    // BitVector operations
    MkBitVec { value: u64, width: u8 },
}

/// Build a name from an index
fn make_name(prefix: &str, idx: u8) -> String {
    format!("{}_{}", prefix, idx % 16)
}

/// Sample strings for fuzzing
const SAMPLE_STRINGS: [&str; 8] = [
    "",
    "hello",
    "world",
    "test",
    "abc",
    "123",
    "foo bar",
    "special!@#",
];

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    // Limit the number of operations to prevent OOM
    let max_ops = 100;

    let mut tm = TermManager::new();
    let mut terms: Vec<TermId> = Vec::new();

    // Generate some initial terms
    terms.push(tm.mk_true());
    terms.push(tm.mk_false());
    terms.push(tm.mk_int(BigInt::from(0)));
    terms.push(tm.mk_int(BigInt::from(1)));

    // Generate a sequence of operations
    for _ in 0..max_ops {
        let op: Result<TermOp, _> = unstructured.arbitrary();
        let op = match op {
            Ok(op) => op,
            Err(_) => break,
        };

        let get_term = |idx: u8| -> Option<TermId> {
            if terms.is_empty() {
                None
            } else {
                Some(terms[idx as usize % terms.len()])
            }
        };

        let get_terms = |indices: &[u8]| -> Vec<TermId> {
            if terms.is_empty() {
                Vec::new()
            } else {
                indices
                    .iter()
                    .map(|&idx| terms[idx as usize % terms.len()])
                    .collect()
            }
        };

        let new_term = match op {
            // Boolean operations
            TermOp::MkTrue => Some(tm.mk_true()),
            TermOp::MkFalse => Some(tm.mk_false()),
            TermOp::MkBoolVar { name_idx } => {
                Some(tm.mk_var(&make_name("b", name_idx), tm.sorts.bool_sort))
            }
            TermOp::MkNot { arg_idx } => get_term(arg_idx).map(|t| tm.mk_not(t)),
            TermOp::MkAnd { ref arg_indices } => {
                let args = get_terms(arg_indices);
                if args.is_empty() {
                    None
                } else {
                    Some(tm.mk_and(args))
                }
            }
            TermOp::MkOr { ref arg_indices } => {
                let args = get_terms(arg_indices);
                if args.is_empty() {
                    None
                } else {
                    Some(tm.mk_or(args))
                }
            }
            TermOp::MkImplies { lhs_idx, rhs_idx } => {
                match (get_term(lhs_idx), get_term(rhs_idx)) {
                    (Some(lhs), Some(rhs)) => Some(tm.mk_implies(lhs, rhs)),
                    _ => None,
                }
            }
            TermOp::MkXor { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_xor(lhs, rhs)),
                _ => None,
            },
            TermOp::MkIte {
                cond_idx,
                then_idx,
                else_idx,
            } => match (get_term(cond_idx), get_term(then_idx), get_term(else_idx)) {
                (Some(cond), Some(then_br), Some(else_br)) => {
                    Some(tm.mk_ite(cond, then_br, else_br))
                }
                _ => None,
            },
            TermOp::MkEq { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_eq(lhs, rhs)),
                _ => None,
            },
            TermOp::MkDistinct { ref arg_indices } => {
                let args = get_terms(arg_indices);
                if args.len() >= 2 {
                    Some(tm.mk_distinct(args))
                } else {
                    None
                }
            }

            // Integer operations
            TermOp::MkInt { value } => Some(tm.mk_int(BigInt::from(value))),
            TermOp::MkIntVar { name_idx } => {
                Some(tm.mk_var(&make_name("i", name_idx), tm.sorts.int_sort))
            }
            TermOp::MkAdd { ref arg_indices } => {
                let args = get_terms(arg_indices);
                if args.is_empty() {
                    None
                } else {
                    Some(tm.mk_add(args))
                }
            }
            TermOp::MkSub { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_sub(lhs, rhs)),
                _ => None,
            },
            TermOp::MkMul { ref arg_indices } => {
                let args = get_terms(arg_indices);
                if args.is_empty() {
                    None
                } else {
                    Some(tm.mk_mul(args))
                }
            }
            TermOp::MkDiv { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_div(lhs, rhs)),
                _ => None,
            },
            TermOp::MkMod { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_mod(lhs, rhs)),
                _ => None,
            },
            TermOp::MkNeg { arg_idx } => get_term(arg_idx).map(|t| tm.mk_neg(t)),
            TermOp::MkLt { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_lt(lhs, rhs)),
                _ => None,
            },
            TermOp::MkLe { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_le(lhs, rhs)),
                _ => None,
            },
            TermOp::MkGt { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_gt(lhs, rhs)),
                _ => None,
            },
            TermOp::MkGe { lhs_idx, rhs_idx } => match (get_term(lhs_idx), get_term(rhs_idx)) {
                (Some(lhs), Some(rhs)) => Some(tm.mk_ge(lhs, rhs)),
                _ => None,
            },

            // Real operations
            TermOp::MkReal { numer, denom } => {
                if denom != 0 {
                    Some(tm.mk_real(Rational64::new(numer.into(), denom.into())))
                } else {
                    None
                }
            }
            TermOp::MkRealVar { name_idx } => {
                Some(tm.mk_var(&make_name("r", name_idx), tm.sorts.real_sort))
            }

            // Array operations
            TermOp::MkSelect {
                array_idx,
                index_idx,
            } => match (get_term(array_idx), get_term(index_idx)) {
                (Some(arr), Some(idx)) => Some(tm.mk_select(arr, idx)),
                _ => None,
            },
            TermOp::MkStore {
                array_idx,
                index_idx,
                value_idx,
            } => match (get_term(array_idx), get_term(index_idx), get_term(value_idx)) {
                (Some(arr), Some(idx), Some(val)) => Some(tm.mk_store(arr, idx, val)),
                _ => None,
            },

            // String operations
            TermOp::MkString { value_idx } => {
                let s = SAMPLE_STRINGS[value_idx as usize % SAMPLE_STRINGS.len()];
                Some(tm.mk_string_lit(s))
            }
            TermOp::MkStrLen { arg_idx } => get_term(arg_idx).map(|t| tm.mk_str_len(t)),
            TermOp::MkStrConcat { s1_idx, s2_idx } => {
                match (get_term(s1_idx), get_term(s2_idx)) {
                    (Some(s1), Some(s2)) => Some(tm.mk_str_concat(s1, s2)),
                    _ => None,
                }
            }
            TermOp::MkStrContains { s_idx, sub_idx } => {
                match (get_term(s_idx), get_term(sub_idx)) {
                    (Some(s), Some(sub)) => Some(tm.mk_str_contains(s, sub)),
                    _ => None,
                }
            }

            // BitVector operations
            TermOp::MkBitVec { value, width } => {
                let w = (width % 64).max(1) as u32;
                Some(tm.mk_bitvec(BigInt::from(value), w))
            }
        };

        if let Some(t) = new_term {
            terms.push(t);

            // Keep the term list bounded to prevent OOM
            if terms.len() > 1000 {
                terms.drain(0..500);
            }
        }
    }
});
