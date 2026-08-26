//! Tests for the model evaluator.

use super::*;
use crate::model::FuncInterp;
use smallvec::SmallVec;

#[test]
fn test_eval_cache() {
    let mut cache = EvalCache::new();
    let t1 = TermId::from(1u32);

    assert!(cache.is_empty());

    cache.insert(t1, Value::Bool(true));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(t1), Some(&Value::Bool(true)));

    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn test_eval_result() {
    let ok = EvalResult::Ok(Value::Int(42));
    assert!(ok.is_ok());
    assert_eq!(ok.value(), Some(&Value::Int(42)));

    let undef = EvalResult::Undefined(TermId::from(1u32));
    assert!(!undef.is_ok());
    assert_eq!(undef.value(), None);
}

// Regression tests for: "Model evaluator silently truncates big integer
// and wide BV constants to 0" — a BigInt IntConst or BitVecConst that
// does not fit the fixed-width `Value` representation must surface an
// explicit `EvalResult::Error`, never a fabricated 0.

#[test]
fn test_eval_int_const_in_range_still_works() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let t = manager.mk_int(num_bigint::BigInt::from(42));
    let result = evaluator.eval(t, &manager);
    assert!(matches!(result, EvalResult::Ok(Value::Int(42))));
}

#[test]
fn test_eval_int_const_too_big_for_i64_errors_not_zero() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // 2^100 does not fit in i64; previously this silently evaluated to 0.
    let huge = num_bigint::BigInt::from(2u64).pow(100);
    let t = manager.mk_int(huge);
    let result = evaluator.eval(t, &manager);
    match result {
        EvalResult::Error(_) => {}
        other => panic!("expected EvalResult::Error for oversized IntConst, got {other:?}"),
    }
}

#[test]
fn test_eval_int_const_negative_too_big_errors_not_zero() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let huge_neg = -num_bigint::BigInt::from(2u64).pow(100);
    let t = manager.mk_int(huge_neg);
    let result = evaluator.eval(t, &manager);
    match result {
        EvalResult::Error(_) => {}
        other => {
            panic!("expected EvalResult::Error for oversized negative IntConst, got {other:?}")
        }
    }
}

#[test]
fn test_eval_bitvec_const_in_range_still_works() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let t = manager.mk_bitvec(num_bigint::BigInt::from(7), 8);
    let result = evaluator.eval(t, &manager);
    assert!(matches!(result, EvalResult::Ok(Value::BitVec(8, 7))));
}

#[test]
fn test_eval_wide_bitvec_const_errors_not_zero() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // 128-bit constant with a value that doesn't fit u64; previously
    // this silently evaluated to 0 via `unwrap_or(0)`.
    let wide_val = num_bigint::BigInt::from(2u64).pow(100);
    let t = manager.mk_bitvec(wide_val, 128);
    let result = evaluator.eval(t, &manager);
    match result {
        EvalResult::Error(_) => {}
        other => panic!("expected EvalResult::Error for wide BitVecConst, got {other:?}"),
    }
}

#[test]
fn test_eval_bitvec_const_width_over_64_with_small_value_still_works() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // Width > 64 with a magnitude that fits u64 is exactly representable
    // (Value::BitVec's Display zero-extends the u64 magnitude out to
    // `width` bits), so this must evaluate successfully, not error.
    let t = manager.mk_bitvec(num_bigint::BigInt::from(3), 128);
    let result = evaluator.eval(t, &manager);
    assert!(matches!(result, EvalResult::Ok(Value::BitVec(128, 3))));
}

#[test]
fn test_eval_mod_min_by_neg_one_errors_not_panic() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // `(mod i64::MIN -1)` triggers `i64::MIN.rem_euclid(-1)`, which
    // overflows and panics in BOTH debug and release. It must surface an
    // explicit error rather than aborting the process.
    let min = manager.mk_int(num_bigint::BigInt::from(i64::MIN));
    let neg_one = manager.mk_int(num_bigint::BigInt::from(-1));
    let m = manager.mk_mod(min, neg_one);
    let result = evaluator.eval(m, &manager);
    match result {
        EvalResult::Error(_) => {}
        other => panic!("expected EvalResult::Error for (mod i64::MIN -1), got {other:?}"),
    }
}

#[test]
fn test_eval_mod_euclidean_still_works() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // (mod -7 2) = 1 under Euclidean semantics.
    let a = manager.mk_int(num_bigint::BigInt::from(-7));
    let b = manager.mk_int(num_bigint::BigInt::from(2));
    let m = manager.mk_mod(a, b);
    assert!(matches!(
        evaluator.eval(m, &manager),
        EvalResult::Ok(Value::Int(1))
    ));
}

// Regression tests for: "arithmetic overflow reachable from model
// evaluation".
//
// Every one of these evaluates a *representable* input to an unrepresentable
// result. Untreated, each aborts the process in a debug build (`attempt to
// … with overflow`) and — worse — silently wraps in a release build, handing
// the caller a model value that is simply wrong. The contract these pin down
// is the one the `IntConst`/`BitVecConst` arms already follow: an
// unrepresentable result is an explicit `EvalResult::Error`, in *both*
// profiles.

/// An Int-sorted variable pinned to `value` in the model.
fn int_var(manager: &mut TermManager, model: &mut Model, name: &str, value: i64) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let term = manager.mk_var(name, int_sort);
    model.assign(term, Value::Int(value));
    term
}

/// A Real-sorted variable pinned to `value` in the model.
fn real_var(manager: &mut TermManager, model: &mut Model, name: &str, value: Rational64) -> TermId {
    let real_sort = manager.sorts.real_sort;
    let term = manager.mk_var(name, real_sort);
    model.assign(term, Value::Rational(value));
    term
}

/// Assert that evaluating `build(..)` reports an explicit error.
fn assert_overflow_error<F>(what: &str, build: F)
where
    F: FnOnce(&mut TermManager, &mut Model) -> TermId,
{
    let mut manager = TermManager::new();
    let mut model = Model::new();
    let term = build(&mut manager, &mut model);
    let mut evaluator = ModelEvaluator::new(&model);
    match evaluator.eval(term, &manager) {
        EvalResult::Error(_) => {}
        other => panic!("{what}: expected EvalResult::Error, got {other:?}"),
    }
}

#[test]
fn test_eval_sub_int_overflow_errors_not_panic() {
    assert_overflow_error("(- i64::MIN 1)", |manager, model| {
        let a = int_var(manager, model, "a", i64::MIN);
        let b = int_var(manager, model, "b", 1);
        manager.mk_sub(a, b)
    });
}

#[test]
fn test_eval_sub_real_overflow_errors_not_panic() {
    assert_overflow_error(
        "(- (/ 1 i64::MAX) (/ 1 (- i64::MAX 1)))",
        |manager, model| {
            let a = real_var(manager, model, "a", Rational64::new(1, i64::MAX));
            let b = real_var(manager, model, "b", Rational64::new(1, i64::MAX - 1));
            manager.mk_sub(a, b)
        },
    );
}

#[test]
fn test_eval_neg_int_min_errors_not_panic() {
    assert_overflow_error("(- i64::MIN)", |manager, model| {
        let a = int_var(manager, model, "a", i64::MIN);
        manager.mk_neg(a)
    });
}

#[test]
fn test_eval_neg_real_min_numerator_errors_not_panic() {
    assert_overflow_error("(- (/ i64::MIN 3))", |manager, model| {
        let a = real_var(manager, model, "a", Rational64::new(i64::MIN, 3));
        manager.mk_neg(a)
    });
}

#[test]
fn test_eval_add_fold_overflow_errors_not_panic() {
    assert_overflow_error("(+ i64::MAX 1)", |manager, model| {
        let a = int_var(manager, model, "a", i64::MAX);
        let b = int_var(manager, model, "b", 1);
        manager.mk_add([a, b])
    });
}

#[test]
fn test_eval_mul_fold_overflow_errors_not_panic() {
    assert_overflow_error("(* 4e9 4e9)", |manager, model| {
        let a = int_var(manager, model, "a", 4_000_000_000);
        let b = int_var(manager, model, "b", 4_000_000_000);
        manager.mk_mul([a, b])
    });
}

#[test]
fn test_eval_real_div_overflow_errors_not_panic() {
    assert_overflow_error("(/ i64::MAX (/ 1 i64::MAX))", |manager, model| {
        let a = real_var(manager, model, "a", Rational64::from_integer(i64::MAX));
        let b = real_var(manager, model, "b", Rational64::new(1, i64::MAX));
        manager.mk_div(a, b)
    });
}

#[test]
fn test_eval_bvextract_width_overflow_errors_not_panic() {
    // `((_ extract u32::MAX 0) x)` computes `high - low + 1`, which overflows
    // u32 — a debug panic, and a wrap to width 0 in release.
    assert_overflow_error("((_ extract u32::MAX 0) x)", |manager, model| {
        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        model.assign(x, Value::BitVec(8, 5));
        manager.intern_term(
            TermKind::BvExtract {
                high: u32::MAX,
                low: 0,
                arg: x,
            },
            bv_sort,
        )
    });
}

#[test]
fn test_eval_arithmetic_near_the_limit_still_works() {
    // The checks must not turn a representable result into an error.
    let mut manager = TermManager::new();
    let mut model = Model::new();
    let a = int_var(&mut manager, &mut model, "a", i64::MAX);
    let b = int_var(&mut manager, &mut model, "b", 1);
    let sub = manager.mk_sub(a, b);
    let neg = manager.mk_neg(b);
    let add = manager.mk_add([b, b]);
    let mul = manager.mk_mul([a, b]);
    let mut evaluator = ModelEvaluator::new(&model);
    for (term, expected) in [(sub, i64::MAX - 1), (neg, -1), (add, 2), (mul, i64::MAX)] {
        match evaluator.eval(term, &manager) {
            EvalResult::Ok(Value::Int(n)) => assert_eq!(n, expected),
            other => panic!("expected Ok(Int({expected})), got {other:?}"),
        }
    }
}

// Regression tests for: "`=` compared `Value`s structurally, so two shapes of
// the same Real number were unequal".
//
// A Real *literal* evaluates to `Value::Rational(1/1)`, while every *computed*
// Real result goes through `combine::from_rational`, which reports an integral
// rational as `Value::Int`. Under the derived `PartialEq` the two were
// different values, so a true equality between them evaluated to `false` —
// a wrong answer on well-sorted, single-sort input, not merely a cosmetic
// mismatch. The same `==` backs `distinct`, `select`'s index lookup and
// `FuncInterp::evaluate`.

/// `(+ 0.5 0.5)` — a Real-sorted term whose value normalizes to `Value::Int`.
fn half_plus_half(manager: &mut TermManager) -> TermId {
    let half = manager.mk_real(Rational64::new(1, 2));
    manager.mk_add([half, half])
}

#[test]
fn test_real_literal_and_computed_real_still_have_different_shapes() {
    // Pins the *premise* of the bug: the two operands really do evaluate to
    // different `Value` variants, which is what the derived `PartialEq` keyed
    // on. If this ever stops holding, the bridge below has become moot rather
    // than wrong.
    let mut manager = TermManager::new();
    let one = manager.mk_real(Rational64::from_integer(1));
    let sum = half_plus_half(&mut manager);
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);
    assert!(matches!(
        evaluator.eval(one, &manager),
        EvalResult::Ok(Value::Rational(_))
    ));
    assert!(matches!(
        evaluator.eval(sum, &manager),
        EvalResult::Ok(Value::Int(1))
    ));
}

#[test]
fn test_eval_eq_real_literal_equals_computed_real() {
    let mut manager = TermManager::new();
    let one = manager.mk_real(Rational64::from_integer(1));
    let sum = half_plus_half(&mut manager);
    let eq = manager.mk_eq(one, sum);
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);
    match evaluator.eval(eq, &manager) {
        EvalResult::Ok(Value::Bool(true)) => {}
        other => panic!("expected (= 1.0 (+ 0.5 0.5)) to be true, got {other:?}"),
    }
}

#[test]
fn test_eval_distinct_real_literal_and_computed_real_is_false() {
    let mut manager = TermManager::new();
    let one = manager.mk_real(Rational64::from_integer(1));
    let sum = half_plus_half(&mut manager);
    let distinct = manager.mk_distinct([one, sum]);
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);
    match evaluator.eval(distinct, &manager) {
        EvalResult::Ok(Value::Bool(false)) => {}
        other => panic!("expected (distinct 1.0 (+ 0.5 0.5)) to be false, got {other:?}"),
    }
}

#[test]
fn test_eval_select_finds_a_store_whose_index_has_the_other_shape() {
    // `(select (store arr 1.0 7) (+ 0.5 0.5))` stores at `Rational(1/1)` and
    // reads at `Int(1)`. Structural index comparison missed the entry and
    // silently returned the array's default instead.
    let mut manager = TermManager::new();
    let real_sort = manager.sorts.real_sort;
    let int_sort = manager.sorts.int_sort;
    let array_sort = manager.sorts.array(real_sort, int_sort);
    let arr = manager.mk_var("arr", array_sort);
    let one = manager.mk_real(Rational64::from_integer(1));
    let seven = manager.mk_int(num_bigint::BigInt::from(7));
    let stored = manager.mk_store(arr, one, seven);
    let sum = half_plus_half(&mut manager);
    let select = manager.mk_select(stored, sum);

    let mut model = Model::new();
    model.assign(arr, Value::Array(Box::new(Value::Int(-1)), Vec::new()));
    let mut evaluator = ModelEvaluator::new(&model);
    match evaluator.eval(select, &manager) {
        EvalResult::Ok(Value::Int(7)) => {}
        other => panic!("expected the stored 7, got {other:?}"),
    }
}

// Regression tests for: "ModelEvaluator cannot evaluate BV
// comparisons/shifts/div, strings, arrays" — truth tables for the
// newly implemented BV division/remainder/shift/comparison/
// concat/extract, array select/store, and string ops.

fn bv(manager: &mut TermManager, value: i64, width: u32) -> TermId {
    manager.mk_bitvec(num_bigint::BigInt::from(value), width)
}

#[test]
fn test_eval_bvudiv_and_bvurem_truth_table() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let a = bv(&mut manager, 7, 4);
    let b = bv(&mut manager, 2, 4);
    let udiv = manager.mk_bv_udiv(a, b);
    let urem = manager.mk_bv_urem(a, b);
    assert!(matches!(
        evaluator.eval(udiv, &manager),
        EvalResult::Ok(Value::BitVec(4, 3))
    ));
    assert!(matches!(
        evaluator.eval(urem, &manager),
        EvalResult::Ok(Value::BitVec(4, 1))
    ));
}

#[test]
fn test_eval_bvudiv_by_zero_is_all_ones() {
    // SMT-LIB total semantics: (bvudiv x #b0000) = #b1111.
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let a = bv(&mut manager, 5, 4);
    let zero = bv(&mut manager, 0, 4);
    let udiv = manager.mk_bv_udiv(a, zero);
    assert!(matches!(
        evaluator.eval(udiv, &manager),
        EvalResult::Ok(Value::BitVec(4, 15))
    ));
}

#[test]
fn test_eval_bvurem_by_zero_is_dividend() {
    // SMT-LIB total semantics: (bvurem x #b0000) = x.
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let a = bv(&mut manager, 5, 4);
    let zero = bv(&mut manager, 0, 4);
    let urem = manager.mk_bv_urem(a, zero);
    assert!(matches!(
        evaluator.eval(urem, &manager),
        EvalResult::Ok(Value::BitVec(4, 5))
    ));
}

#[test]
fn test_eval_bvsdiv_and_bvsrem_truth_table() {
    // -4 (0b1100) / 2 (0b0010) = -2 (0b1110) in 4-bit two's complement.
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let a = bv(&mut manager, 12, 4); // -4
    let b = bv(&mut manager, 2, 4);
    let sdiv = manager.mk_bv_sdiv(a, b);
    assert!(matches!(
        evaluator.eval(sdiv, &manager),
        EvalResult::Ok(Value::BitVec(4, 14)) // -2
    ));

    // -7 (0b1001) srem 2 (0b0010) = -1 (0b1111): truncating division
    // rounds -7/2 toward zero to -3, remainder -7 - 2*(-3) = -1.
    let c = bv(&mut manager, 9, 4); // -7
    let srem = manager.mk_bv_srem(c, b);
    assert!(matches!(
        evaluator.eval(srem, &manager),
        EvalResult::Ok(Value::BitVec(4, 15)) // -1
    ));
}

#[test]
fn test_eval_bvsdiv_by_zero_depends_on_dividend_sign() {
    // SMT-LIB: (bvsdiv s #b0) = all-ones if s is non-negative, else 1.
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let zero = bv(&mut manager, 0, 4);
    let pos = bv(&mut manager, 4, 4); // non-negative
    let neg = bv(&mut manager, 12, 4); // -4, negative

    let sdiv_pos = manager.mk_bv_sdiv(pos, zero);
    assert!(matches!(
        evaluator.eval(sdiv_pos, &manager),
        EvalResult::Ok(Value::BitVec(4, 15))
    ));

    let sdiv_neg = manager.mk_bv_sdiv(neg, zero);
    assert!(matches!(
        evaluator.eval(sdiv_neg, &manager),
        EvalResult::Ok(Value::BitVec(4, 1))
    ));
}

#[test]
fn test_eval_bvsrem_by_zero_is_dividend() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let a = bv(&mut manager, 12, 4); // -4
    let zero = bv(&mut manager, 0, 4);
    let srem = manager.mk_bv_srem(a, zero);
    assert!(matches!(
        evaluator.eval(srem, &manager),
        EvalResult::Ok(Value::BitVec(4, 12))
    ));
}

#[test]
fn test_eval_bv_shifts_truth_table() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // shl(0b0011, 1) = 0b0110
    let three = bv(&mut manager, 3, 4);
    let one = bv(&mut manager, 1, 4);
    let shl = manager.mk_bv_shl(three, one);
    assert!(matches!(
        evaluator.eval(shl, &manager),
        EvalResult::Ok(Value::BitVec(4, 6))
    ));

    // shl by an amount >= width zeroes the result.
    let five = bv(&mut manager, 5, 4);
    let shl_oob = manager.mk_bv_shl(three, five);
    assert!(matches!(
        evaluator.eval(shl_oob, &manager),
        EvalResult::Ok(Value::BitVec(4, 0))
    ));

    // lshr(0b1000, 1) = 0b0100 (zero-filled).
    let eight = bv(&mut manager, 8, 4);
    let lshr = manager.mk_bv_lshr(eight, one);
    assert!(matches!(
        evaluator.eval(lshr, &manager),
        EvalResult::Ok(Value::BitVec(4, 4))
    ));

    // ashr(0b1000, 1) = 0b1100 (sign-filled: 0b1000 is -8, -8>>1 = -4).
    let ashr = manager.mk_bv_ashr(eight, one);
    assert!(matches!(
        evaluator.eval(ashr, &manager),
        EvalResult::Ok(Value::BitVec(4, 12))
    ));

    // ashr by an amount >= width on a negative value fills with all
    // ones (saturates toward -1).
    let ashr_oob = manager.mk_bv_ashr(eight, five);
    assert!(matches!(
        evaluator.eval(ashr_oob, &manager),
        EvalResult::Ok(Value::BitVec(4, 15))
    ));
}

#[test]
fn test_eval_bv_comparisons_signed_vs_unsigned() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // 0b1000 = 8 unsigned, -8 signed; 0b0001 = 1 both ways.
    let eight = bv(&mut manager, 8, 4);
    let one = bv(&mut manager, 1, 4);

    let ult = manager.mk_bv_ult(one, eight);
    assert!(matches!(
        evaluator.eval(ult, &manager),
        EvalResult::Ok(Value::Bool(true))
    ));

    let slt = manager.mk_bv_slt(eight, one);
    assert!(
        matches!(
            evaluator.eval(slt, &manager),
            EvalResult::Ok(Value::Bool(true))
        ),
        "signed: -8 < 1"
    );

    let slt_reversed = manager.mk_bv_slt(one, eight);
    assert!(
        matches!(
            evaluator.eval(slt_reversed, &manager),
            EvalResult::Ok(Value::Bool(false))
        ),
        "signed: 1 < -8 is false"
    );

    let sle_eq = manager.mk_bv_sle(eight, eight);
    assert!(matches!(
        evaluator.eval(sle_eq, &manager),
        EvalResult::Ok(Value::Bool(true))
    ));
}

#[test]
fn test_eval_bvconcat_computes_combined_value_and_width() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let lhs = bv(&mut manager, 0b10, 2);
    let rhs = bv(&mut manager, 0b01, 2);
    let concat = manager
        .try_mk_bv_concat(lhs, rhs)
        .expect("two bit-vector literals concat cleanly");
    assert!(matches!(
        evaluator.eval(concat, &manager),
        EvalResult::Ok(Value::BitVec(4, 0b1001))
    ));
}

#[test]
fn test_eval_bvconcat_result_wider_than_64_bits_errors() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let lhs = bv(&mut manager, 1, 40);
    let rhs = bv(&mut manager, 1, 30);
    // Interned raw: `try_mk_bv_concat` folds two literals into a single wide
    // literal, and the point here is the evaluator's `BvConcat` arm — it
    // must refuse a result too wide for its 64-bit `Value::BitVec` rather
    // than silently truncate it.
    let sort = manager.sorts.bitvec(70);
    let concat = manager.intern_term(TermKind::BvConcat(lhs, rhs), sort);
    match evaluator.eval(concat, &manager) {
        EvalResult::Error(_) => {}
        other => panic!("expected EvalResult::Error for a >64-bit concat, got {other:?}"),
    }
}

#[test]
fn test_eval_bvextract_selects_the_expected_bit_range() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    // 181 = 0b10110101; extract [5:2] = 0b1101 = 13.
    let arg = bv(&mut manager, 181, 8);
    let extract = manager.mk_bv_extract(5, 2, arg);
    assert!(matches!(
        evaluator.eval(extract, &manager),
        EvalResult::Ok(Value::BitVec(4, 13))
    ));
}

#[test]
fn test_eval_select_and_store_array_roundtrip() {
    let mut manager = TermManager::new();
    let mut model = Model::new();

    let int_sort = manager.sorts.int_sort;
    let array_sort = manager.sorts.array(int_sort, int_sort);
    let arr = manager.mk_var("arr", array_sort);
    model.assign(arr, Value::Array(Box::new(Value::Int(0)), Vec::new()));

    let five = manager.mk_int(num_bigint::BigInt::from(5));
    let forty_two = manager.mk_int(num_bigint::BigInt::from(42));
    let stored = manager.mk_store(arr, five, forty_two);

    let mut evaluator = ModelEvaluator::new(&model);

    let select_stored = manager.mk_select(stored, five);
    assert!(matches!(
        evaluator.eval(select_stored, &manager),
        EvalResult::Ok(Value::Int(42))
    ));

    // An index never stored falls back to the array's default value.
    let six = manager.mk_int(num_bigint::BigInt::from(6));
    let select_default = manager.mk_select(stored, six);
    assert!(matches!(
        evaluator.eval(select_default, &manager),
        EvalResult::Ok(Value::Int(0))
    ));

    // A second store to the same index shadows the first.
    let hundred = manager.mk_int(num_bigint::BigInt::from(100));
    let stored_again = manager.mk_store(stored, five, hundred);
    let select_latest = manager.mk_select(stored_again, five);
    assert!(matches!(
        evaluator.eval(select_latest, &manager),
        EvalResult::Ok(Value::Int(100))
    ));
}

#[test]
fn test_eval_string_ops_truth_table() {
    let mut manager = TermManager::new();
    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);

    let hello = manager.mk_string_lit("hello");
    let world = manager.mk_string_lit(" world");

    let len = manager.mk_str_len(hello);
    assert!(matches!(
        evaluator.eval(len, &manager),
        EvalResult::Ok(Value::Int(5))
    ));

    let concat = manager.mk_str_concat(hello, world);
    match &evaluator.eval(concat, &manager) {
        EvalResult::Ok(Value::String(s)) => assert_eq!(s, "hello world"),
        other => panic!("expected concatenated string, got {other:?}"),
    }

    let one = manager.mk_int(num_bigint::BigInt::from(1));
    let at = manager.mk_str_at(hello, one);
    match &evaluator.eval(at, &manager) {
        EvalResult::Ok(Value::String(s)) => assert_eq!(s, "e"),
        other => panic!("expected \"e\", got {other:?}"),
    }

    let ell = manager.mk_string_lit("ell");
    let contains_true = manager.mk_str_contains(hello, ell);
    assert!(matches!(
        evaluator.eval(contains_true, &manager),
        EvalResult::Ok(Value::Bool(true))
    ));

    let xyz = manager.mk_string_lit("xyz");
    let contains_false = manager.mk_str_contains(hello, xyz);
    assert!(matches!(
        evaluator.eval(contains_false, &manager),
        EvalResult::Ok(Value::Bool(false))
    ));

    let three = manager.mk_int(num_bigint::BigInt::from(3));
    let substr = manager.mk_str_substr(hello, one, three);
    match &evaluator.eval(substr, &manager) {
        EvalResult::Ok(Value::String(s)) => assert_eq!(s, "ell"),
        other => panic!("expected \"ell\", got {other:?}"),
    }

    let zero = manager.mk_int(num_bigint::BigInt::from(0));
    let l = manager.mk_string_lit("l");
    let indexof = manager.mk_str_indexof(hello, l, zero);
    assert!(matches!(
        evaluator.eval(indexof, &manager),
        EvalResult::Ok(Value::Int(2))
    ));

    // Not found -> -1.
    let indexof_missing = manager.mk_str_indexof(hello, xyz, zero);
    assert!(matches!(
        evaluator.eval(indexof_missing, &manager),
        EvalResult::Ok(Value::Int(-1))
    ));
}

/// `str.<` / `str.<=` / `str.to_code` / `str.from_code` evaluate under a
/// model, including through a variable the model binds.
#[test]
fn test_eval_string_order_and_char_codes() {
    let mut manager = TermManager::new();
    let string_sort = manager.sorts.string_sort();
    let x = manager.mk_var("x", string_sort);
    let abd = manager.mk_string_lit("abd");

    let mut model = Model::new();
    model.assign(x, Value::String("abc".to_string()));
    let mut evaluator = ModelEvaluator::new(&model);

    // The model binds `x` to "abc", so the order is decided concretely.
    let lt = manager.mk_str_lt(x, abd);
    assert!(matches!(
        evaluator.eval(lt, &manager),
        EvalResult::Ok(Value::Bool(true))
    ));
    let le = manager.mk_str_le(abd, x);
    assert!(matches!(
        evaluator.eval(le, &manager),
        EvalResult::Ok(Value::Bool(false))
    ));

    // `str.to_code` is -1 for a non-singleton string.
    let to_code_x = manager.mk_str_to_code(x);
    assert!(matches!(
        evaluator.eval(to_code_x, &manager),
        EvalResult::Ok(Value::Int(-1))
    ));

    // A surrogate is unrepresentable, so the term stays undefined rather
    // than evaluating to a wrong value.
    let int_sort = manager.sorts.int_sort;
    let n = manager.mk_var("n", int_sort);
    let mut model2 = Model::new();
    model2.assign(n, Value::Int(0xD800));
    let mut evaluator2 = ModelEvaluator::new(&model2);
    let from_code = manager.mk_str_from_code(n);
    assert!(matches!(
        evaluator2.eval(from_code, &manager),
        EvalResult::Undefined(_)
    ));
}

// ===== Deep-nesting regression tests =====
//
// Regression tests for: "ModelEvaluator::eval recurses once per term-nesting
// level and has no depth guard, so a deep term aborts the process with
// `fatal runtime error: stack overflow` instead of returning a result."
//
// Every one of these builds its term with a plain `for` loop — a recursive
// builder would overflow before the evaluation under test even started — and
// runs the evaluation on a thread with an explicitly small (128 KiB) stack, the
// size an embedder's worker thread typically gets. A stack overflow aborts the
// whole process, so "the call returned at all" *is* the assertion; the value
// checks on top of that make sure the iterative driver computes the same
// answer the recursive one did at shallow depths.
//
// Before the conversion, on this 1 MiB stack, a `Not` chain aborted somewhere
// between 1500 and 2000 levels in release and between 3000 and 3500 in debug.

/// Stack size every deep test runs under: the ~1 MiB a non-main thread gets by
/// default on most platforms, and far less than a libtest thread's.
const SMALL_STACK: usize = 1 << 17;

/// A depth well past anything a native-stack recursion could survive.
const DEEP: usize = 12_500;

/// A merely large depth, kept cheap enough to run in every profile.
const LARGE: usize = 5_000;

/// Run `body` on a thread with a deliberately small stack and return its value.
///
/// A stack overflow inside `body` aborts the process rather than unwinding, so
/// this helper cannot turn one into a test failure — that is the point. The
/// test run itself fails, loudly, which is the signal.
fn on_small_stack<T, F>(body: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(body)
        .expect("spawn small-stack thread")
        .join()
        .expect("small-stack thread panicked")
}

/// `(not (not ... (not p)))`, `depth` levels of it.
///
/// Interned raw because `mk_not` folds double negation away, which would leave
/// a depth-1 term instead of the deep one under test.
fn deep_not_chain(manager: &mut TermManager, depth: usize) -> (TermId, TermId) {
    let bool_sort = manager.sorts.bool_sort;
    let leaf = manager.mk_var("p", bool_sort);
    let mut term = leaf;
    for _ in 0..depth {
        term = manager.intern_term(TermKind::Not(term), bool_sort);
    }
    (term, leaf)
}

#[test]
fn test_deep_not_chain_evaluates_on_a_small_stack() {
    for depth in [LARGE, DEEP] {
        let (value, cached) = on_small_stack(move || {
            let mut manager = TermManager::new();
            let (term, leaf) = deep_not_chain(&mut manager, depth);
            let mut model = Model::new();
            model.assign(leaf, Value::Bool(true));
            let mut evaluator = ModelEvaluator::new(&model);
            let result = evaluator.eval(term, &manager);
            (result, evaluator.cache_size())
        });
        // An even number of negations is the identity.
        let expected = depth % 2 == 0;
        match value {
            EvalResult::Ok(Value::Bool(b)) => assert_eq!(b, expected, "depth {depth}"),
            other => panic!("depth {depth}: expected Ok(Bool({expected})), got {other:?}"),
        }
        // One cache entry per `not` plus one for the leaf: proof the chain
        // really was `depth` levels deep and that every level was visited,
        // rather than folded away by a term builder.
        assert_eq!(cached, depth + 1, "depth {depth}: visited term count");
    }
}

#[test]
fn test_deep_not_chain_without_cache_evaluates_on_a_small_stack() {
    // The cacheless evaluator visits the same chain without ever short-cutting
    // through `EvalCache`, so it exercises the driver on its own.
    let value = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (term, leaf) = deep_not_chain(&mut manager, DEEP);
        let mut model = Model::new();
        model.assign(leaf, Value::Bool(false));
        let mut evaluator = ModelEvaluator::without_cache(&model);
        let result = evaluator.eval(term, &manager);
        assert_eq!(evaluator.cache_size(), 0, "caching must stay disabled");
        result
    });
    assert!(matches!(value, EvalResult::Ok(Value::Bool(false))));
}

#[test]
fn test_deep_bvadd_chain_evaluates_on_a_small_stack() {
    // A binary, eagerly-evaluated operator nested `DEEP` levels: every level
    // pushes a frame that must wait for two operands.
    let value = on_small_stack(|| {
        let mut manager = TermManager::new();
        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let one = manager.mk_bitvec(num_bigint::BigInt::from(1), 8);
        let mut term = x;
        for _ in 0..DEEP {
            term = manager.mk_bv_add(term, one);
        }
        let mut model = Model::new();
        model.assign(x, Value::BitVec(8, 5));
        let mut evaluator = ModelEvaluator::new(&model);
        evaluator.eval(term, &manager)
    });
    // 5 + DEEP mod 2^8.
    let expected = (5u64 + DEEP as u64) % 256;
    match value {
        EvalResult::Ok(Value::BitVec(8, v)) => assert_eq!(v, expected),
        other => panic!("expected Ok(BitVec(8, {expected})), got {other:?}"),
    }
}

#[test]
fn test_deep_nested_and_evaluates_on_a_small_stack() {
    // `and` is n-ary and short-circuiting, so it takes the driver's
    // `Op::Connective` path rather than the eager one. Interned raw because
    // `mk_and` flattens nested conjunctions into a single wide one.
    let value = on_small_stack(|| {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let p = manager.mk_var("p", bool_sort);
        let q = manager.mk_var("q", bool_sort);
        let mut term = p;
        for _ in 0..DEEP {
            let args: SmallVec<[TermId; 4]> = SmallVec::from_slice(&[term, q]);
            term = manager.intern_term(TermKind::And(args), bool_sort);
        }
        let mut model = Model::new();
        model.assign(p, Value::Bool(true));
        model.assign(q, Value::Bool(true));
        let mut evaluator = ModelEvaluator::new(&model);
        evaluator.eval(term, &manager)
    });
    assert!(matches!(value, EvalResult::Ok(Value::Bool(true))));
}

#[test]
fn test_deep_nested_add_evaluates_on_a_small_stack() {
    // `+` folds into a running accumulator as operands arrive, so it takes the
    // driver's `Op::Arith` path. Interned raw because `mk_add` flattens.
    let value = on_small_stack(|| {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let one = manager.mk_int(num_bigint::BigInt::from(1));
        let mut term = x;
        for _ in 0..DEEP {
            let args: SmallVec<[TermId; 4]> = SmallVec::from_slice(&[term, one]);
            term = manager.intern_term(TermKind::Add(args), int_sort);
        }
        let mut model = Model::new();
        model.assign(x, Value::Int(7));
        let mut evaluator = ModelEvaluator::new(&model);
        evaluator.eval(term, &manager)
    });
    let expected = 7 + DEEP as i64;
    match value {
        EvalResult::Ok(Value::Int(n)) => assert_eq!(n, expected),
        other => panic!("expected Ok(Int({expected})), got {other:?}"),
    }
}

#[test]
fn test_deep_nested_store_select_evaluates_on_a_small_stack() {
    // A three-operand eager frame (`store`) nested deeply. The depth is kept
    // modest because each `store` copies the array's exception list, which is
    // quadratic in the chain length regardless of how the walk is driven.
    const STORE_DEPTH: usize = 1_000;
    let value = on_small_stack(|| {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let arr = manager.mk_var("arr", array_sort);
        let key = manager.mk_int(num_bigint::BigInt::from(1));
        let mut term = arr;
        for i in 0..STORE_DEPTH {
            let value = manager.mk_int(num_bigint::BigInt::from(i as i64));
            term = manager.mk_store(term, key, value);
        }
        let select = manager.mk_select(term, key);
        let mut model = Model::new();
        model.assign(arr, Value::Array(Box::new(Value::Int(-1)), Vec::new()));
        let mut evaluator = ModelEvaluator::new(&model);
        evaluator.eval(select, &manager)
    });
    // The newest store to the key wins.
    let expected = STORE_DEPTH as i64 - 1;
    match value {
        EvalResult::Ok(Value::Int(n)) => assert_eq!(n, expected),
        other => panic!("expected Ok(Int({expected})), got {other:?}"),
    }
}

/// How many `Value::Array` levels `value` nests through its exception values,
/// counted with a loop rather than a recursive walk.
fn store_nesting_depth(value: &Value) -> usize {
    let mut depth = 0usize;
    let mut node = value;
    while let Value::Array(_, excs) = node {
        depth += 1;
        match excs.first() {
            Some((_, inner)) => node = inner,
            None => break,
        }
    }
    depth
}

#[test]
fn test_store_chain_nests_in_value_position_but_stays_flat_in_array_position() {
    // The mechanism that makes a deeply nested `Value` reachable from
    // evaluation at all: `store`'s combiner keeps the base array's default
    // and *appends* `(index, value)`, so a chain threaded through the VALUE
    // operand nests one `Value::Array` per level, while the usual chain
    // threaded through the ARRAY operand only lengthens a flat exception
    // list. Well-sorted, which is why each level needs its own array sort
    // `(Array Int S_{k-1})`.
    const DEPTH: usize = 200;
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let key = manager.mk_int(num_bigint::BigInt::from(1));
    let mut model = Model::new();

    let mut elem_sort = int_sort;
    let mut nested_term = manager.mk_int(num_bigint::BigInt::from(0));
    for level in 0..DEPTH {
        let array_sort = manager.sorts.array(int_sort, elem_sort);
        let base = manager.mk_var(&format!("a{level}"), array_sort);
        model.assign(base, Value::Array(Box::new(Value::Int(0)), Vec::new()));
        nested_term = manager.mk_store(base, key, nested_term);
        elem_sort = array_sort;
    }

    let flat_sort = manager.sorts.array(int_sort, int_sort);
    let flat_base = manager.mk_var("flat", flat_sort);
    model.assign(flat_base, Value::Array(Box::new(Value::Int(0)), Vec::new()));
    let mut flat_term = flat_base;
    for level in 0..DEPTH {
        let stored = manager.mk_int(num_bigint::BigInt::from(level as i64));
        flat_term = manager.mk_store(flat_term, key, stored);
    }

    let mut evaluator = ModelEvaluator::new(&model);
    match evaluator.eval(nested_term, &manager) {
        EvalResult::Ok(ref value) => assert_eq!(store_nesting_depth(value), DEPTH),
        other => panic!("expected a nested array value, got {other:?}"),
    }
    match evaluator.eval(flat_term, &manager) {
        EvalResult::Ok(Value::Array(_, ref excs)) => {
            assert_eq!(excs.len(), DEPTH, "the array-position chain must stay flat");
        }
        other => panic!("expected a flat array value, got {other:?}"),
    }
}

#[test]
fn test_deep_array_value_in_model_evaluates_on_a_small_stack() {
    // `eval` hands the model's value back by *clone* and caches another
    // clone, and every one of those — plus the final drop of the model, the
    // cache and the result — used to recurse once per nesting level.
    // 6 250 is comfortably past where the derived traits aborted.
    const DEPTH: usize = 6_250;
    let (depth, tail) = on_small_stack(|| {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let arr = manager.mk_var("arr", array_sort);

        let mut value = Value::Int(7);
        for level in 0..DEPTH {
            value = Value::Array(
                Box::new(Value::Int(-1)),
                vec![(Value::Int(level as i64), value)],
            );
        }
        let mut model = Model::new();
        model.assign(arr, value);

        let mut evaluator = ModelEvaluator::new(&model);
        let result = evaluator.eval(arr, &manager);
        let measured = match &result {
            EvalResult::Ok(value) => {
                let mut node = value;
                let mut depth = 0usize;
                while let Value::Array(_, excs) = node {
                    depth += 1;
                    match excs.first() {
                        Some((_, inner)) => node = inner,
                        None => break,
                    }
                }
                (depth, node.clone())
            }
            other => panic!("expected the array value back, got {other:?}"),
        };
        // `result`, the cache and the model are all dropped here, inside the
        // small-stack thread.
        measured
    });
    assert_eq!(depth, DEPTH);
    assert_eq!(tail, Value::Int(7));
}

// ===== Short-circuit contract =====

/// `ite` must evaluate only the taken branch: a failure hiding in the untaken
/// one has to stay invisible, at any nesting depth.
#[test]
fn test_ite_untaken_branch_failure_stays_invisible() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let int_sort = manager.sorts.int_sort;

    let cond = manager.mk_var("c", bool_sort);
    let taken = manager.mk_int(num_bigint::BigInt::from(11));
    // An unassigned variable evaluates to `Undefined`.
    let undefined = manager.mk_var("missing", int_sort);
    // A constant too wide for `Value::Int` evaluates to `Error`.
    let erroring = manager.mk_int(num_bigint::BigInt::from(2u64).pow(100));

    let ite_undefined = manager.mk_ite(cond, taken, undefined);
    let ite_erroring = manager.mk_ite(cond, taken, erroring);

    let mut model = Model::new();
    model.assign(cond, Value::Bool(true));
    let mut evaluator = ModelEvaluator::new(&model);

    assert!(
        matches!(
            evaluator.eval(ite_undefined, &manager),
            EvalResult::Ok(Value::Int(11))
        ),
        "the untaken Undefined branch must not be evaluated"
    );
    assert!(
        matches!(
            evaluator.eval(ite_erroring, &manager),
            EvalResult::Ok(Value::Int(11))
        ),
        "the untaken Error branch must not be evaluated"
    );

    // The untaken branches were never evaluated, so they were never cached.
    assert!(
        evaluator.eval(undefined, &manager).value().is_none(),
        "the undefined branch must still be undefined"
    );
}

/// The same contract under deep nesting: a chain of `ite`s each of which hides
/// an undefined branch still returns the value the taken path leads to.
#[test]
fn test_deep_ite_untaken_branch_failure_stays_invisible() {
    let value = on_small_stack(|| {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let int_sort = manager.sorts.int_sort;

        let cond = manager.mk_var("c", bool_sort);
        let leaf = manager.mk_int(num_bigint::BigInt::from(3));
        let undefined = manager.mk_var("missing", int_sort);

        let mut term = leaf;
        for _ in 0..DEEP {
            term = manager.mk_ite(cond, term, undefined);
        }

        let mut model = Model::new();
        model.assign(cond, Value::Bool(true));
        let mut evaluator = ModelEvaluator::new(&model);
        evaluator.eval(term, &manager)
    });
    match value {
        EvalResult::Ok(Value::Int(3)) => {}
        other => panic!("expected Ok(Int(3)) through DEEP nested ites, got {other:?}"),
    }
}

/// `and` stops at its first `false` operand and `or` at its first `true` one,
/// so a failing operand behind that point stays invisible.
#[test]
fn test_connectives_short_circuit_before_a_failing_operand() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;

    let decided = manager.mk_var("d", bool_sort);
    let undefined = manager.mk_var("missing", bool_sort);
    // Interned raw: `mk_and`/`mk_or` fold a literal operand away, and the
    // point here is the evaluator's own left-to-right short-circuit.
    let conj: SmallVec<[TermId; 4]> = SmallVec::from_slice(&[decided, undefined]);
    let conjunction = manager.intern_term(TermKind::And(conj), bool_sort);
    let disj: SmallVec<[TermId; 4]> = SmallVec::from_slice(&[decided, undefined]);
    let disjunction = manager.intern_term(TermKind::Or(disj), bool_sort);

    let mut false_model = Model::new();
    false_model.assign(decided, Value::Bool(false));
    let mut evaluator = ModelEvaluator::new(&false_model);
    assert!(
        matches!(
            evaluator.eval(conjunction, &manager),
            EvalResult::Ok(Value::Bool(false))
        ),
        "and must stop at its first false operand"
    );

    let mut true_model = Model::new();
    true_model.assign(decided, Value::Bool(true));
    let mut evaluator = ModelEvaluator::new(&true_model);
    assert!(
        matches!(
            evaluator.eval(disjunction, &manager),
            EvalResult::Ok(Value::Bool(true))
        ),
        "or must stop at its first true operand"
    );

    // Conversely, an operand *before* the decision point still decides the
    // result: `and` with a leading true operand must reach the undefined one.
    let mut evaluator = ModelEvaluator::new(&true_model);
    assert!(
        matches!(
            evaluator.eval(conjunction, &manager),
            EvalResult::Undefined(_)
        ),
        "and must keep going while its operands are true"
    );
}

// ===== Failure-precedence contract =====
//
// The eager operators matched on a tuple of both operands' results, whose
// pattern order made `Undefined` outrank `Error` no matter which side it came
// from, and made the leftmost of two same-kind failures win. Those two rules
// decide which sub-term a caller is told about, so they are pinned here.

/// Build an operand pair that fails: an `Error` (an over-wide integer literal)
/// and an `Undefined` (an unassigned variable).
fn failing_operands(manager: &mut TermManager) -> (TermId, TermId) {
    let int_sort = manager.sorts.int_sort;
    let erroring = manager.mk_int(num_bigint::BigInt::from(2u64).pow(100));
    let undefined = manager.mk_var("missing", int_sort);
    (erroring, undefined)
}

#[test]
fn test_eager_binary_prefers_undefined_over_error_from_either_side() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let (erroring, undefined) = failing_operands(&mut manager);
    let model = Model::new();

    // Interned raw so the operand order under test is the one evaluated.
    let error_first = manager.intern_term(TermKind::Eq(erroring, undefined), bool_sort);
    let undefined_first = manager.intern_term(TermKind::Eq(undefined, erroring), bool_sort);

    let mut evaluator = ModelEvaluator::new(&model);
    assert!(
        matches!(
            evaluator.eval(error_first, &manager),
            EvalResult::Undefined(t) if t == undefined
        ),
        "Undefined outranks Error even when the Error operand comes first"
    );
    assert!(
        matches!(
            evaluator.eval(undefined_first, &manager),
            EvalResult::Undefined(t) if t == undefined
        ),
        "Undefined outranks Error when it comes first too"
    );
}

#[test]
fn test_eager_binary_reports_the_leftmost_error() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let model = Model::new();

    // Two distinct over-wide literals, so their messages are distinguishable.
    let left = manager.mk_int(num_bigint::BigInt::from(2u64).pow(100));
    let right = manager.mk_int(num_bigint::BigInt::from(2u64).pow(101));
    let eq = manager.intern_term(TermKind::Eq(left, right), bool_sort);

    let mut evaluator = ModelEvaluator::new(&model);
    match evaluator.eval(eq, &manager) {
        EvalResult::Error(message) => {
            let leftmost = num_bigint::BigInt::from(2u64).pow(100).to_string();
            assert!(
                message.contains(&leftmost),
                "expected the leftmost operand's error, got {message}"
            );
        }
        other => panic!("expected EvalResult::Error, got {other:?}"),
    }
}

#[test]
fn test_eager_ternary_prefers_undefined_over_error() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let array_sort = manager.sorts.array(int_sort, int_sort);
    let (erroring, undefined) = failing_operands(&mut manager);
    let model = Model::new();

    // `store` is the widest eager operator; its failure precedence has to
    // match the two-operand one.
    let store = manager.intern_term(TermKind::Store(erroring, erroring, undefined), array_sort);
    let mut evaluator = ModelEvaluator::new(&model);
    assert!(
        matches!(
            evaluator.eval(store, &manager),
            EvalResult::Undefined(t) if t == undefined
        ),
        "Undefined in the last operand still outranks Errors before it"
    );
}

/// The eager operators evaluate *every* operand, so a failure in the first one
/// does not stop the second from being evaluated (and cached).
#[test]
fn test_eager_binary_still_evaluates_the_second_operand_after_a_failure() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let int_sort = manager.sorts.int_sort;

    let erroring = manager.mk_int(num_bigint::BigInt::from(2u64).pow(100));
    let x = manager.mk_var("x", int_sort);
    let eq = manager.intern_term(TermKind::Eq(erroring, x), bool_sort);

    let mut model = Model::new();
    model.assign(x, Value::Int(4));
    let mut evaluator = ModelEvaluator::new(&model);

    assert!(matches!(evaluator.eval(eq, &manager), EvalResult::Error(_)));
    // `x` was evaluated despite the leading failure, so the model lookup for
    // it landed in the cache.
    assert_eq!(evaluator.cache_size(), 1);
}

// ===== `Apply` wired to `Model::func_interps` =====
//
// Regression tests for: "a model holding a complete `FuncInterp` for `f`
// still evaluates `(f x)` to `EvalResult::Undefined`". `TermKind::Apply`
// used to fall through the `open()` catch-all unconditionally -- so
// `Model::func_interps` was write-only from the evaluator's point of view --
// and because `Undefined` outranks `Error` and propagates through every
// eager frame, one `Apply` anywhere in a formula made the *whole* evaluation
// `Undefined`, no matter how completely the rest of the model was known.
//
// Regex operators (`re.++`, `str.to_re`, `re.union`, ...) lower to `Apply`
// too (`TermManager::mk_regex_op`), sharing the same term shape as
// uninterpreted function application. A `FuncInterp` never exists for one of
// those, so they fall to the same "genuinely absent interpretation ->
// `Undefined`" path as any other unmapped function -- deliberately, rather
// than being special-cased as a different kind of failure.

/// `TermKind::Apply`'s `func` field is a `Spur`, not a `TermId` -- every call
/// site of the same function shares one `Spur` but gets its own `TermId` --
/// so tests recover the exact `Spur` `Model::add_func_interp` must be keyed
/// on by reading it back off the built term, the same way any real caller
/// would have to.
fn apply_func_spur(manager: &TermManager, apply_term: TermId) -> Spur {
    match manager.get(apply_term).map(|t| &t.kind) {
        Some(TermKind::Apply { func, .. }) => *func,
        _ => panic!("expected an Apply term"),
    }
}

#[test]
fn test_eval_apply_with_no_stored_interp_is_undefined() {
    // Preserved behaviour: a function the model says nothing about at all
    // is a genuinely absent interpretation, not an error.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let one = manager.mk_int(num_bigint::BigInt::from(1));
    let g_1 = manager.mk_apply("g", [one], int_sort);

    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);
    assert!(matches!(
        evaluator.eval(g_1, &manager),
        EvalResult::Undefined(t) if t == g_1
    ));
}

#[test]
fn test_eval_apply_consults_stored_func_interp() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let one = manager.mk_int(num_bigint::BigInt::from(1));
    let two = manager.mk_int(num_bigint::BigInt::from(2));
    let f_1_2 = manager.mk_apply("f", [one, two], int_sort);
    let func = apply_func_spur(&manager, f_1_2);

    let mut fi = FuncInterp::new(2, Value::Int(-1));
    fi.add_entry(vec![Value::Int(1), Value::Int(2)], Value::Int(42));
    let mut model = Model::new();
    model.add_func_interp(func, fi);

    let mut evaluator = ModelEvaluator::new(&model);
    assert!(matches!(
        evaluator.eval(f_1_2, &manager),
        EvalResult::Ok(Value::Int(42))
    ));
}

/// An argument tuple not covered by any entry falls back to the stored
/// `FuncInterp`'s `else_value` -- `FuncInterp::evaluate`'s own, pre-existing,
/// total semantics -- rather than `Undefined`. Only a function with *no*
/// stored interpretation at all (the previous test) is `Undefined`.
#[test]
fn test_eval_apply_unmatched_entry_falls_back_to_else_value_not_undefined() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let one = manager.mk_int(num_bigint::BigInt::from(1));
    let three = manager.mk_int(num_bigint::BigInt::from(3));
    let f_1_3 = manager.mk_apply("f", [one, three], int_sort);
    let func = apply_func_spur(&manager, f_1_3);

    let mut fi = FuncInterp::new(2, Value::Int(0));
    fi.add_entry(vec![Value::Int(1), Value::Int(2)], Value::Int(42));
    let mut model = Model::new();
    model.add_func_interp(func, fi);

    let mut evaluator = ModelEvaluator::new(&model);
    assert!(matches!(
        evaluator.eval(f_1_3, &manager),
        EvalResult::Ok(Value::Int(0))
    ));
}

/// The operand is a `Var`, so the only way its value can reach
/// `FuncInterp::evaluate` is through a real evaluation (and, since caching is
/// on, a real cache insertion) -- not by comparing the raw `TermId`.
#[test]
fn test_eval_apply_evaluates_arguments_through_the_frame_machinery() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let f_x = manager.mk_apply("f", [x], int_sort);
    let func = apply_func_spur(&manager, f_x);

    let mut fi = FuncInterp::new(1, Value::Int(-1));
    fi.add_entry(vec![Value::Int(7)], Value::Int(70));
    let mut model = Model::new();
    model.assign(x, Value::Int(7));
    model.add_func_interp(func, fi);

    let mut evaluator = ModelEvaluator::new(&model);
    assert!(matches!(
        evaluator.eval(f_x, &manager),
        EvalResult::Ok(Value::Int(70))
    ));
    // Both `x` and `f_x` were cached, proof `x` was genuinely evaluated.
    assert_eq!(evaluator.cache_size(), 2);
}

/// `FuncInterp::evaluate` matches its entry table with `==`, which bridges
/// `Value::Int` and an integral `Value::Rational` (see `Value`'s `PartialEq`
/// in `model/mod.rs`). Confirms that bridge is live all the way through the
/// `Apply` arm: an entry keyed on a Real literal's value is found by an
/// Int-sorted computed argument.
#[test]
fn test_eval_apply_func_interp_lookup_bridges_int_and_integral_rational() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let one = manager.mk_int(num_bigint::BigInt::from(1));
    let two = manager.mk_int(num_bigint::BigInt::from(2));
    let f_1_2 = manager.mk_apply("f", [one, two], int_sort);
    let func = apply_func_spur(&manager, f_1_2);

    // Entry keyed on Real-shaped values (`Rational64`), looked up with
    // Int-shaped arguments.
    let mut fi = FuncInterp::new(2, Value::Int(-1));
    fi.add_entry(
        vec![
            Value::Rational(Rational64::from_integer(1)),
            Value::Rational(Rational64::from_integer(2)),
        ],
        Value::Int(99),
    );
    let mut model = Model::new();
    model.add_func_interp(func, fi);

    let mut evaluator = ModelEvaluator::new(&model);
    assert!(matches!(
        evaluator.eval(f_1_2, &manager),
        EvalResult::Ok(Value::Int(99))
    ));
}

/// The compounding failure mode reported alongside the missing wiring: since
/// `Apply` used to be *unconditionally* `Undefined`, and `Undefined`
/// outranks `Error` in every eager/connective fold, one `Apply` subterm
/// anywhere in a formula dragged the *whole* result down to `Undefined` even
/// when a matching `FuncInterp` made the actual answer fully determined.
#[test]
fn test_apply_with_resolved_interp_does_not_collapse_the_whole_formula_to_undefined() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let one = manager.mk_int(num_bigint::BigInt::from(1));
    let two = manager.mk_int(num_bigint::BigInt::from(2));
    let f_1_2 = manager.mk_apply("f", [one, two], bool_sort);
    let func = apply_func_spur(&manager, f_1_2);

    let mut fi = FuncInterp::new(2, Value::Bool(false));
    fi.add_entry(vec![Value::Int(1), Value::Int(2)], Value::Bool(true));
    let mut model = Model::new();
    model.add_func_interp(func, fi);

    // `p` is a plain Bool variable (not a `True`/`False` literal), so
    // `mk_and` cannot fold it away -- the conjunction really is built with
    // two operands, the second of which is the `Apply` term under test.
    let p = manager.mk_var("p", bool_sort);
    model.assign(p, Value::Bool(true));
    let conj = manager.mk_and([p, f_1_2]);
    assert!(
        matches!(manager.get(conj).map(|t| &t.kind), Some(TermKind::And(args)) if args.len() == 2),
        "test setup: `mk_and` must not have folded the conjunction away"
    );

    let mut evaluator = ModelEvaluator::new(&model);
    assert!(
        matches!(
            evaluator.eval(conj, &manager),
            EvalResult::Ok(Value::Bool(true))
        ),
        "a resolvable Apply subterm must not drag the whole conjunction down to Undefined"
    );
}

/// A regex operator (`str.to_re`) lowers to `Apply` exactly like an
/// uninterpreted function call, and no `FuncInterp` is ever stored for one.
/// Confirms the deliberate choice: it stays `Undefined`, the same as any
/// other unmapped function, rather than being silently treated as a matched
/// uninterpreted application.
#[test]
fn test_eval_regex_lowered_apply_term_is_undefined_not_an_error() {
    let mut manager = TermManager::new();
    let s = manager.mk_string_lit("abc");
    let re = manager.mk_str_to_re(s);

    let model = Model::new();
    let mut evaluator = ModelEvaluator::new(&model);
    assert!(matches!(
        evaluator.eval(re, &manager),
        EvalResult::Undefined(_)
    ));
}

/// The same contract under deep nesting, mirroring the other
/// `test_deep_*_evaluates_on_a_small_stack` cases: `Apply` is driven by the
/// same explicit frame stack as every other operator, so a chain of nested
/// applications costs heap, not native stack.
#[test]
fn test_deep_apply_chain_evaluates_on_a_small_stack() {
    let value = on_small_stack(|| {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let mut term = manager.mk_int(num_bigint::BigInt::from(0));
        let mut func = None;
        for _ in 0..DEEP {
            term = manager.mk_apply("f", [term], int_sort);
            func.get_or_insert_with(|| apply_func_spur(&manager, term));
        }
        let func = func.expect("DEEP > 0, so at least one Apply was built");

        // No entries: every level's lookup falls straight to `else_value`,
        // so the whole chain must evaluate to that same constant regardless
        // of depth -- what matters here is that the driver visits `DEEP`
        // nested frames without recursing.
        let mut model = Model::new();
        model.add_func_interp(func, FuncInterp::new(1, Value::Int(42)));

        let mut evaluator = ModelEvaluator::new(&model);
        evaluator.eval(term, &manager)
    });
    assert!(
        matches!(value, EvalResult::Ok(Value::Int(42))),
        "expected Ok(Int(42)) through {DEEP} nested Apply terms, got {value:?}"
    );
}
