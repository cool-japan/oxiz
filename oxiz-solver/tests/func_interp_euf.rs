//! End-to-end tests for EUF-derived function interpretation extraction.
//!
//! These exercise `Context::get_func_interp_raw`, which assembles a function
//! interpretation from the EUF congruence closure rather than from raw `Apply`
//! terms alone.  The properties under test are:
//!
//! 1. Congruent applications (e.g. `f(a)` and `f(b)` when `a = b`) collapse to a
//!    single entry keyed by the shared argument equivalence class.
//! 2. Applications with distinct arguments produce distinct entries.
//! 3. The value reported for an application is the canonical model value of its
//!    equivalence class, even when the application term itself has no direct
//!    model assignment.

use oxiz_solver::{Context, SolverResult};

/// `a = b` together with `f(a) = 5` must yield a SINGLE interpretation entry for
/// the `{a, b}` argument class mapping to `5` — not two separate entries.
#[test]
fn test_func_interp_congruence_dedup() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_UFLIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let a = ctx.declare_const("a", int_sort);
    let b = ctx.declare_const("b", int_sort);
    ctx.declare_fun("f", vec![int_sort], int_sort);

    // f(a) and f(b): both interned so the E-graph knows about both applications.
    let fa = ctx.terms.mk_apply("f", [a], int_sort);
    let fb = ctx.terms.mk_apply("f", [b], int_sort);

    // a = b  -> f(a) and f(b) become congruent.
    let a_eq_b = ctx.terms.mk_eq(a, b);
    ctx.assert(a_eq_b);

    // f(a) = 5
    let five = ctx.terms.mk_int(5);
    let fa_eq_5 = ctx.terms.mk_eq(fa, five);
    ctx.assert(fa_eq_5);

    // Reference f(b) in an assertion so it is definitely interned into EUF as an
    // application (f(b) = f(b) is trivially true and does not constrain the model).
    let fb_eq_fb = ctx.terms.mk_eq(fb, fb);
    ctx.assert(fb_eq_fb);

    assert_eq!(ctx.check_sat(), SolverResult::Sat);

    let (entries, _else_value, arity) = ctx
        .get_func_interp_raw("f")
        .expect("f should have an interpretation after SAT");

    assert_eq!(arity, 1, "f has arity 1");
    // Congruence must collapse f(a) and f(b) into ONE entry.
    assert_eq!(
        entries.len(),
        1,
        "f(a) and f(b) must collapse to one entry under a=b, got {:?}",
        entries
    );
    let (args, value) = &entries[0];
    assert_eq!(args.len(), 1);
    assert_eq!(value, "5", "the single entry must map to 5");
    // The argument string is the common model value of the {a, b} class, which
    // is an integer literal (the solver is free to choose which one).
    assert!(
        args[0].parse::<i64>().is_ok(),
        "argument value should be a concrete integer, got {:?}",
        args[0]
    );
}

/// `f(0) = 10` and `f(1) = 20` with `0 != 1` must yield TWO distinct entries.
#[test]
fn test_func_interp_two_distinct_args() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_UFLIA");

    let int_sort = ctx.terms.sorts.int_sort;
    ctx.declare_fun("f", vec![int_sort], int_sort);

    let zero = ctx.terms.mk_int(0);
    let one = ctx.terms.mk_int(1);
    let f0 = ctx.terms.mk_apply("f", [zero], int_sort);
    let f1 = ctx.terms.mk_apply("f", [one], int_sort);

    let ten = ctx.terms.mk_int(10);
    let twenty = ctx.terms.mk_int(20);
    let f0_eq_10 = ctx.terms.mk_eq(f0, ten);
    let f1_eq_20 = ctx.terms.mk_eq(f1, twenty);
    ctx.assert(f0_eq_10);
    ctx.assert(f1_eq_20);

    assert_eq!(ctx.check_sat(), SolverResult::Sat);

    let (entries, _else_value, arity) = ctx
        .get_func_interp_raw("f")
        .expect("f should have an interpretation after SAT");

    assert_eq!(arity, 1);
    assert_eq!(
        entries.len(),
        2,
        "f(0) and f(1) must be two distinct entries, got {:?}",
        entries
    );

    // Both (0 -> 10) and (1 -> 20) must be present (order is unspecified).
    let mut pairs: Vec<(String, String)> = entries
        .iter()
        .map(|(args, value)| (args[0].clone(), value.clone()))
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("0".to_string(), "10".to_string()),
            ("1".to_string(), "20".to_string())
        ]
    );
}

/// The value reported for `f(a)` must be the canonical model value of its
/// equivalence class.  Here `f(a) = c` and `c = 7`, so even though `f(a)` may
/// have no direct model assignment, its class contains `c` (value 7) and the
/// interpretation must report 7.
#[test]
fn test_func_interp_value_via_class_representative() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_UFLIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let a = ctx.declare_const("a", int_sort);
    let c = ctx.declare_const("c", int_sort);
    ctx.declare_fun("f", vec![int_sort], int_sort);

    let fa = ctx.terms.mk_apply("f", [a], int_sort);

    // f(a) = c   and   c = 7
    let fa_eq_c = ctx.terms.mk_eq(fa, c);
    let seven = ctx.terms.mk_int(7);
    let c_eq_7 = ctx.terms.mk_eq(c, seven);
    ctx.assert(fa_eq_c);
    ctx.assert(c_eq_7);

    assert_eq!(ctx.check_sat(), SolverResult::Sat);

    let (entries, _else_value, arity) = ctx
        .get_func_interp_raw("f")
        .expect("f should have an interpretation after SAT");

    assert_eq!(arity, 1);
    assert_eq!(
        entries.len(),
        1,
        "exactly one application f(a), got {:?}",
        entries
    );
    let (args, value) = &entries[0];
    assert_eq!(args.len(), 1);
    assert_eq!(
        value, "7",
        "f(a)'s value must be resolved through its class (c = 7)"
    );
}

/// A declared-but-never-applied function returns an empty entry list with the
/// return sort's default `else_value` (no panic, no spurious entries).
#[test]
fn test_func_interp_unapplied_function() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_UFLIA");

    let int_sort = ctx.terms.sorts.int_sort;
    ctx.declare_fun("g", vec![int_sort], int_sort);

    // Assert something trivial so we get SAT without applying g.
    let t = ctx.terms.mk_true();
    ctx.assert(t);

    assert_eq!(ctx.check_sat(), SolverResult::Sat);

    let (entries, else_value, arity) = ctx
        .get_func_interp_raw("g")
        .expect("declared g should still return an (empty) interpretation");
    assert_eq!(arity, 1);
    assert!(entries.is_empty(), "g is never applied -> no entries");
    assert_eq!(else_value, "0", "default else_value for Int return sort");
}
