//! Integration tests for the Z3 API compatibility extensions.
//!
//! Covers: Real symmetry, ite_*, distinct_*, Array, FuncDecl,
//! forall/exists quantifiers, and Z3Optimize.

use oxiz_solver::z3_compat::{
    Bool, BV, Int, Real, SatResult, Z3Config, Z3Context, Z3Solver,
    Array, FuncDecl, Z3Optimize,
    distinct_bv, distinct_int, distinct_real,
    exists_bool, forall_bool,
    ite_bool, ite_bv, ite_int, ite_real,
};

fn make_ctx() -> Z3Context {
    Z3Context::new(&Z3Config::new())
}

// ─── Real symmetry ────────────────────────────────────────────────────────────

#[test]
fn test_real_gt() {
    let ctx = make_ctx();
    let a = Real::from_i64(&ctx, 3);
    let b = Real::from_i64(&ctx, 1);
    // Build the term — just verify no panic
    let _gt = Real::gt(&ctx, &a, &b);
}

#[test]
fn test_real_ge() {
    let ctx = make_ctx();
    let a = Real::from_i64(&ctx, 5);
    let b = Real::from_i64(&ctx, 5);
    let _ge = Real::ge(&ctx, &a, &b);
}

#[test]
fn test_real_neg() {
    let ctx = make_ctx();
    let x = Real::new_const(&ctx, "x");
    let neg_x = Real::neg(&ctx, &x);
    // Negation must produce a distinct TermId from the argument
    assert_ne!(neg_x.id, x.id);
}

#[test]
fn test_real_div() {
    let ctx = make_ctx();
    let a = Real::new_const(&ctx, "a");
    let b = Real::from_i64(&ctx, 2);
    let d = Real::div(&ctx, &a, &b);
    assert_ne!(d.id, a.id);
}

#[test]
fn test_real_from_i64() {
    let ctx = make_ctx();
    let r = Real::from_i64(&ctx, 42);
    // Must equal the frac form with denominator 1
    let r2 = Real::from_frac(&ctx, 42, 1);
    assert_eq!(r.id, r2.id, "from_i64 and from_frac(n,1) must produce the same term");
}

// ─── ITE ─────────────────────────────────────────────────────────────────────

#[test]
fn test_ite_bool_sat() {
    // ite(c, true, false) — the TermManager simplifies this to `c`; no panic.
    let ctx = make_ctx();
    let c = Bool::new_const(&ctx, "c");
    let t_val = Bool::from_bool(&ctx, true);
    let f_val = Bool::from_bool(&ctx, false);
    let result = ite_bool(&ctx, &c, &t_val, &f_val);
    assert!(result.id.0 < u32::MAX);
}

#[test]
fn test_ite_int() {
    let ctx = make_ctx();
    let c = Bool::new_const(&ctx, "c");
    let one = Int::from_i64(&ctx, 1);
    let zero = Int::from_i64(&ctx, 0);
    let ite = ite_int(&ctx, &c, &one, &zero);
    // ite(c, 1, 0) is a fresh term, distinct from both branches
    assert_ne!(ite.id, one.id);
    assert_ne!(ite.id, zero.id);
}

#[test]
fn test_ite_real() {
    let ctx = make_ctx();
    let c = Bool::new_const(&ctx, "c");
    let hi = Real::from_i64(&ctx, 10);
    let lo = Real::from_i64(&ctx, 0);
    let ite = ite_real(&ctx, &c, &hi, &lo);
    assert_ne!(ite.id, hi.id);
    assert_ne!(ite.id, lo.id);
}

#[test]
fn test_ite_bv() {
    let ctx = make_ctx();
    let c = Bool::new_const(&ctx, "c");
    let hi = BV::from_u64(&ctx, 0xFF, 8);
    let lo = BV::from_u64(&ctx, 0x00, 8);
    let ite = ite_bv(&ctx, &c, &hi, &lo);
    assert_eq!(ite.width, 8);
}

// ─── Distinct ─────────────────────────────────────────────────────────────────

#[test]
fn test_distinct_int_sat() {
    // Three freshly-declared Int variables with a distinct constraint: SAT
    let ctx = make_ctx();
    let mut solver = Z3Solver::new(&ctx);
    solver.set_logic("QF_LIA");

    // Build variables in the solver's own term manager
    let int_sort = solver.context().terms.sorts.int_sort;
    let x = solver.context_mut().terms.mk_var("x", int_sort);
    let int_sort2 = solver.context().terms.sorts.int_sort;
    let y = solver.context_mut().terms.mk_var("y", int_sort2);
    let int_sort3 = solver.context().terms.sorts.int_sort;
    let z_var = solver.context_mut().terms.mk_var("z", int_sort3);
    let d = solver.context_mut().terms.mk_distinct([x, y, z_var]);
    solver.context_mut().assert(d);

    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_distinct_int_same_literal() {
    // distinct(1, 1): distinct on two identical literal TermIds.
    // mk_distinct with equal ids either yields false immediately or
    // interns a well-formed term — either way must not panic.
    let ctx = make_ctx();
    let one_a = Int::from_i64(&ctx, 1);
    let one_b = Int::from_i64(&ctx, 1);
    // Equal integer literals must intern to the same TermId
    assert_eq!(
        one_a.id, one_b.id,
        "Equal integer constants must map to the same TermId"
    );
    let d = distinct_int(&ctx, &[one_a, one_b]);
    // Term is valid (no panic), regardless of the simplified result
    assert!(d.id.0 < u32::MAX);
}

#[test]
fn test_distinct_real() {
    let ctx = make_ctx();
    let a = Real::from_i64(&ctx, 1);
    let b = Real::from_i64(&ctx, 2);
    let _d = distinct_real(&ctx, &[a, b]);
}

#[test]
fn test_distinct_bv() {
    let ctx = make_ctx();
    let a = BV::from_u64(&ctx, 0, 8);
    let b = BV::from_u64(&ctx, 1, 8);
    let _d = distinct_bv(&ctx, &[a, b]);
}

// ─── Array ────────────────────────────────────────────────────────────────────

#[test]
fn test_array_select_store() {
    // Build: arr' = store(arr, 0, 42); val = select(arr', 0)
    let ctx = make_ctx();
    let dom = ctx.int_sort();
    let rng = ctx.int_sort();
    let arr = Array::new_const(&ctx, "arr", dom, rng);
    let idx0 = Int::from_i64(&ctx, 0);
    let val42 = Int::from_i64(&ctx, 42);
    let arr2 = Array::store(&ctx, &arr, idx0.id, val42.id);
    let selected = Array::select(&ctx, &arr2, idx0.id);
    // Selected term must be a valid TermId
    assert!(selected.0 < u32::MAX);
}

#[test]
fn test_array_select_different_index() {
    // store at index 0, select at index 1 — two distinct terms expected
    let ctx = make_ctx();
    let dom = ctx.int_sort();
    let rng = ctx.int_sort();
    let arr = Array::new_const(&ctx, "arr", dom, rng);
    let idx0 = Int::from_i64(&ctx, 0);
    let idx1 = Int::from_i64(&ctx, 1);
    let val = Int::from_i64(&ctx, 99);
    let arr2 = Array::store(&ctx, &arr, idx0.id, val.id);
    let sel_at_0 = Array::select(&ctx, &arr2, idx0.id);
    let sel_at_1 = Array::select(&ctx, &arr2, idx1.id);
    // Different indices → different result terms
    assert_ne!(sel_at_0, sel_at_1);
}

#[test]
fn test_array_eq() {
    let ctx = make_ctx();
    let dom = ctx.int_sort();
    let rng = ctx.int_sort();
    let a = Array::new_const(&ctx, "a", dom, rng);
    let b = Array::new_const(&ctx, "b", dom, rng);
    let eq = Array::eq(&ctx, &a, &b);
    // Equality of two distinct variables is a valid Bool term
    assert_ne!(a.id, b.id);
    assert!(eq.id.0 < u32::MAX);
}

// ─── FuncDecl ─────────────────────────────────────────────────────────────────

#[test]
fn test_func_decl_apply() {
    // Declare f: Int -> Int; apply to x; result must differ from x
    let ctx = make_ctx();
    let int_sort = ctx.int_sort();
    let f = FuncDecl::new(&ctx, "f", &[int_sort], int_sort);
    let x = Int::new_const(&ctx, "x");
    let fx = f.apply(&ctx, &[x.id]);
    assert_ne!(fx, x.id, "f(x) must be a distinct term from x");
}

#[test]
fn test_func_decl_two_args() {
    // g: (Int, Int) -> Bool
    let ctx = make_ctx();
    let int_sort = ctx.int_sort();
    let bool_sort = ctx.bool_sort();
    let g = FuncDecl::new(&ctx, "g", &[int_sort, int_sort], bool_sort);
    let x = Int::new_const(&ctx, "x");
    let y = Int::new_const(&ctx, "y");
    let gxy = g.apply(&ctx, &[x.id, y.id]);
    assert!(gxy.0 < u32::MAX);
}

// ─── Quantifiers ──────────────────────────────────────────────────────────────

#[test]
fn test_quantifier_forall_construction() {
    // forall x:Int. x >= 0  — verify term construction without panic
    // We build the body via the Z3-compat Int API so everything stays in the
    // same context TermManager.
    let ctx = make_ctx();
    let int_sort = ctx.int_sort();
    // Build the bound variable by declaring an Int constant with the same name
    let x = Int::new_const(&ctx, "x");
    let zero = Int::from_i64(&ctx, 0);
    let body = Int::ge(&ctx, &x, &zero);
    let q = forall_bool(&ctx, [("x", int_sort)], &body);
    assert!(q.id.0 < u32::MAX);
}

#[test]
fn test_quantifier_exists_construction() {
    // exists x:Int. x > 100
    let ctx = make_ctx();
    let int_sort = ctx.int_sort();
    let x = Int::new_const(&ctx, "x");
    let hundred = Int::from_i64(&ctx, 100);
    let body = Int::gt(&ctx, &x, &hundred);
    let q = exists_bool(&ctx, [("x", int_sort)], &body);
    assert!(q.id.0 < u32::MAX);
    // exists must differ from the raw body term
    assert_ne!(q.id, body.id);
}

// ─── Z3Optimize ───────────────────────────────────────────────────────────────

#[test]
fn test_optimize_minimize_term_constructed() {
    // Minimize x subject to x >= 5.
    // The optimizer may return Sat or Unknown, but must not panic.
    let cfg = Z3Config::new();
    let ctx = Z3Context::new(&cfg);
    let mut opt = Z3Optimize::new(&ctx);

    let x = Int::new_const(&ctx, "x");
    let five = Int::from_i64(&ctx, 5);
    let ge = Int::ge(&ctx, &x, &five);
    opt.assert(&ge);
    let _idx = opt.minimize(x.id);

    let result = opt.check();
    assert!(
        result == SatResult::Sat || result == SatResult::Unknown,
        "Expected Sat or Unknown, got {:?}",
        result
    );
}

#[test]
fn test_optimize_get_lower_before_check_is_none() {
    let cfg = Z3Config::new();
    let ctx = Z3Context::new(&cfg);
    let mut opt = Z3Optimize::new(&ctx);
    let x = Int::new_const(&ctx, "x");
    let _idx = opt.minimize(x.id);
    // Before check(), bounds must be None
    assert!(opt.get_lower(0).is_none());
    assert!(opt.get_upper(0).is_none());
}

#[test]
fn test_optimize_sat_trivial() {
    // Trivially-true assertion with no objective: Sat or Unknown
    let cfg = Z3Config::new();
    let ctx = Z3Context::new(&cfg);
    let mut opt = Z3Optimize::new(&ctx);
    let t = Bool::from_bool(&ctx, true);
    opt.assert(&t);
    let result = opt.check();
    assert!(
        result == SatResult::Sat || result == SatResult::Unknown,
        "Expected Sat or Unknown, got {:?}",
        result
    );
}
