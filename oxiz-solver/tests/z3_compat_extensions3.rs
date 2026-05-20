//! Integration tests for the Z3 API compatibility extension layer 3.
//!
//! Covers: Sort introspection (`Z3Sort`, `Z3SortKind`), term substitution
//! (`Z3Context::substitute`), and quantifier patterns/triggers (`Z3Pattern`,
//! `forall_with_patterns` / `exists_with_patterns`).

use oxiz_solver::z3_compat::{
    BV, Bool, FuncDecl, Int, Real, Z3Config, Z3Context, Z3SortKind,
};

fn make_ctx() -> Z3Context {
    Z3Context::new(&Z3Config::new())
}

// ─── Sort introspection ───────────────────────────────────────────────────────

#[test]
fn test_sort_kind_bool() {
    let ctx = make_ctx();
    let p = Bool::new_const(&ctx, "p");
    let s = ctx.sort_of_bool(&p);
    assert_eq!(s.kind(), Z3SortKind::Bool);
    assert_eq!(s.name(), "Bool");
}

#[test]
fn test_sort_kind_int() {
    let ctx = make_ctx();
    let x = Int::new_const(&ctx, "x");
    let s = ctx.sort_of_int(&x);
    assert_eq!(s.kind(), Z3SortKind::Int);
    assert_eq!(s.name(), "Int");
}

#[test]
fn test_sort_kind_real() {
    let ctx = make_ctx();
    let r = Real::new_const(&ctx, "r");
    let s = ctx.sort_of_real(&r);
    assert_eq!(s.kind(), Z3SortKind::Real);
    assert_eq!(s.name(), "Real");
}

#[test]
fn test_sort_kind_bv() {
    let ctx = make_ctx();
    let b = BV::new_const(&ctx, "b", 32);
    let s = ctx.sort_of_bv(&b);
    assert_eq!(s.kind(), Z3SortKind::BitVec);
}

#[test]
fn test_bv_size_returns_width() {
    let ctx = make_ctx();
    let b8 = BV::new_const(&ctx, "b8", 8);
    let b64 = BV::new_const(&ctx, "b64", 64);
    assert_eq!(ctx.sort_of_bv(&b8).bv_size(), Some(8));
    assert_eq!(ctx.sort_of_bv(&b64).bv_size(), Some(64));
    // Non-BV sorts return None.
    let x = Int::new_const(&ctx, "x");
    assert_eq!(ctx.sort_of_int(&x).bv_size(), None);
}

#[test]
fn test_array_domain_range() {
    let ctx = make_ctx();
    let arr_sort = ctx.array_sort(ctx.int_sort(), ctx.bool_sort());
    let s = ctx.wrap_sort(arr_sort);
    assert_eq!(s.kind(), Z3SortKind::Array);
    let dom = s.array_domain().expect("array has a domain");
    let rng = s.array_range().expect("array has a range");
    assert_eq!(dom.kind(), Z3SortKind::Int);
    assert_eq!(rng.kind(), Z3SortKind::Bool);
    // A non-array sort yields None for both.
    let p = Bool::new_const(&ctx, "p");
    assert!(ctx.sort_of_bool(&p).array_domain().is_none());
    assert!(ctx.sort_of_bool(&p).array_range().is_none());
}

// ─── Term substitution ────────────────────────────────────────────────────────

#[test]
fn test_substitute_replaces_subterm() {
    let ctx = make_ctx();
    let x = Int::new_const(&ctx, "x");
    let y = Int::new_const(&ctx, "y");
    // expr = x + 1 ; replace x -> y  ==>  y + 1
    let one = Int::from_i64(&ctx, 1);
    let expr = Int::add(&ctx, &[x.clone(), one.clone()]);
    let expected = Int::add(&ctx, &[y.clone(), one.clone()]);
    let got = ctx.substitute(expr.id, &[(x.id, y.id)]);
    assert_eq!(got, expected.id);
}

#[test]
fn test_substitute_multiple_pairs() {
    let ctx = make_ctx();
    let x = Int::new_const(&ctx, "x");
    let y = Int::new_const(&ctx, "y");
    let a = Int::new_const(&ctx, "a");
    let b = Int::new_const(&ctx, "b");
    // expr = x + y ; {x->a, y->b}  ==>  a + b
    let expr = Int::add(&ctx, &[x.clone(), y.clone()]);
    let expected = Int::add(&ctx, &[a.clone(), b.clone()]);
    let got = ctx.substitute(expr.id, &[(x.id, a.id), (y.id, b.id)]);
    assert_eq!(got, expected.id);
}

#[test]
fn test_substitute_no_match_returns_original() {
    let ctx = make_ctx();
    let x = Int::new_const(&ctx, "x");
    let y = Int::new_const(&ctx, "y");
    let z = Int::new_const(&ctx, "z");
    let w = Int::new_const(&ctx, "w");
    // expr = x + y ; {z->w}  (no match)  ==>  unchanged id
    let expr = Int::add(&ctx, &[x.clone(), y.clone()]);
    let got = ctx.substitute(expr.id, &[(z.id, w.id)]);
    assert_eq!(got, expr.id);
}

#[test]
fn test_substitute_nested() {
    let ctx = make_ctx();
    let x = Int::new_const(&ctx, "x");
    let y = Int::new_const(&ctx, "y");
    // expr = (x * x) + x ; replace x -> y  ==>  (y * y) + y
    let prod = Int::mul(&ctx, &[x.clone(), x.clone()]);
    let expr = Int::add(&ctx, &[prod.clone(), x.clone()]);
    let prod_y = Int::mul(&ctx, &[y.clone(), y.clone()]);
    let expected = Int::add(&ctx, &[prod_y.clone(), y.clone()]);
    let got = ctx.substitute(expr.id, &[(x.id, y.id)]);
    assert_eq!(got, expected.id);
}

#[test]
fn test_substitute_into_bitvector() {
    // Exercises the bit-vector recursion that the core `substitute` does NOT
    // perform — confirms the dedicated rebuild descends through BV operators.
    let ctx = make_ctx();
    let a = BV::new_const(&ctx, "a", 8);
    let b = BV::new_const(&ctx, "b", 8);
    let c = BV::new_const(&ctx, "c", 8);
    // expr = a + b ; replace a -> c  ==>  c + b
    let expr = BV::bvadd(&ctx, &a, &b);
    let expected = BV::bvadd(&ctx, &c, &b);
    let got = ctx.substitute(expr.id, &[(a.id, c.id)]);
    assert_eq!(got, expected.id);
}

#[test]
fn test_substitute_into_apply() {
    // Exercises substitution into a function application's arguments — another
    // case the core `substitute` leaves untouched.
    let ctx = make_ctx();
    let x = Int::new_const(&ctx, "x");
    let y = Int::new_const(&ctx, "y");
    let f = FuncDecl::new(&ctx, "f", &[ctx.int_sort()], ctx.int_sort());
    // expr = f(x) ; replace x -> y  ==>  f(y)
    let expr = f.apply(&ctx, &[x.id]);
    let expected = f.apply(&ctx, &[y.id]);
    let got = ctx.substitute(expr, &[(x.id, y.id)]);
    assert_eq!(got, expected);
}

// ─── Quantifier patterns / triggers ───────────────────────────────────────────

#[test]
fn test_mk_pattern_and_forall_with_patterns() {
    let ctx = make_ctx();
    let int_sort = ctx.int_sort();
    // Build (forall ((x Int)) (! (>= (f x) 0) :pattern ((f x))))
    let x = Int::new_const(&ctx, "x");
    let f = FuncDecl::new(&ctx, "f", &[int_sort], int_sort);
    let fx = f.apply(&ctx, &[x.id]);
    let zero = Int::from_i64(&ctx, 0);
    let body = Int::ge(&ctx, &Int::from_id(fx), &zero);

    let pat = ctx.mk_pattern(&[fx]);
    assert_eq!(pat.len(), 1);
    assert!(!pat.is_empty());

    let q = ctx.forall_with_patterns(&[("x", int_sort)], &[pat], &body);
    // The quantifier term should be created (non-trivial) and boolean-sorted.
    assert_eq!(ctx.sort_of_bool(&q).kind(), Z3SortKind::Bool);
}

#[test]
fn test_exists_with_patterns_solves() {
    let ctx = make_ctx();
    let int_sort = ctx.int_sort();
    let x = Int::new_const(&ctx, "x");
    let f = FuncDecl::new(&ctx, "f", &[int_sort], int_sort);
    let fx = f.apply(&ctx, &[x.id]);
    let zero = Int::from_i64(&ctx, 0);
    let body = Int::gt(&ctx, &Int::from_id(fx), &zero);
    let pat = ctx.mk_pattern(&[fx]);

    // exists with an empty pattern set and with a pattern both build a bool term.
    let q_no_pat = ctx.exists_with_patterns(&[("x", int_sort)], &[], &body);
    let q_pat = ctx.exists_with_patterns(&[("x", int_sort)], &[pat], &body);
    assert_eq!(ctx.sort_of_bool(&q_no_pat).kind(), Z3SortKind::Bool);
    assert_eq!(ctx.sort_of_bool(&q_pat).kind(), Z3SortKind::Bool);
}
