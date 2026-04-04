//! Criterion benchmarks for the Simplex arithmetic solver.
//!
//! Covers four scenarios:
//!   - `bench_simplex_small`  — 10-variable LP
//!   - `bench_simplex_medium` — 50-variable LP
//!   - `bench_simplex_large`  — 200-variable LP
//!   - `bench_pivot_hot_path` — isolated pivot() microbenchmark (2-variable tableau)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_rational::Rational64;
use oxiz_theories::arithmetic::{LinExpr, Simplex};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helper builders
// ---------------------------------------------------------------------------

/// Build a Simplex instance with `n` variables and the following constraint
/// structure (a dense chain that forces many pivots):
///
///   for i in 0..n-1:  x[i] - x[i+1] <= 1
///   x[0] >= 0
///   x[n-1] <= n as Rational64
///
/// This creates a feasible LP that the primal simplex must solve by finding
/// a basic feasible solution along the chain.
fn build_chain_lp(n: u32) -> Simplex {
    let mut simplex = Simplex::new();

    // Allocate n variables
    let vars: Vec<u32> = (0..n).map(|_| simplex.new_var()).collect();

    // x[0] >= 0
    simplex.set_lower(vars[0], Rational64::new(0, 1), 0);
    // x[n-1] <= n (gives the simplex something to push against)
    simplex.set_upper(vars[n as usize - 1], Rational64::new(n as i64, 1), 1);

    // Add chain constraints: x[i] - x[i+1] <= 1  (i.e. x[i] - x[i+1] + slack = 1, slack >= 0)
    for i in 0..(n as usize - 1) {
        let mut expr = LinExpr::new();
        expr.add_term(vars[i], Rational64::new(1, 1));
        expr.add_term(vars[i + 1], Rational64::new(-1, 1));
        // expr <= 1  <=>  expr - 1 <= 0
        expr.add_constant(Rational64::new(-1, 1));
        simplex.add_le(expr, (i + 2) as u32);
    }

    // Add a non-trivial objective direction: x[0] <= x[1] + 2 for all pairs
    // (adds density to the tableau and forces more pivots)
    for i in 0..(n as usize).saturating_sub(2) {
        let mut expr = LinExpr::new();
        expr.add_term(vars[i], Rational64::new(1, 1));
        expr.add_term(vars[i + 1], Rational64::new(-1, 1));
        expr.add_constant(Rational64::new(-2, 1));
        simplex.add_le(expr, n + i as u32 + 10);
    }

    simplex
}

/// Build a 2-variable simplex tableau that is already infeasible (upper < lower)
/// after one pivot.  Used as the "hot path" microbenchmark to isolate the
/// cost of a single pivot step with minimal surrounding overhead.
///
/// Setup:
///   x0, x1 non-basic; slack s (basic) = x0 + x1 - 3 with s >= 0
///   Bounds: x0 in [0,2], x1 in [0,2]  →  s = x0+x1-3 must reach >= 0
///   After: pivot x0 into basis, observe update
fn build_pivot_bench() -> Simplex {
    let mut simplex = Simplex::new();

    let x0 = simplex.new_var();
    let x1 = simplex.new_var();

    // x0 in [0, 2], x1 in [0, 2]
    simplex.set_lower(x0, Rational64::new(0, 1), 0);
    simplex.set_upper(x0, Rational64::new(2, 1), 1);
    simplex.set_lower(x1, Rational64::new(0, 1), 2);
    simplex.set_upper(x1, Rational64::new(2, 1), 3);

    // x0 + x1 >= 3  (only satisfiable at the boundary, forces pivoting)
    let mut expr = LinExpr::new();
    expr.add_term(x0, Rational64::new(1, 1));
    expr.add_term(x1, Rational64::new(1, 1));
    expr.add_constant(Rational64::new(-3, 1));
    simplex.add_ge(expr, 4);

    // Additional tight constraint: x0 - x1 <= 0
    let mut expr2 = LinExpr::new();
    expr2.add_term(x0, Rational64::new(1, 1));
    expr2.add_term(x1, Rational64::new(-1, 1));
    simplex.add_le(expr2, 5);

    simplex
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

fn bench_simplex_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex");

    group.bench_function(BenchmarkId::new("small", "10_vars"), |b| {
        b.iter(|| {
            let mut s = build_chain_lp(black_box(10));
            let _ = s.check();
            black_box(s)
        });
    });

    group.finish();
}

fn bench_simplex_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex");

    group.bench_function(BenchmarkId::new("medium", "50_vars"), |b| {
        b.iter(|| {
            let mut s = build_chain_lp(black_box(50));
            let _ = s.check();
            black_box(s)
        });
    });

    group.finish();
}

fn bench_simplex_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex");
    // Large LP is expensive; reduce sample size to keep total bench time bounded.
    group.sample_size(20);

    group.bench_function(BenchmarkId::new("large", "200_vars"), |b| {
        b.iter(|| {
            let mut s = build_chain_lp(black_box(200));
            let _ = s.check();
            black_box(s)
        });
    });

    group.finish();
}

/// Microbenchmark that isolates the pivot() hot path by running `check()` on a
/// minimal 2-variable tableau where the solver must perform exactly one or two
/// pivots before concluding SAT.  Setup cost (build_pivot_bench) is excluded
/// from the measurement via the `iter` closure.
fn bench_pivot_hot_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex");

    group.bench_function(BenchmarkId::new("pivot_hot_path", "2_vars"), |b| {
        b.iter(|| {
            let mut s = build_pivot_bench();
            // check() exercises find_violating → find_pivot_col → pivot → update_assignment
            let result = s.check();
            black_box(result)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    simplex_benches,
    bench_simplex_small,
    bench_simplex_medium,
    bench_simplex_large,
    bench_pivot_hot_path,
);
criterion_main!(simplex_benches);
