//! Benchmarks for oxiz-solver

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_solver::{Context, Solver, SolverConfig};

/// Benchmark simple boolean satisfiability
fn bench_simple_sat(c: &mut Criterion) {
    c.bench_function("simple_sat", |b| {
        b.iter(|| {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);
            let formula = tm.mk_and(vec![p, q]);

            solver.assert(black_box(formula), &mut tm);
            solver.check(&mut tm)
        });
    });
}

/// Benchmark simple boolean unsatisfiability
fn bench_simple_unsat(c: &mut Criterion) {
    c.bench_function("simple_unsat", |b| {
        b.iter(|| {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let not_p = tm.mk_not(p);

            solver.assert(black_box(p), &mut tm);
            solver.assert(black_box(not_p), &mut tm);
            solver.check(&mut tm)
        });
    });
}

/// Benchmark push/pop operations
fn bench_push_pop(c: &mut Criterion) {
    c.bench_function("push_pop", |b| {
        b.iter(|| {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            solver.assert(p, &mut tm);

            for _ in 0..10 {
                solver.push();
                let q = tm.mk_var("q", tm.sorts.bool_sort);
                solver.assert(q, &mut tm);
                solver.pop();
            }

            solver.check(&mut tm)
        });
    });
}

/// Benchmark arithmetic constraints
fn bench_arithmetic(c: &mut Criterion) {
    c.bench_function("arithmetic", |b| {
        b.iter(|| {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::from(0));
            let ten = tm.mk_int(BigInt::from(10));

            // x >= 0
            let c1 = tm.mk_ge(x, zero);
            // y >= 0
            let zero2 = tm.mk_int(BigInt::from(0));
            let c2 = tm.mk_ge(y, zero2);
            // x + y <= 10
            let sum = tm.mk_add(vec![x, y]);
            let c3 = tm.mk_le(sum, ten);

            solver.assert(black_box(c1), &mut tm);
            solver.assert(black_box(c2), &mut tm);
            solver.assert(black_box(c3), &mut tm);
            solver.check(&mut tm)
        });
    });
}

/// Benchmark encoding complexity
fn bench_complex_encoding(c: &mut Criterion) {
    c.bench_function("complex_encoding", |b| {
        b.iter(|| {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            // Create a complex nested formula
            let vars: Vec<_> = (0..20)
                .map(|i| tm.mk_var(&format!("x{}", i), tm.sorts.bool_sort))
                .collect();

            // Create (x0 ∧ x1) ∨ (x2 ∧ x3) ∨ ... ∨ (x18 ∧ x19)
            let mut clauses = Vec::new();
            for i in (0..20).step_by(2) {
                clauses.push(tm.mk_and(vec![vars[i], vars[i + 1]]));
            }

            let formula = tm.mk_or(clauses);

            solver.assert(black_box(formula), &mut tm);
            solver.check(&mut tm)
        });
    });
}

/// Benchmark different solver configurations
fn bench_solver_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_configs");

    for (name, config) in [
        ("fast", SolverConfig::fast()),
        ("balanced", SolverConfig::balanced()),
        ("thorough", SolverConfig::thorough()),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut solver = Solver::with_config(config.clone());
                let mut tm = TermManager::new();

                let p = tm.mk_var("p", tm.sorts.bool_sort);
                let q = tm.mk_var("q", tm.sorts.bool_sort);
                let r = tm.mk_var("r", tm.sorts.bool_sort);

                let and_pq = tm.mk_and(vec![p, q]);
                let formula = tm.mk_or(vec![and_pq, r]);

                solver.assert(black_box(formula), &mut tm);
                solver.check(&mut tm)
            });
        });
    }

    group.finish();
}

/// Benchmark Context API (SMT-LIB2 style)
fn bench_context_api(c: &mut Criterion) {
    c.bench_function("context_api", |b| {
        b.iter(|| {
            let mut ctx = Context::new();

            ctx.set_logic("QF_LIA");
            let x = ctx.declare_const("x", ctx.terms.sorts.int_sort);
            let y = ctx.declare_const("y", ctx.terms.sorts.int_sort);

            let zero = ctx.terms.mk_int(BigInt::from(0));
            let c1 = ctx.terms.mk_ge(x, zero);
            let c2 = ctx.terms.mk_ge(y, zero);

            ctx.assert(black_box(c1));
            ctx.assert(black_box(c2));
            ctx.check_sat()
        });
    });
}

/// Benchmark model generation
fn bench_model_generation(c: &mut Criterion) {
    c.bench_function("model_generation", |b| {
        b.iter(|| {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);

            solver.assert(p, &mut tm);
            solver.assert(q, &mut tm);

            let result = solver.check(&mut tm);
            black_box(result);

            if let Some(model) = solver.model() {
                black_box(model);
            }
        });
    });
}

/// Benchmark simplification
fn bench_simplification(c: &mut Criterion) {
    c.bench_function("simplification", |b| {
        b.iter(|| {
            let config = SolverConfig {
                simplify: true,
                ..SolverConfig::default()
            };
            let mut solver = Solver::with_config(config);
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let t = tm.mk_true();
            let f = tm.mk_false();

            // Create formulas that will be simplified
            let and_true = tm.mk_and(vec![p, t]); // p ∧ true => p
            let or_false = tm.mk_or(vec![p, f]); // p ∨ false => p
            let inner_not = tm.mk_not(p);
            let not_not = tm.mk_not(inner_not); // ¬¬p => p

            solver.assert(black_box(and_true), &mut tm);
            solver.assert(black_box(or_false), &mut tm);
            solver.assert(black_box(not_not), &mut tm);
            solver.check(&mut tm)
        });
    });
}

criterion_group!(
    benches,
    bench_simple_sat,
    bench_simple_unsat,
    bench_push_pop,
    bench_arithmetic,
    bench_complex_encoding,
    bench_solver_configs,
    bench_context_api,
    bench_model_generation,
    bench_simplification
);

criterion_main!(benches);
