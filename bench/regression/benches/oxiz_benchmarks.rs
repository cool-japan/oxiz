//! Criterion benchmarks for OxiZ
//!
//! These benchmarks can be run with:
//! ```bash
//! cargo bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::parse_script;
use oxiz_sat::{DimacsParser, Lit, Solver as SatSolver, Var};
use oxiz_solver::Solver;

/// Benchmark SAT solver with 3-SAT instances of varying sizes
fn bench_sat_3sat(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat_3sat");

    for num_vars in [10, 25, 50].iter() {
        let num_clauses = num_vars * 4; // 4.0 clause-to-variable ratio

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            num_vars,
            |b, &num_vars| {
                b.iter(|| {
                    let mut solver = SatSolver::new();

                    for _ in 0..num_vars {
                        solver.new_var();
                    }

                    for i in 0..num_clauses {
                        let v1 = Var::new((i % num_vars) as u32);
                        let v2 = Var::new(((i * 7 + 3) % num_vars) as u32);
                        let v3 = Var::new(((i * 13 + 7) % num_vars) as u32);
                        let l1 = if i % 2 == 0 { Lit::pos(v1) } else { Lit::neg(v1) };
                        let l2 = if i % 3 == 0 { Lit::pos(v2) } else { Lit::neg(v2) };
                        let l3 = if i % 5 == 0 { Lit::pos(v3) } else { Lit::neg(v3) };
                        solver.add_clause([l1, l2, l3]);
                    }

                    black_box(solver.solve())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark unit propagation
fn bench_sat_propagation(c: &mut Criterion) {
    c.bench_function("sat_unit_propagation", |b| {
        b.iter(|| {
            let mut solver = SatSolver::new();

            for _ in 0..100 {
                solver.new_var();
            }

            // Create implication chain
            for i in 0..99 {
                let vi = Var::new(i);
                let vj = Var::new(i + 1);
                solver.add_clause([Lit::neg(vi), Lit::pos(vj)]);
            }

            solver.add_clause([Lit::pos(Var::new(0))]);

            black_box(solver.solve())
        });
    });
}

/// Benchmark LIA theory solver
fn bench_theory_lia(c: &mut Criterion) {
    c.bench_function("theory_lia_simple", |b| {
        b.iter(|| {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();
            solver.set_logic("QF_LIA");

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let five = tm.mk_int(BigInt::from(5));
            let ten = tm.mk_int(BigInt::from(10));

            solver.assert(tm.mk_ge(x, five), &mut tm);
            solver.assert(tm.mk_le(x, ten), &mut tm);
            let x_plus_1 = tm.mk_add(vec![x, tm.mk_int(BigInt::from(1))]);
            solver.assert(tm.mk_eq(y, x_plus_1), &mut tm);

            black_box(solver.check(&mut tm))
        });
    });
}

/// Benchmark SMT-LIB parser
fn bench_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser");

    let simple_input = r#"
        (declare-const x Int)
        (declare-const y Int)
        (assert (> x 0))
        (assert (< y 10))
        (check-sat)
    "#;

    let medium_input = r#"
        (set-logic QF_LIA)
        (declare-const a Int)
        (declare-const b Int)
        (declare-const c Int)
        (declare-const d Int)
        (declare-const e Int)
        (assert (>= a 0))
        (assert (>= b 0))
        (assert (>= c 0))
        (assert (>= d 0))
        (assert (>= e 0))
        (assert (<= a 100))
        (assert (<= b 100))
        (assert (<= c 100))
        (assert (<= d 100))
        (assert (<= e 100))
        (assert (= (+ a b c) 50))
        (check-sat)
    "#;

    group.bench_function("simple", |b| {
        b.iter(|| {
            let mut tm = TermManager::new();
            black_box(parse_script(simple_input, &mut tm))
        });
    });

    group.bench_function("medium", |b| {
        b.iter(|| {
            let mut tm = TermManager::new();
            black_box(parse_script(medium_input, &mut tm))
        });
    });

    group.finish();
}

/// Benchmark DIMACS parser
fn bench_dimacs_parser(c: &mut Criterion) {
    let dimacs_input = "p cnf 10 20\n1 2 3 0\n-1 2 4 0\n1 -2 5 0\n-1 -2 6 0\n\
        3 4 5 0\n-3 4 6 0\n3 -4 7 0\n-3 -4 8 0\n\
        5 6 7 0\n-5 6 8 0\n5 -6 9 0\n-5 -6 10 0\n\
        7 8 9 0\n-7 8 10 0\n7 -8 1 0\n-7 -8 2 0\n\
        9 10 1 0\n-9 10 2 0\n9 -10 3 0\n-9 -10 4 0\n";

    c.bench_function("parser_dimacs", |b| {
        b.iter(|| {
            black_box(DimacsParser::parse(dimacs_input))
        });
    });
}

criterion_group!(
    benches,
    bench_sat_3sat,
    bench_sat_propagation,
    bench_theory_lia,
    bench_parser,
    bench_dimacs_parser,
);

criterion_main!(benches);
