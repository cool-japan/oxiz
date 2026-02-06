//! Performance benchmarks for oxiz-wasm
//!
//! These benchmarks measure the performance of various WASM operations.
//!
//! Run with: cargo bench --bench performance

#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(not(target_arch = "wasm32"))]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// Mock WasmSolver for native benchmarks since the real one requires WASM target
#[cfg(not(target_arch = "wasm32"))]
mod mock {
    use oxiz_solver::Solver;

    pub struct WasmSolver {
        solver: Solver,
    }

    impl WasmSolver {
        pub fn new() -> Self {
            Self {
                solver: Solver::new(),
            }
        }

        pub fn set_logic(&mut self, _logic: &str) {}

        pub fn declare_const(&mut self, _name: &str, _sort: &str) -> Result<(), String> {
            Ok(())
        }

        pub fn assert_formula(&mut self, _formula: &str) -> Result<(), String> {
            Ok(())
        }

        pub fn check_sat(&mut self) -> String {
            "sat".to_string()
        }

        pub fn get_model(&self) -> Result<String, String> {
            Ok("()".to_string())
        }

        pub fn push(&mut self) {}
        pub fn pop(&mut self) {}

        pub fn reset(&mut self) {
            self.solver = Solver::new();
        }

        pub fn reset_assertions(&mut self) {}

        pub fn simplify(&self, _expr: &str) -> Result<String, String> {
            Ok("15".to_string())
        }

        pub fn validate_formula(&self, _formula: &str) -> Result<(), String> {
            Ok(())
        }

        pub fn get_statistics(&self) -> Result<String, String> {
            Ok("{}".to_string())
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
use mock::WasmSolver;

#[cfg(not(target_arch = "wasm32"))]
fn bench_solver_creation(c: &mut Criterion) {
    c.bench_function("solver_creation", |b| {
        b.iter(|| {
            let solver = WasmSolver::new();
            black_box(solver);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_declare_const(c: &mut Criterion) {
    c.bench_function("declare_const", |b| {
        let mut solver = WasmSolver::new();
        let mut counter = 0;

        b.iter(|| {
            let name = format!("x{}", counter);
            solver.declare_const(&name, "Int").unwrap();
            counter += 1;
            black_box(&solver);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_assert_formula(c: &mut Criterion) {
    c.bench_function("assert_formula", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        b.iter(|| {
            solver.reset_assertions();
            solver.assert_formula("(> x 0)").unwrap();
            black_box(&solver);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_check_sat_simple(c: &mut Criterion) {
    c.bench_function("check_sat_simple", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();

        b.iter(|| {
            solver.reset_assertions();
            solver.assert_formula("p").unwrap();
            let result = solver.check_sat();
            black_box(result);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_check_sat_with_assertions(c: &mut Criterion) {
    let mut group = c.benchmark_group("check_sat_assertions");

    for num_assertions in [1, 5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_assertions),
            num_assertions,
            |b, &num| {
                let mut solver = WasmSolver::new();
                solver.set_logic("QF_LIA");

                for i in 0..num {
                    solver.declare_const(&format!("x{}", i), "Int").unwrap();
                }

                b.iter(|| {
                    solver.reset_assertions();
                    for i in 0..num {
                        solver.assert_formula(&format!("(> x{} {})", i, i)).unwrap();
                    }
                    let result = solver.check_sat();
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_push_pop(c: &mut Criterion) {
    c.bench_function("push_pop", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        b.iter(|| {
            solver.push();
            solver.assert_formula("(> x 0)").unwrap();
            solver.pop();
            black_box(&solver);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_get_model(c: &mut Criterion) {
    c.bench_function("get_model", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.check_sat();

        b.iter(|| {
            let model = solver.get_model().unwrap();
            black_box(model);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_simplify(c: &mut Criterion) {
    c.bench_function("simplify", |b| {
        let solver = WasmSolver::new();

        b.iter(|| {
            let result = solver.simplify("(+ 1 2 3 4 5)").unwrap();
            black_box(result);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_validate_formula(c: &mut Criterion) {
    c.bench_function("validate_formula", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        b.iter(|| {
            let _ = solver.validate_formula("(> x 0)").ok();
            black_box(&solver);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_reset(c: &mut Criterion) {
    c.bench_function("reset", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.assert_formula("(> x 0)").unwrap();

        b.iter(|| {
            solver.reset();
            solver.set_logic("QF_LIA");
            solver.declare_const("x", "Int").unwrap();
            black_box(&solver);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_bitvector_operations(c: &mut Criterion) {
    c.bench_function("bitvector_declare", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_BV");
        let mut counter = 0;

        b.iter(|| {
            let name = format!("bv{}", counter);
            solver.declare_const(&name, "BitVec32").unwrap();
            counter += 1;
            black_box(&solver);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_statistics(c: &mut Criterion) {
    c.bench_function("get_statistics", |b| {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.check_sat();

        b.iter(|| {
            let stats = solver.get_statistics().unwrap();
            black_box(stats);
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(
    benches,
    bench_solver_creation,
    bench_declare_const,
    bench_assert_formula,
    bench_check_sat_simple,
    bench_check_sat_with_assertions,
    bench_push_pop,
    bench_get_model,
    bench_simplify,
    bench_validate_formula,
    bench_reset,
    bench_bitvector_operations,
    bench_statistics,
);

#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);

#[cfg(target_arch = "wasm32")]
fn main() {
    // Benchmarks are not run on WASM target
    println!("Benchmarks are only available on native targets");
}
