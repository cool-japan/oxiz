//! Benchmark definitions for OxiZ performance regression testing
//!
//! This module defines benchmarks for various OxiZ components:
//! - SAT solving (CDCL core)
//! - Theory solving (LIA, LRA, BV, Arrays)
//! - Parser performance
//! - MaxSAT algorithms

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::parse_script;
use oxiz_opt::{MaxSatSolver, Weight};
use oxiz_sat::{DimacsParser, Lit, Solver as SatSolver, SolverResult as SatResult, Var};
use oxiz_solver::{Solver, SolverResult};
use std::time::{Duration, Instant};

/// Result of a single benchmark run
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Category of the benchmark
    pub category: BenchmarkCategory,
    /// Average time in microseconds
    pub avg_time_us: f64,
    /// Minimum time in microseconds
    pub min_time_us: f64,
    /// Maximum time in microseconds
    pub max_time_us: f64,
    /// Number of iterations
    pub iterations: u32,
}

/// Benchmark categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BenchmarkCategory {
    /// SAT solving (CDCL core performance)
    Sat,
    /// Theory solving (LIA, LRA, BV, Arrays)
    Theory,
    /// Parser performance
    Parser,
    /// MaxSAT algorithms
    MaxSat,
}

impl std::fmt::Display for BenchmarkCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkCategory::Sat => write!(f, "sat"),
            BenchmarkCategory::Theory => write!(f, "theory"),
            BenchmarkCategory::Parser => write!(f, "parser"),
            BenchmarkCategory::MaxSat => write!(f, "maxsat"),
        }
    }
}

/// Run a benchmark with the given function and return timing results
fn run_benchmark<F>(name: &str, category: BenchmarkCategory, iterations: u32, mut f: F) -> BenchmarkResult
where
    F: FnMut(),
{
    let mut times = Vec::with_capacity(iterations as usize);

    // Warmup
    for _ in 0..3 {
        f();
    }

    // Actual measurements
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    let times_us: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1_000_000.0).collect();
    let sum: f64 = times_us.iter().sum();
    let avg = sum / iterations as f64;
    let min = times_us.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times_us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    BenchmarkResult {
        name: name.to_string(),
        category,
        avg_time_us: avg,
        min_time_us: min,
        max_time_us: max,
        iterations,
    }
}

/// Run all SAT benchmarks
pub fn run_sat_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Benchmark 1: Simple 3-SAT instance
    results.push(run_benchmark("sat_3sat_small", BenchmarkCategory::Sat, 100, || {
        let mut solver = SatSolver::new();
        // Create a simple 3-SAT instance with 10 variables and 30 clauses
        for _ in 0..10 {
            let _ = solver.new_var();
        }

        // Add random-ish clauses
        for i in 0..30 {
            let v1 = Var::new((i % 10) as u32);
            let v2 = Var::new(((i + 3) % 10) as u32);
            let v3 = Var::new(((i + 7) % 10) as u32);
            let l1 = if i % 2 == 0 { Lit::pos(v1) } else { Lit::neg(v1) };
            let l2 = if i % 3 == 0 { Lit::pos(v2) } else { Lit::neg(v2) };
            let l3 = if i % 5 == 0 { Lit::pos(v3) } else { Lit::neg(v3) };
            solver.add_clause([l1, l2, l3]);
        }

        let _ = solver.solve();
    }));

    // Benchmark 2: Medium 3-SAT instance
    results.push(run_benchmark("sat_3sat_medium", BenchmarkCategory::Sat, 50, || {
        let mut solver = SatSolver::new();
        for _ in 0..50 {
            let _ = solver.new_var();
        }

        for i in 0..200 {
            let v1 = Var::new((i % 50) as u32);
            let v2 = Var::new(((i * 7 + 3) % 50) as u32);
            let v3 = Var::new(((i * 13 + 7) % 50) as u32);
            let l1 = if i % 2 == 0 { Lit::pos(v1) } else { Lit::neg(v1) };
            let l2 = if i % 3 == 0 { Lit::pos(v2) } else { Lit::neg(v2) };
            let l3 = if i % 5 == 0 { Lit::pos(v3) } else { Lit::neg(v3) };
            solver.add_clause([l1, l2, l3]);
        }

        let _ = solver.solve();
    }));

    // Benchmark 3: Unit propagation stress test
    results.push(run_benchmark("sat_unit_propagation", BenchmarkCategory::Sat, 100, || {
        let mut solver = SatSolver::new();
        for _ in 0..100 {
            let _ = solver.new_var();
        }

        // Create a chain of implications
        for i in 0..99 {
            let vi = Var::new(i);
            let vj = Var::new(i + 1);
            solver.add_clause([Lit::neg(vi), Lit::pos(vj)]);
        }

        // Force the first variable to be true
        solver.add_clause([Lit::pos(Var::new(0))]);

        let _ = solver.solve();
    }));

    // Benchmark 4: UNSAT instance
    results.push(run_benchmark("sat_unsat_pigeonhole", BenchmarkCategory::Sat, 20, || {
        let mut solver = SatSolver::new();

        // Pigeonhole problem: 4 pigeons, 3 holes (UNSAT)
        // Variables: p_i_j means pigeon i is in hole j
        let pigeons = 4;
        let holes = 3;

        for _ in 0..(pigeons * holes) {
            let _ = solver.new_var();
        }

        // Each pigeon must be in at least one hole
        for i in 0..pigeons {
            let mut clause = Vec::new();
            for j in 0..holes {
                clause.push(Lit::pos(Var::new((i * holes + j) as u32)));
            }
            solver.add_clause(clause);
        }

        // No two pigeons in the same hole
        for j in 0..holes {
            for i1 in 0..pigeons {
                for i2 in (i1 + 1)..pigeons {
                    solver.add_clause([
                        Lit::neg(Var::new((i1 * holes + j) as u32)),
                        Lit::neg(Var::new((i2 * holes + j) as u32)),
                    ]);
                }
            }
        }

        let _ = solver.solve();
    }));

    results
}

/// Run all theory benchmarks
pub fn run_theory_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Benchmark 1: LIA simple
    results.push(run_benchmark("theory_lia_simple", BenchmarkCategory::Theory, 50, || {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);
        let five = tm.mk_int(BigInt::from(5));
        let ten = tm.mk_int(BigInt::from(10));

        // x >= 5 AND x <= 10 AND y = x + 1
        solver.assert(tm.mk_ge(x, five), &mut tm);
        solver.assert(tm.mk_le(x, ten), &mut tm);
        let x_plus_1 = tm.mk_add(vec![x, tm.mk_int(BigInt::from(1))]);
        solver.assert(tm.mk_eq(y, x_plus_1), &mut tm);

        let _ = solver.check(&mut tm);
    }));

    // Benchmark 2: LIA with multiple constraints
    results.push(run_benchmark("theory_lia_medium", BenchmarkCategory::Theory, 30, || {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        solver.set_logic("QF_LIA");

        let vars: Vec<_> = (0..5)
            .map(|i| tm.mk_var(&format!("x{}", i), tm.sorts.int_sort))
            .collect();

        // Sum of all variables <= 100
        let sum = tm.mk_add(vars.clone());
        let hundred = tm.mk_int(BigInt::from(100));
        solver.assert(tm.mk_le(sum, hundred), &mut tm);

        // Each variable >= 0 and <= 30
        let zero = tm.mk_int(BigInt::from(0));
        let thirty = tm.mk_int(BigInt::from(30));
        for &v in &vars {
            solver.assert(tm.mk_ge(v, zero), &mut tm);
            solver.assert(tm.mk_le(v, thirty), &mut tm);
        }

        let _ = solver.check(&mut tm);
    }));

    // Benchmark 3: Boolean + Arithmetic combination
    results.push(run_benchmark("theory_bool_arith", BenchmarkCategory::Theory, 40, || {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let b = tm.mk_var("b", tm.sorts.bool_sort);
        let five = tm.mk_int(BigInt::from(5));
        let zero = tm.mk_int(BigInt::from(0));

        // b => x > 5
        // !b => x <= 0
        let x_gt_5 = tm.mk_gt(x, five);
        let x_le_0 = tm.mk_le(x, zero);
        let not_b = tm.mk_not(b);

        let impl1 = tm.mk_implies(b, x_gt_5);
        let impl2 = tm.mk_implies(not_b, x_le_0);

        solver.assert(impl1, &mut tm);
        solver.assert(impl2, &mut tm);

        let _ = solver.check(&mut tm);
    }));

    // Benchmark 4: UNSAT LIA
    results.push(run_benchmark("theory_lia_unsat", BenchmarkCategory::Theory, 30, || {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let five = tm.mk_int(BigInt::from(5));
        let three = tm.mk_int(BigInt::from(3));

        // x > 5 AND x < 3 (UNSAT)
        solver.assert(tm.mk_gt(x, five), &mut tm);
        solver.assert(tm.mk_lt(x, three), &mut tm);

        let _ = solver.check(&mut tm);
    }));

    results
}

/// Run all parser benchmarks
pub fn run_parser_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Benchmark 1: Simple SMT-LIB parsing
    let simple_input = r#"
        (declare-const x Int)
        (declare-const y Int)
        (assert (> x 0))
        (assert (< y 10))
        (assert (= (+ x y) 15))
        (check-sat)
    "#;

    results.push(run_benchmark("parser_simple", BenchmarkCategory::Parser, 200, || {
        let mut tm = TermManager::new();
        let _ = parse_script(simple_input, &mut tm);
    }));

    // Benchmark 2: Medium complexity SMT-LIB
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
        (assert (= (+ c d e) 75))
        (assert (< (+ a e) 30))
        (check-sat)
        (get-model)
    "#;

    results.push(run_benchmark("parser_medium", BenchmarkCategory::Parser, 100, || {
        let mut tm = TermManager::new();
        let _ = parse_script(medium_input, &mut tm);
    }));

    // Benchmark 3: Nested expressions
    let nested_input = r#"
        (declare-const x Int)
        (assert (= x (+ 1 (+ 2 (+ 3 (+ 4 (+ 5 (+ 6 (+ 7 (+ 8 (+ 9 10)))))))))))
        (check-sat)
    "#;

    results.push(run_benchmark("parser_nested", BenchmarkCategory::Parser, 200, || {
        let mut tm = TermManager::new();
        let _ = parse_script(nested_input, &mut tm);
    }));

    // Benchmark 4: DIMACS parsing
    let dimacs_input = "p cnf 10 20\n1 2 3 0\n-1 2 4 0\n1 -2 5 0\n-1 -2 6 0\n\
        3 4 5 0\n-3 4 6 0\n3 -4 7 0\n-3 -4 8 0\n\
        5 6 7 0\n-5 6 8 0\n5 -6 9 0\n-5 -6 10 0\n\
        7 8 9 0\n-7 8 10 0\n7 -8 1 0\n-7 -8 2 0\n\
        9 10 1 0\n-9 10 2 0\n9 -10 3 0\n-9 -10 4 0\n";

    results.push(run_benchmark("parser_dimacs", BenchmarkCategory::Parser, 500, || {
        let _ = DimacsParser::parse(dimacs_input);
    }));

    results
}

/// Run all MaxSAT benchmarks
pub fn run_maxsat_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Benchmark 1: Simple MaxSAT
    results.push(run_benchmark("maxsat_simple", BenchmarkCategory::MaxSat, 50, || {
        let mut solver = MaxSatSolver::new();

        // Create variables
        for i in 0..10 {
            // Hard constraint: at least one of consecutive variables
            let v1 = Var::new(i);
            let v2 = Var::new((i + 1) % 10);
            solver.add_hard([Lit::pos(v1), Lit::pos(v2)]);
        }

        // Soft constraints: prefer negative literals
        for i in 0..10 {
            solver.add_soft([Lit::neg(Var::new(i))]);
        }

        let _ = solver.solve();
    }));

    // Benchmark 2: Weighted MaxSAT
    results.push(run_benchmark("maxsat_weighted", BenchmarkCategory::MaxSat, 30, || {
        let mut solver = MaxSatSolver::new();

        // Hard constraints
        for i in 0..5 {
            let v1 = Var::new(i);
            let v2 = Var::new(i + 5);
            solver.add_hard([Lit::pos(v1), Lit::pos(v2)]);
        }

        // Weighted soft constraints
        for i in 0..10 {
            let weight = Weight::from((i + 1) as u64);
            solver.add_soft_weighted([Lit::neg(Var::new(i))], weight);
        }

        let _ = solver.solve();
    }));

    // Benchmark 3: Partial MaxSAT (more hard constraints)
    results.push(run_benchmark("maxsat_partial", BenchmarkCategory::MaxSat, 30, || {
        let mut solver = MaxSatSolver::new();

        // Many hard constraints forming a satisfiable core
        for i in 0..15 {
            let v1 = Var::new(i % 20);
            let v2 = Var::new((i + 7) % 20);
            solver.add_hard([Lit::pos(v1), Lit::pos(v2)]);
        }

        // Some soft constraints that may conflict
        for i in 0..20 {
            solver.add_soft([Lit::neg(Var::new(i))]);
        }

        let _ = solver.solve();
    }));

    results
}

/// Run all benchmarks and return results
pub fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    results.extend(run_sat_benchmarks());
    results.extend(run_theory_benchmarks());
    results.extend(run_parser_benchmarks());
    results.extend(run_maxsat_benchmarks());

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sat_benchmarks_run() {
        let results = run_sat_benchmarks();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_theory_benchmarks_run() {
        let results = run_theory_benchmarks();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_parser_benchmarks_run() {
        let results = run_parser_benchmarks();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_maxsat_benchmarks_run() {
        let results = run_maxsat_benchmarks();
        assert!(!results.is_empty());
    }
}
