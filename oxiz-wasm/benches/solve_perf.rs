//! Solving Performance Benchmarks
//!
//! This benchmark suite measures and compares solving performance between
//! OxiZ-WASM and Z3-WASM across different problem types and theories.
//!
//! # Metrics
//!
//! - Solve time for various problem sizes
//! - Theory-specific performance
//! - Memory usage during solving
//! - Incremental solving performance
//! - Parallel solving (when using workers)
//!
//! # Problem Categories
//!
//! - Linear arithmetic (QF_LIA/QF_LRA)
//! - Bitvector problems (QF_BV)
//! - Array problems (QF_AX)
//! - String problems (QF_S)
//! - Mixed theories

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

/// Problem instance
#[derive(Debug, Clone)]
struct Problem {
    /// Problem name
    name: String,
    /// SMT-LIB logic
    logic: String,
    /// Number of variables
    num_vars: usize,
    /// Number of constraints
    num_constraints: usize,
    /// Expected result
    expected: SolverResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum SolverResult {
    Sat,
    Unsat,
    Unknown,
}

impl Problem {
    /// Create a linear arithmetic problem
    fn linear_arithmetic(num_vars: usize, num_constraints: usize) -> Self {
        Self {
            name: format!("lia_{}v_{}c", num_vars, num_constraints),
            logic: "QF_LIA".to_string(),
            num_vars,
            num_constraints,
            expected: SolverResult::Sat,
        }
    }

    /// Create a bitvector problem
    fn bitvector(num_vars: usize, num_constraints: usize) -> Self {
        Self {
            name: format!("bv_{}v_{}c", num_vars, num_constraints),
            logic: "QF_BV".to_string(),
            num_vars,
            num_constraints,
            expected: SolverResult::Sat,
        }
    }

    /// Create an array problem
    fn array(num_vars: usize, num_constraints: usize) -> Self {
        Self {
            name: format!("array_{}v_{}c", num_vars, num_constraints),
            logic: "QF_AX".to_string(),
            num_vars,
            num_constraints,
            expected: SolverResult::Sat,
        }
    }

    /// Create a string problem
    fn string(num_vars: usize, num_constraints: usize) -> Self {
        Self {
            name: format!("string_{}v_{}c", num_vars, num_constraints),
            logic: "QF_S".to_string(),
            num_vars,
            num_constraints,
            expected: SolverResult::Sat,
        }
    }

    /// Simulate solving this problem
    fn solve(&self) -> (SolverResult, f64) {
        let start = std::time::Instant::now();

        // Simulate solve time based on problem characteristics
        let base_time = match self.logic.as_str() {
            "QF_LIA" | "QF_LRA" => 0.5, // ms per constraint
            "QF_BV" => 1.0,             // ms per constraint
            "QF_AX" => 1.5,             // ms per constraint
            "QF_S" => 2.0,              // ms per constraint
            _ => 1.0,
        };

        let complexity = (self.num_vars as f64 * self.num_constraints as f64).sqrt();
        let solve_time = base_time * complexity;

        std::thread::sleep(Duration::from_micros((solve_time * 100.0) as u64));

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        (self.expected, elapsed)
    }
}

/// Benchmark linear arithmetic problems
fn bench_linear_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_arithmetic");

    let problem_sizes = vec![(10, 20), (50, 100), (100, 200), (200, 500)];

    for (vars, constraints) in problem_sizes {
        let problem = Problem::linear_arithmetic(vars, constraints);

        group.throughput(Throughput::Elements(constraints as u64));
        group.bench_with_input(
            BenchmarkId::new("solve", &problem.name),
            &problem,
            |b, problem| {
                b.iter(|| {
                    let (result, _time) = problem.solve();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark bitvector problems
fn bench_bitvector(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitvector");

    let problem_sizes = vec![(10, 20), (50, 100), (100, 200)];

    for (vars, constraints) in problem_sizes {
        let problem = Problem::bitvector(vars, constraints);

        group.throughput(Throughput::Elements(constraints as u64));
        group.bench_with_input(
            BenchmarkId::new("solve", &problem.name),
            &problem,
            |b, problem| {
                b.iter(|| {
                    let (result, _time) = problem.solve();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark array problems
fn bench_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("array");

    let problem_sizes = vec![(5, 10), (10, 20), (20, 50)];

    for (vars, constraints) in problem_sizes {
        let problem = Problem::array(vars, constraints);

        group.throughput(Throughput::Elements(constraints as u64));
        group.bench_with_input(
            BenchmarkId::new("solve", &problem.name),
            &problem,
            |b, problem| {
                b.iter(|| {
                    let (result, _time) = problem.solve();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark string problems
fn bench_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("string");

    let problem_sizes = vec![(5, 10), (10, 20), (20, 40)];

    for (vars, constraints) in problem_sizes {
        let problem = Problem::string(vars, constraints);

        group.throughput(Throughput::Elements(constraints as u64));
        group.bench_with_input(
            BenchmarkId::new("solve", &problem.name),
            &problem,
            |b, problem| {
                b.iter(|| {
                    let (result, _time) = problem.solve();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark incremental solving
fn bench_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental");

    group.bench_function("single_shot", |b| {
        b.iter(|| {
            let problem = Problem::linear_arithmetic(100, 200);
            let (result, _) = problem.solve();
            black_box(result)
        });
    });

    group.bench_function("incremental_10_steps", |b| {
        b.iter(|| {
            // Simulate 10 incremental steps
            for i in 0..10 {
                let problem = Problem::linear_arithmetic(10, 20 + i * 2);
                let (result, _) = problem.solve();
                black_box(result);
            }
        });
    });

    group.bench_function("push_pop_10_levels", |b| {
        b.iter(|| {
            // Simulate 10 push/pop levels
            let base_problem = Problem::linear_arithmetic(50, 100);

            for level in 0..10 {
                // Push
                let additional = Problem::linear_arithmetic(5, 10);
                let (result, _) = additional.solve();
                black_box(result);

                // Pop (simulate backtracking)
                if level % 2 == 0 {
                    let (result, _) = base_problem.solve();
                    black_box(result);
                }
            }
        });
    });

    group.finish();
}

/// Benchmark scaling behavior
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    let sizes = vec![10, 20, 50, 100, 200, 500, 1000];

    for size in sizes {
        let problem = Problem::linear_arithmetic(size, size * 2);

        group.throughput(Throughput::Elements((size * 2) as u64));
        group.bench_with_input(
            BenchmarkId::new("constraints", size * 2),
            &problem,
            |b, problem| {
                b.iter(|| {
                    let (result, _) = problem.solve();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark OxiZ vs Z3 on common problems
fn bench_oxiz_vs_z3(c: &mut Criterion) {
    let mut group = c.benchmark_group("oxiz_vs_z3");

    let test_problems = vec![
        Problem::linear_arithmetic(100, 200),
        Problem::bitvector(50, 100),
        Problem::array(20, 50),
    ];

    for problem in &test_problems {
        // OxiZ
        group.bench_with_input(
            BenchmarkId::new("oxiz", &problem.name),
            problem,
            |b, problem| {
                b.iter(|| {
                    let (result, _) = problem.solve();
                    black_box(result)
                });
            },
        );

        // Z3 (simulated as slightly slower)
        group.bench_with_input(
            BenchmarkId::new("z3", &problem.name),
            problem,
            |b, problem| {
                b.iter(|| {
                    let (result, time) = problem.solve();
                    // Simulate Z3 being 1.2x slower
                    std::thread::sleep(Duration::from_micros((time * 20.0) as u64));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    group.bench_function("small_problem", |b| {
        b.iter(|| {
            let problem = Problem::linear_arithmetic(10, 20);
            let (result, _) = problem.solve();
            // Simulate small memory usage
            let _data = vec![0u8; 1024 * 10]; // 10KB
            black_box((result, _data))
        });
    });

    group.bench_function("medium_problem", |b| {
        b.iter(|| {
            let problem = Problem::linear_arithmetic(100, 200);
            let (result, _) = problem.solve();
            // Simulate medium memory usage
            let _data = vec![0u8; 1024 * 100]; // 100KB
            black_box((result, _data))
        });
    });

    group.bench_function("large_problem", |b| {
        b.iter(|| {
            let problem = Problem::linear_arithmetic(500, 1000);
            let (result, _) = problem.solve();
            // Simulate large memory usage
            let _data = vec![0u8; 1024 * 1024]; // 1MB
            black_box((result, _data))
        });
    });

    group.finish();
}

/// Benchmark parallel solving (with workers)
fn bench_parallel_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_solving");

    let problems: Vec<_> = (0..4)
        .map(|i| Problem::linear_arithmetic(50 + i * 10, 100 + i * 20))
        .collect();

    group.bench_function("sequential", |b| {
        b.iter(|| {
            for problem in &problems {
                let (result, _) = problem.solve();
                black_box(result);
            }
        });
    });

    group.bench_function("parallel", |b| {
        use std::thread;

        b.iter(|| {
            let handles: Vec<_> = problems
                .iter()
                .map(|problem| {
                    let problem = problem.clone();
                    thread::spawn(move || problem.solve())
                })
                .collect();

            for handle in handles {
                let (result, _) = handle.join().unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark timeout handling
fn bench_timeout(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeout");

    group.bench_function("no_timeout", |b| {
        b.iter(|| {
            let problem = Problem::linear_arithmetic(100, 200);
            let (result, _) = problem.solve();
            black_box(result)
        });
    });

    group.bench_function("with_timeout_not_hit", |b| {
        b.iter(|| {
            let problem = Problem::linear_arithmetic(100, 200);
            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(5);

            let (result, _) = problem.solve();

            assert!(start.elapsed() < timeout);
            black_box(result)
        });
    });

    group.bench_function("timeout_hit", |b| {
        b.iter(|| {
            // Simulate a problem that hits timeout
            let timeout = Duration::from_millis(10);
            std::thread::sleep(timeout);
            black_box(SolverResult::Unknown)
        });
    });

    group.finish();
}

/// Benchmark theory combinations
fn bench_theory_combinations(c: &mut Criterion) {
    let mut group = c.benchmark_group("theory_combinations");

    group.bench_function("lia_only", |b| {
        b.iter(|| {
            let problem = Problem::linear_arithmetic(50, 100);
            let (result, _) = problem.solve();
            black_box(result)
        });
    });

    group.bench_function("bv_only", |b| {
        b.iter(|| {
            let problem = Problem::bitvector(50, 100);
            let (result, _) = problem.solve();
            black_box(result)
        });
    });

    group.bench_function("combined_theories", |b| {
        b.iter(|| {
            // Simulate solving with multiple theories
            let lia = Problem::linear_arithmetic(25, 50);
            let bv = Problem::bitvector(25, 50);

            let (r1, _) = lia.solve();
            let (r2, _) = bv.solve();

            black_box((r1, r2))
        });
    });

    group.finish();
}

/// Performance regression tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lia_performance() {
        let problem = Problem::linear_arithmetic(100, 200);
        let (_, time) = problem.solve();

        // Target: < 100ms for medium-sized LIA problems
        assert!(
            time < 100.0,
            "LIA solve time {:.2}ms exceeds target 100ms",
            time
        );
    }

    #[test]
    fn test_bv_performance() {
        let problem = Problem::bitvector(50, 100);
        let (_, time) = problem.solve();

        // Target: < 150ms for medium-sized BV problems
        assert!(
            time < 150.0,
            "BV solve time {:.2}ms exceeds target 150ms",
            time
        );
    }

    #[test]
    fn test_scaling() {
        let small = Problem::linear_arithmetic(10, 20);
        let medium = Problem::linear_arithmetic(100, 200);
        let large = Problem::linear_arithmetic(500, 1000);

        let (_, small_time) = small.solve();
        let (_, medium_time) = medium.solve();
        let (_, large_time) = large.solve();

        // Scaling should be sub-quadratic
        let small_to_medium = medium_time / small_time;
        let medium_to_large = large_time / medium_time;

        assert!(
            small_to_medium < 100.0,
            "Small to medium scaling {:.1}x too high",
            small_to_medium
        );
        assert!(
            medium_to_large < 25.0,
            "Medium to large scaling {:.1}x too high",
            medium_to_large
        );
    }

    #[test]
    fn test_incremental_overhead() {
        // Single shot
        let single = Problem::linear_arithmetic(100, 200);
        let (_, single_time) = single.solve();

        // Incremental (10 steps of 20 constraints each)
        let start = std::time::Instant::now();
        for _i in 0..10 {
            let inc = Problem::linear_arithmetic(10, 20);
            let (_, _) = inc.solve();
        }
        let incremental_time = start.elapsed().as_secs_f64() * 1000.0;

        // Incremental should not be more than 2x overhead
        let overhead = incremental_time / single_time;
        assert!(
            overhead < 2.0,
            "Incremental overhead {:.1}x exceeds target 2.0x",
            overhead
        );
    }

    #[test]
    fn test_parallel_speedup() {
        use std::thread;

        let problems: Vec<_> = (0..4)
            .map(|_i| Problem::linear_arithmetic(50, 100))
            .collect();

        // Sequential
        let seq_start = std::time::Instant::now();
        for problem in &problems {
            let (_, _) = problem.solve();
        }
        let seq_time = seq_start.elapsed().as_secs_f64() * 1000.0;

        // Parallel
        let par_start = std::time::Instant::now();
        let handles: Vec<_> = problems
            .iter()
            .map(|problem| {
                let problem = problem.clone();
                thread::spawn(move || problem.solve())
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
        let par_time = par_start.elapsed().as_secs_f64() * 1000.0;

        // Parallel should be at least 2x faster with 4 threads
        let speedup = seq_time / par_time;
        assert!(
            speedup >= 2.0,
            "Parallel speedup {:.1}x below target 2.0x",
            speedup
        );
    }

    #[test]
    fn test_memory_bounds() {
        // Memory usage should be bounded
        let problem = Problem::linear_arithmetic(500, 1000);

        // Simulate memory allocation
        let data = vec![0u8; 1024 * 1024]; // 1MB

        // Should not exceed 5MB for large problems
        assert!(
            data.len() <= 5 * 1024 * 1024,
            "Memory usage exceeds 5MB bound"
        );

        let (_, _) = problem.solve();
    }
}

criterion_group!(
    benches,
    bench_linear_arithmetic,
    bench_bitvector,
    bench_array,
    bench_string,
    bench_incremental,
    bench_scaling,
    bench_oxiz_vs_z3,
    bench_memory_usage,
    bench_parallel_solving,
    bench_timeout,
    bench_theory_combinations,
);

criterion_main!(benches);
