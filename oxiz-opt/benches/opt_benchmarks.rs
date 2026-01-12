//! Benchmarks for oxiz-opt
//!
//! Comprehensive benchmarks for optimization algorithms and data structures

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_opt::{
    MaxSatAlgorithm, MaxSatConfig, MaxSatSolver, PreprocessConfig, Preprocessor, Weight,
};
use oxiz_sat::{Lit, Var};

// ============================================================================
// Weight Operations Benchmarks
// ============================================================================

fn bench_weight_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_operations");

    group.bench_function("add_int", |b| {
        let w1 = Weight::from(1000i64);
        let w2 = Weight::from(2000i64);
        b.iter(|| black_box(w1.add(&w2)));
    });

    group.bench_function("add_rational", |b| {
        let r1 = BigRational::new(BigInt::from(1000), BigInt::from(3));
        let r2 = BigRational::new(BigInt::from(2000), BigInt::from(3));
        let w1 = Weight::Rational(r1);
        let w2 = Weight::Rational(r2);
        b.iter(|| black_box(w1.add(&w2)));
    });

    group.bench_function("sub_int", |b| {
        let w1 = Weight::from(2000i64);
        let w2 = Weight::from(1000i64);
        b.iter(|| black_box(w1.sub(&w2)));
    });

    group.bench_function("mul_scalar", |b| {
        let w = Weight::from(1000i64);
        b.iter(|| black_box(w.mul_scalar(5)));
    });

    group.bench_function("comparison", |b| {
        let w1 = Weight::from(1000i64);
        let w2 = Weight::from(2000i64);
        b.iter(|| black_box(w1 < w2));
    });

    group.bench_function("min_weight", |b| {
        let w1 = Weight::from(1000i64);
        let w2 = Weight::from(2000i64);
        b.iter(|| black_box(w1.min_weight(&w2)));
    });

    group.bench_function("max_weight", |b| {
        let w1 = Weight::from(1000i64);
        let w2 = Weight::from(2000i64);
        b.iter(|| black_box(w1.max_weight(&w2)));
    });

    group.bench_function("is_zero", |b| {
        let w = Weight::from(1000i64);
        b.iter(|| black_box(w.is_zero()));
    });

    group.bench_function("clone", |b| {
        let w = Weight::from(1000i64);
        b.iter(|| black_box(w.clone()));
    });

    group.finish();
}

// ============================================================================
// MaxSAT Algorithm Benchmarks
// ============================================================================

fn bench_maxsat_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsat_algorithms");

    for algo in [
        MaxSatAlgorithm::FuMalik,
        MaxSatAlgorithm::Oll,
        MaxSatAlgorithm::Msu3,
        MaxSatAlgorithm::Pmres,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", algo)),
            &algo,
            |b, &algo| {
                b.iter(|| {
                    let config = MaxSatConfig::builder().algorithm(algo).build();
                    let mut solver = MaxSatSolver::with_config(config);

                    // Add a simple instance
                    for i in 0..10 {
                        let lit = Lit::pos(Var::new(i));
                        solver.add_soft([lit]);
                    }

                    black_box(solver.solve())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Preprocessing Benchmarks
// ============================================================================

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    group.bench_function("preprocess_small", |b| {
        let mut preprocessor = Preprocessor::new();
        let soft_clauses: Vec<_> = (0..20)
            .map(|i| {
                use oxiz_opt::SoftClause;
                use oxiz_opt::SoftId;
                let id = SoftId::new(i);
                let lit = Lit::pos(Var::new(i));
                SoftClause::unit(id, lit, Weight::one())
            })
            .collect();

        b.iter(|| black_box(preprocessor.preprocess(&soft_clauses)));
    });

    group.bench_function("preprocess_with_duplicates", |b| {
        let mut preprocessor = Preprocessor::with_config(PreprocessConfig {
            merge_duplicates: true,
            ..Default::default()
        });

        let soft_clauses: Vec<_> = (0..20)
            .flat_map(|i| {
                use oxiz_opt::SoftClause;
                use oxiz_opt::SoftId;
                // Create duplicates
                vec![
                    SoftClause::unit(SoftId::new(i * 2), Lit::pos(Var::new(i)), Weight::one()),
                    SoftClause::unit(SoftId::new(i * 2 + 1), Lit::pos(Var::new(i)), Weight::one()),
                ]
            })
            .collect();

        b.iter(|| black_box(preprocessor.preprocess(&soft_clauses)));
    });

    group.finish();
}

// ============================================================================
// Soft Clause Operations
// ============================================================================

fn bench_soft_clause_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_clause");

    group.bench_function("create_unit", |b| {
        use oxiz_opt::{SoftClause, SoftId};
        b.iter(|| {
            let id = SoftId::new(0);
            let lit = Lit::pos(Var::new(0));
            let weight = Weight::one();
            black_box(SoftClause::unit(id, lit, weight))
        });
    });

    group.bench_function("create_clause", |b| {
        use oxiz_opt::{SoftClause, SoftId};
        b.iter(|| {
            let id = SoftId::new(0);
            let lits = vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))];
            let weight = Weight::one();
            black_box(SoftClause::new(id, lits, weight))
        });
    });

    group.finish();
}

// ============================================================================
// Large Instance Benchmarks
// ============================================================================

fn bench_large_maxsat_instances(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_maxsat");
    group.sample_size(10);

    // Benchmark: Medium instance (50 variables, 100 soft clauses)
    group.bench_function("medium_instance_50vars", |b| {
        b.iter(|| {
            let mut solver = MaxSatSolver::new();

            // Add some hard clauses (at-least-one constraints)
            for chunk in (0..50).collect::<Vec<_>>().chunks(3) {
                let lits: Vec<_> = chunk.iter().map(|&i| Lit::pos(Var::new(i))).collect();
                solver.add_hard(lits);
            }

            // Add soft clauses (preference for variables to be false)
            for i in 0..100 {
                let var_idx = i % 50;
                solver.add_soft([Lit::neg(Var::new(var_idx))]);
            }

            black_box(solver.solve())
        });
    });

    // Benchmark: Weighted instance (30 variables, varied weights)
    group.bench_function("weighted_instance_30vars", |b| {
        b.iter(|| {
            let config = MaxSatConfig::builder()
                .algorithm(MaxSatAlgorithm::WMax)
                .build();
            let mut solver = MaxSatSolver::with_config(config);

            // Add hard constraint: at least one must be true
            let hard_clause: Vec<_> = (0..10).map(|i| Lit::pos(Var::new(i))).collect();
            solver.add_hard(hard_clause);

            // Add weighted soft clauses
            for i in 0..30 {
                let weight = Weight::from((i % 5 + 1) as i64);
                solver.add_soft_weighted([Lit::neg(Var::new(i))], weight);
            }

            black_box(solver.solve())
        });
    });

    // Benchmark: Core-guided on conflicting instance
    group.bench_function("conflicting_instance_20vars", |b| {
        b.iter(|| {
            let mut solver = MaxSatSolver::new();

            // Add conflicting hard clauses
            for i in (0..20).step_by(2) {
                solver.add_hard([Lit::pos(Var::new(i)), Lit::pos(Var::new(i + 1))]);
            }

            // Add soft clauses that conflict with hard clauses
            for i in 0..20 {
                solver.add_soft([Lit::neg(Var::new(i))]);
            }

            black_box(solver.solve())
        });
    });

    // Benchmark: OLL algorithm on medium instance
    group.bench_function("oll_medium_instance", |b| {
        b.iter(|| {
            let config = MaxSatConfig::builder()
                .algorithm(MaxSatAlgorithm::Oll)
                .build();
            let mut solver = MaxSatSolver::with_config(config);

            // Create a pigeon-hole like instance
            for i in 0..5 {
                for j in 0..5 {
                    let var = Var::new(i * 5 + j);
                    solver.add_soft([Lit::pos(var)]);
                }
            }

            // Add constraints: each row must have at least one true
            for i in 0..5 {
                let row: Vec<_> = (0..5).map(|j| Lit::pos(Var::new(i * 5 + j))).collect();
                solver.add_hard(row);
            }

            black_box(solver.solve())
        });
    });

    group.finish();
}

// ============================================================================
// Cardinality Constraint Benchmarks
// ============================================================================

fn bench_cardinality_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("cardinality_constraints");

    group.bench_function("totalizer_10_lits", |b| {
        use oxiz_opt::Totalizer;
        b.iter(|| {
            let lits: Vec<_> = (0..10).map(|i| Lit::pos(Var::new(i))).collect();
            let totalizer = Totalizer::new(&lits, 1000);
            black_box(totalizer)
        });
    });

    group.bench_function("totalizer_50_lits", |b| {
        use oxiz_opt::Totalizer;
        b.iter(|| {
            let lits: Vec<_> = (0..50).map(|i| Lit::pos(Var::new(i))).collect();
            let totalizer = Totalizer::new(&lits, 1000);
            black_box(totalizer)
        });
    });

    group.bench_function("encode_at_most_k_10_lits", |b| {
        use oxiz_opt::{CardinalityEncoding, encode_at_most_k};
        b.iter(|| {
            let lits: Vec<_> = (0..10).map(|i| Lit::pos(Var::new(i))).collect();
            let (clauses, _, _) = encode_at_most_k(&lits, 5, CardinalityEncoding::Totalizer, 1000);
            black_box(clauses)
        });
    });

    group.finish();
}

// ============================================================================
// WCNF Format Benchmarks
// ============================================================================

fn bench_wcnf_format(c: &mut Criterion) {
    use oxiz_opt::WcnfInstance;

    let mut group = c.benchmark_group("wcnf_format");

    group.bench_function("parse_small_instance", |b| {
        let wcnf = r#"c Small WCNF instance
p wcnf 10 15 100
100 1 2 0
100 -1 3 0
50 1 0
30 -2 -3 0
20 4 5 0
100 -4 6 0
10 7 0
5 -7 8 0
100 1 -8 0
25 9 10 0
15 -9 0
35 -10 1 0
100 2 4 0
100 -3 -5 0
45 6 7 0
"#;
        b.iter(|| {
            let instance = WcnfInstance::parse(wcnf.as_bytes()).unwrap();
            black_box(instance)
        });
    });

    group.bench_function("parse_medium_instance", |b| {
        // Generate a medium-sized WCNF instance programmatically
        let mut wcnf = String::from("p wcnf 50 100 1000\n");
        for i in 1..=50 {
            wcnf.push_str(&format!("1000 {} {} 0\n", i, i + 1));
        }
        for i in 1..=50 {
            wcnf.push_str(&format!("{} {} 0\n", i * 10, i));
        }

        b.iter(|| {
            let instance = WcnfInstance::parse(wcnf.as_bytes()).unwrap();
            black_box(instance)
        });
    });

    group.bench_function("parse_and_solve_small", |b| {
        let wcnf = r#"p wcnf 5 8 100
100 1 2 0
100 -1 3 0
50 1 0
30 -2 -3 0
20 4 5 0
10 -4 0
15 5 0
25 -5 1 0
"#;
        b.iter(|| {
            let instance = WcnfInstance::parse(wcnf.as_bytes()).unwrap();
            let mut solver = instance.to_solver();
            let _ = black_box(solver.solve());
        });
    });

    group.bench_function("parse_weighted_instance", |b| {
        // Test parsing of instances with various weight formats
        let wcnf = r#"c Weighted instance
p wcnf 10 12 h
h 1 2 0
h -1 3 0
500 1 0
300 -2 -3 0
200 4 5 0
h -4 6 0
100 7 0
50 -7 8 0
h 1 -8 0
250 9 10 0
150 -9 0
350 -10 1 0
"#;
        b.iter(|| {
            let instance = WcnfInstance::parse(wcnf.as_bytes()).unwrap();
            black_box(instance)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_weight_operations,
    bench_maxsat_algorithms,
    bench_preprocessing,
    bench_soft_clause_operations,
    bench_large_maxsat_instances,
    bench_cardinality_constraints,
    bench_wcnf_format
);
criterion_main!(benches);
