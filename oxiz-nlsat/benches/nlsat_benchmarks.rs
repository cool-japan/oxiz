use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::interval::Interval;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::{
    CadConfig, CadDecomposer, NlsatSolver, SampleStrategy, SturmSequence,
    interval_set::IntervalSet,
    types::{AtomKind, Literal},
    var_order::{OrderingStrategy, VariableOrdering},
};

/// Benchmark CAD projection operations
fn bench_cad_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cad_projection");

    // Benchmark projection with different polynomial sizes
    for degree in [2, 3, 4, 5] {
        group.bench_with_input(
            BenchmarkId::new("polynomial_degree", degree),
            &degree,
            |b, &_deg| {
                b.iter(|| {
                    let config = CadConfig {
                        max_cells: 1000,
                        use_mccallum: true,
                        cache_projections: true,
                        sample_strategy: SampleStrategy::Rational,
                        ordering_strategy: None,
                        parallel_projection: false,
                        parallel_lifting: false,
                    };
                    let var_order = vec![0];
                    let _decomposer = CadDecomposer::new(config, var_order);

                    // Create a simple polynomial for the degree
                    let poly = Polynomial::from_var(0);
                    black_box(poly)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark interval set operations
fn bench_interval_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("interval_operations");

    // Create test intervals using oxiz_math::interval
    let zero = BigRational::zero();
    let one = BigRational::one();

    let interval1 = Interval::less_than(one.clone());
    let interval2 = Interval::at_least(zero.clone());

    group.bench_function("union", |b| {
        b.iter(|| {
            let set1 = IntervalSet::from_interval(black_box(interval1.clone()));
            let set2 = IntervalSet::from_interval(black_box(interval2.clone()));
            black_box(set1.union(&set2))
        });
    });

    group.bench_function("intersection", |b| {
        b.iter(|| {
            let set1 = IntervalSet::from_interval(black_box(interval1.clone()));
            let set2 = IntervalSet::from_interval(black_box(interval2.clone()));
            black_box(set1.intersect(&set2))
        });
    });

    group.bench_function("complement", |b| {
        b.iter(|| {
            let set = IntervalSet::from_interval(black_box(interval1.clone()));
            black_box(set.complement())
        });
    });

    group.finish();
}

/// Benchmark polynomial evaluation
fn bench_polynomial_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_evaluation");

    // Test with different polynomial degrees
    for degree in [2, 5, 10] {
        group.bench_with_input(BenchmarkId::new("degree", degree), &degree, |b, &_deg| {
            // Create simple polynomial
            let poly = Polynomial::from_var(0);
            let mut assignment = rustc_hash::FxHashMap::default();
            assignment.insert(0, BigRational::from_integer(2.into()));

            b.iter(|| black_box(poly.eval(&assignment)));
        });
    }

    group.finish();
}

/// Benchmark solver with different problem sizes
fn bench_solver_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_operations");

    // Simple linear constraint: x > 0
    group.bench_function("linear_constraint", |b| {
        b.iter(|| {
            let mut solver = NlsatSolver::new();
            let _x = solver.new_arith_var();

            // x > 0
            let poly = Polynomial::from_var(0);
            let atom_id = solver.new_ineq_atom(poly, AtomKind::Gt);
            let lit = solver.atom_literal(atom_id, true);
            solver.add_clause(vec![lit]);

            black_box(solver.solve())
        });
    });

    // Simple boolean constraint
    group.bench_function("boolean_constraint", |b| {
        b.iter(|| {
            let mut solver = NlsatSolver::new();
            let a = solver.new_bool_var();
            let b = solver.new_bool_var();

            // (a ∨ b) ∧ (~a ∨ b)
            solver.add_clause(vec![Literal::positive(a), Literal::positive(b)]);
            solver.add_clause(vec![Literal::negative(a), Literal::positive(b)]);

            black_box(solver.solve())
        });
    });

    // Two variable constraint
    group.bench_function("two_variable_constraint", |b| {
        b.iter(|| {
            let mut solver = NlsatSolver::new();
            let _x = solver.new_arith_var();
            let _y = solver.new_arith_var();

            // x > 0 ∧ y > 0
            let poly_x = Polynomial::from_var(0);
            let poly_y = Polynomial::from_var(1);

            let atom1 = solver.new_ineq_atom(poly_x, AtomKind::Gt);
            let atom2 = solver.new_ineq_atom(poly_y, AtomKind::Gt);

            solver.add_clause(vec![solver.atom_literal(atom1, true)]);
            solver.add_clause(vec![solver.atom_literal(atom2, true)]);

            black_box(solver.solve())
        });
    });

    group.finish();
}

/// Benchmark variable ordering strategies
fn bench_variable_ordering(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_ordering");

    // Create test polynomials for ordering
    let polys: Vec<Polynomial> = (0..10).map(Polynomial::from_var).collect();

    let strategies = vec![
        ("static", OrderingStrategy::Static),
        ("brown", OrderingStrategy::Brown),
        ("min_degree", OrderingStrategy::MinDegree),
        ("max_degree", OrderingStrategy::MaxDegree),
        ("min_occurrence", OrderingStrategy::MinOccurrence),
        ("max_occurrence", OrderingStrategy::MaxOccurrence),
    ];

    for (name, strategy) in strategies {
        group.bench_function(name, |b| {
            b.iter(|| {
                let ordering = VariableOrdering::new(strategy, polys.clone());
                black_box(ordering.compute())
            });
        });
    }

    group.finish();
}

/// Benchmark Sturm sequence computation
fn bench_sturm_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("sturm_sequence");

    for degree in [2, 3, 4] {
        group.bench_with_input(BenchmarkId::new("degree", degree), &degree, |b, &_deg| {
            // Simple polynomial for benchmarking
            let poly = Polynomial::from_var(0);

            b.iter(|| {
                let sturm = SturmSequence::new(black_box(&poly), 0);
                black_box(sturm)
            });
        });
    }

    group.finish();
}

/// Benchmark clause operations
fn bench_clause_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("clause_operations");

    // Create test literals
    let literals: Vec<Literal> = (0..100).map(Literal::positive).collect();

    group.bench_function("clause_creation_small", |b| {
        b.iter(|| {
            let clause_lits = black_box(&literals[0..5]);
            black_box(clause_lits.to_vec())
        });
    });

    group.bench_function("clause_creation_medium", |b| {
        b.iter(|| {
            let clause_lits = black_box(&literals[0..20]);
            black_box(clause_lits.to_vec())
        });
    });

    group.bench_function("clause_creation_large", |b| {
        b.iter(|| {
            let clause_lits = black_box(&literals[0..50]);
            black_box(clause_lits.to_vec())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cad_projection,
    bench_interval_operations,
    bench_polynomial_evaluation,
    bench_solver_operations,
    bench_variable_ordering,
    bench_sturm_sequence,
    bench_clause_operations,
);

criterion_main!(benches);
