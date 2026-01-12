use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_math::*;

// Helper to create rationals
fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

fn benchmark_polynomial_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial");

    // Create test polynomials
    let p1 = polynomial::Polynomial::from_coeffs_int(&[
        (1, &[(0, 2)]), // x^2
        (2, &[(0, 1)]), // 2x
        (1, &[]),       // 1
    ]);

    let p2 = polynomial::Polynomial::from_coeffs_int(&[
        (1, &[(0, 1)]), // x
        (1, &[]),       // 1
    ]);

    group.bench_function("add", |b| b.iter(|| black_box(&p1) + black_box(&p2)));

    group.bench_function("mul", |b| b.iter(|| black_box(&p1) * black_box(&p2)));

    group.bench_function("gcd", |b| {
        b.iter(|| black_box(&p1).gcd_univariate(black_box(&p2)))
    });

    // Evaluation benchmark
    let mut assignment = rustc_hash::FxHashMap::default();
    assignment.insert(0, rat(5));

    group.bench_function("eval", |b| {
        b.iter(|| black_box(&p1).eval(black_box(&assignment)))
    });

    group.finish();
}

fn benchmark_grobner_basis(c: &mut Criterion) {
    let mut group = c.benchmark_group("grobner");
    // Reduce sample size for expensive operations
    group.sample_size(10);

    // Small system: x^2 - 2, y - x
    let small_polys = vec![
        polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (-2, &[]),      // -2
        ]),
        polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(1, 1)]),  // y
            (-1, &[(0, 1)]), // -x
        ]),
    ];

    group.bench_function("buchberger_small", |b| {
        b.iter(|| grobner::grobner_basis(black_box(&small_polys)))
    });

    // Skip F4 and F5 benchmarks - they are too slow for regular benchmarking
    // Uncomment to benchmark when needed:
    // group.bench_function("f4_small", |b| {
    //     b.iter(|| grobner::grobner_basis_f4(black_box(&small_polys)))
    // });
    //
    // group.bench_function("f5_small", |b| {
    //     b.iter(|| grobner::grobner_basis_f5(black_box(&small_polys)))
    // });

    group.finish();
}

fn benchmark_root_isolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("root_isolation");

    // x^3 - 2 (has one real root)
    let cubic = polynomial::Polynomial::from_coeffs_int(&[
        (1, &[(0, 3)]), // x^3
        (-2, &[]),      // -2
    ]);

    group.bench_function("cubic", |b| b.iter(|| black_box(&cubic).isolate_roots(0)));

    // x^4 - 5x^2 + 4 (has four real roots)
    let quartic = polynomial::Polynomial::from_coeffs_int(&[
        (1, &[(0, 4)]),  // x^4
        (-5, &[(0, 2)]), // -5x^2
        (4, &[]),        // 4
    ]);

    group.bench_function("quartic", |b| {
        b.iter(|| black_box(&quartic).isolate_roots(0))
    });

    group.finish();
}

fn benchmark_simplex(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex");

    // Create a simple tableau and benchmark row operations
    let mut tableau = simplex::SimplexTableau::new();
    let x = tableau.fresh_var();
    let y = tableau.fresh_var();
    let z = tableau.fresh_var();

    // Create a row: z = 5 + 2x + 3y
    let mut coeffs = rustc_hash::FxHashMap::default();
    coeffs.insert(x, rat(2));
    coeffs.insert(y, rat(3));
    let row = simplex::Row::from_expr(z, rat(5), coeffs.clone());

    group.bench_function("add_row", |b| {
        b.iter(|| {
            let mut t = black_box(tableau.clone());
            let r = black_box(row.clone());
            t.add_row(r)
        })
    });

    // Benchmark row evaluation
    let mut assignment = rustc_hash::FxHashMap::default();
    assignment.insert(x, rat(1));
    assignment.insert(y, rat(2));

    group.bench_function("row_eval", |b| {
        b.iter(|| {
            let r = black_box(&row);
            let a = black_box(&assignment);
            r.eval(a)
        })
    });

    group.finish();
}

fn benchmark_rational_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("rational");

    let a = BigInt::from(123456789);
    let b = BigInt::from(987654321);

    group.bench_function("gcd_euclidean", |b_bench| {
        b_bench.iter(|| rational::gcd_bigint(black_box(a.clone()), black_box(b.clone())))
    });

    group.bench_function("gcd_binary", |b_bench| {
        b_bench.iter(|| rational::gcd_binary(black_box(a.clone()), black_box(b.clone())))
    });

    let n = BigInt::from(1000000007); // Prime
    group.bench_function("is_prime", |b_bench| {
        b_bench.iter(|| rational::is_prime(black_box(&n), 20))
    });

    group.finish();
}

fn benchmark_interval_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("interval");

    let i1 = interval::Interval::closed(rat(1), rat(5));
    let i2 = interval::Interval::closed(rat(3), rat(7));

    group.bench_function("intersect", |b| {
        b.iter(|| black_box(&i1).intersect(black_box(&i2)))
    });

    group.bench_function("union", |b| b.iter(|| black_box(&i1).union(black_box(&i2))));

    group.bench_function("add", |b| b.iter(|| black_box(&i1).add(black_box(&i2))));

    group.bench_function("mul", |b| b.iter(|| black_box(&i1).mul(black_box(&i2))));

    group.finish();
}

fn benchmark_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix");

    use matrix::Matrix;
    use num_rational::Rational64;

    // Small 10x10 matrix
    let size = 10;
    let data: Vec<Rational64> = (0..size * size)
        .map(|i| Rational64::new(i as i64 + 1, 1))
        .collect();

    let m = Matrix::from_vec(size, size, data);

    group.bench_function("gaussian_elimination_10x10", |b| {
        b.iter(|| {
            let m_copy = black_box(m.clone());
            m_copy.gaussian_elimination()
        })
    });

    group.finish();
}

fn benchmark_number_theory(c: &mut Criterion) {
    let mut group = c.benchmark_group("number_theory");

    let n = BigInt::from(10007); // Prime
    let a = BigInt::from(1234);

    group.bench_function("euler_totient", |b| {
        b.iter(|| rational::euler_totient(black_box(&n)))
    });

    group.bench_function("jacobi_symbol", |b| {
        b.iter(|| rational::jacobi_symbol(black_box(&a), black_box(&n)))
    });

    group.bench_function("factorial", |b| {
        b.iter(|| rational::factorial(black_box(20)))
    });

    group.bench_function("binomial_100_50", |b| {
        b.iter(|| rational::binomial(black_box(100), black_box(50)))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_polynomial_operations,
    benchmark_grobner_basis,
    benchmark_root_isolation,
    benchmark_simplex,
    benchmark_rational_arithmetic,
    benchmark_interval_operations,
    benchmark_matrix_operations,
    benchmark_number_theory,
);

criterion_main!(benches);
