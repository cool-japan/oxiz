//! Benchmark ML overhead on solving

use criterion::{Criterion, criterion_group, criterion_main};
use oxiz_ml::branching::BranchingLearner;
use oxiz_ml::restarts::{RestartFeatures, RestartPolicyLearner};
use std::hint::black_box;

fn benchmark_branching_prediction(c: &mut Criterion) {
    let mut learner = BranchingLearner::default_config().unwrap();
    let candidates = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    c.bench_function("branching_prediction", |b| {
        b.iter(|| learner.predict_branch(black_box(&candidates)));
    });
}

fn benchmark_restart_prediction(c: &mut Criterion) {
    let mut learner = RestartPolicyLearner::default_config();
    let features = RestartFeatures::extract(100, 5.0, 0.5, 0.7, 50, 20, 5);

    c.bench_function("restart_prediction", |b| {
        b.iter(|| learner.predict_restart(black_box(&features)));
    });
}

criterion_group!(
    benches,
    benchmark_branching_prediction,
    benchmark_restart_prediction
);
criterion_main!(benches);
