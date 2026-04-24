//! Criterion benchmarks for the EUF (Equality and Uninterpreted Functions) theory solver.
//!
//! Covers five steady-state workloads:
//!   - `bench_euf_intern_leaf_chain`     — intern 10 000 leaf terms
//!   - `bench_euf_intern_app_tree`       — right-leaning 5 000-node application chain
//!   - `bench_euf_merge_congruence_chain`— N=100 chain that drives congruence closure
//!   - `bench_euf_merge_injective`       — N=100 pairs; measures merge cost for injective-style workload
//!   - `bench_euf_push_pop_cycle`        — 100 push/intern/merge/pop cycles (incremental backtrack)

use criterion::{Criterion, criterion_group, criterion_main};
use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::euf::{EufSolver, FunctionProperties};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Benchmark 1 — intern 10 000 leaf terms
// ---------------------------------------------------------------------------

fn bench_euf_intern_leaf_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("euf");

    group.bench_function("intern_leaf_chain", |b| {
        b.iter(|| {
            let mut solver = EufSolver::new();
            for i in 0u32..10_000 {
                let idx = solver.intern(TermId::new(i));
                black_box(idx);
            }
            black_box(solver)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2 — right-leaning application chain  f(f(f(x, x), x), x) ...
// ---------------------------------------------------------------------------

fn bench_euf_intern_app_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("euf");

    group.bench_function("intern_app_tree", |b| {
        b.iter(|| {
            let mut solver = EufSolver::new();
            // func symbol id = 0; use a unique base TermId space so intern doesn't dedup
            let func: u32 = 0;
            // intern the leaf term for x
            let x_term = TermId::new(1);
            let x = solver.intern(x_term);

            // Each iteration interns a new app node f(prev, x).
            // We use unique TermIds (starting at 2) so each is a fresh node.
            let mut prev = x;
            for i in 0u32..5_000 {
                let app_term = TermId::new(i + 2);
                let node = solver.intern_app(app_term, func, [prev, x]);
                prev = node;
            }
            black_box(prev)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3 — congruence chain: merge a_0=a_1, ..., a_{N-1}=a_N;
//               then check that f(a_0) ≡ f(a_N) via congruence closure.
// ---------------------------------------------------------------------------

fn bench_euf_merge_congruence_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("euf");
    const N: u32 = 100;

    group.bench_function("merge_congruence_chain", |b| {
        b.iter(|| {
            let mut solver = EufSolver::new();
            let func: u32 = 1;

            // Intern leaf terms a_0 .. a_N  (TermIds 0..=N)
            let leaves: Vec<u32> = (0..=N).map(|i| solver.intern(TermId::new(i))).collect();

            // Intern f(a_0) .. f(a_N) — TermIds start at N+1 to avoid collisions
            let apps: Vec<u32> = (0..=N)
                .map(|i| solver.intern_app(TermId::new(N + 1 + i), func, [leaves[i as usize]]))
                .collect();

            // Merge chain: a_0 = a_1 = ... = a_N
            for i in 0..N {
                solver
                    .merge(leaves[i as usize], leaves[(i + 1) as usize], TermId::new(i))
                    .expect("merge in congruence chain");
            }

            // After all merges, congruence closure must have propagated:
            // f(a_0) ≡ f(a_N).  Measure the equivalence check cost too.
            let eq = solver.are_equal(apps[0], apps[N as usize]);
            black_box(eq)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4 — injective-style workload: N pairs, merge f(x_i)=f(y_i).
//               The solver does NOT currently implement back-propagation for
//               injectivity; this bench measures the raw merge + congruence cost.
// ---------------------------------------------------------------------------

fn bench_euf_merge_injective(c: &mut Criterion) {
    let mut group = c.benchmark_group("euf");
    const N: u32 = 100;

    group.bench_function("merge_injective", |b| {
        b.iter(|| {
            let mut solver = EufSolver::new();
            let func: u32 = 2;

            // Register function — no injective field, use default props
            solver.register_function(
                func,
                FunctionProperties {
                    associative: false,
                    commutative: false,
                    has_identity: false,
                },
            );

            // TermId layout: x_i = i, y_i = N+i, f(x_i) = 2N+i, f(y_i) = 3N+i
            let xs: Vec<u32> = (0..N).map(|i| solver.intern(TermId::new(i))).collect();
            let ys: Vec<u32> = (0..N).map(|i| solver.intern(TermId::new(N + i))).collect();
            let fxs: Vec<u32> = (0..N)
                .map(|i| solver.intern_app(TermId::new(2 * N + i), func, [xs[i as usize]]))
                .collect();
            let fys: Vec<u32> = (0..N)
                .map(|i| solver.intern_app(TermId::new(3 * N + i), func, [ys[i as usize]]))
                .collect();

            // Merge f(x_i) = f(y_i) for each i
            for i in 0..N as usize {
                solver
                    .merge(fxs[i], fys[i], TermId::new(4 * N + i as u32))
                    .expect("merge in injective bench");
            }

            black_box(solver.node_count())
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5 — push/pop cycle: realistic accumulated state + incremental push
//
// This bench measures the trail-based pop() under conditions where the
// algorithmic advantage is visible: a large pre-push base (2000 leaf terms +
// 1000 app nodes = 3000 nodes) vs. small per-push work (50 leaves + 25 apps).
//
// Old pop() cost: sig_table.clear() + fp_table.clear() + rescan 3000 nodes = O(3000)
// New pop() cost: replay 75 trail entries = O(75)
// Expected ratio: ~40× for the sig/fp rewind portion
//
// The solver is created once and reused across Criterion samples; pop() must
// restore exactly the pre-push state so every iteration sees the same 3000-node
// baseline (correctness is implicitly tested here).
// ---------------------------------------------------------------------------

fn bench_euf_push_pop_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("euf");

    group.bench_function("push_pop_cycle", |b| {
        // ---- Setup phase (outside b.iter) ----
        // Build a realistic pre-push state: 2000 leaf terms + 1000 app nodes.
        // This represents a solver that has processed many constraints before
        // the incremental push.
        let mut solver = EufSolver::new();
        let func: u32 = 3;

        // Intern 2000 leaf terms (TermIds 0..2000)
        for i in 0u32..2000 {
            let leaf = solver.intern(TermId::new(i));
            black_box(leaf);
        }

        // Intern 1000 app nodes f(i*2, i*2+1) with unique TermIds 2000..3000
        for i in 0u32..1000 {
            let a = i * 2;
            let b_t = i * 2 + 1;
            let app = solver.intern_app(TermId::new(2000 + i), func, [a, b_t]);
            black_box(app);
        }

        // TermId counter for per-iter unique IDs (never overlaps with setup range)
        // Start at 10_000 to leave clear headroom above setup IDs (0..3000).
        let mut counter = 10_000u32;

        // ---- Benchmark loop ----
        // Each iteration: push → intern 50 leaves + 25 apps → pop
        // With 3000 pre-push nodes the trail pays off: old code would rescan
        // all 3000; new code replays 75 trail entries.
        b.iter(|| {
            solver.push();

            // Intern 50 fresh leaf terms
            for j in 0u32..50 {
                let leaf = solver.intern(TermId::new(counter + j));
                black_box(leaf);
            }

            // Intern 25 fresh app nodes, args taken modulo the pre-push pool
            // to guarantee the args are valid pre-existing node indices.
            for j in 0u32..25 {
                let a = (counter + 50 + j * 2) % 2000;
                let b_t = (counter + 50 + j * 2 + 1) % 2000;
                let app = solver.intern_app(TermId::new(counter + 100 + j), func, [a, b_t]);
                black_box(app);
            }

            counter += 200; // step far enough to avoid TermId reuse across iters

            solver.pop();

            black_box(solver.node_count())
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    euf_benches,
    bench_euf_intern_leaf_chain,
    bench_euf_intern_app_tree,
    bench_euf_merge_congruence_chain,
    bench_euf_merge_injective,
    bench_euf_push_pop_cycle,
);
criterion_main!(euf_benches);
