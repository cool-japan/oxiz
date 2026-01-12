use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use oxiz_proof::alethe::AletheProof;
use oxiz_proof::compress::ProofCompressor;
use oxiz_proof::conversion::FormatConverter;
use oxiz_proof::diff::{compute_similarity, diff_proofs};
use oxiz_proof::drat::DratProof;
use oxiz_proof::merge::{merge_proofs, slice_proof};
use oxiz_proof::proof::Proof;

fn bench_proof_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_construction");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut proof = Proof::new();
                for i in 0..size {
                    proof.add_axiom(black_box(format!("p{}", i)));
                }
                proof
            });
        });
    }

    group.finish();
}

fn bench_proof_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_inference");

    for size in [5_usize, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                // Add axioms
                for i in 0..size {
                    let id = proof.add_axiom(format!("p{}", i));
                    axiom_ids.push(id);
                }

                // Add inferences
                if size > 0 {
                    for i in 0..(size - 1) {
                        proof.add_inference(
                            "and",
                            vec![axiom_ids[i], axiom_ids[i + 1]],
                            format!("q{}", i),
                        );
                    }
                }

                black_box(proof)
            });
        });
    }

    group.finish();
}

fn bench_proof_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_stats");

    for size in [50, 100, 200].iter() {
        let mut proof = Proof::new();
        for i in 0..*size {
            proof.add_axiom(format!("p{}", i));
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(proof.stats()));
        });
    }

    group.finish();
}

fn bench_proof_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_compression");

    for size in [20_usize, 50, 100].iter() {
        let mut proof = Proof::new();
        let mut ids = Vec::new();

        // Create a proof with some redundancy
        for i in 0..*size {
            ids.push(proof.add_axiom(format!("p{}", i)));
        }

        if *size >= 2 {
            for i in 0..(size - 2) {
                proof.add_inference("and", vec![ids[i], ids[i + 1]], format!("q{}", i));
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let compressor = ProofCompressor::new();
            b.iter(|| black_box(compressor.compress(&proof)));
        });
    }

    group.finish();
}

fn bench_proof_deduplication(c: &mut Criterion) {
    c.bench_function("proof_deduplication", |b| {
        b.iter(|| {
            let mut proof = Proof::new();

            // Add same axiom multiple times
            for _ in 0..100 {
                proof.add_axiom(black_box("same_conclusion"));
            }

            black_box(proof)
        });
    });
}

fn bench_deep_proof_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_proof_tree");

    for depth in [5, 10, 15].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            b.iter(|| {
                let mut proof = Proof::new();
                let mut prev_id = proof.add_axiom("base");

                // Build a deep chain
                for i in 0..depth {
                    prev_id = proof.add_inference("step", vec![prev_id], format!("level{}", i));
                }

                black_box(proof)
            });
        });
    }

    group.finish();
}

fn bench_wide_proof_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("wide_proof_tree");

    for width in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(width), width, |b, &width| {
            b.iter(|| {
                let mut proof = Proof::new();
                let mut axioms = Vec::new();

                // Build a wide tree
                for i in 0..width {
                    axioms.push(proof.add_axiom(format!("axiom{}", i)));
                }

                // Combine all axioms
                if axioms.len() >= 2 {
                    proof.add_inference("combine", axioms, "combined");
                }

                black_box(proof)
            });
        });
    }

    group.finish();
}

fn bench_resolution_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolution_chain");

    for length in [10, 25, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(length), length, |b, &length| {
            b.iter(|| {
                let mut proof = Proof::new();

                // Create a chain of resolution steps
                let mut prev_id = proof.add_axiom("p0");
                for i in 1..length {
                    let axiom = proof.add_axiom(format!("p{}", i));
                    prev_id =
                        proof.add_inference("resolution", vec![prev_id, axiom], format!("r{}", i));
                }

                black_box(proof)
            });
        });
    }

    group.finish();
}

fn bench_unit_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("unit_propagation");

    for size in [20, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut proof = Proof::new();

                // Simulate unit propagation
                let mut units = Vec::new();
                for i in 0..size {
                    units.push(proof.add_axiom(format!("unit{}", i)));
                }

                // Create derived facts through unit propagation
                for i in 0..(size - 1) {
                    proof.add_inference(
                        "unit-propagate",
                        vec![units[i], units[i + 1]],
                        format!("derived{}", i),
                    );
                }

                black_box(proof)
            });
        });
    }

    group.finish();
}

fn bench_theory_lemmas(c: &mut Criterion) {
    let mut group = c.benchmark_group("theory_lemmas");

    for size in [10, 30, 60].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut proof = Proof::new();

                // Create equality chain (common in EUF)
                let mut eqs = Vec::new();
                for i in 0..size {
                    eqs.push(proof.add_axiom(format!("(= x{} x{})", i, i + 1)));
                }

                // Apply transitivity to build longer chains
                let mut prev = eqs[0];
                for (i, &eq) in eqs.iter().enumerate().skip(1) {
                    prev =
                        proof.add_inference("trans", vec![prev, eq], format!("(= x0 x{})", i + 1));
                }

                black_box(proof)
            });
        });
    }

    group.finish();
}

fn bench_batch_axioms(c: &mut Criterion) {
    c.bench_function("batch_add_axioms", |b| {
        b.iter(|| {
            let mut proof = Proof::new();
            let conclusions: Vec<String> = (0..100).map(|i| format!("axiom{}", i)).collect();
            black_box(proof.add_axioms(conclusions));
        });
    });
}

fn bench_node_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_queries");

    // Setup a proof for querying
    let mut proof = Proof::new();
    let axioms: Vec<_> = (0..50)
        .map(|i| proof.add_axiom(format!("p{}", i)))
        .collect();
    for i in 0..40 {
        proof.add_inference(
            if i % 3 == 0 { "and" } else { "or" },
            vec![axioms[i], axioms[i + 1]],
            format!("q{}", i),
        );
    }

    group.bench_function("find_by_rule", |b| {
        b.iter(|| black_box(proof.find_nodes_by_rule("and")));
    });

    group.bench_function("count_rule", |b| {
        b.iter(|| black_box(proof.count_rule("and")));
    });

    if let Some(root) = proof.root() {
        group.bench_function("get_ancestors", |b| {
            b.iter(|| black_box(proof.get_all_ancestors(root)));
        });
    }

    group.finish();
}

fn bench_proof_merging(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_merging");

    for num_proofs in [2, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_proofs),
            num_proofs,
            |b, &num_proofs| {
                // Create multiple proofs to merge
                let proofs: Vec<Proof> = (0..num_proofs)
                    .map(|i| {
                        let mut proof = Proof::new();
                        for j in 0..20 {
                            proof.add_axiom(format!("p{}_{}", i, j));
                        }
                        proof
                    })
                    .collect();

                b.iter(|| black_box(merge_proofs(&proofs)));
            },
        );
    }

    group.finish();
}

fn bench_proof_slicing(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_slicing");

    for depth in [5, 10, 15].iter() {
        let mut proof = Proof::new();
        let mut prev_id = proof.add_axiom("base");

        // Build a deep chain
        for i in 0..*depth {
            prev_id = proof.add_inference("step", vec![prev_id], format!("level{}", i));
        }

        let target = prev_id;

        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, _| {
            b.iter(|| black_box(slice_proof(&proof, target)));
        });
    }

    group.finish();
}

fn bench_proof_diffing(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_diffing");

    for size in [20, 50, 100].iter() {
        let mut proof1 = Proof::new();
        let mut proof2 = Proof::new();

        // Create similar but different proofs
        for i in 0..*size {
            proof1.add_axiom(format!("p{}", i));
            if i % 2 == 0 {
                proof2.add_axiom(format!("p{}", i));
            } else {
                proof2.add_axiom(format!("q{}", i));
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(diff_proofs(&proof1, &proof2)));
        });
    }

    group.finish();
}

fn bench_proof_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_similarity");

    for size in [20, 50, 100].iter() {
        let mut proof1 = Proof::new();
        let mut proof2 = Proof::new();

        // Create similar proofs with 70% overlap
        for i in 0..*size {
            proof1.add_axiom(format!("p{}", i));
            if i < (*size * 7 / 10) {
                proof2.add_axiom(format!("p{}", i));
            } else {
                proof2.add_axiom(format!("q{}", i));
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(compute_similarity(&proof1, &proof2)));
        });
    }

    group.finish();
}

fn bench_drat_to_alethe_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("drat_to_alethe_conversion");

    for num_clauses in [10, 50, 100, 200].iter() {
        let mut drat = DratProof::new();

        // Create a DRAT proof with many clauses
        for i in 0..*num_clauses {
            drat.add_clause(vec![i, -(i + 1)]);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(num_clauses),
            num_clauses,
            |b, _| {
                let converter = FormatConverter::new();
                b.iter(|| black_box(converter.drat_to_alethe(&drat)));
            },
        );
    }

    group.finish();
}

fn bench_alethe_to_lfsc_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("alethe_to_lfsc_conversion");

    for num_assumptions in [10, 30, 50, 100].iter() {
        let mut alethe = AletheProof::new();

        // Create an Alethe proof with many assumptions
        for i in 0..*num_assumptions {
            alethe.assume(format!("(= x{} {})", i, i));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(num_assumptions),
            num_assumptions,
            |b, _| {
                let converter = FormatConverter::new();
                b.iter(|| black_box(converter.alethe_to_lfsc(&alethe)));
            },
        );
    }

    group.finish();
}

fn bench_alethe_resolution_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("alethe_resolution_conversion");

    for num_steps in [5, 10, 20, 50].iter() {
        let mut alethe = AletheProof::new();

        // Build a resolution chain
        let mut indices = Vec::new();
        for i in 0..*num_steps {
            indices.push(alethe.assume(format!("p{}", i)));
        }

        // Add resolution steps
        for i in 0..((*num_steps) - 1) {
            alethe.resolution(vec![format!("q{}", i)], vec![indices[i], indices[i + 1]]);
        }

        group.bench_with_input(BenchmarkId::from_parameter(num_steps), num_steps, |b, _| {
            let converter = FormatConverter::new();
            b.iter(|| black_box(converter.alethe_to_lfsc(&alethe)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_proof_construction,
    bench_proof_inference,
    bench_proof_stats,
    bench_proof_compression,
    bench_proof_deduplication,
    bench_deep_proof_tree,
    bench_wide_proof_tree,
    bench_resolution_chain,
    bench_unit_propagation,
    bench_theory_lemmas,
    bench_batch_axioms,
    bench_node_queries,
    bench_proof_merging,
    bench_proof_slicing,
    bench_proof_diffing,
    bench_proof_similarity,
    bench_drat_to_alethe_conversion,
    bench_alethe_to_lfsc_conversion,
    bench_alethe_resolution_conversion
);

criterion_main!(benches);
