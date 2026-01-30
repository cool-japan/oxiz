# Performance Optimization Guide

## Introduction

This guide provides strategies for optimizing OxiZ performance in various scenarios.

## Profiling

### Built-in Statistics

```rust
let mut solver = Solver::new();
solver.enable_statistics();

// ... solving ...

let stats = solver.get_statistics();
println!("Decisions: {}", stats.decisions);
println!("Propagations: {}", stats.propagations);
println!("Conflicts: {}", stats.conflicts);
println!("Time in SAT: {:?}", stats.time_sat);
println!("Time in theories: {:?}", stats.time_theory);
```

### External Profiling

**CPU Profiling (cargo-flamegraph)**:
```bash
cargo install flamegraph
cargo flamegraph --bin your_solver -- input.smt2
```

**Memory Profiling (heaptrack)**:
```bash
heaptrack ./target/release/oxiz-cli input.smt2
heaptrack_gui heaptrack.oxiz-cli.*.gz
```

## Optimization Strategies

### 1. Set the Logic

**Impact**: 20-50% speedup

```rust
// GOOD: Specific logic enables optimizations
solver.set_logic("QF_LIA");

// AVOID: Generic logic disables optimizations
solver.set_logic("ALL");
```

**Why**: Specific logics enable:
- Specialized decision heuristics
- Theory-specific preprocessing
- Optimized data structures

### 2. Preprocessing with Tactics

**Impact**: 10-100x speedup on some instances

```rust
use oxiz_core::tactic::{StatelessSimplifyTactic, SolveEqsTactic, Tactic};

let mut goal = Goal::new(vec![formula]);

// Simplify
let simplified = StatelessSimplifyTactic.apply(&goal);

// Solve equations (variable elimination)
let solved = SolveEqsTactic.apply(&simplified);

// Assert preprocessed formula
solver.assert_goal(solved, &mut tm);
```

**Effective tactics**:
- `simplify`: Constant folding, boolean simplification
- `solve-eqs`: Variable elimination via substitution
- `propagate-ineqs`: Bound propagation
- `ctx-simplify`: Contextual simplification

### 3. Incremental Solving

**Impact**: 5-50x for repeated queries

```rust
// GOOD: Reuse solver instance
let mut solver = Solver::new();
solver.assert(base_constraints, &mut tm);

for query in queries {
    solver.push();
    solver.assert(query, &mut tm);
    let result = solver.check(&mut tm);
    solver.pop();
}

// AVOID: Creating new solver each time
for query in queries {
    let mut solver = Solver::new();
    solver.assert(base_constraints, &mut tm);
    solver.assert(query, &mut tm);
    let result = solver.check(&mut tm);
}
```

**Why**: Incremental solving reuses:
- Learned clauses
- Simplification results
- Theory state

### 4. Batch Assertions

**Impact**: 10-30% speedup

```rust
// GOOD: Batch assertions
for formula in formulas {
    solver.assert(formula, &mut tm);
}
let result = solver.check(&mut tm);

// AVOID: Check after each assertion
for formula in formulas {
    solver.assert(formula, &mut tm);
    solver.check(&mut tm); // Repeated work
}
```

### 5. Arena Allocation

**Impact**: 2-5x for term-heavy workloads

```rust
use oxiz_core::alloc::Arena;

let mut arena = Arena::new(Default::default());

// Allocate many terms
for i in 0..10000 {
    arena.alloc(Term::new(i));
}

// Fast bulk deallocation (single operation)
drop(arena);
```

**Comparison**:
- Box allocation: O(n) individual deallocations
- Arena: O(1) bulk deallocation

### 6. Resource Limits

**Impact**: Prevents timeout/OOM, enables early termination

```rust
solver.set_time_limit(Duration::from_secs(60));
solver.set_memory_limit(2 * 1024 * 1024 * 1024); // 2 GB

match solver.check(&mut tm) {
    SolverResult::Unknown => {
        // Timeout or resource limit
        println!("Aborted due to limits");
    }
    _ => {}
}
```

### 7. Parallel Solving

**Impact**: Near-linear speedup on multi-core

```rust
use oxiz_solver::ParallelSolver;

let mut parallel = ParallelSolver::new(num_cpus::get());

parallel.add_strategy(Strategy::CDCL);
parallel.add_strategy(Strategy::CDCL_with_restarts);
parallel.add_strategy(Strategy::LocalSearch);

let result = parallel.solve(formula, &mut tm);
```

### 8. Clause Deletion

**Impact**: 20-40% reduction in memory, 10-20% speedup

```rust
use oxiz_core::config::{ClauseDeletionStrategy, Config};

let config = Config {
    sat_params: SatParams {
        clause_deletion: ClauseDeletionStrategy::Activity,
        deletion_threshold: 0.5,
        ..Default::default()
    },
    ..Default::default()
};

let mut solver = Solver::with_config(config);
```

**Strategies**:
- `Activity`: Delete low-activity clauses (Glucose-style)
- `Size`: Delete longer clauses first
- `LBD`: Delete high LBD clauses (keep glue clauses)

### 9. Restart Strategy

**Impact**: 10-50% speedup (instance-dependent)

```rust
use oxiz_sat::RestartStrategy;

let config = Config {
    sat_params: SatParams {
        restart_strategy: RestartStrategy::Luby(100),
        ..Default::default()
    },
    ..Default::default()
};
```

**Strategies**:
- `Fixed(n)`: Restart every n conflicts
- `Geometric { base, factor }`: Geometric growth
- `Luby(n)`: Luby sequence (universal strategy)
- `Glucose`: LBD-based dynamic restarts

## Theory-Specific Optimizations

### Linear Arithmetic (LIA/LRA)

**1. Normalize bounds**:
```rust
// Before: x <= 5, x <= 3, x <= 7
// After: x <= 3 (tightest bound)
```

**2. Use simplex with dual**:
```rust
let config = SimplexConfig {
    use_dual: true,
    pivoting_rule: PivotingRule::Bland, // Prevents cycling
    ..Default::default()
};
```

**3. Enable cutting planes**:
```rust
let config = LIAConfig {
    enable_cuts: true,
    max_cuts_per_round: 10,
    ..Default::default()
};
```

### Arrays

**1. Lazy axiom instantiation**:
```rust
// Only instantiate when necessary
let config = ArrayConfig {
    eager_extensionality: false,
    lazy_writes: true,
    ..Default::default()
};
```

**2. Index canonicalization**:
```rust
// Canonicalize indices to reduce case splits
// a[i] vs a[j] -> check if i = j first
```

### Bitvectors

**1. Bit-blasting threshold**:
```rust
let config = BVConfig {
    bitblast_threshold: 64, // Blast only small widths
    use_symbolic_for_large: true,
    ..Default::default()
};
```

**2. Rewrite to simplify**:
```rust
// bvadd(x, 0) -> x
// bvmul(x, 1) -> x
// bvand(x, x) -> x
```

## Memory Optimization

### 1. Hash Consing

**Automatic in TermManager**:
```rust
let x1 = tm.mk_var("x", tm.sorts.int_sort);
let x2 = tm.mk_var("x", tm.sorts.int_sort);
// x1 == x2 (same pointer, shared memory)
```

### 2. Clause Minimization

```rust
let config = Config {
    sat_params: SatParams {
        minimize_learned_clauses: true,
        minimization_algorithm: MinimizationAlg::Recursive,
        ..Default::default()
    },
    ..Default::default()
};
```

### 3. Watch List Compression

```rust
// Periodically compress watch lists
solver.compress_watches();
```

## Benchmarking

### Microbenchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_term_creation(c: &mut Criterion) {
    c.bench_function("term_creation", |b| {
        b.iter(|| {
            let mut tm = TermManager::new();
            for i in 0..1000 {
                black_box(tm.mk_int(BigInt::from(i)));
            }
        });
    });
}

criterion_group!(benches, bench_term_creation);
criterion_main!(benches);
```

### End-to-End Benchmarks

```rust
use std::time::Instant;

let start = Instant::now();
let result = solver.check(&mut tm);
let elapsed = start.elapsed();

println!("Result: {:?} in {:?}", result, elapsed);
```

## Common Performance Issues

### Issue 1: Slow Parsing

**Symptom**: High CPU time in parsing phase

**Solution**:
```rust
// Reuse TermManager across multiple files
let mut tm = TermManager::new();
for file in files {
    parse_script(file, &mut tm)?;
}
```

### Issue 2: Memory Growth

**Symptom**: Increasing memory usage over time

**Solution**:
```rust
// Enable clause deletion
solver.enable_clause_deletion();

// Periodic garbage collection
solver.gc();
```

### Issue 3: Theory Conflicts

**Symptom**: Many theory conflicts, slow convergence

**Solution**:
```rust
// Use delayed theory combination
solver.set_theory_mode(TheoryMode::DelayedCombination);
```

## Performance Checklist

- [ ] Set specific logic (not "ALL")
- [ ] Enable preprocessing tactics
- [ ] Use incremental solving for multiple queries
- [ ] Batch assertions before checking
- [ ] Set resource limits
- [ ] Enable clause deletion for long-running
- [ ] Choose appropriate restart strategy
- [ ] Profile before optimizing
- [ ] Benchmark before and after changes

## Typical Speedups

| Optimization | Speedup | Applicability |
|--------------|---------|---------------|
| Set logic | 1.2-1.5x | All |
| Simplify tactic | 1.1-100x | Complex formulas |
| Incremental solving | 5-50x | Multiple queries |
| Batch assertions | 1.1-1.3x | Many assertions |
| Arena allocation | 2-5x | AST-heavy |
| Parallel solving | 1.5-4x | Hard instances |
| Clause deletion | 1.1-1.4x | Long runs |
| Theory optimization | 1.2-10x | Theory-heavy |

## References

- "Efficient E-Matching for SMT Solvers" (CAV 2007)
- "Faster and More Accurate: A Faster and More Accurate SAT Solver" (SAT 2009)
- "Predicting Learnt Clauses Quality" (IJCAI 2009)
