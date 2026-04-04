# oxiz-smtcomp

SMT-COMP benchmark infrastructure for OxiZ SMT solver.

## Overview

This crate provides tools for running SMT-COMP benchmarks and evaluating OxiZ performance:

- **Benchmark Runner** - Execute SMT-LIB2 benchmark files
- **Statistics** - Collect and analyze solving statistics
- **Regression Testing** - Track performance changes
- **Parallel Execution** - Run benchmarks concurrently
- **Reporting** - Generate HTML reports and plots

## Modules

| Module | Description |
|:-------|:------------|
| `benchmark` | Benchmark execution and timing |
| `loader` | SMT-LIB2 benchmark file loading |
| `parallel` | Parallel benchmark execution with rayon |
| `statistics` | Statistical analysis of results |
| `regression` | Performance regression detection |
| `filtering` | Benchmark filtering and selection |
| `sampling` | Random benchmark sampling |
| `reporter` | Result reporting utilities |
| `html_report` | HTML report generation |
| `plotting` | Result visualization |
| `dashboard` | Interactive dashboard |
| `starexec` | StarExec format compatibility |
| `virtual_best` | Virtual best solver computation |
| `memory` | Memory usage tracking |
| `model_verify` | Model verification |
| `ci_integration` | CI/CD integration |
| `resumption` | Resume interrupted runs |

## Usage

```rust
use oxiz_smtcomp::{BenchmarkRunner, BenchmarkConfig};

let config = BenchmarkConfig {
    timeout_ms: 30000,
    memory_limit_mb: 4096,
    parallel_jobs: 4,
    ..Default::default()
};

let runner = BenchmarkRunner::new(config);
let results = runner.run_directory("benchmarks/QF_LIA/")?;

for result in results {
    println!("{}: {:?} ({:.2}s)",
        result.name,
        result.status,
        result.time_ms as f64 / 1000.0
    );
}
```

## SMT-COMP Compatibility

This crate supports the SMT-COMP benchmark format:
- SMT-LIB2 input files
- StarExec output format
- Standard status reporting (sat/unsat/unknown/timeout/error)

## Dependencies

- `oxiz-core` - Core SMT-LIB parsing
- `oxiz-solver` - SMT solver
- `rayon` - Parallel execution
- `serde` - Serialization

## Status (v0.2.0)

| Metric | Value |
|:-------|:------|
| Version | 0.2.0 |
| Status | Alpha |
| Tests | 104 passing |
| Rust LoC | 10,853 (20 files) |
| Public API items | 370 |
| `todo!`/`unimplemented!` | 0 |

*Last updated: 2026-04-04*

## License

Apache-2.0
