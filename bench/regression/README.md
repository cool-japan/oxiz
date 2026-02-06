# OxiZ Performance Regression Testing

This directory contains the performance regression testing infrastructure for OxiZ SMT solver.

## Overview

The regression testing framework:
- Runs benchmarks across SAT, Theory, Parser, and MaxSAT components
- Compares results against a stored baseline
- Detects regressions exceeding a configurable threshold (default: 10%)
- Produces CI-friendly output with exit codes

## Directory Structure

```
bench/regression/
├── Cargo.toml           # Benchmark crate configuration
├── baseline.json        # Baseline performance data
├── README.md            # This file
├── src/
│   ├── main.rs          # Regression test runner
│   └── benchmarks.rs    # Benchmark definitions
└── benchmarks/          # Sample SMT files for benchmarking
    ├── sat_simple.cnf
    ├── lia_simple.smt2
    ├── lra_simple.smt2
    ├── bv_simple.smt2
    └── arrays_simple.smt2
```

## Benchmark Categories

### SAT Solving (CDCL Core)
- `sat_3sat_small` - Small 3-SAT instance (10 vars, 30 clauses)
- `sat_3sat_medium` - Medium 3-SAT instance (50 vars, 200 clauses)
- `sat_unit_propagation` - Unit propagation stress test
- `sat_unsat_pigeonhole` - UNSAT pigeonhole problem

### Theory Solving
- `theory_lia_simple` - Simple linear integer arithmetic
- `theory_lia_medium` - Medium complexity LIA with multiple constraints
- `theory_bool_arith` - Boolean-arithmetic combination
- `theory_lia_unsat` - UNSAT LIA instance

### Parser Performance
- `parser_simple` - Simple SMT-LIB parsing
- `parser_medium` - Medium complexity SMT-LIB
- `parser_nested` - Deeply nested expressions
- `parser_dimacs` - DIMACS CNF parsing

### MaxSAT Algorithms
- `maxsat_simple` - Simple MaxSAT instance
- `maxsat_weighted` - Weighted MaxSAT
- `maxsat_partial` - Partial MaxSAT with many hard constraints

## Usage

### Running Benchmarks

```bash
# From the repository root
cd bench/regression
cargo run --release

# With options
cargo run --release -- --threshold 15 --json

# Update baseline with current results
cargo run --release -- --update
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-b, --baseline <FILE>` | Path to baseline file (default: baseline.json) |
| `-t, --threshold <PCT>` | Regression threshold percentage (default: 10) |
| `-u, --update` | Update baseline with current results |
| `--json` | Output in JSON format for automation |
| `--github` | Output with GitHub Actions annotations |
| `-h, --help` | Print help information |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All benchmarks passed (no regressions) |
| 1 | One or more regressions detected |

## CI Integration

The regression tests are integrated into GitHub Actions. See `.github/workflows/perf-regression.yml`.

### Manual Trigger

You can manually trigger the performance regression workflow from the GitHub Actions tab.

### Automated Runs

The workflow runs automatically on:
- Push to main branch
- Pull requests
- Weekly schedule (Sunday at midnight UTC)

## Updating the Baseline

When performance improvements are made, update the baseline:

```bash
cargo run --release -- --update
git add baseline.json
git commit -m "Update performance baseline"
```

## Adding New Benchmarks

1. Add benchmark function in `src/benchmarks.rs`
2. Include it in the appropriate `run_*_benchmarks()` function
3. Run with `--update` to add to baseline
4. Commit the updated baseline

Example:

```rust
// In benchmarks.rs
results.push(run_benchmark("my_new_benchmark", BenchmarkCategory::Sat, 50, || {
    // Benchmark code here
}));
```

## Interpreting Results

### Text Output

```
=== OxiZ Performance Regression Report ===

Summary:
  Total benchmarks: 15
  Regressions:      1 (threshold: 10.0%)
  Improvements:     2
  Unchanged:        11
  New benchmarks:   1

--- sat ---
  [FAIL] sat_3sat_medium                      250.0us    +25.0%  (baseline: 200.0us)
  [ OK ] sat_3sat_small                        52.0us     +4.0%  (baseline: 50.0us)
```

### JSON Output

```json
{
  "has_regression": true,
  "threshold_percent": 10.0,
  "comparisons": [...],
  "summary": {
    "total_benchmarks": 15,
    "regressions": 1,
    "improvements": 2,
    "unchanged": 11,
    "new_benchmarks": 1
  }
}
```

## Troubleshooting

### High Variance in Results

If benchmarks show high variance:
1. Ensure the system is not under load
2. Run with more iterations (modify `run_benchmark` calls)
3. Use a dedicated benchmarking machine

### False Positives

If regressions are detected due to system noise:
1. Re-run the benchmarks
2. Consider increasing the threshold
3. Check for background processes

## License

Apache-2.0 (same as OxiZ)
