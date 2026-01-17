# OxiZ Fuzzing

This directory contains fuzz tests for the OxiZ SMT solver using [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html) and libFuzzer.

## Installation

Install cargo-fuzz (requires nightly Rust):

```bash
cargo install cargo-fuzz
```

## Fuzz Targets

### fuzz_smtlib_parser

Fuzzes the SMT-LIB2 parser with arbitrary byte sequences to find crashes or panics during parsing.

```bash
cd /path/to/oxiz/fuzz
cargo +nightly fuzz run fuzz_smtlib_parser
```

### fuzz_term_builder

Fuzzes term construction with random operations (boolean, integer, real, array, string, bitvector) to ensure the TermManager handles all cases correctly.

```bash
cd /path/to/oxiz/fuzz
cargo +nightly fuzz run fuzz_term_builder
```

### fuzz_solver

Fuzzes the solver with structured random SMT commands to test solver behavior under various constraint combinations.

```bash
cd /path/to/oxiz/fuzz
cargo +nightly fuzz run fuzz_solver
```

## Common Options

### Run with a time limit

```bash
cargo +nightly fuzz run fuzz_smtlib_parser -- -max_total_time=3600
```

### Run with a specific number of iterations

```bash
cargo +nightly fuzz run fuzz_smtlib_parser -- -runs=100000
```

### Run with multiple jobs

```bash
cargo +nightly fuzz run fuzz_smtlib_parser -- -jobs=4 -workers=4
```

### Use a specific seed corpus

```bash
cargo +nightly fuzz run fuzz_smtlib_parser corpus/fuzz_smtlib_parser
```

### Limit memory usage

```bash
cargo +nightly fuzz run fuzz_smtlib_parser -- -rss_limit_mb=2048
```

## Coverage Reporting

Generate coverage reports to see which code paths have been exercised:

```bash
# Run fuzzing with coverage instrumentation
cargo +nightly fuzz coverage fuzz_smtlib_parser

# Generate HTML report (requires llvm-cov)
cargo +nightly fuzz coverage fuzz_smtlib_parser --lcov > coverage.lcov

# Or use grcov for HTML reports
grcov coverage.lcov -t html -o coverage_report/
```

## Reproducing Crashes

When a crash is found, a file will be saved in `artifacts/fuzz_TARGET/`:

```bash
# Reproduce the crash
cargo +nightly fuzz run fuzz_smtlib_parser artifacts/fuzz_smtlib_parser/crash-XXXXX

# Minimize the crash input
cargo +nightly fuzz tmin fuzz_smtlib_parser artifacts/fuzz_smtlib_parser/crash-XXXXX
```

## Directory Structure

```
fuzz/
  Cargo.toml           # Fuzz package configuration
  README.md            # This file
  fuzz_targets/        # Fuzz target source files
    fuzz_smtlib_parser.rs
    fuzz_term_builder.rs
    fuzz_solver.rs
  corpus/              # Corpus of interesting inputs (gitignored)
  artifacts/           # Crash artifacts (gitignored)
```

## Tips

1. **Start with a seed corpus**: Provide valid SMT-LIB2 files to help the fuzzer explore meaningful paths:
   ```bash
   mkdir -p corpus/fuzz_smtlib_parser
   cp ../examples/*.smt2 corpus/fuzz_smtlib_parser/
   ```

2. **Run fuzzing overnight**: Fuzzing benefits from long run times. Consider running overnight or for several hours.

3. **Monitor with AFL-style stats**: Use `-print_final_stats=1` to see coverage statistics.

4. **Address sanitizers**: cargo-fuzz automatically enables AddressSanitizer. For other sanitizers:
   ```bash
   RUSTFLAGS="-Zsanitizer=memory" cargo +nightly fuzz run fuzz_smtlib_parser
   ```

5. **Debug crashes**: To get better stack traces:
   ```bash
   RUST_BACKTRACE=1 cargo +nightly fuzz run fuzz_smtlib_parser artifacts/crash-XXXXX
   ```
