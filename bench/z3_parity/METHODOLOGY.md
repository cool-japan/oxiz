# Z3 Parity Test Methodology

## Overview

This test suite validates OxiZ's correctness by comparing its results against Z3, the de facto standard SMT solver. The goal is to ensure OxiZ produces the same satisfiability decisions as Z3 on a representative set of benchmarks.

## Test Structure

### Benchmark Categories

- **QF_LIA** (16 benchmarks): Quantifier-Free Linear Integer Arithmetic
- **QF_LRA** (16 benchmarks): Quantifier-Free Linear Real Arithmetic
- **QF_BV** (15 benchmarks): Quantifier-Free Bit-Vectors
- **QF_S** (10 benchmarks): Quantifier-Free Strings
- **QF_FP** (10 benchmarks): Quantifier-Free Floating Point
- **QF_DT** (10 benchmarks): Quantifier-Free Datatypes
- **QF_A** (10 benchmarks): Quantifier-Free Arrays

**Total:** 87 benchmarks

### Benchmark Selection Criteria

Each benchmark was selected to:

1. **Cover diverse constraint patterns** - Not just simple satisfiability checks
2. **Mix of SAT/UNSAT/UNKNOWN** - Approximately 40% SAT, 40% UNSAT, 20% UNKNOWN
3. **Varied complexity** - Easy, medium, and hard instances
4. **Execute within timeout** - Both solvers should complete within 60 seconds
5. **Test edge cases** - Boundary conditions, special values, tricky patterns

### Benchmark Sources

Benchmarks are derived from:

- SMT-LIB benchmark repository
- Z3 test suite
- Manually crafted instances targeting specific features
- Real-world verification problems (simplified)

## Running Tests

### Prerequisites

1. **Install Z3**:
   ```bash
   # macOS
   brew install z3

   # Linux
   sudo apt-get install z3

   # Or download from: https://github.com/Z3Prover/z3/releases
   ```

2. **Build OxiZ**:
   ```bash
   cd /Users/kitasan/work/oxiz
   cargo build --release
   ```

### Execute Parity Tests

```bash
cd bench/z3_parity
cargo run --release
```

This will:
1. Discover all `.smt2` files in `benchmarks/`
2. Run each benchmark on both Z3 and OxiZ (in parallel)
3. Compare results
4. Generate a summary report
5. Save detailed results to `results.json`

### Interpreting Results

#### Match Status

- **Correct**: Both solvers agree (SAT/SAT, UNSAT/UNSAT, UNKNOWN/UNKNOWN)
- **Wrong**: Disagreement on SAT/UNSAT (critical bug!)
- **Timeout**: One or both solvers exceeded 60s timeout
- **Error**: Parse error or execution failure

**Note:** If one solver returns UNKNOWN and the other returns SAT or UNSAT, this is considered **Correct** (UNKNOWN is a valid response).

#### Pass Criteria

For **v0.1.3 release**, the following accuracy is required:

- **QF_LIA**: 100% correct (0 wrong)
- **QF_LRA**: 100% correct (0 wrong)
- **QF_BV**: 100% correct (0 wrong)
- **QF_S**: ≥ 80% correct (exploratory)
- **QF_FP**: ≥ 80% correct (exploratory)
- **QF_DT**: ≥ 70% correct (work in progress)
- **QF_A**: ≥ 90% correct (mature theory)

**Overall**: ≥ 95% correct across all logics

## Limitations

### What This Tests

✅ **Correctness of satisfiability decisions**
✅ **Logic-specific feature coverage**
✅ **Regression prevention**
✅ **Relative performance (execution time)**

### What This Does NOT Test

❌ **Model quality** - We don't validate model values, only SAT/UNSAT
❌ **Proof generation** - OxiZ doesn't generate proofs yet
❌ **Incremental solving** - Benchmarks are one-shot
❌ **Quantifiers** - Limited to quantifier-free logics
❌ **Performance limits** - All benchmarks finish in < 60s

## Adding New Benchmarks

To add a new benchmark:

1. Create a `.smt2` file in the appropriate logic directory
2. Include a comment at the top documenting:
   - What feature/pattern it tests
   - Expected result (sat/unsat/unknown)
   - Source (if adapted from elsewhere)

Example:
```smt2
; Test: Branch and bound with negative coefficients
; Expected: unsat
; Source: Adapted from SMT-LIB QF_LIA/20230215-Barrett

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (+ (* -2 x) (* 3 y)) 7))
(assert (>= x 0))
(assert (<= x 5))
(assert (>= y 0))
(assert (<= y 2))

(check-sat)
```

3. Run the parity suite to validate
4. Commit both the benchmark and updated `results.json`

## Troubleshooting

### "Z3 not found"

Ensure Z3 is installed and in your PATH:
```bash
which z3  # Should print /usr/local/bin/z3 or similar
z3 --version  # Should print version info
```

### All OxiZ tests showing "Error"

Check that OxiZ builds successfully:
```bash
cd ../../oxiz
cargo test
```

### Timeout issues

If benchmarks are timing out, increase the timeout in `z3_runner.rs` and `oxiz_runner.rs`:
```rust
const Z3_TIMEOUT_SECS: u64 = 120;  // Increase from 60
```

## Maintenance

This test suite should be run:

- **Before every release** - Blocking requirement
- **Weekly (CI)** - Automated regression testing
- **After major changes** - Especially to core theories
- **When adding new features** - Ensure no regressions

## Future Enhancements

- [ ] Model validation (check SAT models satisfy constraints)
- [ ] UNSAT core comparison
- [ ] Incremental solving tests (push/pop)
- [ ] Performance benchmarking (not just correctness)
- [ ] Fuzz-generated benchmarks
- [ ] Quantified formula tests (once implemented)
