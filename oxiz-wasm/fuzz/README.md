# Fuzz Testing for oxiz-wasm

This directory contains fuzz tests for the oxiz-wasm crate using cargo-fuzz.

## Prerequisites

Install cargo-fuzz if you haven't already:

```bash
cargo install cargo-fuzz
```

## Running Fuzz Tests

To run a specific fuzz target:

```bash
# Fuzz the execute() method
cargo fuzz run fuzz_execute

# Fuzz the assertFormula() method
cargo fuzz run fuzz_assert_formula

# Fuzz the declareConst() method
cargo fuzz run fuzz_declare_const

# Fuzz the simplify() method
cargo fuzz run fuzz_simplify
```

## Fuzz Targets

### fuzz_execute

Tests the `execute()` method with arbitrary SMT-LIB2 scripts. This helps find:
- Parser crashes
- Unexpected panics in script execution
- Memory safety issues

### fuzz_assert_formula

Tests the `assertFormula()` method with arbitrary formulas. This helps find:
- Formula parsing issues
- Type checking crashes
- Assertion handling bugs

### fuzz_declare_const

Tests the `declareConst()` method with arbitrary names and sorts. This helps find:
- Sort validation issues
- Name handling bugs
- Symbol table corruption

### fuzz_simplify

Tests the `simplify()` method with arbitrary expressions. This helps find:
- Simplification crashes
- Expression parsing issues
- Rewrite rule bugs

## Continuous Fuzzing

For continuous fuzzing, you can run all targets in parallel:

```bash
#!/bin/bash
for target in fuzz_execute fuzz_assert_formula fuzz_declare_const fuzz_simplify; do
    cargo fuzz run "$target" -- -max_total_time=3600 &
done
wait
```

## Analyzing Crashes

If a crash is found, it will be saved in `fuzz/artifacts/<target>/<crash-file>`.

To reproduce a crash:

```bash
cargo fuzz run <target> fuzz/artifacts/<target>/<crash-file>
```

To minimize a crash:

```bash
cargo fuzz cmin <target> fuzz/corpus/<target>
```

## Coverage

To generate coverage information:

```bash
cargo fuzz coverage <target>
```

## Tips

- Start with short runs (a few minutes) to catch obvious issues
- For thorough testing, run each target for at least 24 hours
- Monitor memory usage - fuzzing can be resource-intensive
- Use sanitizers for better bug detection (enabled by default in cargo-fuzz)
