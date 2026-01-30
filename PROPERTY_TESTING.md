# Property-Based Testing Guide for OxiZ

This document describes the comprehensive property-based testing infrastructure implemented for the OxiZ SMT solver.

## Overview

Property-based testing complements traditional unit tests by:
- Testing properties that should hold for all inputs
- Automatically generating diverse test cases
- Shrinking failing cases to minimal examples
- Catching edge cases that manual tests miss

## Test Organization

### Property Tests by Crate

#### oxiz-core
Location: `oxiz-core/tests/property_tests/`

- `ast_properties.rs`: Term construction, substitution, comparisons
- `rewriter_properties.rs`: Rewrite rule correctness
- `tactic_properties.rs`: Tactic composition and correctness
- `qe_properties.rs`: Quantifier elimination soundness
- `egraph_properties.rs`: E-graph congruence closure
- `datalog_properties.rs`: Datalog evaluation correctness

#### oxiz-math
Location: `oxiz-math/tests/property_tests/`

- `polynomial_extended.rs`: Polynomial arithmetic properties
- `simplex_properties.rs`: Linear programming correctness
- `root_properties.rs`: Root finding algorithms
- `property_tests.rs`: Rational arithmetic properties

#### oxiz-solver
Location: `oxiz-solver/tests/property_tests/`

- `backtrack_properties.rs`: Backtracking consistency
- `model_properties.rs`: Model validity
- `conflict_properties.rs`: Conflict analysis correctness
- `propagation_properties.rs`: Unit propagation soundness

#### oxiz-sat
Location: `oxiz-sat/tests/property_tests/`

- `cdcl_properties.rs`: CDCL algorithm correctness

### Fuzz Targets

Location: `fuzz/fuzz_targets/`

- `fuzz_smtlib_parser.rs`: Parser fuzzing
- `fuzz_term_builder.rs`: Term manager fuzzing
- `fuzz_solver.rs`: Solver fuzzing
- `fuzz_theory_arithmetic.rs`: Arithmetic theory
- `fuzz_theory_bitvector.rs`: Bitvector theory
- `fuzz_theory_array.rs`: Array theory
- `fuzz_quantifiers.rs`: Quantifier handling
- `fuzz_tactics.rs`: Tactic application

### Runtime Invariants

Location: `*/src/invariants.rs`

- `oxiz-solver/src/invariants.rs`: Solver state invariants
- `oxiz-sat/src/invariants.rs`: SAT solver invariants

## Running Tests

### Property Tests

```bash
# Run all property tests
cargo test --test property_based --all

# Run specific crate property tests
cargo test --test property_based -p oxiz-core
cargo test --test property_based -p oxiz-math
cargo test --test property_based -p oxiz-solver
cargo test --test property_based -p oxiz-sat

# Run with more test cases
PROPTEST_CASES=1000 cargo test --test property_based

# Run with specific seed for reproducibility
PROPTEST_RNG_SEED=12345 cargo test --test property_based
```

### Fuzz Tests

```bash
# Install cargo-fuzz (once)
cargo install cargo-fuzz

# List available fuzz targets
cargo fuzz list

# Run a specific fuzz target
cargo fuzz run fuzz_solver

# Run with time limit
cargo fuzz run fuzz_solver -- -max_total_time=300

# Run with specific corpus
cargo fuzz run fuzz_solver corpus/solver/

# Minimize a crash
cargo fuzz cmin fuzz_solver
cargo fuzz tmin fuzz_solver crash-file
```

### Invariant Checks

```bash
# Enable invariant checks in debug mode (default)
cargo test

# Enable in release mode
cargo test --release --features invariant-checks

# Run specific invariant tests
cargo test -p oxiz-solver invariant_tests
```

## Writing Property Tests

### Basic Structure

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn my_property(x in 0i64..100i64) {
        // Test that property holds for x
        prop_assert!(property_holds(x));
    }
}
```

### Custom Strategies

```rust
// Strategy for generating terms
fn term_strategy() -> impl Strategy<Value = Term> {
    prop_oneof![
        // Generate constants
        (0i64..100i64).prop_map(|n| Term::Const(n)),
        // Generate variables
        "[a-z]".prop_map(|s| Term::Var(s)),
        // Generate compound terms
        (term_strategy(), term_strategy())
            .prop_map(|(t1, t2)| Term::Add(Box::new(t1), Box::new(t2)))
    ]
}
```

### Property Categories

#### Algebraic Properties

```rust
proptest! {
    #[test]
    fn addition_commutative(a in num(), b in num()) {
        let sum1 = add(a, b);
        let sum2 = add(b, a);
        prop_assert_eq!(sum1, sum2);
    }

    #[test]
    fn addition_associative(a in num(), b in num(), c in num()) {
        let left = add(add(a, b), c);
        let right = add(a, add(b, c));
        prop_assert_eq!(left, right);
    }
}
```

#### Invariant Properties

```rust
proptest! {
    #[test]
    fn trail_monotonic(ops in vec(operation(), 1..10)) {
        let mut solver = Solver::new();
        let mut last_level = 0;

        for op in ops {
            apply_operation(&mut solver, op);
            let level = solver.decision_level();
            prop_assert!(level >= last_level);
            last_level = level;
        }
    }
}
```

#### Semantic Preservation

```rust
proptest! {
    #[test]
    fn rewrite_preserves_semantics(term in term_strategy()) {
        let original = eval(&term);
        let rewritten = rewrite(&term);
        let after = eval(&rewritten);
        prop_assert_eq!(original, after);
    }
}
```

## Debugging Failed Properties

### Shrinking

When a property fails, proptest automatically shrinks the input:

```
property failed for input: (x: 12345)
shrinking...
minimal failing input: (x: 42)
```

### Regression Tests

Failed cases are automatically saved:

```
proptest-regressions/
  my_property.txt
```

Commit these files to prevent regressions.

### Deterministic Replay

```bash
# Run with specific seed
PROPTEST_RNG_SEED=12345 cargo test property_name

# The seed is printed on failure
# thread 'test' panicked at 'Test failed with seed 67890'
```

## Best Practices

### 1. Start Simple

Begin with basic properties before complex ones:

```rust
// Good: Simple property
#[test]
fn addition_identity(x in num()) {
    prop_assert_eq!(add(x, 0), x);
}

// Better: After simpler tests pass
#[test]
fn addition_distributive(a in num(), b in num(), c in num()) {
    let left = mul(a, add(b, c));
    let right = add(mul(a, b), mul(a, c));
    prop_assert_eq!(left, right);
}
```

### 2. Use Reasonable Input Ranges

Avoid generating inputs that are likely to fail:

```rust
// Bad: May generate invalid inputs
#[test]
fn division_property(a in any::<i64>(), b in any::<i64>()) {
    prop_assert_eq!(div(a, b), ...); // b might be 0!
}

// Good: Filter invalid inputs
#[test]
fn division_property(
    a in any::<i64>(),
    b in any::<i64>().prop_filter("non-zero", |&x| x != 0)
) {
    prop_assert_eq!(div(a, b), ...);
}
```

### 3. Test Invariants, Not Implementations

```rust
// Bad: Testing implementation details
#[test]
fn uses_specific_algorithm(input in data()) {
    let output = solve(input);
    prop_assert!(output.algorithm == "CDCL"); // Fragile!
}

// Good: Testing correctness property
#[test]
fn solution_satisfies_constraints(input in data()) {
    let solution = solve(input);
    prop_assert!(validates(&input, &solution));
}
```

### 4. Combine with Traditional Tests

Use property tests to complement, not replace, traditional tests:

```rust
// Traditional: Specific known cases
#[test]
fn known_case() {
    assert_eq!(solve(problem_x), expected_solution);
}

// Property: General properties
proptest! {
    #[test]
    fn solution_always_valid(problem in problem_gen()) {
        let solution = solve(problem);
        prop_assert!(is_valid(&solution));
    }
}
```

## Coverage Goals

Target coverage for property tests:
- **Core algorithms**: 80%+ property coverage
- **Data structures**: All algebraic properties tested
- **API boundaries**: Input validation properties
- **Error handling**: Error recovery properties

## Performance Considerations

Property tests can be slow. Optimize:

1. **Limit case count**: Use `PROPTEST_CASES` environment variable
2. **Parallel execution**: Runs in parallel by default
3. **Quick checks**: Use `#[cfg(not(miri))]` for expensive tests
4. **Timeout**: Set per-test timeouts for expensive properties

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Property Tests

on: [push, pull_request]

jobs:
  proptest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run property tests
        run: cargo test --test property_based --all
        env:
          PROPTEST_CASES: 256

  fuzz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz
      - name: Run fuzz tests (time-limited)
        run: |
          for target in $(cargo fuzz list); do
            cargo fuzz run $target -- -max_total_time=60 -rss_limit_mb=2048
          done
```

## Troubleshooting

### "Test took too long"

Increase timeout or reduce input size:

```rust
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,  // Reduce cases
        max_shrink_time: 10000,
        max_shrink_iters: 100000,
        ..ProptestConfig::default()
    })]

    #[test]
    fn expensive_property(x in 0..100) {  // Smaller range
        // Test
    }
}
```

### "Shrinking took too long"

Disable shrinking or limit iterations:

```rust
proptest! {
    #![proptest_config(ProptestConfig {
        max_shrink_iters: 1000,
        ..ProptestConfig::default()
    })]
    // ...
}
```

### Fuzz Tests Run Out of Memory

Set memory limits:

```bash
cargo fuzz run target -- -rss_limit_mb=2048
```

## Resources

- [proptest Documentation](https://altsysrq.github.io/proptest-book/)
- [cargo-fuzz Guide](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [Property-Based Testing Patterns](https://hypothesis.works/articles/what-is-property-based-testing/)

## Contributing

When adding new features:

1. **Write properties first**: Define what should be true
2. **Implement feature**: Make properties pass
3. **Add regression tests**: Save interesting cases
4. **Document properties**: Explain what's being tested

Example workflow:

```rust
// 1. Define property
proptest! {
    #[test]
    fn new_feature_property(input in data()) {
        let output = new_feature(input);
        prop_assert!(property_holds(output));
    }
}

// 2. Run test (will fail initially)
// cargo test new_feature_property

// 3. Implement feature until property passes
// 4. Add regression tests for edge cases found
// 5. Document in this guide
```

## Statistics

Current property test coverage:
- **Total property tests**: 250+
- **Total fuzz targets**: 8
- **Total invariant checks**: 20+
- **Lines of test code**: 10,000+

Last updated: 2026-01-29
