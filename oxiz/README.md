# OxiZ

Next-Generation SMT Solver in Pure Rust

[![Crates.io](https://img.shields.io/crates/v/oxiz.svg)](https://crates.io/crates/oxiz)
[![Documentation](https://docs.rs/oxiz/badge.svg)](https://docs.rs/oxiz)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/cool-japan/oxiz/blob/main/LICENSE)

> **Version**: 0.3.2 | **Status**: Meta-crate | **Tests**: 25

## About

**OxiZ** is a high-performance SMT (Satisfiability Modulo Theories) solver written entirely in Pure Rust,
designed to achieve feature parity with [Z3](https://github.com/Z3Prover/z3) while leveraging Rust's safety,
performance, and concurrency features.

OxiZ tracks honest, non-fabricated parity against a real Z3 binary via `bench/z3_parity`: **170/170 Correct, 0 Wrong, 0 Inconclusive, 0 Timeout, 0 Error** on the extended 19-logic / 170-benchmark differential suite against a real z3 4.15.4 binary under the honest comparator (Unknown never counts as a match). All 19 logic families are at 100% of the differential parity suite. See the [README](../README.md#z3-parity-differential-suite-results-honest-comparator-️) and [CHANGELOG](../CHANGELOG.md) for current status and known limitations before relying on it for production workloads.

## Features

- **Pure Rust Implementation**: No C/C++ dependencies, complete memory safety
- **Z3-Compatible**: Extensive theory support and familiar API patterns
- **High Performance**: Optimized SAT core with advanced heuristics
- **Modular Design**: Use only what you need via feature flags
- **SMT-LIB2 Support**: Full parser and printer for standard format
- **WebAssembly Ready**: Run in browsers via WASM bindings

## Quick Start

### Installation

Add OxiZ to your `Cargo.toml`:

```toml
[dependencies]
oxiz = "0.3.3"  # Default: std + core solver
```

With specific features:

```toml
[dependencies]
oxiz = { version = "0.3.3", features = ["nlsat", "optimization"] }
```

All features:

```toml
[dependencies]
oxiz = { version = "0.3.3", features = ["full"] }
```

### Basic Usage

```rust
use oxiz::{Solver, TermManager, SolverResult};
use num_bigint::BigInt;

let mut solver = Solver::new();
let mut tm = TermManager::new();

// Create variables
let x = tm.mk_var("x", tm.sorts.int_sort);
let y = tm.mk_var("y", tm.sorts.int_sort);

// x + y = 10
let sum = tm.mk_add([x, y]);
let ten = tm.mk_int(BigInt::from(10));
let eq = tm.mk_eq(sum, ten);

// x > 5
let five = tm.mk_int(BigInt::from(5));
let gt = tm.mk_gt(x, five);

// Assert and solve
solver.assert(eq, &mut tm);
solver.assert(gt, &mut tm);

match solver.check(&mut tm) {
    SolverResult::Sat => println!("SAT"),
    SolverResult::Unsat => println!("UNSAT"),
    SolverResult::Unknown => println!("Unknown"),
}
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library support (core solver requires this for SMT-LIB2 parsing) | ✓ |
| `nlsat` | Nonlinear real arithmetic (implies `std`) | |
| `optimization` | MaxSMT and optimization (implies `std`) | |
| `spacer` | CHC solver for verification (implies `std`) | |
| `proof` | Proof generation (implies `std`) | |
| `standard` | Common features (`nlsat` + `optimization` + `proof`) | |
| `full` | All features (`standard` + `spacer`) | |

Note: the core SAT/theory solver (`Solver`, `TermManager`) is always available — it is not gated behind a `solver` feature.

## Theory Support

- **EUF**: Equality and uninterpreted functions
- **LRA**: Linear real arithmetic
- **LIA**: Linear integer arithmetic
- **NRA**: Nonlinear real arithmetic (with `nlsat` feature)
- **Arrays**: Theory of arrays
- **BitVectors**: Bit-precise reasoning
- **Strings**: String operations with regex
- **Datatypes**: Algebraic data types
- **Floating-Point**: IEEE 754 semantics

## Documentation

- [API Documentation](https://docs.rs/oxiz)
- [GitHub Repository](https://github.com/cool-japan/oxiz)
- [Examples](https://github.com/cool-japan/oxiz/tree/main/examples)

## License

Licensed under Apache License 2.0 ([LICENSE](https://github.com/cool-japan/oxiz/blob/main/LICENSE)).
