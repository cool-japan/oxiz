# OxiZ

Next-Generation SMT Solver in Pure Rust

[![Crates.io](https://img.shields.io/crates/v/oxiz.svg)](https://crates.io/crates/oxiz)
[![Documentation](https://docs.rs/oxiz/badge.svg)](https://docs.rs/oxiz)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/oxiz/blob/main/LICENSE)

## About

**OxiZ** is a high-performance SMT (Satisfiability Modulo Theories) solver written entirely in Pure Rust,
designed to achieve feature parity with [Z3](https://github.com/Z3Prover/z3) while leveraging Rust's safety,
performance, and concurrency features.

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
oxiz = "0.1.1"  # Default includes solver
```

With specific features:

```toml
[dependencies]
oxiz = { version = "0.1.1", features = ["nlsat", "optimization"] }
```

All features:

```toml
[dependencies]
oxiz = { version = "0.1.1", features = ["full"] }
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
| `solver` | Core SMT solver (SAT + theories) | ✓ |
| `nlsat` | Nonlinear real arithmetic | |
| `optimization` | MaxSMT and optimization | |
| `spacer` | CHC solver for verification | |
| `proof` | Proof generation | |
| `standard` | Common features (solver + nlsat + opt + proof) | |
| `full` | All features | |

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

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/cool-japan/oxiz/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/cool-japan/oxiz/blob/main/LICENSE-MIT))

at your option.
