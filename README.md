# OxiZ

Next-Generation SMT Solver in Pure Rust

[![Crates.io](https://img.shields.io/crates/v/oxiz-solver.svg)](https://crates.io/crates/oxiz-solver)
[![Documentation](https://docs.rs/oxiz-solver/badge.svg)](https://docs.rs/oxiz-solver)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## About This Project

OxiZ is a high-performance Satisfiability Modulo Theories (SMT) solver written entirely in Rust. This project reimplements [Z3](https://github.com/Z3Prover/z3) in Pure Rust, achieving **~90%+ feature parity** with only **~25% of the codebase size**.

**Pure Rust is a fundamental requirement** - no C/C++ dependencies, no FFI bindings, just clean, safe Rust code.

## Features

- **Pure Rust** - No C/C++ dependencies, memory-safe by design
- **CDCL(T) Architecture** - Conflict-Driven Clause Learning with theory integration
- **Comprehensive Theory Support** - EUF, LRA, LIA, BV, Arrays, Strings, FP, Datatypes
- **Advanced Quantifier Handling** - MBQI, E-matching, Skolemization, DER
- **SMT-LIB2 Support** - Full standard input/output format
- **WebAssembly Ready** - Run in browsers via WASM bindings
- **Incremental Solving** - Push/pop for efficient constraint management
- **Proof Generation** - DRAT, Alethe, LFSC, Coq/Lean/Isabelle export
- **Optimization** - MaxSAT, OMT with Pareto optimization
- **Model Checking** - CHC solving with PDR/IC3

## Project Statistics

| Metric | Value |
|--------|-------|
| Rust Lines of Code | ~173,500 |
| Test Count | 3,670 |
| Crates | 11 |
| Z3 Feature Parity | ~90-92% |

## Workspace Structure

```
oxiz/
├── oxiz-core/      # Core AST, sorts, SMT-LIB parser, tactics, rewriters
├── oxiz-math/      # Mathematical algorithms (polynomials, matrices, LP)
├── oxiz-sat/       # CDCL SAT solver with VSIDS/LRB/VMTF
├── oxiz-nlsat/     # Nonlinear arithmetic (CAD, algebraic numbers)
├── oxiz-theories/  # Theory solvers (EUF, Arith, BV, Arrays, Strings, FP, ADT)
├── oxiz-solver/    # Main CDCL(T) orchestration, MBQI
├── oxiz-opt/       # Optimization (MaxSAT, OMT)
├── oxiz-spacer/    # CHC solving, PDR/IC3, BMC
├── oxiz-proof/     # Proof generation and verification
├── oxiz-wasm/      # WebAssembly bindings
└── oxiz-cli/       # Command-line interface
```

## Quick Start

### Installation

```bash
# Add to your Cargo.toml
[dependencies]
oxiz-solver = "0.1"
```

### Building from Source

```bash
git clone https://github.com/cool-japan/oxiz
cd oxiz
cargo build --release
```

### Running Tests

```bash
cargo nextest run --all-features
```

### Using the CLI

```bash
# Solve an SMT-LIB2 file
cargo run --release -p oxiz-cli -- input.smt2

# Interactive mode
cargo run --release -p oxiz-cli -- --interactive

# With verbose output
cargo run --release -p oxiz-cli -- -v input.smt2
```

### Library Usage

```rust
use oxiz_solver::Context;

fn main() {
    let mut ctx = Context::new();

    let results = ctx.execute_script(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const y Int)
        (assert (> x 0))
        (assert (< y 10))
        (assert (= (+ x y) 15))
        (check-sat)
        (get-model)
    "#).unwrap();

    for result in results {
        println!("{}", result);
    }
}
```

## Supported Logics

| Logic | Description | Status |
|-------|-------------|--------|
| QF_UF | Uninterpreted Functions | ✅ Complete |
| QF_LRA | Linear Real Arithmetic | ✅ Complete |
| QF_LIA | Linear Integer Arithmetic | ✅ Complete |
| QF_BV | Fixed-size BitVectors | ✅ Complete |
| QF_S | Strings | ✅ Complete |
| QF_FP | Floating Point | ✅ Complete |
| QF_DT | Datatypes (ADT) | ✅ Complete |
| QF_A | Arrays | ✅ Complete |
| QF_NRA | Nonlinear Real Arithmetic | ✅ Complete |
| QF_NIA | Nonlinear Integer Arithmetic | ✅ Partial |
| UFLIA | UF + LIA with Quantifiers | ✅ Complete |
| AUFBV | Arrays + UF + BV | ✅ Complete |
| HORN | Constrained Horn Clauses | ✅ Complete |

## Key Components

### SAT Solver
- CDCL with two-watched literals
- Multiple branching heuristics (VSIDS, LRB, VMTF, CHB)
- Clause learning with minimization
- Preprocessing (BCE, BVE, subsumption)
- DRAT proof generation
- Local search and lookahead
- AllSAT enumeration

### Theory Solvers
- EUF with congruence closure
- LRA with Simplex
- LIA with branch-and-bound, Cuts
- BV with bit-blasting and word-level reasoning
- Arrays with extensionality
- Strings with automata
- Floating-point with bit-precise semantics
- Datatypes (ADT) with testers/selectors
- Pseudo-Boolean constraints
- Special relations (partial/total orders)

### Quantifier Handling
- E-matching with triggers
- MBQI (Model-Based Quantifier Instantiation)
- Skolemization
- DER (Destructive Equality Resolution)
- Model-Based Projection

### Optimization
- MaxSAT (Fu-Malik, RC2, LNS)
- OMT with lexicographic/Pareto optimization
- Weighted soft constraints

### Model Checking
- PDR/IC3 for CHC solving
- BMC (Bounded Model Checking)
- Lemma generalization
- Craig interpolation

## Architecture

OxiZ follows a layered CDCL(T) architecture:

1. **SAT Core** (`oxiz-sat`) - CDCL solver with modern heuristics
2. **Theory Solvers** (`oxiz-theories`) - Modular theory implementations
3. **SMT Orchestration** (`oxiz-solver`) - Theory combination and DPLL(T)
4. **Tactics** (`oxiz-core`) - Preprocessing and simplification
5. **Proof Layer** (`oxiz-proof`) - Proof generation and verification

## Requirements

- Rust 1.85+ (Edition 2024)
- No external C/C++ dependencies

## WebAssembly

OxiZ can be compiled to WebAssembly for browser use:

```bash
cd oxiz-wasm
wasm-pack build --target web
```

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

MIT OR Apache-2.0

## Authors

COOLJAPAN OU (Team KitaSan)

## Acknowledgments

This project is inspired by and references the algorithms in:
- Z3 (Microsoft Research)
- CVC5 (Stanford/Iowa)
- MiniSat/Glucose
- Various academic papers on SMT solving
