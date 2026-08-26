# Contributing to OxiZ

OxiZ is a next-generation **Satisfiability Modulo Theories (SMT) solver** written entirely in pure
Rust. It implements a modular CDCL(T) architecture that closely follows the design of Z3 while
leveraging Rust's safety guarantees and modern language features.

Issues and pull requests are welcome. The sections below describe how the code is organized,
styled, and tested — treat this as a technical reference for working in the codebase rather than a
process document.

**Quick Links:**
- [Documentation](docs/)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Issue Tracker](https://github.com/cool-japan/oxiz/issues)
- [Repository](https://github.com/cool-japan/oxiz)

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following tools installed:

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| **Rust** | 1.88+ (Edition 2024) | Compilation |
| **cargo-clippy** | Latest | Linting |
| **cargo-fmt** | Latest | Formatting |
| **cargo-nextest** | Recommended | Fast test runner |
| **wasm-pack** | For WASM | WebAssembly builds |

### Setting Up Your Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cool-japan/oxiz.git
   cd oxiz
   ```

2. **Verify your Rust installation:**
   ```bash
   rustc --version   # Should be 1.88 or higher (the declared `rust-version`)
   cargo --version
   ```

3. **Install development tools:**
   ```bash
   rustup component add clippy rustfmt
   cargo install cargo-nextest  # Recommended for faster testing
   ```

4. **Build the project:**
   ```bash
   cargo build
   ```

5. **Run the test suite:**
   ```bash
   # Using cargo test
   cargo test --all-features

   # Using nextest (recommended, faster)
   cargo nextest run --all-features
   ```

6. **Build in release mode:**
   ```bash
   cargo build --release
   ```

7. **Run the CLI:**
   ```bash
   cargo run --release -p oxiz-cli -- --help
   ```

### Building Specific Crates

```bash
# Build a specific crate
cargo build -p oxiz-core

# Build with all features
cargo build -p oxiz-solver --all-features

# Build WASM bindings
cd oxiz-wasm && wasm-pack build --target web
```

---

## Code Style

OxiZ follows strict code quality standards, enforced in CI on every PR.

### NO WARNINGS POLICY

**This is critical:** OxiZ enforces a strict NO WARNINGS policy. Code must compile without any
warnings from both the compiler and Clippy.

```bash
# Check for warnings (this must pass with no output)
cargo clippy --all-features --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Rust Style Guidelines

1. **Formatting:** Always run `cargo fmt` before committing:
   ```bash
   cargo fmt --all
   ```

2. **Linting:** All code must pass Clippy with warnings as errors:
   ```bash
   cargo clippy --all-features --all-targets -- -D warnings
   ```

3. **Documentation:** All public APIs must be documented:
   ```rust
   /// Computes the satisfiability of the given formula.
   ///
   /// # Arguments
   ///
   /// * `formula` - The SMT formula to check
   ///
   /// # Returns
   ///
   /// Returns `Sat` with a model if satisfiable, `Unsat` with a proof
   /// if unsatisfiable, or `Unknown` if the solver cannot determine.
   ///
   /// # Examples
   ///
   /// ```
   /// use oxiz_solver::Solver;
   ///
   /// let mut solver = Solver::new();
   /// solver.assert(formula);
   /// let result = solver.check_sat();
   /// ```
   pub fn check_sat(&mut self) -> SolverResult {
       // ...
   }
   ```

### Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Types | PascalCase | `TheorySolver`, `TermId` |
| Functions | snake_case | `check_sat`, `add_clause` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_CLAUSE_SIZE` |
| Modules | snake_case | `theory_solver`, `proof_gen` |
| Type parameters | Single uppercase or descriptive | `T`, `Term` |

### Module Organization

Follow this structure for new modules:

```rust
//! Module-level documentation explaining purpose.
//!
//! # Overview
//!
//! Brief description of what this module provides.

// Imports grouped by: std, external crates, internal crates, local modules
use std::collections::HashMap;

use indexmap::IndexMap;
use rayon::prelude::*;

use oxiz_core::ast::TermId;

use crate::internal_module;

// Public re-exports
pub use self::submodule::PublicType;

// Type definitions
type InternalAlias = Vec<TermId>;

// Constants
const INTERNAL_CONSTANT: usize = 42;

// Main implementations
pub struct MainType { /* ... */ }

impl MainType { /* ... */ }

// Trait implementations
impl SomeTrait for MainType { /* ... */ }

// Private helpers at the bottom
fn internal_helper() { /* ... */ }

// Tests in a submodule
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_functionality() { /* ... */ }
}
```

### Error Handling

- Use `Result<T, E>` for operations that can fail
- Define error types using `thiserror`
- Avoid `unwrap()` and `expect()` except in tests or truly impossible cases. `clippy::unwrap_used`
  is set to `deny` in the workspace lint table, and every member crate is covered — 13 declare it
  directly in their own `[lints.clippy]`, while `oxiz`, `oxiz-smtcomp`, `oxiz-py` and `oxiz-ml`
  inherit it via `[lints] workspace = true`. A stray `unwrap()` in production code therefore fails
  the build rather than merely warning.
  - When converting a native-recursive walk to an explicit heap stack (to
    remove a stack-overflow risk), drive the loop from
    `while let Some(frame) = stack.pop()` (own the frame, push back what
    still needs resuming) or carry the resume state inside the frame enum
    itself (e.g. `Expand(T)` / `Combine(T, n)`), so that "the stack emptied
    but I still expected a frame" is a case the code cannot even express.
    `expect("just matched via last_mut/last() above")` from a separate
    peek-then-pop pair is not a "truly impossible case" exemption -- it is
    exactly the shape this rule exists to rule out, because every future
    conversion done the same way reproduces it. See
    `oxiz-nlsat/src/solver/conflict.rs::is_redundant_literal`,
    `oxiz-theories/src/checking/proof.rs::validate_step`,
    `oxiz-theories/src/euf/incremental.rs::node_size`, or
    `oxiz-core/src/ast/manager/query/substitute.rs` for the pattern to copy.
- Document error conditions in function documentation

---

## Testing Requirements

OxiZ maintains high test coverage. New code should include tests appropriate to the change.

### Unit Tests

Every public API should have unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_basic_case() {
        let result = my_function(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_function_edge_case() {
        let result = my_function(edge_input);
        assert_eq!(result, edge_expected);
    }

    #[test]
    #[should_panic(expected = "specific error message")]
    fn test_function_invalid_input() {
        my_function(invalid_input);
    }
}
```

### Integration Tests

For features that span multiple modules:

```rust
// In tests/integration_test.rs
use oxiz_solver::Solver;
use oxiz_core::ast::*;

#[test]
fn test_solver_with_lra_theory() {
    let mut solver = Solver::new();
    // Set up and test complete solving pipeline
}
```

### Doc Tests

All code examples in documentation must be runnable:

```rust
/// Checks satisfiability of the current assertions.
///
/// # Examples
///
/// ```
/// use oxiz_solver::Solver;
///
/// let mut solver = Solver::new();
/// let x = solver.declare_const("x", Sort::Int);
/// solver.assert(solver.mk_gt(x, solver.mk_int(0)));
/// assert!(solver.check_sat().is_sat());
/// ```
pub fn check_sat(&mut self) -> SolverResult {
    // ...
}
```

### Test Naming Conventions

```rust
#[test]
fn test_<module>_<function>_<scenario>() {
    // test_solver_check_sat_unsat_formula
    // test_cdcl_propagate_unit_clause
    // test_simplex_feasibility_with_strict_inequality
}
```

### Running Tests

```bash
# Run all tests
cargo nextest run --all-features

# Run tests for a specific crate
cargo nextest run -p oxiz-sat

# Run tests matching a pattern
cargo nextest run test_cdcl

# Run with output visible
cargo nextest run -- --nocapture
```

### Coverage Expectations

- New features: aim for >80% coverage of new code
- Bug fixes: include a test that reproduces the bug
- Critical paths (solving, proof generation): aim for >90% coverage

### Running the full test suite within a memory cap

On 2026-07-31 a single test process reached 32 GB of virtual memory during a full
`--workspace --all-features` run, which motivated capping both memory and per-test wall time.

**Linux:** run the suite inside a transient cgroup scope so a runaway process is killed by the
kernel instead of pushing the whole machine into swap:

```bash
systemd-run --user --scope --collect -p MemoryMax=10G -q -- \
  cargo nextest run --workspace --all-features
```

**macOS:** there is no cgroup equivalent, so cap memory pressure indirectly by limiting
parallelism instead. `.config/nextest.toml`'s `slow-timeout` (`period = "60s"`,
`terminate-after = 3` on `default`, `2` on `ci`) now also kills any individual test that runs past
3×60s (2×60s in CI), so a stuck/runaway test no longer runs unbounded until something else
intervenes.

On a 16-thread / 14.6 GB machine, the parallelism caps below were needed to stay within memory:

```bash
cargo build -j 6
cargo nextest run --workspace --all-features --test-threads 8
```

---

## Pull Requests

Fork the repository, create a feature branch, and open a PR against `master`. Keep changes focused
and include tests for new behavior. All of the following must pass:

```bash
cargo build --all-features
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --no-deps
```

PRs are merged via squash-and-merge.

---

## Architecture Overview

OxiZ is organized as a Cargo workspace with 17 members: 15 crates plus the two benchmark
harnesses under `bench/`. Everything except `oxiz-py` (which needs maturin for Python linking)
is in `default-members`, so a plain `cargo build` covers the whole tree.

### Crate Hierarchy

```
oxiz (meta-crate: unified API)
  |
  +-- oxiz-cli (Command-line interface)
  +-- oxiz-wasm (WebAssembly bindings)
  +-- oxiz-py (Python bindings via PyO3/maturin)
  +-- oxiz-smtcomp (SMT-COMP entry package and runners)
  +-- oxiz-ml (ML-guided heuristics)
  +-- oxiz-opt (MaxSAT/OMT optimization)
  |
  +-- oxiz-solver (CDCL(T) orchestration)
        |
        +-- oxiz-spacer (PDR/CHC solving)
        +-- oxiz-theories (Theory solvers: EUF, LRA, BV, etc.)
        +-- oxiz-proof (Proof generation: DRAT, Alethe, LFSC)
              |
              +-- oxiz-sat (CDCL SAT solver)
              +-- oxiz-nlsat (Non-linear arithmetic)
                    |
                    +-- oxiz-math (Mathematical foundations)
                          |
                          +-- oxiz-core (AST, sorts, parser, tactics)
```

### Key Abstractions

| Abstraction | Location | Purpose |
|-------------|----------|---------|
| `TermId` | oxiz-core | Hash-consed term references |
| `Solver` | oxiz-solver | Main SMT solver interface |
| `SatSolver` | oxiz-sat | CDCL SAT solving core |
| `TheorySolver` | oxiz-theories | Theory solver trait |
| `Proof` | oxiz-proof | Proof DAG representation |

### Adding New Components

- **New Theory:** Implement `TheorySolver` trait in `oxiz-theories`
- **New Tactic:** Implement `Tactic` trait in `oxiz-core`
- **New Proof Format:** Implement `ProofFormatter` trait in `oxiz-proof`

For detailed architecture information, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## License

Licensed under Apache License 2.0 ([LICENSE](LICENSE)).
