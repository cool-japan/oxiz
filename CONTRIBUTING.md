# Contributing to OxiZ

Welcome to OxiZ, and thank you for your interest in contributing!

OxiZ is a next-generation **Satisfiability Modulo Theories (SMT) solver** written entirely in pure Rust. We implement a modular CDCL(T) architecture that closely follows the design of Z3 while leveraging Rust's safety guarantees and modern features.

**Project Statistics:**
- ~173,500 lines of Rust code
- 3,670+ tests
- 12 workspace crates
- ~90-95% Z3 feature parity

We welcome contributions of all kinds: bug fixes, new features, documentation improvements, test additions, and performance optimizations. Every contribution helps make OxiZ better for the entire formal verification community.

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
| **Rust** | 1.75+ (Edition 2024) | Compilation |
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
   rustc --version   # Should be 1.75 or higher
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

OxiZ follows strict code quality standards. All contributions must adhere to these guidelines.

### NO WARNINGS POLICY

**This is critical:** OxiZ enforces a strict NO WARNINGS policy. Your code must compile without any warnings from both the compiler and Clippy.

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
- Avoid `unwrap()` and `expect()` except in tests or truly impossible cases
- Document error conditions in function documentation

---

## Pull Request Process

### Fork and Branch Workflow

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/oxiz.git
   cd oxiz
   git remote add upstream https://github.com/cool-japan/oxiz.git
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # Or for bug fixes:
   git checkout -b fix/issue-123-description
   ```

4. **Keep your branch updated:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Commit Message Format

We follow a structured commit message format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without feature changes
- `perf`: Performance improvements
- `test`: Adding or modifying tests
- `chore`: Build process or auxiliary tool changes

**Examples:**

```
feat(oxiz-theories): add support for floating-point theory

Implement IEEE 754 floating-point theory solver with:
- Bit-blasting approach for precise semantics
- Support for all rounding modes
- Integration with existing CDCL(T) engine

Closes #42
```

```
fix(oxiz-sat): correct VSIDS decay computation

The activity decay was being applied incorrectly during
conflict analysis, causing suboptimal branching decisions.

Fixes #123
```

### Pull Request Description Template

When opening a PR, use this template:

```markdown
## Summary

Brief description of what this PR does and why.

## Changes

- List of specific changes made
- One item per line
- Be specific and clear

## Testing

Describe how you tested these changes:
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Doc tests added/updated
- [ ] Manual testing performed

## Related Issues

Fixes #123
Related to #456

## Checklist

- [ ] Code compiles without warnings (`cargo clippy -- -D warnings`)
- [ ] Code is formatted (`cargo fmt --all`)
- [ ] All tests pass (`cargo nextest run --all-features`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow the format guidelines
```

### Review Process

1. **Automated Checks:** All PRs must pass CI:
   - Compilation on Linux/macOS/Windows
   - All tests pass
   - No Clippy warnings
   - Code is formatted

2. **Code Review:** At least one maintainer must approve the PR

3. **Review Criteria:**
   - Code correctness and quality
   - Test coverage
   - Documentation completeness
   - Performance considerations
   - Consistency with existing codebase

4. **Addressing Feedback:**
   - Respond to all review comments
   - Make requested changes in new commits
   - Mark conversations as resolved when addressed

### CI Requirements

All PRs must pass these checks:

```bash
# These commands must all succeed
cargo build --all-features
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --no-deps
```

### Merge Policy

- PRs are merged using **squash and merge**
- The squash commit message should summarize all changes
- Branch is deleted after merge

---

## Testing Requirements

OxiZ maintains high test coverage. All contributions must include appropriate tests.

### Unit Tests

Every public API must have unit tests:

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

# Check test coverage (requires cargo-tarpaulin)
cargo tarpaulin --all-features
```

### Coverage Expectations

- New features: Aim for >80% coverage of new code
- Bug fixes: Include a test that reproduces the bug
- Critical paths (solving, proof generation): Aim for >90% coverage

---

## Issue Guidelines

### Bug Reports

When reporting a bug, include:

```markdown
## Bug Description

Clear description of the bug.

## Steps to Reproduce

1. Step one
2. Step two
3. ...

## Expected Behavior

What should happen.

## Actual Behavior

What actually happens.

## Environment

- OxiZ version:
- Rust version:
- OS:

## Minimal Reproduction

```smt2
; SMT-LIB2 input that triggers the bug
(set-logic QF_LIA)
(declare-const x Int)
...
```

## Additional Context

Any other relevant information.
```

### Feature Requests

When requesting a feature:

```markdown
## Feature Description

Clear description of the proposed feature.

## Motivation

Why this feature would be useful.

## Proposed API

```rust
// Example of how the feature might look
pub fn new_feature(&mut self, param: Type) -> Result<Output> {
    // ...
}
```

## Alternatives Considered

Other approaches you've considered.
```

### Labels

We use these labels for issues:

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention is needed |
| `performance` | Performance improvements |
| `theory/*` | Specific theory solver |
| `crate/*` | Specific crate |

---

## Architecture Overview

OxiZ is organized as a Cargo workspace with 12 crates:

### Crate Hierarchy

```
oxiz (meta-crate: unified API)
  |
  +-- oxiz-cli (Command-line interface)
  +-- oxiz-wasm (WebAssembly bindings)
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

## Communication

### GitHub Issues

Use GitHub Issues for:
- Bug reports
- Feature requests
- Documentation issues
- Questions about implementation

### GitHub Discussions

Use GitHub Discussions for:
- General questions about usage
- Design discussions
- Community announcements
- Sharing benchmarks and use cases

### Response Times

We aim to:
- Triage new issues within 48 hours
- Provide initial PR review within 1 week
- Respond to questions within a few days

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

Unacceptable behavior includes harassment, trolling, insults, and other unprofessional conduct.

---

## Thank You!

Thank you for taking the time to contribute to OxiZ. Your efforts help advance the state of SMT solving and formal verification. We look forward to your contributions!

*If you have any questions not covered in this guide, please open an issue or start a discussion.*
