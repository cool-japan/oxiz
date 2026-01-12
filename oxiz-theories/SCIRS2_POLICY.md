# SCIRS2 Policy Compliance - oxiz-theories

This document tracks compliance with SCIRS2 (SciRS2) project policies for the oxiz-theories crate.

## Dependencies

The oxiz-theories crate uses **standard Rust libraries** instead of SciRS2-Core:

- `oxiz-core` - Core AST, error handling, and SMT-LIB parsing (sibling crate)
- `oxiz-sat` - SAT solver (sibling crate)
- `oxiz-math` - Mathematical utilities (sibling crate)
- `oxiz-nlsat` - Nonlinear arithmetic solver (sibling crate)
- `num-rational` - Arbitrary precision rational numbers
- `num-traits` - Numerical traits
- `rustc-hash` - Fast hash map implementation
- `smallvec` - Stack-allocated vectors
- `thiserror` - Error handling

### Rationale for Standard Libraries

oxiz-theories is a **foundational SMT theory solver library** that provides:
- Equality with Uninterpreted Functions (EUF)
- Linear Rational/Integer Arithmetic (LRA/LIA)
- BitVector theory with bit-blasting and word-level propagation
- Array theory with read-over-write axioms
- String theory with regular expressions
- Floating-point arithmetic (IEEE 754)
- Algebraic Datatypes
- Theory combination (Nelson-Oppen)

These are **core SMT solver components** that implement the foundational theories needed for automated theorem proving. Using SciRS2-Core would be inappropriate as this is domain-specific SMT solving infrastructure, not general scientific computing.

## Code Quality Standards

### No Warnings Policy ✅
- All code compiles without warnings
- `cargo clippy --all-features -- -D warnings` passes cleanly
- All clippy suggestions have been addressed
- Added `#[allow(dead_code)]` for placeholder code (future LIA cuts, E-matching code trees)

### No Unwrap Policy ⚠️
- **Status**: Partial compliance
- Test code uses `.unwrap()` (acceptable)
- Some production code still contains `.unwrap()` calls that should be refactored
- Uses `.unwrap_or()`, `.unwrap_or_else()`, `.unwrap_or_default()`, and `?` operator in most places
- **Action Item**: Refactor remaining production unwraps to use proper error handling

### Latest Crates Policy ✅
- All dependencies use workspace-level version management
- Versions are specified in root `Cargo.toml`
- Using latest stable versions available on crates.io

### Workspace Policy ✅
- All dependency versions use `.workspace = true`
- No version control in individual crate `Cargo.toml`
- Keywords and categories are crate-specific as required

### Refactoring Policy ✅
- Current codebase: **7,926 lines of Rust code** across 26 files
- Largest file: `src/bv/solver.rs` at 1,268 lines
- All files well under 2,000 lines per file limit
- Modular design with separate files for each theory and component

### Naming Convention Policy ✅
- All variables use `snake_case`
- All types use `PascalCase`
- All constants use `SCREAMING_SNAKE_CASE`
- Follows Rust standard naming conventions

### Testing Policy ✅
- **126 unit tests** covering all modules
- All tests pass with `cargo nextest run --all-features`
- Tests use temporary file handling with `std::env::temp_dir()` where needed
- Comprehensive coverage of:
  - EUF congruence closure and explanations
  - Simplex algorithm for LRA
  - BitVector operations (add, sub, mul, division, shifts, comparisons)
  - Word-level propagators for BitVectors
  - Array read-over-write and extensionality
  - String operations and regex automata
  - Floating-point constraints
  - Algebraic datatype reasoning
  - Theory combination and purification

## Module Structure

```
oxiz-theories/
├── src/
│   ├── lib.rs                  - Crate root and public API
│   ├── theory.rs               - Theory trait definition
│   ├── combination.rs          - Nelson-Oppen theory combination
│   ├── arithmetic/
│   │   ├── mod.rs              - Arithmetic theory exports
│   │   ├── solver.rs           - Main LRA/LIA solver
│   │   ├── simplex.rs          - Simplex algorithm (1,205 lines)
│   │   ├── delta.rs            - Delta-rational numbers
│   │   ├── lia.rs              - Linear Integer Arithmetic
│   │   └── optimize.rs         - LRA optimization (NEW)
│   ├── bv/
│   │   ├── mod.rs              - BitVector theory exports
│   │   ├── solver.rs           - BitVector solver with bit-blasting (1,268 lines)
│   │   └── propagator.rs       - Word-level propagators (NEW)
│   ├── array/
│   │   ├── mod.rs              - Array theory exports
│   │   └── solver.rs           - Array theory solver
│   ├── string/
│   │   ├── mod.rs              - String theory exports
│   │   ├── solver.rs           - String solver (1,024 lines)
│   │   └── regex.rs            - Regular expression automata
│   ├── fp/
│   │   ├── mod.rs              - Floating-point theory exports
│   │   └── solver.rs           - IEEE 754 floating-point (918 lines)
│   ├── datatype/
│   │   ├── mod.rs              - Datatype theory exports
│   │   └── solver.rs           - Algebraic datatype solver
│   └── euf/
│       ├── mod.rs              - EUF theory exports
│       ├── solver.rs           - Congruence closure
│       ├── union_find.rs       - Backtrackable union-find
│       ├── ematching.rs        - E-matching for quantifiers
│       └── proof.rs            - Proof generation
└── tests/                      - Integration tests
```

## Recent Enhancements

### BitVector Theory
- ✅ Implemented division operations (udiv, sdiv, urem, srem)
- ✅ Added word-level propagators with interval analysis
- ✅ Constraints: Eq, Ult, Ule, Add, Sub, And, Or
- ✅ Propagation for shifts (shl, lshr)

### LRA Optimization
- ✅ Implemented maximize/minimize operations
- ✅ Objective function builder with linear expressions
- ✅ Constraint management (Le, Ge, Eq)
- ✅ Optimal value extraction and model management

## Performance Characteristics

- Pure Rust implementation with zero-cost abstractions
- Bit-blasting to SAT for BitVector theory
- Word-level propagation to avoid full bit-blasting
- Simplex algorithm for linear arithmetic
- Congruence closure with path compression and union-by-rank
- Efficient hash-based data structures using `rustc-hash`
- Stack-allocated small vectors using `smallvec`

## Safety

- 100% safe Rust code
- No unsafe blocks
- No FFI dependencies
- No C/C++ bindings

## Documentation

- All public APIs documented with doc comments
- Module-level documentation explaining purpose and algorithms
- Examples in doc comments where appropriate
- Reference to Z3 implementation noted for algorithm provenance
- Detailed comments for complex algorithms (Simplex, congruence closure, etc.)

## Algorithm Sources

The implementations in oxiz-theories are **heavily based on** the Z3 SMT solver:
- Z3 repository: `../z3/`
- Algorithms adapted and transformed into idiomatic Rust
- Core SMT techniques from research papers and Z3's proven implementations
- No direct code translation - redesigned for Rust's ownership model and type system

## TODO Items

From TODO.md, pending enhancements:
- [ ] Dynamic arity functions (EUF)
- [ ] Preprocessing - Gaussian elimination (LRA)
- [ ] Lazy encoding strategies (BitVectors)
- [ ] AIGS representation (BitVectors)
- [ ] Incremental array solving with push/pop
- [ ] String-integer conversion (str.to_int, str.from_int)
- [ ] Unicode support (String theory)
- [ ] IEEE 754 floating-point enhancements

## Compliance Status: ⚠️ MOSTLY COMPLIANT

### Compliant Policies:
- ✅ No Warnings Policy - passes `cargo clippy --all-features -- -D warnings`
- ✅ Latest Crates Policy - using workspace dependencies
- ✅ Workspace Policy - all deps use `.workspace = true`
- ✅ Refactoring Policy - all files under 2,000 lines
- ✅ Naming Convention Policy - follows Rust standards
- ✅ Testing Policy - 126 comprehensive tests

### Action Items:
- ⚠️ **No Unwrap Policy** - Some production code still uses `.unwrap()`
  - Test code unwraps are acceptable
  - Production unwraps should be refactored to proper error handling
  - Estimated: ~10-15 production unwraps to fix

### Notes:
- SciRS2-Core is not used because oxiz-theories is domain-specific SMT solving infrastructure
- The crate implements foundational automated theorem proving algorithms
- Dependencies are appropriate for the problem domain
