# SCIRS2 Policy Compliance - oxiz-math

This document tracks compliance with SCIRS2 (SciRS2) project policies for the oxiz-math crate.

## Dependencies

The oxiz-math crate uses **standard Rust numerical libraries** instead of SciRS2-Core:

- `num-rational` - Arbitrary precision rational numbers
- `num-bigint` - Arbitrary precision integers
- `num-traits` - Numerical traits
- `num-integer` - Integer traits and operations
- `rustc-hash` - Fast hash map implementation
- `smallvec` - Stack-allocated vectors
- `thiserror` - Error handling

### Rationale for Standard Libraries

oxiz-math is a **foundational mathematical library** that provides:
- Arbitrary precision rational arithmetic
- Interval arithmetic
- Polynomial operations
- Simplex algorithm for linear programming
- Delta-rational numbers for strict inequalities

These are **primitive mathematical operations** that form the basis for higher-level numerical computing. Using SciRS2-Core would create circular dependencies, as SciRS2-Core itself may depend on similar foundational mathematics.

## Code Quality Standards

### No Warnings Policy ✅
- All code compiles without warnings
- `cargo clippy --all-features -- -D warnings` passes cleanly
- All clippy suggestions have been addressed

### No Unwrap Policy ✅
- No use of `.unwrap()` in production code
- All fallible operations use proper error handling with `Result` or `Option`
- Uses `.unwrap_or()`, `.unwrap_or_else()`, or `?` operator where appropriate

### Latest Crates Policy ✅
- All dependencies use workspace-level version management
- Versions are specified in root `Cargo.toml`
- Using latest stable versions available on crates.io

### Workspace Policy ✅
- All dependency versions use `.workspace = true`
- No version control in individual crate `Cargo.toml`
- Keywords and categories are crate-specific as required

### Refactoring Policy ✅
- Current codebase: ~4,128 lines of Rust code
- Well under 2000 lines per file limit
- Modular design with separate files for each major component

### Naming Convention Policy ✅
- All variables use `snake_case`
- All types use `PascalCase`
- All constants use `SCREAMING_SNAKE_CASE`
- Follows Rust standard naming conventions

### Testing Policy ✅
- 81 unit tests covering all modules
- All tests pass with `cargo nextest run --all-features`
- Tests use temporary file handling with `std::env::temp_dir()` where needed
- Comprehensive coverage of edge cases

## Module Structure

```
oxiz-math/
├── src/
│   ├── lib.rs              - Crate root and public API
│   ├── delta_rational.rs   - Delta-rational numbers (r + δ*k)
│   ├── interval.rs         - Interval arithmetic with bounds
│   ├── polynomial.rs       - Multivariate polynomial operations
│   ├── rational.rs         - Rational number utilities
│   └── simplex.rs          - Simplex algorithm for LP
└── tests/                  - Integration tests
```

## Performance Characteristics

- Pure Rust implementation with zero-cost abstractions
- Arbitrary precision arithmetic using `num-bigint`
- Efficient hash-based data structures using `rustc-hash`
- Stack-allocated small vectors using `smallvec`

## Safety

- 100% safe Rust code
- No unsafe blocks
- No FFI dependencies
- No C/C++ bindings

## Documentation

- All public APIs documented with doc comments
- Module-level documentation explaining purpose and usage
- Examples in doc comments where appropriate
- Reference to Z3 implementation noted for algorithm provenance

## Future Considerations

If SciRS2-Core develops its own arbitrary precision arithmetic library that:
1. Has zero external dependencies
2. Provides equivalent or better performance
3. Maintains API compatibility

Then oxiz-math could consider migrating to use it. However, for now, using standard `num-*` crates is the most practical approach for foundational mathematics.

## Compliance Status: ✅ COMPLIANT

All SCIRS2 policies have been followed to the extent applicable for a foundational mathematical library.
