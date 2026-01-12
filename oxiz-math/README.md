# oxiz-math

Mathematical foundations for the OxiZ SMT solver.

## Overview

This crate provides Pure Rust implementations of mathematical algorithms required for SMT solving. It serves as the foundation for arithmetic theories (LRA, LIA, NRA, NIA) and optimization.

## Modules

| Module | Description | Z3 Reference |
|:-------|:------------|:-------------|
| `simplex` | Simplex algorithm for linear programming | `math/simplex/` |
| `polynomial` | Polynomial arithmetic | `math/polynomial/` |
| `interval` | Interval arithmetic for bounds | `math/interval/` |
| `rational` | Arbitrary precision rationals | - |
| `grobner` | Gröbner basis computation (planned) | `math/grobner/` |
| `realclosure` | Real closed field arithmetic (planned) | `math/realclosure/` |

## Usage

```rust
use oxiz_math::simplex::Simplex;
use oxiz_math::polynomial::Polynomial;
use oxiz_math::interval::Interval;
```

## Design Principles

- **Pure Rust**: No C/C++ dependencies
- **Generic**: Works with various numeric types
- **Incremental**: Supports incremental updates for SMT integration
- **Efficient**: Optimized for SMT workloads

## License

MIT OR Apache-2.0
