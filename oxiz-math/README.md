# oxiz-math

[![Version](https://img.shields.io/badge/version-0.2.1-blue)](https://github.com/cool-japan/oxiz)
[![Status](https://img.shields.io/badge/status-stable-green)]()

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
| `grobner` | Gröbner basis computation | `math/grobner/` |
| `realclosure` | Real closed field arithmetic | `math/realclosure/` |

## Usage

```rust
use oxiz_math::simplex::Simplex;
use oxiz_math::polynomial::Polynomial;
use oxiz_math::interval::Interval;
```

## Status (v0.2.1)

| Metric | Value |
|:-------|:------|
| Tests | 586 passing |
| Rust LoC | 30,748 (60 files) |
| Public API items | 1,036 |
| `todo!`/`unimplemented!` | 0 |
| Status | Stable |

## Design Principles

- **Pure Rust**: No C/C++ dependencies
- **Generic**: Works with various numeric types
- **Incremental**: Supports incremental updates for SMT integration
- **Efficient**: Optimized for SMT workloads

## License

Apache-2.0
