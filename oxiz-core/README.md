# oxiz-core

Core data structures and utilities for OxiZ SMT solver.

## Status (v0.3.4)

| Metric | Value |
|:-------|:------|
| Version | 0.3.4 |
| Status | Stable |
| Tests | 2,216 passing |
| Release Date | 2026-10-08 |
| Source files | 237 |
| Public API items | 3,025 (recorded at 0.3.1) |

Changes in 0.3.1:

- **BREAKING (0.x)**: `ModelValue::BitVec` now carries a `num_bigint::BigUint`
  instead of a `u64`, so bit-vectors wider than 64 bits are represented
  losslessly. New helpers: `ModelValue::from_bitvec_int`,
  `ModelValue::from_bitvec_bits`, `ModelValue::as_bitvec`,
  `ast::model::bitvec_mask`, and
  `Model::assign_bitvec_big` (the `u64` `Model::assign_bitvec` is kept for
  narrow widths).
- Every remaining unguarded recursive term walk — parser, printer, model
  evaluator, substitution, and the `Drop`/`Clone`/`PartialEq` impls on the deep
  public enums — now runs on an explicit heap stack, so deeply nested input can
  no longer overflow the stack.
- The SMT-LIB2 parser is fully iterative and rejects unhandled or malformed
  input with an error instead of silently dropping or defaulting it; mixed-width
  bit-vector binary operands are now rejected at parse time, as Z3 does.
- E-matching trigger inference is restricted to uninterpreted heads (matching
  Z3's `pattern_inference`), which removes a matching loop.
- New `ast::normal_forms::to_cnf_tseitin` — equisatisfiable linear-size CNF via
  Tseitin encoding; `TseitinCnfTactic` is rewired to it.

## Overview

This crate provides the foundational components used across all OxiZ crates:

- **AST** - Term representation with hash-consing for memory efficiency
- **Sorts** - Type system for SMT sorts (Bool, Int, Real, BitVec, etc.)
- **SMT-LIB2** - Lexer, parser, and printer for the standard SMT format
- **Tactics** - Framework for solver strategies and term transformations
- **Models** - Model values (`ModelValue`) and evaluation under a model

## Modules

### `ast`

Hash-consed term representation using arena allocation:

```rust
use oxiz_core::ast::TermManager;

let mut tm = TermManager::new();
let x = tm.mk_var("x", tm.sorts.bool_sort);
let y = tm.mk_var("y", tm.sorts.bool_sort);
let and_xy = tm.mk_and([x, y]);
```

Key types:
- `TermId` - Lightweight handle to a term
- `Term` - Term data (kind, sort, children)
- `TermKind` - Enum of all term kinds (And, Or, Not, Eq, etc.)
- `TermManager` - Creates and interns terms

### `sort`

Sort (type) system. Each `TermManager` owns a `SortManager` as its `sorts` field; the three common
sorts are pre-allocated, the rest are interned on demand:

```rust
use oxiz_core::sort::SortManager;

let mut sm = SortManager::new();
let bool_sort = sm.bool_sort;
let int_sort = sm.int_sort;
let bv32 = sm.bitvec(32);
let array_sort = sm.array(int_sort, bool_sort);
```

### `smtlib`

SMT-LIB2 parsing and printing. The parser is iterative (no recursion on term depth) and reports an error for
input it cannot handle rather than silently dropping it:

```rust
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::{Command, parse_script};

let mut tm = TermManager::new();
let input = "(declare-const x Int) (assert (> x 0)) (check-sat)";
let commands: Vec<Command> = parse_script(input, &mut tm)?;
```

### `tactic`

Tactic framework for solver strategies:

```rust
use oxiz_core::ast::TermManager;
use oxiz_core::tactic::{Goal, SimplifyTactic};

let mut tm = TermManager::new();
let p = tm.mk_var("p", tm.sorts.bool_sort);
let goal = Goal::new(vec![p]);

let mut tactic = SimplifyTactic::new(&mut tm);
let result = tactic.apply_mut(&goal)?;
```

### `model`

Model values and evaluation. As of 0.3.1 a bit-vector value is an
arbitrary-precision unsigned bit pattern, so widths beyond 64 bits are exact:

```rust
use num_bigint::BigUint;
use oxiz_core::ast::ModelValue;

let value = ModelValue::from_bitvec_bits(BigUint::from(255u32), 128);
let Some((bits, width)) = value.as_bitvec() else {
    unreachable!("built as a bit-vector value")
};
assert_eq!(width, 128);
assert_eq!(*bits, BigUint::from(255u32));
```

## Dependencies

- `rustc-hash` - Fast hash maps
- `smallvec` - Stack-allocated vectors
- `thiserror` - Error handling
- `num-bigint` / `num-rational` / `num-integer` / `num-traits` - Arbitrary-precision
  integers, rationals and bit-vector values

All of them are Pure Rust; the crate builds with `no_std` when the `std`
feature is disabled.

## License

Apache-2.0
