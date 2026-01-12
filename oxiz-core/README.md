# oxiz-core

Core data structures and utilities for OxiZ SMT solver.

## Overview

This crate provides the foundational components used across all OxiZ crates:

- **AST** - Term representation with hash-consing for memory efficiency
- **Sorts** - Type system for SMT sorts (Bool, Int, Real, BitVec, etc.)
- **SMT-LIB2** - Lexer, parser, and printer for the standard SMT format
- **Tactics** - Framework for solver strategies and term transformations

## Modules

### `ast`

Hash-consed term representation using arena allocation:

```rust
use oxiz_core::ast::TermManager;

let mut tm = TermManager::new();
let x = tm.mk_var("x", sort_bool);
let y = tm.mk_var("y", sort_bool);
let and_xy = tm.mk_and([x, y]);
```

Key types:
- `TermId` - Lightweight handle to a term
- `Term` - Term data (kind, sort, children)
- `TermKind` - Enum of all term kinds (And, Or, Not, Eq, etc.)
- `TermManager` - Creates and interns terms

### `sort`

Sort (type) system:

```rust
use oxiz_core::sort::SortManager;

let mut sm = SortManager::new();
let bool_sort = sm.bool();
let int_sort = sm.int();
let bv32 = sm.bitvec(32);
let array_sort = sm.array(int_sort, bool_sort);
```

### `smtlib`

SMT-LIB2 parsing and printing:

```rust
use oxiz_core::smtlib::{Lexer, Parser};

let input = "(declare-const x Int) (assert (> x 0)) (check-sat)";
let mut parser = Parser::new(input, &mut term_manager, &mut sort_manager);
let commands = parser.parse_script()?;
```

### `tactic`

Tactic framework for solver strategies:

```rust
use oxiz_core::tactic::{Goal, SimplifyTactic, Tactic};

let mut tactic = SimplifyTactic::new();
let result = tactic.apply(goal)?;
```

## Dependencies

- `rustc-hash` - Fast hash maps
- `smallvec` - Stack-allocated vectors
- `thiserror` - Error handling

## License

MIT OR Apache-2.0
