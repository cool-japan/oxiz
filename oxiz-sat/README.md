# oxiz-sat

CDCL SAT solver implementation for OxiZ.

## Overview

This crate implements a modern Conflict-Driven Clause Learning (CDCL) SAT solver with:

- **Two-Watched Literals** - Efficient unit propagation
- **VSIDS** - Variable State Independent Decaying Sum branching heuristic
- **Clause Learning** - First-UIP conflict analysis
- **Incremental Solving** - Push/pop for assumption-based solving

## Architecture

```
┌─────────────────────────────────────────┐
│              Solver                     │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │  Trail  │  │ Watched │  │  VSIDS  │  │
│  │         │  │  Lists  │  │         │  │
│  └─────────┘  └─────────┘  └─────────┘  │
├─────────────────────────────────────────┤
│           Clause Database               │
└─────────────────────────────────────────┘
```

## Usage

```rust
use oxiz_sat::{Solver, SolverResult, Lit, Var};

let mut solver = Solver::new();

// Create variables
let x = solver.new_var();
let y = solver.new_var();
let z = solver.new_var();

// Add clauses: (x OR y) AND (NOT x OR z) AND (NOT y OR NOT z)
solver.add_clause([Lit::pos(x), Lit::pos(y)]);
solver.add_clause([Lit::neg(x), Lit::pos(z)]);
solver.add_clause([Lit::neg(y), Lit::neg(z)]);

match solver.solve() {
    SolverResult::Sat => {
        let model = solver.model();
        println!("SAT: {:?}", model);
    }
    SolverResult::Unsat => println!("UNSAT"),
    SolverResult::Unknown => println!("UNKNOWN"),
}
```

## Modules

### `literal`

Literal and variable representation:
- `Var` - Variable index
- `Lit` - Signed literal (variable + polarity)
- `LBool` - Three-valued logic (True, False, Undef)

### `clause`

Clause representation and database:
- `Clause` - Immutable clause with literals
- `ClauseRef` - Reference to clause in database
- `ClauseDatabase` - Storage for all clauses

### `trail`

Assignment trail for backtracking:
- Decision levels
- Propagation reasons
- Efficient backtracking

### `watched`

Two-watched literal scheme:
- O(1) watch updates during propagation
- Lazy watch list maintenance

### `vsids`

VSIDS branching heuristic:
- Activity-based variable selection
- Exponential decay
- Conflict-driven bumping

## Performance

The solver is optimized for:
- Cache-friendly clause storage
- Minimal allocations during solving
- Fast unit propagation via two-watched literals

## License

MIT OR Apache-2.0
