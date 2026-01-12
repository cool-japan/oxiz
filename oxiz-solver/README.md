# oxiz-solver

Main CDCL(T) SMT solver orchestration for OxiZ.

## Overview

This crate integrates the SAT solver with theory solvers to provide complete SMT solving:

- **CDCL(T)** - SAT solver with theory propagation
- **Context** - High-level API for SMT-LIB2 interaction
- **Model Generation** - Extract satisfying assignments

## Architecture

```
┌────────────────────────────────────────────────────┐
│                    Context                         │
│  (SMT-LIB2 interface, declaration management)      │
├────────────────────────────────────────────────────┤
│                    Solver                          │
│  (CDCL(T) orchestration, theory combination)       │
├────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ SAT Core │  │   EUF    │  │   Arithmetic     │  │
│  │(oxiz-sat)│  │  Solver  │  │     Solver       │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└────────────────────────────────────────────────────┘
```

## Usage

### High-Level API (Context)

```rust
use oxiz_solver::Context;

let mut ctx = Context::new();

// Execute SMT-LIB2 script
let results = ctx.execute_script(r#"
    (set-logic QF_LIA)
    (declare-const x Int)
    (declare-const y Int)
    (assert (> x 0))
    (assert (< y 10))
    (assert (= (+ x y) 15))
    (check-sat)
    (get-model)
"#)?;

for line in results {
    println!("{}", line);
}
```

### Low-Level API (Solver)

```rust
use oxiz_solver::{Solver, SolverResult};
use oxiz_core::ast::TermId;

let mut solver = Solver::new();
solver.set_logic("QF_UF");

// Assert terms
solver.assert(term_id);

// Check satisfiability
match solver.check() {
    SolverResult::Sat => {
        if let Some(model) = solver.model() {
            // Extract assignments
        }
    }
    SolverResult::Unsat => println!("Unsatisfiable"),
    SolverResult::Unknown => println!("Unknown"),
}
```

## Modules

### `context`

High-level SMT-LIB2 context:
- Declaration management (constants, functions, sorts)
- Script execution
- Result formatting

### `solver`

CDCL(T) solver:
- SAT/theory integration
- Boolean encoding
- Model construction
- Push/pop state management

## Supported Logics

- `QF_UF` - Quantifier-free Uninterpreted Functions
- `QF_LRA` - Quantifier-free Linear Real Arithmetic
- `QF_LIA` - Quantifier-free Linear Integer Arithmetic (partial)
- `QF_BV` - Quantifier-free BitVectors (partial)

## Dependencies

- `oxiz-core` - AST and parsing
- `oxiz-sat` - SAT solver
- `oxiz-theories` - Theory solvers

## License

MIT OR Apache-2.0
