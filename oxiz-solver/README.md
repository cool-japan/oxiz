# oxiz-solver

Main CDCL(T) SMT solver orchestration for OxiZ.

**Version**: 0.3.4 | **Status**: Stable | **Tests**: 2,510 passing | **LoC**: 80,325 code in `src/` (194 files) | **Public API**: 1,637 items (recorded at 0.3.1)

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
use oxiz_core::ast::TermManager;

let mut solver = Solver::new();
let mut tm = TermManager::new();
solver.set_logic("QF_UF");

let p = tm.mk_var("p", tm.sorts.bool_sort);

// Assert terms
solver.assert(p, &mut tm);

// Check satisfiability
match solver.check(&mut tm) {
    SolverResult::Sat => {
        if let Some(model) = solver.model() {
            // Extract assignments
        }
    }
    SolverResult::Unsat => println!("Unsatisfiable"),
    SolverResult::Unknown => println!("Unknown"),
}
```

The model, unsat core and proof belong to the check that produced them: any
`assert`, `push`, `pop` or `reset` invalidates them, so a stale answer can never
be read back. Repeating `check` on an untouched goal is an O(1) verdict-cache
hit; the cache is dropped by those same mutations and by every settings
mutator.

## Modules

### `context`

High-level SMT-LIB2 context:
- Declaration management (constants, functions, sorts)
- Script execution
- Result formatting

### `solver`

CDCL(T) solver:
- SAT/theory integration
- Boolean encoding (depth-memoized, so a shared DAG is encoded once rather than
  once per path)
- Model construction
- Push/pop state management, with the Tseitin memo retracted entry by entry
  through an undo journal so repeated push/pop/check cycles do not re-encode a
  goal from scratch

### `mbqi`

Model-based quantifier instantiation:
- Finite-range quantifier expansion
- Skolem witness synthesis with CEGAR refinement
- Symbolic model certification over the reals, plus quasi-macro detection
- Search state checkpointed and restored around each check, so instantiation
  does not silently stop on a repeatedly checked goal

## Supported Logics

Status below is what the `bench/z3_parity` differential suite measures against a
real `z3` 4.15.4 binary under the honest comparator (an `Unknown` from either
solver never counts as a match). All 19 logic families in the suite are at 100%
Correct in 0.3.4; per-logic counts and the suite's coverage limits are in the
root [`README.md`](../README.md) and [`TODO.md`](../TODO.md).

- `QF_UF`, `QF_UFLIA`, `QF_UFLRA` - Uninterpreted functions, alone and with arithmetic
- `QF_LRA` (16/16), `QF_LIA` (16/16) - Linear real and integer arithmetic
- `QF_BV` (15/15) - Fixed-size bit-vectors, including widths beyond 64 bits
- `QF_A`, `QF_ALIA`, `QF_ABV`, `QF_AUFBV`, `QF_AUFLIA` - Arrays and their combinations
- `QF_DT` (10/10), `QF_S` (10/10), `QF_FP` (12/12) - Datatypes, strings, floating point
- `QF_NIA` (1/1), `QF_NIRA` (5/5) - Nonlinear integer and mixed integer/real
  arithmetic; the suite's `QF_NIA` coverage is a single benchmark, and broader
  NIA branch-and-bound has known gaps (see the root `TODO.md`)
- `AUFLIA`, `AUFLIRA`, `UFLIA`, `UFLRA` - Quantified logics, via MBQI with
  finite-range expansion, Skolem-witness CEGAR and symbolic model certification

MBQI is not complete in general; the quantified results above are the suite's,
not a general completeness claim.

## Dependencies

- `oxiz-core` - AST and parsing
- `oxiz-sat` - SAT solver
- `oxiz-theories` - Theory solvers
- `oxiz-proof` (optional) - Proof replay and verification
- `num-bigint` / `num-rational` / `num-traits` - Arbitrary-precision arithmetic
- `rustc-hash`, `smallvec`, `hashbrown` - Data structures
- `rayon` (optional) - Parallel portfolio solving

## License

Apache-2.0
