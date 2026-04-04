# Z3 to OxiZ Migration Guide

Complete guide for migrating from Z3 to OxiZ, covering API mappings,
feature comparisons, and common migration patterns.

---

## Table of Contents

1. [Architecture Differences](#architecture-differences)
2. [API Mapping: Z3 Python to OxiZ Rust](#api-mapping-z3-python-to-oxiz-rust)
3. [API Mapping: Z3 Python to OxiZ Python](#api-mapping-z3-python-to-oxiz-python)
4. [SMT-LIB2 Compatibility](#smt-lib2-compatibility)
5. [Feature Comparison](#feature-comparison)
6. [Common Migration Patterns](#common-migration-patterns)
7. [Performance Differences](#performance-differences)
8. [Unsupported Features](#unsupported-features)

---

## Architecture Differences

| Aspect | Z3 | OxiZ |
|--------|-----|------|
| Language | C++ with Python/C#/Java bindings | Pure Rust with Python (PyO3) binding |
| Term management | GC-based Context | Explicit `TermManager` (arena-based) |
| Memory model | Shared references via GC | Ownership + `TermId` handles |
| Thread safety | Context per thread | `Send + Sync` solver, shared `TermManager` via `Arc` |
| no_std support | No | Yes (core solver works without std) |

Key philosophical difference: Z3's Python API uses operator overloading
(`x + y`, `x == y`) for term construction. OxiZ uses explicit builder methods
(`tm.mk_add`, `tm.mk_eq`) or SMT-LIB2 string parsing.

---

## API Mapping: Z3 Python to OxiZ Rust

### Solver Creation

**Z3 (Python):**
```python
import z3
s = z3.Solver()
s.set("timeout", 5000)
```

**OxiZ (Rust):**
```rust
use oxiz_solver::{Solver, SolverConfig};

let mut config = SolverConfig::balanced();
config.timeout_ms = 5000;
let mut solver = Solver::with_config(config);
```

### Variable Declaration

**Z3 (Python):**
```python
x = z3.Int('x')
y = z3.Real('y')
p = z3.Bool('p')
bv = z3.BitVec('bv', 32)
```

**OxiZ (Rust):**
```rust
use oxiz_core::ast::TermManager;

let mut tm = TermManager::new();
let x = tm.mk_var("x", tm.sorts.int_sort);
let y = tm.mk_var("y", tm.sorts.real_sort);
let p = tm.mk_var("p", tm.sorts.bool_sort);
let bv = tm.mk_var("bv", tm.sorts.bv_sort(32));
```

### Formula Construction

**Z3 (Python):**
```python
constraint = z3.And(x + y == 10, x > 0, y > 0)
s.add(constraint)
```

**OxiZ (Rust):**
```rust
use num_bigint::BigInt;

let ten = tm.mk_int(BigInt::from(10));
let zero = tm.mk_int(BigInt::from(0));
let sum_eq = tm.mk_eq(tm.mk_add(vec![x, y]), ten);
let x_pos = tm.mk_gt(x, zero);
let y_pos = tm.mk_gt(y, zero);
let constraint = tm.mk_and(vec![sum_eq, x_pos, y_pos]);
solver.assert(constraint, &mut tm);
```

### Solving and Model Extraction

**Z3 (Python):**
```python
if s.check() == z3.sat:
    m = s.model()
    print(m[x], m[y])
```

**OxiZ (Rust):**
```rust
use oxiz_solver::SolverResult;

match solver.check(&mut tm) {
    SolverResult::Sat => {
        if let Some(model) = solver.get_model() {
            // model provides variable assignments
        }
    }
    SolverResult::Unsat => { /* handle unsat */ }
    SolverResult::Unknown => { /* handle timeout/resource limit */ }
}
```

### Z3 Compatibility Layer

OxiZ provides a Z3-style API wrapper for faster migration:

```rust
use oxiz_solver::z3_compat::{Z3Config, Z3Context, Z3Solver, Bool, Int, SatResult};

let cfg = Z3Config::new();
let ctx = Z3Context::new(&cfg);
let mut solver = Z3Solver::new(&ctx);

let p = Bool::new_const(&ctx, "p");
let q = Bool::new_const(&ctx, "q");
solver.assert(&Bool::and(&ctx, &[p.clone(), q.clone()]));

match solver.check() {
    SatResult::Sat => { /* ... */ }
    SatResult::Unsat => { /* ... */ }
    SatResult::Unknown => { /* ... */ }
}
```

---

## API Mapping: Z3 Python to OxiZ Python

| Z3 Python | OxiZ Python | Notes |
|-----------|-------------|-------|
| `z3.Solver()` | `oxiz.Solver()` | Direct equivalent |
| `z3.Context()` | `oxiz.TermManager()` | Separate term construction from solving |
| `z3.Bool('p')` | `tm.mk_var('p', 'Bool')` | Sort as string |
| `z3.Int('x')` | `tm.mk_var('x', 'Int')` | |
| `z3.Real('r')` | `tm.mk_var('r', 'Real')` | |
| `z3.BitVec('b', 32)` | `tm.mk_var('b', 'BitVec[32]')` | Width in sort string |
| `z3.IntVal(42)` | `tm.mk_int(42)` | |
| `z3.And(a, b)` | `tm.mk_and([a, b])` | Takes a list |
| `z3.Or(a, b)` | `tm.mk_or([a, b])` | Takes a list |
| `z3.Not(p)` | `tm.mk_not(p)` | |
| `z3.Implies(a, b)` | `tm.mk_implies(a, b)` | |
| `a + b` | `tm.mk_add([a, b])` | No operator overloading |
| `a == b` | `tm.mk_eq(a, b)` | |
| `a < b` | `tm.mk_lt(a, b)` | |
| `z3.If(c, t, e)` | `tm.mk_ite(c, t, e)` | |
| `z3.Select(a, i)` | `tm.mk_select(a, i)` | Array read |
| `z3.Store(a, i, v)` | `tm.mk_store(a, i, v)` | Array write |
| `solver.add(f)` | `solver.assert_term(t, tm)` | |
| `solver.check()` | `solver.check_sat(tm)` | |
| `solver.model()` | `solver.get_model(tm)` | Returns `dict[str, str]` |
| `solver.push()` | `solver.push()` | Identical |
| `solver.pop()` | `solver.pop()` | Identical |

---

## SMT-LIB2 Compatibility

OxiZ supports standard SMT-LIB2 input files:

```bash
oxiz solve problem.smt2
```

### Supported Commands

| Command | Status | Notes |
|---------|--------|-------|
| `set-logic` | Full | All standard logics |
| `declare-const` | Full | |
| `declare-fun` | Full | |
| `declare-sort` | Full | |
| `declare-datatypes` | Full | Parametric and recursive |
| `define-fun` | Full | |
| `assert` | Full | |
| `check-sat` | Full | |
| `get-model` | Full | |
| `get-value` | Full | |
| `push` / `pop` | Full | |
| `reset` | Full | |
| `set-option` | Partial | Common options supported |
| `get-unsat-core` | Full | Must enable via option |
| `get-proof` | Full | |
| `check-sat-assuming` | Full | |
| `echo` | Full | |
| `exit` | Full | |

### SMT-LIB2 Differences

- OxiZ accepts `(set-option :timeout N)` in milliseconds
- MBQI options use `(set-option :mbqi.max_instantiations N)`
- Proof format uses a simplified format (not full SMT-LIB2 proof format)

---

## Feature Comparison

| Feature | Z3 | OxiZ | Notes |
|---------|-----|------|-------|
| QF_LIA | Full | Full | Omega test + cutting planes |
| QF_LRA | Full | Full | Dual Simplex |
| QF_NIA | Full | Full | NLSAT (incomplete for integers) |
| QF_NRA | Full | Full | NLSAT + CAD |
| QF_BV | Full | Full | Eager bit-blasting |
| QF_A / Arrays | Full | Full | Read-over-write + extensionality |
| QF_UF | Full | Full | Congruence closure |
| QF_FP | Full | Full | IEEE 754 via bit-blasting |
| QF_DT | Full | Full | ADTs with acyclicity |
| QF_S (Strings) | Full | Full | Brzozowski derivatives |
| Quantifiers (MBQI) | Full | Full | Model-based instantiation |
| E-matching | Full | Full | Trigger-based instantiation |
| MaxSAT/MaxSMT | Full | Full | RC2, PM-RES, MaxHS, IHS, SortMax |
| Optimization (OMT) | Full | Full | Minimize/maximize objectives |
| Pareto enumeration | Partial | Full | Multi-objective optimization |
| CHC/Spacer | Full | Full | PDR/IC3 with Spacer |
| Proof generation | Full | Full | |
| Unsat cores | Full | Full | |
| Tactic framework | Rich | Limited | CLI strategy options |
| Python operator overloading | Full | None | Use explicit API |
| Lambda / array lambda | Full | Not yet | |
| Recursive functions | Full | Partial | |
| Parallel tactics (ParOr) | Full | Portfolio | CLI `--parallel` |
| no_std / embedded | No | Yes | Core solver is no_std |
| WASM target | No | Yes | Via oxiz-wasm |
| Pure Rust / no C deps | No (C++) | Yes | Zero C/Fortran dependencies |

---

## Common Migration Patterns

### Pattern 1: Simple Satisfiability Check

**Z3:**
```python
from z3 import *
s = Solver()
x, y = Ints('x y')
s.add(x + y == 10, x > 0, y > 0)
print(s.check())
```

**OxiZ (SMT-LIB2 string API, easiest migration path):**
```python
import oxiz
tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('QF_LIA')
s.assert_formula('(declare-const x Int)', tm)
s.assert_formula('(declare-const y Int)', tm)
s.assert_formula('(= (+ x y) 10)', tm)
s.assert_formula('(> x 0)', tm)
s.assert_formula('(> y 0)', tm)
print(s.check_sat(tm))
```

### Pattern 2: CEGAR Loop with Push/Pop

**Z3:**
```python
s = Solver()
s.push()
s.add(candidate_lemma)
if s.check() == unsat:
    s.pop()
    # try different refinement
```

**OxiZ:**
```python
s = oxiz.Solver()
s.push()
s.assert_term(candidate_lemma, tm)
if s.check_sat(tm) == oxiz.SolverResult.Unsat:
    s.pop()
    # try different refinement
```

### Pattern 3: Optimization

**Z3:**
```python
opt = Optimize()
x = Int('x')
opt.add(x >= 0, x <= 100)
opt.minimize(x)
opt.check()
```

**OxiZ:**
```python
opt = oxiz.Optimizer()
opt.set_logic('QF_LIA')
x = tm.mk_var('x', 'Int')
opt.assert_term(tm.mk_ge(x, tm.mk_int(0)))
opt.assert_term(tm.mk_le(x, tm.mk_int(100)))
opt.minimize(x)
result = opt.optimize(tm)
```

---

## Performance Differences

| Scenario | Z3 | OxiZ | Reason |
|----------|-----|------|--------|
| Startup time | ~50ms | ~5ms | No GC, no C++ runtime init |
| QF_LIA (small) | Fast | Fast | Similar Simplex implementations |
| QF_BV (wide) | Fast | Comparable | Both use eager bit-blasting |
| Quantified (AUFLIA) | Mature heuristics | Good, still tuning | Z3 has decades of heuristic refinement |
| Memory usage | Higher (GC overhead) | Lower (arena allocation) | Rust ownership model |
| Parallel portfolio | Mature | Good | Both use independent threads |
| Incremental (push/pop) | Very optimized | Good | Z3 has more incremental theory solvers |
| WASM deployment | Not possible | Supported | Pure Rust compiles to WASM |

General guidance: For most QF logics, OxiZ performs comparably to Z3. For
heavily quantified problems, Z3's mature heuristics may still have an edge.
OxiZ's advantage is in deployment flexibility (no_std, WASM, embedded) and
lower memory footprint.

---

## Unsupported Features

Features present in Z3 that are not yet available in OxiZ:

| Feature | Z3 API | Workaround |
|---------|--------|------------|
| Python operator overloading | `x + y`, `x == y` | Use `tm.mk_add()`, `tm.mk_eq()` |
| `z3.Tactic` combinators | `Then(t1, t2)` | Use CLI strategy flags |
| `z3.Probe` | Formula probing | Not available |
| `z3.Lambda` | Array lambdas | Not yet; use explicit `store` chains |
| `z3.RecFunction` | Recursive function definitions | Partial; use SMT-LIB2 `define-fun-rec` |
| `z3.FPSort` in Python | FP sort construction | Use SMT-LIB2 string API |
| `solver.assertions()` | Inspect current assertions | Not exposed |
| Algebraic number models | Exact algebraic numbers | Rational approximations returned |
| Custom propagators (Python) | `z3.UserPropagator` | Rust `UserPropagator` trait available |

**Migration tip:** The fastest path is to translate Z3 Python formula construction
into SMT-LIB2 strings via `assert_formula`, then incrementally migrate to the
native term API for performance-critical code paths.
