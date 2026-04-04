# Z3 to OxiZ Migration Guide

This guide maps common Z3 Python API patterns to their OxiZ equivalents. OxiZ
is a Pure Rust SMT solver that exposes a Python interface via the `oxiz` package
(backed by PyO3). The core difference in philosophy is that OxiZ builds terms
either through explicit `TermManager` API calls or through SMT-LIB2 string
assertions, whereas Z3's Python API builds terms using operator-overloaded Python
objects.

---

## Table of Contents

1. [Installation and Import](#installation-and-import)
2. [Core API Mapping Table](#core-api-mapping-table)
3. [Sorts and Variables](#sorts-and-variables)
4. [Assertions and Solving](#assertions-and-solving)
5. [Model Extraction](#model-extraction)
6. [Incremental Solving (push/pop)](#incremental-solving-pushpop)
7. [Optimization](#optimization)
8. [SMT-LIB2 String Interface](#smtlib2-string-interface)
9. [Supported Logics](#supported-logics)
10. [Known Limitations vs Z3](#known-limitations-vs-z3)

---

## Installation and Import

**Z3:**
```python
import z3
```

**OxiZ:**
```python
import oxiz
```

---

## Core API Mapping Table

| Z3 Python API | OxiZ Python API | Notes |
|---|---|---|
| `z3.Solver()` | `oxiz.Solver()` | Direct equivalent |
| `z3.Context()` | `oxiz.TermManager()` | OxiZ separates term construction (TermManager) from solving (Solver) |
| `z3.Bool('p')` | `tm.mk_var('p', 'Bool')` | Sort is a string: `'Bool'` |
| `z3.Int('x')` | `tm.mk_var('x', 'Int')` | Sort is a string: `'Int'` |
| `z3.Real('r')` | `tm.mk_var('r', 'Real')` | Sort is a string: `'Real'` |
| `z3.BitVec('b', 32)` | `tm.mk_var('b', 'BitVec[32]')` | Width encoded in sort string |
| `z3.BoolVal(True)` | `tm.mk_bool(True)` | |
| `z3.IntVal(42)` | `tm.mk_int(42)` | |
| `z3.RealVal('3/4')` | `tm.mk_real(3, 4)` | Numerator/denominator as integers |
| `z3.BitVecVal(0xFF, 8)` | `tm.mk_bv(0xFF, 8)` | |
| `z3.Not(p)` | `tm.mk_not(p)` | |
| `z3.And(a, b, c)` | `tm.mk_and([a, b, c])` | Takes a list |
| `z3.Or(a, b, c)` | `tm.mk_or([a, b, c])` | Takes a list |
| `z3.Implies(a, b)` | `tm.mk_implies(a, b)` | |
| `z3.Xor(a, b)` | `tm.mk_xor(a, b)` | |
| `a == b` | `tm.mk_eq(a, b)` | No operator overloading in OxiZ terms |
| `a != b` | `tm.mk_not(tm.mk_eq(a, b))` | Use distinct for more than two |
| `z3.Distinct(a, b, c)` | `tm.mk_distinct([a, b, c])` | |
| `a + b` | `tm.mk_add([a, b])` | |
| `a - b` | `tm.mk_sub(a, b)` | |
| `a * b` | `tm.mk_mul([a, b])` | |
| `z3.UDiv(a, b)` | `tm.mk_bv_udiv(a, b)` | BV unsigned division |
| `a / b` (Int) | `tm.mk_div(a, b)` | Integer division |
| `a % b` | `tm.mk_mod(a, b)` | |
| `-a` | `tm.mk_neg(a)` | |
| `a < b` | `tm.mk_lt(a, b)` | |
| `a <= b` | `tm.mk_le(a, b)` | |
| `a > b` | `tm.mk_gt(a, b)` | |
| `a >= b` | `tm.mk_ge(a, b)` | |
| `z3.If(c, t, e)` | `tm.mk_ite(c, t, e)` | |
| `z3.Select(a, i)` | `tm.mk_select(a, i)` | Array read |
| `z3.Store(a, i, v)` | `tm.mk_store(a, i, v)` | Array write |
| `z3.Concat(a, b)` | `tm.mk_bv_concat(a, b)` | BV concat |
| `z3.Extract(hi, lo, b)` | `tm.mk_bv_extract(hi, lo, b)` | BV bit extract |
| `solver.add(f)` | `solver.assert_term(t, tm)` | Term-based assertion |
| `solver.add(f)` | `solver.assert_formula("(> x 0)", tm)` | String-based assertion |
| `solver.check()` | `solver.check_sat(tm)` | Returns `SolverResult.Sat/Unsat/Unknown` |
| `solver.model()` | `solver.get_model(tm)` | Returns `dict` of `{name: value_str}` |
| `solver.push()` | `solver.push()` | Identical |
| `solver.pop()` | `solver.pop()` | Identical |
| `solver.reset()` | `solver.reset()` | Identical |
| `solver.set('logic', 'QF_LIA')` | `solver.set_logic('QF_LIA')` | |
| `solver.set('timeout', 5000)` | `solver.set_option('timeout', '5000')` | Milliseconds |
| `solver.unsat_core()` | `solver.get_unsat_core()` | Must enable via `set_option` first |
| `solver.num_scopes()` | `solver.context_level` | Property, not a method |
| `solver.num_scopes()` | `solver.num_assertions` | Assertion count |

---

## Sorts and Variables

**Z3:**
```python
import z3

x = z3.Int('x')
y = z3.Real('y')
p = z3.Bool('p')
b = z3.BitVec('b', 32)
```

**OxiZ:**
```python
import oxiz

tm = oxiz.TermManager()

x = tm.mk_var('x', 'Int')
y = tm.mk_var('y', 'Real')
p = tm.mk_var('p', 'Bool')
b = tm.mk_var('b', 'BitVec[32]')
```

The `TermManager` is the single owner of all term nodes. It must be passed to
`Solver` methods that need to inspect or extend the term graph. Keep one
`TermManager` per solving session.

---

## Assertions and Solving

**Z3:**
```python
import z3

s = z3.Solver()
x = z3.Int('x')
y = z3.Int('y')

s.add(x + y == 10)
s.add(x > 0)
s.add(y > 0)

result = s.check()   # z3.sat / z3.unsat / z3.unknown
```

**OxiZ (term API):**
```python
import oxiz

tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('QF_LIA')

x = tm.mk_var('x', 'Int')
y = tm.mk_var('y', 'Int')

ten = tm.mk_int(10)
zero = tm.mk_int(0)

s.assert_term(tm.mk_eq(tm.mk_add([x, y]), ten), tm)
s.assert_term(tm.mk_gt(x, zero), tm)
s.assert_term(tm.mk_gt(y, zero), tm)

result = s.check_sat(tm)
# result == oxiz.SolverResult.Sat
```

**OxiZ (SMT-LIB2 string API — simpler for most use cases):**
```python
import oxiz

tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('QF_LIA')

s.assert_formula('(declare-const x Int)', tm)   # not needed for term API
s.assert_formula('(declare-const y Int)', tm)
s.assert_formula('(= (+ x y) 10)', tm)
s.assert_formula('(> x 0)', tm)
s.assert_formula('(> y 0)', tm)

result = s.check_sat(tm)
```

> **Note:** `assert_formula` parses the string as an SMT-LIB2 term expression.
> Variable declarations are handled implicitly when variables are created via
> `mk_var`, or can be embedded as SMT-LIB2 `declare-const` forms.

---

## Model Extraction

**Z3:**
```python
if s.check() == z3.sat:
    m = s.model()
    print(m[x], m[y])
```

**OxiZ:**
```python
if s.check_sat(tm) == oxiz.SolverResult.Sat:
    model = s.get_model(tm)          # dict[str, str]
    print(model['x'], model['y'])    # e.g. "3", "7"
```

Model values in OxiZ are always returned as strings. Parse them as needed:

| Sort | Example value string |
|------|---------------------|
| `Bool` | `"true"` / `"false"` |
| `Int` | `"42"` / `"-7"` |
| `Real` | `"3"` / `"3/4"` (rational) |
| `BitVec[8]` | `"#xff"` |

---

## Incremental Solving (push/pop)

The `push()` and `pop()` interface is identical between Z3 and OxiZ.

**Z3:**
```python
s = z3.Solver()
s.push()
s.add(x > 5)
r1 = s.check()
s.pop()
s.push()
s.add(x < 3)
r2 = s.check()
s.pop()
```

**OxiZ:**
```python
s = oxiz.Solver()
s.set_logic('QF_LIA')
tm = oxiz.TermManager()
x = tm.mk_var('x', 'Int')

s.push()
s.assert_term(tm.mk_gt(x, tm.mk_int(5)), tm)
r1 = s.check_sat(tm)   # Sat
s.pop()

s.push()
s.assert_term(tm.mk_lt(x, tm.mk_int(3)), tm)
r2 = s.check_sat(tm)   # Sat
s.pop()
```

Current scope depth can be inspected with `solver.context_level`.

---

## Optimization

Z3's `Optimize` class maps to OxiZ's `Optimizer`:

**Z3:**
```python
opt = z3.Optimize()
x = z3.Int('x')
opt.add(x >= 0)
opt.add(x <= 10)
opt.minimize(x)
opt.check()
m = opt.model()
```

**OxiZ:**
```python
import oxiz

tm = oxiz.TermManager()
opt = oxiz.Optimizer()
opt.set_logic('QF_LIA')

x = tm.mk_var('x', 'Int')
opt.assert_term(tm.mk_ge(x, tm.mk_int(0)))
opt.assert_term(tm.mk_le(x, tm.mk_int(10)))
opt.minimize(x)

result = opt.optimize(tm)
if result == oxiz.OptimizationResult.Optimal:
    model = opt.get_model(tm)
    print(model['x'])    # "0"
```

---

## SMT-LIB2 String Interface

For users migrating from Z3 `.smt2` files or who prefer working with SMT-LIB2
syntax directly, OxiZ's CLI accepts `.smt2` files natively:

```bash
oxiz solve problem.smt2
```

The `assert_formula` method on `Solver` also accepts raw SMT-LIB2 term strings,
making it easy to embed existing SMT-LIB2 logic into Python scripts:

```python
import oxiz

tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('QF_BV')

# Assert directly using SMT-LIB2 syntax
s.assert_formula('(= (bvand x #xff) #x0f)', tm)
result = s.check_sat(tm)
```

The OxiZ CLI also supports unsat core extraction, proof generation, and
interactive REPL mode — features accessible from the terminal that complement
the Python API.

---

## Supported Logics

OxiZ supports the following SMT-LIB2 logic identifiers. Pass them to
`solver.set_logic(...)` or include `(set-logic ...)` in `.smt2` files.

### Quantifier-Free Logics (benchmarks in `bench/z3_parity/benchmarks/`)

| Logic | Theories | Benchmark directory |
|-------|----------|-------------------|
| `QF_LIA` | Quantifier-free linear integer arithmetic | `qf_lia/` |
| `QF_LRA` | Quantifier-free linear real arithmetic | `qf_lra/` |
| `QF_NIA` | Quantifier-free nonlinear integer arithmetic (NLSAT) | `qf_nia/` |
| `QF_NIRA` | Quantifier-free nonlinear mixed arithmetic | `QF_NIRA/` |
| `QF_BV` | Quantifier-free bitvectors | `qf_bv/` |
| `QF_ABV` | Quantifier-free arrays + bitvectors | `QF_ABV/` |
| `QF_ALIA` | Quantifier-free arrays + linear integers | `QF_ALIA/` |
| `QF_AUFBV` | Quantifier-free arrays + UF + bitvectors | `QF_AUFBV/` |
| `QF_AUFLIA` | Quantifier-free arrays + UF + integers | `QF_AUFLIA/` |
| `QF_UFLIA` | Quantifier-free uninterpreted functions + integers | `QF_UFLIA/` |
| `QF_UFLRA` | Quantifier-free UF + reals | `QF_UFLRA/` |
| `QF_FP` | Quantifier-free floating point | `qf_fp/` |
| `QF_DT` | Quantifier-free algebraic datatypes | `qf_dt/` |
| `QF_S` | Quantifier-free strings | `qf_s/` |
| `QF_A` | Quantifier-free arrays (generic) | `qf_a/` |

### Logics with Quantifiers

| Logic | Theories | Benchmark directory |
|-------|----------|-------------------|
| `AUFLIA` | Arrays + UF + linear integers + quantifiers | `AUFLIA/` |
| `AUFLIRA` | Arrays + UF + linear integers/reals + quantifiers | `AUFLIRA/` |
| `UFLIA` | UF + linear integers + quantifiers | `UFLIA/` |
| `UFLRA` | UF + linear reals + quantifiers | `UFLRA/` |

---

## Known Limitations vs Z3

| Feature | Z3 | OxiZ Status |
|---------|-----|------------|
| Python operator overloading (`x + y`, `x == y`) | Full support | Not supported — use `tm.mk_*` methods or `assert_formula` |
| Tactic framework (`z3.Tactic`, `z3.Then`, `z3.Or`) | Rich tactic combinators | Limited — CLI exposes strategy options; Python API does not yet expose tactics |
| `z3.Solver.assertions()` | Returns assertion list as Z3 exprs | Not available — assertion introspection is not yet exposed |
| `z3.ForAll` / `z3.Exists` | Python-level quantifier creation | Not available in Python API — use SMT-LIB2 string assertions with `(forall ...)` |
| `z3.FP*` floating-point operations | Full IEEE 754 support | Partial — `QF_FP` benchmarks pass; Python term API for FP not yet exposed |
| Algebraic numbers in models | Z3 returns algebraic number objects | OxiZ returns rational string approximations |
| `z3.Probe` | Formula probing | Not available |
| `z3.ParOr` / parallel tactics | Native parallel tactics | Portfolio solving available via CLI `--parallel` flag |
| Lambda / array lambda | `z3.Lambda` | Not yet supported |
| Recursive functions | `z3.RecFunction` | Not yet supported |
| SMTCOMP-mode output | `--smt2` flag | `--smtcomp` flag on CLI |
| Unsat core production | Automatic | Must enable: `solver.set_option('produce-unsat-cores', 'true')` |
| Model completion for UF | Automatic in Z3 | Partial — may return `unknown` for underspecified UF models |

> **Tip for porting Z3 scripts:** The fastest migration path is to translate
> Python-level formula construction into SMT-LIB2 string calls via
> `assert_formula`, then switch to the native term API incrementally as needed.
