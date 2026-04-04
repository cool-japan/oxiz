# OxiZ Performance Tuning Guide

This guide explains how to get the best performance from OxiZ across different
problem classes. OxiZ is a multi-theory SMT solver implemented in Pure Rust;
understanding how it routes logic families to specialized sub-solvers is the
single most effective tuning lever available.

---

## Table of Contents

1. [Setting the Right Logic](#setting-the-right-logic)
2. [Logic Family → Solver Dispatch Table](#logic-family--solver-dispatch-table)
3. [Incremental Solving (push/pop) — When to Use, Overhead](#incremental-solving-pushpop)
4. [Timeout Configuration](#timeout-configuration)
5. [MBQI Budget Hints for Quantified Formulas](#mbqi-budget-hints-for-quantified-formulas)
6. [Portfolio Mode](#portfolio-mode)
7. [Reading SLoC Coverage Metrics](#reading-sloc-coverage-metrics)
8. [Benchmark-Specific Tips](#benchmark-specific-tips)

---

## Setting the Right Logic

The `(set-logic ...)` declaration (or `solver.set_logic(...)` in Python) is
the highest-impact single optimization. When no logic is declared OxiZ falls
back to an "ALL" mode that enables every theory simultaneously, which prevents
several important optimizations:

- **Arithmetic dispatch:** Without a logic hint, OxiZ uses the LRA (real
  arithmetic) solver by default, even for integer problems. LIA (integer
  arithmetic) enables the Omega/cutting-planes path, which can be exponentially
  faster on integer-only benchmarks.
- **NLSAT activation:** `QF_NIA` and `QF_NRA` are the only logics that activate
  the nonlinear arithmetic (NLSAT) solver. If you assert nonlinear terms without
  declaring the logic, OxiZ attempts to handle them inside the linear solver and
  will quickly return `unknown`.
- **Theory combination overhead:** Declaring a quantifier-free logic (prefix
  `QF_`) disables the MBQI and E-matching engines entirely, reducing per-check
  overhead significantly.

**SMT-LIB2:**
```smt2
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (>= x 0))
(assert (<= x 100))
(assert (= (+ (* 3 x) (* 7 y)) 42))
(check-sat)
```

**Python:**
```python
import oxiz

tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('QF_LIA')   # <-- critical: selects LIA solver path
```

**Rule of thumb:** Always declare the most specific logic that covers your
problem. Use `QF_` if you have no quantifiers. Use `LIA`/`LRA` not `NIA`/`NRA`
if all arithmetic is linear.

---

## Logic Family → Solver Dispatch Table

The following table shows which internal solver components are activated for
each logic family. Components listed as "disabled" incur no overhead.

| Logic | Arithmetic solver | NLSAT | E-matching | MBQI | Array theory | BV theory | UF |
|-------|------------------|-------|------------|------|-------------|-----------|-----|
| `QF_LIA` / `QF_IDL` | LIA (Omega/CP) | off | off | off | off | off | off |
| `QF_LRA` / `QF_RDL` | LRA (Simplex) | off | off | off | off | off | off |
| `QF_NIA` | LIA + NLSAT | **on** | off | off | off | off | off |
| `QF_NRA` / `QF_NIRA` | LRA + NLSAT | **on** | off | off | off | off | off |
| `QF_BV` / `QF_ABV` / `QF_AUFBV` | LIA (bounded) | off | off | off | conditional | **on** | conditional |
| `QF_A` / `QF_ALIA` / `QF_AUFLIA` | LIA | off | off | off | **on** | off | conditional |
| `QF_UF` / `QF_UFLIA` / `QF_UFLRA` | LIA/LRA | off | off | off | off | off | **on** |
| `QF_FP` | FP (IEEE 754) | off | off | off | off | off | off |
| `QF_DT` | LIA | off | off | off | off | off | off |
| `QF_S` | String (re-based) | off | off | off | off | off | off |
| `AUFLIA` / `UFLIA` | LIA | off | **on** | **on** | **on** | off | **on** |
| `AUFLIRA` / `UFLRA` | LIA+LRA | off | **on** | **on** | **on** | off | **on** |
| `ALL` (default) | LRA | off | **on** | **on** | **on** | **on** | **on** |

**Key takeaway:** Quantifier-free logics are handled by specialized, highly
optimized paths. Quantified logics (`AUFLIA`, `AUFLIRA`, `UFLIA`, `UFLRA`)
enable both E-matching and MBQI, which are much more expensive and may time out
on large instances.

---

## Incremental Solving (push/pop)

### When to use push/pop

- **Assume-guarantee reasoning:** Assert shared background theory once, then
  use `push`/`pop` to check many different hypotheses against the same base.
- **All-SAT enumeration:** After finding a model, push, assert the negation of
  the current model, check again, pop, repeat.
- **CEGAR loops:** Push before adding a candidate lemma; if unsatisfiable, pop
  and try a different refinement.

```python
import oxiz

tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('QF_LIA')

# Shared background
x = tm.mk_var('x', 'Int')
s.assert_term(tm.mk_ge(x, tm.mk_int(0)), tm)
s.assert_term(tm.mk_le(x, tm.mk_int(100)), tm)

# Query 1
s.push()
s.assert_term(tm.mk_gt(x, tm.mk_int(90)), tm)
r1 = s.check_sat(tm)     # Sat
s.pop()

# Query 2 — shared constraints still in place
s.push()
s.assert_term(tm.mk_lt(x, tm.mk_int(5)), tm)
r2 = s.check_sat(tm)     # Sat
s.pop()
```

### Overhead considerations

Each `push()` snapshots the current solver state. In OxiZ this includes:

- The DPLL(T) trail and assignment stack
- The arithmetic solver's tableau
- The congruence closure graph (for UF logics)
- MBQI instantiation history (for quantified logics)

**Minimize push depth:** Deeply nested push/pop trees (depth > 20) can cause
significant memory pressure because each level retains a full state snapshot.
If you need to test many independent queries, prefer calling `reset()` and
re-asserting the common background rather than accumulating push levels.

**Avoid pushing inside hot loops** where the same constraint is asserted and
popped thousands of times. Instead, use the assumption literal pattern:
conditionally assert a Boolean selector variable and let the solver handle
conditional reasoning internally.

---

## Timeout Configuration

### CLI

```bash
# Timeout in seconds
oxiz solve --timeout 30 problem.smt2

# Portfolio mode with per-solver timeout
oxiz solve --parallel --portfolio-timeout 10 problem.smt2
```

### SMT-LIB2

```smt2
(set-option :timeout 30000)   ; milliseconds
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(check-sat)
```

### Python

```python
import oxiz

s = oxiz.Solver()
s.set_option('timeout', '30000')   ; milliseconds as string
```

### How timeout interacts with the solver

When a timeout fires, `check_sat()` returns `SolverResult.Unknown`. The solver
state is preserved — you can still call `get_model()` (it will return an empty
dict) and continue with `push()`/`pop()`. Calling `check_sat()` again after a
timeout will restart the search from scratch unless incremental state was
preserved via `push()`.

For quantified logics, the MBQI instantiation budget (see next section) provides
a softer form of resource bounding that is preferable to hard timeouts when you
want partial results.

---

## MBQI Budget Hints for Quantified Formulas

MBQI (Model-Based Quantifier Instantiation) is OxiZ's primary engine for
deciding formulas with universal quantifiers. It works by:

1. Constructing a candidate model ignoring quantifiers.
2. Checking whether the model satisfies all quantifiers.
3. If not, extracting witnesses from the model and instantiating the quantifier
   body with those witnesses as new ground lemmas.
4. Repeating until a satisfying model is found or the budget is exhausted.

### Default budget

The default instantiation budget is **1,000 instantiations per quantifier
instance**. For problems with many quantifiers or long instantiation chains this
budget is often insufficient.

### Adjusting via CLI option

```bash
# Allow up to 10,000 MBQI instantiation steps
oxiz solve --set-option mbqi.max_instantiations=10000 quantified.smt2
```

### SMT-LIB2

```smt2
(set-option :mbqi.max_instantiations 10000)
(set-logic AUFLIA)
(declare-fun f (Int) Int)
(assert (forall ((x Int)) (>= (f x) 0)))
(assert (= (f 5) 3))
(check-sat)
```

### Python

MBQI budget is not yet directly exposed in the Python binding. Use the CLI or
embed SMT-LIB2 option strings.

### Tuning strategy

| Problem characteristic | Recommended budget |
|------------------------|-------------------|
| Few quantifiers (< 5), shallow structure | 500–1,000 (default) |
| Many quantifiers (5–20), linear bodies | 2,000–5,000 |
| Deeply nested quantifiers, nonlinear bodies | 10,000+ or switch to E-matching only |
| Universally quantified arithmetic (LIA/LRA) | Use MBQI; E-matching alone is weak |
| Quantified UF with known triggers | Lower MBQI budget, rely on E-matching patterns |

When MBQI exceeds its budget and returns `unknown`, inspect whether:
- The formula has quantifier alternation (∀∃) which makes it undecidable in
  general.
- The witnesses being generated form an infinite chain (a sign of a
  non-stratified formula — see the Quantifier Handling Guide).
- The problem can be reformulated in a quantifier-free fragment.

---

## Portfolio Mode

OxiZ's CLI supports a portfolio solver that runs multiple solver configurations
in parallel and returns the first answer:

```bash
oxiz solve --parallel --threads 4 problem.smt2
```

The portfolio spawns independent solver threads with different:
- Restart strategies (frequent / moderate / rare)
- Branching heuristics (VSIDS / LRB)
- Simplification levels
- Lookahead settings

Portfolio mode is most effective for:
- Hard combinatorial problems where the right strategy is unknown
- Timeouts caused by unlucky variable ordering in a single configuration
- SMT-COMP style benchmarking

Portfolio mode is **not recommended** for:
- Interactive workflows where determinism matters
- Programs that call `push()`/`pop()` extensively (only single-threaded mode
  supports incremental solving)
- Memory-constrained environments (each thread maintains an independent solver
  state)

---

## Reading SLoC Coverage Metrics

OxiZ coverage of the Z3 feature set can be estimated from the SLoC of each
subcrate relative to Z3's C++ implementation. Use `tokei` to inspect the current
state:

```bash
tokei /path/to/oxiz/
```

Key subcrates and what they cover:

| Crate | Primary coverage area |
|-------|-----------------------|
| `oxiz-core` | AST, term management, E-matching, sort system |
| `oxiz-solver` | DPLL(T) core, theory combination, MBQI, model extraction |
| `oxiz-theories` | Arithmetic (LIA/LRA/NLSAT), BV, arrays, UF, datatypes, FP, strings |
| `oxiz-sat` | SAT solver (CDCL, portfolio, parallel) |
| `oxiz-proof` | Proof logging, unsat core extraction, proof checking |
| `oxiz-opt` | Optimization (OMT, MaxSAT, Pareto enumeration) |
| `oxiz-spacer` | PDR/IC3 for CHC solving |
| `oxiz-nlsat` | Nonlinear arithmetic over reals/integers (cylindrical algebraic) |

A higher SLoC count in a subcrate relative to Z3's equivalent module generally
indicates deeper coverage — but review the benchmark parity results in
`bench/z3_parity/` for ground-truth functional equivalence data.

---

## Benchmark-Specific Tips

The benchmark suite in `bench/z3_parity/benchmarks/` is organized by logic
family. Each category tests specific solver paths:

### `qf_lia/` — Linear Integer Arithmetic

```smt2
; Good: declare logic, use simple linear constraints
(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 0))
(assert (<= (* 3 x) 99))
(check-sat)
```

Tuning: Prefer `QF_LIA` over `QF_NIA` for linear problems. The LIA path uses
the Omega test and cutting planes, which are complete for linear integer
arithmetic without requiring NLSAT.

### `qf_bv/` — Bitvectors

```smt2
; Good: declare width-matching operations
(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(assert (= (bvand x #xff) #x0f))
(check-sat)
```

Tuning: Keep bit widths as small as possible. Bit-blasting cost scales linearly
with width and can be superlinear for multiplication. For widths > 64, consider
whether your problem can be modeled using `QF_LIA` with explicit modular
arithmetic constraints.

### `qf_lra/` — Linear Real Arithmetic

```smt2
; Good: Simplex is very fast for sparse systems
(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)
(assert (>= (+ x y) 1.0))
(assert (<= x 0.5))
(check-sat)
```

Tuning: LRA via the Simplex method handles thousands of variables efficiently.
Avoid unnecessary `distinct` constraints on reals; they expand to quadratic
pairwise inequality sets.

### `AUFLIA/` and `AUFLIRA/` — Quantified Array + Arithmetic

```smt2
; Add patterns to guide E-matching when possible
(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const n Int)
(assert (forall ((i Int))
  (! (=> (and (>= i 0) (< i n))
         (>= (select a i) 0))
     :pattern ((select a i)))))   ; trigger pattern
(check-sat)
```

Tuning: Always provide `:pattern` triggers for quantifiers over arrays. Without
triggers, E-matching cannot fire and MBQI must enumerate model-based witnesses,
which is expensive. See the Quantifier Handling Guide for details.

### `qf_s/` — Strings

String solving is handled by a regular-expression-based theory. Performance
degrades with long string lengths and complex regex intersections. Prefer
concrete string length bounds when possible:

```smt2
(set-logic QF_S)
(declare-const s String)
(assert (str.in.re s (re.* (str.to.re "ab"))))
(assert (= (str.len s) 4))
(check-sat)
```
