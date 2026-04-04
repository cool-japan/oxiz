# OxiZ Quantifier Handling Guide

This guide covers how OxiZ handles universally and existentially quantified
formulas, when each instantiation engine fires, and how to write formulas that
the solver can handle efficiently.

---

## Table of Contents

1. [Overview: Two Instantiation Engines](#overview-two-instantiation-engines)
2. [How MBQI Works in OxiZ](#how-mbqi-works-in-oxiz)
3. [How E-matching Works in OxiZ](#how-e-matching-works-in-oxiz)
4. [When MBQI Fires vs E-matching](#when-mbqi-fires-vs-e-matching)
5. [Writing Quantifier-Friendly Formulas](#writing-quantifier-friendly-formulas)
6. [Triggers and Patterns (`:pattern`, `:no-pattern`)](#triggers-and-patterns)
7. [Common Pitfalls](#common-pitfalls)
8. [Translating Quantified Z3 Python Formulas](#translating-quantified-z3-python-formulas)
9. [Benchmark Reference](#benchmark-reference)

---

## Overview: Two Instantiation Engines

OxiZ uses two complementary engines to handle quantified formulas. Both operate
inside the DPLL(T) framework and communicate with the SAT solver through a theory
callback interface.

| Engine | When active | Primary strength |
|--------|-------------|-----------------|
| **E-matching** | Always when quantifiers are present | Speed: fires on concrete ground terms |
| **MBQI** | When E-matching is insufficient or no triggers exist | Completeness: model-driven, can find witnesses without syntactic triggers |

The two engines are not mutually exclusive — OxiZ runs E-matching first on each
check iteration and falls back to MBQI when E-matching produces no new
instantiations.

---

## How MBQI Works in OxiZ

MBQI (Model-Based Quantifier Instantiation) is implemented in
`oxiz-solver/src/mbqi/`. The algorithm follows Ge & de Moura (CAV 2009):

### Step-by-step algorithm

1. **Candidate model extraction:** The DPLL(T) core asks the SAT solver for a
   partial propositional model. Theory solvers (arithmetic, arrays, UF) are
   asked to complete it to a full first-order model `M`.

2. **Model completion:** The `model_completion` module handles underspecified
   uninterpreted functions by assigning arbitrary values consistent with the
   current equalities. Projection functions map values in potentially infinite
   domains to a finite representative set.

3. **Quantifier specialization:** For each universal quantifier `∀x.φ(x)`, the
   engine evaluates `φ` under `M` by substituting representatives from the
   finite model for `x`. This is the "counterexample search" step.

4. **Counterexample extraction:** If a witness `x'` is found such that `¬φ(x')`
   holds under `M`, the ground instance `φ(x')` is added as a new lemma to the
   clause database. This restricts the search space for the next iteration.

5. **Refinement loop:** The solver re-checks satisfiability with the new lemma.
   If the model can no longer be constructed without satisfying all quantifiers,
   the result is `sat`. If a contradiction is derived, the result is `unsat`.
   If the budget is exhausted, the result is `unknown`.

### MBQI sub-modules in OxiZ

| Sub-module | Responsibility |
|------------|---------------|
| `mbqi::model_completion` | Completes partial UF models; handles uninterpreted sorts |
| `mbqi::counterexample` | Generates witnesses that falsify quantifier bodies under the model |
| `mbqi::finite_model` | Finite model finder with bounded universe enumeration and symmetry breaking |
| `mbqi::lazy_instantiation` | Generates instantiations on demand rather than eagerly |
| `mbqi::instantiation` | Core instantiation engine; manages the budget counter |
| `mbqi::heuristics` | Selection strategies (which quantifiers to instantiate first) |
| `mbqi::patterns` | Pattern extraction and relevancy filtering |
| `mbqi::integration` | Callback interface that connects MBQI to the main DPLL(T) loop |

### Budget tracking

The `InstantiationContext` struct tracks a `max_instantiations` limit (default
1,000). Each time a witness is instantiated, the counter increments. When the
limit is reached, MBQI stops and reports `unknown` rather than risking an
infinite loop.

---

## How E-matching Works in OxiZ

E-matching is implemented in `oxiz-core/src/ematching/`. It is a syntactic
matching algorithm that fires when ground terms in the current formula match
the trigger patterns of a universally quantified formula.

### Key components

| Component | Role |
|-----------|------|
| `ematching::code_tree` | Compiled instruction tree for fast pattern matching |
| `ematching::index` | Inverted term index mapping function symbols to ground terms |
| `ematching::mod_time` | Tracks which terms are "new" since the last matching round, avoiding redundant work |
| `ematching::trigger` | Trigger (pattern) extraction heuristics |
| `ematching::multi_pattern` | Optimized matching when a quantifier has multiple trigger terms |
| `ematching::relevancy` | Relevancy propagation — only instantiates on terms likely to be useful |
| `ematching::quantifier_inst` | Deduplicates identical instantiations |

### How E-matching fires

Given:
```smt2
(assert (forall ((x Int)) (! (>= (f x) 0) :pattern ((f x)))))
(assert (= (f 5) 3))
```

E-matching fires when the ground term `(f 5)` appears in the clause database.
It matches the pattern `(f x)` with substitution `x ↦ 5`, and instantiates
the quantifier body as the lemma `(>= (f 5) 0)`, which resolves the query.

Without the `:pattern` annotation, OxiZ uses heuristic trigger selection
(`ematching::trigger`) to infer patterns from the quantifier body.

---

## When MBQI Fires vs E-matching

```
Check iteration:
  1. Run E-matching round
     ├─ New instantiations produced? → add lemmas → restart SAT → goto 1
     └─ No new instantiations?
          2. Run MBQI round
             ├─ Counterexample found? → add lemma → restart SAT → goto 1
             ├─ No counterexample? → model satisfies all quantifiers → SAT
             └─ Budget exhausted? → UNKNOWN
```

**E-matching fires first** because it is fast (indexed lookup) and precise
(only fires on exact syntactic matches). It is the preferred engine for:
- Formulas with function applications where the function appears in ground terms
- Quantifiers with explicit `:pattern` triggers
- Repetitive instantiation over a known finite domain

**MBQI fires second** (or exclusively) when:
- No ground terms match any trigger pattern
- The quantifier body is purely arithmetic (no function symbol to use as anchor)
- The formula requires witnesses not currently in the ground term set
- E-matching has been exhausted and the current model still fails quantifiers

---

## Writing Quantifier-Friendly Formulas

### Use quantifier-free alternatives when possible

If your problem can be expressed without quantifiers, always prefer the
quantifier-free formulation. Compare:

**Quantified (slow):**
```smt2
(set-logic AUFLIA)
(declare-const a (Array Int Int))
(assert (forall ((i Int)) (>= (select a i) 0)))
(assert (= (select a 7) 5))
(check-sat)
```

**Quantifier-free (fast — use QF_ALIA and concrete selects):**
```smt2
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(assert (>= (select a 7) 0))
(assert (= (select a 7) 5))
(check-sat)
```

### Stratify your quantifiers

A formula is **stratified** if no quantified variable can be substituted back
into a position that generates a new instance of the same quantifier. Non-
stratified formulas can cause MBQI to loop indefinitely.

**Stratified (terminating):**
```smt2
; f : Int -> Int, g : Int -> Int; f feeds g but not the reverse
(assert (forall ((x Int)) (>= (g (f x)) 0)))
```

**Non-stratified (may diverge):**
```smt2
; f : Int -> Int; f's output feeds back into f's input
(assert (forall ((x Int)) (>= (f (f x)) (f x))))
```

For non-stratified formulas, either:
- Add a `:pattern` that avoids the recursive case
- Bound the depth explicitly: `(assert (forall ((x Int)) (=> (and (>= x 0) (< x 100)) ...)))`

### Prefer shallow nesting

Deeply nested quantifiers (`∀x.∀y.∀z.φ`) require combinatorial instantiation.
Each level multiplies the number of required witnesses. Flatten when possible:

```smt2
; Instead of deeply nested:
; (forall ((x Int)) (forall ((y Int)) (forall ((z Int)) (= (f x y z) 0))))

; Prefer multi-variable binding (single quantifier prefix):
(forall ((x Int) (y Int) (z Int)) (= (f x y z) 0))
```

The multi-variable form is semantically equivalent but allows MBQI to search for
witnesses jointly rather than iterating three independent loops.

### Use arithmetic bounds on quantified variables

Unbounded quantifiers over infinite domains (`∀x : Int`) require MBQI to search
an infinite space. Bounding the search region dramatically reduces counterexample
search time:

```smt2
; Bounded: MBQI checks finitely many witnesses
(assert (forall ((i Int))
  (=> (and (>= i 0) (< i 100))
      (>= (select a i) 0))))
```

```smt2
; Unbounded: MBQI may diverge or time out
(assert (forall ((i Int)) (>= (select a i) 0)))
```

---

## Triggers and Patterns

OxiZ's E-matching engine reads `:pattern` and `:no-pattern` annotations from
SMT-LIB2 formulas. These are the primary mechanism for guiding instantiation.

### `:pattern` — positive trigger

A `:pattern` annotation specifies the ground term shape that must appear before
the quantifier is instantiated. The pattern must contain all bound variables.

```smt2
(set-logic AUFLIA)
(declare-fun f (Int) Int)
(declare-fun g (Int) Int)

; Pattern: (f x) — E-matching fires when any ground (f c) appears
(assert (forall ((x Int))
  (! (=> (>= (f x) 0) (>= (g x) 0))
     :pattern ((f x)))))

(assert (= (f 3) 5))   ; triggers instantiation with x=3
(check-sat)
```

### Multi-pattern triggers

When a single term cannot cover all bound variables, use a multi-pattern (a
comma-separated set of terms that must all appear simultaneously):

```smt2
(assert (forall ((x Int) (y Int))
  (! (= (f x y) (f y x))
     :pattern ((f x y) (f y x)))))   ; both must appear
```

### `:no-pattern` — excluded trigger

Prevents a specific term from being used as a trigger. Useful when heuristic
trigger selection picks a term that causes excessive instantiation:

```smt2
(assert (forall ((x Int))
  (! (>= (f (g x)) 0)
     :no-pattern ((g x)))))  ; don't trigger on (g x), it's too generic
```

### Heuristic trigger selection

When no `:pattern` is provided, OxiZ uses the `ematching::trigger` module to
infer triggers. The heuristic prefers:

1. Terms that contain all bound variables
2. Terms with function symbols that appear in ground assertions
3. Terms that are maximal (not a strict sub-term of another pattern)

If no good heuristic trigger exists, OxiZ falls back to MBQI for that quantifier.

### Trigger priority

When multiple triggers are present, OxiZ fires on the first match and avoids
duplicate instantiations via `ematching::quantifier_inst`'s deduplication table.
The `mod_time` optimization ensures that only terms modified since the last
matching round are re-checked, keeping per-iteration cost linear in the number
of new ground terms rather than in the total term count.

---

## Common Pitfalls

### 1. Quantifier over an unbounded array domain without a trigger

```smt2
; Problem: no (f x) term exists in the ground context to trigger E-matching
(declare-const a (Array Int Int))
(assert (forall ((i Int)) (>= (select a i) 0)))
```

Fix: Add a trigger or use MBQI with bounds:
```smt2
(assert (forall ((i Int))
  (! (>= (select a i) 0)
     :pattern ((select a i)))))
(assert (= (select a 0) 5))   ; now E-matching fires with i=0
```

### 2. Quantifier alternation (∀∃)

OxiZ's MBQI handles `∀x.∃y.φ(x,y)` by Skolemization: the existential is
replaced with a Skolem function `f_y(x)` and the formula becomes
`∀x.φ(x, f_y(x))`. This introduces an uninterpreted function, shifting the
problem into `AUFLIA`/`AUFLIRA`. Deep alternation (`∀∃∀∃...`) is in general
undecidable and will eventually exhaust the MBQI budget.

```smt2
; Acceptable: single alternation, bounded
(set-logic AUFLIA)
(assert (forall ((x Int))
  (exists ((y Int))
    (and (> y x) (= (f y) 0)))))
```

### 3. Non-linear bodies with quantifiers

Quantified formulas with nonlinear arithmetic in the body are handled by MBQI
but not by E-matching alone. Expected behavior: MBQI will attempt to find
witnesses, but may time out on complex nonlinear instances.

```smt2
; This is in NIA (nonlinear) territory — MBQI only, no E-matching trigger
(assert (forall ((x Int)) (>= (* x x) 0)))
```

Workaround: Use `QF_NIA` (dropping the quantifier) if the bound domain is
small enough to enumerate explicitly.

### 4. Nested quantifiers with shared variables

```smt2
; x appears in both the outer and inner binder — shadowing
(assert (forall ((x Int))
  (forall ((x Int))   ; shadows outer x — almost certainly a mistake
    (= (f x) 0))))
```

OxiZ follows standard lexical scoping, so the inner `x` shadows the outer one.
Use distinct variable names to avoid confusion.

### 5. Universally quantified Boolean variables

```smt2
; This is valid SMT-LIB2 but OxiZ may return unknown for complex bodies
(assert (forall ((p Bool)) (or p (not p))))  ; tautology, but triggers MBQI
```

For pure propositional tautologies, assert them as ground facts without
quantification.

---

## Translating Quantified Z3 Python Formulas

Z3's Python API has first-class support for quantifier creation. OxiZ's Python
API does not yet expose quantifier term constructors directly; quantified
formulas must be expressed as SMT-LIB2 strings via `assert_formula`.

### Z3: ForAll

**Z3 Python:**
```python
import z3

f = z3.Function('f', z3.IntSort(), z3.IntSort())
x = z3.Int('x')

s = z3.Solver()
s.add(z3.ForAll([x], f(x) >= 0))
s.add(f(z3.IntVal(5)) == 3)
print(s.check())   # sat
```

**OxiZ Python (via SMT-LIB2 strings):**
```python
import oxiz

tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('UFLIA')

# Declare uninterpreted function
s.assert_formula('(declare-fun f (Int) Int)', tm)

# Universal quantifier with explicit pattern
s.assert_formula(
    '(assert (forall ((x Int)) (! (>= (f x) 0) :pattern ((f x)))))',
    tm
)
s.assert_formula('(assert (= (f 5) 3))', tm)

result = s.check_sat(tm)
print(result)   # SolverResult.Sat
model = s.get_model(tm)
```

### Z3: Exists

**Z3 Python:**
```python
x = z3.Int('x')
s = z3.Solver()
s.add(z3.Exists([x], z3.And(x > 0, x < 10)))
print(s.check())   # sat
```

**OxiZ Python:**
```python
s.assert_formula('(assert (exists ((x Int)) (and (> x 0) (< x 10))))', tm)
result = s.check_sat(tm)   # Sat
```

### Z3: ForAll with pattern

**Z3 Python:**
```python
f = z3.Function('f', z3.IntSort(), z3.IntSort())
x = z3.Int('x')
pat = z3.PatternRef(z3.MultiPattern(f(x)))

s.add(z3.ForAll([x], f(x) >= 0, patterns=[f(x)]))
```

**OxiZ Python:**
```python
s.assert_formula(
    '(assert (forall ((x Int)) (! (>= (f x) 0) :pattern ((f x)))))',
    tm
)
```

### Z3: Array quantifier (initialization check)

This pattern from `bench/z3_parity/benchmarks/AUFLIA/array_forall_init.smt2`:

**Z3 Python:**
```python
a = z3.Array('a', z3.IntSort(), z3.IntSort())
n = z3.Int('n')
i = z3.Int('i')

s = z3.Solver()
s.add(n == 5)
s.add(z3.ForAll([i],
    z3.Implies(z3.And(i >= 0, i < n), z3.Select(a, i) == 0)))
print(s.check())   # sat
```

**OxiZ Python:**
```python
import oxiz

tm = oxiz.TermManager()
s = oxiz.Solver()
s.set_logic('AUFLIA')

s.assert_formula('(declare-const a (Array Int Int))', tm)
s.assert_formula('(declare-const n Int)', tm)
s.assert_formula('(assert (= n 5))', tm)
s.assert_formula(
    '''(assert (forall ((i Int))
         (! (=> (and (>= i 0) (< i n))
                (= (select a i) 0))
            :pattern ((select a i)))))''',
    tm
)
s.assert_formula('(assert (= (select a 0) 0))', tm)
s.assert_formula('(assert (= (select a 4) 0))', tm)

result = s.check_sat(tm)
print(result)   # SolverResult.Sat
```

### Z3: Uninterpreted function monotonicity

**Z3 Python:**
```python
f = z3.Function('f', z3.IntSort(), z3.IntSort())
x, y = z3.Ints('x y')

s = z3.Solver()
# forall x y. x <= y => f(x) <= f(y)
s.add(z3.ForAll([x, y],
    z3.Implies(x <= y, f(x) <= f(y))))
s.add(f(z3.IntVal(3)) > f(z3.IntVal(5)))   # contradicts monotonicity
print(s.check())   # unsat
```

**OxiZ (SMT-LIB2 file for CLI):**
```smt2
(set-logic UFLIA)
(declare-fun f (Int) Int)

; Monotonicity: forall x y. x <= y => f(x) <= f(y)
; Pattern: use both (f x) and (f y) as multi-pattern
(assert (forall ((x Int) (y Int))
  (! (=> (<= x y) (<= (f x) (f y)))
     :pattern ((f x) (f y)))))

; Contradiction: f(3) > f(5)
(assert (> (f 3) (f 5)))

(check-sat)
; expected: unsat
```

---

## Benchmark Reference

The following benchmark files in `bench/z3_parity/benchmarks/` demonstrate
quantifier handling scenarios directly:

| File | Logic | Quantifier pattern |
|------|-------|--------------------|
| `AUFLIA/array_forall_init.smt2` | AUFLIA | `∀i. 0≤i<n ⇒ a[i]=0` |
| `AUFLIA/array_search.smt2` | AUFLIA | Search result quantification |
| `AUFLIA/array_sorted.smt2` | AUFLIA | Sortedness invariant `∀i. a[i]≤a[i+1]` |
| `AUFLIA/array_permutation.smt2` | AUFLIA | Permutation witness quantifier |
| `AUFLIA/array_unique.smt2` | AUFLIA | Uniqueness: `∀i≠j. a[i]≠a[j]` |
| `AUFLIRA/auflira_quantified.smt2` | AUFLIRA | `∀x. f(x)≥0` with contradiction |
| `UFLIA/` | UFLIA | UF with quantified integer constraints |
| `UFLRA/` | UFLRA | UF with quantified real constraints |

Run these benchmarks via the CLI to observe solver behavior:

```bash
oxiz solve --verbose bench/z3_parity/benchmarks/AUFLIA/array_forall_init.smt2
oxiz solve --verbose bench/z3_parity/benchmarks/AUFLIRA/auflira_quantified.smt2
```

The `--verbose` flag reports MBQI instantiation counts and E-matching match
rounds, which is useful for diagnosing performance problems with quantified
formulas.
