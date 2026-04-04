# OxiZ Common Pitfalls and Solutions

Practical guide to avoiding the most common mistakes when using OxiZ,
with concrete solutions for each issue.

---

## Table of Contents

1. [Timeout Issues](#timeout-issues)
2. [Memory Issues](#memory-issues)
3. [Incorrect or Unexpected Results](#incorrect-or-unexpected-results)
4. [API Misuse](#api-misuse)
5. [Performance Traps](#performance-traps)

---

## Timeout Issues

### Problem: Solver returns Unknown immediately

**Cause:** No logic declared, causing all theories to load.

**Solution:** Always declare the most specific logic:
```rust
use oxiz_solver::Solver;
use oxiz_core::ast::TermManager;

let mut solver = Solver::new();
let mut tm = TermManager::new();
solver.set_logic("QF_LIA"); // Not just "ALL" or omitting this call
```

### Problem: Quantified formula times out

**Cause:** MBQI budget exhausted or no E-matching triggers.

**Solutions:**
1. Increase MBQI budget via `(set-option :mbqi.max_instantiations 10000)`
2. Add `:pattern` triggers to quantified formulas:
```smt2
(assert (forall ((x Int))
  (! (>= (f x) 0) :pattern ((f x)))))
```
3. Consider reformulating in a quantifier-free fragment if possible

### Problem: Nonlinear arithmetic returns Unknown

**Cause:** QF_NIA is incomplete for integers. The NLSAT solver uses CAD
which is complete for reals but not integers.

**Solutions:**
1. If the problem is over reals, use `QF_NRA` instead
2. Add tighter bounds on variables to help search
3. Try linearization: replace `x*y` with a fresh variable `z` and add
   bounds derived from the ranges of `x` and `y`

### Problem: Appropriate timeout values

**Guidelines:**

| Problem class | Suggested timeout |
|---------------|------------------|
| QF_LIA (small, < 100 vars) | 5-10 seconds |
| QF_LIA (large, > 1000 vars) | 60-300 seconds |
| QF_BV (narrow, < 32 bits) | 10-30 seconds |
| QF_BV (wide, 64+ bits) | 60-600 seconds |
| Quantified (AUFLIA) | 60-600 seconds |
| Nonlinear (QF_NIA/NRA) | 120-600 seconds |
| MaxSAT/optimization | 60-1800 seconds |

Set timeouts via:
```rust
use oxiz_solver::SolverConfig;

let mut config = SolverConfig::balanced();
config.timeout_ms = 30_000; // 30 seconds
```

---

## Memory Issues

### Problem: Out of memory on large formulas

**Solutions:**

1. **Disable proof generation** (major memory saving):
```rust
let mut config = SolverConfig::balanced();
config.proof = false;
config.model = true; // Keep model if needed
```

2. **Use incremental solving** to share work:
```rust
// Assert common background once, use push/pop for queries
solver.assert(background, &mut tm);
for query in queries {
    solver.push();
    solver.assert(query, &mut tm);
    let _ = solver.check(&mut tm);
    solver.pop();
}
```

3. **Reduce clause retention**: Lower the clause deletion threshold
   at the SAT level for memory-constrained environments.

4. **Reset between independent queries**: If queries share no state,
   `solver.reset()` frees all internal allocations.

### Problem: Push/pop depth causing memory bloat

**Cause:** Each `push()` snapshots solver state. Deep nesting (> 20 levels)
accumulates significant memory.

**Solution:** Flatten the push/pop structure or use `reset()` + re-assert:
```rust
// Instead of deep nesting:
// solver.push(); solver.push(); solver.push(); ...

// Prefer flat structure:
solver.push();
solver.assert(query, &mut tm);
let _ = solver.check(&mut tm);
solver.pop();
// State is back to baseline, push again for next query
```

### Problem: Bit-vector operations consuming excessive memory

**Cause:** Bit-blasting wide BV operations (64+ bit multiplication)
generates millions of clauses.

**Solutions:**
- Use narrower bit-widths when the full range is not needed
- Factor common BV subexpressions
- For pure comparison/addition, consider QF_LIA with bounded integers

---

## Incorrect or Unexpected Results

### Problem: Solver returns Sat but model seems wrong

**Cause:** Often a formulation error. The solver found a model that
satisfies the constraints as written, but not as intended.

**Debugging steps:**
1. Extract the model and verify each assertion manually
2. Check sort mismatches: an `Int` variable used where `Real` was intended
   produces different semantics for division
3. Check for missing constraints: ensure all intended invariants are asserted

### Problem: Unknown instead of Sat/Unsat

**Common causes and fixes:**

| Cause | Fix |
|-------|-----|
| No logic declared | Add `solver.set_logic(...)` |
| Timeout hit | Increase timeout or simplify formula |
| Resource limit | Increase `max_conflicts` or `max_decisions` |
| Incomplete theory (QF_NIA) | Add bounds, try different encoding |
| MBQI budget exhausted | Increase instantiation budget |

### Problem: Different result than Z3

**Cause:** Both solvers are correct; the difference is usually:
- A timeout in one solver but not the other
- Different default options (Z3 enables some options OxiZ does not)
- Different model completion behavior for underspecified UF

**Debugging:** Test with the same SMT-LIB2 file on both solvers with
explicit timeouts and options to isolate the difference.

### Problem: Theory interaction produces unexpected conflicts

**Cause:** Nelson-Oppen equality propagation between theories can produce
non-obvious conflicts.

**Example:** An EUF equality `f(x) = f(y)` may force `x = y`, which
then conflicts with an arithmetic constraint `x != y`.

**Solution:** Enable verbose/tracing mode to see the conflict explanation
chain and understand which theory propagations led to the conflict.

---

## API Misuse

### Problem: TermManager lifetime issues

**Cause:** Terms (`TermId`) are handles into the `TermManager` arena.
Using a `TermId` from one `TermManager` with a different one is undefined.

**Rule:** Keep one `TermManager` per solving session:
```rust
let mut tm = TermManager::new();
let mut solver = Solver::new();

// All terms built from the same tm
let x = tm.mk_var("x", tm.sorts.int_sort);
let y = tm.mk_var("y", tm.sorts.int_sort);
solver.assert(tm.mk_eq(x, y), &mut tm);
```

### Problem: Forgetting to pass TermManager to check()

**Solution:** The Rust API requires `&mut TermManager` for `check()`:
```rust
// Correct:
let result = solver.check(&mut tm);

// The borrow checker prevents using tm while solver borrows it.
// Do all term construction before or after check().
```

### Problem: Using push/pop asymmetrically

**Cause:** Every `push()` must have a matching `pop()`. Orphaned pushes
leak memory; orphaned pops may panic or corrupt state.

**Rule:** Use RAII-style scoping when possible:
```rust
solver.push();
solver.assert(hypothesis, &mut tm);
let result = solver.check(&mut tm);
solver.pop(); // Always pop, even if check() returned Unknown
```

### Problem: Asserting after check() without push/pop

**Note:** Asserting new constraints after `check()` without `push()`/`pop()`
is valid but makes the solver state monotonically grow. Previous learned
clauses are retained, which is usually beneficial. Use `reset()` only if
you want a completely fresh solver.

---

## Performance Traps

### Trap: Using QF_NIA for linear problems

**Impact:** 10-100x slower than QF_LIA.

The NLSAT solver activates for `QF_NIA`, adding CAD overhead even for
linear constraints. Always use `QF_LIA` for purely linear integer problems.

### Trap: Unnecessary distinct constraints

**Impact:** O(n^2) pairwise inequality clauses.

```smt2
; Slow: creates n*(n-1)/2 clauses
(assert (distinct x1 x2 x3 ... x100))

; Better: if values are bounded, use at-most-one encoding
; or pigeonhole reasoning
```

### Trap: Creating new variables in a loop

**Impact:** Linear growth in solver state per iteration.

If you create fresh variables in each iteration of a CEGAR or enumeration
loop, the solver accumulates dead variables. Use `push()`/`pop()` to
scope temporary variables, or `reset()` between independent queries.

### Trap: Wide bit-vector multiplication

**Impact:** Quadratic clause count in bit-width.

32-bit multiplication creates ~1,000 clauses. 64-bit creates ~4,000.
128-bit creates ~16,000. If you only need the low bits of a product,
extract them early.

### Trap: Unbounded string variables with regex

**Impact:** Potential non-termination.

Always bound string lengths when using regex constraints:
```smt2
(assert (<= (str.len s) 100))  ; Explicit bound
(assert (str.in.re s pattern))
```

### Trap: Forgetting to set logic for FP problems

**Impact:** FP theory not activated, solver returns Unknown.

Floating-point requires `QF_FP`:
```rust
solver.set_logic("QF_FP");
```

### Trap: Excessive push depth in incremental solving

**Impact:** Linear memory growth per push level. For > 20 levels deep,
restructure to flat push/pop or reset-and-reassert for independent queries.
