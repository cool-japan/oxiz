# OxiZ Theory-Specific Guide

Detailed guide covering each theory solver in OxiZ: capabilities, performance
characteristics, formulation tips, and common pitfalls.

---

## Table of Contents

1. [Theory Architecture](#theory-architecture)
2. [Integer Arithmetic (LIA)](#integer-arithmetic-lia)
3. [Real Arithmetic (LRA)](#real-arithmetic-lra)
4. [Bit-Vectors (BV)](#bit-vectors-bv)
5. [Arrays](#arrays)
6. [Strings](#strings)
7. [Floating-Point (FP)](#floating-point-fp)
8. [Algebraic Datatypes (DT)](#algebraic-datatypes-dt)
9. [Nonlinear Arithmetic (NIA/NRA)](#nonlinear-arithmetic-nianra)
10. [Difference Logic (IDL/RDL)](#difference-logic-idlrdl)
11. [Uninterpreted Functions (UF)](#uninterpreted-functions-uf)
12. [Theory Combination](#theory-combination)

---

## Theory Architecture

All theory solvers implement the `Theory` trait:

```rust
// Each theory provides:
// - assert_literal: Process a new Boolean assignment
// - propagate: Deduce implied literals
// - check: Verify theory consistency
// - explain: Generate conflict explanations
// - backtrack: Undo state on backtracking
```

Theory solvers are combined via the Nelson-Oppen method for disjoint theories,
with shared term detection for overlapping signatures.

---

## Integer Arithmetic (LIA)

**Logic:** `QF_LIA`, `QF_IDL`, `UFLIA`, `AUFLIA`

### Solver Components

- **Simplex-based relaxation**: Solves the rational relaxation first
- **Omega test**: Complete decision procedure for Presburger arithmetic
- **Cutting planes**: Branch-and-cut for larger systems
- **Difference logic specialization**: Bellman-Ford for x - y <= c constraints

### Strengths

- Complete for quantifier-free linear integer arithmetic
- Efficient handling of bounded integer variables
- Specialized paths for common patterns (IDL, UTVPI)

### Formulation Tips

```smt2
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

; Good: simple linear constraints
(assert (>= (+ (* 3 x) (* 2 y)) 10))
(assert (<= x 100))

; Good: use div/mod for divisibility
(assert (= (mod x 3) 0))
```

- Prefer `QF_LIA` over `QF_NIA` when all arithmetic is linear
- Use `div` and `mod` for divisibility constraints instead of auxiliary variables
- Bound variables explicitly when possible; unbounded integers require more
  expensive reasoning
- For systems of the form x - y <= c, use `QF_IDL` for Bellman-Ford optimization

### Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Feasibility (bounded) | Polynomial (Simplex) | Fast for sparse systems |
| Integer feasibility | NP-complete | Omega test + branch-and-cut |
| Divisibility (mod/div) | Moderate | Expanded to linear constraints |

### Common Pitfalls

- Forgetting to declare `QF_LIA` causes OxiZ to use LRA (reals), which
  may return rational solutions that do not satisfy integer constraints
- Large coefficients (> 10^9) can cause numerical precision issues in the
  Simplex tableau; consider normalizing constraints
- `distinct` on many integers creates O(n^2) inequality pairs

---

## Real Arithmetic (LRA)

**Logic:** `QF_LRA`, `QF_RDL`, `UFLRA`, `AUFLIRA`

### Solver Components

- **Dual Simplex**: Primary decision procedure
- **Bland's rule**: Anti-cycling pivot selection
- **Bound propagation**: Derives tighter bounds from constraints

### Strengths

- Polynomial-time decision procedure (Simplex is practical polynomial)
- Handles thousands of variables and constraints efficiently
- Exact rational arithmetic (no floating-point rounding)

### Formulation Tips

```smt2
(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

; Simplex excels at sparse constraint systems
(assert (>= (+ x y) 1.0))
(assert (<= (- x y) 0.5))
(assert (>= x 0.0))
```

- Simplex performance depends on constraint matrix density, not variable count
- Avoid `distinct` on reals: expands to O(n^2) disequalities
- For optimization over reals, the Simplex method naturally produces optimal
  vertices; combine with OxiZ's optimizer

### Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Feasibility | Polynomial | Dual Simplex, very fast in practice |
| Disequalities | Linear per disequality | Checked after Simplex |
| Optimization | Polynomial | Natural extension of Simplex |

### Common Pitfalls

- Mixing integer and real variables without declaring `AUFLIRA` or similar
  mixed logic can lead to incomplete results
- Very large rational coefficients (from user input) can slow Simplex pivots
- Strict inequalities (< , >) require infinitesimal handling; prefer
  non-strict (<=, >=) when semantically equivalent

---

## Bit-Vectors (BV)

**Logic:** `QF_BV`, `QF_ABV`, `QF_AUFBV`

### Solver Components

- **Eager bit-blasting**: Converts BV constraints to propositional logic
- **Word-level propagation**: Propagates at the word level before bit-blasting
- **Constant folding**: Evaluates constant BV expressions at compile time

### Strengths

- Complete decision procedure for fixed-width bit-vectors
- Efficient for hardware verification and protocol analysis
- Supports all SMT-LIB2 BV operations (arithmetic, bitwise, shifts, comparisons)

### Formulation Tips

```smt2
(set-logic QF_BV)
(declare-const x (_ BitVec 32))
(declare-const y (_ BitVec 32))

; Good: explicit width matching
(assert (= (bvadd x y) (_ bv100 32)))

; Good: use extract for bit-field access
(assert (= ((_ extract 7 0) x) #x0f))

; Avoid: unnecessary wide bit-vectors
; Use the minimum width that covers your value range
```

- Keep bit-widths as small as possible; cost scales linearly with width
  and superlinearly for multiplication
- For widths > 64, consider QF_LIA with explicit modular arithmetic
- Use `bvand`/`bvor` masks instead of complex conditional logic when possible
- Factor repeated subexpressions to reduce the bit-blasted formula size

### Performance Characteristics

| Operation | Cost (relative to width w) | Notes |
|-----------|---------------------------|-------|
| Comparison, add, sub | O(w) | Linear encoding |
| Multiplication | O(w^2) | Quadratic; avoid wide multiplies |
| Division/remainder | O(w^2) | Expensive; bound when possible |
| Shift by constant | O(w) | Variable shifts are O(w log w) |
| Concat/extract | O(1) | Just index remapping |

### Common Pitfalls

- BV multiplication of two 64-bit values creates ~4,096 clauses;
  128-bit multiplication creates ~16,384 clauses
- Signed vs unsigned operations (`bvslt` vs `bvult`) have different encodings;
  mixing them inadvertently can cause unexpected results
- Overflow semantics in BV are modular (wrap-around), not undefined

---

## Arrays

**Logic:** `QF_A`, `QF_ALIA`, `QF_ABV`, `QF_AUFLIA`, `QF_AUFBV`

### Solver Components

- **Read-over-write axioms**: `select(store(a, i, v), i) = v`
- **Extensionality**: Two arrays are equal iff they agree on all indices
- **Eager expansion**: Instantiates axioms on-demand as new read/write terms appear

### Formulation Tips

```smt2
(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const i Int)
(declare-const v Int)

; Read-over-write: reading what was just written
(assert (= (select (store a i v) i) v))  ; Always true by axiom

; Reading a different index
(declare-const j Int)
(assert (not (= i j)))
(assert (= (select (store a i v) j) (select a j)))  ; Axiom
```

- Minimize the number of distinct array terms; each new `store` creates
  a new array term that must be tracked
- For quantified array properties, always provide `:pattern` triggers:

```smt2
(assert (forall ((k Int))
  (! (=> (and (>= k 0) (< k n))
         (>= (select a k) 0))
     :pattern ((select a k)))))
```

- Without patterns, MBQI must enumerate witnesses, which is expensive

### Common Pitfalls

- Array extensionality can be expensive: proving two arrays equal requires
  showing agreement on all indices
- Nested arrays (`Array Int (Array Int Int))`) multiply the axiom count
- Large constant arrays with many stores degrade to O(n) lookups per read

---

## Strings

**Logic:** `QF_S`

### Solver Components

- **Word equation solver**: Handles string equality and concatenation
- **Brzozowski derivatives**: Regular expression membership via derivative computation
- **Length reasoning**: Integrates with arithmetic for `str.len` constraints
- **Unicode support**: Full Unicode character handling

### Formulation Tips

```smt2
(set-logic QF_S)
(declare-const s String)

; Good: bound string length for performance
(assert (= (str.len s) 4))
(assert (str.in.re s (re.* (str.to.re "ab"))))

; Good: use str.contains for substring checks
(assert (str.contains s "bc"))
```

- Always bound string lengths when possible; unbounded strings can cause
  divergence in the word equation solver
- Prefer `str.contains` over complex regex when checking substring membership
- Regex intersection is expensive; minimize the number of intersected patterns
- `str.replace` and `str.replace_all` add complexity; use sparingly

### Performance Characteristics

| Operation | Cost | Notes |
|-----------|------|-------|
| Concatenation | Low | Word equation solving |
| Length constraints | Low | Delegated to arithmetic |
| Regex membership | Moderate-High | Depends on regex complexity |
| Regex intersection | High | Can be exponential |
| Word equations | Moderate | NP-hard in general |

### Common Pitfalls

- Unbounded string variables with regex constraints can cause non-termination
- Complex regex patterns (nested Kleene stars) can cause exponential blowup
  in derivative computation
- String-integer conversion (`str.to.int`, `int.to.str`) introduces arithmetic
  coupling

---

## Floating-Point (FP)

**Logic:** `QF_FP`

### Solver Components

- **Bit-level encoding**: IEEE 754 semantics via bit-blasting
- **Rounding mode support**: All five IEEE 754 rounding modes
- **Special value handling**: NaN, infinity, signed zeros, denormals

### Supported Rounding Modes

| Mode | SMT-LIB2 Name | Description |
|------|---------------|-------------|
| RNE | `roundNearestTiesToEven` | Default; most common |
| RNA | `roundNearestTiesToAway` | Round half away from zero |
| RTP | `roundTowardPositive` | Ceiling |
| RTN | `roundTowardNegative` | Floor |
| RTZ | `roundTowardZero` | Truncation |

### Formulation Tips

```smt2
(set-logic QF_FP)
(declare-const x (_ FloatingPoint 8 24))  ; float32
(declare-const y (_ FloatingPoint 8 24))

; Always specify rounding mode for arithmetic
(assert (fp.eq (fp.add roundNearestTiesToEven x y)
               (fp #b0 #x42 #b01000000000000000000000)))
```

- FP reasoning reduces to bit-vector reasoning; 64-bit FP creates very large
  SAT instances
- Prefer 32-bit or 16-bit floats when precision is not critical
- Use `fp.eq` (IEEE equality) vs `=` (structural equality): NaN considerations
- Denormal numbers add complexity; if not needed, assert inputs are normal

### Common Pitfalls

- `fp.eq NaN NaN` is false (IEEE semantics) but `= NaN NaN` is true
- Signed zero: `fp.eq +0.0 -0.0` is true
- 64-bit FP bit-blasting produces very large formulas; expect long solving times
- Rounding mode must be specified for every arithmetic operation

---

## Algebraic Datatypes (DT)

**Logic:** `QF_DT`

### Solver Components

- **Constructor reasoning**: Tracks which constructor each term uses
- **Selector axioms**: `sel(cons(x)) = x`
- **Acyclicity**: Ensures no circular datatype values
- **Exhaustiveness**: Every DT value matches exactly one constructor

### Formulation Tips

```smt2
(set-logic QF_DT)
(declare-datatypes ((List 1))
  ((par (T) ((nil) (cons (hd T) (tl (List T)))))))

(declare-const xs (List Int))
(assert (not (= xs (as nil (List Int)))))
(assert (>= (hd xs) 0))
(check-sat)
```

- Use `is-Constructor` testers to branch on constructors
- Recursive datatypes (lists, trees) with quantifiers require MBQI
- Keep datatype nesting shallow; deeply nested ADTs multiply the axiom count

### Common Pitfalls

- Acyclicity constraints are checked lazily; cycles are detected as conflicts
- Mixing datatypes with other theories (arithmetic) requires proper theory
  combination via Nelson-Oppen
- Parametric datatypes increase the sort universe size

---

## Nonlinear Arithmetic (NIA/NRA)

**Logic:** `QF_NIA`, `QF_NRA`, `QF_NIRA`

### Solver Components

- **NLSAT**: Cylindrical Algebraic Decomposition (CAD) based
- **Groebner bases**: For polynomial ideal membership
- **Interval arithmetic**: Bound propagation for polynomials
- **Linearization**: Approximates nonlinear terms with linear constraints

### Strengths and Limitations

Nonlinear arithmetic is decidable for reals (Tarski) but undecidable for
integers (Hilbert's 10th problem). OxiZ uses NLSAT for both, which is
complete for QF_NRA but incomplete for QF_NIA.

### Formulation Tips

```smt2
(set-logic QF_NRA)
(declare-const x Real)
(declare-const y Real)

; Good: minimize polynomial degree
(assert (>= (* x x) 4.0))
(assert (<= (* y y) 9.0))

; Good: factor common subexpressions
(declare-const xy Real)
(assert (= xy (* x y)))
(assert (>= xy 0.0))
(assert (<= xy 10.0))
```

- Minimize the number of nonlinear variables and polynomial degree
- Factor common subexpressions to reduce the CAD complexity
- For QF_NIA, consider whether the problem can be encoded in QF_LIA
  (e.g., bounded multiplication can be encoded with case splits)
- Provide explicit bounds on all variables to help interval propagation

### Performance Characteristics

| Polynomial degree | Variables | Expected behavior |
|------------------|-----------|-------------------|
| 2 (quadratic) | < 10 | Usually fast |
| 2 | 10-50 | Moderate, depends on structure |
| 3+ | Any | Can be very slow |
| Any | > 50 nonlinear | Likely timeout |

### Common Pitfalls

- Forgetting to declare `QF_NIA`/`QF_NRA` causes NLSAT to not activate
- QF_NIA is incomplete: the solver may return `unknown` for satisfiable instances
- Division by a variable creates implicit constraints (denominator != 0)
- Very high-degree polynomials cause exponential CAD cell count

---

## Difference Logic (IDL/RDL)

**Logic:** `QF_IDL`, `QF_RDL` -- Specialized for x - y <= c constraints,
solved via Bellman-Ford. O(V*E) complexity, orders of magnitude faster than
general LIA for pure difference constraints (scheduling, timing analysis).

---

## Uninterpreted Functions (UF)

**Logic:** `QF_UF`, `QF_UFLIA`, `QF_UFLRA` -- Congruence closure (E-graph)
with union-find. O(n log n) for ground terms. Key axiom: `f(a) = f(b)` when
`a = b`.

---

## Theory Combination

OxiZ uses Nelson-Oppen to combine disjoint theories via equality propagation.
For best performance: minimize shared variables between theories, declare the
most specific logic, and use `balanced()` or `thorough()` configuration.
