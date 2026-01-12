# oxiz-theories

Theory solvers for OxiZ SMT solver.

## Overview

This crate provides modular theory solver implementations for CDCL(T):

- **EUF** - Equality with Uninterpreted Functions (congruence closure)
- **Arithmetic** - Linear Real/Integer Arithmetic (Simplex)
- **BitVector** - Fixed-size bit-vector operations

## Theory Trait

All theory solvers implement the `Theory` trait:

```rust
pub trait Theory {
    fn id(&self) -> TheoryId;
    fn name(&self) -> &str;
    fn can_handle(&self, term: TermId) -> bool;
    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult>;
    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult>;
    fn check(&mut self) -> Result<TheoryResult>;
    fn push(&mut self);
    fn pop(&mut self);
    fn reset(&mut self);
}
```

## EUF Solver

Implements Equality with Uninterpreted Functions using:
- **Union-Find** with path compression and union by rank
- **Congruence Closure** for function application
- **Disequality tracking** for conflict detection

```rust
use oxiz_theories::euf::EufSolver;

let mut solver = EufSolver::new();
let a = solver.intern(term_a);
let b = solver.intern(term_b);
let fa = solver.intern_app(term_fa, func_f, [a]);
let fb = solver.intern_app(term_fb, func_f, [b]);

solver.merge(a, b, reason)?;  // a = b
assert!(solver.are_equal(fa, fb));  // f(a) = f(b) by congruence
```

## Arithmetic Solver

Implements Linear Real Arithmetic (LRA) using the Simplex algorithm:
- **Tableau-based Simplex** with slack variables
- **Bound propagation** for conflict detection
- **Pivoting** for feasibility search

```rust
use oxiz_theories::arithmetic::ArithSolver;
use num_rational::Rational64;

let mut solver = ArithSolver::lra();

// x >= 0
solver.assert_ge(&[(x, Rational64::one())], Rational64::zero(), reason);

// x + y <= 10
solver.assert_le(
    &[(x, Rational64::one()), (y, Rational64::one())],
    Rational64::from_integer(10),
    reason
);

let result = solver.check()?;
```

## BitVector Solver

Implements bit-vector operations:
- **Bit-blasting** to propositional logic
- **Word-level reasoning** for efficiency
- **Eager/lazy encoding** strategies

```rust
use oxiz_theories::bv::BvSolver;

let mut solver = BvSolver::new();
solver.assert_eq(bv_term_a, bv_term_b, reason);
let result = solver.check()?;
```

## Dependencies

- `num-rational` - Exact rational arithmetic
- `rustc-hash` - Fast hash maps
- `smallvec` - Stack-allocated vectors

## License

MIT OR Apache-2.0
