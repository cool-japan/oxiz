# SCIRS2 Policy for oxiz-nlsat

## Policy Statement

This crate (`oxiz-nlsat`) is a **non-linear arithmetic SMT solver** implementing Cylindrical Algebraic Decomposition (CAD) and does **NOT** require SciRS2 dependencies.

## Rationale

### Nature of the Project
- **oxiz-nlsat** implements the NLSAT algorithm for solving non-linear real and integer arithmetic constraints
- Primary focus: symbolic polynomial manipulation, real algebraic number computation, CAD-based decision procedures
- Domain: SMT solving for QF_NRA (quantifier-free non-linear real arithmetic) and QF_NIA (quantifier-free non-linear integer arithmetic)

### Dependency Analysis

The project uses:
- `num-bigint`, `num-rational`, `num-traits`: For exact arbitrary-precision arithmetic (required for exact real algebraic computation)
- `oxiz-core`: Core SMT infrastructure (AST, sorts, SMT-LIB parsing)
- `oxiz-math`: Mathematical primitives (polynomials, intervals, Sturm sequences)
- `rustc-hash`: Fast hashing for internal data structures
- `smallvec`: Stack-allocated vectors for performance
- **NO** usage of `rand` or `rand_distr` (no random sampling needed)
- **NO** usage of `ndarray` (no multi-dimensional array computations)
- **NO** statistical or scientific computing requirements

### CAD and Polynomial Operations

The project performs:
- **Symbolic polynomial arithmetic**: Exact operations on multivariate polynomials
- **Real root isolation**: Sturm sequences and interval arithmetic for exact root bounds
- **Sign determination**: Evaluating polynomial signs at algebraic points
- **Projection operations**: Resultants, discriminants, and leading coefficients

All operations are **exact and symbolic** - no floating-point approximations or numerical optimization.

### When SciRS2 Would Be Required

SciRS2 (and projects like NumRS2, ToRSh, etc.) would be required if this project:
- Performed statistical sampling or Monte Carlo methods
- Required multi-dimensional array operations
- Implemented machine learning algorithms
- Needed numerical optimization with gradient descent
- Performed floating-point scientific simulations
- Used approximate numerical methods instead of exact symbolic computation

### Current Status

✅ **COMPLIANT**: This project does not need to use SciRS2 as it performs exact symbolic computation, not numerical/statistical computing.

The project's arithmetic needs are satisfied by:
- `num-bigint`: Arbitrary precision integers (for polynomial coefficients and integer arithmetic)
- `num-rational`: Exact rational numbers (for real algebraic numbers and polynomial evaluation)
- `num-traits`: Generic numeric traits
- `oxiz-math`: Polynomial algebra and interval arithmetic

These are appropriate for **exact symbolic computation** in an SMT/CAD context.

## Review Date

Last reviewed: 2025-12-28

---

**Note**: If future features require random sampling, distributions, or array operations (e.g., for heuristic search, ML-guided decision making, or approximate numerical methods), this policy should be updated to use SciRS2-Core instead of rand/ndarray.
