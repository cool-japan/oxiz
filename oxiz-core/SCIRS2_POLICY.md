# SCIRS2 Policy for oxiz-core

## Policy Statement

This crate (`oxiz-core`) is a **pure logic SMT solver core** and does **NOT** require SciRS2 dependencies.

## Rationale

### Nature of the Project
- **oxiz-core** implements an SMT (Satisfiability Modulo Theories) solver
- Primary focus: symbolic logic, term manipulation, constraint solving
- Domain: formal verification, theorem proving, program analysis

### Dependency Analysis

The project uses:
- `num-bigint`, `num-rational`, `num-traits`: For arbitrary-precision arithmetic (required for SMT theory of integers/reals)
- **NO** usage of `rand` or `rand_distr` (no random sampling needed)
- **NO** usage of `ndarray` (no multi-dimensional array computations)
- **NO** statistical or scientific computing requirements

### When SciRS2 Would Be Required

SciRS2 (and projects like NumRS2, ToRSh, etc.) would be required if this project:
- Performed statistical sampling or Monte Carlo methods
- Required multi-dimensional array operations
- Implemented machine learning algorithms
- Needed numerical optimization with gradient descent
- Performed scientific simulations

### Current Status

✅ **COMPLIANT**: This project does not need to use SciRS2 as it does not perform scientific/statistical computing.

The project's arithmetic needs are satisfied by:
- `num-bigint`: Arbitrary precision integers (for SMT-LIB Int theory)
- `num-rational`: Rational numbers (for SMT-LIB Real theory)
- `num-traits`: Generic numeric traits

These are appropriate for **exact symbolic computation** in an SMT solver context.

## Review Date

Last reviewed: 2025-12-28

---

**Note**: If future features require random sampling, distributions, or array operations, this policy should be updated to use SciRS2-Core instead of rand/ndarray.
