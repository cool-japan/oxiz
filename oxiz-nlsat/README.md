# oxiz-nlsat

Non-linear arithmetic solver for OxiZ using Cylindrical Algebraic Decomposition (CAD).

## Overview

This crate implements the NLSAT algorithm for solving non-linear real arithmetic (QF_NRA) and non-linear integer arithmetic (QF_NIA) problems. NLSAT is one of the most complex components in Z3 (~180k lines of C++).

## Supported Logics

- **QF_NRA**: Quantifier-free non-linear real arithmetic
- **QF_NIA**: Quantifier-free non-linear integer arithmetic
- **QF_NIRA**: Combined non-linear integer/real arithmetic

## Key Components

| Module | Description | Z3 Reference |
|:-------|:------------|:-------------|
| `solver` | Main NLSAT solver | `nlsat_solver.cpp` |
| `types` | Core type definitions | `nlsat_types.h` |
| `clause` | Clause representation | `nlsat_clause.h` |
| `assignment` | Variable assignments | `nlsat_assignment.h` |
| `interval_set` | Solution intervals | `nlsat_interval_set.cpp` |
| `explain` | Conflict explanation | `nlsat_explain.cpp` |

## Algorithm Overview

1. **Variable Ordering**: Choose order for CAD projection
2. **Constraint Propagation**: Evaluate constraints under partial assignment
3. **Conflict Detection**: Find conflicting cells in CAD
4. **Explanation**: Generate lemmas for CDCL integration
5. **Backtracking**: Learn from conflicts and continue search

## Dependencies

- `oxiz-math`: Polynomial arithmetic, interval operations

## References

- Jovanović, D., & de Moura, L. (2012). Solving Non-linear Arithmetic. IJCAR.
- Collins, G. E. (1975). Quantifier elimination for real closed fields by cylindrical algebraic decomposition.

## License

MIT OR Apache-2.0
