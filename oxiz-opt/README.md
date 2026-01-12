# oxiz-opt

Optimization engine for OxiZ: MaxSMT and Optimization Modulo Theories (OMT).

## Overview

This crate provides optimization capabilities on top of the SMT solver. It supports weighted MaxSAT, MaxSMT (soft constraints with theories), and objective function optimization.

## Features

| Feature | Description | Z3 Reference |
|:--------|:------------|:-------------|
| **MaxSAT** | Maximize satisfied clauses | `maxcore.cpp`, `wmax.cpp` |
| **MaxSMT** | MaxSAT with theory constraints | `maxsmt.cpp` |
| **OMT** | Optimize linear objectives | `optsmt.cpp` |
| **Pareto** | Multi-objective optimization | `opt_pareto.cpp` |

## Algorithms

### MaxSAT Algorithms
- **Core-guided**: Fu-Malik, OLL, MSU3
- **Weighted**: WMax, Stratified
- **Hybrid**: Combining multiple strategies

### OMT Algorithms
- **Linear search**: Iterative optimization
- **Binary search**: For bounded objectives
- **Symba**: Symbolic objective optimization

## Usage

```rust
use oxiz_opt::OptContext;

fn main() {
    let mut opt = OptContext::new();

    // Add hard constraints
    // opt.assert(...);

    // Add soft constraints with weights
    // opt.add_soft(..., weight);

    // Add objective to minimize/maximize
    // opt.minimize(...);

    // Solve
    // let result = opt.check();
}
```

## SMT-LIB2 Commands

```smt2
(minimize (+ x y))
(maximize z)
(assert-soft (> x 0) :weight 5)
(check-sat)
(get-objectives)
```

## License

MIT OR Apache-2.0
