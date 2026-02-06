# Getting Started with OxiZ

## Introduction

OxiZ is a next-generation SMT (Satisfiability Modulo Theories) solver written in Pure Rust. This guide will help you get started with using OxiZ in your projects.

## Installation

### From crates.io

```bash
cargo add oxiz
```

### From source

```bash
git clone https://github.com/cool-japan/oxiz
cd oxiz
cargo build --release
```

## Quick Start

### Example 1: Basic Boolean Satisfiability

```rust
use oxiz_solver::{Solver, SolverResult};
use oxiz_core::ast::TermManager;

fn main() {
    // Create solver and term manager
    let mut solver = Solver::new();
    let mut tm = TermManager::new();

    // Create boolean variables
    let p = tm.mk_var("p", tm.sorts.bool_sort);
    let q = tm.mk_var("q", tm.sorts.bool_sort);

    // Assert: p AND q
    let formula = tm.mk_and(vec![p, q]);
    solver.assert(formula, &mut tm);

    // Check satisfiability
    match solver.check(&mut tm) {
        SolverResult::Sat => {
            println!("Satisfiable!");
            if let Some(model) = solver.get_model(&tm) {
                println!("Model: p={:?}, q={:?}",
                    model.eval(p, &tm),
                    model.eval(q, &tm)
                );
            }
        }
        SolverResult::Unsat => println!("Unsatisfiable!"),
        SolverResult::Unknown => println!("Unknown"),
    }
}
```

### Example 2: Integer Arithmetic

```rust
use oxiz_solver::{Solver, SolverResult};
use oxiz_core::ast::TermManager;
use num_bigint::BigInt;

fn main() {
    let mut solver = Solver::new();
    solver.set_logic("QF_LIA"); // Quantifier-Free Linear Integer Arithmetic
    let mut tm = TermManager::new();

    // Variables
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);

    // Constants
    let five = tm.mk_int(BigInt::from(5));
    let ten = tm.mk_int(BigInt::from(10));

    // Assert: x + y = 10
    let sum = tm.mk_add(vec![x, y]);
    solver.assert(tm.mk_eq(sum, ten), &mut tm);

    // Assert: x > 5
    solver.assert(tm.mk_gt(x, five), &mut tm);

    // Solve
    match solver.check(&mut tm) {
        SolverResult::Sat => {
            if let Some(model) = solver.get_model(&tm) {
                println!("Solution found:");
                println!("  x = {:?}", model.eval(x, &tm));
                println!("  y = {:?}", model.eval(y, &tm));
            }
        }
        SolverResult::Unsat => println!("No solution exists"),
        SolverResult::Unknown => println!("Could not determine"),
    }
}
```

### Example 3: SMT-LIB2 Format

```rust
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::parse_script;
use oxiz_solver::Solver;

fn main() {
    let input = r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const y Int)
        (assert (= (+ x y) 10))
        (assert (> x 5))
        (check-sat)
        (get-model)
    "#;

    let mut tm = TermManager::new();
    let commands = parse_script(input, &mut tm).unwrap();

    let mut solver = Solver::new();
    for cmd in commands {
        // Execute commands
        solver.execute_command(cmd, &mut tm);
    }
}
```

## Core Concepts

### 1. Term Manager

The `TermManager` is the central data structure for creating and managing terms:

```rust
let mut tm = TermManager::new();

// Create variables
let x = tm.mk_var("x", tm.sorts.int_sort);
let p = tm.mk_var("p", tm.sorts.bool_sort);

// Create constants
let five = tm.mk_int(BigInt::from(5));
let true_val = tm.mk_true();

// Build expressions
let x_plus_5 = tm.mk_add(vec![x, five]);
let comparison = tm.mk_gt(x_plus_5, ten);
```

**Key Features**:
- Hash consing: identical terms are shared (memory efficient)
- Immutable: terms cannot be modified after creation
- Type-safe: sort system prevents type errors

### 2. Solver

The `Solver` orchestrates the solving process:

```rust
let mut solver = Solver::new();

// Set logic (optional, improves performance)
solver.set_logic("QF_LIA");

// Assert formulas
solver.assert(formula1, &mut tm);
solver.assert(formula2, &mut tm);

// Check satisfiability
let result = solver.check(&mut tm);

// Get model (if SAT)
if let Some(model) = solver.get_model(&tm) {
    // Evaluate terms
    let value = model.eval(x, &tm);
}

// Get proof (if UNSAT)
if let Some(proof) = solver.get_proof() {
    // Verify unsatisfiability
}

// Get unsat core
if let Some(core) = solver.get_unsat_core() {
    // Minimal unsatisfiable subset
}
```

### 3. Incremental Solving

OxiZ supports incremental solving with push/pop:

```rust
let mut solver = Solver::new();
let mut tm = TermManager::new();

// Base assertions
solver.assert(base_formula, &mut tm);

// Push scope
solver.push();
solver.assert(additional_constraint, &mut tm);

match solver.check(&mut tm) {
    SolverResult::Sat => println!("Satisfiable with constraint"),
    SolverResult::Unsat => println!("Unsatisfiable with constraint"),
    _ => {}
}

// Pop scope (remove additional_constraint)
solver.pop();

// Check again with only base formula
let result2 = solver.check(&mut tm);
```

## Supported Theories

### Quantifier-Free Fragments (Decidable)

| Logic     | Description | Example |
|-----------|-------------|---------|
| QF_UF     | Uninterpreted Functions | `f(x) = f(y) ∧ x ≠ y` |
| QF_LIA    | Linear Integer Arithmetic | `2x + 3y ≤ 10` |
| QF_LRA    | Linear Real Arithmetic | `0.5x + 1.2y = 3.7` |
| QF_BV     | Bitvectors | `bvadd(x, y) = #x08` |
| QF_AUFLIA | Arrays + UF + LIA | `select(store(a, i, v), i) = v` |

### Quantified Fragments (Semi-decidable)

| Logic | Description | Example |
|-------|-------------|---------|
| LIA   | LIA with quantifiers | `∀x. (x ≥ 0 → x + 1 > 0)` |
| UFLIA | UF + LIA + quantifiers | `∀x. f(x) ≥ 0` |

## Configuration

### Resource Limits

```rust
use oxiz_core::config::{Config, ResourceLimits};
use std::time::Duration;

let config = Config {
    resource_limits: ResourceLimits {
        max_time_ms: Some(5000),      // 5 second timeout
        max_memory_mb: Some(1024),    // 1 GB memory limit
        max_iterations: Some(100000), // Iteration limit
        max_depth: Some(100),         // Recursion depth
    },
    ..Default::default()
};

let mut solver = Solver::with_config(config);
```

### Solver Options

```rust
use oxiz_core::config::{Config, SatParams};

let config = Config {
    sat_params: SatParams {
        restart_strategy: RestartStrategy::Luby(100),
        phase_saving: PhaseSaving::Always,
        clause_deletion: ClauseDeletionStrategy::Activity,
        ..Default::default()
    },
    ..Default::default()
};
```

## Advanced Features

### Tactics for Preprocessing

```rust
use oxiz_core::tactic::{Goal, StatelessSimplifyTactic, Tactic};

let mut tm = TermManager::new();

// Create complex formula
let formula = tm.mk_and(vec![
    tm.mk_or(vec![p, tm.mk_true()]),  // Simplifies to true
    tm.mk_and(vec![q, tm.mk_false()]), // Simplifies to false
]);

// Create goal and apply tactic
let goal = Goal::new(vec![formula]);
let tactic = StatelessSimplifyTactic;
let result = tactic.apply(&goal);

// Use simplified formula
for subgoal in result.subgoals {
    println!("Simplified: {:?}", subgoal.formulas);
}
```

### Quantifier Elimination

```rust
use oxiz_core::qe::{QeLiteSolver, QeLiteConfig};

let mut qe_solver = QeLiteSolver::new(QeLiteConfig::default());
let mut tm = TermManager::new();

// ∃x. (x > 0 ∧ y = x + 5)
// Eliminate x, get: y > 5
let x = tm.mk_var("x", tm.sorts.int_sort);
let y = tm.mk_var("y", tm.sorts.int_sort);

let body = /* ... formula with x ... */;
let result = qe_solver.eliminate(&[x], body, &mut tm);

match result {
    Ok(qe_result) => {
        println!("Quantifier-free formula: {:?}", qe_result.formula);
    }
    Err(e) => println!("QE failed: {:?}", e),
}
```

### Optimization (MaxSMT)

```rust
use oxiz_solver::{Optimizer, Objective, ObjectiveKind};

let mut optimizer = Optimizer::new();
let mut tm = TermManager::new();

// Minimize: x + y
let x = tm.mk_var("x", tm.sorts.int_sort);
let y = tm.mk_var("y", tm.sorts.int_sort);
let sum = tm.mk_add(vec![x, y]);

optimizer.add_objective(Objective {
    kind: ObjectiveKind::Minimize,
    expr: sum,
});

// Constraints
optimizer.assert(tm.mk_ge(x, zero), &mut tm);
optimizer.assert(tm.mk_ge(y, zero), &mut tm);
optimizer.assert(tm.mk_le(tm.mk_add(vec![x, y]), ten), &mut tm);

match optimizer.optimize(&mut tm) {
    Ok(result) => {
        println!("Optimal value: {:?}", result.optimal_value);
        println!("Solution: {:?}", result.model);
    }
    Err(e) => println!("Optimization failed: {:?}", e),
}
```

## Best Practices

### 1. Set the Logic

Always set the logic if you know it:

```rust
solver.set_logic("QF_LIA"); // Enables specialized algorithms
```

### 2. Use Incremental Solving

Reuse solver instances with push/pop:

```rust
// Good: incremental
solver.push();
solver.assert(new_constraint, &mut tm);
let result = solver.check(&mut tm);
solver.pop();

// Avoid: creating new solver each time (slower)
let mut new_solver = Solver::new();
// ...
```

### 3. Resource Limits

Set timeouts for production use:

```rust
solver.set_time_limit(Duration::from_secs(60));
```

### 4. Reuse TermManager

Create one TermManager per solving session:

```rust
// Good: single TermManager
let mut tm = TermManager::new();
let x = tm.mk_var("x", tm.sorts.int_sort);
let y = tm.mk_var("y", tm.sorts.int_sort);

// Avoid: multiple TermManagers (terms not comparable)
let mut tm1 = TermManager::new();
let mut tm2 = TermManager::new();
let x1 = tm1.mk_var("x", tm1.sorts.int_sort);
let x2 = tm2.mk_var("x", tm2.sorts.int_sort);
// x1 and x2 are incomparable!
```

## Error Handling

```rust
use oxiz_core::error::{Result, OxizError};

fn solve_formula(input: &str) -> Result<bool> {
    let mut tm = TermManager::new();
    let commands = parse_script(input, &mut tm)?;

    let mut solver = Solver::new();
    for cmd in commands {
        solver.execute_command(cmd, &mut tm)?;
    }

    match solver.check(&mut tm) {
        SolverResult::Sat => Ok(true),
        SolverResult::Unsat => Ok(false),
        SolverResult::Unknown => Err(OxizError::Unknown),
    }
}

// Usage
match solve_formula(input_str) {
    Ok(is_sat) => println!("Result: {}", is_sat),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Performance Tips

### 1. Arena Allocation

Use arena for bulk term creation:

```rust
use oxiz_core::alloc::Arena;

let mut arena = Arena::new(Default::default());
for i in 0..10000 {
    arena.alloc(create_term(i));
}
// Fast bulk deallocation when arena drops
```

### 2. Batch Assertions

Assert multiple formulas before checking:

```rust
// Good: batch assertions
for formula in formulas {
    solver.assert(formula, &mut tm);
}
let result = solver.check(&mut tm);

// Avoid: check after each assertion (slower)
for formula in formulas {
    solver.assert(formula, &mut tm);
    solver.check(&mut tm); // Repeated work
}
```

### 3. Simplify Before Solving

Use tactics to preprocess:

```rust
let simplified = StatelessSimplifyTactic.apply(&goal);
solver.assert_goal(simplified, &mut tm);
```

## Examples

See the `examples/` directory for more:

- `oxiz-core/examples/` - Core functionality (10 examples)
- `oxiz-solver/examples/` - Solver usage (8 examples)
- `oxiz-math/examples/` - Math algorithms (6 examples)
- `oxiz-theories/examples/` - Theory solvers (6 examples)

## Next Steps

- Read the [Architecture Guide](../architecture/solver-architecture.md)
- Explore [Theory Combination](../architecture/theory-combination.md)
- Check [API Documentation](https://docs.rs/oxiz)
- Join the community at [GitHub](https://github.com/cool-japan/oxiz)

## Troubleshooting

### Common Issues

**Issue**: `TermManager` mismatch errors

**Solution**: Use a single `TermManager` instance per solving session.

---

**Issue**: Timeout or Unknown result

**Solution**: Increase time limit, simplify formula, or use QF fragment.

---

**Issue**: Out of memory

**Solution**: Set memory limits, enable clause deletion, or use smaller formulas.

---

**Issue**: Performance is slow

**Solution**: Set logic, use tactics, batch assertions, enable parallelism.

## Support

- Documentation: <https://docs.rs/oxiz>
- Issues: <https://github.com/cool-japan/oxiz/issues>
- Discussions: <https://github.com/cool-japan/oxiz/discussions>
