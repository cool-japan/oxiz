# OxiZ-Opt Examples

This directory contains practical examples demonstrating real-world usage of the oxiz-opt MaxSAT solver.

## Running Examples

Each example can be run using:

```bash
cargo run --example <example_name>
```

## Available Examples

### 1. Scheduling (`scheduling.rs`)

**Problem**: Schedule tasks to time slots with preferences and constraints.

**Demonstrates**:
- Hard constraints (exclusivity, conflicts)
- Soft constraints with weights (preferences)
- Model extraction and interpretation
- Basic MaxSAT problem encoding

**Run**:
```bash
cargo run --example scheduling
```

**Use Case**: Task scheduling, meeting planning, resource reservation systems

---

### 2. Resource Allocation (`resource_allocation.rs`)

**Problem**: Allocate limited resources to projects to maximize total value.

**Demonstrates**:
- Weighted MaxSAT (WMax algorithm)
- Capacity constraints
- Value maximization
- Combinatorial optimization

**Run**:
```bash
cargo run --example resource_allocation
```

**Use Case**: Budget allocation, portfolio optimization, project selection

---

### 3. Algorithm Comparison (`algorithm_comparison.rs`)

**Problem**: Compare different MaxSAT algorithms on a graph coloring problem.

**Demonstrates**:
- Algorithm selection (Fu-Malik, OLL, MSU3, PMRES)
- Performance benchmarking
- Statistics collection
- Algorithm trade-offs

**Run**:
```bash
cargo run --example algorithm_comparison
```

**Use Case**: Algorithm selection, performance analysis, understanding algorithm behavior

---

## Problem Encoding Tips

### Hard Constraints vs Soft Constraints

- **Hard constraints**: Must be satisfied (use `add_hard`)
  - Example: "Task A must be scheduled"
  - Example: "Total cost must not exceed budget"

- **Soft constraints**: Preferences that can be violated (use `add_soft` or `add_soft_weighted`)
  - Example: "Prefer task A in the morning" (weight 5)
  - Example: "Prefer low-cost solutions" (weight proportional to cost)

### Variable Encoding

Common patterns for encoding variables:

1. **Binary choice**: One boolean variable per choice
   - `x_i = true` means option i is selected

2. **One-hot encoding**: Multiple variables for mutually exclusive choices
   - `x_{task,slot}` = task is assigned to slot
   - Add constraint: exactly one slot per task

3. **Integer encoding**: Use multiple boolean variables to represent integers
   - Binary encoding for values 0-7 uses 3 boolean variables

### Weight Selection

- Use integer weights when possible (faster than rationals)
- Higher weight = stronger preference
- Consider the ratio between weights (relative importance)
- Use `Weight::infinite()` for "almost hard" constraints

## Algorithm Selection Guide

| Algorithm | Best For | Characteristics |
|-----------|----------|-----------------|
| **Fu-Malik** | General-purpose, simple problems | Basic core-guided, easy to understand |
| **OLL** | Many overlapping cores | Uses incremental cardinality constraints |
| **MSU3** | Simple unweighted MaxSAT | Iterative relaxation |
| **WMax** | Weighted instances | Stratified approach by weight |
| **PMRES** | Partial MaxSAT with many hard constraints | Resolution-based |
| **RC2** | Modern instances | Advanced cardinality constraints |

## Performance Tips

1. **Preprocessing**: Enable preprocessing for large instances
   ```rust
   let config = MaxSatConfig::builder()
       .core_minimization(true)
       .build();
   ```

2. **Algorithm Selection**: Try different algorithms for your problem
   - Start with Fu-Malik for small problems
   - Use WMax for weighted instances
   - Try OLL for problems with many cores

3. **Stratified Solving**: Enable for weighted instances
   ```rust
   let config = MaxSatConfig::builder()
       .stratified(true)
       .build();
   ```

4. **Iteration Limits**: Set appropriate limits
   ```rust
   let config = MaxSatConfig::builder()
       .max_iterations(10000)
       .build();
   ```

## Further Reading

- MaxSAT solver documentation: See `../src/maxsat.rs`
- Algorithm details: Check individual algorithm implementations
- Benchmarks: See `../benches/opt_benchmarks.rs` for performance tests
