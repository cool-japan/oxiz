# OxiZ WASM Examples

This directory contains example HTML files demonstrating various features of OxiZ WASM.

## Running the Examples

1. **Build the WASM package:**
   ```bash
   cd oxiz-wasm
   wasm-pack build --target web --release
   ```

2. **Serve the examples:**
   ```bash
   # Using Python
   python3 -m http.server 8000

   # Or using Node.js
   npx serve .
   ```

3. **Open in browser:**
   Navigate to `http://localhost:8000/examples/`

## Examples

### 1. Simple SAT (`simple-sat.html`)

Demonstrates basic satisfiability checking with a simple constraint.

**Concepts covered:**
- Setting logic
- Declaring constants
- Asserting formulas
- Checking satisfiability
- Extracting models

**Problem:** Find an integer x such that x > 0

### 2. UNSAT Core (`unsat-core.html`)

Shows how to extract an unsatisfiable core when constraints contradict.

**Concepts covered:**
- Enabling unsat core production
- Working with contradictory constraints
- Extracting and interpreting unsat cores

**Problem:** Proving that p and ¬p cannot both be true

### 3. Incremental Solving (`incremental-solving.html`)

Demonstrates efficient incremental solving with push/pop operations.

**Concepts covered:**
- Incremental constraint addition
- Push/pop stack operations
- Exploring multiple scenarios efficiently
- Backtracking to try different constraints

**Problem:** Finding valid age ranges with progressively stricter constraints

### 4. Optimization & MaxSMT (`optimization.html`)

Demonstrates optimization capabilities including minimization, maximization, and MaxSMT.

**Concepts covered:**
- Minimization objectives
- Maximization objectives
- Soft constraints with weights (MaxSMT)
- Lexicographic (multi-objective) optimization
- Resource allocation problems

**Problems:**
- Linear programming (minimize/maximize objectives)
- Preference satisfaction with weighted soft constraints
- Multi-objective optimization with priorities
- Resource allocation to maximize profit

### 5. Craig Interpolation (`interpolation.html`)

Demonstrates Craig interpolation for computing interpolants from UNSAT formula partitions.

**Concepts covered:**
- Enabling proof production
- Partitioning formulas into A and B
- Computing Craig interpolants
- Modular verification techniques
- Abstraction refinement

**Use cases:**
- Modular program verification
- Invariant generation
- Abstraction refinement in model checking
- Compositional reasoning

**Problem:** Given UNSAT formulas A and B, compute interpolant I where:
- A implies I
- I and B is UNSAT
- I only contains symbols common to A and B

## Performance Tips

1. **Reuse solver instances** - Don't create new solvers for each query
2. **Set logic early** - Enables specialized optimizations
3. **Use push/pop** - More efficient than reset + re-assert
4. **Disable unneeded features** - Turn off model/core production if not needed

## Further Reading

- [API Reference](../docs/API_REFERENCE.md)
- [Performance Tuning Guide](../docs/PERFORMANCE_TUNING.md)
- [Tutorial (Beginner)](../docs/TUTORIAL_BEGINNER.md)
