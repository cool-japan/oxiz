# Performance Tuning Guide

This guide provides strategies for optimizing the performance of OxiZ WASM in your applications.

## Table of Contents

- [Quick Wins](#quick-wins)
- [Logic Selection](#logic-selection)
- [Incremental Solving](#incremental-solving)
- [WASM Binary Size Optimization](#wasm-binary-size-optimization)
- [Memory Management](#memory-management)
- [Browser-Specific Optimizations](#browser-specific-optimizations)
- [Profiling and Benchmarking](#profiling-and-benchmarking)
- [Common Performance Pitfalls](#common-performance-pitfalls)

---

## Quick Wins

### 1. Always Set the Logic

Setting the SMT logic enables specialized solvers and optimizations.

**Bad:**
```javascript
const solver = new WasmSolver();
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
solver.checkSat(); // Uses generic solver
```

**Good:**
```javascript
const solver = new WasmSolver();
solver.setLogic("QF_LIA"); // Enables LIA-specific optimizations
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
solver.checkSat(); // Uses optimized LIA solver
```

**Impact:** 2-10x speedup depending on problem

### 2. Use Appropriate Presets

Apply configuration presets to quickly optimize for your use case.

```javascript
const solver = new WasmSolver();

// For fastest solving (no models or cores)
solver.applyPreset("fast");

// For debugging
solver.applyPreset("debug");

// For incremental solving
solver.applyPreset("incremental");
```

### 3. Prefer `checkSatAssuming` Over `push/pop`

For temporary constraints, use `checkSatAssuming` instead of push/pop.

**Slower:**
```javascript
solver.push();
solver.assertFormula("(> x 10)");
const result = solver.checkSat();
solver.pop();
```

**Faster:**
```javascript
const result = solver.checkSatAssuming(["(> x 10)"]);
// No need to pop - assertion stack unchanged
```

**Impact:** 20-50% speedup for incremental queries

---

## Logic Selection

Choose the most specific logic for your problem:

### Logic Performance Hierarchy (fastest to slowest)

1. **QF_UF** - Pure uninterpreted functions
   - Best for: Boolean reasoning, equivalence checking
   - Example: `(= (f a) (f b))`

2. **QF_LIA** - Linear integer arithmetic
   - Best for: Integer constraints, counting problems
   - Example: `(and (>= x 0) (<= x 10))`

3. **QF_LRA** - Linear real arithmetic
   - Best for: Continuous optimization, scheduling
   - Example: `(and (>= x 0.0) (<= x 10.5))`

4. **QF_BV** - Bitvectors
   - Best for: Low-level verification, hardware
   - Example: `(bvult x #x0000000a)`

5. **ALL** - Combined theories (slowest)
   - Use only when necessary
   - Avoid if possible

### Logic Selection Example

```javascript
// If you only need integers and linear constraints
solver.setLogic("QF_LIA"); // Fast

// If you mix different theories
solver.setLogic("ALL"); // Slower, but necessary
```

---

## Incremental Solving

### Strategy 1: Reuse Solver Instances

**Bad:**
```javascript
for (const constraint of constraints) {
  const solver = new WasmSolver(); // ❌ Creating new solver each time
  solver.setLogic("QF_LIA");
  solver.declareConst("x", "Int");
  solver.assertFormula(constraint);
  const result = solver.checkSat();
  console.log(result);
}
```

**Good:**
```javascript
const solver = new WasmSolver();
solver.setLogic("QF_LIA");
solver.declareConst("x", "Int");

for (const constraint of constraints) {
  solver.push(); // ✅ Reuse solver
  solver.assertFormula(constraint);
  const result = solver.checkSat();
  console.log(result);
  solver.pop();
}
```

**Impact:** 10-100x speedup for multiple queries

### Strategy 2: Shared Declarations

Declare all constants upfront to avoid repeated declarations.

**Bad:**
```javascript
// Declaring variables as needed
solver.assertFormula("(> x 0)");
solver.declareConst("x", "Int"); // Declaration after use
```

**Good:**
```javascript
// Declare all variables upfront
solver.declareConst("x", "Int");
solver.declareConst("y", "Int");
solver.declareConst("z", "Int");

// Then assert formulas
solver.assertFormula("(> x 0)");
solver.assertFormula("(< y 10)");
```

### Strategy 3: Minimize Push/Pop Depth

Avoid deep nesting of push/pop operations.

**Bad:**
```javascript
solver.push();
  solver.push();
    solver.push();
      solver.push(); // Too deep!
```

**Good:**
```javascript
// Keep nesting shallow
solver.push();
  solver.assertFormula("...");
  solver.checkSat();
solver.pop();
```

---

## WASM Binary Size Optimization

### Build Configuration

Optimize the WASM binary size during build:

```bash
# Standard build
wasm-pack build --target web

# Optimized build
wasm-pack build --target web --release

# With wasm-opt (requires wasm-opt installed)
wasm-pack build --target web --release
wasm-opt -Oz -o pkg/oxiz_wasm_bg.wasm pkg/oxiz_wasm_bg.wasm
```

### Size Optimization Levels

- `-O` - Optimize for size (moderate)
- `-Oz` - Optimize aggressively for size
- `-O3` - Optimize for speed (larger binary)

**Recommendation:** Use `-Oz` for production web apps

### Compression

Serve WASM files with compression:

```nginx
# Nginx example
location ~* \.wasm$ {
    gzip on;
    gzip_types application/wasm;
}
```

Or use Brotli compression for even better results (30-40% smaller).

---

## Memory Management

### Avoid Creating Large Models

Only request models when necessary:

```javascript
// If you don't need the model
solver.setOption("produce-models", "false");
solver.checkSat(); // Faster, uses less memory

// Only enable when needed
solver.setOption("produce-models", "true");
const result = solver.checkSat();
if (result === "sat") {
  const model = solver.getModel();
}
```

### Reset Assertions Periodically

For long-running applications, reset assertions to free memory:

```javascript
// After many iterations
if (iterations % 100 === 0) {
  solver.resetAssertions(); // Clears assertions, keeps declarations
}
```

### Use Garbage Collection Hints

In JavaScript, help the GC by nulling out large objects:

```javascript
let solver = new WasmSolver();
// ... use solver ...

// When done
solver.reset();
solver = null; // Help GC reclaim memory
```

---

## Browser-Specific Optimizations

### Use Web Workers

Run solver in a Web Worker to avoid blocking the UI:

**worker.js:**
```javascript
import init, { WasmSolver } from './oxiz-wasm/oxiz_wasm.js';

await init();

self.onmessage = async (e) => {
  const solver = new WasmSolver();
  solver.setLogic(e.data.logic);
  // ... solve ...
  self.postMessage({ result });
};
```

**main.js:**
```javascript
const worker = new Worker('worker.js', { type: 'module' });

worker.postMessage({
  logic: "QF_LIA",
  constraints: [/* ... */]
});

worker.onmessage = (e) => {
  console.log("Result:", e.data.result);
};
```

### Use Async Methods

For long-running operations, use async methods:

```javascript
// Synchronous (blocks UI)
const result = solver.checkSat();

// Asynchronous (doesn't block UI)
const result = await solver.checkSatAsync();
```

### Lazy Loading

Load WASM module only when needed:

```javascript
let wasmModule = null;

async function getSolver() {
  if (!wasmModule) {
    wasmModule = await import('./oxiz-wasm/oxiz_wasm.js');
    await wasmModule.default();
  }
  return new wasmModule.WasmSolver();
}

// Use only when needed
const solver = await getSolver();
```

---

## Profiling and Benchmarking

### Measure Performance

Use the browser's performance API:

```javascript
const start = performance.now();

solver.setLogic("QF_LIA");
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
const result = solver.checkSat();

const end = performance.now();
console.log(`checkSat took ${end - start}ms`);
```

### Get Solver Statistics

Use `getStatistics()` to understand solver behavior:

```javascript
const stats = solver.getStatistics();
console.log("Assertions:", stats.num_assertions);
console.log("Last result:", stats.last_result);
```

### Profile Memory Usage

Monitor memory usage:

```javascript
if (performance.memory) {
  console.log("Used:", performance.memory.usedJSHeapSize);
  console.log("Total:", performance.memory.totalJSHeapSize);
}
```

### Benchmark Suite

Create a benchmark suite for your specific use case:

```javascript
function benchmark(name, fn, iterations = 1000) {
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = performance.now();
  const avg = (end - start) / iterations;
  console.log(`${name}: ${avg.toFixed(3)}ms per iteration`);
}

benchmark("declareConst", () => {
  const solver = new WasmSolver();
  solver.declareConst("x", "Int");
});

benchmark("checkSat simple", () => {
  const solver = new WasmSolver();
  solver.setLogic("QF_UF");
  solver.declareConst("p", "Bool");
  solver.assertFormula("p");
  solver.checkSat();
});
```

---

## Common Performance Pitfalls

### Pitfall 1: Not Setting Logic

```javascript
// ❌ Slow - uses generic solver
const solver = new WasmSolver();
solver.declareConst("x", "Int");
solver.checkSat();

// ✅ Fast - uses specialized solver
const solver = new WasmSolver();
solver.setLogic("QF_LIA");
solver.declareConst("x", "Int");
solver.checkSat();
```

### Pitfall 2: Creating New Solvers Repeatedly

```javascript
// ❌ Slow - creates new solver each time
function check(constraint) {
  const solver = new WasmSolver();
  solver.setLogic("QF_LIA");
  solver.declareConst("x", "Int");
  solver.assertFormula(constraint);
  return solver.checkSat();
}

// ✅ Fast - reuses solver
const solver = new WasmSolver();
solver.setLogic("QF_LIA");
solver.declareConst("x", "Int");

function check(constraint) {
  solver.push();
  solver.assertFormula(constraint);
  const result = solver.checkSat();
  solver.pop();
  return result;
}
```

### Pitfall 3: Unnecessary Model Generation

```javascript
// ❌ Slow - generates model even when not needed
solver.setOption("produce-models", "true");
for (const constraint of constraints) {
  solver.assertFormula(constraint);
  solver.checkSat(); // Generates model every time
}

// ✅ Fast - only generates model when needed
solver.setOption("produce-models", "false");
for (const constraint of constraints) {
  solver.assertFormula(constraint);
  solver.checkSat(); // No model generation
}
```

### Pitfall 4: Synchronous Operations in UI Thread

```javascript
// ❌ Bad - blocks UI
button.onclick = () => {
  const solver = new WasmSolver();
  solver.setLogic("QF_LIA");
  // ... complex solving ...
  const result = solver.checkSat(); // UI freezes
  updateUI(result);
};

// ✅ Good - doesn't block UI
button.onclick = async () => {
  const solver = new WasmSolver();
  solver.setLogic("QF_LIA");
  // ... complex solving ...
  const result = await solver.checkSatAsync(); // UI stays responsive
  updateUI(result);
};
```

### Pitfall 5: Over-Deep Push/Pop Nesting

```javascript
// ❌ Slow - deep nesting
solver.push();
  solver.push();
    solver.push();
      solver.push();
        solver.checkSat();
      solver.pop();
    solver.pop();
  solver.pop();
solver.pop();

// ✅ Fast - shallow nesting or use assumptions
solver.checkSatAssuming(["constraint1", "constraint2", "constraint3"]);
```

---

## Performance Checklist

- [ ] Set the most specific logic for your problem
- [ ] Use appropriate configuration presets
- [ ] Reuse solver instances for multiple queries
- [ ] Prefer `checkSatAssuming` over `push/pop` for temporary constraints
- [ ] Declare all variables upfront
- [ ] Disable model generation when not needed
- [ ] Use Web Workers for heavy computations
- [ ] Use async methods in browser environments
- [ ] Minimize push/pop nesting depth
- [ ] Profile and benchmark your specific use cases
- [ ] Compress WASM files during deployment
- [ ] Use lazy loading for on-demand solver instantiation

---

## Performance Metrics

Expected performance for typical problems:

| Operation | Time (typical) | Notes |
|-----------|---------------|-------|
| Create solver | < 1ms | Very fast |
| Set logic | < 1ms | One-time cost |
| Declare const | < 0.1ms | Per variable |
| Assert formula | 0.1-10ms | Depends on complexity |
| Check-sat (SAT) | 1ms-10s | Highly problem-dependent |
| Check-sat (UNSAT) | 1ms-60s | Can be slower than SAT |
| Get model | 0.1-5ms | After SAT result |
| Push/pop | 0.1-1ms | Per operation |

*Note: These are rough estimates and actual performance varies significantly based on problem complexity, logic, and hardware.*

---

## Further Reading

- [API Reference](API_REFERENCE.md)
- [Tutorial (Intermediate)](TUTORIAL_INTERMEDIATE.md)
- [Architecture Documentation](ARCHITECTURE.md)

---

## Reporting Performance Issues

If you encounter performance issues:

1. Create a minimal reproducible example
2. Include timing information
3. Specify the logic and problem size
4. Note your browser and platform
5. Run with debug tracing enabled: `solver.setTracing(true)`
6. Capture statistics with `solver.getStatistics()`

Report issues at: https://github.com/cool-japan/oxiz/issues
