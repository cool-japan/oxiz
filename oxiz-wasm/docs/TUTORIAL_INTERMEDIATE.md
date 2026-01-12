# OxiZ WASM Tutorial: Intermediate Guide

This tutorial covers intermediate topics for using OxiZ effectively.

## Prerequisites

- Completed the [Beginner Tutorial](TUTORIAL_BEGINNER.md)
- Understanding of basic SMT-LIB2 syntax
- Familiarity with JavaScript promises and async/await

## Topics Covered

1. [Incremental Solving](#incremental-solving)
2. [Check-SAT with Assumptions](#check-sat-with-assumptions)
3. [Unsat Cores](#unsat-cores)
4. [Function Definitions](#function-definitions)
5. [Simplification](#simplification)
6. [Solver Options and Presets](#solver-options-and-presets)
7. [Performance Optimization](#performance-optimization)
8. [Web Worker Integration](#web-worker-integration)

## Incremental Solving

Incremental solving lets you explore different scenarios without recreating the solver.

### Push and Pop

Use `push()` and `pop()` to create and remove assertion contexts:

```javascript
import init, { WasmSolver } from 'oxiz-wasm';

async function incrementalExample() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');
  solver.declareConst('x', 'Int');
  solver.declareConst('y', 'Int');

  // Base constraints (always active)
  solver.assertFormula('(> x 0)');
  solver.assertFormula('(> y 0)');

  // Scenario 1: x + y = 10
  solver.push(); // Save current state
  solver.assertFormula('(= (+ x y) 10)');
  console.log('Scenario 1:', solver.checkSat()); // sat
  console.log('Model 1:', solver.getModel());
  solver.pop(); // Restore previous state

  // Scenario 2: x > 100
  solver.push();
  solver.assertFormula('(> x 100)');
  console.log('Scenario 2:', solver.checkSat()); // sat
  console.log('Model 2:', solver.getModel());
  solver.pop();

  // Scenario 3: x + y < 3 (impossible with x, y > 0)
  solver.push();
  solver.assertFormula('(< (+ x y) 3)');
  console.log('Scenario 3:', solver.checkSat()); // unsat
  solver.pop();
}
```

### Multiple Levels

You can nest push/pop:

```javascript
solver.assertFormula('(> x 0)'); // Level 0

solver.push(); // Level 1
solver.assertFormula('(< x 10)');

solver.push(); // Level 2
solver.assertFormula('(= x 5)');
solver.checkSat(); // Checks: x > 0 AND x < 10 AND x = 5

solver.pop(); // Back to level 1
solver.checkSat(); // Checks: x > 0 AND x < 10

solver.pop(); // Back to level 0
solver.checkSat(); // Checks: x > 0
```

### Practical Example: Configuration Testing

```javascript
async function testConfigurations() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');
  solver.declareConst('cpu', 'Int');
  solver.declareConst('memory', 'Int');
  solver.declareConst('cost', 'Int');

  // Base requirements
  solver.assertFormula('(>= cpu 1)');
  solver.assertFormula('(>= memory 512)');

  // Cost model
  solver.assertFormula('(= cost (+ (* cpu 100) (div memory 10)))');

  const configs = [
    { name: 'Budget', constraints: ['(= cpu 2)', '(= memory 1024)', '(< cost 250)'] },
    { name: 'Standard', constraints: ['(= cpu 4)', '(= memory 2048)', '(< cost 500)'] },
    { name: 'Premium', constraints: ['(= cpu 8)', '(= memory 4096)', '(< cost 1000)'] },
  ];

  for (const config of configs) {
    solver.push();

    for (const constraint of config.constraints) {
      solver.assertFormula(constraint);
    }

    const result = solver.checkSat();
    console.log(`${config.name}: ${result}`);

    if (result === 'sat') {
      const model = solver.getModel();
      console.log(`  CPU: ${model.cpu.value}, Memory: ${model.memory.value}, Cost: ${model.cost.value}`);
    }

    solver.pop();
  }
}
```

## Check-SAT with Assumptions

Instead of push/pop, use temporary assumptions:

```javascript
async function assumptionsExample() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_UF');
  solver.declareConst('p', 'Bool');
  solver.declareConst('q', 'Bool');
  solver.declareConst('r', 'Bool');

  // Base constraint
  solver.assertFormula('(or p q r)');

  // Check under different assumptions
  let result1 = solver.checkSatAssuming(['p', 'q']);
  console.log('With p and q:', result1); // sat

  let result2 = solver.checkSatAssuming(['(not p)', '(not q)', '(not r)']);
  console.log('With ¬p, ¬q, ¬r:', result2); // unsat
}
```

### When to Use Assumptions vs Push/Pop

**Use Assumptions when:**
- You need to test many temporary constraints
- You don't want to modify the assertion stack
- You're exploring related scenarios quickly

**Use Push/Pop when:**
- You want to build up constraints incrementally
- You need nested contexts
- You're exploring hierarchical scenarios

## Unsat Cores

When a formula is unsatisfiable, find which constraints caused the conflict:

```javascript
async function unsatCoreExample() {
  await init();
  const solver = new WasmSolver();

  // Enable unsat core generation
  solver.setOption('produce-unsat-cores', 'true');

  solver.setLogic('QF_LIA');
  solver.declareConst('x', 'Int');

  // Add conflicting constraints
  solver.assertFormula('(> x 10)');
  solver.assertFormula('(< x 5)');
  solver.assertFormula('(> x 0)'); // This one is fine

  const result = solver.checkSat();

  if (result === 'unsat') {
    const core = solver.getUnsatCore();
    console.log('Unsat core:', core);
    // Shows which assertions conflict
  }
}
```

## Function Definitions

Define reusable functions:

```javascript
async function functionExample() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');

  // Define helper functions
  solver.defineFun('double', ['x Int'], 'Int', '(* 2 x)');
  solver.defineFun('square', ['x Int'], 'Int', '(* x x)');
  solver.defineFun('max', ['a Int', 'b Int'], 'Int', '(ite (> a b) a b)');

  solver.declareConst('n', 'Int');

  // Use defined functions
  solver.assertFormula('(= (double n) 10)');
  // Equivalent to: (= (* 2 n) 10)

  const result = solver.checkSat();
  if (result === 'sat') {
    const model = solver.getModel();
    console.log('n =', model.n.value); // 5
  }
}
```

### Complex Function Example

```javascript
async function complexFunctionExample() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');

  // Define abs(x) = if x >= 0 then x else -x
  solver.defineFun('abs', ['x Int'], 'Int',
    '(ite (>= x 0) x (- x))');

  // Define distance(a, b) = abs(a - b)
  solver.defineFun('distance', ['a Int', 'b Int'], 'Int',
    '(abs (- a b))');

  solver.declareConst('x', 'Int');
  solver.declareConst('y', 'Int');

  // Distance between x and y is 10
  solver.assertFormula('(= (distance x y) 10)');

  // x is positive
  solver.assertFormula('(> x 0)');

  const result = solver.checkSat();
  if (result === 'sat') {
    const model = solver.getModel();
    console.log('x =', model.x.value);
    console.log('y =', model.y.value);
    console.log('Distance:', Math.abs(parseInt(model.x.value) - parseInt(model.y.value)));
  }
}
```

## Simplification

Simplify complex expressions:

```javascript
async function simplificationExample() {
  await init();
  const solver = new WasmSolver();

  // Simplify arithmetic
  let result1 = solver.simplify('(+ 1 2 3)');
  console.log('1 + 2 + 3 =', result1); // "6"

  // Simplify boolean
  let result2 = solver.simplify('(and true false)');
  console.log('true AND false =', result2); // "false"

  // Simplify expressions with variables
  solver.declareConst('x', 'Int');
  let result3 = solver.simplify('(+ x 0)');
  console.log('x + 0 =', result3); // "x"

  let result4 = solver.simplify('(* x 1)');
  console.log('x * 1 =', result4); // "x"

  let result5 = solver.simplify('(* x 0)');
  console.log('x * 0 =', result5); // "0"
}
```

## Solver Options and Presets

### Using Presets

Quick configuration for common use cases:

```javascript
solver.applyPreset('complete');  // All features enabled
solver.applyPreset('fast');      // Optimized for speed
solver.applyPreset('debug');     // Verbose output
solver.applyPreset('incremental'); // Optimized for push/pop
```

### Custom Options

```javascript
solver.setOption('produce-models', 'true');
solver.setOption('produce-unsat-cores', 'true');

// Check what's set
const modelsEnabled = solver.getOption('produce-models');
console.log('Models enabled:', modelsEnabled);
```

### Statistics and Diagnostics

```javascript
// Get solver statistics
const stats = solver.getStatistics();
console.log('Assertions:', stats.num_assertions);
console.log('Last result:', stats.last_result);

// Get solver info
const name = solver.getInfo('name');
const version = solver.getInfo('version');
console.log(`${name} v${version}`);

// Get diagnostics
const warnings = solver.getDiagnostics();
warnings.forEach(w => console.warn(w));

// Check usage patterns
const rec = solver.checkPattern('incremental');
console.log('Incremental recommendation:', rec);
```

## Performance Optimization

### Tip 1: Set Logic Early

```javascript
// ❌ Slow
solver.declareConst('x', 'Int');
solver.setLogic('QF_LIA');

// ✅ Fast
solver.setLogic('QF_LIA');
solver.declareConst('x', 'Int');
```

### Tip 2: Use Assumptions Over Push/Pop for Rapid Testing

```javascript
// ❌ Slower for many scenarios
for (const scenario of scenarios) {
  solver.push();
  // ... add constraints ...
  solver.checkSat();
  solver.pop();
}

// ✅ Faster
for (const scenario of scenarios) {
  solver.checkSatAssuming(scenario.constraints);
}
```

### Tip 3: Reuse Solvers

```javascript
// ❌ Expensive
for (const problem of problems) {
  const solver = new WasmSolver();
  // ... solve problem ...
}

// ✅ Better
const solver = new WasmSolver();
for (const problem of problems) {
  solver.resetAssertions(); // Keep declarations
  // ... solve problem ...
}
```

### Tip 4: Validate Once, Assert Many

```javascript
// Validate formula before asserting in loops
const formula = '(> x 0)';
try {
  solver.validateFormula(formula);
  // Now safe to use in loops
  for (let i = 0; i < 1000; i++) {
    solver.assertFormula(formula);
  }
} catch (e) {
  console.error('Invalid formula:', e.message);
}
```

## Web Worker Integration

For long-running operations, use a Web Worker:

### worker.js
```javascript
import init, { WasmSolver } from 'oxiz-wasm';

let solver = null;

self.onmessage = async function(e) {
  const { id, type, payload } = e.data;

  try {
    if (type === 'init') {
      await init();
      solver = new WasmSolver();
      self.postMessage({ id, success: true });
    }
    else if (type === 'checkSat') {
      const result = solver.checkSat();
      self.postMessage({ id, success: true, result });
    }
    else if (type === 'assertFormula') {
      solver.assertFormula(payload);
      self.postMessage({ id, success: true });
    }
    // ... handle other operations ...
  } catch (error) {
    self.postMessage({ id, success: false, error: error.message });
  }
};
```

### main.js
```javascript
const worker = new Worker('worker.js', { type: 'module' });
let messageId = 0;
const pending = new Map();

function sendMessage(type, payload) {
  return new Promise((resolve, reject) => {
    const id = messageId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, payload });
  });
}

worker.onmessage = function(e) {
  const { id, success, result, error } = e.data;
  const handlers = pending.get(id);

  if (handlers) {
    pending.delete(id);
    if (success) {
      handlers.resolve(result);
    } else {
      handlers.reject(new Error(error));
    }
  }
};

// Usage
await sendMessage('init');
await sendMessage('assertFormula', '(> x 0)');
const result = await sendMessage('checkSat');
console.log('Result:', result);
```

### Using Framework Wrappers

For easier integration, use our framework wrappers:

**React:**
```javascript
import { useSolverWorker } from '@oxiz/react';

function MyComponent() {
  const { ready, checkSat, assertFormula } = useSolverWorker('/worker.js', 'QF_LIA');

  async function solve() {
    await assertFormula('(> x 0)');
    const result = await checkSat();
    console.log(result);
  }

  return <button onClick={solve} disabled={!ready}>Solve</button>;
}
```

**Vue:**
```javascript
import { useSolverWorker } from '@oxiz/vue';

export default {
  setup() {
    const { ready, checkSat, assertFormula } = useSolverWorker('/worker.js', 'QF_LIA');

    async function solve() {
      await assertFormula('(> x 0)');
      const result = await checkSat();
      console.log(result);
    }

    return { ready, solve };
  }
};
```

## Debugging Tips

### Use Debug Dump

```javascript
solver.declareConst('x', 'Int');
solver.assertFormula('(> x 0)');
solver.checkSat();

const dump = solver.debugDump();
console.log(dump);
// Shows: logic, assertions, last result, options, model, etc.
```

### Enable Tracing

```javascript
solver.setTracing(true);
solver.checkSat(); // Will emit detailed trace information
solver.setTracing(false);
```

### Safe Assertion

```javascript
try {
  solver.assertFormulaSafe('(> x 0)');
} catch (e) {
  console.error(e.message);
  // Provides helpful hints like:
  // "Make sure all variables are declared with declareConst()..."
}
```

## Next Steps

1. Read the [Advanced Tutorial](TUTORIAL_ADVANCED.md) for:
   - Theory-specific optimizations
   - Custom solvers
   - Advanced bitvector operations

2. See [Performance Tuning Guide](PERFORMANCE_GUIDE.md)

3. Check [Real-world Examples](../examples/real-world/)

## Practice Problems

1. Write a configuration validator for a web server
2. Implement a simple Sudoku solver
3. Create a resource allocation optimizer
4. Build a logic puzzle solver

Happy solving! 🚀
