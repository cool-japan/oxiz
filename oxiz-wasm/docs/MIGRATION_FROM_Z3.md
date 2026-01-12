# Migration Guide: Z3 JavaScript to OxiZ WASM

This guide helps you migrate from Z3's JavaScript bindings to OxiZ WASM.

## Overview

OxiZ WASM provides a similar API to Z3's JavaScript bindings but with some differences in initialization, naming conventions, and available features.

## Key Differences

### 1. Initialization

**Z3:**
```javascript
const { init } = require('z3-solver');
const { Context } = await init();
const solver = new Context('main').solver();
```

**OxiZ:**
```javascript
import init, { WasmSolver } from 'oxiz-wasm';
await init();
const solver = new WasmSolver();
```

### 2. Setting Logic

**Z3:**
```javascript
solver.set('logic', 'QF_LIA');
```

**OxiZ:**
```javascript
solver.setLogic('QF_LIA');
```

### 3. Declaring Constants

**Z3:**
```javascript
const x = ctx.Int.const('x');
const y = ctx.Int.const('y');
```

**OxiZ:**
```javascript
solver.declareConst('x', 'Int');
solver.declareConst('y', 'Int');
```

### 4. Adding Assertions

**Z3:**
```javascript
solver.add(x.gt(0));
solver.add(y.lt(10));
```

**OxiZ:**
```javascript
solver.assertFormula('(> x 0)');
solver.assertFormula('(< y 10)');
```

### 5. Checking Satisfiability

**Z3:**
```javascript
const result = await solver.check();
if (result === 'sat') {
  const model = solver.model();
  console.log(model.get(x));
}
```

**OxiZ:**
```javascript
const result = solver.checkSat();
if (result === 'sat') {
  const model = solver.getModel();
  console.log(model.x.value);
}
```

## Feature Comparison

| Feature | Z3 | OxiZ | Notes |
|---------|-----|------|-------|
| Basic solving | ✅ | ✅ | Both support SAT/UNSAT/UNKNOWN |
| Integer arithmetic | ✅ | ✅ | QF_LIA logic |
| Real arithmetic | ✅ | ✅ | QF_LRA logic |
| Boolean logic | ✅ | ✅ | QF_UF logic |
| Bitvectors | ✅ | ✅ | QF_BV logic |
| Quantifiers | ✅ | ⚠️ | Limited support |
| Push/Pop | ✅ | ✅ | Incremental solving |
| Unsat cores | ✅ | ✅ | With appropriate options |
| Model generation | ✅ | ✅ | Full model extraction |
| Simplification | ✅ | ✅ | Expression simplification |
| Optimization | ✅ | 🔜 | Planned feature |
| Proof generation | ✅ | 🔜 | Planned feature |

## Complete Examples

### Example 1: Basic Integer Arithmetic

**Z3:**
```javascript
const { init } = require('z3-solver');

async function solve() {
  const { Context } = await init();
  const ctx = new Context('main');
  const solver = ctx.solver();

  solver.set('logic', 'QF_LIA');

  const x = ctx.Int.const('x');
  const y = ctx.Int.const('y');

  solver.add(x.gt(0));
  solver.add(y.lt(10));
  solver.add(x.add(y).eq(15));

  const result = await solver.check();
  if (result === 'sat') {
    const model = solver.model();
    console.log('x =', model.get(x));
    console.log('y =', model.get(y));
  }
}
```

**OxiZ:**
```javascript
import init, { WasmSolver } from 'oxiz-wasm';

async function solve() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');

  solver.declareConst('x', 'Int');
  solver.declareConst('y', 'Int');

  solver.assertFormula('(> x 0)');
  solver.assertFormula('(< y 10)');
  solver.assertFormula('(= (+ x y) 15)');

  const result = solver.checkSat();
  if (result === 'sat') {
    const model = solver.getModel();
    console.log('x =', model.x.value);
    console.log('y =', model.y.value);
  }
}
```

### Example 2: Incremental Solving

**Z3:**
```javascript
solver.push();
solver.add(x.gt(100));
await solver.check(); // Check with additional constraint
solver.pop();
await solver.check(); // Check without additional constraint
```

**OxiZ:**
```javascript
solver.push();
solver.assertFormula('(> x 100)');
solver.checkSat(); // Check with additional constraint
solver.pop();
solver.checkSat(); // Check without additional constraint
```

### Example 3: Boolean Logic

**Z3:**
```javascript
const p = ctx.Bool.const('p');
const q = ctx.Bool.const('q');
const r = ctx.Bool.const('r');

solver.add(p.implies(q));
solver.add(q.implies(r));
solver.add(p);
solver.add(r.not());

const result = await solver.check(); // unsat
```

**OxiZ:**
```javascript
solver.setLogic('QF_UF');
solver.declareConst('p', 'Bool');
solver.declareConst('q', 'Bool');
solver.declareConst('r', 'Bool');

solver.assertFormula('(=> p q)');
solver.assertFormula('(=> q r)');
solver.assertFormula('p');
solver.assertFormula('(not r)');

const result = solver.checkSat(); // "unsat"
```

### Example 4: Getting Unsat Core

**Z3:**
```javascript
solver.set('unsat_core', true);
// ... add assertions with tracking ...
const result = await solver.check();
if (result === 'unsat') {
  const core = solver.unsatCore();
  console.log(core);
}
```

**OxiZ:**
```javascript
solver.setOption('produce-unsat-cores', 'true');
// ... add assertions ...
const result = solver.checkSat();
if (result === 'unsat') {
  const core = solver.getUnsatCore();
  console.log(core);
}
```

## API Translation Table

| Z3 API | OxiZ API | Notes |
|--------|----------|-------|
| `new Context()` | `new WasmSolver()` | Direct solver creation |
| `solver.set(key, value)` | `solver.setOption(key, value)` | Option setting |
| `solver.add(expr)` | `solver.assertFormula(expr)` | Takes SMT-LIB2 string |
| `await solver.check()` | `solver.checkSat()` | Synchronous in OxiZ |
| `solver.model()` | `solver.getModel()` | Returns JS object |
| `solver.push()` | `solver.push()` | Same |
| `solver.pop()` | `solver.pop()` | Same |
| `solver.reset()` | `solver.reset()` | Same |
| `ctx.Int.const(name)` | `solver.declareConst(name, 'Int')` | String-based |
| `ctx.Bool.const(name)` | `solver.declareConst(name, 'Bool')` | String-based |
| `x.gt(y)` | `'(> x y)'` | SMT-LIB2 syntax |
| `x.lt(y)` | `'(< x y)'` | SMT-LIB2 syntax |
| `x.eq(y)` | `'(= x y)'` | SMT-LIB2 syntax |
| `x.add(y)` | `'(+ x y)'` | SMT-LIB2 syntax |
| `x.mul(y)` | `'(* x y)'` | SMT-LIB2 syntax |
| `p.and(q)` | `'(and p q)'` | SMT-LIB2 syntax |
| `p.or(q)` | `'(or p q)'` | SMT-LIB2 syntax |
| `p.not()` | `'(not p)'` | SMT-LIB2 syntax |
| `p.implies(q)` | `'(=> p q)'` | SMT-LIB2 syntax |

## SMT-LIB2 Syntax Quick Reference

OxiZ uses SMT-LIB2 syntax for formulas. Here's a quick reference:

### Arithmetic Operators
- Addition: `(+ x y)`
- Subtraction: `(- x y)`
- Multiplication: `(* x y)`
- Division: `(/ x y)` (integer or real)
- Modulo: `(mod x y)`
- Absolute value: `(abs x)`

### Comparison Operators
- Equal: `(= x y)`
- Not equal: `(distinct x y)`
- Greater than: `(> x y)`
- Less than: `(< x y)`
- Greater or equal: `(>= x y)`
- Less or equal: `(<= x y)`

### Boolean Operators
- And: `(and p q)`
- Or: `(or p q)`
- Not: `(not p)`
- Implies: `(=> p q)`
- If-then-else: `(ite p x y)`

## Performance Considerations

### Z3
- Runs in a separate thread/worker for async operations
- May have higher memory overhead due to full Z3 feature set
- Supports complex theories and optimizations

### OxiZ
- Smaller WASM binary size
- Faster initialization
- Lower memory footprint
- Optimized for web environments
- Built-in support for Web Workers

## Migration Checklist

- [ ] Replace Z3 imports with OxiZ imports
- [ ] Update initialization code
- [ ] Convert declarative API to SMT-LIB2 strings
- [ ] Update assertion syntax
- [ ] Update model extraction code
- [ ] Test with your specific use cases
- [ ] Update error handling (OxiZ has typed errors)
- [ ] Review performance characteristics
- [ ] Update documentation/comments

## Common Pitfalls

### 1. Forgetting to Initialize
```javascript
// ❌ Wrong
const solver = new WasmSolver();

// ✅ Correct
await init();
const solver = new WasmSolver();
```

### 2. Using JavaScript Operators Instead of SMT-LIB2
```javascript
// ❌ Wrong (this won't work)
solver.assertFormula('x > 0');

// ✅ Correct
solver.assertFormula('(> x 0)');
```

### 3. Not Setting Logic
```javascript
// ⚠️ Works but suboptimal
solver.declareConst('x', 'Int');

// ✅ Better - enables optimizations
solver.setLogic('QF_LIA');
solver.declareConst('x', 'Int');
```

### 4. Forgetting to Declare Variables
```javascript
// ❌ Wrong - y not declared
solver.declareConst('x', 'Int');
solver.assertFormula('(> (+ x y) 0)'); // Error!

// ✅ Correct
solver.declareConst('x', 'Int');
solver.declareConst('y', 'Int');
solver.assertFormula('(> (+ x y) 0)');
```

## Getting Help

- **Documentation**: Check the OxiZ WASM README
- **Examples**: See the `examples/` directory
- **Issues**: Report bugs on GitHub
- **SMT-LIB2 Reference**: See `SMTLIB2_REFERENCE.md`

## Additional Resources

- [SMT-LIB2 Standard](http://smtlib.cs.uiowa.edu/)
- [OxiZ Examples](../examples/)
- [Web Worker Integration](../examples/webworker/)
- [Framework Wrappers](../wrappers/)
