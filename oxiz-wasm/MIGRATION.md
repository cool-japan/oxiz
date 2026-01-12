# Migration Guide: Z3 JavaScript Bindings to OxiZ WASM

This guide helps you migrate from Z3's JavaScript bindings to OxiZ WASM.

## Table of Contents

- [Overview](#overview)
- [Key Differences](#key-differences)
- [API Mapping](#api-mapping)
- [Common Migration Patterns](#common-migration-patterns)
- [Feature Comparison](#feature-comparison)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

OxiZ WASM is a pure Rust implementation of an SMT solver compiled to WebAssembly, providing:

- **Pure Rust/WASM**: No C++ dependencies, smaller binary size
- **Modern API**: Simpler, more JavaScript-friendly API
- **Better TypeScript Support**: Full TypeScript declarations included
- **Async Support**: Built-in async methods for long-running operations
- **Memory Efficiency**: Optimized memory pooling and string handling

## Key Differences

### Architecture

**Z3:**
- C++ codebase compiled to WASM via Emscripten
- Large binary size (several MB)
- Complex initialization with memory management

**OxiZ:**
- Pure Rust compiled to WASM via wasm-bindgen
- Smaller binary size
- Simple initialization, automatic memory management

### API Style

**Z3:**
- Object-oriented API with separate Context, Solver, and Term objects
- Explicit reference counting and memory management
- Complex type hierarchy

**OxiZ:**
- Simplified single `WasmSolver` object
- Automatic memory management
- String-based formulas (SMT-LIB2 syntax)

## API Mapping

### Initialization

**Z3:**
```javascript
const { init } = require('z3-solver');
const { Context } = await init();
const ctx = new Context('main');
const solver = new ctx.Solver();
```

**OxiZ:**
```javascript
const { WasmSolver } = require('oxiz-wasm');
const solver = new WasmSolver();
```

### Declaring Variables

**Z3:**
```javascript
const x = ctx.Int.const('x');
const y = ctx.Bool.const('y');
const bv = ctx.BitVec.const('bv', 32);
```

**OxiZ:**
```javascript
solver.declareConst('x', 'Int');
solver.declareConst('y', 'Bool');
solver.declareConst('bv', 'BitVec32');
```

### Setting Logic

**Z3:**
```javascript
solver.set('logic', 'QF_LIA');
```

**OxiZ:**
```javascript
solver.setLogic('QF_LIA');
```

### Adding Assertions

**Z3:**
```javascript
const x = ctx.Int.const('x');
solver.add(x.eq(42));
solver.add(x.gt(0));
```

**OxiZ:**
```javascript
solver.declareConst('x', 'Int');
solver.assertFormula('(= x 42)');
solver.assertFormula('(> x 0)');
```

### Checking Satisfiability

**Z3:**
```javascript
const result = await solver.check();
if (result === 'sat') {
  const model = solver.model();
  console.log(model.eval(x).value());
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

### Push/Pop

**Z3:**
```javascript
solver.push();
solver.add(constraint);
await solver.check();
solver.pop();
```

**OxiZ:**
```javascript
solver.push();
solver.assertFormula('constraint');
solver.checkSat();
solver.pop();
```

### Reset

**Z3:**
```javascript
solver.reset();
```

**OxiZ:**
```javascript
solver.reset(); // Clears everything
solver.resetAssertions(); // Keeps declarations
```

### Simplification

**Z3:**
```javascript
const x = ctx.Int.const('x');
const expr = x.add(1).add(2);
const simplified = expr.simplify();
```

**OxiZ:**
```javascript
const result = solver.simplify('(+ (+ x 1) 2)');
```

## Common Migration Patterns

### Pattern 1: Basic SAT Checking

**Z3:**
```javascript
const { init } = require('z3-solver');

async function checkSat() {
  const { Context } = await init();
  const ctx = new Context('main');
  const solver = new ctx.Solver();

  const p = ctx.Bool.const('p');
  const q = ctx.Bool.const('q');

  solver.add(p.or(q));
  solver.add(p.not());

  const result = await solver.check();
  console.log(result); // 'sat'

  if (result === 'sat') {
    const model = solver.model();
    console.log('q =', model.eval(q).value());
  }
}
```

**OxiZ:**
```javascript
const { WasmSolver } = require('oxiz-wasm');

function checkSat() {
  const solver = new WasmSolver();

  solver.declareConst('p', 'Bool');
  solver.declareConst('q', 'Bool');

  solver.assertFormula('(or p q)');
  solver.assertFormula('(not p)');

  const result = solver.checkSat();
  console.log(result); // 'sat'

  if (result === 'sat') {
    const model = solver.getModel();
    console.log('q =', model.q.value);
  }
}
```

### Pattern 2: Integer Arithmetic

**Z3:**
```javascript
const x = ctx.Int.const('x');
const y = ctx.Int.const('y');

solver.add(x.eq(10));
solver.add(y.eq(20));
solver.add(x.add(y).eq(30));

const result = await solver.check();
```

**OxiZ:**
```javascript
solver.declareConst('x', 'Int');
solver.declareConst('y', 'Int');

solver.assertFormula('(= x 10)');
solver.assertFormula('(= y 20)');
solver.assertFormula('(= (+ x y) 30)');

const result = solver.checkSat();
```

### Pattern 3: Using Push/Pop for Backtracking

**Z3:**
```javascript
solver.add(baseConstraints);

solver.push();
solver.add(temporaryConstraint);
const result1 = await solver.check();
solver.pop();

solver.push();
solver.add(anotherConstraint);
const result2 = await solver.check();
solver.pop();
```

**OxiZ:**
```javascript
solver.assertFormula('base-constraints');

solver.push();
solver.assertFormula('temporary-constraint');
const result1 = solver.checkSat();
solver.pop();

solver.push();
solver.assertFormula('another-constraint');
const result2 = solver.checkSat();
solver.pop();
```

### Pattern 4: BitVector Operations

**Z3:**
```javascript
const bv1 = ctx.BitVec.const('bv1', 8);
const bv2 = ctx.BitVec.const('bv2', 8);

solver.add(bv1.eq(ctx.BitVec.val(1, 8)));
solver.add(bv2.eq(ctx.BitVec.val(2, 8)));
```

**OxiZ:**
```javascript
solver.declareConst('bv1', 'BitVec8');
solver.declareConst('bv2', 'BitVec8');

solver.assertFormula('(= bv1 #b00000001)');
solver.assertFormula('(= bv2 #b00000010)');
```

## Feature Comparison

| Feature | Z3 | OxiZ | Notes |
|---------|----|----- |-------|
| Boolean Logic | ✓ | ✓ | Full support |
| Integer Arithmetic | ✓ | ✓ | Linear arithmetic |
| Real Arithmetic | ✓ | ✓ | Linear arithmetic |
| BitVectors | ✓ | ✓ | Basic operations |
| Arrays | ✓ | ⚠️ | Limited support |
| Strings | ✓ | ⚠️ | In development |
| Quantifiers | ✓ | ✗ | Not yet supported |
| Optimization | ✓ | ✗ | Not yet supported |
| Proof Generation | ✓ | ✗ | Not yet supported |
| Unsat Core | ✓ | ✓ | Supported |
| Model Extraction | ✓ | ✓ | Supported |
| Async API | ⚠️ | ✓ | Built-in |
| TypeScript | ⚠️ | ✓ | Full declarations |

## Performance Considerations

### Startup Time

**Z3:**
- Longer initialization due to Emscripten runtime
- Larger WASM binary to load

**OxiZ:**
- Faster initialization with wasm-bindgen
- Smaller binary size

### Memory Usage

**Z3:**
- Manual memory management required
- Potential memory leaks if not careful

**OxiZ:**
- Automatic memory management
- Built-in memory pooling
- More predictable memory usage

### Execution Speed

For simple problems, OxiZ may be faster due to:
- Lower overhead
- Optimized Rust code
- Better memory locality

For complex problems, Z3 may be faster due to:
- Mature solver algorithms
- Decades of optimization
- More solver strategies

### Recommendations

- **Use OxiZ for:**
  - Simple to medium complexity problems
  - Browser-based applications
  - TypeScript projects
  - Applications requiring predictable memory usage
  - Projects prioritizing small binary size

- **Use Z3 for:**
  - Complex problems requiring advanced solver features
  - Problems needing quantifiers or optimization
  - Applications requiring proof generation
  - Critical applications where Z3's maturity is important

## Troubleshooting

### Common Issues

#### Issue: "Cannot find module 'oxiz-wasm'"

**Solution:**
```bash
# Make sure to build the WASM package first
wasm-pack build --target nodejs
```

#### Issue: Formula parsing errors

**Z3:**
```javascript
// Object-oriented API
solver.add(x.eq(y));
```

**OxiZ:**
```javascript
// Must use SMT-LIB2 syntax
solver.assertFormula('(= x y)');
```

#### Issue: Model values different format

**Z3:**
```javascript
const value = model.eval(x).value(); // Number
```

**OxiZ:**
```javascript
const value = model.x.value; // String
```

#### Issue: Missing features (quantifiers, optimization)

**Solution:**
OxiZ is under active development. If you need these features:
1. Continue using Z3 for now
2. Watch the OxiZ repository for updates
3. Consider contributing to the project

### Getting Help

- **GitHub Issues**: https://github.com/cool-japan/oxiz/issues
- **Documentation**: See README.md and examples/
- **Z3 Documentation**: https://z3prover.github.io/api/html/namespacez3py.html

## Migration Checklist

- [ ] Install oxiz-wasm package
- [ ] Update imports to use `WasmSolver`
- [ ] Convert variable declarations to `declareConst()`
- [ ] Convert assertions to SMT-LIB2 string format
- [ ] Update check() calls to `checkSat()`
- [ ] Update model extraction to use `getModel()`
- [ ] Test async methods if needed
- [ ] Verify all features are supported
- [ ] Update TypeScript types if applicable
- [ ] Run integration tests
- [ ] Measure performance differences
- [ ] Update documentation

## Example: Complete Migration

### Before (Z3)

```javascript
const { init } = require('z3-solver');

async function solve() {
  const { Context } = await init();
  const ctx = new Context('main');
  const solver = new ctx.Solver();

  solver.set('logic', 'QF_LIA');

  const x = ctx.Int.const('x');
  const y = ctx.Int.const('y');

  solver.add(x.add(y).eq(10));
  solver.add(x.sub(y).eq(2));

  const result = await solver.check();

  if (result === 'sat') {
    const model = solver.model();
    console.log('x =', model.eval(x).value());
    console.log('y =', model.eval(y).value());
  }
}

solve();
```

### After (OxiZ)

```javascript
const { WasmSolver } = require('oxiz-wasm');

function solve() {
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');

  solver.declareConst('x', 'Int');
  solver.declareConst('y', 'Int');

  solver.assertFormula('(= (+ x y) 10)');
  solver.assertFormula('(= (- x y) 2)');

  const result = solver.checkSat();

  if (result === 'sat') {
    const model = solver.getModel();
    console.log('x =', model.x.value);
    console.log('y =', model.y.value);
  }
}

solve();
```

## Conclusion

OxiZ WASM offers a simpler, more modern API compared to Z3's JavaScript bindings. While it may not yet support all of Z3's advanced features, it's an excellent choice for many SMT solving tasks, especially in web environments where binary size and ease of use matter.

The migration process is straightforward for most applications, primarily involving:
1. Changing the initialization code
2. Converting to SMT-LIB2 string syntax for formulas
3. Updating model extraction code

For applications requiring advanced features like quantifiers or optimization, continue using Z3 while monitoring OxiZ's development progress.
