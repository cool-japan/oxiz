# SMT-LIB2 Quick Reference for OxiZ

This guide provides a quick reference for SMT-LIB2 syntax used in OxiZ WASM.

## Table of Contents
- [Basics](#basics)
- [Logics](#logics)
- [Sorts (Types)](#sorts-types)
- [Operators](#operators)
- [Commands](#commands)
- [Examples](#examples)

## Basics

SMT-LIB2 uses prefix notation (also called Polish notation) where operators come before operands:

```
Standard notation: x + y
SMT-LIB2 notation: (+ x y)

Standard notation: x > 0 && y < 10
SMT-LIB2 notation: (and (> x 0) (< y 10))
```

## Logics

Common logics supported by OxiZ:

| Logic | Description | Use Cases |
|-------|-------------|-----------|
| `QF_UF` | Quantifier-Free Uninterpreted Functions | Boolean logic, equality |
| `QF_LIA` | Quantifier-Free Linear Integer Arithmetic | Integer constraints |
| `QF_LRA` | Quantifier-Free Linear Real Arithmetic | Real number constraints |
| `QF_NIA` | Quantifier-Free Nonlinear Integer Arithmetic | Polynomial constraints |
| `QF_BV` | Quantifier-Free Bitvectors | Bit-level operations |
| `ALL` | All theories | General problems |

Set logic with:
```javascript
solver.setLogic('QF_LIA');
```

## Sorts (Types)

### Built-in Sorts

| Sort | Description | Example Values |
|------|-------------|----------------|
| `Bool` | Boolean | `true`, `false` |
| `Int` | Integers | `0`, `42`, `-17` |
| `Real` | Real numbers | `0.0`, `3.14`, `-2.5` |
| `BitVec<N>` | N-bit bitvector | `#b101`, `#x1F` |

Declare constants:
```javascript
solver.declareConst('x', 'Int');
solver.declareConst('flag', 'Bool');
solver.declareConst('bv', 'BitVec32');
```

### Custom Sorts

Define sort aliases:
```javascript
solver.defineSort('Word', 'BitVec32');
```

## Operators

### Integer Arithmetic

| Operator | Syntax | Description | Example |
|----------|--------|-------------|---------|
| Addition | `(+ x y ...)` | Sum | `(+ 1 2 3)` → `6` |
| Subtraction | `(- x y)` | Difference | `(- 10 3)` → `7` |
| Negation | `(- x)` | Unary minus | `(- 5)` → `-5` |
| Multiplication | `(* x y ...)` | Product | `(* 2 3 4)` → `24` |
| Division | `(div x y)` | Integer division | `(div 7 2)` → `3` |
| Modulo | `(mod x y)` | Remainder | `(mod 7 3)` → `1` |
| Absolute | `(abs x)` | Absolute value | `(abs -5)` → `5` |

### Real Arithmetic

| Operator | Syntax | Description | Example |
|----------|--------|-------------|---------|
| Addition | `(+ x y ...)` | Sum | `(+ 1.5 2.5)` → `4.0` |
| Subtraction | `(- x y)` | Difference | `(- 10.0 3.5)` → `6.5` |
| Multiplication | `(* x y ...)` | Product | `(* 2.0 3.5)` → `7.0` |
| Division | `(/ x y)` | Real division | `(/ 7.0 2.0)` → `3.5` |

### Comparison Operators

| Operator | Syntax | Description | Example |
|----------|--------|-------------|---------|
| Equal | `(= x y)` | Equality | `(= 5 5)` → `true` |
| Distinct | `(distinct x y ...)` | Not equal | `(distinct 1 2 3)` → `true` |
| Less than | `(< x y)` | x < y | `(< 3 5)` → `true` |
| Less or equal | `(<= x y)` | x ≤ y | `(<= 5 5)` → `true` |
| Greater than | `(> x y)` | x > y | `(> 10 5)` → `true` |
| Greater or equal | `(>= x y)` | x ≥ y | `(>= 5 5)` → `true` |

### Boolean Operators

| Operator | Syntax | Description | Example |
|----------|--------|-------------|---------|
| And | `(and p q ...)` | Conjunction | `(and true false)` → `false` |
| Or | `(or p q ...)` | Disjunction | `(or true false)` → `true` |
| Not | `(not p)` | Negation | `(not true)` → `false` |
| Implies | `(=> p q)` | Implication | `(=> true false)` → `false` |
| XOR | `(xor p q)` | Exclusive or | `(xor true false)` → `true` |
| If-then-else | `(ite cond then else)` | Conditional | `(ite true 1 2)` → `1` |

### Bitvector Operators

| Operator | Syntax | Description |
|----------|--------|-------------|
| Bitwise AND | `(bvand x y)` | AND operation |
| Bitwise OR | `(bvor x y)` | OR operation |
| Bitwise NOT | `(bvnot x)` | NOT operation |
| Bitwise XOR | `(bvxor x y)` | XOR operation |
| Left shift | `(bvshl x y)` | Shift left |
| Right shift | `(bvlshr x y)` | Logical shift right |
| Arithmetic shift | `(bvashr x y)` | Arithmetic shift right |
| Addition | `(bvadd x y)` | Bitvector addition |
| Subtraction | `(bvsub x y)` | Bitvector subtraction |
| Multiplication | `(bvmul x y)` | Bitvector multiplication |

## Commands

### Declaration Commands

```javascript
// Declare a constant
solver.declareConst('x', 'Int');

// Declare a function (nullary only currently)
solver.declareFun('f', [], 'Int');

// Define a function with body
solver.defineFun('double', ['x Int'], 'Int', '(* 2 x)');

// Define a sort alias
solver.defineSort('Word', 'BitVec32');
```

### Assertion Commands

```javascript
// Assert a formula
solver.assertFormula('(> x 0)');

// Assert with validation
solver.assertFormulaSafe('(> x 0)');

// Validate without asserting
solver.validateFormula('(> x 0)');
```

### Solving Commands

```javascript
// Check satisfiability
const result = solver.checkSat(); // "sat", "unsat", or "unknown"

// Check with assumptions
const result = solver.checkSatAssuming(['(> x 5)', '(< y 10)']);

// Asynchronous check
const result = await solver.checkSatAsync();
```

### Model Commands

```javascript
// Get full model
const model = solver.getModel();

// Get model as string
const modelStr = solver.getModelString();

// Get specific values
const values = solver.getValue(['x', 'y', '(+ x y)']);
```

### Stack Commands

```javascript
// Push new context
solver.push();

// Pop context
solver.pop();

// Reset solver
solver.reset();

// Reset only assertions
solver.resetAssertions();
```

## Examples

### Example 1: Linear Integer Arithmetic

```javascript
solver.setLogic('QF_LIA');
solver.declareConst('x', 'Int');
solver.declareConst('y', 'Int');

// x + y = 10
solver.assertFormula('(= (+ x y) 10)');

// x > 0
solver.assertFormula('(> x 0)');

// y < 8
solver.assertFormula('(< y 8)');

const result = solver.checkSat(); // "sat"
if (result === 'sat') {
  const model = solver.getModel();
  console.log('x =', model.x.value);
  console.log('y =', model.y.value);
}
```

### Example 2: Boolean Logic

```javascript
solver.setLogic('QF_UF');
solver.declareConst('p', 'Bool');
solver.declareConst('q', 'Bool');
solver.declareConst('r', 'Bool');

// (p => q) ∧ (q => r) ∧ p ∧ ¬r
solver.assertFormula('(and (=> p q) (=> q r) p (not r))');

const result = solver.checkSat(); // "unsat"
if (result === 'unsat') {
  const core = solver.getUnsatCore();
  console.log('Unsat core:', core);
}
```

### Example 3: Conditional Expressions

```javascript
solver.setLogic('QF_LIA');
solver.declareConst('x', 'Int');
solver.declareConst('y', 'Int');
solver.declareConst('max', 'Int');

// max = if x > y then x else y
solver.assertFormula('(= max (ite (> x y) x y))');

// x = 5, y = 3
solver.assertFormula('(= x 5)');
solver.assertFormula('(= y 3)');

const result = solver.checkSat(); // "sat"
const model = solver.getModel();
console.log('max =', model.max.value); // 5
```

### Example 4: Nested Expressions

```javascript
solver.setLogic('QF_LIA');
solver.declareConst('a', 'Int');
solver.declareConst('b', 'Int');
solver.declareConst('c', 'Int');

// (a + b) * c = 20
solver.assertFormula('(= (* (+ a b) c) 20)');

// a > 0, b > 0, c > 0
solver.assertFormula('(and (> a 0) (> b 0) (> c 0))');

const result = solver.checkSat();
```

### Example 5: Define and Use Functions

```javascript
solver.setLogic('QF_LIA');

// Define double(x) = 2 * x
solver.defineFun('double', ['x Int'], 'Int', '(* 2 x)');

// Define max(a, b)
solver.defineFun('max', ['a Int', 'b Int'], 'Int', '(ite (> a b) a b)');

solver.declareConst('n', 'Int');

// double(n) = 10
solver.assertFormula('(= (double n) 10)');

const result = solver.checkSat(); // "sat"
const model = solver.getModel();
console.log('n =', model.n.value); // 5
```

### Example 6: Multiple Solutions

```javascript
solver.setLogic('QF_LIA');
solver.declareConst('x', 'Int');

solver.assertFormula('(> x 0)');
solver.assertFormula('(< x 10)');

const solutions = [];

while (true) {
  const result = solver.checkSat();
  if (result !== 'sat') break;

  const model = solver.getModel();
  const value = model.x.value;
  solutions.push(value);

  // Block this solution
  solver.assertFormula(`(distinct x ${value})`);

  if (solutions.length >= 5) break; // Limit solutions
}

console.log('Solutions:', solutions);
```

## Advanced Features

### Quantifiers (Limited Support)

```javascript
// Universal quantification
'(forall ((x Int)) (>= (* x x) 0))'

// Existential quantification
'(exists ((x Int)) (= (* x x) 4))'
```

### Let Bindings

```javascript
// Local variable binding
'(let ((temp (+ x y))) (* temp temp))'
```

### Named Expressions

```javascript
// Give names to sub-expressions for tracking
'(! (> x 0) :named pos_x)'
```

## Tips and Best Practices

1. **Set Logic Early**: Always call `setLogic()` before declarations for better performance
2. **Use Simplify**: Use `solver.simplify(expr)` to simplify complex expressions
3. **Validate First**: Use `validateFormula()` to check syntax before asserting
4. **Incremental Solving**: Use `push()`/`pop()` for exploring multiple scenarios
5. **Type Consistency**: Ensure operands have compatible types (Int with Int, Real with Real)
6. **Parentheses**: Every operation needs parentheses in SMT-LIB2
7. **Nesting**: You can nest expressions arbitrarily deep

## Common Errors

### Missing Parentheses
```
❌ > x 0
✅ (> x 0)
```

### Wrong Operator Order
```
❌ (x + y)
✅ (+ x y)
```

### Type Mismatch
```
❌ (+ 1 1.5)      // Mixing Int and Real
✅ (+ 1.0 1.5)    // Both Real
```

### Undeclared Variables
```
❌ solver.assertFormula('(> x 0)'); // x not declared
✅ solver.declareConst('x', 'Int');
   solver.assertFormula('(> x 0)');
```

## Resources

- [SMT-LIB2 Official Site](http://smtlib.cs.uiowa.edu/)
- [SMT-LIB2 Standard v2.6](http://smtlib.cs.uiowa.edu/papers/smt-lib-reference-v2.6-r2021-05-12.pdf)
- [OxiZ Examples](../examples/)
