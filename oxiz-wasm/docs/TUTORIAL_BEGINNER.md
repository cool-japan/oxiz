# OxiZ WASM Tutorial: Beginner's Guide

Welcome to OxiZ! This tutorial will teach you the basics of using the OxiZ SMT solver in your JavaScript applications.

## What is an SMT Solver?

An SMT (Satisfiability Modulo Theories) solver is a tool that can answer questions like:
- "Is there a value of x that makes x > 5 AND x < 3 true?" (Answer: No, it's unsatisfiable)
- "Find values of x and y where x + y = 10 and x > 7" (Answer: x=8, y=2 is one solution)

SMT solvers are used in:
- Program verification
- Test case generation
- Constraint solving
- Optimization problems
- Automated theorem proving

## Installation

### For Web Browsers

```html
<!DOCTYPE html>
<html>
<head>
  <script type="module">
    import init, { WasmSolver } from './path/to/oxiz_wasm.js';

    async function main() {
      await init();
      const solver = new WasmSolver();
      // ... use solver ...
    }

    main();
  </script>
</head>
<body>
  <h1>OxiZ Example</h1>
</body>
</html>
```

### For Node.js

```bash
npm install oxiz-wasm
```

```javascript
import init, { WasmSolver } from 'oxiz-wasm';

async function main() {
  await init();
  const solver = new WasmSolver();
  // ... use solver ...
}

main();
```

## Your First Program

Let's solve a simple problem: "Find a number x where x > 5"

```javascript
import init, { WasmSolver } from 'oxiz-wasm';

async function firstExample() {
  // 1. Initialize the WASM module
  await init();

  // 2. Create a solver instance
  const solver = new WasmSolver();

  // 3. Set the logic (QF_LIA = Quantifier-Free Linear Integer Arithmetic)
  solver.setLogic('QF_LIA');

  // 4. Declare a variable
  solver.declareConst('x', 'Int');

  // 5. Add a constraint
  solver.assertFormula('(> x 5)');

  // 6. Check if there's a solution
  const result = solver.checkSat();
  console.log('Result:', result); // "sat" (satisfiable)

  // 7. Get the solution
  if (result === 'sat') {
    const model = solver.getModel();
    console.log('x =', model.x.value);
  }
}

firstExample();
```

**Output:**
```
Result: sat
x = 6
```

## Understanding the Code

Let's break down each step:

### 1. Initialize the WASM Module

```javascript
await init();
```

This loads the WebAssembly module. You must do this before creating any solvers.

### 2. Create a Solver

```javascript
const solver = new WasmSolver();
```

Creates a new solver instance. Each solver maintains its own state.

### 3. Set the Logic

```javascript
solver.setLogic('QF_LIA');
```

Tells the solver what kind of math you'll be using:
- `QF_LIA` - Integer arithmetic (whole numbers)
- `QF_LRA` - Real arithmetic (decimals)
- `QF_UF` - Boolean logic
- `QF_BV` - Bitvectors

### 4. Declare Variables

```javascript
solver.declareConst('x', 'Int');
```

Creates a variable named 'x' of type 'Int' (integer).

### 5. Add Constraints

```javascript
solver.assertFormula('(> x 5)');
```

Adds a constraint. The formula uses SMT-LIB2 syntax:
- `(> x 5)` means "x is greater than 5"
- Operators come first: `(operator operand1 operand2)`

### 6. Check Satisfiability

```javascript
const result = solver.checkSat();
```

Asks the solver: "Can all constraints be satisfied?"
Returns:
- `"sat"` - Yes, there's a solution
- `"unsat"` - No, it's impossible
- `"unknown"` - The solver couldn't determine

### 7. Get the Solution

```javascript
const model = solver.getModel();
```

If satisfiable, gets a solution. Returns an object like:
```javascript
{
  x: { sort: "Int", value: "6" }
}
```

## Example 2: Two Variables

Let's solve: "Find x and y where x + y = 10 and x > y"

```javascript
async function twoVariables() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');

  // Declare two variables
  solver.declareConst('x', 'Int');
  solver.declareConst('y', 'Int');

  // Add constraints
  solver.assertFormula('(= (+ x y) 10)'); // x + y = 10
  solver.assertFormula('(> x y)');        // x > y

  const result = solver.checkSat();
  console.log('Result:', result);

  if (result === 'sat') {
    const model = solver.getModel();
    console.log('x =', model.x.value);
    console.log('y =', model.y.value);
    console.log('Sum:', parseInt(model.x.value) + parseInt(model.y.value));
  }
}
```

**Output:**
```
Result: sat
x = 6
y = 4
Sum: 10
```

## Example 3: Boolean Logic

Let's solve a logic puzzle: "If it's raining, then it's cloudy. It's raining. Is it cloudy?"

```javascript
async function booleanLogic() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_UF'); // For boolean logic

  // Declare boolean variables
  solver.declareConst('raining', 'Bool');
  solver.declareConst('cloudy', 'Bool');

  // If raining then cloudy
  solver.assertFormula('(=> raining cloudy)');

  // It's raining
  solver.assertFormula('raining');

  // Is it cloudy?
  const result = solver.checkSat();

  if (result === 'sat') {
    const model = solver.getModel();
    console.log('Raining:', model.raining.value);
    console.log('Cloudy:', model.cloudy.value);
  }
}
```

**Output:**
```
Raining: true
Cloudy: true
```

## Example 4: Finding Impossible Constraints

What if there's no solution?

```javascript
async function unsatisfiable() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');
  solver.declareConst('x', 'Int');

  // These constraints can't both be true!
  solver.assertFormula('(> x 10)');  // x > 10
  solver.assertFormula('(< x 5)');   // x < 5

  const result = solver.checkSat();
  console.log('Result:', result); // "unsat"

  if (result === 'unsat') {
    console.log('No solution exists!');
  }
}
```

**Output:**
```
Result: unsat
No solution exists!
```

## Common SMT-LIB2 Syntax

Here are the most common operations:

### Arithmetic
```javascript
// Addition
'(+ x y)'        // x + y
'(+ 1 2 3)'      // 1 + 2 + 3

// Subtraction
'(- x y)'        // x - y
'(- x)'          // -x (negation)

// Multiplication
'(* x y)'        // x * y

// Division (integer)
'(div x y)'      // x / y (rounds down)

// Modulo
'(mod x y)'      // x % y
```

### Comparison
```javascript
'(= x y)'        // x == y
'(< x y)'        // x < y
'(<= x y)'       // x <= y
'(> x y)'        // x > y
'(>= x y)'       // x >= y
'(distinct x y)' // x != y
```

### Boolean
```javascript
'(and p q)'      // p AND q
'(or p q)'       // p OR q
'(not p)'        // NOT p
'(=> p q)'       // p IMPLIES q (if p then q)
```

## Error Handling

Always handle potential errors:

```javascript
async function withErrorHandling() {
  try {
    await init();
    const solver = new WasmSolver();

    // This will fail - 'x' not declared
    solver.assertFormula('(> x 0)');
  } catch (error) {
    console.error('Error:', error.message);
  }
}
```

Better approach - validate first:

```javascript
async function withValidation() {
  await init();
  const solver = new WasmSolver();
  solver.setLogic('QF_LIA');

  try {
    // Validate before asserting
    solver.validateFormula('(> x 0)');
    solver.assertFormula('(> x 0)');
  } catch (error) {
    console.error('Invalid formula:', error.message);
    // Suggests: "Make sure all variables are declared..."
  }
}
```

## Practical Example: Sudoku Cell

Let's check if a Sudoku cell can be filled:

```javascript
async function sudokuCell() {
  await init();
  const solver = new WasmSolver();

  solver.setLogic('QF_LIA');
  solver.declareConst('cell', 'Int');

  // Sudoku rules for a cell:
  // - Must be between 1 and 9
  solver.assertFormula('(>= cell 1)');
  solver.assertFormula('(<= cell 9)');

  // Can't be 5 (already in row)
  solver.assertFormula('(distinct cell 5)');

  // Can't be 3 (already in column)
  solver.assertFormula('(distinct cell 3)');

  // Can't be 7 (already in box)
  solver.assertFormula('(distinct cell 7)');

  const result = solver.checkSat();

  if (result === 'sat') {
    const model = solver.getModel();
    console.log('Valid value for cell:', model.cell.value);
  } else {
    console.log('No valid value for this cell!');
  }
}
```

## Next Steps

Now that you understand the basics:

1. Read the [Intermediate Tutorial](TUTORIAL_INTERMEDIATE.md) to learn about:
   - Incremental solving with push/pop
   - Working with multiple solvers
   - Performance optimization

2. Check the [SMT-LIB2 Reference](SMTLIB2_REFERENCE.md) for all operators

3. See [Examples](../examples/) for real-world use cases

## Exercises

Try solving these problems yourself:

1. Find two numbers where their sum is 20 and their difference is 4
2. Check if a number can be both even (divisible by 2) and odd
3. Find three numbers x, y, z where x + y + z = 15 and x < y < z

Solutions are in `examples/exercises/`.

## Tips for Beginners

1. **Always set logic first** - It enables optimizations
2. **Declare before use** - Variables must be declared before appearing in formulas
3. **Use parentheses** - Every operation needs parentheses in SMT-LIB2
4. **Start simple** - Begin with one or two variables
5. **Validate formulas** - Use `validateFormula()` to catch errors early
6. **Check diagnostics** - Use `getDiagnostics()` for usage tips

## Getting Help

- Check the [FAQ](FAQ.md)
- See [Common Errors](COMMON_ERRORS.md)
- Read the [API Reference](API_REFERENCE.md)
- Ask on GitHub Issues

Happy solving! 🎉
