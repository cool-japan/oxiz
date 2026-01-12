# OxiZ WASM API Reference

Complete reference documentation for the OxiZ WASM API.

## Table of Contents

- [WasmSolver](#wasmsolver)
  - [Constructor](#constructor)
  - [Core Methods](#core-methods)
  - [Declaration Methods](#declaration-methods)
  - [Definition Methods](#definition-methods)
  - [Assertion Methods](#assertion-methods)
  - [Query Methods](#query-methods)
  - [Stack Methods](#stack-methods)
  - [Configuration Methods](#configuration-methods)
  - [Debugging Methods](#debugging-methods)
  - [Async Methods](#async-methods)
- [Error Types](#error-types)
- [Utility Functions](#utility-functions)

---

## WasmSolver

The main solver class for interacting with OxiZ from JavaScript/TypeScript.

### Constructor

#### `new WasmSolver()`

Creates a new solver instance.

**Returns:** `WasmSolver`

**Example:**
```javascript
const solver = new WasmSolver();
```

---

### Core Methods

#### `execute(script: string): string`

Executes an SMT-LIB2 script and returns the output.

**Parameters:**
- `script` (string): An SMT-LIB2 script

**Returns:** `string` - The output from executing the script

**Throws:** Error if the script is invalid or execution fails

**Example:**
```javascript
const script = `
  (set-logic QF_LIA)
  (declare-const x Int)
  (assert (> x 0))
  (check-sat)
`;
const result = solver.execute(script);
console.log(result); // "sat"
```

---

#### `setLogic(logic: string): void`

Sets the SMT logic for the solver.

**Parameters:**
- `logic` (string): The SMT-LIB2 logic name

**Common Logics:**
- `QF_UF` - Quantifier-free uninterpreted functions
- `QF_LIA` - Quantifier-free linear integer arithmetic
- `QF_LRA` - Quantifier-free linear real arithmetic
- `QF_BV` - Quantifier-free bitvectors
- `ALL` - All supported theories

**Example:**
```javascript
solver.setLogic("QF_LIA");
```

---

#### `checkSat(): string`

Checks satisfiability of the current assertions.

**Returns:** `string` - One of: `"sat"`, `"unsat"`, or `"unknown"`

**Example:**
```javascript
solver.declareConst("p", "Bool");
solver.assertFormula("p");
const result = solver.checkSat();
if (result === "sat") {
  const model = solver.getModel();
  console.log(model);
}
```

---

#### `checkSatAssuming(assumptions: string[]): string`

Checks satisfiability under temporary assumptions without modifying the assertion stack.

**Parameters:**
- `assumptions` (string[]): Array of SMT-LIB2 boolean expressions

**Returns:** `string` - One of: `"sat"`, `"unsat"`, or `"unknown"`

**Throws:** Error if assumptions array is empty or contains invalid formulas

**Example:**
```javascript
solver.declareConst("p", "Bool");
solver.declareConst("q", "Bool");
solver.assertFormula("(or p q)");

// Check with assumption p is true
const result1 = solver.checkSatAssuming(["p"]); // "sat"

// Check with both false
const result2 = solver.checkSatAssuming(["(not p)", "(not q)"]); // "unsat"
```

---

### Declaration Methods

#### `declareConst(name: string, sortName: string): void`

Declares a constant (0-ary function).

**Parameters:**
- `name` (string): The constant name
- `sortName` (string): The sort/type name

**Valid Sorts:**
- `Bool` - Boolean values
- `Int` - Integer values
- `Real` - Real number values
- `BitVecN` - Bitvector of width N (e.g., `BitVec32`)

**Throws:** Error if the sort is invalid

**Example:**
```javascript
solver.declareConst("x", "Int");
solver.declareConst("flag", "Bool");
solver.declareConst("bv", "BitVec32");
```

---

#### `declareFun(name: string, argSorts: string[], retSort: string): void`

Declares a function.

**Parameters:**
- `name` (string): The function name
- `argSorts` (string[]): Array of argument sort names
- `retSort` (string): Return sort name

**Note:** Currently only nullary functions (constants) are fully supported.

**Throws:** Error if sorts are invalid or non-nullary functions are used

**Example:**
```javascript
// Declare a constant (nullary function)
solver.declareFun("c", [], "Int");
```

---

### Definition Methods

#### `defineSort(name: string, sortName: string): void`

Creates an alias for an existing sort.

**Parameters:**
- `name` (string): The new sort alias name
- `sortName` (string): The existing sort to alias

**Throws:** Error if the base sort is invalid

**Example:**
```javascript
solver.defineSort("Word", "BitVec32");
solver.declareConst("reg", "Word"); // Uses BitVec32
```

---

#### `defineFun(name: string, params: string[], retSort: string, body: string): void`

Defines a function with an implementation.

**Parameters:**
- `name` (string): The function name
- `params` (string[]): Array of parameter specifications (format: "name sort")
- `retSort` (string): Return sort
- `body` (string): SMT-LIB2 expression for the function body

**Throws:** Error if parameters are malformed or body is invalid

**Example:**
```javascript
solver.setLogic("QF_LIA");

// Define a function that doubles its input
solver.defineFun("double", ["x Int"], "Int", "(* 2 x)");

// Define max of two integers
solver.defineFun("max2", ["a Int", "b Int"], "Int", "(ite (> a b) a b)");

// Use the functions
solver.declareConst("n", "Int");
solver.assertFormula("(= (double n) 10)");
```

---

### Assertion Methods

#### `assertFormula(formula: string): void`

Asserts a formula (adds it to the assertion stack).

**Parameters:**
- `formula` (string): SMT-LIB2 boolean expression

**Throws:** Error if formula is invalid or references undeclared variables

**Example:**
```javascript
solver.setLogic("QF_LIA");
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
solver.assertFormula("(< x 10)");
```

---

#### `assertFormulaSafe(formula: string): void`

Asserts a formula with enhanced error messages and suggestions.

**Parameters:**
- `formula` (string): SMT-LIB2 boolean expression

**Throws:** Error with helpful hints if formula is invalid

**Example:**
```javascript
try {
  solver.assertFormulaSafe("(> x y)"); // y not declared
} catch (e) {
  console.error(e.message); // Includes hint about declaring y
}
```

---

#### `validateFormula(formula: string): boolean`

Validates a formula without asserting it.

**Parameters:**
- `formula` (string): SMT-LIB2 expression to validate

**Returns:** `boolean` - `true` if valid

**Throws:** Error if formula is invalid

**Example:**
```javascript
solver.declareConst("x", "Int");

// Validate before asserting
try {
  solver.validateFormula("(> x 0)");
  solver.assertFormula("(> x 0)"); // Safe to assert
} catch (e) {
  console.error("Invalid formula:", e.message);
}
```

---

### Query Methods

#### `getModel(): object`

Returns the model as a JavaScript object.

**Returns:** Object mapping variable names to `{sort: string, value: string}`

**Throws:** Error if `checkSat()` didn't return `"sat"`

**Example:**
```javascript
solver.declareConst("x", "Int");
solver.assertFormula("(> x 5)");
if (solver.checkSat() === "sat") {
  const model = solver.getModel();
  console.log(model.x.value); // e.g., "6"
  console.log(model.x.sort);  // "Int"
}
```

---

#### `getModelString(): string`

Returns the model as an SMT-LIB2 formatted string.

**Returns:** `string` - SMT-LIB2 representation of the model

**Throws:** Error if `checkSat()` didn't return `"sat"`

**Example:**
```javascript
if (solver.checkSat() === "sat") {
  const modelStr = solver.getModelString();
  console.log(modelStr);
}
```

---

#### `getValue(terms: string[]): string`

Evaluates expressions in the current model.

**Parameters:**
- `terms` (string[]): Array of SMT-LIB2 expressions to evaluate

**Returns:** `string` - SMT-LIB2 representation of the values

**Throws:** Error if `checkSat()` didn't return `"sat"` or terms are invalid

**Example:**
```javascript
solver.declareConst("x", "Int");
solver.assertFormula("(> x 5)");
if (solver.checkSat() === "sat") {
  const values = solver.getValue(["x", "(+ x 1)", "(* x 2)"]);
  console.log(values);
}
```

---

#### `getUnsatCore(): string`

Returns the unsatisfiable core.

**Returns:** `string` - SMT-LIB2 representation of the unsat core

**Throws:** Error if `checkSat()` didn't return `"unsat"`

**Example:**
```javascript
solver.declareConst("p", "Bool");
solver.assertFormula("p");
solver.assertFormula("(not p)");
if (solver.checkSat() === "unsat") {
  const core = solver.getUnsatCore();
  console.log(core);
}
```

---

#### `getAssertions(): string`

Returns all current assertions as an SMT-LIB2 string.

**Returns:** `string` - SMT-LIB2 representation of assertions

**Example:**
```javascript
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
const assertions = solver.getAssertions();
console.log(assertions); // "((> x 0))"
```

---

#### `simplify(expr: string): string`

Simplifies an SMT-LIB2 expression.

**Parameters:**
- `expr` (string): Expression to simplify

**Returns:** `string` - Simplified expression

**Throws:** Error if expression is invalid

**Example:**
```javascript
const result = solver.simplify("(+ 1 2)"); // "3"
const result2 = solver.simplify("(and true false)"); // "false"
```

---

### Stack Methods

#### `push(): void`

Creates a new backtracking point.

**Example:**
```javascript
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
solver.push();
solver.assertFormula("(< x 5)");
console.log(solver.checkSat()); // "sat"
solver.pop(); // Remove (< x 5)
```

---

#### `pop(): void`

Backtracks to the previous backtracking point.

**Example:**
```javascript
solver.push();
solver.assertFormula("(> x 10)");
solver.pop(); // Remove (> x 10)
```

---

#### `reset(): void`

Completely resets the solver to initial state.

**Example:**
```javascript
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
solver.reset(); // Everything cleared
```

---

#### `resetAssertions(): void`

Removes all assertions but keeps declarations and options.

**Example:**
```javascript
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
solver.resetAssertions(); // x still declared, assertion removed
solver.assertFormula("(< x 0)"); // Can still use x
```

---

### Configuration Methods

#### `setOption(key: string, value: string): void`

Sets a solver option.

**Parameters:**
- `key` (string): Option name
- `value` (string): Option value

**Common Options:**
- `produce-models` - Enable model generation ("true"/"false")
- `produce-unsat-cores` - Enable unsat core generation ("true"/"false")

**Example:**
```javascript
solver.setOption("produce-models", "true");
solver.setOption("produce-unsat-cores", "true");
```

---

#### `getOption(key: string): string | undefined`

Gets the current value of a solver option.

**Parameters:**
- `key` (string): Option name

**Returns:** `string | undefined` - Option value or undefined if not set

**Example:**
```javascript
solver.setOption("produce-models", "true");
const value = solver.getOption("produce-models"); // "true"
```

---

#### `applyPreset(preset: string): void`

Applies a configuration preset.

**Parameters:**
- `preset` (string): Preset name

**Available Presets:**
- `default` - Default configuration with model production
- `fast` - Optimized for fast solving
- `complete` - All features enabled
- `debug` - Debugging configuration with verbose output
- `unsat-core` - Optimized for unsat core extraction
- `incremental` - Optimized for incremental solving

**Throws:** Error if preset is unknown

**Example:**
```javascript
solver.applyPreset("complete");
```

---

### Debugging Methods

#### `getStatistics(): object`

Returns solver statistics.

**Returns:** Object containing:
- `num_assertions` (number): Number of current assertions
- `last_result` (string): Last check-sat result
- `cancelled` (boolean): Cancellation status
- `logic` (string): Current logic (if set)

**Example:**
```javascript
const stats = solver.getStatistics();
console.log("Assertions:", stats.num_assertions);
console.log("Result:", stats.last_result);
```

---

#### `getInfo(key: string): string`

Gets solver metadata.

**Parameters:**
- `key` (string): Info key (with or without `:` prefix)

**Supported Keys:**
- `name` / `:name` - Solver name
- `version` / `:version` - Solver version
- `authors` / `:authors` - Authors
- `all-statistics` - Statistics availability
- `error-behavior` - Error handling behavior
- `reason-unknown` - Reason for unknown result

**Returns:** `string` - The requested information

**Throws:** Error if key is unknown

**Example:**
```javascript
const name = solver.getInfo("name"); // "OxiZ"
const version = solver.getInfo("version");
console.log(`${name} v${version}`);
```

---

#### `debugDump(): string`

Returns a comprehensive debug dump of solver state.

**Returns:** `string` - Formatted debug information

**Example:**
```javascript
solver.setLogic("QF_LIA");
solver.declareConst("x", "Int");
solver.assertFormula("(> x 0)");
solver.checkSat();

const dump = solver.debugDump();
console.log(dump);
```

---

#### `setTracing(enabled: boolean): void`

Enables or disables debug tracing.

**Parameters:**
- `enabled` (boolean): Whether to enable tracing

**Example:**
```javascript
solver.setTracing(true); // Enable tracing
solver.checkSat();
solver.setTracing(false); // Disable tracing
```

---

#### `getDiagnostics(): string[]`

Returns diagnostic warnings about solver usage.

**Returns:** `string[]` - Array of warning messages

**Example:**
```javascript
const warnings = solver.getDiagnostics();
if (warnings.length > 0) {
  console.warn("Solver warnings:");
  warnings.forEach(w => console.warn("  -", w));
}
```

---

#### `checkPattern(pattern: string): string`

Gets recommendations for usage patterns.

**Parameters:**
- `pattern` (string): Pattern name

**Supported Patterns:**
- `incremental` - Incremental solving patterns
- `assumptions` - When to use checkSatAssuming
- `async` - Async operation recommendations
- `validation` - Formula validation patterns

**Returns:** `string` - Recommendation message

**Example:**
```javascript
const rec = solver.checkPattern("assumptions");
console.log(rec);
```

---

#### `cancel(): void`

Requests cancellation of the current operation.

**Note:** This is a hint and may not take effect immediately.

**Example:**
```javascript
setTimeout(() => solver.cancel(), 5000); // Cancel after 5 seconds
```

---

#### `isCancelled(): boolean`

Checks if cancellation has been requested.

**Returns:** `boolean` - `true` if cancelled

**Example:**
```javascript
if (solver.isCancelled()) {
  console.log("Operation was cancelled");
}
```

---

### Async Methods

#### `checkSatAsync(): Promise<string>`

Async version of `checkSat()`.

**Returns:** `Promise<string>` - Resolves to `"sat"`, `"unsat"`, or `"unknown"`

**Example:**
```javascript
const result = await solver.checkSatAsync();
if (result === "sat") {
  const model = solver.getModel();
  console.log(model);
}
```

---

#### `executeAsync(script: string): Promise<string>`

Async version of `execute()`.

**Parameters:**
- `script` (string): SMT-LIB2 script

**Returns:** `Promise<string>` - Resolves to the script output

**Throws:** Promise rejects if script is invalid

**Example:**
```javascript
const script = `
  (set-logic QF_LIA)
  (declare-const x Int)
  (assert (> x 0))
  (check-sat)
`;
const result = await solver.executeAsync(script);
console.log(result);
```

---

## Error Types

All errors thrown by the API have the following structure:

```typescript
{
  kind: string,    // Error type (e.g., "ParseError", "InvalidSort")
  message: string  // Error message
}
```

**Error Kinds:**
- `ParseError` - Parse error in SMT-LIB2 script
- `InvalidSort` - Invalid sort name
- `NoModel` - No model available
- `NoUnsatCore` - No unsat core available
- `InvalidState` - Solver in invalid state
- `InvalidInput` - Invalid input or argument
- `NotSupported` - Operation not supported
- `Unknown` - Unknown error

---

## Utility Functions

#### `version(): string`

Returns the OxiZ WASM version.

**Returns:** `string` - Version in semver format

**Example:**
```javascript
import { version } from 'oxiz-wasm';
console.log(`OxiZ WASM version: ${version()}`);
```

---

## TypeScript Support

OxiZ WASM includes full TypeScript declarations. Import types as:

```typescript
import { WasmSolver, WasmErrorKind } from 'oxiz-wasm';

const solver = new WasmSolver();
solver.setLogic("QF_LIA");
```

---

## Best Practices

1. **Always set logic** for better performance
2. **Use `checkSatAssuming`** instead of push/pop for temporary constraints
3. **Validate user input** with `validateFormula()` or `assertFormulaSafe()`
4. **Use async methods** in browsers to avoid blocking the UI
5. **Check diagnostics** with `getDiagnostics()` to identify issues
6. **Apply presets** for quick configuration

---

## See Also

- [Tutorial (Beginner)](TUTORIAL_BEGINNER.md)
- [Tutorial (Intermediate)](TUTORIAL_INTERMEDIATE.md)
- [Migration from Z3](MIGRATION_FROM_Z3.md)
- [SMT-LIB2 Reference](SMTLIB2_REFERENCE.md)
