# OxiZ WASM TypeScript Examples

This directory contains TypeScript examples demonstrating type-safe usage of the OxiZ WASM SMT solver.

## Features Demonstrated

- **Type Safety**: Full TypeScript type definitions for all APIs
- **Error Handling**: Proper error handling with typed error objects
- **Async Operations**: Using async/await with the solver
- **Incremental Solving**: Using push/pop for backtracking
- **Multiple Theories**: Boolean logic, integer arithmetic, bitvectors

## Prerequisites

1. Build the WASM package:
   ```bash
   cd ../..
   wasm-pack build --target web
   ```

2. Install dependencies:
   ```bash
   cd examples/typescript
   npm install
   ```

## Running the Examples

```bash
# Build TypeScript
npm run build

# Run the examples
npm start

# Or do both in one command
npm run dev
```

## Examples Included

### basic.ts

Demonstrates:
- Boolean satisfiability checking
- Integer linear arithmetic
- Error handling with TypeScript
- Async operations
- Incremental solving with push/pop
- Bitvector operations

## Type Definitions

The project uses the TypeScript declarations from `oxiz-wasm.d.ts` which provide:

- **WasmSolver class**: Fully typed solver interface
- **Model interface**: Type-safe model extraction
- **SatResult type**: Union type for satisfiability results
- **Error types**: Structured error handling

## Example Usage Patterns

### Basic Solving

```typescript
import init, { WasmSolver } from 'oxiz-wasm';

await init();

const solver = new WasmSolver();
solver.setLogic('QF_LIA');
solver.declareConst('x', 'Int');
solver.assertFormula('(> x 0)');

const result = solver.checkSat(); // Type: "sat" | "unsat" | "unknown"
if (result === 'sat') {
    const model = solver.getModel(); // Type: Model
    console.log(model.x.value);
}
```

### Error Handling

```typescript
try {
    solver.declareConst('x', 'InvalidSort');
} catch (error: any) {
    console.error(`${error.kind}: ${error.message}`);
}
```

### Async Operations

```typescript
const result = await solver.checkSatAsync();
if (result === 'sat') {
    const model = solver.getModel();
    // Process model...
}
```

## Type Safety Benefits

TypeScript provides:
- Autocomplete for all solver methods
- Type checking for sort names
- Compile-time error detection
- Better IDE support
- Self-documenting code

## Building for Production

For production use:

1. Use release build of WASM:
   ```bash
   wasm-pack build --target web --release
   ```

2. Enable TypeScript strict mode (already enabled in tsconfig.json)

3. Build your TypeScript:
   ```bash
   npm run build
   ```

## IDE Integration

For VS Code:
1. Install "TypeScript" extension
2. Open the typescript folder
3. TypeScript will automatically use the type definitions
4. You'll get full autocomplete and type checking

## Troubleshooting

### Cannot find module 'oxiz-wasm'

Make sure you've built the WASM package first:
```bash
cd ../..
wasm-pack build --target web
```

### Type errors in node_modules

Run:
```bash
npm install
```

This will install the correct type definitions.

## Next Steps

- Check out `basic.ts` for comprehensive examples
- Read the API documentation in `../../oxiz-wasm.d.ts`
- Experiment with different SMT logics
- Try the browser examples in `../basic.html`
