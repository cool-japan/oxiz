# OxiZ WASM - WebAssembly Bindings for OxiZ SMT Solver

WebAssembly bindings for the OxiZ SMT solver, enabling high-performance SMT solving directly in web browsers and Node.js.

## Overview

This crate provides comprehensive JavaScript/TypeScript bindings for running OxiZ in web browsers and Node.js environments. It features a complete API with async support, Web Worker compatibility, and structured error handling.

### Key Features

- **Pure Rust**: 100% safe Rust code compiled to WebAssembly (no unsafe, no FFI)
- **Complete API**: Full SMT-LIB2 operations support (declarations, assertions, models, unsat cores)
- **Async Support**: Non-blocking operations with `async`/`await`
- **Web Worker Ready**: Run solver in background threads
- **Type-Safe Errors**: Structured error objects with detailed messages
- **Well Documented**: Comprehensive JSDoc comments for all APIs
- **Examples Included**: Interactive HTML demos and Web Worker examples
- **TypeScript Support**: Full type definitions and TypeScript examples
- **Performance Benchmarks**: Comprehensive benchmarking suite included
- **Build Automation**: Smart build script with optimization support

## Building

### Quick Start

```bash
# Using the build script (recommended)
./build.sh dev          # Fast development build
./build.sh release      # Optimized release build
./build.sh optimized    # Maximum optimization (requires wasm-opt)
./build.sh all          # Build for all targets
```

### Manual Building

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
wasm-pack build --target web

# Build for Node.js
wasm-pack build --target nodejs

# Build for bundlers (webpack, etc.)
wasm-pack build --target bundler
```

### Build Script Options

The `build.sh` script supports multiple build types:

- **dev** - Fast development build with debug info
- **release** - Standard optimized release build
- **profiling** - Release build with debug symbols
- **optimized** - Maximum size optimization using wasm-opt
- **all** - Build for all targets (web, nodejs, bundler)
- **clean** - Remove all build artifacts

**Note**: The `optimized` build requires [Binaryen](https://github.com/WebAssembly/binaryen) for wasm-opt:
```bash
# macOS
brew install binaryen

# Ubuntu/Debian
sudo apt-get install binaryen
```

## Usage

### JavaScript (ES Modules)

```javascript
import init, { WasmSolver, version } from './pkg/oxiz_wasm.js';

async function main() {
    await init();

    console.log(`OxiZ version: ${version()}`);

    const solver = new WasmSolver();
    solver.setLogic("QF_LIA");

    const result = solver.execute(`
        (declare-const x Int)
        (declare-const y Int)
        (assert (> x 0))
        (assert (< y 10))
        (assert (= (+ x y) 15))
        (check-sat)
    `);

    console.log(result);  // "sat"
}

main();
```

### TypeScript

```typescript
import init, { WasmSolver, version, type Model, type SatResult } from './pkg/oxiz_wasm';

async function solve(): Promise<void> {
    await init();

    const solver = new WasmSolver();
    solver.setLogic("QF_LIA");

    // TypeScript provides full type safety
    solver.declareConst('x', 'Int');  // 'Int' is validated by TypeScript
    solver.assertFormula('(> x 0)');

    const result: SatResult = solver.checkSat();  // Type: "sat" | "unsat" | "unknown"

    if (result === 'sat') {
        const model: Model = solver.getModel();
        console.log(model.x.value);  // Autocomplete works here!
        console.log(model.x.sort);   // Type: string
    }
}
```

The package includes comprehensive TypeScript declarations (`oxiz-wasm.d.ts`) providing:
- Full type safety for all API methods
- IntelliSense/autocomplete support
- Type-checked sort names and logic names
- Structured error types
- Complete JSDoc documentation

### Node.js (CommonJS)

```javascript
const { WasmSolver, version } = require('./pkg/oxiz_wasm');

const solver = new WasmSolver();
console.log(solver.checkSat());
```

## API Reference

### `WasmSolver`

Main solver class providing access to all SMT operations.

```typescript
class WasmSolver {
    // Constructor
    constructor();

    // Script execution
    execute(script: string): string;
    executeAsync(script: string): Promise<string>;

    // Logic and options
    setLogic(logic: string): void;
    setOption(key: string, value: string): void;
    getOption(key: string): string | undefined;

    // Variable and function declarations
    declareConst(name: string, sort: string): void;  // sorts: "Bool", "Int", "Real", "BitVecN"
    declareFun(name: string, argSorts: string[], retSort: string): void;

    // Assertions
    assertFormula(formula: string): void;
    getAssertions(): string;
    resetAssertions(): void;

    // Satisfiability checking
    checkSat(): string;  // "sat" | "unsat" | "unknown"
    checkSatAsync(): Promise<string>;

    // Model extraction (after sat result)
    getModel(): object;  // { varName: { sort: string, value: string } }
    getModelString(): string;
    getValue(terms: string[]): string;

    // Unsat core (after unsat result)
    getUnsatCore(): string;

    // Context management
    push(): void;
    pop(): void;
    reset(): void;

    // Utilities
    simplify(expr: string): string;
    cancel(): void;
    isCancelled(): boolean;
}
```

### Global Functions

```typescript
// Get OxiZ WASM version
function version(): string;

// Initialize WASM module (auto-called on import)
function init(): Promise<void>;
```

### Error Objects

All operations that can fail throw structured error objects:

```typescript
interface WasmError {
    kind: "ParseError" | "InvalidSort" | "InvalidInput" |
          "NoModel" | "NoUnsatCore" | "InvalidState" |
          "NotSupported" | "Unknown";
    message: string;
}
```

## Examples

The [examples](./examples) directory contains complete working examples:

### Browser Examples

- **basic.html**: Interactive browser examples demonstrating:
  - Boolean satisfiability
  - Integer linear arithmetic
  - Unsatisfiable formulas and unsat cores
  - Async operations
  - Custom SMT-LIB2 scripts

- **worker.html**: Web Worker demonstration showing:
  - Running solver in background thread
  - UI responsiveness during solving
  - Progress reporting and cancellation

- **benchmark.html**: Performance benchmarking suite with:
  - Initialization benchmarks
  - Declaration and assertion benchmarks
  - SAT solving benchmarks
  - Model extraction benchmarks
  - Scalability tests

- **solver-worker.js**: Reusable Web Worker implementation

### TypeScript Examples

- **typescript/basic.ts**: Comprehensive TypeScript examples with:
  - Full type safety and autocomplete
  - Error handling with typed errors
  - Async operations
  - Incremental solving
  - Bitvector operations

See [examples/README.md](./examples/README.md) and [examples/typescript/README.md](./examples/typescript/README.md) for detailed usage patterns and setup instructions.

## Browser Compatibility

Requires modern browser with WebAssembly support:
- Chrome/Edge 85+
- Firefox 78+
- Safari 14+
- Node.js 14+

## Performance Tips

- Use **release builds** with optimizations for production
- Use **async methods** (`checkSatAsync()`, `executeAsync()`) for long-running operations
- Run solver in **Web Worker** for truly non-blocking computation
- **Reuse solver instances** when solving multiple related problems
- Use **push/pop** for incremental solving instead of creating new solvers

## Limitations

Current limitations (see [TODO.md](./TODO.md) for planned features):
- Non-nullary functions not yet supported (requires AST changes)
- Quantifier support is limited
- Some advanced SMT-LIB2 features not implemented

## Bundle Size

The WASM binary is approximately 200KB gzipped (may vary based on build options).

## Development

### Running Tests

```bash
# WASM tests (requires wasm-pack)
wasm-pack test --headless --firefox
wasm-pack test --headless --chrome
```

### Code Structure

- `src/lib.rs` - Main WASM bindings implementation
- `examples/` - Browser and Web Worker examples
- `TODO.md` - Planned features and improvements
- `SCIRS2_POLICY.md` - Dependency policy documentation

## Contributing

Contributions welcome! Please ensure:
- All code is safe Rust (no `unsafe`)
- Add tests for new features
- Update documentation
- Follow existing code style
- Adhere to the "no warnings" policy

## Related Crates

- [oxiz-core](../oxiz-core) - Core SMT solver implementation
- [oxiz-solver](../oxiz-solver) - High-level solver interface
- [oxiz-cli](../oxiz-cli) - Command-line interface

## License

Licensed under Apache License 2.0 ([LICENSE](../LICENSE)).
