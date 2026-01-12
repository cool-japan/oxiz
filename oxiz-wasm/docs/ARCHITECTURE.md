# OxiZ WASM Architecture

This document describes the architecture and design of the OxiZ WASM bindings.

## Table of Contents

- [Overview](#overview)
- [Architecture Diagram](#architecture-diagram)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)
- [Module Structure](#module-structure)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)
- [Future Enhancements](#future-enhancements)

---

## Overview

OxiZ WASM provides WebAssembly bindings for the OxiZ SMT solver, enabling SMT solving directly in web browsers and Node.js environments. The architecture is designed to:

1. **Provide a clean JavaScript/TypeScript API** - Hide Rust internals
2. **Maintain type safety** - Leverage Rust's type system
3. **Optimize for WASM constraints** - Minimize binary size and maximize performance
4. **Support incremental solving** - Enable efficient query workflows
5. **Be production-ready** - Comprehensive error handling and validation

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     JavaScript/TypeScript                    │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐   │
│  │   React    │  │    Vue     │  │   Vanilla JS/TS     │   │
│  │  Wrapper   │  │  Wrapper   │  │   Applications      │   │
│  └────────────┘  └────────────┘  └─────────────────────┘   │
│         │              │                     │               │
│         └──────────────┴─────────────────────┘               │
│                        │                                     │
│                        ▼                                     │
│              ┌──────────────────┐                            │
│              │   WasmSolver     │                            │
│              │  (JavaScript)    │                            │
│              └──────────────────┘                            │
└─────────────────────────┬───────────────────────────────────┘
                          │ wasm-bindgen FFI
┌─────────────────────────┴───────────────────────────────────┐
│                         Rust/WASM                            │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   oxiz-wasm (lib.rs)                  │   │
│  │                                                        │   │
│  │  ┌───────────┐  ┌────────────┐  ┌────────────────┐  │   │
│  │  │ WasmSolver│  │ WasmError  │  │  Validation    │  │   │
│  │  └───────────┘  └────────────┘  └────────────────┘  │   │
│  │                                                        │   │
│  │  ┌───────────────┐  ┌────────────────────────────┐  │   │
│  │  │  pool.rs      │  │   string_utils.rs          │  │   │
│  │  │  (Memory      │  │   (String optimization)    │  │   │
│  │  │   pooling)    │  │                            │  │   │
│  │  └───────────────┘  └────────────────────────────┘  │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│         ┌────────────────────────────────┐                   │
│         │      oxiz-solver (Context)     │                   │
│         │                                 │                   │
│         │  • SAT/UNSAT decision           │                   │
│         │  • Model generation             │                   │
│         │  • Unsat core extraction        │                   │
│         │  • Incremental solving          │                   │
│         └────────────────────────────────┘                   │
│                         │                                    │
│         ┌───────────────┴────────────────┐                   │
│         │                                │                   │
│         ▼                                ▼                   │
│  ┌────────────┐                  ┌───────────────┐          │
│  │ oxiz-core  │                  │ oxiz-theories │          │
│  │            │                  │               │          │
│  │ • Terms    │                  │ • Arithmetic  │          │
│  │ • Sorts    │                  │ • Bitvectors  │          │
│  │ • AST      │                  │ • Arrays      │          │
│  └────────────┘                  └───────────────┘          │
│                                                               │
│         ┌─────────────────────────────────┐                  │
│         │         oxiz-sat                │                  │
│         │  (CDCL SAT Solver)              │                  │
│         └─────────────────────────────────┘                  │
└───────────────────────────────────────────────────────────────┘
```

---

## Core Components

### WasmSolver

The main entry point for users. Wraps `oxiz_solver::Context` and provides:

- **API Methods** - All public-facing SMT operations
- **State Management** - Tracks last result, cancellation flags
- **Validation** - Input validation before passing to core
- **Error Translation** - Converts Rust errors to JavaScript errors

**Key responsibilities:**
- Maintain solver state (`last_result`, `cancelled`)
- Validate all inputs before calling core solver
- Translate between JavaScript and Rust types
- Provide clear error messages

### Error System

Structured error handling with `WasmError` and `WasmErrorKind`:

```rust
pub enum WasmErrorKind {
    ParseError,      // SMT-LIB2 parsing errors
    InvalidSort,     // Invalid sort names
    NoModel,         // Model not available
    NoUnsatCore,     // Unsat core not available
    InvalidState,    // Invalid solver state
    InvalidInput,    // Invalid user input
    NotSupported,    // Feature not yet implemented
    Unknown,         // Catch-all
}

pub struct WasmError {
    kind: WasmErrorKind,
    message: String,
}
```

Errors are converted to JavaScript objects with `kind` and `message` fields.

### Memory Pool (`pool.rs`)

Optimizes memory allocation in WASM:

```rust
pub struct StringPool {
    pool: RefCell<VecDeque<String>>,
    max_size: usize,
    initial_capacity: usize,
}
```

**Features:**
- Thread-local global pools
- Automatic capacity management
- Reuses strings across operations
- Reduces GC pressure

### String Utilities (`string_utils.rs`)

Efficient string handling for WASM:

- `JsStringBuilder` - StringBuilder pattern for constructing output
- `join_lines()` - Optimized string joining with pre-calculated capacity
- `is_effectively_empty()` - Whitespace-aware empty check
- `vec_to_js_array()` / `js_array_to_vec()` - Array conversion helpers

---

## Data Flow

### Typical Solving Flow

1. **User creates solver**
   ```javascript
   const solver = new WasmSolver();
   ```
   - Rust: `WasmSolver::new()` called
   - Creates `Context::new()` internally
   - Initializes state fields

2. **User sets logic**
   ```javascript
   solver.setLogic("QF_LIA");
   ```
   - JavaScript → WASM boundary crossed
   - String passed to Rust
   - `Context::set_logic()` called
   - Logic stored in context

3. **User declares constants**
   ```javascript
   solver.declareConst("x", "Int");
   ```
   - Input validation (non-empty names/sorts)
   - Sort parsing (`parse_sort()`)
   - `Context::declare_const()` called
   - Symbol added to symbol table

4. **User asserts formula**
   ```javascript
   solver.assertFormula("(> x 0)");
   ```
   - Input validation (non-empty)
   - SMT-LIB2 script constructed: `(assert (> x 0))`
   - `Context::execute_script()` parses and adds assertion
   - Formula added to assertion stack

5. **User checks satisfiability**
   ```javascript
   const result = solver.checkSat();
   ```
   - `Context::check_sat()` invoked
   - SAT solver runs
   - Result (`Sat`/`Unsat`/`Unknown`) converted to string
   - Result stored in `last_result`
   - String returned to JavaScript

6. **User gets model** (if SAT)
   ```javascript
   const model = solver.getModel();
   ```
   - Validates `last_result == "sat"`
   - `Context::get_model()` called
   - Model converted to JavaScript object
   - Object returned across WASM boundary

### Error Handling Flow

```
User Input
    │
    ▼
Input Validation ──► Error? ──► WasmError ──► JavaScript Error
    │                                              │
    │ (valid)                                      │
    ▼                                              │
Core Solver                                        │
    │                                              │
    ▼                                              │
Result? ──────────────► Error? ──► WasmError ─────┘
    │
    │ (success)
    ▼
Return Result
```

---

## Design Decisions

### Why `#![forbid(unsafe_code)]`?

- **Safety first** - No undefined behavior in WASM
- **Trust** - Users can trust the library is memory-safe
- **Reliability** - Reduces entire classes of bugs

**Trade-off:** Slightly less flexibility, but worth it for safety guarantees.

### Why Thread-Local Pools?

Memory pools use `thread_local!` instead of global statics:

**Advantages:**
- No synchronization overhead (WASM is single-threaded)
- Automatically cleaned up when thread ends
- Safe without `unsafe` code

**Trade-off:** Each worker thread has its own pool (acceptable in WASM context).

### Why Separate String Utilities?

Separated into `string_utils.rs` for:

1. **Testability** - Easier to unit test string operations
2. **Reusability** - Can be used across the codebase
3. **Clarity** - Keeps `lib.rs` focused on API
4. **Optimization** - Pre-calculated capacities for joining

### Why Store `last_result` in WasmSolver?

- **Validation** - Ensures `getModel()` only called after SAT result
- **User Experience** - Clearer error messages
- **State Tracking** - Knows when models/cores are available

**Alternative considered:** Query core solver each time
**Rejected because:** Extra overhead, less clear API

### Why Both Sync and Async APIs?

- **Sync** - Simple, direct for small problems
- **Async** - Non-blocking for long-running operations in browsers

**Implementation note:** Current async methods wrap sync (TODO: Add true async with periodic yields)

---

## Module Structure

```
oxiz-wasm/
├── src/
│   ├── lib.rs          # Main API (WasmSolver)
│   ├── pool.rs         # Memory pooling
│   └── string_utils.rs # String optimization
├── tests/
│   ├── browser.rs      # Browser tests (wasm-pack test)
│   ├── nodejs.rs       # Node.js tests
│   └── regression.rs   # Regression test suite
├── fuzz/
│   └── fuzz_targets/   # Fuzzing targets
├── benches/
│   └── performance.rs  # Benchmarks (criterion)
└── docs/
    ├── API_REFERENCE.md
    ├── PERFORMANCE_TUNING.md
    └── ARCHITECTURE.md (this file)
```

### Dependency Graph

```
oxiz-wasm
    │
    ├─► wasm-bindgen (FFI)
    ├─► js-sys (JavaScript types)
    ├─► console_error_panic_hook (Better errors)
    │
    └─► oxiz-solver
            │
            ├─► oxiz-core (Terms, Sorts, AST)
            ├─► oxiz-sat (SAT solver)
            ├─► oxiz-theories (Theory solvers)
            └─► oxiz-proof (Proof generation)
```

---

## Error Handling

### Error Conversion Chain

```
Core Solver Error
    │
    ▼
WasmError::new(kind, message)
    │
    ▼
impl From<WasmError> for JsValue
    │
    ▼
JavaScript Error Object
    {
      kind: "ParseError",
      message: "Failed to parse: ..."
    }
```

### Validation Layers

1. **JavaScript Layer** - Type checking (TypeScript)
2. **WASM Boundary** - Input validation (`is_effectively_empty`, etc.)
3. **Core Solver** - Semantic validation (parsing, type checking)

**Rationale:** Fail fast with clear errors at each layer.

---

## Performance Considerations

### WASM Binary Size

Optimizations applied:

1. **Compiler flags** - `opt-level = "z"` for size
2. **LTO** - Link-time optimization
3. **Code-gen units** - Single unit for better optimization
4. **wasm-opt** - Post-processing with `-Oz`
5. **Compression** - Serve with gzip/brotli

**Result:** ~200-500KB compressed (depending on features)

### Memory Management

1. **String Pooling** - Reduces allocations
2. **Pre-allocated Capacities** - Avoids reallocation during growth
3. **`RefCell` instead of `Mutex`** - No thread sync overhead
4. **Minimal Copies** - Use references where possible

### String Conversion Optimization

- **Pre-calculated capacities** - `join_lines` calculates total size upfront
- **Single allocation** - Build entire string in one go
- **Avoid intermediate strings** - Direct construction when possible

### Future Optimizations

1. **Streaming API** - For large results
2. **Incremental model extraction** - Get values on-demand
3. **SIMD** - For bitvector operations (when WASM SIMD stable)
4. **Shared memory** - For web worker communication (experimental)

---

## Future Enhancements

### Short Term

- [ ] True async solving (with periodic yields)
- [ ] Non-nullary function support
- [ ] Incremental compilation caching
- [ ] Streaming results API

### Medium Term

- [ ] Proof production support
- [ ] MaxSMT optimization
- [ ] Model minimization
- [ ] Quantifier support

### Long Term

- [ ] Interpolation
- [ ] Quantifier elimination
- [ ] Custom theory integration
- [ ] Distributed solving (multi-worker)

---

## Testing Strategy

### Test Pyramid

```
       ┌─────────────┐
       │   Browser   │   Integration tests
       │   Tests     │   (wasm-pack test)
       ├─────────────┤
       │   Node.js   │   Integration tests
       │   Tests     │   (wasm-pack test)
       ├─────────────┤
       │ Regression  │   Specific bugs
       │   Tests     │   and edge cases
       ├─────────────┤
       │    Fuzz     │   Random inputs
       │   Tests     │   (cargo fuzz)
       ├─────────────┤
       │    Unit     │   Individual
       │   Tests     │   components
       └─────────────┘
```

### Test Coverage Goals

- **Unit tests** - 80%+ coverage
- **Integration tests** - All API methods
- **Regression tests** - All fixed bugs
- **Fuzz tests** - Continuous (CI)

---

## Build Process

### Development Build

```bash
wasm-pack build --target web --dev
```

### Production Build

```bash
# 1. Build with release optimizations
wasm-pack build --target web --release

# 2. Further optimize with wasm-opt
wasm-opt -Oz -o pkg/oxiz_wasm_bg.wasm pkg/oxiz_wasm_bg.wasm

# 3. Compress for distribution
gzip -k pkg/oxiz_wasm_bg.wasm
brotli pkg/oxiz_wasm_bg.wasm
```

---

## Deployment

### NPM Package Structure

```
oxiz-wasm/
├── package.json
├── oxiz_wasm.js         # JavaScript glue code
├── oxiz_wasm.d.ts       # TypeScript declarations
├── oxiz_wasm_bg.wasm    # WASM binary
└── README.md
```

### CDN Distribution

Serve from CDN for easy inclusion:

```html
<script type="module">
  import init, { WasmSolver } from 'https://cdn.jsdelivr.net/npm/oxiz-wasm/oxiz_wasm.js';

  await init();
  const solver = new WasmSolver();
</script>
```

---

## Contributing

When extending the architecture:

1. **Maintain safety** - No `unsafe` code
2. **Add tests** - Unit + integration for new features
3. **Update docs** - API reference, architecture docs
4. **Benchmark** - Ensure no performance regression
5. **Validate inputs** - Always check user input
6. **Clear errors** - Provide helpful error messages

---

## References

- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)
- [WASM Performance Guide](https://web.dev/webassembly-performance/)
- [SMT-LIB2 Standard](http://smtlib.cs.uiowa.edu/)
- [OxiZ Core Documentation](../../oxiz-core/README.md)

---

## Glossary

- **WASM** - WebAssembly, binary instruction format for web
- **FFI** - Foreign Function Interface, boundary between languages
- **SMT** - Satisfiability Modulo Theories
- **SAT** - Boolean Satisfiability
- **AST** - Abstract Syntax Tree
- **CDCL** - Conflict-Driven Clause Learning (SAT algorithm)
- **LTO** - Link-Time Optimization

---

For questions about the architecture, please open an issue on GitHub.
