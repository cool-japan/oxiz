# oxiz-wasm TODO

Last Updated: 2026-01-05

## Progress: ~99% Complete

---

## BEYOND Z3: WASM-First Architecture

**OxiZ is designed for the browser from day one!**

| Metric | Z3 WASM | OxiZ WASM |
|--------|---------|-----------|
| Bundle Size | ~20MB | **Target <2MB** |
| Load Time | Slow | **Fast** |
| Memory | Heavy | Optimized |
| Async Support | Limited | **Full** |

**Benefits**:
- Client-side verification (no server roundtrip)
- Edge computing (CDN-deployed verification)
- Offline-capable web applications
- Embedded in web IDEs and playgrounds

**Framework Wrappers**: React, Vue, Svelte, Deno ready!

---

## Dependencies
- **oxiz-core**: SMT-LIB2 parser
- **oxiz-solver**: Main solver API

## Provides (enables other crates)
- JavaScript/TypeScript SMT solver API
- Browser-native verification
- WebWorker support for non-blocking solving

---

## API Improvements

- [x] Add `declareConst` method
- [x] Add `declareFun` method (supports non-nullary functions)
- [x] Add `assertFormula` method (from string)
- [x] Add `getValue` method for model extraction
- [x] Add `getUnsatCore` method
- [x] Add async/streaming API for long-running solves
- [x] Add cancellation support
- [x] Implement true async with periodic event loop yielding (checkSatAsync, executeAsync)
- [x] Add executeWithProgress() method with progress callbacks
- [x] Add async utility helpers for responsive browser operations
- [x] Enhance `declareFun` to support non-nullary functions
- [x] Add `defineSort` method for custom sorts
- [x] Add `defineFun` method for function definitions
- [x] Add `checkSatAssuming` method
- [x] Add `validateFormula` method for pre-validation
- [x] Add `assertFormulaSafe` method with error recovery
- [x] Add formula builder methods (mkEq, mkAnd, mkOr, mkNot, mkImplies, etc.)
- [x] Add comparison builders (mkLt, mkLe, mkGt, mkGe)
- [x] Add arithmetic builders (mkAdd, mkSub, mkMul)
- [x] Add mkIte for if-then-else expressions
- [x] Add mkDistinct for all-different constraints
- [x] Add mkDiv for division operations
- [x] Add mkMod for modulo operations
- [x] Add mkNeg for arithmetic negation
- [x] Add mkXor for exclusive-or operations
- [x] Add declareFuns for batch constant declarations
- [x] Add assertFormulas for batch assertions
- [x] Add assertNamed for labeled assertions (useful for unsat cores)
- [x] Add bitvector operations:
  - [x] mkBvAnd - bitvector AND
  - [x] mkBvOr - bitvector OR
  - [x] mkBvXor - bitvector XOR
  - [x] mkBvNot - bitvector NOT
  - [x] mkBvNeg - bitvector negation
  - [x] mkBvAdd - bitvector addition
  - [x] mkBvSub - bitvector subtraction
  - [x] mkBvMul - bitvector multiplication

## Performance

- [x] Reduce WASM binary size (build.sh with wasm-opt support)
- [x] Implement memory pooling for term allocation (src/pool.rs)
- [x] Add WebWorker support for non-blocking solving
- [x] Optimize string passing between JS and WASM (src/string_utils.rs)
- [x] Add incremental compilation option (.cargo/config.toml)
- [x] Profile and optimize hot paths (benchmark.html)
- [x] Add performance benchmarking suite

## Error Handling

- [x] Improve error messages for JS consumers
- [x] Add typed error classes
- [x] Add validation for input scripts
- [x] Add error recovery mechanisms (validateFormula, assertFormulaSafe)
- [x] Add better parse error reporting with helpful hints
- [x] Add warnings for suboptimal usage patterns (getDiagnostics, checkPattern)

## TypeScript Support

- [x] Generate TypeScript declarations (.d.ts files)
- [x] Add JSDoc comments
- [x] Create TypeScript example project (examples/typescript/)
- [x] Add type definitions for error objects
- [x] Create TypeScript-friendly API wrappers

## Packaging

- [ ] Publish to npm (ready - use ./publish.sh when ready)
- [x] Add CDN distribution (unpkg/jsdelivr) - CDN_USAGE.md created
- [x] Create package.json configuration
- [x] Create .npmignore for clean npm packages
- [x] Create React wrapper component (wrappers/react/)
- [x] Create Vue wrapper component (wrappers/vue/)
- [x] Create Svelte wrapper component (wrappers/svelte/)
- [x] Add Deno support (deno/)
- [x] Add CommonJS build target (build.sh supports nodejs target)
- [x] Add ESM build target (build.sh supports web target)
- [x] Optimize package size (build.sh with wasm-opt)
- [x] Add build automation script (build.sh)
- [x] Add NPM publishing automation (publish.sh, version-bump.sh)

## Testing

- [x] Add browser-based tests (using wasm-pack test) - tests/browser.rs
- [x] Add Node.js integration tests - tests/nodejs.rs
- [x] Add performance benchmarks - benches/performance.rs (criterion-based)
- [x] Test in all target environments (Chrome, Firefox, Safari, Edge) - BROWSER_TESTING.md created with comprehensive testing guide
- [x] Add CI/CD pipeline for automated testing (.github/workflows/ci.yml)
- [x] Add release automation (.github/workflows/release.yml)
- [x] Add fuzzing tests (fuzz/)
- [x] Add regression tests (tests/regression.rs)

## Documentation

- [x] Create basic examples (HTML/JS)
- [x] Add WebWorker example
- [x] Add API reference documentation
- [x] Create tutorials (beginner to advanced) - docs/TUTORIAL_*.md
- [x] Add interactive playground (online demo) - examples/playground.html created with full-featured UI
- [x] Add migration guide from Z3 JavaScript bindings - docs/MIGRATION_FROM_Z3.md
- [x] Add SMT-LIB2 quick reference - docs/SMTLIB2_REFERENCE.md
- [x] Add performance tuning guide
- [x] Add architecture documentation
- [x] Add CDN usage guide - CDN_USAGE.md
- [x] Add browser testing guide - BROWSER_TESTING.md

## Future Features

- [x] Add proof production support (getProof method)
- [x] Add statistics API (getStatistics, getInfo methods)
- [x] Add solver configuration presets (applyPreset method)
- [x] Add debugging/tracing support (debugDump, setTracing methods)
- [x] Add optimization (MaxSMT) support
  - [x] minimize() method for minimization objectives
  - [x] maximize() method for maximization objectives
  - [x] optimize() method to run optimization
  - [x] assertSoft() method for soft constraints (MaxSMT)
- [x] Add model minimization
  - [x] getMinimalModel() method to get model with only specified variables
- [x] Add interpolation support (computeInterpolant method - basic implementation)
- [x] Add quantifier elimination API (eliminateQuantifiers method - returns NotSupported, full implementation planned)

## Completed

- [x] Basic WasmSolver wrapper
- [x] execute() method
- [x] setLogic() method
- [x] checkSat() method
- [x] checkSatAsync() method (async version)
- [x] executeAsync() method (async version)
- [x] push/pop/reset methods
- [x] resetAssertions() method
- [x] version() function
- [x] Panic hook setup
- [x] declareConst() method with full validation
- [x] declareFun() method (supports non-nullary functions)
- [x] assertFormula() method with validation
- [x] getValue() method
- [x] getModel() method (returns JS object)
- [x] getModelString() method (returns SMT-LIB2 string)
- [x] getUnsatCore() method
- [x] getProof() method for proof production
- [x] getAssertions() method
- [x] setOption() / getOption() methods
- [x] simplify() method
- [x] mkApp() method for creating function applications
- [x] Formula builder methods:
  - [x] mkEq() - equality
  - [x] mkAnd() - conjunction
  - [x] mkOr() - disjunction
  - [x] mkNot() - negation
  - [x] mkImplies() - implication
  - [x] mkIte() - if-then-else
  - [x] mkLt(), mkLe(), mkGt(), mkGe() - comparisons
  - [x] mkAdd(), mkSub(), mkMul() - arithmetic
  - [x] mkDiv() - division
  - [x] mkMod() - modulo
  - [x] mkNeg() - arithmetic negation
  - [x] mkXor() - exclusive-or
  - [x] mkDistinct() - all-different
- [x] cancel() method
- [x] isCancelled() method
- [x] Comprehensive error handling with typed errors
- [x] Input validation for all methods
- [x] JSDoc comments for all public APIs
- [x] WebWorker example implementation
- [x] HTML/JavaScript examples
- [x] Example README with usage patterns
- [x] TypeScript declarations file (oxiz-wasm.d.ts)
- [x] TypeScript example project with full type safety
- [x] Performance benchmark suite (benchmark.html)
- [x] Build automation script (build.sh)
- [x] Support for multiple build targets (web, nodejs, bundler)
- [x] Optimized builds with wasm-opt integration
- [x] checkSatAssuming() method for incremental solving with assumptions
- [x] defineSort() method for creating sort aliases
- [x] defineFun() method for defining functions with bodies
- [x] validateFormula() method for pre-validation without assertion
- [x] assertFormulaSafe() method with enhanced error messages and hints
- [x] getDiagnostics() method for detecting suboptimal usage patterns
- [x] checkPattern() method for usage pattern recommendations
- [x] getStatistics() method for performance monitoring
- [x] getInfo() method for solver metadata (name, version, capabilities, etc.)
- [x] applyPreset() method for quick configuration (default, fast, complete, debug, etc.)
- [x] debugDump() method for comprehensive state inspection
- [x] setTracing() method for enabling/disabling debug tracing
- [x] Comprehensive unit tests for all new features
- [x] Browser-based tests (tests/browser.rs) with 60+ test cases
- [x] Node.js integration tests (tests/nodejs.rs) with comprehensive coverage
- [x] React wrapper with hooks and context (@oxiz/react)
- [x] Vue 3 wrapper with composables and stores (@oxiz/vue)
- [x] Svelte wrapper with stores and composables (@oxiz/svelte)
- [x] Deno support with examples and documentation
- [x] Migration guide from Z3 JavaScript bindings (docs/MIGRATION_FROM_Z3.md)
- [x] SMT-LIB2 quick reference guide (docs/SMTLIB2_REFERENCE.md)
- [x] Beginner tutorial (docs/TUTORIAL_BEGINNER.md)
- [x] Intermediate tutorial (docs/TUTORIAL_INTERMEDIATE.md)
- [x] Performance benchmarks with criterion (benches/performance.rs)
- [x] Optimization support (minimize, maximize, optimize methods)
- [x] MaxSMT support (assertSoft method with weights)
- [x] Comprehensive tests for optimization features
- [x] Model minimization (getMinimalModel method)
- [x] Tests for model minimization (4 comprehensive tests)
- [x] Interpolation support (computeInterpolant method for Craig interpolation)
- [x] Quantifier elimination API (eliminateQuantifiers method with NotSupported response)
- [x] Comprehensive tests for interpolation (7 tests covering edge cases and validation)
- [x] Comprehensive tests for quantifier elimination (7 tests)
- [x] Interactive HTML example for Craig interpolation (examples/interpolation.html)
- [x] Updated examples README with interpolation documentation
- [x] Batch operations for efficiency:
  - [x] declareFuns() - declare multiple constants at once
  - [x] assertFormulas() - assert multiple formulas at once
  - [x] assertNamed() - assert formulas with labels for unsat core tracking
- [x] Comprehensive tests for new operators and batch operations (27 new tests)
- [x] Bitvector operations (8 methods):
  - [x] mkBvAnd(), mkBvOr(), mkBvXor() - bitwise logical operations
  - [x] mkBvNot(), mkBvNeg() - bitwise negation and arithmetic negation
  - [x] mkBvAdd(), mkBvSub(), mkBvMul() - bitvector arithmetic
- [x] Comprehensive tests for bitvector operations (12 new tests)
- [x] Interactive batch operations example (examples/batch-operations.html)
- [x] Updated TypeScript declarations with all new methods
- [x] NPM publishing automation scripts (publish.sh, version-bump.sh)
- [x] Interactive playground demo (examples/playground.html)
- [x] CDN distribution documentation (CDN_USAGE.md)
- [x] Browser testing guide (BROWSER_TESTING.md)
- [x] Async utilities module (src/async_utils.rs) with event loop yielding
- [x] Enhanced checkSatAsync() with true async and periodic yields
- [x] Enhanced executeAsync() with chunked execution and cancellation support
- [x] New executeWithProgress() method for progress callbacks
- [x] Progress callback demo (examples/progress-demo.html)
- [x] Updated TypeScript declarations for async methods
