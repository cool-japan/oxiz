# SCIRS2 POLICY - oxiz-wasm

## Purpose

This document defines the policy for using SciRS2 libraries in the oxiz-wasm crate.

## Policy Statement

The oxiz-wasm crate is a WebAssembly binding layer for the OxiZ SMT solver. As such, it primarily interfaces with JavaScript and does NOT directly use numerical or scientific computing libraries.

## Dependencies

### Allowed External Dependencies

The following external dependencies are allowed for WASM binding purposes:

- `wasm-bindgen` - Core WASM bindings
- `wasm-bindgen-futures` - Async/await support in WASM
- `js-sys` - JavaScript standard library bindings
- `console_error_panic_hook` - Better panic messages in browser console
- `serde` and `serde-wasm-bindgen` - Serialization support for JS interop

### SciRS2 Usage

**This crate does NOT require SciRS2 dependencies** because:

1. It is a thin binding layer over oxiz-core and oxiz-solver
2. All numerical and scientific computation is handled by the core oxiz libraries
3. Its only responsibility is to provide a JavaScript-friendly API

### Prohibited Dependencies

The following dependencies are NOT allowed in this crate:

- `rand`, `rand_distr` - Random number generation (not needed in WASM bindings)
- `ndarray` - N-dimensional arrays (not needed in WASM bindings)
- Any other numerical/scientific libraries that duplicate SciRS2 functionality

## Numerical Computing Needs

If oxiz-wasm ever requires numerical computing capabilities (which it currently does not), the following approach must be taken:

1. **First choice**: Delegate to oxiz-core or oxiz-solver which already use SciRS2-Core
2. **Second choice**: Add minimal SciRS2-Core dependency only if absolutely necessary
3. **Never**: Add rand, ndarray, or other non-SciRS2 alternatives

## Compliance

This crate is currently **COMPLIANT** with the SCIRS2 POLICY because:

- It has no numerical computing dependencies
- All computation is delegated to core libraries
- It focuses solely on WASM binding concerns

## Future Considerations

If future features require numerical capabilities:

1. Evaluate whether the feature belongs in oxiz-core instead
2. If WASM-specific numerical work is needed, use SciRS2-Core
3. Update this policy document to reflect the new dependencies

## Last Updated

2025-12-28
