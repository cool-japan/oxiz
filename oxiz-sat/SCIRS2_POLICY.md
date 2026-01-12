# SCIRS2 Policy for oxiz-sat

## Purpose
This document describes the SciRS2 compliance policy for the oxiz-sat crate, which is a high-performance CDCL SAT solver for the OxiZ SMT solver framework.

## Dependency Policy

### Prohibited Dependencies
The following dependencies are **NOT ALLOWED** per SciRS2 policy:
- `rand` / `rand_distr` - Use SciRS2-Core random number generation instead
- `ndarray` - Use SciRS2-Core array types instead

### Current Dependencies
This crate uses the following dependencies, all of which are compliant:

1. **oxiz-core** (workspace dependency)
   - Core types and utilities for OxiZ
   - Provides error types, AST structures, and common utilities

2. **tracing** (workspace dependency)
   - Logging and instrumentation framework
   - Allowed for diagnostic purposes

3. **smallvec** (workspace dependency)
   - Stack-allocated small vectors
   - Performance optimization, no SciRS2 alternative needed

4. **rustc-hash** (workspace dependency)
   - Fast hash map implementation
   - Performance optimization, no SciRS2 alternative needed

### Random Number Generation
Instead of using the `rand` crate, this implementation includes a custom **xorshift64** PRNG:
- Located in `src/solver.rs` (methods: `rand_u64()`, `rand_f64()`, `rand_bool()`)
- Used for random polarity selection in branching heuristic
- Simple, deterministic, and reproducible
- No external dependencies required

## Compliance Status

✅ **COMPLIANT** - This crate does not use any prohibited dependencies.

### Implementation Notes
1. All random number generation is done via the internal xorshift64 implementation
2. No array/matrix operations requiring ndarray
3. All numerical operations are basic scalar operations
4. Hash maps use `rustc-hash::FxHashMap` for performance

## Future Considerations
If numerical/statistical operations are needed in the future:
- Use **SciRS2-Core** for random distributions
- Use **SciRS2-Core** for array operations
- Consult with the SciRS2 team for other numerical needs

## Verification
To verify compliance, run:
```bash
cargo tree --edges normal | grep -E "(rand|ndarray)"
```
This should return no results.

**Note**: Dev-dependencies (e.g., `proptest` which uses `rand`) are allowed for testing purposes only.
They do not affect the production build.

Last Updated: 2025-12-28
