# SCIRS2 Policy for oxiz-spacer

## Overview

This crate follows the SCIRS2 (Scientific Rust 2) policy for scientific computing in Rust.

## Dependencies

Instead of using standard scientific computing crates like `rand`, `rand_distr`, or `ndarray`, we use **SciRS2-Core** when numerical or random computations are needed.

### Current Status

✅ **Compliant** - This crate currently does not require numerical/random computations beyond what's provided by standard Rust and the core oxiz infrastructure.

## Rationale

The oxiz-spacer crate focuses on:
- **Symbolic reasoning**: PDR/IC3 algorithm for CHC solving
- **SMT integration**: Symbolic satisfiability checking
- **Logical formulas**: Term manipulation and frame management

These operations are inherently symbolic and do not require numerical scientific computing libraries.

## Future Considerations

If numerical computations are needed in the future (e.g., for heuristics, statistics, or optimization metrics), we will:
1. Evaluate if SciRS2-Core provides the required functionality
2. Use SciRS2-Core instead of `rand`/`ndarray` where applicable
3. Document any exceptions with clear justification

## Current Dependencies

The crate relies on:
- `oxiz-core`: Core term management and SMT-LIB parsing
- `oxiz-solver`: SMT solver integration
- `smallvec`: Stack-allocated vectors for performance
- `rustc_hash`: Fast hashing for internal data structures
- `thiserror`: Error handling
- `tracing`: Logging and diagnostics

All dependencies are workspace-managed and compliant with project policies.

## Verification

Run `cargo tree -p oxiz-spacer` to verify no prohibited dependencies are included.
