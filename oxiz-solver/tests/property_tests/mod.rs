//! Property-based tests for oxiz-solver
//!
//! Comprehensive testing of CDCL(T) solver invariants
//!
//! Note: These tests are disabled by default due to API incompatibilities.
//! Enable with `--features property-tests` when the API is stabilized.

#![cfg(feature = "property-tests")]

mod backtrack_properties;
mod conflict_properties;
mod model_properties;
mod propagation_properties;
