//! Property-based testing entry point for oxiz-solver
//!
//! Note: These tests are disabled by default due to API incompatibilities.
//! Run with: cargo test --test property_based --features property-tests

#![cfg(feature = "property-tests")]

mod property_tests;
