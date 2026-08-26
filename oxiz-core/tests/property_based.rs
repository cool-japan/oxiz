//! Property-based testing entry point for oxiz-core
//!
//! Runs by default (property-tests is on in the default feature set); a
//! consumer that opted out with `default-features = false` gets it back
//! via: cargo test --test property_based --features property-tests

#![cfg(feature = "property-tests")]

mod property_tests;
