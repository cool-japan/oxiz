//! Enhanced JavaScript/TypeScript API
//!
//! This module provides an enhanced JavaScript/TypeScript API for OxiZ WASM,
//! with support for modern web features like Promises, Workers, and streaming.
//!
//! # Features
//!
//! - **Promise Wrapper**: Async/await support for solver operations
//! - **Memory Management**: Efficient memory sharing between JS and WASM
//! - **Worker Support**: Web Worker integration for background solving
//! - **Streaming**: Stream large results incrementally
//!
//! # Example
//!
//! ```javascript
//! import init, { AsyncSolver } from 'oxiz-wasm';
//!
//! await init();
//! const solver = new AsyncSolver();
//!
//! // Use async/await
//! const result = await solver.checkSatAsync();
//! console.log(result); // "sat" or "unsat"
//!
//! // Stream model values
//! for await (const [name, value] of solver.streamModel()) {
//!     console.log(`${name} = ${value}`);
//! }
//! ```

pub mod memory_management;
pub mod promise_wrapper;
pub mod streaming;
pub mod worker_support;

pub use memory_management::*;
pub use promise_wrapper::*;
pub use streaming::*;
pub use worker_support::*;
