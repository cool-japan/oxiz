//! Lock-free data structures for parallel solving.
//!
//! Provides lock-free work queues and clause sharing structures
//! for use in parallel SAT/SMT solving. Feature-gated behind `parallel`.

#[cfg(feature = "std")]
mod clause_sharing;
#[cfg(feature = "std")]
mod queue;

#[cfg(feature = "std")]
pub use clause_sharing::LockFreeClauseSharing;
#[cfg(feature = "std")]
pub use queue::LockFreeQueue;

#[cfg(all(test, feature = "std"))]
mod tests;
