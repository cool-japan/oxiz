//! Custom Memory Allocators for SMT Solver.
//!
//! Provides specialized allocators for AST nodes, clauses, and solver state
//! to improve cache locality and reduce allocation overhead.

pub mod arena;
pub mod pool;
pub mod region;

pub use arena::{Arena, ArenaConfig, ArenaError, ArenaHandle};
pub use pool::{ObjectPool, PoolConfig, PoolGuard, PoolStats, SharedObjectPool, SharedPoolGuard};
pub use region::{Region, RegionAllocator, RegionRef, RegionSlice};
