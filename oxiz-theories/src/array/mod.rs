//! Array theory solver
//!
//! Implements the theory of arrays with read (select) and write (store) operations.
//! Uses the axioms:
//! - Read-over-write same: select(store(a, i, v), i) = v
//! - Read-over-write different: i ≠ j → select(store(a, i, v), j) = select(a, j)
//! - Extensionality: (∀i. select(a, i) = select(b, i)) → a = b

mod solver;

pub use solver::ArraySolver;
