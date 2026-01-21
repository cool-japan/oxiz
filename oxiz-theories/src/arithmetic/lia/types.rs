//! Linear Integer Arithmetic (LIA) Solver
//!
//! Extends the Simplex-based LRA solver with integer-specific reasoning:
//! - Branch-and-bound for integer feasibility
//! - Gomory cuts, MIR cuts, and CG cuts
//! - GCD-based infeasibility detection
//! - Integer bound propagation

use super::super::simplex::{Simplex, VarId};
use crate::config::LiaConfig;
use num_rational::Rational64;
use rustc_hash::FxHashMap;

/// A bound on an integer variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum IntBound {
    /// Lower bound: x >= value
    Lower(i64),
    /// Upper bound: x <= value
    Upper(i64),
}

/// Branch-and-bound node
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct BranchNode {
    /// Variable to branch on
    pub(super) var: VarId,
    /// Branch direction: true = x >= ceil(value), false = x <= floor(value)
    pub(super) branch_up: bool,
    /// The fractional value that triggered the branch
    pub(super) fractional_value: Rational64,
}

/// Cut metadata for management and aging
#[derive(Debug, Clone)]
pub(super) struct CutInfo {
    /// The cut constraint (stored as slack variable ID in simplex)
    #[allow(dead_code)]
    pub(super) slack_var: VarId,
    /// Age: number of LP solves since cut was added
    pub(super) age: u32,
    /// Activity: how often the cut has been tight/binding
    /// (incremented when slack variable is at bound)
    pub(super) activity: u32,
    /// Generation iteration when cut was added
    #[allow(dead_code)]
    pub(super) generation: usize,
}

/// Linear Integer Arithmetic Solver
#[derive(Debug)]
pub struct LiaSolver {
    /// Underlying Simplex solver (works with rationals)
    pub(super) simplex: Simplex,
    /// Integer variables (all variables are integers in LIA)
    pub(super) int_vars: FxHashMap<VarId, IntBound>,
    /// Branch-and-bound stack
    pub(super) branch_stack: Vec<BranchNode>,
    /// Maximum branch depth (to prevent infinite loops)
    pub(super) max_depth: usize,
    /// Number of cuts generated
    pub(super) cuts_generated: usize,
    /// Conflict-driven cut selection: track which variables appear in recent conflicts
    pub(super) conflict_vars: FxHashMap<VarId, u32>,
    /// Number of conflicts seen
    pub(super) num_conflicts: u32,
    /// Configuration
    pub(super) config: LiaConfig,
    /// Pseudo-cost tracking for branching heuristic (variable -> (down_cost, up_cost, count))
    pub(super) pseudo_costs: FxHashMap<VarId, (f64, f64, u32)>,
    /// Active cuts being managed
    pub(super) active_cuts: Vec<CutInfo>,
    /// Number of LP solves performed (for cut aging)
    #[allow(dead_code)]
    pub(super) lp_solve_count: u32,
}
