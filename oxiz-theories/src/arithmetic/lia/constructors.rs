//! Constructor and variable creation methods for LIA solver

use super::super::simplex::{Simplex, VarId};
use super::types::{IntBound, LiaSolver};
use crate::config::{LiaConfig, SimplexConfig};
use rustc_hash::FxHashMap;

impl LiaSolver {
    /// Create a new LIA solver
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LiaConfig::default())
    }

    /// Create a new LIA solver with custom configuration
    #[must_use]
    pub fn with_config(config: LiaConfig) -> Self {
        Self::with_configs(config, SimplexConfig::default())
    }

    /// Create a new LIA solver with custom LIA and Simplex configurations
    #[must_use]
    pub fn with_configs(lia_config: LiaConfig, simplex_config: SimplexConfig) -> Self {
        Self {
            simplex: Simplex::with_config(simplex_config),
            int_vars: FxHashMap::default(),
            branch_stack: Vec::new(),
            max_depth: lia_config.max_depth,
            cuts_generated: 0,
            conflict_vars: FxHashMap::default(),
            num_conflicts: 0,
            config: lia_config,
            pseudo_costs: FxHashMap::default(),
            active_cuts: Vec::new(),
            lp_solve_count: 0,
        }
    }

    /// Create a new integer variable
    pub fn new_var(&mut self) -> VarId {
        let var = self.simplex.new_var();
        self.int_vars.insert(var, IntBound::Lower(i64::MIN));
        var
    }
}
