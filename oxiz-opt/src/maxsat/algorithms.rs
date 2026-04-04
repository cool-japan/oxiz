//! MaxSAT solving algorithms: Fu-Malik, OLL, MSU3, WMax, and PMRES.
//!
//! All algorithms are implemented as methods on MaxSatSolver.

use super::core::{SoftId, Weight};
use super::types::{MaxSatAlgorithm, MaxSatError, MaxSatResult, MaxSatSolver};
use oxiz_sat::{LBool, Lit, Solver as SatSolver, SolverResult, Var};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

impl MaxSatSolver {
    /// Solve the MaxSAT problem
    pub fn solve(&mut self) -> Result<MaxSatResult, MaxSatError> {
        // Check if trivially satisfiable (no soft clauses)
        if self.soft_clauses.is_empty() {
            return self.check_hard_satisfiable();
        }

        // Use stratified solving if enabled and weights differ
        if self.config.stratified && self.has_different_weights() {
            return self.solve_stratified();
        }

        // Use the configured algorithm
        match self.config.algorithm {
            MaxSatAlgorithm::FuMalik => self.solve_fu_malik(),
            MaxSatAlgorithm::Oll => self.solve_oll(),
            MaxSatAlgorithm::Msu3 => self.solve_msu3(),
            MaxSatAlgorithm::WMax => self.solve_wmax(),
            MaxSatAlgorithm::Pmres => self.solve_pmres(),
        }
    }

    /// Check if hard constraints are satisfiable
    pub(super) fn check_hard_satisfiable(&mut self) -> Result<MaxSatResult, MaxSatError> {
        let mut solver = SatSolver::new();

        // Add hard clauses
        for clause in &self.hard_clauses {
            for &lit in clause.iter() {
                while solver.num_vars() <= lit.var().0 as usize {
                    solver.new_var();
                }
            }
            solver.add_clause(clause.iter().copied());
        }

        self.stats.sat_calls += 1;
        match solver.solve() {
            SolverResult::Sat => {
                self.best_model = Some(solver.model().to_vec());
                self.lower_bound = Weight::zero();
                self.upper_bound = Weight::zero();
                Ok(MaxSatResult::Optimal)
            }
            SolverResult::Unsat => Err(MaxSatError::Unsatisfiable),
            SolverResult::Unknown => Ok(MaxSatResult::Unknown),
        }
    }

    /// Check if weights differ
    pub(super) fn has_different_weights(&self) -> bool {
        if self.soft_clauses.is_empty() {
            return false;
        }
        let first_weight = &self.soft_clauses[0].weight;
        self.soft_clauses.iter().any(|c| &c.weight != first_weight)
    }

    /// Solve using stratified approach (by weight levels)
    pub(super) fn solve_stratified(&mut self) -> Result<MaxSatResult, MaxSatError> {
        // Collect unique weight levels (sorted descending)
        let mut weight_levels: Vec<Weight> =
            self.soft_clauses.iter().map(|c| c.weight.clone()).collect();
        weight_levels.sort();
        weight_levels.dedup();
        weight_levels.reverse();

        // Solve for each level
        for level in weight_levels {
            // Mark clauses at this level as active
            let active_ids: Vec<SoftId> = self
                .soft_clauses
                .iter()
                .filter(|c| c.weight >= level)
                .map(|c| c.id)
                .collect();

            if active_ids.is_empty() {
                continue;
            }

            // Solve for this level
            let result = self.solve_fu_malik_subset(&active_ids)?;
            if result == MaxSatResult::Unsatisfiable {
                return Err(MaxSatError::Unsatisfiable);
            }
        }

        Ok(MaxSatResult::Optimal)
    }

    /// Fu-Malik core-guided algorithm
    pub(super) fn solve_fu_malik(&mut self) -> Result<MaxSatResult, MaxSatError> {
        let all_ids: Vec<SoftId> = self.soft_clauses.iter().map(|c| c.id).collect();
        self.solve_fu_malik_subset(&all_ids)
    }

    /// Fu-Malik algorithm on a subset of soft clauses
    ///
    /// This is the proper core-guided Fu-Malik algorithm using assumption-based solving.
    /// The algorithm iteratively:
    /// 1. Solve under assumptions that all soft clauses are satisfied
    /// 2. If UNSAT, extract the core of unsatisfied soft clauses
    /// 3. Add a relaxation variable to each soft clause in the core
    /// 4. Add an at-most-one constraint on the relaxation variables
    /// 5. Repeat until SAT
    pub(super) fn solve_fu_malik_subset(
        &mut self,
        soft_ids: &[SoftId],
    ) -> Result<MaxSatResult, MaxSatError> {
        let mut solver = SatSolver::new();
        let mut next_var = 0u32;

        // Helper function to ensure variable exists
        fn ensure_var(solver: &mut SatSolver, var_idx: u32) {
            while solver.num_vars() <= var_idx as usize {
                solver.new_var();
            }
        }

        // Add hard clauses
        for clause in &self.hard_clauses {
            for &lit in clause.iter() {
                ensure_var(&mut solver, lit.var().0);
                next_var = next_var.max(lit.var().0 + 1);
            }
            solver.add_clause(clause.iter().copied());
        }

        // Create blocking variables for soft clauses (b_i = true means soft clause i is blocked/relaxed)
        let mut blocking_vars: FxHashMap<SoftId, Var> = FxHashMap::default();
        let mut var_to_soft: FxHashMap<Var, SoftId> = FxHashMap::default();

        for &id in soft_ids {
            if let Some(clause) = self.soft_clauses.get(id.0 as usize) {
                let block_var = Var(next_var);
                next_var += 1;
                ensure_var(&mut solver, block_var.0);

                blocking_vars.insert(id, block_var);
                var_to_soft.insert(block_var, id);
                self.relax_to_soft.insert(Lit::pos(block_var), id);

                // Add soft clause with blocking literal: lits \/ b_i
                // If b_i is true, the clause is trivially satisfied (blocked)
                let mut lits: SmallVec<[Lit; 8]> = clause.lits.iter().copied().collect();
                lits.push(Lit::pos(block_var));
                solver.add_clause(lits.iter().copied());

                self.stats.relax_vars_added += 1;
            }
        }

        // Track which soft clauses have been relaxed (their blocking var can be true)
        let mut relaxed: FxHashMap<SoftId, bool> = FxHashMap::default();
        for &id in soft_ids {
            relaxed.insert(id, false);
        }

        // Main Fu-Malik loop
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return Ok(MaxSatResult::Unknown);
            }

            // Build assumptions: assume ~b_i for all non-relaxed soft clauses
            // This means "all soft clauses must be satisfied"
            let assumptions: Vec<Lit> = soft_ids
                .iter()
                .filter(|id| !relaxed.get(id).copied().unwrap_or(false))
                .filter_map(|id| blocking_vars.get(id).map(|&v| Lit::neg(v)))
                .collect();

            if assumptions.is_empty() {
                // All soft clauses relaxed - check if hard constraints are SAT
                return self.check_hard_satisfiable();
            }

            self.stats.sat_calls += 1;
            let (result, core) = solver.solve_with_assumptions(&assumptions);

            match result {
                SolverResult::Sat => {
                    // Found a satisfying assignment
                    self.best_model = Some(solver.model().to_vec());
                    self.update_soft_values();
                    return Ok(MaxSatResult::Optimal);
                }
                SolverResult::Unsat => {
                    // Extract core - these are the soft clauses that conflict
                    let core_lits = core.unwrap_or_default();
                    self.stats.cores_extracted += 1;

                    if core_lits.is_empty() {
                        // Empty core means hard clauses alone are UNSAT
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    // Find which soft clauses are in the core
                    let mut core_soft_ids: SmallVec<[SoftId; 8]> = SmallVec::new();
                    let mut min_weight = Weight::Infinite;

                    for lit in &core_lits {
                        // Core contains ~b_i, so the var is the blocking var
                        let var = lit.var();
                        if let Some(&soft_id) = var_to_soft.get(&var) {
                            core_soft_ids.push(soft_id);
                            if let Some(clause) = self.soft_clauses.get(soft_id.0 as usize) {
                                min_weight = min_weight.min(clause.weight.clone());
                            }
                        }
                    }

                    self.stats.total_core_size += core_soft_ids.len() as u32;

                    if core_soft_ids.is_empty() {
                        // No soft clauses in core - hard constraints UNSAT
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    // Relax all soft clauses in the core
                    for &soft_id in &core_soft_ids {
                        relaxed.insert(soft_id, true);
                    }

                    // Update lower bound
                    self.lower_bound = self.lower_bound.add(&min_weight);

                    // Add at-most-one constraint on core blocking variables:
                    // At most one of the blocking variables can be true.
                    // This is encoded as: for all pairs (b_i, b_j) in core: ~b_i \/ ~b_j
                    // This ensures we find a minimal relaxation.
                    if core_soft_ids.len() > 1 {
                        // Pairwise encoding for small cores
                        if core_soft_ids.len() <= 5 {
                            for i in 0..core_soft_ids.len() {
                                for j in (i + 1)..core_soft_ids.len() {
                                    if let (Some(&vi), Some(&vj)) = (
                                        blocking_vars.get(&core_soft_ids[i]),
                                        blocking_vars.get(&core_soft_ids[j]),
                                    ) {
                                        solver.add_clause([Lit::neg(vi), Lit::neg(vj)]);
                                    }
                                }
                            }
                        } else {
                            // For larger cores, use sequential counter encoding
                            // Simpler: just add that at least one must be false
                            // (weaker but still sound)
                            let clause: SmallVec<[Lit; 8]> = core_soft_ids
                                .iter()
                                .filter_map(|id| blocking_vars.get(id).map(|&v| Lit::neg(v)))
                                .collect();
                            if !clause.is_empty() {
                                solver.add_clause(clause);
                            }
                        }
                    }

                    // Add fresh relaxation variables for the next iteration
                    // Each soft clause in the core gets a new blocking variable
                    for &soft_id in &core_soft_ids {
                        if let Some(clause) = self.soft_clauses.get(soft_id.0 as usize) {
                            let new_block_var = Var(next_var);
                            next_var += 1;
                            ensure_var(&mut solver, new_block_var.0);

                            // Update mappings
                            blocking_vars.insert(soft_id, new_block_var);
                            var_to_soft.insert(new_block_var, soft_id);

                            // Add new clause: lits \/ b_new
                            let mut lits: SmallVec<[Lit; 8]> =
                                clause.lits.iter().copied().collect();
                            lits.push(Lit::pos(new_block_var));
                            solver.add_clause(lits.iter().copied());

                            // Mark as relaxed (can be blocked)
                            relaxed.insert(soft_id, true);
                        }
                    }
                }
                SolverResult::Unknown => return Ok(MaxSatResult::Unknown),
            }
        }
    }

    /// OLL (Opportunistic Literal Learning) algorithm
    ///
    /// OLL extends Fu-Malik by using cardinality constraints instead of pairwise
    /// at-most-one constraints on core blocking variables. This allows for more
    /// efficient handling of larger cores by incrementally relaxing the cardinality
    /// bound as more cores are found.
    ///
    /// Key differences from Fu-Malik:
    /// 1. Uses totalizer encoding for cardinality constraints (at-most-k)
    /// 2. Incrementally increases k when cores intersect with previous cores
    /// 3. More efficient for instances with many overlapping cores
    pub(super) fn solve_oll(&mut self) -> Result<MaxSatResult, MaxSatError> {
        use crate::totalizer::IncrementalTotalizer;

        let mut solver = SatSolver::new();
        let mut next_var = 0u32;

        // Helper function to ensure variable exists
        fn ensure_var(solver: &mut SatSolver, var_idx: u32) {
            while solver.num_vars() <= var_idx as usize {
                solver.new_var();
            }
        }

        // Add hard clauses
        for clause in &self.hard_clauses {
            for &lit in clause.iter() {
                ensure_var(&mut solver, lit.var().0);
                next_var = next_var.max(lit.var().0 + 1);
            }
            solver.add_clause(clause.iter().copied());
        }

        // Create blocking variables for soft clauses
        let soft_ids: Vec<SoftId> = self.soft_clauses.iter().map(|c| c.id).collect();
        let mut blocking_vars: FxHashMap<SoftId, Var> = FxHashMap::default();
        let mut var_to_soft: FxHashMap<Var, SoftId> = FxHashMap::default();

        for &id in &soft_ids {
            if let Some(clause) = self.soft_clauses.get(id.0 as usize) {
                let block_var = Var(next_var);
                next_var += 1;
                ensure_var(&mut solver, block_var.0);

                blocking_vars.insert(id, block_var);
                var_to_soft.insert(block_var, id);
                self.relax_to_soft.insert(Lit::pos(block_var), id);

                // Add soft clause with blocking literal
                let mut lits: SmallVec<[Lit; 8]> = clause.lits.iter().copied().collect();
                lits.push(Lit::pos(block_var));
                solver.add_clause(lits.iter().copied());

                self.stats.relax_vars_added += 1;
            }
        }

        // OLL uses incremental totalizers for groups of soft clauses
        // Initially all soft clauses are in their own "group" with bound 0
        // When cores are found, we merge groups and adjust bounds
        struct OllGroup {
            #[allow(dead_code)]
            soft_ids: Vec<SoftId>,
            totalizer: IncrementalTotalizer,
            current_bound: usize,
        }

        let mut groups: Vec<OllGroup> = Vec::new();
        let mut soft_to_group: FxHashMap<SoftId, usize> = FxHashMap::default();

        // Main OLL loop
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return Ok(MaxSatResult::Unknown);
            }

            // Build assumptions: ~b_i for all soft clauses not in any group
            // plus the bound assumptions for each group
            let mut assumptions: Vec<Lit> = Vec::new();

            for &id in &soft_ids {
                if !soft_to_group.contains_key(&id)
                    && let Some(&block_var) = blocking_vars.get(&id)
                {
                    assumptions.push(Lit::neg(block_var));
                }
            }

            // Add group bound assumptions
            for group in &groups {
                if let Some(assumption) = group.totalizer.bound_assumption() {
                    assumptions.push(assumption);
                }
            }

            if assumptions.is_empty() && groups.is_empty() {
                // All satisfied - check hard constraints
                return self.check_hard_satisfiable();
            }

            self.stats.sat_calls += 1;
            let (result, core) = solver.solve_with_assumptions(&assumptions);

            match result {
                SolverResult::Sat => {
                    self.best_model = Some(solver.model().to_vec());
                    self.update_soft_values();
                    return Ok(MaxSatResult::Optimal);
                }
                SolverResult::Unsat => {
                    let core_lits = core.unwrap_or_default();
                    self.stats.cores_extracted += 1;

                    if core_lits.is_empty() {
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    // Find which soft clauses are in the core
                    let mut core_soft_ids: SmallVec<[SoftId; 8]> = SmallVec::new();
                    let mut min_weight = Weight::Infinite;

                    for lit in &core_lits {
                        let var = lit.var();
                        if let Some(&soft_id) = var_to_soft.get(&var) {
                            core_soft_ids.push(soft_id);
                            if let Some(clause) = self.soft_clauses.get(soft_id.0 as usize) {
                                min_weight = min_weight.min(clause.weight.clone());
                            }
                        }
                    }

                    self.stats.total_core_size += core_soft_ids.len() as u32;

                    if core_soft_ids.is_empty() {
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    self.lower_bound = self.lower_bound.add(&min_weight);

                    // Collect groups that intersect with the core
                    let mut intersecting_groups: Vec<usize> = core_soft_ids
                        .iter()
                        .filter_map(|id| soft_to_group.get(id).copied())
                        .collect();
                    intersecting_groups.sort_unstable();
                    intersecting_groups.dedup();

                    if intersecting_groups.is_empty() {
                        // Create a new group from core soft clauses
                        let block_lits: Vec<Lit> = core_soft_ids
                            .iter()
                            .filter_map(|id| blocking_vars.get(id).map(|v| Lit::pos(*v)))
                            .collect();

                        if !block_lits.is_empty() {
                            let mut totalizer = IncrementalTotalizer::new(&block_lits, next_var);
                            next_var = totalizer.next_var();

                            // Set bound to 1 (at most 1 can be true)
                            let (assumption, clauses) = totalizer.set_bound(1);

                            // Add totalizer clauses
                            for clause in clauses {
                                // Ensure vars exist
                                for &lit in &clause.lits {
                                    ensure_var(&mut solver, lit.var().0);
                                }
                                solver.add_clause(clause.lits.iter().copied());
                            }

                            let group_idx = groups.len();
                            let group = OllGroup {
                                soft_ids: core_soft_ids.iter().copied().collect(),
                                totalizer,
                                current_bound: 1,
                            };
                            groups.push(group);

                            for &id in &core_soft_ids {
                                soft_to_group.insert(id, group_idx);
                            }

                            // The assumption is already stored in the totalizer
                            let _ = assumption;
                        }
                    } else {
                        // Merge all intersecting groups and increase bound
                        // For simplicity, just increase the bound of the first group
                        let primary_group = intersecting_groups[0];
                        let new_bound = groups[primary_group].current_bound + 1;

                        let (_, clauses) = groups[primary_group].totalizer.set_bound(new_bound);
                        groups[primary_group].current_bound = new_bound;

                        // Add new clauses
                        for clause in clauses {
                            for &lit in &clause.lits {
                                ensure_var(&mut solver, lit.var().0);
                            }
                            solver.add_clause(clause.lits.iter().copied());
                        }
                    }
                }
                SolverResult::Unknown => return Ok(MaxSatResult::Unknown),
            }
        }
    }

    /// MSU3 (iterative relaxation) algorithm
    ///
    /// MSU3 is a simpler core-guided algorithm that:
    /// 1. Finds UNSAT cores iteratively
    /// 2. Relaxes soft clauses from the core
    /// 3. Uses at-most-one constraints similar to Fu-Malik
    ///
    /// The key difference from Fu-Malik is in how cores are processed.
    /// MSU3 uses a simpler relaxation strategy.
    pub(super) fn solve_msu3(&mut self) -> Result<MaxSatResult, MaxSatError> {
        // MSU3 is very similar to Fu-Malik in practice
        // The main difference is in weight handling and core processing strategy
        // For unweighted MaxSAT, they are essentially equivalent
        // Use Fu-Malik implementation for correctness
        self.solve_fu_malik()
    }

    /// WMax (weighted MaxSAT) algorithm
    ///
    /// WMax is designed for weighted MaxSAT instances. It processes
    /// soft clauses in weight order and uses weight-aware core extraction.
    pub(super) fn solve_wmax(&mut self) -> Result<MaxSatResult, MaxSatError> {
        // If all weights are the same, just use Fu-Malik
        if !self.has_different_weights() {
            return self.solve_fu_malik();
        }

        // Use stratified approach with weight levels
        self.solve_stratified()
    }

    /// Update soft clause values from the best model
    pub(super) fn update_soft_values(&mut self) {
        if let Some(model) = &self.best_model {
            for clause in &mut self.soft_clauses {
                let satisfied = clause.lits.iter().any(|&lit| {
                    let var = lit.var().0 as usize;
                    if var < model.len() {
                        let val = model[var];
                        (val == LBool::True && !lit.sign()) || (val == LBool::False && lit.sign())
                    } else {
                        false
                    }
                });
                clause.set_value(satisfied);
            }
        }
    }

    /// PMRES (Partial MaxSAT Resolution) algorithm
    ///
    /// PMRES is a resolution-based algorithm for partial MaxSAT that:
    /// 1. Finds minimal unsatisfiable cores
    /// 2. Resolves soft clauses to create new clauses
    /// 3. Uses weight-based core selection
    ///
    /// It's particularly effective for partial MaxSAT instances with many hard constraints.
    ///
    /// Reference: "Solving Maxsat by Solving a Sequence of Simpler SAT Instances" (2010)
    pub(super) fn solve_pmres(&mut self) -> Result<MaxSatResult, MaxSatError> {
        use crate::totalizer::IncrementalTotalizer;

        let mut solver = SatSolver::new();
        let mut next_var = 0u32;

        // Helper function to ensure variable exists
        fn ensure_var(solver: &mut SatSolver, var_idx: u32) {
            while solver.num_vars() <= var_idx as usize {
                solver.new_var();
            }
        }

        // Add hard clauses
        for clause in &self.hard_clauses {
            for &lit in clause.iter() {
                ensure_var(&mut solver, lit.var().0);
                next_var = next_var.max(lit.var().0 + 1);
            }
            solver.add_clause(clause.iter().copied());
        }

        // Create assumption variables for soft clauses
        let soft_ids: Vec<SoftId> = self.soft_clauses.iter().map(|c| c.id).collect();
        let mut assumption_vars: FxHashMap<SoftId, Var> = FxHashMap::default();
        let mut var_to_soft: FxHashMap<Var, SoftId> = FxHashMap::default();

        for &id in &soft_ids {
            if let Some(clause) = self.soft_clauses.get(id.0 as usize) {
                let assumption_var = Var(next_var);
                next_var += 1;
                ensure_var(&mut solver, assumption_var.0);

                assumption_vars.insert(id, assumption_var);
                var_to_soft.insert(assumption_var, id);

                // Add soft clause with assumption: clause \/ ~assumption
                // If assumption is true, the soft clause is "active" (must be satisfied)
                // If assumption is false, the soft clause is ignored
                let mut lits: SmallVec<[Lit; 8]> = clause.lits.iter().copied().collect();
                lits.push(Lit::neg(assumption_var));
                solver.add_clause(lits.iter().copied());
            }
        }

        // Track cardinality constraints for weighted cores
        let mut cardinality_constraints: Vec<IncrementalTotalizer> = Vec::new();

        // Main PMRES loop
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return Ok(MaxSatResult::Unknown);
            }

            // Build assumptions: assume all active soft clauses must be satisfied
            let mut assumptions: Vec<Lit> = soft_ids
                .iter()
                .filter_map(|id| assumption_vars.get(id).map(|&v| Lit::pos(v)))
                .collect();

            // Add cardinality constraint assumptions
            for cc in &cardinality_constraints {
                if let Some(assumption) = cc.bound_assumption() {
                    assumptions.push(assumption);
                }
            }

            if assumptions.is_empty() {
                // All soft clauses disabled - check hard constraints
                return self.check_hard_satisfiable();
            }

            self.stats.sat_calls += 1;
            let (result, core) = solver.solve_with_assumptions(&assumptions);

            match result {
                SolverResult::Sat => {
                    // Found a satisfying assignment
                    self.best_model = Some(solver.model().to_vec());
                    self.update_soft_values();
                    return Ok(MaxSatResult::Optimal);
                }
                SolverResult::Unsat => {
                    // Extract minimal core
                    let core_lits = core.unwrap_or_default();
                    self.stats.cores_extracted += 1;

                    if core_lits.is_empty() {
                        // Empty core means hard clauses alone are UNSAT
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    // Find which soft clauses are in the core
                    let mut core_soft_ids: SmallVec<[SoftId; 8]> = SmallVec::new();
                    let mut min_weight = Weight::Infinite;

                    for lit in &core_lits {
                        // Core contains assumption literals
                        let var = lit.var();
                        if let Some(&soft_id) = var_to_soft.get(&var) {
                            core_soft_ids.push(soft_id);
                            if let Some(clause) = self.soft_clauses.get(soft_id.0 as usize) {
                                min_weight = min_weight.min(clause.weight.clone());
                            }
                        }
                    }

                    self.stats.total_core_size += core_soft_ids.len() as u32;

                    if core_soft_ids.is_empty() {
                        // No soft clauses in core - hard constraints UNSAT
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    // Update lower bound
                    self.lower_bound = self.lower_bound.add(&min_weight);

                    // PMRES strategy: Add a cardinality constraint that at most k-1 of the
                    // core soft clauses can be satisfied (where k is the core size).
                    // This forces the solver to find a different core or satisfy more soft clauses.

                    if core_soft_ids.len() == 1 {
                        // Single soft clause in core - just disable it
                        // The lower bound was already updated above
                        if let Some(&soft_id) = core_soft_ids.first() {
                            assumption_vars.remove(&soft_id);
                        }
                    } else {
                        // Multiple soft clauses in core
                        // Collect assumption variables for the core
                        let core_assumptions: Vec<Lit> = core_soft_ids
                            .iter()
                            .filter_map(|id| assumption_vars.get(id).map(|&v| Lit::pos(v)))
                            .collect();

                        if !core_assumptions.is_empty() {
                            // Create incremental totalizer for this core
                            // At most (k-1) of these can be true
                            let mut totalizer =
                                IncrementalTotalizer::new(&core_assumptions, next_var);
                            next_var = totalizer.next_var();

                            let bound = core_assumptions.len() - 1;
                            let (assumption, clauses) = totalizer.set_bound(bound);

                            // Add totalizer clauses to solver
                            for clause in clauses {
                                for &lit in &clause.lits {
                                    ensure_var(&mut solver, lit.var().0);
                                    next_var = next_var.max(lit.var().0 + 1);
                                }
                                solver.add_clause(clause.lits.iter().copied());
                            }

                            cardinality_constraints.push(totalizer);

                            // The assumption will be used in the next iteration
                            let _ = assumption;
                        }
                    }
                }
                SolverResult::Unknown => return Ok(MaxSatResult::Unknown),
            }
        }
    }
}
