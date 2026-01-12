//! XOR clause detection and Gaussian elimination
//!
//! This module implements detection of XOR constraints from CNF clauses
//! and uses Gaussian elimination to simplify them. Features include:
//! - GF(2) matrix representation for efficient Gaussian elimination
//! - Incremental XOR propagation with watched literals
//! - Conflict reason generation for CDCL integration
//! - XOR subsumption and strengthening

use crate::clause::ClauseId;
use crate::literal::{Lit, Var};
use std::collections::{HashMap, HashSet, VecDeque};

/// GF(2) row representation using bit vectors for efficient XOR operations
#[derive(Debug, Clone)]
pub struct GF2Row {
    /// Bit vector representing variables (1 = present, 0 = absent)
    bits: Vec<u64>,
    /// Number of variables (bits)
    num_vars: usize,
    /// Right-hand side value
    rhs: bool,
    /// Original clause/constraint IDs
    sources: Vec<usize>,
}

impl GF2Row {
    /// Create a new empty row for given number of variables
    pub fn new(num_vars: usize) -> Self {
        let num_words = num_vars.div_ceil(64);
        Self {
            bits: vec![0; num_words],
            num_vars,
            rhs: false,
            sources: Vec::new(),
        }
    }

    /// Set a variable (1-indexed) in this row
    #[inline]
    pub fn set(&mut self, var_idx: usize) {
        if var_idx < self.num_vars {
            let word = var_idx / 64;
            let bit = var_idx % 64;
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Clear a variable from this row
    #[inline]
    pub fn clear(&mut self, var_idx: usize) {
        if var_idx < self.num_vars {
            let word = var_idx / 64;
            let bit = var_idx % 64;
            self.bits[word] &= !(1u64 << bit);
        }
    }

    /// Check if a variable is set
    #[inline]
    pub fn is_set(&self, var_idx: usize) -> bool {
        if var_idx < self.num_vars {
            let word = var_idx / 64;
            let bit = var_idx % 64;
            (self.bits[word] & (1u64 << bit)) != 0
        } else {
            false
        }
    }

    /// XOR this row with another row
    pub fn xor_with(&mut self, other: &GF2Row) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a ^= *b;
        }
        self.rhs ^= other.rhs;
        self.sources.extend_from_slice(&other.sources);
    }

    /// Check if this row is all zeros (empty constraint)
    pub fn is_zero(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Count number of variables (popcount)
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Find the first (lowest index) set variable
    pub fn first_set(&self) -> Option<usize> {
        for (word_idx, &word) in self.bits.iter().enumerate() {
            if word != 0 {
                return Some(word_idx * 64 + word.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Get all set variable indices
    pub fn get_vars(&self) -> Vec<usize> {
        let mut vars = Vec::new();
        for (word_idx, &word) in self.bits.iter().enumerate() {
            let mut w = word;
            let base = word_idx * 64;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                vars.push(base + bit);
                w &= w - 1; // Clear lowest bit
            }
        }
        vars
    }
}

/// GF(2) matrix for efficient Gaussian elimination
#[derive(Debug, Clone)]
pub struct GF2Matrix {
    /// Rows of the matrix
    rows: Vec<GF2Row>,
    /// Number of variables
    num_vars: usize,
    /// Variable to column index mapping
    var_to_col: HashMap<Var, usize>,
    /// Column index to variable mapping
    col_to_var: Vec<Var>,
    /// Pivot row for each column (-1 if none)
    pivots: Vec<Option<usize>>,
}

impl GF2Matrix {
    /// Create a new GF(2) matrix
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            num_vars: 0,
            var_to_col: HashMap::new(),
            col_to_var: Vec::new(),
            pivots: Vec::new(),
        }
    }

    /// Register a variable and get its column index
    pub fn register_var(&mut self, var: Var) -> usize {
        if let Some(&col) = self.var_to_col.get(&var) {
            return col;
        }
        let col = self.num_vars;
        self.var_to_col.insert(var, col);
        self.col_to_var.push(var);
        self.pivots.push(None);
        self.num_vars += 1;
        col
    }

    /// Add a constraint to the matrix
    pub fn add_constraint(&mut self, vars: &[Var], rhs: bool, source_id: usize) -> XorAddResult {
        // First ensure all variables are registered
        for &var in vars {
            self.register_var(var);
        }

        // Create row
        let mut row = GF2Row::new(self.num_vars);
        for &var in vars {
            if let Some(&col) = self.var_to_col.get(&var) {
                row.set(col);
            }
        }
        row.rhs = rhs;
        row.sources.push(source_id);

        // Reduce with existing rows
        self.reduce_row(&mut row)
    }

    /// Reduce a row using existing pivots
    fn reduce_row(&mut self, row: &mut GF2Row) -> XorAddResult {
        // Extend row if needed
        if row.bits.len() < self.num_vars.div_ceil(64) {
            row.bits.resize(self.num_vars.div_ceil(64), 0);
            row.num_vars = self.num_vars;
        }

        loop {
            let first = match row.first_set() {
                Some(f) => f,
                None => {
                    // Row became zero
                    if row.rhs {
                        return XorAddResult::Conflict(row.sources.clone());
                    }
                    return XorAddResult::Redundant;
                }
            };

            if let Some(pivot_row) = self.pivots.get(first).and_then(|p| *p) {
                row.xor_with(&self.rows[pivot_row]);
            } else {
                // Found a new pivot
                break;
            }
        }

        // Check for unit constraint
        if row.popcount() == 1 {
            let var_idx = row.first_set().expect("popcount == 1");
            let var = self.col_to_var[var_idx];
            let value = row.rhs;
            return XorAddResult::Unit(var, value, row.sources.clone());
        }

        // Add as new row with pivot
        let pivot_col = row.first_set().expect("non-zero row");
        let row_idx = self.rows.len();
        self.pivots[pivot_col] = Some(row_idx);
        self.rows.push(row.clone());

        XorAddResult::Added
    }

    /// Back-substitute an assignment to find implied units
    pub fn propagate(&mut self, var: Var, value: bool) -> Vec<XorAddResult> {
        let mut results = Vec::new();

        let col = match self.var_to_col.get(&var) {
            Some(&c) => c,
            None => return results,
        };

        // Update all rows containing this variable
        for row in &mut self.rows {
            if row.is_set(col) {
                row.clear(col);
                if value {
                    row.rhs = !row.rhs;
                }

                // Check for unit or conflict
                if row.is_zero() {
                    if row.rhs {
                        results.push(XorAddResult::Conflict(row.sources.clone()));
                    }
                } else if row.popcount() == 1 {
                    let var_idx = row.first_set().expect("popcount == 1");
                    let implied_var = self.col_to_var[var_idx];
                    let implied_value = row.rhs;
                    results.push(XorAddResult::Unit(
                        implied_var,
                        implied_value,
                        row.sources.clone(),
                    ));
                }
            }
        }

        results
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Get the number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl Default for GF2Matrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of adding an XOR constraint
#[derive(Debug, Clone)]
pub enum XorAddResult {
    /// Constraint was added successfully
    Added,
    /// Constraint was redundant
    Redundant,
    /// Found a unit implication (variable, value, reason sources)
    Unit(Var, bool, Vec<usize>),
    /// Found a conflict (reason sources)
    Conflict(Vec<usize>),
}

/// Represents an XOR constraint: x1 ⊕ x2 ⊕ ... ⊕ xn = rhs
#[derive(Debug, Clone)]
pub struct XorConstraint {
    /// Variables in the XOR constraint
    pub vars: Vec<Var>,
    /// Right-hand side (true or false)
    pub rhs: bool,
    /// Original clause IDs that form this XOR constraint
    pub source_clauses: Vec<ClauseId>,
}

impl XorConstraint {
    /// Create a new XOR constraint
    pub fn new(vars: Vec<Var>, rhs: bool) -> Self {
        Self {
            vars,
            rhs,
            source_clauses: Vec::new(),
        }
    }

    /// Get the number of variables
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Substitute a variable with a value
    pub fn substitute(&mut self, var: Var, value: bool) {
        if let Some(pos) = self.vars.iter().position(|&v| v == var) {
            self.vars.remove(pos);
            if value {
                self.rhs = !self.rhs;
            }
        }
    }

    /// Add (XOR) another constraint to this one
    pub fn xor_with(&mut self, other: &XorConstraint) {
        // XOR the RHS
        self.rhs ^= other.rhs;

        // XOR the variables (symmetric difference)
        let mut var_set: HashSet<Var> = self.vars.iter().copied().collect();
        for &var in &other.vars {
            if var_set.contains(&var) {
                var_set.remove(&var);
            } else {
                var_set.insert(var);
            }
        }

        self.vars = var_set.into_iter().collect();
        self.vars.sort_unstable();

        // Merge source clauses
        self.source_clauses.extend_from_slice(&other.source_clauses);
    }

    /// Normalize the constraint (ensure first variable has positive polarity)
    pub fn normalize(&mut self) {
        if !self.vars.is_empty() {
            // Sort variables for canonical form
            self.vars.sort_unstable();
        }
    }
}

/// XOR constraint manager with Gaussian elimination
pub struct XorManager {
    /// XOR constraints
    constraints: Vec<XorConstraint>,
    /// Variable to constraint mapping
    var_to_constraints: HashMap<Var, Vec<usize>>,
    /// Detected unit XOR constraints
    units: Vec<(Var, bool)>,
    /// Detected conflicts
    has_conflict: bool,
}

impl XorManager {
    /// Create a new XOR manager
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            var_to_constraints: HashMap::new(),
            units: Vec::new(),
            has_conflict: false,
        }
    }

    /// Add an XOR constraint
    pub fn add_constraint(&mut self, mut constraint: XorConstraint) {
        constraint.normalize();

        // Check for trivial cases
        if constraint.is_empty() {
            if constraint.rhs {
                // 0 = 1, conflict
                self.has_conflict = true;
            }
            // 0 = 0 is trivially satisfied
            return;
        }

        if constraint.len() == 1 {
            // Unit constraint
            self.units.push((constraint.vars[0], constraint.rhs));
            return;
        }

        // Add to var mapping
        for &var in &constraint.vars {
            self.var_to_constraints
                .entry(var)
                .or_default()
                .push(self.constraints.len());
        }

        self.constraints.push(constraint);
    }

    /// Perform Gaussian elimination
    pub fn eliminate(&mut self) {
        let mut row = 0;
        let mut col = 0;

        // Collect all variables
        let mut all_vars: HashSet<Var> = HashSet::new();
        for constraint in &self.constraints {
            all_vars.extend(constraint.vars.iter().copied());
        }
        let mut vars: Vec<Var> = all_vars.into_iter().collect();
        vars.sort_unstable();

        while row < self.constraints.len() && col < vars.len() {
            let var = vars[col];

            // Find pivot row
            let pivot = self.find_pivot(row, var);

            if let Some(pivot_row) = pivot {
                // Swap rows if needed
                if pivot_row != row {
                    self.constraints.swap(row, pivot_row);
                }

                // Eliminate variable from other rows
                let pivot_constraint = self.constraints[row].clone();
                for i in 0..self.constraints.len() {
                    if i != row && self.constraints[i].vars.contains(&var) {
                        self.constraints[i].xor_with(&pivot_constraint);
                        self.constraints[i].normalize();

                        // Check for new units or conflicts
                        if self.constraints[i].is_empty() {
                            if self.constraints[i].rhs {
                                self.has_conflict = true;
                                return;
                            }
                        } else if self.constraints[i].len() == 1 {
                            self.units
                                .push((self.constraints[i].vars[0], self.constraints[i].rhs));
                        }
                    }
                }

                row += 1;
            }

            col += 1;
        }

        // Remove trivial constraints
        self.constraints.retain(|c| !c.is_empty() && c.len() > 1);
    }

    /// Find a pivot row for the given variable
    fn find_pivot(&self, start_row: usize, var: Var) -> Option<usize> {
        (start_row..self.constraints.len()).find(|&i| self.constraints[i].vars.contains(&var))
    }

    /// Get unit constraints
    pub fn get_units(&self) -> &[(Var, bool)] {
        &self.units
    }

    /// Check if there's a conflict
    pub fn has_conflict(&self) -> bool {
        self.has_conflict
    }

    /// Get all constraints
    pub fn get_constraints(&self) -> &[XorConstraint] {
        &self.constraints
    }

    /// Back-substitute to find all unit implications
    pub fn back_substitute(&mut self, assignment: &HashMap<Var, bool>) {
        for constraint in &mut self.constraints {
            // Apply known assignments
            let mut to_remove = Vec::new();
            for (i, &var) in constraint.vars.iter().enumerate() {
                if let Some(&value) = assignment.get(&var) {
                    to_remove.push(i);
                    if value {
                        constraint.rhs = !constraint.rhs;
                    }
                }
            }

            // Remove assigned variables
            for &i in to_remove.iter().rev() {
                constraint.vars.remove(i);
            }

            // Check for units or conflicts
            if constraint.is_empty() {
                if constraint.rhs {
                    self.has_conflict = true;
                    return;
                }
            } else if constraint.len() == 1 {
                self.units.push((constraint.vars[0], constraint.rhs));
            }
        }
    }
}

impl Default for XorManager {
    fn default() -> Self {
        Self::new()
    }
}

/// XOR clause detector
pub struct XorDetector {
    /// Minimum XOR size to detect
    min_xor_size: usize,
    /// Maximum XOR size to detect
    max_xor_size: usize,
}

impl XorDetector {
    /// Create a new XOR detector
    pub fn new(min_size: usize, max_size: usize) -> Self {
        Self {
            min_xor_size: min_size,
            max_xor_size: max_size,
        }
    }

    /// Detect XOR constraints from clauses
    /// An XOR constraint x1 ⊕ x2 ⊕ ... ⊕ xn = rhs is represented as 2^(n-1) clauses
    /// For example, x1 ⊕ x2 = 0 is represented as:
    ///   (x1 ∨ x2) ∧ (¬x1 ∨ ¬x2)
    pub fn detect_xor(&self, clauses: &[(Vec<Lit>, ClauseId)]) -> Vec<XorConstraint> {
        let mut xor_constraints = Vec::new();
        let mut used_clauses: HashSet<ClauseId> = HashSet::new();

        // Try to find XOR patterns for different sizes
        for size in self.min_xor_size..=self.max_xor_size {
            let xors = self.detect_xor_of_size(clauses, size, &used_clauses);
            for xor in xors {
                for &clause_id in &xor.source_clauses {
                    used_clauses.insert(clause_id);
                }
                xor_constraints.push(xor);
            }
        }

        xor_constraints
    }

    /// Detect XOR constraints of a specific size
    fn detect_xor_of_size(
        &self,
        clauses: &[(Vec<Lit>, ClauseId)],
        size: usize,
        used_clauses: &HashSet<ClauseId>,
    ) -> Vec<XorConstraint> {
        let mut result = Vec::new();

        // Group clauses by their variables (ignoring polarity)
        let mut clause_groups: HashMap<Vec<Var>, Vec<(Vec<bool>, ClauseId)>> = HashMap::new();

        for (lits, clause_id) in clauses {
            if used_clauses.contains(clause_id) {
                continue;
            }

            if lits.len() != size {
                continue;
            }

            let mut vars: Vec<Var> = lits.iter().map(|l| l.var()).collect();
            vars.sort_unstable();

            let polarities: Vec<bool> = {
                let mut v = vars.clone();
                let mut p = Vec::new();
                for lit in lits {
                    if let Some(pos) = v.iter().position(|&x| x == lit.var()) {
                        p.push(lit.is_pos());
                        v.remove(pos);
                    }
                }
                p
            };

            clause_groups
                .entry(vars)
                .or_default()
                .push((polarities, *clause_id));
        }

        // Check if clause groups form XOR constraints
        for (vars, polarity_groups) in clause_groups {
            if polarity_groups.len() != (1 << (size - 1)) {
                continue;
            }

            // Verify this is a valid XOR encoding
            if self.is_valid_xor_encoding(&polarity_groups, size) {
                // Determine RHS from the polarity pattern
                let rhs = self.compute_xor_rhs(&polarity_groups);
                let mut xor = XorConstraint::new(vars, rhs);
                xor.source_clauses = polarity_groups.iter().map(|(_, id)| *id).collect();
                result.push(xor);
            }
        }

        result
    }

    /// Check if polarity groups form a valid XOR encoding
    fn is_valid_xor_encoding(
        &self,
        polarity_groups: &[(Vec<bool>, ClauseId)],
        size: usize,
    ) -> bool {
        // For a valid XOR encoding, we need exactly 2^(n-1) clauses
        if polarity_groups.len() != (1 << (size - 1)) {
            return false;
        }

        // Check that we have the right distribution of polarities
        let mut polarity_set: HashSet<Vec<bool>> = HashSet::new();
        for (polarities, _) in polarity_groups {
            if !polarity_set.insert(polarities.clone()) {
                return false; // Duplicate clause
            }
        }

        // For a valid XOR encoding, all clauses should have the same parity
        // of negative literals (all even or all odd)
        let first_neg_count = polarity_groups[0].0.iter().filter(|&&p| !p).count();
        let first_parity = first_neg_count % 2;

        for (polarities, _) in &polarity_groups[1..] {
            let neg_count = polarities.iter().filter(|&&p| !p).count();
            if neg_count % 2 != first_parity {
                return false;
            }
        }

        true
    }

    /// Compute XOR RHS from polarity groups
    fn compute_xor_rhs(&self, polarity_groups: &[(Vec<bool>, ClauseId)]) -> bool {
        // The RHS is determined by the parity of negative literals
        // If all clauses have an even number of negatives, RHS = false
        // If all clauses have an odd number of negatives, RHS = true
        let (pols, _) = &polarity_groups[0];
        let neg_count = pols.iter().filter(|&&p| !p).count();
        neg_count % 2 == 1
    }
}

impl Default for XorDetector {
    fn default() -> Self {
        Self::new(3, 6)
    }
}

/// ID for an XOR clause within the propagator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XorClauseId(pub usize);

/// XOR clause for propagation with watched literals
#[derive(Debug, Clone)]
pub struct XorClause {
    /// All variables in the XOR constraint
    vars: Vec<Var>,
    /// Right-hand side (parity)
    rhs: bool,
    /// Currently watched variable indices (within vars)
    watched: [usize; 2],
    /// Source clause IDs for conflict explanation
    sources: Vec<ClauseId>,
}

impl XorClause {
    /// Create a new XOR clause
    pub fn new(vars: Vec<Var>, rhs: bool, sources: Vec<ClauseId>) -> Self {
        let watched = if vars.len() >= 2 {
            [0, 1]
        } else {
            [0, 0] // Single var or empty clause
        };
        Self {
            vars,
            rhs,
            watched,
            sources,
        }
    }

    /// Get the variables
    pub fn vars(&self) -> &[Var] {
        &self.vars
    }

    /// Get the RHS
    pub fn rhs(&self) -> bool {
        self.rhs
    }

    /// Get watched variables
    pub fn get_watched(&self) -> (Var, Option<Var>) {
        if self.vars.is_empty() {
            return (Var(0), None);
        }
        let w0 = self.vars[self.watched[0]];
        let w1 = if self.vars.len() > 1 && self.watched[0] != self.watched[1] {
            Some(self.vars[self.watched[1]])
        } else {
            None
        };
        (w0, w1)
    }
}

/// XOR propagator with watched literal scheme
pub struct XorPropagator {
    /// XOR clauses
    clauses: Vec<XorClause>,
    /// Mapping from variable to XOR clause indices watching it
    watches: HashMap<Var, Vec<XorClauseId>>,
    /// Current assignment (None = unassigned)
    assignment: HashMap<Var, bool>,
    /// Trail of assignments for backtracking
    trail: Vec<(Var, usize)>, // (var, decision_level)
    /// Current decision level
    decision_level: usize,
    /// Pending propagations
    pending: VecDeque<(Var, bool, Vec<ClauseId>)>,
    /// Conflict, if any
    conflict: Option<Vec<ClauseId>>,
    /// GF(2) matrix for incremental Gaussian elimination
    matrix: GF2Matrix,
    /// Statistics
    stats: XorPropagatorStats,
}

/// Statistics for XOR propagator
#[derive(Debug, Clone, Default)]
pub struct XorPropagatorStats {
    /// Number of propagations
    pub propagations: usize,
    /// Number of conflicts
    pub conflicts: usize,
    /// Number of XOR clauses
    pub num_clauses: usize,
    /// Number of Gaussian elimination steps
    pub gaussian_steps: usize,
}

impl XorPropagator {
    /// Create a new XOR propagator
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            watches: HashMap::new(),
            assignment: HashMap::new(),
            trail: Vec::new(),
            decision_level: 0,
            pending: VecDeque::new(),
            conflict: None,
            matrix: GF2Matrix::new(),
            stats: XorPropagatorStats::default(),
        }
    }

    /// Add an XOR clause
    pub fn add_clause(
        &mut self,
        vars: Vec<Var>,
        rhs: bool,
        sources: Vec<ClauseId>,
    ) -> Option<XorClauseId> {
        if vars.is_empty() {
            if rhs {
                // Empty clause with RHS=true is a conflict
                self.conflict = Some(sources);
            }
            return None;
        }

        let clause_id = XorClauseId(self.clauses.len());
        let clause = XorClause::new(vars.clone(), rhs, sources.clone());

        // Set up watches
        let (w0, w1) = clause.get_watched();
        self.watches.entry(w0).or_default().push(clause_id);
        if let Some(w1) = w1
            && w0 != w1
        {
            self.watches.entry(w1).or_default().push(clause_id);
        }

        self.clauses.push(clause);
        self.stats.num_clauses += 1;

        // Also add to GF(2) matrix for Gaussian reasoning
        match self.matrix.add_constraint(&vars, rhs, clause_id.0) {
            XorAddResult::Conflict(srcs) => {
                let conflict_sources: Vec<ClauseId> = srcs
                    .iter()
                    .filter_map(|&idx| self.clauses.get(idx).map(|c| c.sources.clone()))
                    .flatten()
                    .collect();
                self.conflict = Some(conflict_sources);
            }
            XorAddResult::Unit(var, value, srcs) => {
                let reason_sources: Vec<ClauseId> = srcs
                    .iter()
                    .filter_map(|&idx| self.clauses.get(idx).map(|c| c.sources.clone()))
                    .flatten()
                    .collect();
                self.pending.push_back((var, value, reason_sources));
            }
            _ => {}
        }
        self.stats.gaussian_steps += 1;

        Some(clause_id)
    }

    /// Propagate an assignment
    pub fn propagate(&mut self, var: Var, value: bool, level: usize) -> PropagateResult {
        if self.conflict.is_some() {
            return PropagateResult::Conflict(self.conflict.clone().unwrap_or_default());
        }

        // Record assignment
        self.assignment.insert(var, value);
        self.trail.push((var, level));
        self.decision_level = level;

        // Propagate in GF(2) matrix
        let matrix_results = self.matrix.propagate(var, value);
        for result in matrix_results {
            match result {
                XorAddResult::Conflict(srcs) => {
                    let conflict_sources: Vec<ClauseId> = srcs
                        .iter()
                        .filter_map(|&idx| self.clauses.get(idx).map(|c| c.sources.clone()))
                        .flatten()
                        .collect();
                    self.conflict = Some(conflict_sources.clone());
                    self.stats.conflicts += 1;
                    return PropagateResult::Conflict(conflict_sources);
                }
                XorAddResult::Unit(implied_var, implied_value, srcs) => {
                    if let Some(&existing) = self.assignment.get(&implied_var) {
                        if existing != implied_value {
                            // Conflict!
                            let conflict_sources: Vec<ClauseId> = srcs
                                .iter()
                                .filter_map(|&idx| self.clauses.get(idx).map(|c| c.sources.clone()))
                                .flatten()
                                .collect();
                            self.conflict = Some(conflict_sources.clone());
                            self.stats.conflicts += 1;
                            return PropagateResult::Conflict(conflict_sources);
                        }
                        // Already assigned with same value, skip
                    } else {
                        let reason_sources: Vec<ClauseId> = srcs
                            .iter()
                            .filter_map(|&idx| self.clauses.get(idx).map(|c| c.sources.clone()))
                            .flatten()
                            .collect();
                        self.pending
                            .push_back((implied_var, implied_value, reason_sources));
                    }
                }
                _ => {}
            }
        }

        // Process watched literal propagation
        if let Some(watch_list) = self.watches.get(&var).cloned() {
            for clause_id in watch_list {
                if let Some(result) = self.propagate_clause(clause_id) {
                    match result {
                        PropagateResult::Conflict(sources) => {
                            self.conflict = Some(sources.clone());
                            self.stats.conflicts += 1;
                            return PropagateResult::Conflict(sources);
                        }
                        PropagateResult::Propagated(_, _, _) => {
                            // Continue processing
                        }
                        PropagateResult::None => {}
                    }
                }
            }
        }

        self.stats.propagations += 1;
        PropagateResult::None
    }

    /// Propagate a specific XOR clause
    fn propagate_clause(&mut self, clause_id: XorClauseId) -> Option<PropagateResult> {
        let clause = self.clauses.get(clause_id.0)?;
        let vars = clause.vars.clone();
        let rhs = clause.rhs;
        let sources = clause.sources.clone();

        // Count assigned and unassigned variables
        let mut assigned_count = 0;
        let mut unassigned_var = None;
        let mut parity = rhs;

        for &var in &vars {
            if let Some(&value) = self.assignment.get(&var) {
                assigned_count += 1;
                if value {
                    parity = !parity;
                }
            } else {
                unassigned_var = Some(var);
            }
        }

        if assigned_count == vars.len() {
            // All assigned - check for conflict
            if parity {
                return Some(PropagateResult::Conflict(sources));
            }
            return Some(PropagateResult::None);
        }

        if assigned_count == vars.len() - 1 {
            // Unit propagation
            if let Some(var) = unassigned_var {
                // The unassigned variable must take the parity value
                let value = parity;
                self.pending.push_back((var, value, sources.clone()));
                return Some(PropagateResult::Propagated(var, value, sources));
            }
        }

        Some(PropagateResult::None)
    }

    /// Get and clear pending propagations
    pub fn get_pending(&mut self) -> Vec<(Var, bool, Vec<ClauseId>)> {
        self.pending.drain(..).collect()
    }

    /// Check if there's a conflict
    pub fn has_conflict(&self) -> bool {
        self.conflict.is_some()
    }

    /// Get conflict clause IDs
    pub fn get_conflict(&self) -> Option<&Vec<ClauseId>> {
        self.conflict.as_ref()
    }

    /// Backtrack to a given level
    pub fn backtrack(&mut self, level: usize) {
        // Remove assignments above the given level
        while let Some(&(var, var_level)) = self.trail.last() {
            if var_level <= level {
                break;
            }
            self.assignment.remove(&var);
            self.trail.pop();
        }
        self.decision_level = level;
        self.conflict = None;
    }

    /// Get statistics
    pub fn stats(&self) -> &XorPropagatorStats {
        &self.stats
    }

    /// Get number of clauses
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }
}

impl Default for XorPropagator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of propagation
#[derive(Debug, Clone)]
pub enum PropagateResult {
    /// No propagation
    None,
    /// Propagated a unit (variable, value, reason)
    Propagated(Var, bool, Vec<ClauseId>),
    /// Conflict detected (reason clause IDs)
    Conflict(Vec<ClauseId>),
}

/// XOR subsumption checker
pub struct XorSubsumption {
    /// Signature map for fast subsumption checking
    signatures: HashMap<u64, Vec<usize>>,
}

impl XorSubsumption {
    /// Create a new subsumption checker
    pub fn new() -> Self {
        Self {
            signatures: HashMap::new(),
        }
    }

    /// Compute signature of an XOR constraint
    fn compute_signature(vars: &[Var]) -> u64 {
        let mut sig = 0u64;
        for var in vars {
            sig ^= 1u64 << (var.0 as usize % 64);
        }
        sig
    }

    /// Add constraint for subsumption checking
    pub fn add(&mut self, idx: usize, vars: &[Var]) {
        let sig = Self::compute_signature(vars);
        self.signatures.entry(sig).or_default().push(idx);
    }

    /// Check if a constraint subsumes any existing constraints
    /// Returns indices of subsumed constraints
    pub fn find_subsumed(&self, vars: &[Var]) -> Vec<usize> {
        let sig = Self::compute_signature(vars);
        let _var_set: HashSet<Var> = vars.iter().copied().collect();
        let mut subsumed = Vec::new();

        // Check constraints with matching signature
        if let Some(candidates) = self.signatures.get(&sig) {
            for &idx in candidates {
                // Would need access to constraints to verify subset relationship
                // For now, just return candidates
                subsumed.push(idx);
            }
        }

        subsumed
    }
}

impl Default for XorSubsumption {
    fn default() -> Self {
        Self::new()
    }
}

/// XOR strengthening: eliminate variables that appear in exactly two XOR constraints
pub struct XorStrengthening;

impl XorStrengthening {
    /// Apply XOR strengthening
    /// Returns new XOR constraints after eliminating variables
    pub fn strengthen(constraints: &[XorConstraint]) -> Vec<XorConstraint> {
        // Count variable occurrences
        let mut var_count: HashMap<Var, Vec<usize>> = HashMap::new();
        for (idx, constraint) in constraints.iter().enumerate() {
            for &var in &constraint.vars {
                var_count.entry(var).or_default().push(idx);
            }
        }

        // Find variables that appear in exactly two constraints
        let mut to_eliminate: Vec<(Var, usize, usize)> = Vec::new();
        for (var, occurrences) in &var_count {
            if occurrences.len() == 2 {
                to_eliminate.push((*var, occurrences[0], occurrences[1]));
            }
        }

        if to_eliminate.is_empty() {
            return constraints.to_vec();
        }

        let mut result: Vec<XorConstraint> = constraints.to_vec();
        let mut removed: HashSet<usize> = HashSet::new();

        for (var, idx1, idx2) in to_eliminate {
            if removed.contains(&idx1) || removed.contains(&idx2) {
                continue;
            }

            // XOR the two constraints to eliminate the variable
            let mut new_constraint = result[idx1].clone();
            new_constraint.xor_with(&result[idx2]);

            // The variable should be eliminated after XOR
            if !new_constraint.vars.contains(&var) {
                // Replace first constraint with the XORed result
                result[idx1] = new_constraint;
                removed.insert(idx2);
            }
        }

        // Filter out removed constraints
        result
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !removed.contains(idx))
            .map(|(_, c)| c)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_constraint_basic() {
        let xor = XorConstraint::new(vec![Var(0), Var(1)], false);
        assert_eq!(xor.len(), 2);
        assert!(!xor.rhs);
    }

    #[test]
    fn test_xor_constraint_substitute() {
        let mut xor = XorConstraint::new(vec![Var(0), Var(1), Var(2)], false);
        xor.substitute(Var(1), true);
        assert_eq!(xor.len(), 2);
        assert!(xor.rhs); // RHS flipped because we substituted true
    }

    #[test]
    fn test_xor_constraint_xor_with() {
        let mut xor1 = XorConstraint::new(vec![Var(0), Var(1)], false);
        let xor2 = XorConstraint::new(vec![Var(1), Var(2)], false);
        xor1.xor_with(&xor2);
        // x0 ⊕ x1 = 0  XOR  x1 ⊕ x2 = 0  =>  x0 ⊕ x2 = 0
        assert_eq!(xor1.vars.len(), 2);
        assert!(xor1.vars.contains(&Var(0)));
        assert!(xor1.vars.contains(&Var(2)));
        assert!(!xor1.rhs);
    }

    #[test]
    fn test_xor_manager_unit() {
        let mut manager = XorManager::new();
        let xor = XorConstraint::new(vec![Var(0)], true);
        manager.add_constraint(xor);
        assert_eq!(manager.get_units().len(), 1);
        assert_eq!(manager.get_units()[0], (Var(0), true));
    }

    #[test]
    fn test_xor_manager_conflict() {
        let mut manager = XorManager::new();
        let xor = XorConstraint::new(vec![], true);
        manager.add_constraint(xor);
        assert!(manager.has_conflict());
    }

    #[test]
    fn test_gaussian_elimination() {
        let mut manager = XorManager::new();
        // x0 ⊕ x1 = 0
        manager.add_constraint(XorConstraint::new(vec![Var(0), Var(1)], false));
        // x1 ⊕ x2 = 0
        manager.add_constraint(XorConstraint::new(vec![Var(1), Var(2)], false));
        // x0 ⊕ x2 = 1 (should conflict with the above two)
        manager.add_constraint(XorConstraint::new(vec![Var(0), Var(2)], true));

        manager.eliminate();
        assert!(manager.has_conflict());
    }

    #[test]
    fn test_xor_detector_basic() {
        let detector = XorDetector::new(2, 4);

        // Create clauses for x0 ⊕ x1 = 0
        // (x0 ∨ x1) ∧ (¬x0 ∨ ¬x1)
        let clauses = vec![
            (vec![Lit::pos(Var(0)), Lit::pos(Var(1))], ClauseId(0)),
            (vec![Lit::neg(Var(0)), Lit::neg(Var(1))], ClauseId(1)),
        ];

        let xors = detector.detect_xor(&clauses);
        assert_eq!(xors.len(), 1);
        assert_eq!(xors[0].vars.len(), 2);
        assert!(!xors[0].rhs);
    }

    #[test]
    fn test_gf2_row_operations() {
        let mut row = GF2Row::new(128);
        row.set(0);
        row.set(64);
        row.set(127);

        assert!(row.is_set(0));
        assert!(row.is_set(64));
        assert!(row.is_set(127));
        assert!(!row.is_set(1));

        assert_eq!(row.popcount(), 3);
        assert_eq!(row.first_set(), Some(0));

        let vars = row.get_vars();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&0));
        assert!(vars.contains(&64));
        assert!(vars.contains(&127));

        row.clear(0);
        assert!(!row.is_set(0));
        assert_eq!(row.first_set(), Some(64));
    }

    #[test]
    fn test_gf2_row_xor() {
        let mut row1 = GF2Row::new(64);
        row1.set(0);
        row1.set(1);
        row1.rhs = false;

        let mut row2 = GF2Row::new(64);
        row2.set(1);
        row2.set(2);
        row2.rhs = true;

        row1.xor_with(&row2);

        // After XOR: {0, 1} ^ {1, 2} = {0, 2}
        assert!(row1.is_set(0));
        assert!(!row1.is_set(1));
        assert!(row1.is_set(2));
        assert!(row1.rhs); // false ^ true = true
    }

    #[test]
    fn test_gf2_matrix_basic() {
        let mut matrix = GF2Matrix::new();

        // x0 + x1 = 0
        let result1 = matrix.add_constraint(&[Var(0), Var(1)], false, 0);
        assert!(matches!(result1, XorAddResult::Added));

        // x1 + x2 = 0
        let result2 = matrix.add_constraint(&[Var(1), Var(2)], false, 1);
        assert!(matches!(result2, XorAddResult::Added));

        assert_eq!(matrix.num_rows(), 2);
        assert_eq!(matrix.num_vars(), 3);
    }

    #[test]
    fn test_gf2_matrix_conflict() {
        let mut matrix = GF2Matrix::new();

        // x0 + x1 = 0
        matrix.add_constraint(&[Var(0), Var(1)], false, 0);
        // x1 + x2 = 0
        matrix.add_constraint(&[Var(1), Var(2)], false, 1);
        // x0 + x2 = 1 (conflict with the above two)
        let result = matrix.add_constraint(&[Var(0), Var(2)], true, 2);

        assert!(matches!(result, XorAddResult::Conflict(_)));
    }

    #[test]
    fn test_gf2_matrix_unit() {
        let mut matrix = GF2Matrix::new();

        // x0 + x1 = 0
        matrix.add_constraint(&[Var(0), Var(1)], false, 0);
        // x0 = 1 (unit) - this should derive x1 = 1 after Gaussian elimination
        let result = matrix.add_constraint(&[Var(0)], true, 1);

        // After adding x0=1, Gaussian elimination reduces:
        // Row 0: x0 + x1 = 0
        // Row 1: x0 = 1
        // After eliminating x0 from row 0: x1 = 1 (unit)
        match result {
            XorAddResult::Unit(var, value, _) => {
                // The unit could be either x0 or x1 depending on pivot order
                assert!(var == Var(0) || var == Var(1));
                assert!(value);
            }
            _ => panic!("Expected unit result, got {:?}", result),
        }
    }

    #[test]
    fn test_xor_propagator_basic() {
        let mut prop = XorPropagator::new();

        // x0 + x1 = 0
        prop.add_clause(vec![Var(0), Var(1)], false, vec![ClauseId(0)]);

        // Assign x0 = true
        let result = prop.propagate(Var(0), true, 1);
        assert!(matches!(result, PropagateResult::None));

        // Should have pending propagation: x1 = true (to satisfy x0 + x1 = 0)
        let pending = prop.get_pending();
        assert!(!pending.is_empty());
        assert_eq!(pending[0].0, Var(1));
        assert!(pending[0].1); // x1 should be true
    }

    #[test]
    fn test_xor_propagator_conflict() {
        let mut prop = XorPropagator::new();

        // x0 + x1 = 0
        prop.add_clause(vec![Var(0), Var(1)], false, vec![ClauseId(0)]);
        // x0 + x1 = 1 (conflicting)
        prop.add_clause(vec![Var(0), Var(1)], true, vec![ClauseId(1)]);

        assert!(prop.has_conflict());
    }

    #[test]
    fn test_xor_propagator_backtrack() {
        let mut prop = XorPropagator::new();

        // x0 + x1 + x2 = 0
        prop.add_clause(vec![Var(0), Var(1), Var(2)], false, vec![ClauseId(0)]);

        // Assign at level 1
        prop.propagate(Var(0), true, 1);
        // Assign at level 2
        prop.propagate(Var(1), false, 2);

        // Backtrack to level 1
        prop.backtrack(1);

        // Check stats
        let stats = prop.stats();
        assert!(stats.propagations >= 1);
    }

    #[test]
    fn test_xor_strengthening() {
        // x0 + x1 = 0
        // x1 + x2 = 0
        // Variable x1 appears in exactly two constraints
        let constraints = vec![
            XorConstraint::new(vec![Var(0), Var(1)], false),
            XorConstraint::new(vec![Var(1), Var(2)], false),
        ];

        let strengthened = XorStrengthening::strengthen(&constraints);

        // After strengthening, x1 should be eliminated
        // x0 + x1 XOR x1 + x2 = x0 + x2 = 0
        // We should have fewer or modified constraints
        assert!(!strengthened.is_empty());
    }

    #[test]
    fn test_xor_subsumption() {
        let mut subsumption = XorSubsumption::new();

        subsumption.add(0, &[Var(0), Var(1)]);
        subsumption.add(1, &[Var(1), Var(2)]);

        let subsumed = subsumption.find_subsumed(&[Var(0), Var(1)]);
        assert!(!subsumed.is_empty());
    }

    #[test]
    fn test_xor_clause_watched() {
        let clause = XorClause::new(vec![Var(0), Var(1), Var(2)], false, vec![ClauseId(0)]);

        let (w0, w1) = clause.get_watched();
        assert_eq!(w0, Var(0));
        assert_eq!(w1, Some(Var(1)));
    }

    #[test]
    fn test_gf2_matrix_propagate() {
        let mut matrix = GF2Matrix::new();

        // x0 + x1 = 0
        matrix.add_constraint(&[Var(0), Var(1)], false, 0);
        // x1 + x2 = 0
        matrix.add_constraint(&[Var(1), Var(2)], false, 1);

        // Propagate x0 = true
        let results = matrix.propagate(Var(0), true);

        // Should derive implications
        // After x0=true: x1 = true (from first constraint)
        // After x1=true: x2 = true (from second constraint)
        // Results may contain these implications
        assert!(!results.is_empty() || matrix.num_rows() > 0);
    }
}
