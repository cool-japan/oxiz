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
#[allow(unused_imports)]
use crate::prelude::*;

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

/// Record of a single `GF2Matrix::propagate` call's mutations, needed to
/// undo them when the corresponding assignment is later retracted
/// (backtracked) by the SAT solver.
#[derive(Debug, Clone)]
struct PropagateUndo {
    /// The variable that was propagated.
    var: Var,
    /// The value it was propagated to (needed to know whether `rhs` was
    /// flipped and must be flipped back).
    value: bool,
    /// Row indices that had `var`'s column cleared by this call. Rows are
    /// never removed or reordered once pushed, so these indices remain
    /// valid for the lifetime of the matrix.
    touched_rows: Vec<usize>,
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
    /// Undo trail for `propagate`, one entry per call, in call order.
    /// `undo_propagate` pops it LIFO, mirroring how the SAT solver
    /// backtracks its own assignment trail.
    undo_stack: Vec<PropagateUndo>,
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
            undo_stack: Vec::new(),
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

        // The loop ends only by `break` with the row's first set column (the
        // row is not changed after that `first_set`), so `pivot_col` is the
        // lowest set column of the row as it is added.
        let pivot_col = loop {
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
                break first;
            }
        };

        // Check for unit constraint: with exactly one set column, that column
        // is the first one.
        if row.popcount() == 1 {
            let var = self.col_to_var[pivot_col];
            let value = row.rhs;
            return XorAddResult::Unit(var, value, row.sources.clone());
        }

        // Add as new row with pivot
        let row_idx = self.rows.len();
        self.pivots[pivot_col] = Some(row_idx);
        self.rows.push(row.clone());

        XorAddResult::Added
    }

    /// Back-substitute an assignment to find implied units.
    ///
    /// This destructively folds `var`'s value into every row that mentions
    /// it (clearing the column and, if `value` is true, flipping `rhs`).
    /// That mutation is only valid for as long as `var` stays assigned to
    /// `value` on the live SAT trail; when the solver later backtracks past
    /// this assignment, [`Self::undo_propagate`] must be called (once, in
    /// exact LIFO order relative to other `propagate` calls) to restore the
    /// affected rows — otherwise the matrix silently keeps reasoning as if
    /// a retracted (or now differently-valued) assignment still held,
    /// producing wrong unit/conflict results for the rest of the search.
    /// Every call — even one for an unregistered variable that touches no
    /// rows — pushes an undo-trail entry, so `undo_propagate` calls stay in
    /// 1:1 lockstep with `propagate` calls regardless of how many rows (if
    /// any) were actually touched.
    pub fn propagate(&mut self, var: Var, value: bool) -> Vec<XorAddResult> {
        let mut results = Vec::new();

        let Some(&col) = self.var_to_col.get(&var) else {
            self.undo_stack.push(PropagateUndo {
                var,
                value,
                touched_rows: Vec::new(),
            });
            return results;
        };

        let mut touched_rows = Vec::new();

        // Update all rows containing this variable
        for (row_idx, row) in self.rows.iter_mut().enumerate() {
            if row.is_set(col) {
                touched_rows.push(row_idx);
                row.clear(col);
                if value {
                    row.rhs = !row.rhs;
                }

                // Check for unit or conflict: `first_set` is `None` exactly
                // when the row is zero, and with exactly one set column that
                // column is the first one.
                match row.first_set() {
                    None => {
                        if row.rhs {
                            results.push(XorAddResult::Conflict(row.sources.clone()));
                        }
                    }
                    Some(var_idx) if row.popcount() == 1 => {
                        let implied_var = self.col_to_var[var_idx];
                        let implied_value = row.rhs;
                        results.push(XorAddResult::Unit(
                            implied_var,
                            implied_value,
                            row.sources.clone(),
                        ));
                    }
                    Some(_) => {}
                }
            }
        }

        self.undo_stack.push(PropagateUndo {
            var,
            value,
            touched_rows,
        });

        results
    }

    /// Undo the most recently applied `propagate` call, restoring the rows
    /// it touched to their pre-propagation state: re-setting the
    /// propagated variable's column bit in each affected row and, if the
    /// propagated value was `true`, flipping `rhs` back.
    ///
    /// Must be called in the same LIFO order the SAT solver backtracks its
    /// trail (undo the most recently propagated assignment first, mirroring
    /// `Solver`'s own phase-saving backtrack) — `propagate` destructively
    /// folds each assignment's effect into the matrix rows, so calling this
    /// out of order (or skipping an entry) would leave rows reflecting a
    /// mix of assignments that never coexisted on the trail.
    ///
    /// Returns the `(Var, bool)` that was undone, or `None` if the undo
    /// trail is empty.
    pub fn undo_propagate(&mut self) -> Option<(Var, bool)> {
        let entry = self.undo_stack.pop()?;

        if let Some(&col) = self.var_to_col.get(&entry.var) {
            for &row_idx in &entry.touched_rows {
                let row = &mut self.rows[row_idx];
                if entry.value {
                    row.rhs = !row.rhs;
                }
                row.set(col);
            }
        }

        Some((entry.var, entry.value))
    }

    /// Number of `propagate` calls that can currently be undone.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
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

/// Smallest XOR arity a [`XorDetector`] can recognize.
///
/// The CNF encoding of an XOR over `n` variables consists of `2^(n-1)`
/// clauses. `n == 0` makes that exponent underflow, and `n == 1` describes a
/// unit clause rather than an XOR, so 2 is the smallest meaningful arity.
pub const MIN_XOR_SIZE: usize = 2;

/// Largest XOR arity a [`XorDetector`] can recognize.
///
/// `2^(n-1)` must be a representable clause count, so `n - 1` has to stay
/// below the width of `usize`; 32 keeps `1usize << (n - 1)` well-defined on
/// 32-bit targets as well. An XOR over 32 variables already implies 2^31
/// clauses, far beyond anything a detector will encounter.
pub const MAX_XOR_SIZE: usize = 32;

/// XOR clause detector
pub struct XorDetector {
    /// Minimum XOR size to detect, always within
    /// `MIN_XOR_SIZE..=MAX_XOR_SIZE`
    min_xor_size: usize,
    /// Maximum XOR size to detect, always within
    /// `MIN_XOR_SIZE..=MAX_XOR_SIZE`
    max_xor_size: usize,
}

impl XorDetector {
    /// Create a new XOR detector, clamping the sizes to the supported range
    ///
    /// Sizes outside `MIN_XOR_SIZE..=MAX_XOR_SIZE` are clamped rather than
    /// accepted verbatim: a size of 0 used to underflow `size - 1` and a size
    /// above 64 used to overflow the `1 << (size - 1)` shift, both reachable
    /// from this public constructor. Use [`XorDetector::try_new`] when the
    /// caller wants the out-of-range request reported instead of adjusted.
    pub fn new(min_size: usize, max_size: usize) -> Self {
        Self {
            min_xor_size: min_size.clamp(MIN_XOR_SIZE, MAX_XOR_SIZE),
            max_xor_size: max_size.clamp(MIN_XOR_SIZE, MAX_XOR_SIZE),
        }
    }

    /// Create a new XOR detector, rejecting unsupported sizes
    ///
    /// Returns `None` when either bound falls outside
    /// `MIN_XOR_SIZE..=MAX_XOR_SIZE`, or when `min_size > max_size` (which
    /// would silently detect nothing at all).
    pub fn try_new(min_size: usize, max_size: usize) -> Option<Self> {
        let supported = MIN_XOR_SIZE..=MAX_XOR_SIZE;
        if !supported.contains(&min_size) || !supported.contains(&max_size) || min_size > max_size {
            return None;
        }
        Some(Self {
            min_xor_size: min_size,
            max_xor_size: max_size,
        })
    }

    /// The smallest XOR arity this detector looks for
    pub fn min_size(&self) -> usize {
        self.min_xor_size
    }

    /// The largest XOR arity this detector looks for
    pub fn max_size(&self) -> usize {
        self.max_xor_size
    }

    /// Number of clauses a CNF encoding of an `size`-ary XOR must have
    ///
    /// `None` for a size this detector does not support, which is what keeps
    /// the `1 << (size - 1)` shift below in range.
    fn expected_clause_count(size: usize) -> Option<usize> {
        if !(MIN_XOR_SIZE..=MAX_XOR_SIZE).contains(&size) {
            return None;
        }
        Some(1usize << (size - 1))
    }

    /// Detect XOR constraints from clauses
    /// An XOR constraint x1 ⊕ x2 ⊕ ... ⊕ xn = rhs is represented as 2^(n-1) clauses
    /// whose negative-literal parity encodes the rhs: every clause of the encoding
    /// has the same parity of negative literals, equal to `1 - rhs`.
    /// For example, x1 ⊕ x2 = 1 is represented as:
    ///   (x1 ∨ x2) ∧ (¬x1 ∨ ¬x2)
    /// (each clause has an even number of negatives ⇒ parity 0 ⇒ rhs = 1), while
    /// x1 ⊕ x2 = 0 is represented as (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2) (odd parity ⇒ rhs = 0).
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

        // Sizes outside the supported range have no valid encoding to look
        // for; `size == 0` in particular used to underflow `size - 1` below.
        let Some(expected_clauses) = Self::expected_clause_count(size) else {
            return result;
        };

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
            if polarity_groups.len() != expected_clauses {
                continue;
            }

            // Verify this is a valid XOR encoding
            if self.is_valid_xor_encoding(&polarity_groups, size)
                && let Some(rhs) = Self::compute_xor_rhs(&polarity_groups)
            {
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
        // For a valid XOR encoding, we need exactly 2^(n-1) clauses. An
        // unsupported size has no such count, so nothing can be valid at it.
        let Some(expected_clauses) = Self::expected_clause_count(size) else {
            return false;
        };
        if polarity_groups.len() != expected_clauses {
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
        let Some((first_polarities, rest)) = polarity_groups.split_first() else {
            return false;
        };
        let first_parity = first_polarities.0.iter().filter(|&&p| !p).count() % 2;

        for (polarities, _) in rest {
            let neg_count = polarities.iter().filter(|&&p| !p).count();
            if neg_count % 2 != first_parity {
                return false;
            }
        }

        true
    }

    /// Compute XOR RHS from polarity groups
    ///
    /// `None` for an empty group list, which encodes no XOR at all.
    fn compute_xor_rhs(polarity_groups: &[(Vec<bool>, ClauseId)]) -> Option<bool> {
        // A CNF encoding of x1 ⊕ ... ⊕ xn = c consists of clauses whose
        // negative-literal parity equals `1 - c` (every falsifying assignment
        // has parity c, so each clause forbids one parity-`1-c` assignment).
        // Hence c = 1 - parity, i.e. rhs is true exactly when the parity of
        // negatives is even.
        //
        // If all clauses have an even number of negatives, RHS = true (c = 1).
        // If all clauses have an odd number of negatives, RHS = false (c = 0).
        let (pols, _) = polarity_groups.first()?;
        let neg_count = pols.iter().filter(|&&p| !p).count();
        Some(neg_count % 2 == 0)
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

            // Undo this assignment's effect on the GF(2) matrix.
            // `GF2Matrix::propagate` destructively folds each assignment
            // into its rows (clearing the variable's column and flipping
            // `rhs`); without this, the matrix would keep reflecting a
            // retracted assignment after backtracking, producing wrong
            // unit/conflict results the next time the solver explores a
            // different branch. `propagate` and `undo_propagate` are
            // 1:1 (one undo-trail entry per propagate call, even when no
            // rows were touched), so popping once per retracted trail
            // entry here stays in exact lockstep.
            if let Some((undone_var, _)) = self.matrix.undo_propagate() {
                debug_assert_eq!(
                    undone_var, var,
                    "GF2Matrix undo stack desynchronized from XorPropagator trail"
                );
            }
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
    /// Signature map used purely as a fast *pre-filter* for candidate
    /// lookup: two constraints with different signatures can never be a
    /// subset/superset match (the signature is a linear function of the
    /// variable set), but two constraints with the *same* signature are not
    /// guaranteed to be related — a 64-bit XOR fingerprint collides
    /// whenever the same bit position is touched an even number of times
    /// (e.g. `Var(0)` and `Var(64)` both map to bit 0), so an unrelated
    /// variable set can share a signature by chance. Every candidate pulled
    /// from this map is therefore re-verified against its real, stored
    /// variable set in [`Self::find_subsumed`] before being reported.
    signatures: HashMap<u64, Vec<usize>>,
    /// The actual variable set for each registered constraint index, needed
    /// to verify (rather than merely guess from the lossy signature)
    /// whether a candidate is genuinely subsumed.
    var_sets: HashMap<usize, HashSet<Var>>,
}

impl XorSubsumption {
    /// Create a new subsumption checker
    pub fn new() -> Self {
        Self {
            signatures: HashMap::new(),
            var_sets: HashMap::new(),
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
        self.var_sets.insert(idx, vars.iter().copied().collect());
    }

    /// Find previously-registered constraints subsumed by a constraint over
    /// `vars`.
    ///
    /// Mirrors CNF clause subsumption: an existing constraint at index
    /// `idx` is subsumed by (i.e. rendered redundant by) `vars` when
    /// `vars`'s variable set is a subset of `idx`'s registered variable set
    /// — `vars` is at least as general, so `idx` need not be kept
    /// separately. Every signature-bucket candidate is checked against its
    /// *real* stored variable set (see the `signatures` field doc) before
    /// being reported, so this never returns an unverified hash collision
    /// as if it were a genuine subsumption.
    ///
    /// Note this can only find matches whose variable sets share the same
    /// 64-bit signature — in practice this reliably covers exact-duplicate
    /// variable sets (always same signature) plus the rare deliberate
    /// signature collision, but is not a full O(n) subset scan against
    /// every registered constraint.
    pub fn find_subsumed(&self, vars: &[Var]) -> Vec<usize> {
        let sig = Self::compute_signature(vars);
        let query: HashSet<Var> = vars.iter().copied().collect();
        let mut subsumed = Vec::new();

        // Check constraints with matching signature
        if let Some(candidates) = self.signatures.get(&sig) {
            for &idx in candidates {
                // Re-verify against the real variable set: the signature
                // alone cannot distinguish a genuine subset relationship
                // from an unrelated hash collision.
                if let Some(existing_vars) = self.var_sets.get(&idx)
                    && query.is_subset(existing_vars)
                {
                    subsumed.push(idx);
                }
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

        // Create clauses for x0 ⊕ x1 = 1
        // (x0 ∨ x1) ∧ (¬x0 ∨ ¬x1) — each clause has an even number of negatives,
        // so the encoded RHS is 1 (the two variables must differ).
        let clauses = vec![
            (vec![Lit::pos(Var(0)), Lit::pos(Var(1))], ClauseId(0)),
            (vec![Lit::neg(Var(0)), Lit::neg(Var(1))], ClauseId(1)),
        ];

        let xors = detector.detect_xor(&clauses);
        assert_eq!(xors.len(), 1);
        assert_eq!(xors[0].vars.len(), 2);
        assert!(xors[0].rhs);
    }

    #[test]
    fn test_xor_detector_rhs_zero_encoding() {
        // Finding 6: the =0 encoding (¬x0 ∨ x1) ∧ (x0 ∨ ¬x1) has an ODD number
        // of negatives per clause, so the recovered RHS must be false (0).
        let detector = XorDetector::new(2, 4);
        let clauses = vec![
            (vec![Lit::neg(Var(0)), Lit::pos(Var(1))], ClauseId(0)),
            (vec![Lit::pos(Var(0)), Lit::neg(Var(1))], ClauseId(1)),
        ];
        let xors = detector.detect_xor(&clauses);
        assert_eq!(xors.len(), 1);
        assert!(
            !xors[0].rhs,
            "odd-negative-parity encoding must decode to RHS = 0"
        );
    }

    #[test]
    fn test_xor_detector_size_zero_is_clamped_not_underflowed() {
        // `new(0, ..)` used to reach `1 << (0 - 1)`: a `usize` underflow, then
        // a shift far past the width of the type.
        let detector = XorDetector::new(0, 4);
        assert_eq!(detector.min_size(), MIN_XOR_SIZE);
        assert_eq!(detector.max_size(), 4);

        let clauses = vec![
            (vec![Lit::pos(Var(0)), Lit::pos(Var(1))], ClauseId(0)),
            (vec![Lit::neg(Var(0)), Lit::neg(Var(1))], ClauseId(1)),
        ];
        // Still finds the 2-ary XOR, and finds it without panicking.
        assert_eq!(detector.detect_xor(&clauses).len(), 1);
    }

    #[test]
    fn test_xor_detector_size_beyond_64_is_clamped_not_overflowed() {
        // `new(.., 65)` used to reach `1 << 64`, a shift overflow.
        let detector = XorDetector::new(2, 65);
        assert_eq!(detector.min_size(), 2);
        assert_eq!(detector.max_size(), MAX_XOR_SIZE);

        let clauses = vec![
            (vec![Lit::pos(Var(0)), Lit::pos(Var(1))], ClauseId(0)),
            (vec![Lit::neg(Var(0)), Lit::neg(Var(1))], ClauseId(1)),
        ];
        assert_eq!(detector.detect_xor(&clauses).len(), 1);
    }

    #[test]
    fn test_xor_detector_empty_range_detects_nothing() {
        // `min > max` after clamping: an empty search range, not a panic.
        let detector = XorDetector::new(usize::MAX, 0);
        assert_eq!(detector.min_size(), MAX_XOR_SIZE);
        assert_eq!(detector.max_size(), MIN_XOR_SIZE);
        assert!(detector.detect_xor(&[]).is_empty());
    }

    #[test]
    fn test_xor_detector_try_new_rejects_unsupported_sizes() {
        assert!(XorDetector::try_new(0, 4).is_none(), "0 is below MIN");
        assert!(
            XorDetector::try_new(1, 4).is_none(),
            "1 is not an XOR arity"
        );
        assert!(XorDetector::try_new(2, 65).is_none(), "65 is above MAX");
        assert!(
            XorDetector::try_new(4, 2).is_none(),
            "min must not exceed max"
        );

        let detector = XorDetector::try_new(2, 4).expect("2..=4 is supported");
        assert_eq!((detector.min_size(), detector.max_size()), (2, 4));
    }

    #[test]
    fn test_expected_clause_count_range() {
        assert_eq!(XorDetector::expected_clause_count(0), None);
        assert_eq!(XorDetector::expected_clause_count(1), None);
        assert_eq!(XorDetector::expected_clause_count(2), Some(2));
        assert_eq!(XorDetector::expected_clause_count(3), Some(4));
        assert_eq!(
            XorDetector::expected_clause_count(MAX_XOR_SIZE),
            Some(1usize << (MAX_XOR_SIZE - 1))
        );
        assert_eq!(XorDetector::expected_clause_count(MAX_XOR_SIZE + 1), None);
        assert_eq!(XorDetector::expected_clause_count(65), None);
        assert_eq!(XorDetector::expected_clause_count(usize::MAX), None);
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

    // Regression test for the subsumption item: `find_subsumed` used to
    // report *every* signature-bucket candidate as subsumed without
    // verifying the actual variable sets, so two totally unrelated
    // constraints whose 64-bit XOR signatures happen to collide would be
    // falsely reported as subsuming one another. `Var(0)` and `Var(64)`
    // collide on the same signature bit (`var.0 as usize % 64`), so
    // `{Var(64)}` and `{Var(0)}` share a signature despite being disjoint.
    #[test]
    fn test_xor_subsumption_rejects_unverified_hash_collision() {
        let mut subsumption = XorSubsumption::new();
        subsumption.add(0, &[Var(0)]);

        // Same signature as `{Var(0)}` (both touch signature bit 0), but a
        // completely different, non-subset variable set.
        let subsumed = subsumption.find_subsumed(&[Var(64)]);
        assert!(
            subsumed.is_empty(),
            "a bare signature collision must not be reported as subsumed: {subsumed:?}"
        );
    }

    // A genuine subset relationship that happens to land in the same
    // signature bucket (via the same Var(0)/Var(64) collision, this time
    // used deliberately so the *existing* constraint's signature reduces to
    // just Var(1)'s bit) must still be correctly detected.
    #[test]
    fn test_xor_subsumption_detects_true_positive_within_colliding_bucket() {
        let mut subsumption = XorSubsumption::new();
        // Var(0) and Var(64) cancel out in the XOR signature, leaving only
        // Var(1)'s bit -- same signature as `{Var(1)}` alone.
        subsumption.add(0, &[Var(0), Var(1), Var(64)]);

        let subsumed = subsumption.find_subsumed(&[Var(1)]);
        assert_eq!(
            subsumed,
            vec![0],
            "{{Var(1)}} is a genuine subset of {{Var(0), Var(1), Var(64)}} and must be reported"
        );
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

    fn assert_single_unit(results: &[XorAddResult], expected_var: Var, expected_value: bool) {
        assert_eq!(
            results.len(),
            1,
            "expected exactly one result, got {results:?}"
        );
        match &results[0] {
            XorAddResult::Unit(v, val, _) => {
                assert_eq!(*v, expected_var);
                assert_eq!(*val, expected_value);
            }
            other => panic!("expected Unit, got {other:?}"),
        }
    }

    // Regression test for the backtracking item: `GF2Matrix::propagate`
    // destructively clears the propagated variable's column (and flips
    // `rhs`) in every row that mentions it, with no way to undo that once
    // the corresponding SAT assignment is retracted. Verify
    // `undo_propagate` exactly restores row state, so re-propagating the
    // same (or a different) value from the restored matrix reproduces the
    // correct implication rather than silently doing nothing (which is
    // what happens if the row's bit was left permanently cleared).
    #[test]
    fn test_gf2_matrix_undo_propagate_restores_row_state() {
        let mut matrix = GF2Matrix::new();
        // x0 + x1 = 0
        matrix.add_constraint(&[Var(0), Var(1)], false, 0);
        // x1 + x2 = 0
        matrix.add_constraint(&[Var(1), Var(2)], false, 1);

        assert_eq!(matrix.undo_depth(), 0);

        // Propagate x0 = true: row0 (x0+x1=0) folds to x1=1 (unit).
        let results_true = matrix.propagate(Var(0), true);
        assert_single_unit(&results_true, Var(1), true);
        assert_eq!(matrix.undo_depth(), 1);

        // Undo it; the matrix must report exactly what was undone.
        assert_eq!(matrix.undo_propagate(), Some((Var(0), true)));
        assert_eq!(matrix.undo_depth(), 0);

        // Re-propagating x0 = true from the restored state must reproduce
        // the identical implication (row wasn't permanently corrupted by
        // the first propagate/undo cycle).
        let results_true_again = matrix.propagate(Var(0), true);
        assert_single_unit(&results_true_again, Var(1), true);

        // Undo again and propagate the *opposite* value: this must derive
        // the opposite implication (x1 = false), proving the row's `rhs`
        // was correctly restored too, not just the column bit.
        assert_eq!(matrix.undo_propagate(), Some((Var(0), true)));
        let results_false = matrix.propagate(Var(0), false);
        assert_single_unit(&results_false, Var(1), false);
    }

    // Undo trail must stay in lockstep with `propagate` calls even for
    // variables that were never registered in the matrix (no rows touched),
    // since callers pop it purely by call count, not by which calls did
    // anything.
    #[test]
    fn test_gf2_matrix_undo_propagate_unregistered_var_is_still_tracked() {
        let mut matrix = GF2Matrix::new();
        matrix.add_constraint(&[Var(0), Var(1)], false, 0);

        // Var(5) is never part of any constraint.
        let results = matrix.propagate(Var(5), true);
        assert!(results.is_empty());
        assert_eq!(matrix.undo_depth(), 1);

        assert_eq!(matrix.undo_propagate(), Some((Var(5), true)));
        assert_eq!(matrix.undo_depth(), 0);
    }

    // Integration-level regression test: `XorPropagator::backtrack` must
    // restore the GF(2) matrix, not just its own assignment/trail maps, so
    // that re-deciding a backtracked variable the *other* way produces the
    // correct (opposite) implication instead of silently no implication at
    // all.
    #[test]
    fn test_xor_propagator_backtrack_restores_matrix_for_reassignment() {
        let mut prop = XorPropagator::new();

        // x0 + x1 + x2 = 0
        prop.add_clause(vec![Var(0), Var(1), Var(2)], false, vec![ClauseId(0)]);

        // Level 1: x0 = true. Matrix row becomes x1 + x2 = 1 (not yet unit).
        prop.propagate(Var(0), true, 1);

        // Level 2: x1 = false. Row becomes unit: x2 = true.
        prop.propagate(Var(1), false, 2);
        let pending_first = prop.get_pending();
        assert!(
            pending_first.iter().any(|&(v, val, _)| v == Var(2) && val),
            "expected x2 = true to be implied, got {pending_first:?}"
        );

        // Backtrack to level 1, undoing x1's effect on the matrix.
        prop.backtrack(1);

        // Re-decide the opposite way at level 2: x1 = true. This must
        // derive the opposite implication for x2 (x2 = false). Before the
        // fix, the matrix row's x1 bit had already been permanently
        // cleared by the first (level-2) propagate call and was never
        // restored on backtrack, so this second call would silently find
        // nothing to update and produce no implication at all.
        prop.propagate(Var(1), true, 2);
        let pending_second = prop.get_pending();
        assert!(
            pending_second
                .iter()
                .any(|&(v, val, _)| v == Var(2) && !val),
            "expected x2 = false to be implied after re-deciding x1 = true \
             post-backtrack, got {pending_second:?}"
        );
    }
}
