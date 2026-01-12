//! Simplex algorithm for linear arithmetic.
//!
//! This module implements the Simplex algorithm for solving linear programming
//! problems and checking satisfiability of linear real arithmetic constraints.
//!
//! The implementation follows the two-phase simplex method:
//! - Phase 1: Find a basic feasible solution (or prove infeasibility)
//! - Phase 2: Optimize the objective function (or detect unboundedness)
//!
//! Reference: Z3's `math/simplex/` directory and standard LP textbooks.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Variable identifier for simplex.
pub type VarId = u32;

/// Constraint identifier.
pub type ConstraintId = u32;

/// Bound type for a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundType {
    /// Lower bound: x >= value
    Lower,
    /// Upper bound: x <= value
    Upper,
    /// Equality: x = value
    Equal,
}

/// A bound on a variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bound {
    /// The variable this bound applies to.
    pub var: VarId,
    /// The type of bound (lower, upper, or equality).
    pub bound_type: BoundType,
    /// The bound value.
    pub value: BigRational,
}

impl Bound {
    /// Create a lower bound: var >= value.
    pub fn lower(var: VarId, value: BigRational) -> Self {
        Self {
            var,
            bound_type: BoundType::Lower,
            value,
        }
    }

    /// Create an upper bound: var <= value.
    pub fn upper(var: VarId, value: BigRational) -> Self {
        Self {
            var,
            bound_type: BoundType::Upper,
            value,
        }
    }

    /// Create an equality bound: var = value.
    pub fn equal(var: VarId, value: BigRational) -> Self {
        Self {
            var,
            bound_type: BoundType::Equal,
            value,
        }
    }
}

/// Variable classification in the tableau.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarClass {
    /// Basic variable (appears on LHS of a row).
    Basic,
    /// Non-basic variable (appears on RHS).
    NonBasic,
}

/// A row in the simplex tableau represents: basic_var = constant + sum(coeff_i * nonbasic_var_i).
#[derive(Debug, Clone)]
pub struct Row {
    /// The basic variable for this row.
    pub basic_var: VarId,
    /// The constant term.
    pub constant: BigRational,
    /// Coefficients for non-basic variables: var_id -> coefficient.
    pub coeffs: FxHashMap<VarId, BigRational>,
}

impl Row {
    /// Create a new row with a basic variable.
    pub fn new(basic_var: VarId) -> Self {
        Self {
            basic_var,
            constant: BigRational::zero(),
            coeffs: FxHashMap::default(),
        }
    }

    /// Create a row representing: basic_var = constant + sum(coeffs).
    pub fn from_expr(
        basic_var: VarId,
        constant: BigRational,
        coeffs: FxHashMap<VarId, BigRational>,
    ) -> Self {
        let mut row = Self {
            basic_var,
            constant,
            coeffs: FxHashMap::default(),
        };
        for (var, coeff) in coeffs {
            if !coeff.is_zero() {
                row.coeffs.insert(var, coeff);
            }
        }
        row
    }

    /// Get the value of the basic variable given values of non-basic variables.
    pub fn eval(&self, non_basic_values: &FxHashMap<VarId, BigRational>) -> BigRational {
        let mut value = self.constant.clone();
        for (var, coeff) in &self.coeffs {
            if let Some(val) = non_basic_values.get(var) {
                value += coeff * val;
            }
        }
        value
    }

    /// Add a multiple of another row to this row.
    /// self += multiplier * other
    pub fn add_row(&mut self, multiplier: &BigRational, other: &Row) {
        if multiplier.is_zero() {
            return;
        }

        self.constant += multiplier * &other.constant;

        for (var, coeff) in &other.coeffs {
            let new_coeff = self
                .coeffs
                .get(var)
                .cloned()
                .unwrap_or_else(BigRational::zero)
                + multiplier * coeff;
            if new_coeff.is_zero() {
                self.coeffs.remove(var);
            } else {
                self.coeffs.insert(*var, new_coeff);
            }
        }
    }

    /// Multiply the row by a scalar.
    pub fn scale(&mut self, scalar: &BigRational) {
        if scalar.is_zero() {
            self.constant = BigRational::zero();
            self.coeffs.clear();
            return;
        }

        self.constant *= scalar;
        for coeff in self.coeffs.values_mut() {
            *coeff *= scalar;
        }
    }

    /// Substitute a non-basic variable using another row.
    /// If var appears in this row, replace it using: var = row.constant + sum(row.coeffs).
    pub fn substitute(&mut self, var: VarId, row: &Row) {
        if let Some(coeff) = self.coeffs.remove(&var) {
            // Add coeff * row to self
            self.add_row(&coeff, row);
        }
    }

    /// Check if the row is valid (no basic variable in RHS).
    pub fn is_valid(&self, basic_vars: &FxHashSet<VarId>) -> bool {
        for var in self.coeffs.keys() {
            if basic_vars.contains(var) {
                return false;
            }
        }
        true
    }

    /// Normalize the row by dividing by the GCD of coefficients.
    pub fn normalize(&mut self) {
        if self.coeffs.is_empty() {
            return;
        }

        // Compute GCD of all coefficients
        let mut gcd: Option<BigInt> = None;
        for coeff in self.coeffs.values() {
            if !coeff.is_zero() {
                let num = coeff.numer().clone();
                gcd = Some(match gcd {
                    None => num.abs(),
                    Some(g) => gcd_bigint(g, num.abs()),
                });
            }
        }

        if let Some(g) = gcd
            && !g.is_one()
        {
            let divisor = BigRational::from_integer(g);
            self.scale(&(BigRational::one() / divisor));
        }
    }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x{} = {}", self.basic_var, self.constant)?;
        for (var, coeff) in &self.coeffs {
            if coeff.is_positive() {
                write!(f, " + {}*x{}", coeff, var)?;
            } else {
                write!(f, " - {}*x{}", -coeff, var)?;
            }
        }
        Ok(())
    }
}

/// Compute GCD of two BigInts using Euclidean algorithm.
fn gcd_bigint(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let t = &a % &b;
        a = b;
        b = t;
    }
    a.abs()
}

/// Result of a simplex operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimplexResult {
    /// The system is satisfiable.
    Sat,
    /// The system is unsatisfiable.
    Unsat,
    /// The objective is unbounded.
    Unbounded,
    /// Unknown (shouldn't happen in simplex).
    Unknown,
}

/// Explanation for why a constraint caused unsatisfiability.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The constraints involved in the conflict.
    pub constraints: Vec<ConstraintId>,
}

/// The Simplex tableau.
#[derive(Debug, Clone)]
pub struct SimplexTableau {
    /// Rows of the tableau, indexed by basic variable.
    rows: FxHashMap<VarId, Row>,
    /// Set of basic variables.
    basic_vars: FxHashSet<VarId>,
    /// Set of non-basic variables.
    non_basic_vars: FxHashSet<VarId>,
    /// Current assignment to all variables.
    assignment: FxHashMap<VarId, BigRational>,
    /// Lower bounds for variables.
    lower_bounds: FxHashMap<VarId, BigRational>,
    /// Upper bounds for variables.
    upper_bounds: FxHashMap<VarId, BigRational>,
    /// Mapping from variables to the constraints that bound them.
    var_to_constraints: FxHashMap<VarId, Vec<ConstraintId>>,
    /// Next fresh variable ID.
    next_var_id: VarId,
    /// Use Bland's rule to prevent cycling.
    use_blands_rule: bool,
}

impl SimplexTableau {
    /// Create a new empty tableau.
    pub fn new() -> Self {
        Self {
            rows: FxHashMap::default(),
            basic_vars: FxHashSet::default(),
            non_basic_vars: FxHashSet::default(),
            assignment: FxHashMap::default(),
            lower_bounds: FxHashMap::default(),
            upper_bounds: FxHashMap::default(),
            var_to_constraints: FxHashMap::default(),
            next_var_id: 0,
            use_blands_rule: true,
        }
    }

    /// Create a fresh variable.
    pub fn fresh_var(&mut self) -> VarId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.non_basic_vars.insert(id);
        self.assignment.insert(id, BigRational::zero());
        id
    }

    /// Add a variable with initial bounds.
    pub fn add_var(&mut self, lower: Option<BigRational>, upper: Option<BigRational>) -> VarId {
        let var = self.fresh_var();
        if let Some(lb) = lower {
            self.lower_bounds.insert(var, lb.clone());
            self.assignment.insert(var, lb);
        }
        if let Some(ub) = upper {
            self.upper_bounds.insert(var, ub);
        }
        var
    }

    /// Add a row to the tableau.
    /// The row represents: basic_var = constant + sum(coeffs).
    pub fn add_row(&mut self, row: Row) -> Result<(), String> {
        let basic_var = row.basic_var;

        // Ensure basic_var is not already basic
        if self.basic_vars.contains(&basic_var) {
            return Err(format!("Variable x{} is already basic", basic_var));
        }

        // Ensure all non-basic vars in coeffs are indeed non-basic
        for var in row.coeffs.keys() {
            if self.basic_vars.contains(var) {
                return Err(format!("Variable x{} appears in RHS but is basic", var));
            }
            self.non_basic_vars.insert(*var);
        }

        // Move basic_var from non-basic to basic
        self.non_basic_vars.remove(&basic_var);
        self.basic_vars.insert(basic_var);

        // Compute initial value for basic_var
        let value = row.eval(&self.assignment);
        self.assignment.insert(basic_var, value);

        self.rows.insert(basic_var, row);
        Ok(())
    }

    /// Add a bound constraint.
    pub fn add_bound(
        &mut self,
        var: VarId,
        bound_type: BoundType,
        value: BigRational,
        constraint_id: ConstraintId,
    ) -> Result<(), Conflict> {
        self.var_to_constraints
            .entry(var)
            .or_default()
            .push(constraint_id);

        match bound_type {
            BoundType::Lower => {
                let current_lb = self.lower_bounds.get(&var);
                if let Some(lb) = current_lb {
                    if &value > lb {
                        self.lower_bounds.insert(var, value.clone());
                    }
                } else {
                    self.lower_bounds.insert(var, value.clone());
                }

                // Check for immediate conflict
                if let Some(ub) = self.upper_bounds.get(&var)
                    && &value > ub
                {
                    return Err(Conflict {
                        constraints: vec![constraint_id],
                    });
                }
            }
            BoundType::Upper => {
                let current_ub = self.upper_bounds.get(&var);
                if let Some(ub) = current_ub {
                    if &value < ub {
                        self.upper_bounds.insert(var, value.clone());
                    }
                } else {
                    self.upper_bounds.insert(var, value.clone());
                }

                // Check for immediate conflict
                if let Some(lb) = self.lower_bounds.get(&var)
                    && &value < lb
                {
                    return Err(Conflict {
                        constraints: vec![constraint_id],
                    });
                }
            }
            BoundType::Equal => {
                self.lower_bounds.insert(var, value.clone());
                self.upper_bounds.insert(var, value.clone());

                // Check existing bounds
                if let Some(lb) = self.lower_bounds.get(&var)
                    && &value < lb
                {
                    return Err(Conflict {
                        constraints: vec![constraint_id],
                    });
                }
                if let Some(ub) = self.upper_bounds.get(&var)
                    && &value > ub
                {
                    return Err(Conflict {
                        constraints: vec![constraint_id],
                    });
                }
            }
        }

        Ok(())
    }

    /// Get the current assignment to a variable.
    pub fn get_value(&self, var: VarId) -> Option<&BigRational> {
        self.assignment.get(&var)
    }

    /// Check if a basic variable violates its bounds.
    fn violates_bounds(&self, var: VarId) -> bool {
        if let Some(val) = self.assignment.get(&var) {
            if let Some(lb) = self.lower_bounds.get(&var)
                && val < lb
            {
                return true;
            }
            if let Some(ub) = self.upper_bounds.get(&var)
                && val > ub
            {
                return true;
            }
        }
        false
    }

    /// Find a basic variable that violates its bounds.
    fn find_violating_basic_var(&self) -> Option<VarId> {
        if self.use_blands_rule {
            // Use Bland's rule: pick the smallest index
            self.basic_vars
                .iter()
                .filter(|&&var| self.violates_bounds(var))
                .min()
                .copied()
        } else {
            self.basic_vars
                .iter()
                .find(|&&var| self.violates_bounds(var))
                .copied()
        }
    }

    /// Find a non-basic variable to pivot with.
    /// Returns (non_basic_var, improving_direction).
    fn find_pivot_non_basic(
        &self,
        basic_var: VarId,
        target_increase: bool,
    ) -> Option<(VarId, bool)> {
        let row = self.rows.get(&basic_var)?;

        let mut candidates = Vec::new();

        for (nb_var, coeff) in &row.coeffs {
            let current_val = self.assignment.get(nb_var)?;
            let lb = self.lower_bounds.get(nb_var);
            let ub = self.upper_bounds.get(nb_var);

            // Determine if we can increase or decrease nb_var
            let can_increase = ub.is_none_or(|bound| bound > current_val);
            let can_decrease = lb.is_none_or(|bound| bound < current_val);

            // If coeff > 0: increasing nb_var increases basic_var
            // If coeff < 0: increasing nb_var decreases basic_var
            let increases_basic = coeff.is_positive();

            if target_increase {
                // We want to increase basic_var
                if increases_basic && can_increase {
                    candidates.push((*nb_var, true));
                } else if !increases_basic && can_decrease {
                    candidates.push((*nb_var, false));
                }
            } else {
                // We want to decrease basic_var
                if increases_basic && can_decrease {
                    candidates.push((*nb_var, false));
                } else if !increases_basic && can_increase {
                    candidates.push((*nb_var, true));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Use Bland's rule: pick smallest index
        if self.use_blands_rule {
            candidates.sort_by_key(|(var, _)| *var);
        }

        Some(candidates[0])
    }

    /// Perform a pivot operation.
    /// Swap basic_var (currently basic) with non_basic_var (currently non-basic).
    pub fn pivot(&mut self, basic_var: VarId, non_basic_var: VarId) -> Result<(), String> {
        // Get the row for basic_var
        let row = self
            .rows
            .get(&basic_var)
            .ok_or_else(|| format!("No row for basic variable x{}", basic_var))?;

        // Get coefficient of non_basic_var in this row
        let coeff = row
            .coeffs
            .get(&non_basic_var)
            .ok_or_else(|| {
                format!(
                    "Non-basic variable x{} not in row for x{}",
                    non_basic_var, basic_var
                )
            })?
            .clone();

        if coeff.is_zero() {
            return Err(format!("Coefficient of x{} is zero", non_basic_var));
        }

        // Solve for non_basic_var in terms of basic_var
        // Old: basic_var = constant + coeff * non_basic_var + sum(...)
        // New: non_basic_var = (basic_var - constant - sum(...)) / coeff

        let mut new_row = Row::new(non_basic_var);
        new_row.constant = -&row.constant / &coeff;
        new_row
            .coeffs
            .insert(basic_var, BigRational::one() / &coeff);

        for (var, c) in &row.coeffs {
            if var != &non_basic_var {
                new_row.coeffs.insert(*var, -c / &coeff);
            }
        }

        // Substitute non_basic_var in all other rows
        let rows_to_update: Vec<VarId> = self
            .rows
            .keys()
            .filter(|&&v| v != basic_var)
            .copied()
            .collect();

        for row_var in rows_to_update {
            if let Some(r) = self.rows.get_mut(&row_var) {
                r.substitute(non_basic_var, &new_row);
            }
        }

        // Remove old row and add new row
        self.rows.remove(&basic_var);
        self.rows.insert(non_basic_var, new_row);

        // Update basic/non-basic sets
        self.basic_vars.remove(&basic_var);
        self.basic_vars.insert(non_basic_var);
        self.non_basic_vars.remove(&non_basic_var);
        self.non_basic_vars.insert(basic_var);

        // Update assignments
        self.update_assignment();

        Ok(())
    }

    /// Update the assignment based on current tableau.
    fn update_assignment(&mut self) {
        // Evaluate all basic variables
        for (basic_var, row) in &self.rows {
            let value = row.eval(&self.assignment);
            self.assignment.insert(*basic_var, value);
        }
    }

    /// Check feasibility and fix violations using pivoting.
    pub fn check(&mut self) -> Result<SimplexResult, Conflict> {
        let max_iterations = 10000;
        let mut iterations = 0;

        while let Some(violating_var) = self.find_violating_basic_var() {
            iterations += 1;
            if iterations > max_iterations {
                return Ok(SimplexResult::Unknown);
            }

            let current_val = self
                .assignment
                .get(&violating_var)
                .cloned()
                .ok_or_else(|| Conflict {
                    constraints: vec![],
                })?;

            let lb = self.lower_bounds.get(&violating_var);
            let ub = self.upper_bounds.get(&violating_var);

            // Determine if we need to increase or decrease the variable
            let need_increase = lb.is_some_and(|l| &current_val < l);
            let need_decrease = ub.is_some_and(|u| &current_val > u);

            if !need_increase && !need_decrease {
                continue;
            }

            // Find a non-basic variable to pivot with
            if let Some((nb_var, _direction)) =
                self.find_pivot_non_basic(violating_var, need_increase)
            {
                // Compute the new value for nb_var
                let target_value = if need_increase {
                    lb.cloned().unwrap_or_else(BigRational::zero)
                } else {
                    ub.cloned().unwrap_or_else(BigRational::zero)
                };

                // Update nb_var to move basic_var to target
                let row = self.rows.get(&violating_var).ok_or_else(|| Conflict {
                    constraints: vec![],
                })?;
                let coeff = row.coeffs.get(&nb_var).cloned().ok_or_else(|| Conflict {
                    constraints: vec![],
                })?;

                let delta = &target_value - &current_val;
                let nb_delta = &delta / &coeff;
                let current_nb = self
                    .assignment
                    .get(&nb_var)
                    .cloned()
                    .ok_or_else(|| Conflict {
                        constraints: vec![],
                    })?;
                let new_nb = current_nb + nb_delta;

                // Clamp to bounds
                let new_nb = if let Some(lb) = self.lower_bounds.get(&nb_var) {
                    new_nb.max(lb.clone())
                } else {
                    new_nb
                };
                let new_nb = if let Some(ub) = self.upper_bounds.get(&nb_var) {
                    new_nb.min(ub.clone())
                } else {
                    new_nb
                };

                self.assignment.insert(nb_var, new_nb);
                self.update_assignment();
            } else {
                // No pivot found - problem is infeasible or unbounded
                let constraints = self
                    .var_to_constraints
                    .get(&violating_var)
                    .cloned()
                    .unwrap_or_default();
                return Err(Conflict { constraints });
            }
        }

        Ok(SimplexResult::Sat)
    }

    /// Dual simplex algorithm for Linear Programming.
    ///
    /// The dual simplex maintains dual feasibility (all reduced costs non-negative)
    /// while working toward primal feasibility (all variables within bounds).
    ///
    /// This is particularly useful for:
    /// - Reoptimization after adding constraints
    /// - Branch-and-bound in integer programming
    /// - Problems that naturally start dual-feasible
    ///
    /// Reference: Standard LP textbooks and Z3's dual simplex implementation.
    pub fn check_dual(&mut self) -> Result<SimplexResult, Conflict> {
        let max_iterations = 10000;
        let mut iterations = 0;

        // Dual simplex loop: restore primal feasibility
        while let Some(leaving_var) = self.find_violating_basic_var() {
            iterations += 1;
            if iterations > max_iterations {
                return Ok(SimplexResult::Unknown);
            }

            // Get the row for the leaving variable
            let row = match self.rows.get(&leaving_var) {
                Some(r) => r.clone(),
                None => continue,
            };

            let current_val = self
                .assignment
                .get(&leaving_var)
                .cloned()
                .unwrap_or_else(BigRational::zero);

            let lb = self.lower_bounds.get(&leaving_var);
            let ub = self.upper_bounds.get(&leaving_var);

            // Determine which bound is violated
            let violated_lower = lb.is_some_and(|l| &current_val < l);
            let violated_upper = ub.is_some_and(|u| &current_val > u);

            if !violated_lower && !violated_upper {
                continue;
            }

            // For dual simplex, find entering variable using dual pricing rule
            let entering_var = self.find_entering_var_dual(&row, violated_lower)?;

            // Pivot leaving_var out, entering_var in
            self.pivot(leaving_var, entering_var)
                .map_err(|_| Conflict {
                    constraints: self
                        .var_to_constraints
                        .get(&leaving_var)
                        .cloned()
                        .unwrap_or_default(),
                })?;
        }

        Ok(SimplexResult::Sat)
    }

    /// Find the entering variable for dual simplex using dual pricing rule.
    ///
    /// For dual simplex:
    /// - If leaving var violates lower bound: choose entering var with negative coeff
    /// - If leaving var violates upper bound: choose entering var with positive coeff
    /// - Among candidates, choose one that maintains dual feasibility
    fn find_entering_var_dual(&self, row: &Row, violated_lower: bool) -> Result<VarId, Conflict> {
        let mut best_var = None;
        let mut best_ratio: Option<BigRational> = None;

        // Iterate through non-basic variables in the row
        for (&nb_var, coeff) in &row.coeffs {
            // For dual simplex pricing:
            // - If violated_lower, we need coeff < 0 (to increase basic var)
            // - If violated_upper, we need coeff > 0 (to decrease basic var)
            let sign_ok = if violated_lower {
                coeff.is_negative()
            } else {
                coeff.is_positive()
            };

            if !sign_ok || coeff.is_zero() {
                continue;
            }

            // Compute the dual ratio for maintaining dual feasibility
            // This is a simplified version; full implementation would compute reduced costs
            let ratio = coeff.abs();

            match &best_ratio {
                None => {
                    best_ratio = Some(ratio);
                    best_var = Some(nb_var);
                }
                Some(current_best) => {
                    // Choose the variable with smallest ratio (steepest descent in dual)
                    // Use Bland's rule for tie-breaking if enabled
                    if &ratio < current_best {
                        best_ratio = Some(ratio);
                        best_var = Some(nb_var);
                    } else if self.use_blands_rule && &ratio == current_best {
                        // Bland's rule: choose smaller index
                        // best_var is guaranteed Some when best_ratio is Some
                        if best_var.is_none_or(|bv| nb_var < bv) {
                            best_var = Some(nb_var);
                        }
                    }
                }
            }
        }

        best_var.ok_or(Conflict {
            constraints: vec![],
        })
    }

    /// Get all variables.
    pub fn vars(&self) -> Vec<VarId> {
        let mut vars: Vec<VarId> = self
            .basic_vars
            .iter()
            .chain(self.non_basic_vars.iter())
            .copied()
            .collect();
        vars.sort_unstable();
        vars
    }

    /// Get all basic variables.
    pub fn basic_vars(&self) -> Vec<VarId> {
        let mut vars: Vec<VarId> = self.basic_vars.iter().copied().collect();
        vars.sort_unstable();
        vars
    }

    /// Get all non-basic variables.
    pub fn non_basic_vars(&self) -> Vec<VarId> {
        let mut vars: Vec<VarId> = self.non_basic_vars.iter().copied().collect();
        vars.sort_unstable();
        vars
    }

    /// Get the number of rows in the tableau.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Get the number of variables.
    pub fn num_vars(&self) -> usize {
        self.basic_vars.len() + self.non_basic_vars.len()
    }

    /// Enable or disable Bland's anti-cycling rule.
    pub fn set_blands_rule(&mut self, enable: bool) {
        self.use_blands_rule = enable;
    }

    /// Get the current model (satisfying assignment).
    /// Returns None if the system is not known to be satisfiable.
    pub fn get_model(&self) -> Option<FxHashMap<VarId, BigRational>> {
        // Check if all variables satisfy their bounds
        for (var, val) in &self.assignment {
            if let Some(lb) = self.lower_bounds.get(var)
                && val < lb
            {
                return None;
            }
            if let Some(ub) = self.upper_bounds.get(var)
                && val > ub
            {
                return None;
            }
        }
        Some(self.assignment.clone())
    }

    /// Check if the current assignment satisfies all bounds.
    pub fn is_feasible(&self) -> bool {
        for (var, val) in &self.assignment {
            if let Some(lb) = self.lower_bounds.get(var)
                && val < lb
            {
                return false;
            }
            if let Some(ub) = self.upper_bounds.get(var)
                && val > ub
            {
                return false;
            }
        }
        true
    }

    /// Find a variable that violates its bounds, if any.
    pub fn find_violated_bound(&self) -> Option<VarId> {
        for (var, val) in &self.assignment {
            if let Some(lb) = self.lower_bounds.get(var)
                && val < lb
            {
                return Some(*var);
            }
            if let Some(ub) = self.upper_bounds.get(var)
                && val > ub
            {
                return Some(*var);
            }
        }
        None
    }

    /// Get all constraints associated with a variable.
    pub fn get_constraints(&self, var: VarId) -> Vec<ConstraintId> {
        self.var_to_constraints
            .get(&var)
            .cloned()
            .unwrap_or_default()
    }

    /// Extract a minimal conflicting core from constraints.
    /// This is a simple implementation that returns all constraints involved.
    /// A more sophisticated version would compute a true minimal unsat core.
    pub fn get_unsat_core(&self, conflict: &Conflict) -> Vec<ConstraintId> {
        conflict.constraints.clone()
    }

    /// Theory propagation: deduce new bounds from existing constraints.
    /// Returns a list of (var, bound_type, value) tuples representing deduced bounds.
    pub fn propagate(&self) -> Vec<(VarId, BoundType, BigRational)> {
        let mut propagated = Vec::new();

        // For each row: basic_var = constant + sum(coeff * non_basic_var)
        // We can deduce bounds on basic_var from bounds on non_basic vars
        for row in self.rows.values() {
            let basic_var = row.basic_var;

            // Compute lower bound: min value of basic_var
            // basic_var >= constant + sum(min(coeff * lb, coeff * ub))
            let mut lower_bound = row.constant.clone();
            let mut has_finite_lower = true;

            for (var, coeff) in &row.coeffs {
                if coeff.is_positive() {
                    // Positive coeff: use lower bound of var
                    if let Some(lb) = self.lower_bounds.get(var) {
                        lower_bound += coeff * lb;
                    } else {
                        has_finite_lower = false;
                        break;
                    }
                } else {
                    // Negative coeff: use upper bound of var
                    if let Some(ub) = self.upper_bounds.get(var) {
                        lower_bound += coeff * ub;
                    } else {
                        has_finite_lower = false;
                        break;
                    }
                }
            }

            if has_finite_lower {
                // Check if this is a tighter bound
                if let Some(current_lb) = self.lower_bounds.get(&basic_var) {
                    if &lower_bound > current_lb {
                        propagated.push((basic_var, BoundType::Lower, lower_bound.clone()));
                    }
                } else {
                    propagated.push((basic_var, BoundType::Lower, lower_bound.clone()));
                }
            }

            // Compute upper bound: max value of basic_var
            let mut upper_bound = row.constant.clone();
            let mut has_finite_upper = true;

            for (var, coeff) in &row.coeffs {
                if coeff.is_positive() {
                    // Positive coeff: use upper bound of var
                    if let Some(ub) = self.upper_bounds.get(var) {
                        upper_bound += coeff * ub;
                    } else {
                        has_finite_upper = false;
                        break;
                    }
                } else {
                    // Negative coeff: use lower bound of var
                    if let Some(lb) = self.lower_bounds.get(var) {
                        upper_bound += coeff * lb;
                    } else {
                        has_finite_upper = false;
                        break;
                    }
                }
            }

            if has_finite_upper {
                // Check if this is a tighter bound
                if let Some(current_ub) = self.upper_bounds.get(&basic_var) {
                    if &upper_bound < current_ub {
                        propagated.push((basic_var, BoundType::Upper, upper_bound.clone()));
                    }
                } else {
                    propagated.push((basic_var, BoundType::Upper, upper_bound.clone()));
                }
            }
        }

        propagated
    }

    /// Assert a linear constraint: sum(coeffs) + constant {<=, >=, ==} 0.
    /// Returns a new slack variable if needed, and the constraint ID.
    pub fn assert_constraint(
        &mut self,
        coeffs: FxHashMap<VarId, BigRational>,
        constant: BigRational,
        bound_type: BoundType,
        constraint_id: ConstraintId,
    ) -> Result<VarId, Conflict> {
        // Create a fresh slack variable for the constraint
        let slack_var = self.fresh_var();

        // slack_var = sum(coeffs) + constant
        let row = Row::from_expr(slack_var, constant, coeffs);
        self.add_row(row).map_err(|_| Conflict {
            constraints: vec![constraint_id],
        })?;

        // Add appropriate bound on slack_var based on bound_type
        let zero = BigRational::zero();
        match bound_type {
            BoundType::Lower => {
                // sum(coeffs) + constant >= 0  =>  slack_var >= 0
                self.add_bound(slack_var, BoundType::Lower, zero, constraint_id)?;
            }
            BoundType::Upper => {
                // sum(coeffs) + constant <= 0  =>  slack_var <= 0
                self.add_bound(slack_var, BoundType::Upper, zero, constraint_id)?;
            }
            BoundType::Equal => {
                // sum(coeffs) + constant == 0  =>  slack_var == 0
                self.add_bound(slack_var, BoundType::Equal, zero.clone(), constraint_id)?;
                // Equal means both lower and upper bound
                self.add_bound(slack_var, BoundType::Lower, zero.clone(), constraint_id)?;
                self.add_bound(slack_var, BoundType::Upper, zero, constraint_id)?;
            }
        }

        Ok(slack_var)
    }
}

impl Default for SimplexTableau {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SimplexTableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Simplex Tableau:")?;
        writeln!(f, "  Basic variables: {:?}", self.basic_vars())?;
        writeln!(f, "  Non-basic variables: {:?}", self.non_basic_vars())?;
        writeln!(f, "  Rows:")?;
        for row in self.rows.values() {
            writeln!(f, "    {}", row)?;
        }
        writeln!(f, "  Assignment:")?;
        for var in self.vars() {
            if let Some(val) = self.assignment.get(&var) {
                write!(f, "    x{} = {}", var, val)?;
                if let Some(lb) = self.lower_bounds.get(&var) {
                    write!(f, " (>= {})", lb)?;
                }
                if let Some(ub) = self.upper_bounds.get(&var) {
                    write!(f, " (<= {})", ub)?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_row_eval() {
        let mut row = Row::new(0);
        row.constant = rat(5);
        row.coeffs.insert(1, rat(2));
        row.coeffs.insert(2, rat(3));

        let mut values = FxHashMap::default();
        values.insert(1, rat(1));
        values.insert(2, rat(2));

        // 5 + 2*1 + 3*2 = 5 + 2 + 6 = 13
        assert_eq!(row.eval(&values), rat(13));
    }

    #[test]
    fn test_row_add() {
        let mut row1 = Row::new(0);
        row1.constant = rat(5);
        row1.coeffs.insert(1, rat(2));

        let mut row2 = Row::new(0);
        row2.constant = rat(3);
        row2.coeffs.insert(1, rat(1));
        row2.coeffs.insert(2, rat(4));

        // row1 += 2 * row2
        // row1 = 5 + 2*x1 + 2*(3 + x1 + 4*x2)
        // row1 = 5 + 2*x1 + 6 + 2*x1 + 8*x2
        // row1 = 11 + 4*x1 + 8*x2
        row1.add_row(&rat(2), &row2);
        assert_eq!(row1.constant, rat(11));
        assert_eq!(row1.coeffs.get(&1), Some(&rat(4)));
        assert_eq!(row1.coeffs.get(&2), Some(&rat(8)));
    }

    #[test]
    fn test_simplex_basic() {
        let mut tableau = SimplexTableau::new();

        // Variables: x0, x1
        // Constraint: x0 + x1 <= 10
        // Introduce slack: x2 = 10 - x0 - x1

        let x0 = tableau.fresh_var();
        let x1 = tableau.fresh_var();
        let x2 = tableau.fresh_var();

        // x2 = 10 - x0 - x1
        let mut row = Row::new(x2);
        row.constant = rat(10);
        row.coeffs.insert(x0, rat(-1));
        row.coeffs.insert(x1, rat(-1));

        tableau.add_row(row).unwrap();

        // Bounds: x0 >= 0, x1 >= 0, x2 >= 0
        tableau.add_bound(x0, BoundType::Lower, rat(0), 0).unwrap();
        tableau.add_bound(x1, BoundType::Lower, rat(0), 1).unwrap();
        tableau.add_bound(x2, BoundType::Lower, rat(0), 2).unwrap();

        let result = tableau.check().unwrap();
        assert_eq!(result, SimplexResult::Sat);
    }

    #[test]
    fn test_simplex_infeasible() {
        let mut tableau = SimplexTableau::new();

        let x = tableau.fresh_var();

        // x >= 5 and x <= 3 (conflicting bounds)
        tableau.add_bound(x, BoundType::Lower, rat(5), 0).unwrap();
        let result = tableau.add_bound(x, BoundType::Upper, rat(3), 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_simplex_pivot() {
        let mut tableau = SimplexTableau::new();

        let x0 = tableau.fresh_var();
        let x1 = tableau.fresh_var();
        let x2 = tableau.fresh_var();

        // x2 = 10 - 2*x0 - 3*x1
        let mut row = Row::new(x2);
        row.constant = rat(10);
        row.coeffs.insert(x0, rat(-2));
        row.coeffs.insert(x1, rat(-3));

        tableau.add_row(row).unwrap();

        // Pivot x2 and x0
        tableau.pivot(x2, x0).unwrap();

        // After pivot: x0 = (10 - x2 - 3*x1) / 2 = 5 - x2/2 - 3*x1/2
        assert!(tableau.basic_vars.contains(&x0));
        assert!(tableau.non_basic_vars.contains(&x2));

        let new_row = tableau.rows.get(&x0).unwrap();
        assert_eq!(new_row.constant, rat(5));
    }

    #[test]
    fn test_simplex_dual() {
        let mut tableau = SimplexTableau::new();

        // Test dual simplex with a simple LP
        // Variables: x0, x1
        // Constraint: x0 + x1 <= 10
        // Bounds: x0 >= 0, x1 >= 0

        let x0 = tableau.fresh_var();
        let x1 = tableau.fresh_var();
        let x2 = tableau.fresh_var(); // slack variable

        // x2 = 10 - x0 - x1
        let mut row = Row::new(x2);
        row.constant = rat(10);
        row.coeffs.insert(x0, rat(-1));
        row.coeffs.insert(x1, rat(-1));

        tableau.add_row(row).unwrap();

        // Bounds: x0 >= 0, x1 >= 0, x2 >= 0
        tableau.add_bound(x0, BoundType::Lower, rat(0), 0).unwrap();
        tableau.add_bound(x1, BoundType::Lower, rat(0), 1).unwrap();
        tableau.add_bound(x2, BoundType::Lower, rat(0), 2).unwrap();

        // Use dual simplex
        let result = tableau.check_dual().unwrap();
        assert_eq!(result, SimplexResult::Sat);

        // Verify the solution is feasible
        assert!(tableau.is_feasible());
    }

    #[test]
    fn test_simplex_dual_with_bounds() {
        let mut tableau = SimplexTableau::new();

        // Test dual simplex with tighter bounds
        let x0 = tableau.fresh_var();
        let x1 = tableau.fresh_var();
        let x2 = tableau.fresh_var();

        // x2 = 10 - x0 - x1
        let mut row = Row::new(x2);
        row.constant = rat(10);
        row.coeffs.insert(x0, rat(-1));
        row.coeffs.insert(x1, rat(-1));

        tableau.add_row(row).unwrap();

        // Bounds: 0 <= x0 <= 5, 0 <= x1 <= 5, x2 >= 0
        tableau.add_bound(x0, BoundType::Lower, rat(0), 0).unwrap();
        tableau.add_bound(x0, BoundType::Upper, rat(5), 1).unwrap();
        tableau.add_bound(x1, BoundType::Lower, rat(0), 2).unwrap();
        tableau.add_bound(x1, BoundType::Upper, rat(5), 3).unwrap();
        tableau.add_bound(x2, BoundType::Lower, rat(0), 4).unwrap();

        // Use dual simplex
        let result = tableau.check_dual().unwrap();
        assert_eq!(result, SimplexResult::Sat);

        // Verify the solution is feasible
        assert!(tableau.is_feasible());
    }
}
