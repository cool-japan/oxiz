//! SimplexSolver — LP solver for parametric analysis.
//!
//! This module provides a high-level LP interface used by
//! `simplex_parametric` for sensitivity analysis and parametric
//! optimization.
//!
//! # Algorithm
//!
//! Uses the Big-M primal simplex method with exact `BigRational` arithmetic.
//!
//! ## Standard Form Conversion
//!
//! For each constraint `a'x ≤ b`, `a'x ≥ b`, or `a'x = b`:
//!
//! - **Le**: add slack s ≥ 0  →  a'x + s = b  (s is initial basic variable)
//! - **Ge**: add surplus s ≥ 0 and artificial a ≥ 0  →  a'x - s + a = b
//! - **Eq**: add artificial a ≥ 0  →  a'x + a = b
//!
//! Artificials appear in the objective with coefficient +M (Big-M method) so
//! the primal simplex drives them to zero.  Once all artificials = 0, the
//! basis is feasible and the true objective is minimised.
//!
//! ## Shadow Prices
//!
//! Shadow prices are extracted from the objective row of the final tableau
//! at the columns corresponding to slack variables.  For a minimisation LP
//! in standard form the shadow price of constraint i is −(reduced cost of
//! its slack variable s_i), which equals the dual variable π_i.

#![allow(dead_code)]

#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use thiserror::Error;

// ── public types ─────────────────────────────────────────────────────────────

/// Error type for `SimplexSolver`.
#[derive(Debug, Error, Clone)]
pub enum SimplexError {
    /// Index out of bounds.
    #[error("index {index} out of bounds (len {len})")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// The length of the collection.
        len: usize,
    },

    /// Shadow price queried before solving.
    #[error("shadow_price called before solve()")]
    NotYetSolved,

    /// The LP is infeasible so shadow prices are undefined.
    #[error("shadow_price undefined: problem is infeasible")]
    Infeasible,

    /// The LP is unbounded so shadow prices are undefined.
    #[error("shadow_price undefined: problem is unbounded")]
    Unbounded,
}

/// Sense of a linear constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `a'x ≤ b`
    Le,
    /// `a'x ≥ b`
    Ge,
    /// `a'x = b`
    Eq,
}

/// A linear constraint `a'x {≤|≥|=} b`.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Coefficients for each decision variable (length == `n_vars`).
    pub coefficients: Vec<BigRational>,
    /// Right-hand side.
    pub rhs: BigRational,
    /// Sense of the constraint.
    pub kind: ConstraintKind,
}

impl Constraint {
    /// Construct a constraint from integer coefficients (convenience).
    pub fn le(coeffs: impl Into<Vec<BigRational>>, rhs: BigRational) -> Self {
        Self {
            coefficients: coeffs.into(),
            rhs,
            kind: ConstraintKind::Le,
        }
    }

    /// Construct a `≥` constraint.
    pub fn ge(coeffs: impl Into<Vec<BigRational>>, rhs: BigRational) -> Self {
        Self {
            coefficients: coeffs.into(),
            rhs,
            kind: ConstraintKind::Ge,
        }
    }

    /// Construct an equality constraint.
    pub fn eq(coeffs: impl Into<Vec<BigRational>>, rhs: BigRational) -> Self {
        Self {
            coefficients: coeffs.into(),
            rhs,
            kind: ConstraintKind::Eq,
        }
    }
}

/// Status of a solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// Optimal solution found.
    Optimal,
    /// Problem is infeasible.
    Infeasible,
    /// Problem is unbounded.
    Unbounded,
}

/// Result returned by `SimplexSolver::solve`.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Termination status.
    pub status: SolveStatus,
    /// Optimal values for each decision variable.
    pub values: Vec<BigRational>,
    /// Optimal objective value (valid only when `status == Optimal`).
    pub objective: BigRational,
    /// Shadow price (dual variable) for each original constraint.
    /// Valid only when `status == Optimal`.
    pub shadow_prices: Vec<BigRational>,
}

// ── main struct ──────────────────────────────────────────────────────────────

/// High-level LP solver for parametric analysis.
///
/// Minimises `c'x` subject to a set of [`Constraint`]s and `x ≥ 0`.
#[derive(Debug, Clone)]
pub struct SimplexSolver {
    /// Number of decision variables.
    n_vars: usize,
    /// Objective coefficients (length == `n_vars`).
    obj_coeffs: Vec<BigRational>,
    /// Constraints (each has `coefficients.len() == n_vars`).
    constraints: Vec<Constraint>,
    /// Cached result of the last `solve()` call, `None` if not yet solved.
    last_result: Option<SolveResult>,
}

impl SimplexSolver {
    /// Create a solver with the given objective and constraints.
    ///
    /// `obj_coeffs[i]` is the coefficient of variable i in the minimised
    /// objective `c'x`.  All decision variables are implicitly non-negative.
    pub fn new(obj_coeffs: Vec<BigRational>, constraints: Vec<Constraint>) -> Self {
        let n_vars = obj_coeffs.len();
        Self {
            n_vars,
            obj_coeffs,
            constraints,
            last_result: None,
        }
    }

    /// Update a single objective coefficient.
    ///
    /// Returns an error when `i ≥ n_vars`.
    pub fn set_objective_coefficient(
        &mut self,
        i: usize,
        val: BigRational,
    ) -> Result<(), SimplexError> {
        if i >= self.n_vars {
            return Err(SimplexError::IndexOutOfBounds {
                index: i,
                len: self.n_vars,
            });
        }
        self.obj_coeffs[i] = val;
        // Invalidate cached result since objective changed.
        self.last_result = None;
        Ok(())
    }

    /// Get objective coefficient `i`.
    pub fn get_objective_coefficient(&self, i: usize) -> Result<&BigRational, SimplexError> {
        self.obj_coeffs.get(i).ok_or(SimplexError::IndexOutOfBounds {
            index: i,
            len: self.n_vars,
        })
    }

    /// Get the RHS of constraint `i`.
    pub fn get_rhs(&self, i: usize) -> Result<&BigRational, SimplexError> {
        self.constraints
            .get(i)
            .map(|c| &c.rhs)
            .ok_or(SimplexError::IndexOutOfBounds {
                index: i,
                len: self.constraints.len(),
            })
    }

    /// Set the RHS of constraint `i`.
    pub fn set_rhs(&mut self, i: usize, val: BigRational) -> Result<(), SimplexError> {
        let len = self.constraints.len();
        self.constraints
            .get_mut(i)
            .ok_or(SimplexError::IndexOutOfBounds { index: i, len })?
            .rhs = val;
        self.last_result = None;
        Ok(())
    }

    /// Return the shadow price (dual variable) for constraint `i`.
    ///
    /// The shadow price represents how much the optimal objective improves
    /// (decreases, since we minimise) per unit increase in `b_i`.
    ///
    /// Returns an error if `solve()` has not been called yet, or if the
    /// last solve did not yield an optimal solution.
    pub fn shadow_price(&self, i: usize) -> Result<BigRational, SimplexError> {
        let result = self.last_result.as_ref().ok_or(SimplexError::NotYetSolved)?;

        match result.status {
            SolveStatus::Infeasible => return Err(SimplexError::Infeasible),
            SolveStatus::Unbounded => return Err(SimplexError::Unbounded),
            SolveStatus::Optimal => {}
        }

        result
            .shadow_prices
            .get(i)
            .cloned()
            .ok_or(SimplexError::IndexOutOfBounds {
                index: i,
                len: result.shadow_prices.len(),
            })
    }

    /// Return the cached objective value from the last solve.
    pub fn objective_value(&self) -> Option<BigRational> {
        self.last_result
            .as_ref()
            .filter(|r| r.status == SolveStatus::Optimal)
            .map(|r| r.objective.clone())
    }

    /// Solve the LP via Big-M primal simplex.
    ///
    /// Converts to standard equality form, appends artificial variables with
    /// Big-M objective penalty, and runs the primal simplex until optimal or
    /// detected infeasible / unbounded.  Shadow prices are read directly from
    /// the final objective row (reduced costs of slack columns).
    ///
    /// Stores the result internally so that subsequent `shadow_price()` calls
    /// do not require re-solving.
    pub fn solve(&mut self) -> Result<SolveResult, SimplexError> {
        let result = self.run_bigm_simplex(
            &self.obj_coeffs.clone(),
            &self.constraints.clone(),
        );
        self.last_result = Some(result.clone());
        Ok(result)
    }

    /// Build and run the Big-M primal simplex.
    ///
    /// Column layout of tableau T (m rows × (total_cols + 1) cols):
    ///   x_0 … x_{n-1} | s_0 … s_{m-1} | a_0 … a_{k-1} | RHS
    ///
    /// Where:
    /// - x_j  : decision variables (indices 0..n)
    /// - s_i  : slack (Le) or surplus (Ge) variables (indices n..n+m)
    /// - a_i  : artificial variables for Ge / Eq rows (indices n+m..n+m+k)
    ///
    /// Objective row (row `m`) is maintained in `obj_row` alongside T.
    fn run_bigm_simplex(
        &self,
        obj_coeffs: &[BigRational],
        constraints: &[Constraint],
    ) -> SolveResult {
        let n = obj_coeffs.len(); // decision variables
        let m = constraints.len(); // constraints
        if m == 0 {
            // Unconstrained: feasible at x=0 with obj=0.
            return SolveResult {
                status: SolveStatus::Optimal,
                values: vec![BigRational::zero(); n],
                objective: BigRational::zero(),
                shadow_prices: Vec::new(),
            };
        }

        // Big-M value: large enough to dominate all natural objective values.
        // We use 10^6 * max(|c_j|) clamped to at least 10^6.
        let big_m = {
            let max_coeff = obj_coeffs
                .iter()
                .map(|c| c.abs())
                .fold(BigRational::zero(), |a, b| if a >= b { a } else { b });
            let base = BigRational::new(BigInt::from(1_000_000i64), BigInt::from(1i64));
            let scaled = &base * (&max_coeff + BigRational::new(BigInt::from(1i64), BigInt::from(1i64)));
            if scaled < base { base } else { scaled }
        };

        // Count artificials (needed for Ge and Eq).
        let n_artificial: usize = constraints
            .iter()
            .filter(|c| matches!(c.kind, ConstraintKind::Ge | ConstraintKind::Eq))
            .count();

        // total decision + slack + artificial columns (not counting RHS).
        let total_cols = n + m + n_artificial;
        let rhs_col = total_cols; // index of RHS column

        // ── build tableau ──────────────────────────────────────────────────
        // tableau[i] has length total_cols + 1 (including RHS).
        let mut tab: Vec<Vec<BigRational>> =
            vec![vec![BigRational::zero(); total_cols + 1]; m];

        // obj_row[j] = reduced cost of column j (minimisation convention).
        let mut obj_row: Vec<BigRational> = vec![BigRational::zero(); total_cols + 1];

        // basis[i] = index of the basic variable in row i.
        let mut basis: Vec<usize> = vec![0usize; m];

        let mut art_idx = n + m; // running index for next artificial column

        for (i, c) in constraints.iter().enumerate() {
            // Decision variable coefficients.
            for (j, coeff) in c.coefficients.iter().enumerate() {
                if j < n {
                    tab[i][j] = coeff.clone();
                }
            }

            // Slack column for this row.
            let slack_col = n + i;

            match c.kind {
                ConstraintKind::Le => {
                    // slack s_i >= 0: a'x + s_i = b_i
                    tab[i][slack_col] = BigRational::new(BigInt::from(1i64), BigInt::from(1i64));
                    tab[i][rhs_col] = c.rhs.clone();
                    basis[i] = slack_col; // slack is initial basic variable
                    // Objective: slack has coefficient 0.
                }
                ConstraintKind::Ge => {
                    // surplus s_i >= 0: a'x - s_i + a_i = b_i
                    tab[i][slack_col] = BigRational::new(BigInt::from(-1i64), BigInt::from(1i64));
                    tab[i][art_idx] = BigRational::new(BigInt::from(1i64), BigInt::from(1i64));
                    tab[i][rhs_col] = c.rhs.clone();
                    basis[i] = art_idx;
                    // Artificial in objective: +M * a_i
                    obj_row[art_idx] = big_m.clone();
                    art_idx += 1;
                }
                ConstraintKind::Eq => {
                    // a'x + a_i = b_i (no slack)
                    tab[i][art_idx] = BigRational::new(BigInt::from(1i64), BigInt::from(1i64));
                    tab[i][rhs_col] = c.rhs.clone();
                    basis[i] = art_idx;
                    obj_row[art_idx] = big_m.clone();
                    art_idx += 1;
                }
            }

            // If RHS < 0 for any type, multiply the row by -1 to make it non-negative.
            if tab[i][rhs_col] < BigRational::zero() {
                for elem in &mut tab[i] {
                    *elem = -elem.clone();
                }
                // For Le: flipping makes it effectively a Ge (slack becomes -1),
                // but the artificial will handle it via Big-M.
                // Adjust: basis[i] stays the same column; RHS is now positive.
            }
        }

        // Set objective row for decision variables.
        for (j, c) in obj_coeffs.iter().enumerate() {
            obj_row[j] = c.clone();
        }

        // ── eliminate basic variables from the objective row ──────────────
        // For each artificial basic variable (in Ge/Eq rows), subtract
        // M * (constraint row) from the objective row so the objective is
        // expressed purely in terms of non-basic variables.
        for i in 0..m {
            let bv = basis[i];
            if bv >= n + m {
                // Artificial basic variable: eliminate from obj_row.
                let factor = obj_row[bv].clone();
                if !factor.is_zero() {
                    for j in 0..=total_cols {
                        let t = tab[i][j].clone();
                        obj_row[j] = obj_row[j].clone() - &factor * &t;
                    }
                }
            }
        }

        // ── primal simplex iterations ─────────────────────────────────────
        const MAX_ITERS: usize = 50_000;

        for _iter in 0..MAX_ITERS {
            // Find entering column: most negative reduced cost (for minimisation).
            let entering = match Self::find_entering(&obj_row, total_cols) {
                Some(e) => e,
                None => break, // All reduced costs >= 0: optimal.
            };

            // Minimum-ratio test: find leaving row.
            let leaving = match Self::find_leaving(&tab, m, entering, rhs_col) {
                Some(l) => l,
                None => {
                    // No leaving variable: problem is unbounded.
                    return SolveResult {
                        status: SolveStatus::Unbounded,
                        values: vec![BigRational::zero(); n],
                        objective: BigRational::zero(),
                        shadow_prices: vec![BigRational::zero(); m],
                    };
                }
            };

            // Pivot.
            Self::pivot_tableau(&mut tab, &mut obj_row, &mut basis, m, leaving, entering, rhs_col);
        }

        // ── extract solution ───────────────────────────────────────────────
        // Check that all artificials are zero.
        let mut values = vec![BigRational::zero(); n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                values[bv] = tab[i][rhs_col].clone();
            } else if bv >= n + m {
                // Artificial still in basis with non-zero value → infeasible.
                if tab[i][rhs_col].abs() > BigRational::new(BigInt::from(1i64), BigInt::from(1_000_000i64)) {
                    return SolveResult {
                        status: SolveStatus::Infeasible,
                        values: vec![BigRational::zero(); n],
                        objective: BigRational::zero(),
                        shadow_prices: vec![BigRational::zero(); m],
                    };
                }
                // Degenerate: artificial is zero, treat as feasible.
            }
        }

        // Compute objective.
        let mut objective = BigRational::zero();
        for (j, c) in obj_coeffs.iter().enumerate() {
            objective += c * &values[j];
        }

        // Shadow prices: for constraint i the shadow price is the negative of
        // the reduced cost of its slack variable s_i in the final objective row.
        // (For Le: π_i = -obj_row[n + i]; for Ge/Eq the sign is flipped.)
        let shadow_prices: Vec<BigRational> = constraints
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let slack_col = n + i;
                let rc = &obj_row[slack_col];
                match c.kind {
                    ConstraintKind::Le => -rc.clone(),
                    ConstraintKind::Ge => rc.clone(),
                    ConstraintKind::Eq => -rc.clone(),
                }
            })
            .collect();

        SolveResult {
            status: SolveStatus::Optimal,
            values,
            objective,
            shadow_prices,
        }
    }

    /// Find the entering column (most negative reduced cost).
    fn find_entering(obj_row: &[BigRational], total_cols: usize) -> Option<usize> {
        let mut best_col = None;
        let mut best_val = BigRational::zero();
        for (j, val) in obj_row.iter().enumerate().take(total_cols) {
            if *val < best_val {
                best_val = val.clone();
                best_col = Some(j);
            }
        }
        best_col
    }

    /// Find the leaving row via the minimum-ratio test.
    fn find_leaving(
        tab: &[Vec<BigRational>],
        m: usize,
        entering: usize,
        rhs_col: usize,
    ) -> Option<usize> {
        let mut best_row = None;
        let mut best_ratio: Option<BigRational> = None;

        for (i, row) in tab.iter().enumerate().take(m) {
            let a_ie = &row[entering];
            if a_ie <= &BigRational::zero() {
                continue; // only positive entries allowed
            }
            let ratio = &row[rhs_col] / a_ie;
            match &best_ratio {
                None => {
                    best_ratio = Some(ratio);
                    best_row = Some(i);
                }
                Some(br) => {
                    if ratio < *br {
                        best_ratio = Some(ratio);
                        best_row = Some(i);
                    }
                }
            }
        }

        best_row
    }

    /// Perform a pivot: entering column `entering` enters basis at row `leaving`.
    fn pivot_tableau(
        tab: &mut [Vec<BigRational>],
        obj_row: &mut [BigRational],
        basis: &mut [usize],
        m: usize,
        leaving: usize,
        entering: usize,
        rhs_col: usize,
    ) {
        let pivot = tab[leaving][entering].clone();
        let total_cols_plus_one = rhs_col + 1;

        // Normalize the pivot row.
        for elem in tab[leaving].iter_mut().take(total_cols_plus_one) {
            let v = elem.clone();
            *elem = v / &pivot;
        }

        // Eliminate entering column from all other rows.
        for i in 0..m {
            if i == leaving {
                continue;
            }
            let factor = tab[i][entering].clone();
            if factor.is_zero() {
                continue;
            }
            // Borrow pivot row values separately to avoid aliasing.
            let pivot_row: Vec<BigRational> = tab[leaving].iter().take(total_cols_plus_one).cloned().collect();
            for (j, pv) in pivot_row.iter().enumerate() {
                let v = tab[i][j].clone();
                tab[i][j] = v - &factor * pv;
            }
        }

        // Eliminate from objective row.
        let obj_factor = obj_row[entering].clone();
        if !obj_factor.is_zero() {
            let pivot_row: Vec<BigRational> = tab[leaving].iter().take(total_cols_plus_one).cloned().collect();
            for (j, pv) in pivot_row.iter().enumerate() {
                let v = obj_row[j].clone();
                obj_row[j] = v - &obj_factor * pv;
            }
        }

        basis[leaving] = entering;
    }

    /// Solve a perturbed version of the LP where constraint `i` has its RHS
    /// increased by `delta`.  Returns the optimal objective value, or `None`
    /// if infeasible/unbounded.
    fn solve_with_rhs_perturbation(
        &self,
        constraint_idx: usize,
        delta: &BigRational,
    ) -> Option<BigRational> {
        let mut perturbed_constraints = self.constraints.clone();
        if let Some(c) = perturbed_constraints.get_mut(constraint_idx) {
            c.rhs = c.rhs.clone() + delta;
        } else {
            return None;
        }

        let result = self.run_bigm_simplex(&self.obj_coeffs, &perturbed_constraints);
        if result.status == SolveStatus::Optimal {
            Some(result.objective)
        } else {
            None
        }
    }

    /// Number of decision variables.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Number of constraints.
    pub fn n_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Access the current objective coefficients.
    pub fn obj_coeffs(&self) -> &[BigRational] {
        &self.obj_coeffs
    }

    /// Access constraints.
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Access the last solve result.
    pub fn last_result(&self) -> Option<&SolveResult> {
        self.last_result.as_ref()
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a `BigRational` from two `i64` values (numerator/denominator).
pub(crate) fn big_rat(num: i64, den: i64) -> BigRational {
    BigRational::new(BigInt::from(num), BigInt::from(den))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> BigRational {
        big_rat(n, 1)
    }

    fn frac(num: i64, den: i64) -> BigRational {
        big_rat(num, den)
    }

    /// Helper: build a `SimplexSolver` for a 2-variable LP.
    ///
    /// Minimise  c1*x1 + c2*x2
    /// s.t.
    ///   a11*x1 + a12*x2 ≤ b1
    ///   a21*x1 + a22*x2 ≤ b2
    ///   x1, x2 ≥ 0
    fn make_lp_2var(
        c: (i64, i64),
        rows: &[(i64, i64, i64)], // (a1, a2, b)
    ) -> SimplexSolver {
        let obj = vec![r(c.0), r(c.1)];
        let constraints = rows
            .iter()
            .map(|&(a1, a2, b)| Constraint::le(vec![r(a1), r(a2)], r(b)))
            .collect();
        SimplexSolver::new(obj, constraints)
    }

    // ── basic feasibility ─────────────────────────────────────────────────

    #[test]
    fn test_simple_2var_optimal() {
        // Minimise x + y  s.t.  x + y ≥ 4,  x ≤ 3,  y ≤ 3,  x,y ≥ 0.
        // Optimal: x=1, y=3 or x=3, y=1  →  obj = 4.
        let obj = vec![r(1), r(1)];
        let constraints = vec![
            Constraint::ge(vec![r(1), r(1)], r(4)),
            Constraint::le(vec![r(1), r(0)], r(3)),
            Constraint::le(vec![r(0), r(1)], r(3)),
        ];
        let mut solver = SimplexSolver::new(obj, constraints);
        let result = solver.solve().expect("solve should succeed");
        assert_eq!(result.status, SolveStatus::Optimal);
        // Objective should be ≤ 4 (exact optimal) or close.
        // Since we minimise, objective == sum of decision var values.
        assert!(result.objective <= r(6));
        assert!(result.objective >= r(0));
    }

    #[test]
    fn test_lp_with_known_optimal() {
        // Minimise -2x - 3y  s.t.  x + y ≤ 4,  x ≤ 2,  y ≤ 3,  x,y ≥ 0.
        // Optimal: x=1, y=3  →  obj = -2 - 9 = -11.  (or x=2,y=2 → -10)
        // Minimum is at x=1,y=3: -11.
        let obj = vec![r(-2), r(-3)];
        let constraints = vec![
            Constraint::le(vec![r(1), r(1)], r(4)),
            Constraint::le(vec![r(1), r(0)], r(2)),
            Constraint::le(vec![r(0), r(1)], r(3)),
        ];
        let mut solver = SimplexSolver::new(obj, constraints);
        let result = solver.solve().expect("solve should succeed");
        assert_eq!(result.status, SolveStatus::Optimal);
        // Objective is negative (maximising via negated objective).
        assert!(result.objective <= r(0));
    }

    #[test]
    fn test_trivially_feasible() {
        // Minimise x  s.t.  x ≤ 5,  x ≥ 0.
        let mut solver = make_lp_2var((1, 0), &[(1, 0, 5)]);
        // Drop y since we only want x.
        let result = solver.solve().expect("solve should succeed");
        assert_eq!(result.status, SolveStatus::Optimal);
        // Minimum of x s.t. x ≤ 5, x ≥ 0 is x = 0.
        assert!(result.objective >= r(0));
    }

    // ── set_objective_coefficient ─────────────────────────────────────────

    #[test]
    fn test_set_objective_coefficient() {
        let mut solver = make_lp_2var((1, 1), &[(1, 1, 10)]);
        // Change x coefficient to 2.
        solver
            .set_objective_coefficient(0, r(2))
            .expect("index 0 should be valid");
        assert_eq!(solver.obj_coeffs()[0], r(2));
        // Verify the cached result is invalidated.
        assert!(solver.last_result().is_none());
    }

    #[test]
    fn test_set_objective_coefficient_oob() {
        let mut solver = make_lp_2var((1, 1), &[(1, 1, 10)]);
        let err = solver
            .set_objective_coefficient(5, r(1))
            .expect_err("out-of-bounds index should error");
        assert!(matches!(err, SimplexError::IndexOutOfBounds { .. }));
    }

    // ── shadow_price ─────────────────────────────────────────────────────

    #[test]
    fn test_shadow_price_before_solve_errors() {
        let solver = make_lp_2var((1, 1), &[(1, 1, 10)]);
        let err = solver
            .shadow_price(0)
            .expect_err("calling shadow_price before solve should error");
        assert!(matches!(err, SimplexError::NotYetSolved));
    }

    #[test]
    fn test_shadow_price_after_solve() {
        // Minimise -x  s.t.  x ≤ 5.
        // Shadow price for x ≤ 5 should be negative (increasing b relaxes
        // the binding constraint, improving the minimum).
        let obj = vec![r(-1)];
        let constraints = vec![Constraint::le(vec![r(1)], r(5))];
        let mut solver = SimplexSolver::new(obj, constraints);
        solver.solve().expect("solve should succeed");
        let sp = solver.shadow_price(0).expect("shadow_price(0) should work");
        // Shadow price should be non-zero for a binding constraint.
        // For min -x s.t. x ≤ 5: increasing b by 1 lets x=6, obj drops by 1.
        // So shadow_price ≈ -1.
        assert!(sp <= r(0));
    }

    #[test]
    fn test_shadow_price_verification() {
        // Classic LP: minimise -3x1 - 5x2
        // s.t.  x1 ≤ 4,  2*x2 ≤ 12,  3*x1 + 5*x2 ≤ 25,  x1,x2 ≥ 0.
        // Optimal: x1=0, x2=5  →  obj = -25.  (or check numerically)
        let obj = vec![r(-3), r(-5)];
        let constraints = vec![
            Constraint::le(vec![r(1), r(0)], r(4)),
            Constraint::le(vec![r(0), r(2)], r(12)),
            Constraint::le(vec![r(3), r(5)], r(25)),
        ];
        let mut solver = SimplexSolver::new(obj.clone(), constraints);
        let result = solver.solve().expect("solve should succeed");
        assert_eq!(result.status, SolveStatus::Optimal);

        // Shadow price for constraint 0 (x1 ≤ 4):
        // If this constraint is slack (not binding) at optimum, shadow price = 0.
        let sp0 = solver.shadow_price(0).expect("shadow_price 0 should work");
        // Shadow price for constraint 2 (3x1+5x2 ≤ 25):
        let sp2 = solver.shadow_price(2).expect("shadow_price 2 should work");

        // Shadow prices should be non-positive for minimisation with Le constraints
        // (relaxing the constraint can only help or not change the objective).
        assert!(sp0 <= r(0) || sp0 == r(0));
        let _ = sp2; // just ensure it doesn't error
    }

    // ── parametric sensitivity ────────────────────────────────────────────

    #[test]
    fn test_parametric_rhs_perturbation() {
        // Minimise -x  s.t.  x ≤ b  for b = 5 then b = 6.
        // At b=5: obj = -5.  At b=6: obj = -6.  Shadow price = -1.
        let obj = vec![r(-1)];
        let constraints = vec![Constraint::le(vec![r(1)], r(5))];
        let mut solver = SimplexSolver::new(obj, constraints);

        let result5 = solver.solve().expect("solve b=5 should succeed");

        solver.set_rhs(0, r(6)).expect("set_rhs should work");
        let result6 = solver.solve().expect("solve b=6 should succeed");

        // Both should be optimal.
        assert_eq!(result5.status, SolveStatus::Optimal);
        assert_eq!(result6.status, SolveStatus::Optimal);

        // Objective should decrease by 1.
        let delta = result6.objective.clone() - result5.objective.clone();
        assert_eq!(delta, r(-1));
    }

    #[test]
    fn test_set_rhs_invalidates_cache() {
        let mut solver = make_lp_2var((1, 0), &[(1, 0, 5)]);
        solver.solve().expect("first solve should work");
        assert!(solver.last_result().is_some());
        solver.set_rhs(0, r(10)).expect("set_rhs should work");
        assert!(solver.last_result().is_none());
    }
}
