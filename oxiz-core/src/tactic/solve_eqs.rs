//! Equation solving tactics.

use super::core::*;
use super::probe::Probe;
use crate::ast::normal_forms::{to_cnf_tseitin, to_nnf};
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
#[allow(unused_imports)]
use crate::prelude::*;
use num_integer::Integer;
use num_rational::Rational64;
use num_traits::Signed;

/// Equation solving tactic using Gaussian elimination
pub struct SolveEqsTactic<'a> {
    manager: &'a mut TermManager,
    /// Maximum iterations for solving
    max_iterations: usize,
}

impl<'a> SolveEqsTactic<'a> {
    /// Create a new solve-eqs tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            max_iterations: 100,
        }
    }

    /// Create with custom max iterations
    pub fn with_max_iterations(manager: &'a mut TermManager, max_iterations: usize) -> Self {
        Self {
            manager,
            max_iterations,
        }
    }

    /// Check if a variable appears in a term
    fn var_occurs_in(&self, var: TermId, term: TermId) -> bool {
        use crate::ast::traversal::contains_term;
        contains_term(term, var, self.manager)
    }

    /// Apply Gaussian elimination to solve equations.
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        Ok(self.apply_mut_with_converter(goal)?.0)
    }

    /// Apply Gaussian elimination, additionally returning a [`ModelConverter`]
    /// that reconstructs the eliminated variables' values from a model of the
    /// transformed sub-goal.
    ///
    /// Each solved equation `x = expr` removes `x` and substitutes `expr` for
    /// it. Given a model over the surviving variables, the converter recovers
    /// each eliminated `x` by evaluating its `expr` under the model (in
    /// reverse elimination order, since an earlier-eliminated variable's
    /// defining expression may reference a later-eliminated one). Returns
    /// `Some(converter)` only for the `SubGoals` transformation case.
    pub fn apply_mut_with_converter(
        &mut self,
        goal: &Goal,
    ) -> Result<(TacticResult, Option<Box<dyn ModelConverter>>)> {
        if goal.assertions.is_empty() {
            return Ok((TacticResult::NotApplicable, None));
        }

        let mut current_assertions = goal.assertions.clone();
        let mut total_changed = false;
        // Ordered record of (eliminated_var, defining_expr) for the converter.
        let mut eliminations: Vec<(TermId, TermId)> = Vec::new();

        for _ in 0..self.max_iterations {
            let mut iteration_changed = false;
            let mut solved_equation_index = None;
            let mut solution: Option<(TermId, TermId)> = None;

            // Find an equation we can solve
            for (i, &assertion) in current_assertions.iter().enumerate() {
                if let Some((var, expr)) = self.try_solve_equality_concrete(assertion) {
                    solved_equation_index = Some(i);
                    solution = Some((var, expr));
                    break;
                }
            }

            // If we found a solvable equation, apply the substitution
            if let (Some(idx), Some((var, expr))) = (solved_equation_index, solution) {
                iteration_changed = true;
                total_changed = true;
                eliminations.push((var, expr));

                // Build substitution map
                let mut subst = crate::prelude::FxHashMap::default();
                subst.insert(var, expr);

                // Apply substitution to all assertions except the solved one
                let mut new_assertions: Vec<TermId> = Vec::with_capacity(current_assertions.len());

                for (i, &assertion) in current_assertions.iter().enumerate() {
                    if i == idx {
                        // Skip the solved equation (it's now redundant)
                        continue;
                    }

                    let substituted = self.manager.substitute(assertion, &subst);
                    let simplified = self.manager.simplify(substituted);
                    new_assertions.push(simplified);
                }

                current_assertions = new_assertions;
            }

            if !iteration_changed {
                break;
            }
        }

        if !total_changed {
            return Ok((TacticResult::NotApplicable, None));
        }

        // Check for trivially true/false
        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        // Check if any assertion is false
        if current_assertions.contains(&false_id) {
            return Ok((TacticResult::Solved(SolveResult::Unsat), None));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = current_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        // If all assertions are true, goal is SAT
        if filtered.is_empty() {
            return Ok((TacticResult::Solved(SolveResult::Sat), None));
        }

        let converter: Box<dyn ModelConverter> = Box::new(SolveEqsConverter { eliminations });

        Ok((
            TacticResult::SubGoals(vec![Goal {
                assertions: filtered,
                precision: goal.precision,
            }]),
            Some(converter),
        ))
    }

    /// Concrete implementation that returns actual TermIds (not pending computations)
    fn try_solve_equality_concrete(&mut self, eq_term: TermId) -> Option<(TermId, TermId)> {
        let term = self.manager.get(eq_term)?;

        if let TermKind::Eq(lhs, rhs) = term.kind {
            let lhs_kind = self.manager.get(lhs).map(|t| t.kind.clone());
            let rhs_kind = self.manager.get(rhs).map(|t| t.kind.clone());

            // Case 1: x = expr (where x doesn't appear in expr)
            if let Some(TermKind::Var(_)) = &lhs_kind
                && !self.var_occurs_in(lhs, rhs)
            {
                return Some((lhs, rhs));
            }

            // Case 2: expr = x (where x doesn't appear in expr)
            if let Some(TermKind::Var(_)) = &rhs_kind
                && !self.var_occurs_in(rhs, lhs)
            {
                return Some((rhs, lhs));
            }

            // Case 3: Handle linear addition: (x + a) = b => x = b - a
            if let Some(TermKind::Add(args)) = lhs_kind.clone()
                && let Some((var, result)) = self.solve_linear_add_concrete(&args, rhs)
            {
                return Some((var, result));
            }

            // Case 4: Handle subtraction: (x - a) = b => x = b + a
            if let Some(TermKind::Sub(minuend, subtrahend)) = lhs_kind
                && let Some((var, result)) =
                    self.solve_linear_sub_concrete(minuend, subtrahend, rhs)
            {
                return Some((var, result));
            }
        }

        None
    }

    /// Solve (x + a1 + a2 + ...) = b => x = b - a1 - a2 - ...
    fn solve_linear_add_concrete(
        &mut self,
        args: &smallvec::SmallVec<[TermId; 4]>,
        rhs: TermId,
    ) -> Option<(TermId, TermId)> {
        for (i, &arg) in args.iter().enumerate() {
            let arg_term = self.manager.get(arg)?;
            if let TermKind::Var(_) = &arg_term.kind {
                // Collect other arguments
                let other_args: smallvec::SmallVec<[TermId; 4]> = args
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &a)| a)
                    .collect();

                // Check if variable doesn't appear in other args or rhs
                let var_in_others = other_args.iter().any(|&a| self.var_occurs_in(arg, a));
                let var_in_rhs = self.var_occurs_in(arg, rhs);

                if !var_in_others && !var_in_rhs {
                    // Compute result: rhs - sum(other_args)
                    let result = if other_args.is_empty() {
                        rhs
                    } else if other_args.len() == 1 {
                        self.manager.mk_sub(rhs, other_args[0])
                    } else {
                        // rhs - (a1 + a2 + ...)
                        let sum = self.manager.mk_add(other_args);
                        self.manager.mk_sub(rhs, sum)
                    };

                    return Some((arg, result));
                }
            }
        }
        None
    }

    /// Solve (x - a) = b => x = b + a
    fn solve_linear_sub_concrete(
        &mut self,
        minuend: TermId,
        subtrahend: TermId,
        rhs: TermId,
    ) -> Option<(TermId, TermId)> {
        let minuend_term = self.manager.get(minuend)?;

        if let TermKind::Var(_) = &minuend_term.kind
            && !self.var_occurs_in(minuend, subtrahend)
            && !self.var_occurs_in(minuend, rhs)
        {
            // x = rhs + subtrahend
            let result = self.manager.mk_add([rhs, subtrahend]);
            return Some((minuend, result));
        }
        None
    }
}

/// Model converter for [`SolveEqsTactic`].
///
/// Stores the ordered `(eliminated_var, defining_expr)` pairs produced by the
/// tactic. To convert a model of the transformed sub-goal into a model of the
/// original goal, it evaluates each defining expression under the current
/// (progressively extended) model in **reverse** elimination order: the first
/// variable eliminated may be defined in terms of a variable eliminated later,
/// so later ones must be reconstructed first.
struct SolveEqsConverter {
    eliminations: Vec<(TermId, TermId)>,
}

impl ModelConverter for SolveEqsConverter {
    fn convert(&self, model: &TacticModel, manager: &mut TermManager) -> TacticModel {
        let mut result = model.clone();
        for (var, expr) in self.eliminations.iter().rev() {
            // Substitute the known values into the defining expression and
            // simplify; with a complete model over the surviving variables
            // this yields the eliminated variable's concrete value.
            let substituted = manager.substitute(*expr, &result.values);
            let value = manager.simplify(substituted);
            result.values.insert(*var, value);
        }
        result
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessSolveEqsTactic;

impl Tactic for StatelessSolveEqsTactic {
    fn name(&self) -> &str {
        "solve-eqs"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // Gaussian elimination substitutes solved variables, which requires a
        // `&mut TermManager` to build the substituted terms. The manager-free
        // `Tactic::apply` cannot do this, so it honestly reports
        // NotApplicable rather than returning the goal unchanged as if it had
        // solved anything. Use `SolveEqsTactic::apply_mut` (or the registry's
        // `create_managed` path) for the real transformation.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Gaussian elimination for linear equations - solves x = expr and substitutes \
         (requires a TermManager; the manager-free path is NotApplicable)"
    }
}

// ============================================================================
// Fourier-Motzkin Elimination Tactic
// ============================================================================

/// Coefficient in a linear constraint
///
/// Represented as a pair of BigInts for exact rational arithmetic:
/// the coefficient is `numerator / denominator`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Coefficient {
    numerator: num_bigint::BigInt,
    denominator: num_bigint::BigInt,
}

impl Coefficient {
    fn zero() -> Self {
        Self {
            numerator: num_bigint::BigInt::ZERO,
            denominator: num_bigint::BigInt::from(1),
        }
    }

    fn one() -> Self {
        Self {
            numerator: num_bigint::BigInt::from(1),
            denominator: num_bigint::BigInt::from(1),
        }
    }

    fn from_int(n: impl Into<num_bigint::BigInt>) -> Self {
        Self {
            numerator: n.into(),
            denominator: num_bigint::BigInt::from(1),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator == num_bigint::BigInt::ZERO
    }

    fn is_positive(&self) -> bool {
        let sign = self.numerator.sign() * self.denominator.sign();
        sign == num_bigint::Sign::Plus && !self.is_zero()
    }

    fn is_negative(&self) -> bool {
        let sign = self.numerator.sign() * self.denominator.sign();
        sign == num_bigint::Sign::Minus
    }

    fn negate(&self) -> Self {
        Self {
            numerator: -&self.numerator,
            denominator: self.denominator.clone(),
        }
    }

    fn abs(&self) -> Self {
        Self {
            numerator: num_bigint::BigInt::from(self.numerator.magnitude().clone()),
            denominator: num_bigint::BigInt::from(self.denominator.magnitude().clone()),
        }
    }

    /// Whether this coefficient is exactly ±1 (a magnitude-one integer).
    ///
    /// Used by the integer-soundness guard in [`FourierMotzkinTactic`]: an
    /// integer variable is exactly eliminable (real shadow == integer
    /// projection) only when, for every lower/upper pair, its coefficient is a
    /// unit on at least one side — the Omega-test exact-shadow condition.
    fn is_unit_magnitude(&self) -> bool {
        self.denominator == num_bigint::BigInt::from(1)
            && Signed::abs(&self.numerator) == num_bigint::BigInt::from(1)
    }

    /// Multiply two coefficients
    fn multiply(&self, other: &Self) -> Self {
        let num = &self.numerator * &other.numerator;
        let den = &self.denominator * &other.denominator;
        Self::simplify(num, den)
    }

    /// Add two coefficients
    fn add(&self, other: &Self) -> Self {
        let num = &self.numerator * &other.denominator + &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;
        Self::simplify(num, den)
    }

    /// Simplify by dividing by GCD
    fn simplify(num: num_bigint::BigInt, den: num_bigint::BigInt) -> Self {
        if num == num_bigint::BigInt::ZERO {
            return Self::zero();
        }
        let g = Integer::gcd(&num, &den);
        let mut simplified_num = &num / &g;
        let mut simplified_den = &den / &g;
        // Ensure denominator is positive
        if simplified_den.sign() == num_bigint::Sign::Minus {
            simplified_num = -simplified_num;
            simplified_den = -simplified_den;
        }
        Self {
            numerator: simplified_num,
            denominator: simplified_den,
        }
    }
}

/// A linear constraint in the form: Σ aᵢxᵢ ≤ c (or < c if strict)
///
/// The constraint can also have boolean literals as a disjunctive prefix:
/// lit₁ ∨ lit₂ ∨ ... ∨ (Σ aᵢxᵢ ≤ c)
#[derive(Debug, Clone)]
struct LinearConstraint {
    /// Unique identifier (reserved for future use)
    #[allow(dead_code)]
    id: usize,
    /// Coefficients for each variable (indexed by variable id)
    /// Only non-zero coefficients are stored
    coefficients: crate::prelude::FxHashMap<TermId, Coefficient>,
    /// Right-hand side constant
    constant: Coefficient,
    /// Whether this is a strict inequality (<) or non-strict (≤)
    strict: bool,
    /// Boolean literals (disjunctive prefix)
    literals: smallvec::SmallVec<[TermId; 4]>,
    /// Whether this constraint has been eliminated
    dead: bool,
}

impl LinearConstraint {
    fn new(id: usize) -> Self {
        Self {
            id,
            coefficients: crate::prelude::FxHashMap::default(),
            constant: Coefficient::zero(),
            strict: false,
            literals: smallvec::SmallVec::new(),
            dead: false,
        }
    }

    fn get_coeff(&self, var: TermId) -> Coefficient {
        self.coefficients
            .get(&var)
            .cloned()
            .unwrap_or_else(Coefficient::zero)
    }

    fn set_coeff(&mut self, var: TermId, coeff: Coefficient) {
        if coeff.is_zero() {
            self.coefficients.remove(&var);
        } else {
            self.coefficients.insert(var, coeff);
        }
    }

    /// Check if this constraint has no variables (is a constant bound)
    fn is_constant(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Check if this is a trivially true constraint (0 ≤ c where c > 0)
    fn is_tautology(&self) -> bool {
        if !self.is_constant() {
            return false;
        }
        // 0 ≤ c is true if c ≥ 0
        // 0 < c is true if c > 0
        if self.strict {
            self.constant.is_positive()
        } else {
            !self.constant.is_negative()
        }
    }

    /// Check if this is a trivially false constraint (0 ≤ c where c < 0)
    fn is_contradiction(&self) -> bool {
        if !self.is_constant() {
            return false;
        }
        // 0 ≤ c is false if c < 0
        // 0 < c is false if c ≤ 0
        if self.strict {
            !self.constant.is_positive()
        } else {
            self.constant.is_negative()
        }
    }

    /// Get all variables with non-zero coefficients
    fn variables(&self) -> impl Iterator<Item = TermId> + '_ {
        self.coefficients.keys().copied()
    }

    /// Normalize coefficients by dividing by their GCD
    fn normalize(&mut self) {
        if self.coefficients.is_empty() {
            return;
        }

        // Collect all numerators (convert to common denominator first)
        let mut lcm = num_bigint::BigInt::from(1);
        for coeff in self.coefficients.values() {
            lcm = Integer::lcm(&lcm, &coeff.denominator);
        }
        lcm = Integer::lcm(&lcm, &self.constant.denominator);

        // Convert to integers using the LCM
        let mut int_coeffs: Vec<num_bigint::BigInt> = self
            .coefficients
            .values()
            .map(|c| &c.numerator * (&lcm / &c.denominator))
            .collect();
        let int_const = &self.constant.numerator * (&lcm / &self.constant.denominator);
        int_coeffs.push(int_const.clone());

        // Find GCD of all
        let mut gcd = Signed::abs(&int_coeffs[0]);
        for c in &int_coeffs[1..] {
            gcd = Integer::gcd(&gcd, &Signed::abs(c));
        }

        if gcd > num_bigint::BigInt::from(1) {
            // Divide all coefficients by GCD
            for coeff in self.coefficients.values_mut() {
                let int_val = &coeff.numerator * (&lcm / &coeff.denominator);
                let new_val = &int_val / &gcd;
                *coeff = Coefficient::from_int(new_val);
            }
            let new_const = &int_const / &gcd;
            self.constant = Coefficient::from_int(new_const);
        }
    }
}

/// Fourier-Motzkin elimination tactic for linear real arithmetic
///
/// This tactic performs quantifier elimination for linear real arithmetic
/// by systematically eliminating variables through pairwise resolution of
/// upper and lower bounds.
///
/// ## Algorithm
///
/// For each variable x to eliminate:
/// 1. Collect lower bounds: constraints where x has negative coefficient (after isolating x)
/// 2. Collect upper bounds: constraints where x has positive coefficient
/// 3. For each pair (lower, upper), resolve to eliminate x
/// 4. Remove original constraints involving x, add resolved constraints
///
/// ## Complexity
///
/// - Worst case: O(n²) new constraints per variable elimination
/// - Uses cutoff parameters to avoid exponential blowup
///
/// ## Limitations
///
/// - Currently only handles real arithmetic (not integer)
/// - May not terminate for very dense constraint systems
#[derive(Debug)]
pub struct FourierMotzkinTactic<'a> {
    manager: &'a mut TermManager,
    /// Maximum number of operations before giving up
    op_limit: usize,
    /// Current operation count
    op_count: usize,
    /// Cutoff: skip if |lowers| > cutoff1 AND |uppers| > cutoff1
    cutoff1: usize,
    /// Cutoff: skip if |lowers| × |uppers| > cutoff2
    cutoff2: usize,
    /// Next constraint ID
    next_constraint_id: usize,
}

impl<'a> FourierMotzkinTactic<'a> {
    /// Create a new Fourier-Motzkin elimination tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            op_limit: 500_000,
            op_count: 0,
            cutoff1: 8,
            cutoff2: 256,
            next_constraint_id: 0,
        }
    }

    /// Set the operation limit
    pub fn with_op_limit(mut self, limit: usize) -> Self {
        self.op_limit = limit;
        self
    }

    /// Set the cutoff parameters
    pub fn with_cutoffs(mut self, cutoff1: usize, cutoff2: usize) -> Self {
        self.cutoff1 = cutoff1;
        self.cutoff2 = cutoff2;
        self
    }

    /// Apply Fourier-Motzkin elimination to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        // Phase 1: Extract linear constraints from goal
        let mut constraints = Vec::new();
        let mut non_linear = Vec::new();

        for &assertion in &goal.assertions {
            match self.extract_constraint(assertion) {
                Some(c) => constraints.push(c),
                None => non_linear.push(assertion),
            }
        }

        if constraints.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 2: Collect all variables
        let mut all_vars: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();
        for c in &constraints {
            for var in c.variables() {
                all_vars.insert(var);
            }
        }

        // Phase 3: Build variable bounds index
        let mut lowers: crate::prelude::FxHashMap<TermId, Vec<usize>> =
            crate::prelude::FxHashMap::default();
        let mut uppers: crate::prelude::FxHashMap<TermId, Vec<usize>> =
            crate::prelude::FxHashMap::default();

        for (idx, c) in constraints.iter().enumerate() {
            for var in c.variables() {
                let coeff = c.get_coeff(var);
                if coeff.is_negative() {
                    lowers.entry(var).or_default().push(idx);
                } else if coeff.is_positive() {
                    uppers.entry(var).or_default().push(idx);
                }
            }
        }

        // Phase 4: Eliminate variables
        let mut eliminated_any = false;

        // Real-sorted variables are exactly eliminable by Fourier-Motzkin;
        // any other sort (Int, or an unexpected sort) requires the
        // integer-exactness guard below before a pairwise elimination is
        // sound (P4-1104).
        let real_sort = self.manager.sorts.real_sort;

        // Sort variables by elimination cost
        let mut vars_to_eliminate: Vec<_> = all_vars.iter().copied().collect();
        vars_to_eliminate.sort_by_key(|&var| {
            let l = lowers.get(&var).map_or(0, |v| v.len());
            let u = uppers.get(&var).map_or(0, |v| v.len());
            l * u
        });

        for var in vars_to_eliminate {
            if self.op_count >= self.op_limit {
                break;
            }

            let lower_indices = lowers.get(&var).map_or(&[] as &[usize], |v| v.as_slice());
            let upper_indices = uppers.get(&var).map_or(&[] as &[usize], |v| v.as_slice());

            // Apply cutoffs
            if lower_indices.len() > self.cutoff1 && upper_indices.len() > self.cutoff1 {
                continue;
            }
            if lower_indices.len() * upper_indices.len() > self.cutoff2 {
                continue;
            }

            // Trivial elimination: no lower or upper bounds.
            //
            // A variable that is unbounded below (only upper bounds) or above
            // (only lower bounds) can always be given a value satisfying its
            // constraints — over the reals *and* over the integers — so
            // dropping those constraints is a sound projection for both. No
            // integer-exactness guard is needed here.
            if lower_indices.is_empty() || upper_indices.is_empty() {
                // Mark constraints involving this variable as dead
                for &idx in lower_indices.iter().chain(upper_indices.iter()) {
                    constraints[idx].dead = true;
                }
                eliminated_any = true;
                continue;
            }

            // Integer-soundness guard (P4-1104): Fourier-Motzkin resolution
            // computes the *real* shadow of the projection. For an integer
            // variable, the real shadow coincides with the exact integer
            // projection only when every (lower, upper) pair has a unit (±1)
            // coefficient on at least one side (the Omega-test exact-shadow
            // condition). If some pair has |coeff| > 1 on both sides, the real
            // shadow is a strict over-approximation of the integer projection,
            // so eliminating the variable would be unsound — e.g. the system
            // 1 <= 2x <= 1 is real-feasible (x = 1/2) but integer-infeasible,
            // and eliminating x used to collapse it to `true` and report Sat.
            // In that case we skip this variable, leaving its constraints
            // intact rather than fabricating an exact integer projection.
            let requires_integral = self.manager.get(var).is_none_or(|t| t.sort != real_sort);
            if requires_integral
                && !Self::integer_elimination_exact(&constraints, var, lower_indices, upper_indices)
            {
                continue;
            }

            // Perform pairwise resolution
            let mut new_constraints = Vec::new();
            let mut op_limit_hit = false;

            'pairs: for &lower_idx in lower_indices {
                for &upper_idx in upper_indices {
                    self.op_count += 1;
                    if self.op_count >= self.op_limit {
                        // Abort the elimination for this variable before all
                        // pairs have been resolved. Do NOT mark the original
                        // constraints dead below: discarding them here would
                        // over-approximate the system (and could hide a
                        // contradiction among the unresolved pairs), turning a
                        // resource limit into an unsound result. Instead we
                        // leave this variable's constraints untouched and stop
                        // eliminating further variables.
                        op_limit_hit = true;
                        break 'pairs;
                    }

                    if constraints[lower_idx].dead || constraints[upper_idx].dead {
                        continue;
                    }

                    if let Some(resolved) =
                        self.resolve(&constraints[lower_idx], &constraints[upper_idx], var)
                    {
                        // Check for contradiction
                        if resolved.is_contradiction() {
                            return Ok(TacticResult::Solved(SolveResult::Unsat));
                        }

                        // Skip tautologies
                        if resolved.is_tautology() {
                            continue;
                        }

                        new_constraints.push(resolved);
                    }
                }
            }

            if op_limit_hit {
                // Keep this variable's original (lower/upper) constraints
                // alive and stop the elimination loop entirely: the op
                // budget is exhausted, so further variables would hit the
                // same guard on the next outer-loop iteration anyway.
                break;
            }

            // Mark old constraints as dead
            for &idx in lower_indices.iter().chain(upper_indices.iter()) {
                constraints[idx].dead = true;
            }

            // Add new constraints
            constraints.extend(new_constraints);
            eliminated_any = true;

            // Rebuild index for remaining variables
            lowers.clear();
            uppers.clear();
            for (idx, c) in constraints.iter().enumerate() {
                if c.dead {
                    continue;
                }
                for v in c.variables() {
                    let coeff = c.get_coeff(v);
                    if coeff.is_negative() {
                        lowers.entry(v).or_default().push(idx);
                    } else if coeff.is_positive() {
                        uppers.entry(v).or_default().push(idx);
                    }
                }
            }
        }

        if !eliminated_any {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 5: Convert remaining constraints back to assertions
        let mut new_assertions = non_linear;

        for c in &constraints {
            if c.dead {
                continue;
            }
            if c.is_tautology() {
                continue;
            }
            if let Some(term) = self.constraint_to_term(c) {
                new_assertions.push(term);
            }
        }

        // Check if all constraints eliminated to true
        if new_assertions.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Whether eliminating the integer variable `var` by pairwise resolution
    /// is *exact* over the integers.
    ///
    /// Returns `true` iff for every live (lower, upper) pair the coefficient
    /// of `var` is a unit (±1) on at least one side. Under that condition the
    /// Omega-test dark shadow equals the real shadow, so Fourier-Motzkin's
    /// real projection coincides with the exact integer projection and the
    /// elimination preserves (integer) equisatisfiability. If any pair has
    /// |coeff| > 1 on both sides, the elimination is a strict
    /// over-approximation and must be skipped (P4-1104).
    fn integer_elimination_exact(
        constraints: &[LinearConstraint],
        var: TermId,
        lowers: &[usize],
        uppers: &[usize],
    ) -> bool {
        for &li in lowers {
            let lc = &constraints[li];
            if lc.dead {
                continue;
            }
            let lower_is_unit = lc.get_coeff(var).is_unit_magnitude();
            for &ui in uppers {
                let uc = &constraints[ui];
                if uc.dead {
                    continue;
                }
                // The pair is exact iff at least one side has a unit
                // coefficient on `var`.
                if !lower_is_unit && !uc.get_coeff(var).is_unit_magnitude() {
                    return false;
                }
            }
        }
        true
    }

    /// Extract a linear constraint from a term
    fn extract_constraint(&mut self, term_id: TermId) -> Option<LinearConstraint> {
        let term = self.manager.get(term_id)?;

        match &term.kind {
            // a ≤ b  →  a - b ≤ 0
            TermKind::Le(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = false;
                self.extract_linear(*lhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*rhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a < b  →  a - b < 0
            TermKind::Lt(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = true;
                self.extract_linear(*lhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*rhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a ≥ b  →  b - a ≤ 0
            TermKind::Ge(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = false;
                self.extract_linear(*rhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*lhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a > b  →  b - a < 0
            TermKind::Gt(lhs, rhs) => {
                let mut c = LinearConstraint::new(self.next_constraint_id);
                self.next_constraint_id += 1;
                c.strict = true;
                self.extract_linear(*rhs, &Coefficient::one(), &mut c)?;
                self.extract_linear(*lhs, &Coefficient::one().negate(), &mut c)?;
                Some(c)
            }

            // a = b  →  a ≤ b ∧ b ≤ a (not directly handled as a single constraint)
            // For FM, we'd need to split equalities, which we skip for simplicity
            _ => None,
        }
    }

    /// Extract linear terms recursively
    /// Returns None if the term is not linear
    fn extract_linear(
        &self,
        term_id: TermId,
        scale: &Coefficient,
        constraint: &mut LinearConstraint,
    ) -> Option<()> {
        let term = self.manager.get(term_id)?;

        match &term.kind {
            // Integer constant
            TermKind::IntConst(n) => {
                let coeff = Coefficient::from_int(n.clone()).multiply(scale);
                constraint.constant = constraint.constant.add(&coeff.negate());
                Some(())
            }

            // Rational constant
            TermKind::RealConst(r) => {
                let coeff = Coefficient {
                    numerator: num_bigint::BigInt::from(*r.numer()),
                    denominator: num_bigint::BigInt::from(*r.denom()),
                }
                .multiply(scale);
                constraint.constant = constraint.constant.add(&coeff.negate());
                Some(())
            }

            // Variable
            TermKind::Var(_) => {
                let existing = constraint.get_coeff(term_id);
                constraint.set_coeff(term_id, existing.add(scale));
                Some(())
            }

            // Addition
            TermKind::Add(args) => {
                for &arg in args {
                    self.extract_linear(arg, scale, constraint)?;
                }
                Some(())
            }

            // Subtraction
            TermKind::Sub(lhs, rhs) => {
                self.extract_linear(*lhs, scale, constraint)?;
                self.extract_linear(*rhs, &scale.negate(), constraint)?;
                Some(())
            }

            // Negation
            TermKind::Neg(arg) => self.extract_linear(*arg, &scale.negate(), constraint),

            // Multiplication by constant
            TermKind::Mul(args) => {
                // Check if all but one are constants
                let mut const_part = Coefficient::one();
                let mut var_part = None;

                for &arg in args {
                    let arg_term = self.manager.get(arg)?;
                    match &arg_term.kind {
                        TermKind::IntConst(n) => {
                            const_part = const_part.multiply(&Coefficient::from_int(n.clone()));
                        }
                        TermKind::RealConst(r) => {
                            const_part = const_part.multiply(&Coefficient {
                                numerator: num_bigint::BigInt::from(*r.numer()),
                                denominator: num_bigint::BigInt::from(*r.denom()),
                            });
                        }
                        _ => {
                            if var_part.is_some() {
                                // Multiple non-constant terms - not linear
                                return None;
                            }
                            var_part = Some(arg);
                        }
                    }
                }

                let new_scale = scale.multiply(&const_part);
                match var_part {
                    Some(v) => self.extract_linear(v, &new_scale, constraint),
                    None => {
                        // All constants
                        constraint.constant = constraint.constant.add(&new_scale.negate());
                        Some(())
                    }
                }
            }

            // Not linear
            _ => None,
        }
    }

    /// Resolve two constraints to eliminate a variable
    ///
    /// Given:
    /// - lower: ... + (-a)*x + ... ≤ c₁  (a > 0, so x ≥ (... - c₁) / a)
    /// - upper: ... + b*x + ... ≤ c₂     (b > 0, so x ≤ (c₂ - ...) / b)
    ///
    /// Produces: b*(lower_without_x) + a*(upper_without_x) ≤ b*c₁ + a*c₂
    fn resolve(
        &mut self,
        lower: &LinearConstraint,
        upper: &LinearConstraint,
        var: TermId,
    ) -> Option<LinearConstraint> {
        let coeff_l = lower.get_coeff(var); // negative
        let coeff_u = upper.get_coeff(var); // positive

        if !coeff_l.is_negative() || !coeff_u.is_positive() {
            return None;
        }

        let abs_a = coeff_l.abs();
        let b = coeff_u.clone();

        let mut result = LinearConstraint::new(self.next_constraint_id);
        self.next_constraint_id += 1;

        // Combine strictness
        result.strict = lower.strict || upper.strict;

        // Combine literals (union)
        result.literals.extend(lower.literals.iter().copied());
        for lit in &upper.literals {
            if !result.literals.contains(lit) {
                result.literals.push(*lit);
            }
        }

        // Combine coefficients: b * (lower coeffs) + a * (upper coeffs)
        // but skip the eliminated variable
        for (&v, coeff) in &lower.coefficients {
            if v == var {
                continue;
            }
            let scaled = coeff.multiply(&b);
            let existing = result.get_coeff(v);
            result.set_coeff(v, existing.add(&scaled));
        }

        for (&v, coeff) in &upper.coefficients {
            if v == var {
                continue;
            }
            let scaled = coeff.multiply(&abs_a);
            let existing = result.get_coeff(v);
            result.set_coeff(v, existing.add(&scaled));
        }

        // Combine constants: b * c₁ + a * c₂
        let scaled_c1 = lower.constant.multiply(&b);
        let scaled_c2 = upper.constant.multiply(&abs_a);
        result.constant = scaled_c1.add(&scaled_c2);

        // Normalize
        result.normalize();

        Some(result)
    }

    /// Convert a constraint back to a term
    fn constraint_to_term(&mut self, c: &LinearConstraint) -> Option<TermId> {
        if c.coefficients.is_empty() {
            // Constant constraint - already checked for tautology/contradiction
            return None;
        }

        // Build: Σ aᵢxᵢ ≤/< c
        // Which is: Σ aᵢxᵢ - c ≤/< 0
        // Rearranged: sum ≤/< -constant (since we stored as Σ - c ≤ 0)

        let mut positive_terms: Vec<TermId> = Vec::new();
        let mut negative_terms: Vec<TermId> = Vec::new();

        for (&var, coeff) in &c.coefficients {
            if coeff.is_zero() {
                continue;
            }

            // Handle coefficient
            let term = if Signed::abs(&coeff.numerator) == num_bigint::BigInt::from(1)
                && coeff.denominator == num_bigint::BigInt::from(1)
            {
                // Just the variable (possibly negated)
                if coeff.is_negative() {
                    negative_terms.push(var);
                    continue;
                } else {
                    var
                }
            } else {
                // c * var
                let abs_coeff = self.coeff_to_term(&coeff.abs());
                self.manager.mk_mul([abs_coeff, var])
            };

            if coeff.is_positive() {
                positive_terms.push(term);
            } else {
                negative_terms.push(term);
            }
        }

        // Build LHS sum
        let lhs = if positive_terms.is_empty() && negative_terms.is_empty() {
            self.manager.mk_int(0)
        } else if positive_terms.is_empty() {
            // All negative: return -(a + b + ...)
            let sum = if negative_terms.len() == 1 {
                negative_terms[0]
            } else {
                self.manager.mk_add(negative_terms)
            };
            self.manager.mk_neg(sum)
        } else if negative_terms.is_empty() {
            if positive_terms.len() == 1 {
                positive_terms[0]
            } else {
                self.manager.mk_add(positive_terms)
            }
        } else {
            // Mix: (pos_sum) - (neg_sum)
            let pos_sum = if positive_terms.len() == 1 {
                positive_terms[0]
            } else {
                self.manager.mk_add(positive_terms)
            };
            let neg_sum = if negative_terms.len() == 1 {
                negative_terms[0]
            } else {
                self.manager.mk_add(negative_terms)
            };
            self.manager.mk_sub(pos_sum, neg_sum)
        };

        // RHS is -constant (since we stored as Σ coeff*x ≤ -constant)
        let rhs = self.coeff_to_term(&c.constant.negate());

        // Build inequality
        if c.strict {
            Some(self.manager.mk_lt(lhs, rhs))
        } else {
            Some(self.manager.mk_le(lhs, rhs))
        }
    }

    /// Convert a coefficient to a term
    fn coeff_to_term(&mut self, c: &Coefficient) -> TermId {
        if c.denominator == num_bigint::BigInt::from(1) {
            // Integer
            return self.manager.mk_int(c.numerator.clone());
        }

        // Non-integer coefficient: build the *exact* Real value rather than
        // truncating to an approximate integer, which silently changed the
        // constraint's semantics (e.g. a coefficient of 3/2 becoming the
        // integer 1). Reduce by the gcd first so a ratio whose raw
        // numerator/denominator don't individually fit in `i64` (the
        // backing type of `RealConst`'s `Rational64`) still has a chance to
        // after cancellation.
        let g = c.numerator.gcd(&c.denominator);
        let (n, d) = if g == num_bigint::BigInt::from(1) {
            (c.numerator.clone(), c.denominator.clone())
        } else {
            (&c.numerator / &g, &c.denominator / &g)
        };

        match (i64::try_from(&n), i64::try_from(&d)) {
            (Ok(n64), Ok(d64)) if d64 != 0 => self.manager.mk_real(Rational64::new(n64, d64)),
            _ => {
                // The AST's `RealConst` cannot represent a ratio this large
                // exactly (no arbitrary-precision Real constant exists in
                // this term representation, unlike `IntConst`'s `BigInt`).
                // Falling back to a truncated integer approximation here is
                // the same historical trade-off, but now only reached for
                // genuinely huge coefficients instead of any non-integer one.
                let approx = &c.numerator / &c.denominator;
                self.manager.mk_int(approx)
            }
        }
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessFourierMotzkinTactic;

impl Tactic for StatelessFourierMotzkinTactic {
    fn name(&self) -> &str {
        "fm"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // Fourier-Motzkin resolution builds new constraint terms, requiring a
        // `&mut TermManager`. The manager-free path honestly reports
        // NotApplicable instead of returning the goal unchanged. Use
        // `FourierMotzkinTactic::apply_mut` (or `create_managed`) for the real
        // elimination.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Fourier-Motzkin variable elimination for linear arithmetic \
         (requires a TermManager; the manager-free path is NotApplicable)"
    }
}

// ============================================================================
// Scriptable Tactic (Rhai-based)
// ============================================================================
//
// Everything from here to the end of `impl Tactic for ScriptableTactic` is
// behind the `scripting` Cargo feature (ON by default) -- it is the only code
// in the crate that reaches `rhai`. See oxiz-core/Cargo.toml for why that is
// its own feature rather than part of `std`.

/// A tactic that executes user-defined Rhai scripts
///
/// This allows users to create custom simplification strategies by writing
/// Rhai scripts. The scripts have access to goal assertions and can return
/// simplified goals or solve results.
///
/// Requires the `scripting` Cargo feature (enabled by default). With it off
/// this type does not exist; every other tactic in this module is unaffected.
///
/// # Example Script
///
/// ```rhai
/// // Check if goal has any false assertions
/// fn apply_script(assertions) {
///     for assertion in assertions {
///         if is_false(assertion) {
///             return #{result: "unsat"};
///         }
///     }
///     #{result: "unchanged", assertions: assertions}
/// }
/// ```
#[cfg(feature = "scripting")]
#[derive(Debug)]
pub struct ScriptableTactic {
    engine: rhai::Engine,
    script: String,
    name: String,
    description: String,
}

#[cfg(feature = "scripting")]
impl ScriptableTactic {
    /// Create a new scriptable tactic with the given Rhai script
    ///
    /// # Arguments
    ///
    /// * `name` - Name for this tactic
    /// * `script` - Rhai script that defines an `apply_script` function
    /// * `description` - Description of what this tactic does
    ///
    /// # Errors
    ///
    /// Returns an error if the script fails to compile
    pub fn new(name: String, script: String, description: String) -> Result<Self> {
        let mut engine = rhai::Engine::new();

        // Register helper functions that scripts can use
        Self::register_builtins(&mut engine);

        // Validate that the script compiles
        engine.compile(&script).map_err(|e| {
            crate::error::OxizError::Internal(format!("Script compilation failed: {}", e))
        })?;

        Ok(Self {
            engine,
            script,
            name,
            description,
        })
    }

    /// Register built-in helper functions for scripts
    fn register_builtins(engine: &mut rhai::Engine) {
        // Register helper to check if an assertion ID represents false
        // In a real implementation, this would need access to the term manager
        engine.register_fn("is_false", |_id: i64| -> bool {
            // Placeholder - in real usage, we'd check against TermManager
            false
        });

        // Register helper to check if an assertion ID represents true
        engine.register_fn("is_true", |_id: i64| -> bool {
            // Placeholder - in real usage, we'd check against TermManager
            false
        });

        // Register len getter for Vec<i64>
        engine.register_get("len", |arr: &mut Vec<i64>| -> i64 { arr.len() as i64 });

        // Register indexer for Vec<i64>
        engine.register_indexer_get(|arr: &mut Vec<i64>, idx: i64| -> i64 {
            arr.get(idx as usize).copied().unwrap_or(0)
        });

        // Register indexer setter for Vec<i64>
        engine.register_indexer_set(|arr: &mut Vec<i64>, idx: i64, value: i64| {
            if (idx as usize) < arr.len() {
                arr[idx as usize] = value;
            }
        });
    }

    /// Apply the script to a goal using a term manager for context
    ///
    /// This is the stateful version that can access term information
    pub fn apply_with_manager(&self, goal: &Goal, _manager: &TermManager) -> Result<TacticResult> {
        // Convert goal assertions to Rhai array
        let assertions: Vec<i64> = goal.assertions.iter().map(|id| id.0 as i64).collect();

        // Create scope with assertions
        let mut scope = rhai::Scope::new();
        scope.push("assertions", assertions.clone());

        // Execute the script
        let result: rhai::Dynamic = self
            .engine
            .eval_with_scope(&mut scope, &self.script)
            .map_err(|e| {
                crate::error::OxizError::Internal(format!("Script execution failed: {}", e))
            })?;

        // Parse the result
        if let Some(map) = result.try_cast::<rhai::Map>() {
            if let Some(result_type) = map.get("result") {
                match result_type.to_string().as_str() {
                    "sat" => return Ok(TacticResult::Solved(SolveResult::Sat)),
                    "unsat" => return Ok(TacticResult::Solved(SolveResult::Unsat)),
                    "unknown" => return Ok(TacticResult::Solved(SolveResult::Unknown)),
                    "unchanged" => return Ok(TacticResult::NotApplicable),
                    _ => {}
                }
            }

            // Check if script returned modified assertions
            if let Some(new_assertions) = map.get("assertions")
                && let Some(arr) = new_assertions.clone().try_cast::<rhai::Array>()
            {
                let new_ids: Vec<TermId> = arr
                    .iter()
                    .filter_map(|v| v.as_int().ok().map(|i| TermId(i as u32)))
                    .collect();

                if new_ids != goal.assertions {
                    return Ok(TacticResult::SubGoals(vec![Goal::new(new_ids)]));
                }
            }
        }

        Ok(TacticResult::NotApplicable)
    }
}

#[cfg(feature = "scripting")]
impl Tactic for ScriptableTactic {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Stateless version - limited functionality without TermManager
        // Convert goal assertions to Rhai array
        let assertions: Vec<i64> = goal.assertions.iter().map(|id| id.0 as i64).collect();

        // Create scope with assertions
        let mut scope = rhai::Scope::new();
        scope.push("assertions", assertions.clone());

        // Execute the script
        let result: rhai::Dynamic = self
            .engine
            .eval_with_scope(&mut scope, &self.script)
            .map_err(|e| {
                crate::error::OxizError::Internal(format!("Script execution failed: {}", e))
            })?;

        // Parse the result
        if let Some(map) = result.try_cast::<rhai::Map>() {
            if let Some(result_type) = map.get("result") {
                match result_type.to_string().as_str() {
                    "sat" => return Ok(TacticResult::Solved(SolveResult::Sat)),
                    "unsat" => return Ok(TacticResult::Solved(SolveResult::Unsat)),
                    "unknown" => return Ok(TacticResult::Solved(SolveResult::Unknown)),
                    "unchanged" => return Ok(TacticResult::NotApplicable),
                    _ => {}
                }
            }

            // Check if script returned modified assertions
            if let Some(new_assertions) = map.get("assertions")
                && let Some(arr) = new_assertions.clone().try_cast::<rhai::Array>()
            {
                let new_ids: Vec<TermId> = arr
                    .iter()
                    .filter_map(|v| v.as_int().ok().map(|i| TermId(i as u32)))
                    .collect();

                if new_ids != goal.assertions {
                    return Ok(TacticResult::SubGoals(vec![Goal::new(new_ids)]));
                }
            }
        }

        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        &self.description
    }
}

/// Conditional tactic - chooses between tactics based on a probe value
///
/// This tactic evaluates a probe on the goal and selects one of two tactics
/// based on whether the probe value exceeds a threshold.
pub struct CondTactic {
    probe: std::sync::Arc<dyn Probe>,
    threshold: f64,
    if_true: std::sync::Arc<dyn Tactic>,
    if_false: std::sync::Arc<dyn Tactic>,
}

impl core::fmt::Debug for CondTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CondTactic")
            .field("probe_name", &self.probe.name())
            .field("threshold", &self.threshold)
            .field("if_true", &self.if_true.name())
            .field("if_false", &self.if_false.name())
            .finish()
    }
}

impl CondTactic {
    /// Create a new conditional tactic
    ///
    /// # Arguments
    /// * `probe` - The probe to evaluate
    /// * `threshold` - If probe value > threshold, use if_true, else use if_false
    /// * `if_true` - Tactic to use when probe > threshold
    /// * `if_false` - Tactic to use when probe <= threshold
    pub fn new(
        probe: std::sync::Arc<dyn Probe>,
        threshold: f64,
        if_true: std::sync::Arc<dyn Tactic>,
        if_false: std::sync::Arc<dyn Tactic>,
    ) -> Self {
        Self {
            probe,
            threshold,
            if_true,
            if_false,
        }
    }

    /// Create from boxed values
    pub fn from_box(
        probe: Box<dyn Probe>,
        threshold: f64,
        if_true: Box<dyn Tactic>,
        if_false: Box<dyn Tactic>,
    ) -> Self {
        Self {
            probe: probe.into(),
            threshold,
            if_true: if_true.into(),
            if_false: if_false.into(),
        }
    }
}

impl Tactic for CondTactic {
    fn name(&self) -> &str {
        "cond"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Note: We need a TermManager to evaluate probes properly
        // For now, we'll use a dummy evaluation that creates a temporary manager
        // In production, the probe should be evaluated with proper context
        let probe_value = {
            // Create a minimal manager for probe evaluation
            // This is a workaround - ideally we'd have access to the real manager
            let manager = crate::ast::TermManager::new();
            self.probe.evaluate(goal, &manager)
        };

        if probe_value > self.threshold {
            self.if_true.apply(goal)
        } else {
            self.if_false.apply(goal)
        }
    }

    fn description(&self) -> &str {
        "Conditional tactic selection based on probe value"
    }
}

/// When tactic - applies a tactic only when a probe condition is met
///
/// This is a convenience wrapper around CondTactic that returns NotApplicable
/// when the condition is not met.
pub struct WhenTactic {
    probe: std::sync::Arc<dyn Probe>,
    threshold: f64,
    tactic: std::sync::Arc<dyn Tactic>,
}

impl core::fmt::Debug for WhenTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WhenTactic")
            .field("probe_name", &self.probe.name())
            .field("threshold", &self.threshold)
            .field("tactic", &self.tactic.name())
            .finish()
    }
}

impl WhenTactic {
    /// Create a new when tactic
    pub fn new(
        probe: std::sync::Arc<dyn Probe>,
        threshold: f64,
        tactic: std::sync::Arc<dyn Tactic>,
    ) -> Self {
        Self {
            probe,
            threshold,
            tactic,
        }
    }
}

impl Tactic for WhenTactic {
    fn name(&self) -> &str {
        "when"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let probe_value = {
            let manager = crate::ast::TermManager::new();
            self.probe.evaluate(goal, &manager)
        };

        if probe_value > self.threshold {
            self.tactic.apply(goal)
        } else {
            Ok(TacticResult::NotApplicable)
        }
    }

    fn description(&self) -> &str {
        "Apply tactic only when probe condition is met"
    }
}

/// FailIf tactic - fails if a probe condition is met
///
/// Useful for checking preconditions before applying tactics.
pub struct FailIfTactic {
    probe: std::sync::Arc<dyn Probe>,
    threshold: f64,
    message: String,
}

impl core::fmt::Debug for FailIfTactic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FailIfTactic")
            .field("probe_name", &self.probe.name())
            .field("threshold", &self.threshold)
            .finish()
    }
}

impl FailIfTactic {
    /// Create a new fail-if tactic
    pub fn new(probe: std::sync::Arc<dyn Probe>, threshold: f64, message: String) -> Self {
        Self {
            probe,
            threshold,
            message,
        }
    }
}

impl Tactic for FailIfTactic {
    fn name(&self) -> &str {
        "fail-if"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        let probe_value = {
            let manager = crate::ast::TermManager::new();
            self.probe.evaluate(goal, &manager)
        };

        if probe_value > self.threshold {
            Ok(TacticResult::Failed(self.message.clone()))
        } else {
            Ok(TacticResult::NotApplicable)
        }
    }

    fn description(&self) -> &str {
        "Fail if probe condition is met"
    }
}

/// NNF tactic - converts formulas to Negation Normal Form
///
/// In NNF, negations are pushed inward so they only appear directly
/// before atoms, and only AND, OR, NOT operations remain.
#[derive(Debug)]
pub struct NnfTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> NnfTactic<'a> {
    /// Create a new NNF tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply NNF conversion to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());

        for &assertion in &goal.assertions {
            let nnf = to_nnf(assertion, self.manager);
            if nnf != assertion {
                changed = true;
            }
            new_assertions.push(nnf);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Stateless NNF tactic
#[derive(Debug, Default)]
pub struct StatelessNnfTactic;

impl Tactic for StatelessNnfTactic {
    fn name(&self) -> &str {
        "nnf"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // NNF conversion rebuilds terms with negations pushed inward, which
        // needs a `&mut TermManager`. The manager-free path honestly reports
        // NotApplicable instead of returning the goal unchanged. Use
        // `NnfTactic::apply_mut` (or `create_managed`) for the real conversion.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Convert formulas to Negation Normal Form \
         (requires a TermManager; the manager-free path is NotApplicable)"
    }
}

/// Tseitin CNF tactic - converts formulas to Conjunctive Normal Form
///
/// Uses the Tseitin transformation ([`to_cnf_tseitin`]) which introduces
/// auxiliary `tseitin!*` variables to avoid exponential blowup in formula
/// size.  The result is **equisatisfiable**, not equivalent: every model of
/// the output restricts to a model of the input and every model of the input
/// extends to one of the output, but the auxiliary variables appear in
/// models of the transformed goal.  This matches Z3's `tseitin-cnf` tactic.
#[derive(Debug)]
pub struct TseitinCnfTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> TseitinCnfTactic<'a> {
    /// Create a new Tseitin CNF tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply CNF conversion to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());

        for &assertion in &goal.assertions {
            let cnf = to_cnf_tseitin(assertion, self.manager);
            if cnf != assertion {
                changed = true;
            }
            new_assertions.push(cnf);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Stateless CNF tactic
#[derive(Debug, Default)]
pub struct StatelessCnfTactic;

impl Tactic for StatelessCnfTactic {
    fn name(&self) -> &str {
        "tseitin-cnf"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // Tseitin CNF conversion introduces auxiliary variables and rebuilt
        // clauses, requiring a `&mut TermManager`. The manager-free path
        // honestly reports NotApplicable instead of returning the goal
        // unchanged. Use `TseitinCnfTactic::apply_mut` (or `create_managed`)
        // for the real conversion.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Convert formulas to Conjunctive Normal Form using Tseitin transformation \
         (requires a TermManager; the manager-free path is NotApplicable)"
    }
}
