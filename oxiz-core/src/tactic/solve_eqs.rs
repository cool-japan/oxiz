//! Equation solving tactics.

use super::core::*;
use super::probe::Probe;
use crate::ast::normal_forms::{to_cnf, to_nnf};
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use num_integer::Integer;
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

    /// Apply Gaussian elimination to solve equations
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        if goal.assertions.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut current_assertions = goal.assertions.clone();
        let mut total_changed = false;

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

                // Build substitution map
                let mut subst = rustc_hash::FxHashMap::default();
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
            return Ok(TacticResult::NotApplicable);
        }

        // Check for trivially true/false
        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        // Check if any assertion is false
        if current_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = current_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        // If all assertions are true, goal is SAT
        if filtered.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
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

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessSolveEqsTactic;

impl Tactic for StatelessSolveEqsTactic {
    fn name(&self) -> &str {
        "solve-eqs"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![(*goal).clone()]))
    }

    fn description(&self) -> &str {
        "Gaussian elimination for linear equations - solves x = expr and substitutes"
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
    coefficients: rustc_hash::FxHashMap<TermId, Coefficient>,
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
            coefficients: rustc_hash::FxHashMap::default(),
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
        let mut all_vars: rustc_hash::FxHashSet<TermId> = rustc_hash::FxHashSet::default();
        for c in &constraints {
            for var in c.variables() {
                all_vars.insert(var);
            }
        }

        // Phase 3: Build variable bounds index
        let mut lowers: rustc_hash::FxHashMap<TermId, Vec<usize>> =
            rustc_hash::FxHashMap::default();
        let mut uppers: rustc_hash::FxHashMap<TermId, Vec<usize>> =
            rustc_hash::FxHashMap::default();

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

            // Trivial elimination: no lower or upper bounds
            if lower_indices.is_empty() || upper_indices.is_empty() {
                // Mark constraints involving this variable as dead
                for &idx in lower_indices.iter().chain(upper_indices.iter()) {
                    constraints[idx].dead = true;
                }
                eliminated_any = true;
                continue;
            }

            // Perform pairwise resolution
            let mut new_constraints = Vec::new();

            for &lower_idx in lower_indices {
                for &upper_idx in upper_indices {
                    self.op_count += 1;
                    if self.op_count >= self.op_limit {
                        break;
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
            self.manager.mk_int(c.numerator.clone())
        } else {
            // Rational - approximate as integer for now
            // A more sophisticated implementation would use Real sort
            let approx = &c.numerator / &c.denominator;
            self.manager.mk_int(approx)
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

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![(*goal).clone()]))
    }

    fn description(&self) -> &str {
        "Fourier-Motzkin variable elimination for linear arithmetic"
    }
}

// ============================================================================
// Scriptable Tactic (Rhai-based)
// ============================================================================

/// A tactic that executes user-defined Rhai scripts
///
/// This allows users to create custom simplification strategies by writing
/// Rhai scripts. The scripts have access to goal assertions and can return
/// simplified goals or solve results.
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
#[derive(Debug)]
pub struct ScriptableTactic {
    engine: rhai::Engine,
    script: String,
    name: String,
    description: String,
}

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

impl std::fmt::Debug for CondTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Debug for WhenTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Debug for FailIfTactic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![(*goal).clone()]))
    }

    fn description(&self) -> &str {
        "Convert formulas to Negation Normal Form"
    }
}

/// Tseitin CNF tactic - converts formulas to Conjunctive Normal Form
///
/// Uses the Tseitin transformation which introduces auxiliary variables
/// to avoid exponential blowup in formula size.
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
            let cnf = to_cnf(assertion, self.manager);
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

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![(*goal).clone()]))
    }

    fn description(&self) -> &str {
        "Convert formulas to Conjunctive Normal Form using Tseitin transformation"
    }
}

// ============================================================================
// Pseudo-Boolean to Bit-Vector Tactic
// ============================================================================

/// Pseudo-Boolean to Bit-Vector tactic
///
/// This tactic converts pseudo-boolean constraints (linear combinations of
/// booleans with integer coefficients) into bit-vector arithmetic.
///
/// # Example
///
/// `2*x + 3*y + z <= 5` where x, y, z are booleans
///
/// becomes a bit-vector constraint using:
/// - Each boolean as a 1-bit BV (or zero-extended)
/// - Integer coefficients as BV constants
/// - Addition and comparison in BV arithmetic
///
/// # Reference
///
/// Based on Z3's `pb2bv_tactic` in `src/tactic/arith/pb2bv_tactic.cpp`
#[derive(Debug)]
pub struct Pb2BvTactic<'a> {
    manager: &'a mut TermManager,
    /// Bit width for intermediate results (auto-computed or specified)
    bit_width: Option<u32>,
}

/// A term in a pseudo-boolean constraint: coefficient * boolean_var
#[derive(Debug, Clone)]
struct PbTerm {
    /// The coefficient (positive or negative)
    coefficient: i64,
    /// The boolean variable
    var: TermId,
}

/// A pseudo-boolean constraint
#[derive(Debug)]
struct PbConstraint {
    /// Linear combination of boolean variables
    terms: Vec<PbTerm>,
    /// Constant term (right-hand side)
    bound: i64,
    /// Constraint type: true for <=, false for =
    is_le: bool,
}

impl<'a> Pb2BvTactic<'a> {
    /// Create a new PB to BV tactic with auto bit-width
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            bit_width: None,
        }
    }

    /// Create with explicit bit width
    pub fn with_bit_width(manager: &'a mut TermManager, width: u32) -> Self {
        Self {
            manager,
            bit_width: Some(width),
        }
    }

    /// Apply the tactic mutably
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut new_assertions = Vec::new();
        let mut changed = false;

        for &assertion in &goal.assertions {
            if let Some(converted) = self.convert_constraint(assertion) {
                new_assertions.push(converted);
                changed = true;
            } else {
                new_assertions.push(assertion);
            }
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Try to convert a constraint to bit-vector form
    fn convert_constraint(&mut self, term: TermId) -> Option<TermId> {
        let pb = self.extract_pb_constraint(term)?;

        // Compute required bit width
        let width = self.compute_bit_width(&pb);

        // Convert to bit-vector constraint
        self.encode_pb_as_bv(&pb, width)
    }

    /// Extract a PB constraint from a term
    fn extract_pb_constraint(&self, term: TermId) -> Option<PbConstraint> {
        let t = self.manager.get(term)?;

        match &t.kind {
            TermKind::Le(lhs, rhs) => {
                // lhs <= rhs
                // Convert to: lhs - rhs <= 0, then to: lhs <= rhs
                let (terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                Some(PbConstraint {
                    terms,
                    bound: rhs_val,
                    is_le: true,
                })
            }
            TermKind::Ge(lhs, rhs) => {
                // lhs >= rhs => -lhs <= -rhs => rhs <= lhs
                // Convert to: -lhs + rhs <= 0
                let (mut terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                // Negate all coefficients
                for term in &mut terms {
                    term.coefficient = -term.coefficient;
                }

                Some(PbConstraint {
                    terms,
                    bound: -rhs_val,
                    is_le: true,
                })
            }
            TermKind::Lt(lhs, rhs) => {
                // lhs < rhs => lhs <= rhs - 1
                let (terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                Some(PbConstraint {
                    terms,
                    bound: rhs_val - 1,
                    is_le: true,
                })
            }
            TermKind::Gt(lhs, rhs) => {
                // lhs > rhs => rhs < lhs => rhs <= lhs - 1
                let (mut terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                for term in &mut terms {
                    term.coefficient = -term.coefficient;
                }

                Some(PbConstraint {
                    terms,
                    bound: -rhs_val - 1,
                    is_le: true,
                })
            }
            TermKind::Eq(lhs, rhs) => {
                let (terms, _lhs_const) = self.extract_linear_bool_comb(*lhs)?;
                let rhs_val = self.extract_int_const(*rhs)?;

                Some(PbConstraint {
                    terms,
                    bound: rhs_val,
                    is_le: false, // equality
                })
            }
            _ => None,
        }
    }

    /// Extract a linear combination of boolean variables
    /// Returns (terms, constant) where the expression is Σ(coeff * var) + constant
    fn extract_linear_bool_comb(&self, term: TermId) -> Option<(Vec<PbTerm>, i64)> {
        let t = self.manager.get(term)?;

        match &t.kind {
            TermKind::Add(args) => {
                let mut all_terms = Vec::new();
                let mut total_const = 0i64;

                for &arg in args.iter() {
                    if let Some((terms, c)) = self.extract_linear_bool_comb(arg) {
                        all_terms.extend(terms);
                        total_const += c;
                    } else {
                        return None;
                    }
                }

                Some((all_terms, total_const))
            }
            TermKind::Mul(args) if args.len() == 2 => {
                // coeff * var or var * coeff
                let first = self.manager.get(args[0])?;
                let second = self.manager.get(args[1])?;

                if let TermKind::IntConst(c) = &first.kind {
                    // c * var
                    if self.is_boolean_term(args[1]) {
                        let coeff = c.try_into().ok()?;
                        return Some((
                            vec![PbTerm {
                                coefficient: coeff,
                                var: args[1],
                            }],
                            0,
                        ));
                    }
                }

                if let TermKind::IntConst(c) = &second.kind {
                    // var * c
                    if self.is_boolean_term(args[0]) {
                        let coeff = c.try_into().ok()?;
                        return Some((
                            vec![PbTerm {
                                coefficient: coeff,
                                var: args[0],
                            }],
                            0,
                        ));
                    }
                }

                None
            }
            TermKind::IntConst(c) => {
                let val = c.try_into().ok()?;
                Some((Vec::new(), val))
            }
            TermKind::Ite(cond, then_br, else_br) => {
                // if cond then 1 else 0 (common pattern for bool-to-int)
                let then_t = self.manager.get(*then_br)?;
                let else_t = self.manager.get(*else_br)?;

                if matches!(then_t.kind, TermKind::IntConst(ref v) if *v == 1.into())
                    && matches!(else_t.kind, TermKind::IntConst(ref v) if *v == 0.into())
                {
                    // This is (ite bool 1 0), treat as bool with coefficient 1
                    Some((
                        vec![PbTerm {
                            coefficient: 1,
                            var: *cond,
                        }],
                        0,
                    ))
                } else {
                    None
                }
            }
            _ => {
                // Check if it's a standalone boolean variable
                if self.is_boolean_term(term) {
                    Some((
                        vec![PbTerm {
                            coefficient: 1,
                            var: term,
                        }],
                        0,
                    ))
                } else {
                    None
                }
            }
        }
    }

    /// Check if a term is a boolean
    fn is_boolean_term(&self, term: TermId) -> bool {
        if let Some(t) = self.manager.get(term) {
            t.sort == self.manager.sorts.bool_sort
        } else {
            false
        }
    }

    /// Extract an integer constant
    fn extract_int_const(&self, term: TermId) -> Option<i64> {
        let t = self.manager.get(term)?;
        if let TermKind::IntConst(c) = &t.kind {
            c.try_into().ok()
        } else {
            None
        }
    }

    /// Compute the required bit width for a PB constraint
    fn compute_bit_width(&self, pb: &PbConstraint) -> u32 {
        if let Some(w) = self.bit_width {
            return w;
        }

        // Compute the maximum possible sum
        let mut max_sum: i64 = 0;
        for term in &pb.terms {
            max_sum += term.coefficient.abs();
        }
        max_sum = max_sum.max(pb.bound.abs());

        // Compute bits needed (including sign bit for safety)
        let bits_needed = if max_sum == 0 {
            1
        } else {
            (64 - max_sum.leading_zeros()).max(1) + 1
        };

        bits_needed.min(64) // Cap at 64 bits
    }

    /// Encode a PB constraint as bit-vector arithmetic
    fn encode_pb_as_bv(&mut self, pb: &PbConstraint, width: u32) -> Option<TermId> {
        let _bv_sort = self.manager.sorts.bitvec(width);

        // Build the sum: Σ(coeff * bool_to_bv(var))
        let mut sum_terms: Vec<TermId> = Vec::new();

        for term in &pb.terms {
            // Convert boolean to BV: (ite var 1bv 0bv)
            let bv_one = self.manager.mk_bitvec(1u64, width);
            let bv_zero = self.manager.mk_bitvec(0u64, width);
            let var_bv = self.manager.mk_ite(term.var, bv_one, bv_zero);

            // Multiply by coefficient
            let coeff_bv = if term.coefficient >= 0 {
                self.manager.mk_bitvec(term.coefficient as u64, width)
            } else {
                // Negative coefficient: use two's complement
                let abs_coeff = self.manager.mk_bitvec((-term.coefficient) as u64, width);
                self.manager.mk_bv_neg(abs_coeff)
            };

            let prod = self.manager.mk_bv_mul(coeff_bv, var_bv);
            sum_terms.push(prod);
        }

        // Sum all terms
        let sum = if sum_terms.is_empty() {
            self.manager.mk_bitvec(0u64, width)
        } else if sum_terms.len() == 1 {
            sum_terms[0]
        } else {
            let mut acc = sum_terms[0];
            for &term in &sum_terms[1..] {
                acc = self.manager.mk_bv_add(acc, term);
            }
            acc
        };

        // Create the bound as BV
        let bound_bv = if pb.bound >= 0 {
            self.manager.mk_bitvec(pb.bound as u64, width)
        } else {
            let abs_bound = self.manager.mk_bitvec((-pb.bound) as u64, width);
            self.manager.mk_bv_neg(abs_bound)
        };

        // Create the comparison
        if pb.is_le {
            // sum <= bound (signed comparison)
            Some(self.manager.mk_bv_sle(sum, bound_bv))
        } else {
            // sum = bound
            Some(self.manager.mk_eq(sum, bound_bv))
        }
    }
}

/// Stateless wrapper for PB2BV tactic
#[derive(Debug, Default, Clone, Copy)]
pub struct StatelessPb2BvTactic;

impl Tactic for StatelessPb2BvTactic {
    fn name(&self) -> &str {
        "pb2bv"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![(*goal).clone()]))
    }

    fn description(&self) -> &str {
        "Convert pseudo-boolean constraints to bit-vector arithmetic"
    }
}

