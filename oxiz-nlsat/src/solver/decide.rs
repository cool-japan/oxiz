//! Decision making and feasibility region computation for the NLSAT solver.
//!
//! Implements variable ordering, decision heuristics (VSIDS), phase saving,
//! and cylindrical algebraic decomposition (CAD) projection for feasibility.

use super::NlsatSolver;
use crate::interval_set::IntervalSet;
use crate::types::{Atom, AtomKind, BoolVar, Literal};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;

impl NlsatSolver {
    /// Make a decision.
    pub(super) fn decide(&mut self) -> Option<Literal> {
        // Random decision
        if self.config.random_decisions
            && self.random() < self.config.random_freq
            && let Some(lit) = self.random_decision()
        {
            return Some(lit);
        }

        // VSIDS-like decision: pick the unassigned variable with highest activity
        let mut best_var: Option<BoolVar> = None;
        let mut best_activity = f64::NEG_INFINITY;

        for var in 0..self.num_bool_vars {
            if self.assignment.is_bool_assigned(var) {
                continue;
            }

            let activity = self.var_activity.get(var as usize).copied().unwrap_or(0.0);
            if activity > best_activity {
                best_activity = activity;
                best_var = Some(var);
            }
        }

        best_var.map(|var| {
            // Use saved phase (phase saving heuristic)
            let polarity = self.saved_phase.get(var as usize).copied().unwrap_or(true);
            Literal::new(var, polarity)
        })
    }

    /// Save the phase (polarity) of a literal assignment.
    pub(super) fn save_phase(&mut self, lit: Literal) {
        let var = lit.var();
        let polarity = !lit.is_negated();
        if (var as usize) < self.saved_phase.len() {
            self.saved_phase[var as usize] = polarity;
        }
    }

    /// Make a random decision.
    pub(super) fn random_decision(&mut self) -> Option<Literal> {
        let mut unassigned = Vec::new();
        for var in 0..self.num_bool_vars {
            if !self.assignment.is_bool_assigned(var) {
                unassigned.push(var);
            }
        }

        if unassigned.is_empty() {
            return None;
        }

        let idx = (self.random_int() as usize) % unassigned.len();
        let var = unassigned[idx];
        let positive = self.random_int().is_multiple_of(2);

        Some(if positive {
            Literal::positive(var)
        } else {
            Literal::negative(var)
        })
    }

    /// Get the next arithmetic variable to assign.
    pub(super) fn next_arith_var(&self) -> Option<Var> {
        // Return the first unassigned variable in the ordering
        self.var_order
            .iter()
            .find(|&&var| !self.assignment.is_arith_assigned(var))
            .copied()
    }

    /// Pick a value for an arithmetic variable.
    pub(super) fn pick_arith_value(&mut self, var: Var) -> Option<BigRational> {
        // Get the feasible region for this variable
        let feasible = self.compute_feasible_region(var);

        // Early termination: if feasible region is empty, record and return None
        if feasible.is_empty() {
            if self.config.early_termination {
                self.stats.early_terminations += 1;
            }
            return None;
        }

        // Sample a point from the feasible region
        feasible.sample()
    }

    /// Compute the feasible region for an arithmetic variable.
    pub(super) fn compute_feasible_region(&self, var: Var) -> IntervalSet {
        let mut region = IntervalSet::reals();

        // Intersect with constraints from all satisfied atoms
        for atom in &self.atoms {
            if let Atom::Ineq(ineq) = atom {
                // Check if this atom involves the variable
                let involves_var = ineq.factors.iter().any(|f| f.poly.vars().contains(&var));
                if !involves_var {
                    continue;
                }

                // Check if this atom is assigned
                let val = self.assignment.bool_value(ineq.bool_var);
                if val.is_undef() {
                    continue;
                }

                // Get the constraint on var from this atom
                let constraint = self.atom_constraint_on_var(atom, var, val.is_true());
                region = region.intersect(&constraint);

                if region.is_empty() {
                    break;
                }
            }
        }

        region
    }

    /// Get the constraint that an atom places on a variable.
    pub(super) fn atom_constraint_on_var(
        &self,
        atom: &Atom,
        var: Var,
        atom_is_true: bool,
    ) -> IntervalSet {
        match atom {
            Atom::Ineq(ineq) => {
                // For now, only handle single-factor atoms
                if ineq.factors.len() != 1 {
                    return IntervalSet::reals();
                }

                let factor = &ineq.factors[0];

                // Substitute all assigned variables except `var`
                let mut sub_poly = factor.poly.clone();
                for v in factor.poly.vars() {
                    if v != var
                        && let Some(val) = self.assignment.arith_value(v)
                    {
                        sub_poly = sub_poly.substitute(v, &Polynomial::constant(val.clone()));
                    }
                }

                // Now sub_poly should be univariate in `var`
                if !sub_poly.is_univariate() && !sub_poly.is_constant() {
                    // Can't simplify further
                    return IntervalSet::reals();
                }

                // Find roots
                let roots = self.find_univariate_roots(&sub_poly, var);

                // Determine signs between roots
                let signs = self.compute_signs_between_roots(&sub_poly, var, &roots);

                // Create interval set based on constraint kind and polarity
                let target_sign = match (ineq.kind, atom_is_true) {
                    (AtomKind::Eq, true) => 0,    // p = 0
                    (AtomKind::Eq, false) => 127, // p != 0 (special case)
                    (AtomKind::Lt, true) => -1,   // p < 0
                    (AtomKind::Lt, false) => 1,   // p >= 0 (includes 0)
                    (AtomKind::Gt, true) => 1,    // p > 0
                    (AtomKind::Gt, false) => -1,  // p <= 0 (includes 0)
                    _ => return IntervalSet::reals(),
                };

                if target_sign == 127 {
                    // p != 0: complement of {roots}
                    let zero_set = IntervalSet::sign_set(&roots, &signs, 0);
                    zero_set.complement()
                } else if target_sign == 1 && !atom_is_true {
                    // p >= 0: positive or zero
                    let pos_set = IntervalSet::sign_set(&roots, &signs, 1);
                    let zero_set = IntervalSet::sign_set(&roots, &signs, 0);
                    pos_set.union(&zero_set)
                } else if target_sign == -1 && !atom_is_true {
                    // p <= 0: negative or zero
                    let neg_set = IntervalSet::sign_set(&roots, &signs, -1);
                    let zero_set = IntervalSet::sign_set(&roots, &signs, 0);
                    neg_set.union(&zero_set)
                } else {
                    IntervalSet::sign_set(&roots, &signs, target_sign)
                }
            }
            Atom::Root(root) => {
                use crate::cad::SturmSequence;

                // For root atoms, we need to isolate the roots and determine the constraint
                // x op root[i](p) where op is =, <, >, <=, >=

                // First, check if this root atom actually involves the variable `var`
                if root.var != var && !root.poly.vars().contains(&var) {
                    return IntervalSet::reals();
                }

                // If the atom involves `var` in the polynomial (not as the root variable),
                // we cannot easily extract a constraint on `var` alone
                if root.var != var {
                    return IntervalSet::reals();
                }

                // Substitute all assigned variables (except var) into the polynomial
                let mut sub_poly = root.poly.clone();
                for v in root.poly.vars() {
                    if v != var {
                        if let Some(val) = self.assignment.arith_value(v) {
                            sub_poly = sub_poly.substitute(v, &Polynomial::constant(val.clone()));
                        } else {
                            return IntervalSet::reals();
                        }
                    }
                }

                // If the polynomial is constant, no roots exist
                if sub_poly.is_constant() {
                    return IntervalSet::empty();
                }

                // Isolate the roots
                let sturm = SturmSequence::new(&sub_poly, var);
                let root_intervals = sturm.isolate_roots();

                // Check if we have enough roots
                if (root.root_index as usize) > root_intervals.len() || root.root_index == 0 {
                    return IntervalSet::empty();
                }

                // Get the i-th root interval
                let (root_lo, root_hi) = &root_intervals[(root.root_index - 1) as usize];

                // Create interval set based on the atom kind and polarity
                match (root.kind, atom_is_true) {
                    (AtomKind::RootEq, true) => {
                        // x = root[i](p)
                        IntervalSet::from_point(root_lo.clone())
                    }
                    (AtomKind::RootEq, false) => {
                        // x != root[i](p) - complement of the point
                        IntervalSet::from_point(root_lo.clone()).complement()
                    }
                    (AtomKind::RootLt, true) => {
                        // x < root[i](p) - approximately (-∞, root_hi)
                        IntervalSet::lt(root_hi.clone())
                    }
                    (AtomKind::RootLt, false) => {
                        // x >= root[i](p) - approximately [root_lo, +∞)
                        IntervalSet::ge(root_lo.clone())
                    }
                    (AtomKind::RootGt, true) => {
                        // x > root[i](p) - approximately (root_lo, +∞)
                        IntervalSet::gt(root_lo.clone())
                    }
                    (AtomKind::RootGt, false) => {
                        // x <= root[i](p) - approximately (-∞, root_hi]
                        IntervalSet::le(root_hi.clone())
                    }
                    (AtomKind::RootLe, true) => {
                        // x <= root[i](p)
                        IntervalSet::le(root_hi.clone())
                    }
                    (AtomKind::RootLe, false) => {
                        // x > root[i](p)
                        IntervalSet::gt(root_lo.clone())
                    }
                    (AtomKind::RootGe, true) => {
                        // x >= root[i](p)
                        IntervalSet::ge(root_lo.clone())
                    }
                    (AtomKind::RootGe, false) => {
                        // x < root[i](p)
                        IntervalSet::lt(root_hi.clone())
                    }
                    _ => IntervalSet::reals(),
                }
            }
        }
    }

    /// Find roots of a univariate polynomial.
    pub(super) fn find_univariate_roots(&self, poly: &Polynomial, var: Var) -> Vec<BigRational> {
        // For now, use a simple approach for low-degree polynomials
        let degree = poly.degree(var);

        if degree == 0 {
            return Vec::new();
        }

        if degree == 1 {
            // Linear: ax + b = 0  =>  x = -b/a
            return self.find_linear_root(poly);
        }

        if degree == 2 {
            // Quadratic: use quadratic formula (rational roots only)
            return self.find_quadratic_roots(poly);
        }

        // For higher degrees, we would need more sophisticated root isolation
        // For now, return empty (conservative but safe)
        Vec::new()
    }

    /// Find the root of a linear polynomial.
    pub(super) fn find_linear_root(&self, poly: &Polynomial) -> Vec<BigRational> {
        // p = ax + b, find x = -b/a
        let terms = poly.terms();
        if terms.len() > 2 {
            return Vec::new();
        }

        let mut a = BigRational::zero();
        let mut b = BigRational::zero();

        for term in terms {
            if term.monomial.is_unit() {
                b = term.coeff.clone();
            } else if term.monomial.total_degree() == 1 {
                a = term.coeff.clone();
            }
        }

        if a.is_zero() {
            return Vec::new();
        }

        vec![-b / a]
    }

    /// Find rational roots of a quadratic polynomial.
    pub(super) fn find_quadratic_roots(&self, poly: &Polynomial) -> Vec<BigRational> {
        // p = ax^2 + bx + c
        // Discriminant = b^2 - 4ac
        // If discriminant is a perfect square, roots are rational

        let terms = poly.terms();
        if terms.len() > 3 {
            return Vec::new();
        }

        let mut a = BigRational::zero();
        let mut b = BigRational::zero();
        let mut c = BigRational::zero();

        for term in terms {
            match term.monomial.total_degree() {
                0 => c = term.coeff.clone(),
                1 => b = term.coeff.clone(),
                2 => a = term.coeff.clone(),
                _ => return Vec::new(),
            }
        }

        if a.is_zero() {
            // Actually linear
            if b.is_zero() {
                return Vec::new();
            }
            return vec![-c.clone() / b.clone()];
        }

        // Discriminant
        let disc = &b * &b - BigRational::from_integer(4.into()) * &a * &c;

        if disc.is_negative() {
            return Vec::new();
        }

        if disc.is_zero() {
            let root = -b / (BigRational::from_integer(2.into()) * a);
            return vec![root];
        }

        // Check if discriminant is a perfect square
        // For rational discriminant p/q, we need both p and q to be perfect squares
        let numer = disc.numer().clone();
        let denom = disc.denom().clone();

        if let (Some(sqrt_n), Some(sqrt_d)) =
            (super::integer_sqrt(&numer), super::integer_sqrt(&denom))
        {
            let sqrt_disc = BigRational::new(sqrt_n, sqrt_d);
            let two_a = BigRational::from_integer(2.into()) * &a;
            let root1 = (-&b + &sqrt_disc) / &two_a;
            let root2 = (-&b - &sqrt_disc) / &two_a;

            let mut roots = vec![root1, root2];
            roots.sort();
            roots.dedup();
            roots
        } else {
            // Irrational roots - cannot represent exactly
            Vec::new()
        }
    }

    /// Compute signs of polynomial between roots.
    pub(super) fn compute_signs_between_roots(
        &self,
        poly: &Polynomial,
        var: Var,
        roots: &[BigRational],
    ) -> Vec<i8> {
        if roots.is_empty() {
            // No roots - evaluate at any point
            let test_val = BigRational::zero();
            let mut eval_map = FxHashMap::default();
            eval_map.insert(var, test_val);
            let val = poly.eval(&eval_map);
            let sign = if val.is_zero() {
                0
            } else if val.is_positive() {
                1
            } else {
                -1
            };
            return vec![sign];
        }

        let mut signs = Vec::with_capacity(roots.len() + 1);

        // Before first root
        let before = &roots[0] - BigRational::one();
        signs.push(self.eval_sign(poly, var, &before));

        // Between roots
        for i in 0..roots.len() - 1 {
            let mid = (&roots[i] + &roots[i + 1]) / BigRational::from_integer(2.into());
            signs.push(self.eval_sign(poly, var, &mid));
        }

        // After last root
        if let Some(last_root) = roots.last() {
            let after = last_root + BigRational::one();
            signs.push(self.eval_sign(poly, var, &after));
        }

        signs
    }

    /// Evaluate the sign of a polynomial at a point.
    pub(super) fn eval_sign(&self, poly: &Polynomial, var: Var, val: &BigRational) -> i8 {
        let mut eval_map = FxHashMap::default();
        eval_map.insert(var, val.clone());
        let result = poly.eval(&eval_map);
        if result.is_zero() {
            0
        } else if result.is_positive() {
            1
        } else {
            -1
        }
    }
}
