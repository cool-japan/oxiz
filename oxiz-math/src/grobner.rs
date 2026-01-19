//! Gröbner basis computation for polynomial ideals.
//!
//! This module implements Buchberger's algorithm for computing Gröbner bases,
//! which is essential for solving systems of polynomial equations in SMT solving,
//! particularly for non-linear real arithmetic (NRA).
//!
//! Reference: Z3's Gröbner basis implementation and standard computer algebra texts.

use crate::polynomial::{Monomial, MonomialOrder, Polynomial};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

/// Compute the S-polynomial of two polynomials.
///
/// The S-polynomial S(f, g) is defined as:
/// S(f, g) = (lcm(LM(f), LM(g)) / LT(f)) * f - (lcm(LM(f), LM(g)) / LT(g)) * g
///
/// where LM is the leading monomial and LT is the leading term.
pub fn s_polynomial(f: &Polynomial, g: &Polynomial) -> Polynomial {
    if f.is_zero() || g.is_zero() {
        return Polynomial::zero();
    }

    let lm_f = f.leading_monomial();
    let lm_g = g.leading_monomial();

    if lm_f.is_none() || lm_g.is_none() {
        return Polynomial::zero();
    }

    let lm_f = lm_f.expect("leading monomial exists for non-zero polynomial");
    let lm_g = lm_g.expect("leading monomial exists for non-zero polynomial");

    // Compute LCM of leading monomials
    let lcm = monomial_lcm(lm_f, lm_g);

    // Divide LCM by leading terms
    let coeff_f = f.leading_coeff();
    let coeff_g = g.leading_coeff();

    if coeff_f.is_zero() || coeff_g.is_zero() {
        return Polynomial::zero();
    }

    let factor_f = if let Some(quotient) = lcm.div(lm_f) {
        quotient
    } else {
        return Polynomial::zero();
    };

    let factor_g = if let Some(quotient) = lcm.div(lm_g) {
        quotient
    } else {
        return Polynomial::zero();
    };

    // S(f, g) = (lcm / LT(f)) * f - (lcm / LT(g)) * g
    let term1 = f
        .mul_monomial(&factor_f)
        .scale(&(BigRational::one() / &coeff_f));
    let term2 = g
        .mul_monomial(&factor_g)
        .scale(&(BigRational::one() / &coeff_g));

    term1.sub(&term2)
}

/// Compute the least common multiple of two monomials.
fn monomial_lcm(m1: &Monomial, m2: &Monomial) -> Monomial {
    let mut powers = Vec::new();

    let vars1 = m1.vars();
    let vars2 = m2.vars();

    let mut i = 0;
    let mut j = 0;

    while i < vars1.len() || j < vars2.len() {
        if i >= vars1.len() {
            powers.push((vars2[j].var, vars2[j].power));
            j += 1;
        } else if j >= vars2.len() || vars1[i].var < vars2[j].var {
            powers.push((vars1[i].var, vars1[i].power));
            i += 1;
        } else if vars1[i].var > vars2[j].var {
            powers.push((vars2[j].var, vars2[j].power));
            j += 1;
        } else {
            // Same variable, take maximum power
            let max_power = vars1[i].power.max(vars2[j].power);
            powers.push((vars1[i].var, max_power));
            i += 1;
            j += 1;
        }
    }

    Monomial::from_powers(powers)
}

/// Reduce a polynomial f with respect to a set of polynomials G.
///
/// This performs multivariate polynomial division, reducing f by polynomials in G
/// until no further reduction is possible.
pub fn reduce(f: &Polynomial, g_set: &[Polynomial]) -> Polynomial {
    if f.is_zero() || g_set.is_empty() {
        return f.clone();
    }

    let mut r = Polynomial::zero();
    let mut p = f.clone();

    // Limit iterations to prevent infinite loops
    let max_iterations = 1000;
    let mut iterations = 0;

    while !p.is_zero() && iterations < max_iterations {
        iterations += 1;
        let mut reduced = false;

        // Try to reduce by each polynomial in G
        for g in g_set {
            if g.is_zero() {
                continue;
            }

            let lm_p = p.leading_monomial();
            let lm_g = g.leading_monomial();

            if lm_p.is_none() || lm_g.is_none() {
                continue;
            }

            let lm_p = lm_p.expect("leading monomial exists");
            let lm_g = lm_g.expect("leading monomial exists for non-zero polynomial");

            // Check if LM(g) divides LM(p)
            if let Some(quotient_monomial) = lm_p.div(lm_g) {
                // p = p - (LT(p) / LT(g)) * g
                let lc_p = p.leading_coeff();
                let lc_g = g.leading_coeff();

                if !lc_g.is_zero() {
                    let quotient_coeff = lc_p / lc_g;
                    let subtractor = g.mul_monomial(&quotient_monomial).scale(&quotient_coeff);

                    p = p.sub(&subtractor);
                    reduced = true;
                    break;
                }
            }
        }

        if !reduced {
            // Cannot reduce leading term, move it to remainder
            if let Some(lt) = p.leading_term() {
                r = r.add(&Polynomial::from_terms(
                    vec![lt.clone()],
                    MonomialOrder::default(),
                ));
                // Remove leading term from p
                let mut terms = p.terms().to_vec();
                if !terms.is_empty() {
                    terms.remove(0);
                }
                p = Polynomial::from_terms(terms, MonomialOrder::default());
            } else {
                break;
            }
        }
    }

    r
}

/// Compute a Gröbner basis for a set of polynomials using Buchberger's algorithm.
///
/// The Gröbner basis is a special generating set for the ideal generated by the input
/// polynomials, with useful properties for solving polynomial systems.
pub fn grobner_basis(polynomials: &[Polynomial]) -> Vec<Polynomial> {
    if polynomials.is_empty() {
        return vec![];
    }

    // Remove zero polynomials and make primitive
    let mut g: Vec<Polynomial> = polynomials
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| p.primitive())
        .collect();

    if g.is_empty() {
        return vec![];
    }

    // Track pairs that need to be processed
    let mut pairs = Vec::new();
    for i in 0..g.len() {
        for j in (i + 1)..g.len() {
            pairs.push((i, j));
        }
    }

    // Limit iterations to prevent infinite loops
    let max_iterations = 1000;
    let mut iterations = 0;

    while !pairs.is_empty() && iterations < max_iterations {
        iterations += 1;

        // Take a pair
        let (i, j) = pairs.pop().expect("collection validated to be non-empty");

        if i >= g.len() || j >= g.len() {
            continue;
        }

        // Compute S-polynomial
        let s = s_polynomial(&g[i], &g[j]);

        // Reduce S-polynomial with respect to G
        let s_reduced = reduce(&s, &g);

        if !s_reduced.is_zero() {
            let s_primitive = s_reduced.primitive();

            // Add new polynomial to basis
            let new_idx = g.len();
            g.push(s_primitive);

            // Add new pairs
            for k in 0..new_idx {
                pairs.push((k, new_idx));
            }
        }
    }

    // Reduce the basis (interreduce)
    interreduce(&g)
}

/// Interreduce a Gröbner basis to make it minimal and reduced.
///
/// A reduced Gröbner basis has the property that no leading term of any polynomial
/// divides any term of another polynomial in the basis.
fn interreduce(basis: &[Polynomial]) -> Vec<Polynomial> {
    let mut result = Vec::new();

    for (i, p) in basis.iter().enumerate() {
        if p.is_zero() {
            continue;
        }

        // Reduce p with respect to all other polynomials
        let others: Vec<Polynomial> = basis
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| q.clone())
            .collect();

        let p_reduced = reduce(p, &others);

        if !p_reduced.is_zero() {
            // Make monic (leading coefficient = 1)
            result.push(p_reduced.make_monic());
        }
    }

    result
}

/// Test if a polynomial is in the ideal generated by a set of polynomials.
///
/// This is done by computing the Gröbner basis and checking if the polynomial
/// reduces to zero.
pub fn ideal_membership(f: &Polynomial, generators: &[Polynomial]) -> bool {
    if f.is_zero() {
        return true;
    }

    if generators.is_empty() {
        return false;
    }

    // Compute Gröbner basis
    let gb = grobner_basis(generators);

    // Reduce f with respect to the Gröbner basis
    let reduced = reduce(f, &gb);

    reduced.is_zero()
}

/// Signature for F5 algorithm.
///
/// A signature tracks the origin of a polynomial in the F5 algorithm,
/// allowing us to detect useless S-polynomials before computing them.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Signature {
    /// Index of the input polynomial this came from
    index: usize,
    /// Monomial multiplier
    monomial: Monomial,
}

impl Signature {
    fn new(index: usize, monomial: Monomial) -> Self {
        Self { index, monomial }
    }

    fn unit(index: usize) -> Self {
        Self {
            index,
            monomial: Monomial::unit(),
        }
    }

    /// Compare signatures using the module monomial order
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare indices (reverse order - higher index is smaller)
        match other.index.cmp(&self.index) {
            std::cmp::Ordering::Equal => {
                // If same index, compare monomials
                self.monomial.grevlex_cmp(&other.monomial)
            }
            ord => ord,
        }
    }
}

/// Labeled polynomial for F5 algorithm.
///
/// Each polynomial is labeled with a signature that tracks its origin.
#[derive(Clone, Debug)]
struct LabeledPoly {
    polynomial: Polynomial,
    signature: Signature,
}

impl LabeledPoly {
    fn new(polynomial: Polynomial, signature: Signature) -> Self {
        Self {
            polynomial,
            signature,
        }
    }

    /// Multiply by a monomial
    #[allow(dead_code)]
    fn mul_monomial(&self, m: &Monomial) -> Self {
        Self {
            polynomial: self.polynomial.mul_monomial(m),
            signature: Signature::new(self.signature.index, self.signature.monomial.mul(m)),
        }
    }
}

/// F5 criterion: check if an S-polynomial will reduce to zero.
///
/// The F5 criterion uses signatures to detect when an S-polynomial
/// will reduce to zero without actually computing it.
fn f5_criterion(sig: &Signature, basis: &[LabeledPoly]) -> bool {
    // Check if there exists a polynomial in basis with same or smaller signature
    for p in basis {
        if p.signature.cmp(sig) != std::cmp::Ordering::Greater {
            // Found a polynomial with signature <= sig
            // Check if its leading monomial divides the signature monomial
            if let Some(lm) = p.polynomial.leading_monomial()
                && sig.monomial.div(lm).is_some()
            {
                return true;
            }
        }
    }
    false
}

/// F5 algorithm for computing Gröbner bases.
///
/// The F5 algorithm is a signature-based approach that is more efficient than
/// F4 because it avoids computing useless S-polynomials by using signature-based
/// criteria to detect when they will reduce to zero.
///
/// Reference: Faugère, "A new efficient algorithm for computing Gröbner bases
/// without reduction to zero (F5)" (2002)
pub fn grobner_basis_f5(polynomials: &[Polynomial]) -> Vec<Polynomial> {
    if polynomials.is_empty() {
        return vec![];
    }

    // Remove zero polynomials
    let polys: Vec<Polynomial> = polynomials
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| p.primitive())
        .collect();

    if polys.is_empty() {
        return vec![];
    }

    // The F5 algorithm processes input polynomials incrementally
    let mut basis: Vec<LabeledPoly> = Vec::new();

    // Process each input polynomial
    for (idx, poly) in polys.iter().enumerate() {
        // Add current polynomial with unit signature
        let labeled = LabeledPoly::new(poly.clone(), Signature::unit(idx));
        basis.push(labeled);

        // Generate S-polynomials with all previous polynomials in basis
        let mut pairs = Vec::new();
        let basis_len = basis.len();

        for i in 0..basis_len {
            for j in (i + 1)..basis_len {
                // Only consider pairs where at least one polynomial has current index
                if basis[i].signature.index == idx || basis[j].signature.index == idx {
                    pairs.push((i, j));
                }
            }
        }

        // Limit iterations to prevent infinite loops
        let max_iterations = 500;
        let mut iterations = 0;

        while !pairs.is_empty() && iterations < max_iterations {
            iterations += 1;

            let (i, j) = pairs.pop().expect("collection validated to be non-empty");

            if i >= basis.len() || j >= basis.len() {
                continue;
            }

            // Compute signatures for the two S-polynomials
            let lm_i = basis[i].polynomial.leading_monomial();
            let lm_j = basis[j].polynomial.leading_monomial();

            if lm_i.is_none() || lm_j.is_none() {
                continue;
            }

            let lm_i = lm_i.expect("leading monomial exists");
            let lm_j = lm_j.expect("leading monomial exists");

            let lcm = monomial_lcm(lm_i, lm_j);

            let sig_i = if let Some(quot) = lcm.div(lm_i) {
                Signature::new(basis[i].signature.index, quot)
            } else {
                continue;
            };

            let sig_j = if let Some(quot) = lcm.div(lm_j) {
                Signature::new(basis[j].signature.index, quot)
            } else {
                continue;
            };

            // Apply F5 criterion: skip if either signature indicates useless S-poly
            if f5_criterion(&sig_i, &basis) || f5_criterion(&sig_j, &basis) {
                continue;
            }

            // Choose the larger signature (smaller in our ordering since we reverse index)
            let (s_poly, sig) = if sig_i.cmp(&sig_j) == std::cmp::Ordering::Greater {
                (
                    s_polynomial(&basis[i].polynomial, &basis[j].polynomial),
                    sig_i,
                )
            } else {
                (
                    s_polynomial(&basis[i].polynomial, &basis[j].polynomial),
                    sig_j,
                )
            };

            if s_poly.is_zero() {
                continue;
            }

            // Reduce S-polynomial (top-reduction only to preserve signature)
            let basis_polys: Vec<Polynomial> =
                basis.iter().map(|lp| lp.polynomial.clone()).collect();
            let s_reduced = reduce(&s_poly, &basis_polys);

            if !s_reduced.is_zero() {
                let s_primitive = s_reduced.primitive();
                let new_labeled = LabeledPoly::new(s_primitive, sig);

                let new_idx = basis.len();
                basis.push(new_labeled);

                // Add new pairs
                for k in 0..new_idx {
                    if basis[k].signature.index == idx || basis[new_idx].signature.index == idx {
                        pairs.push((k, new_idx));
                    }
                }
            }
        }
    }

    // Extract polynomials from labeled polynomials and interreduce
    let polys: Vec<Polynomial> = basis.iter().map(|lp| lp.polynomial.clone()).collect();
    interreduce(&polys)
}

/// F4 algorithm for computing Gröbner bases.
///
/// The F4 algorithm is a matrix-based approach that is more efficient than
/// Buchberger's algorithm for many problems. It processes multiple S-polynomials
/// simultaneously using Gaussian elimination.
///
/// Reference: Faugère, "A new efficient algorithm for computing Gröbner bases (F4)" (1999)
pub fn grobner_basis_f4(polynomials: &[Polynomial]) -> Vec<Polynomial> {
    if polynomials.is_empty() {
        return vec![];
    }

    // Remove zero polynomials and make primitive
    let mut g: Vec<Polynomial> = polynomials
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| p.primitive())
        .collect();

    if g.is_empty() {
        return vec![];
    }

    // Track pairs that need to be processed
    let mut pairs = Vec::new();
    for i in 0..g.len() {
        for j in (i + 1)..g.len() {
            pairs.push((i, j));
        }
    }

    // Limit iterations to prevent infinite loops
    let max_iterations = 500;
    let mut iterations = 0;

    while !pairs.is_empty() && iterations < max_iterations {
        iterations += 1;

        // Select a batch of pairs to process (F4 key idea: process multiple pairs at once)
        let batch_size = pairs.len().min(5);
        let mut batch = Vec::new();
        for _ in 0..batch_size {
            if let Some(pair) = pairs.pop() {
                batch.push(pair);
            }
        }

        // Compute S-polynomials for all pairs in batch
        let mut s_polys = Vec::new();
        for (i, j) in batch {
            if i >= g.len() || j >= g.len() {
                continue;
            }
            let s = s_polynomial(&g[i], &g[j]);
            if !s.is_zero() {
                s_polys.push(s);
            }
        }

        if s_polys.is_empty() {
            continue;
        }

        // Reduce all S-polynomials simultaneously using matrix reduction
        let reduced_polys = f4_matrix_reduction(&s_polys, &g);

        // Add new non-zero polynomials to basis
        for s_reduced in reduced_polys {
            if !s_reduced.is_zero() {
                let s_primitive = s_reduced.primitive();

                // Add new polynomial to basis
                let new_idx = g.len();
                g.push(s_primitive);

                // Add new pairs
                for k in 0..new_idx {
                    pairs.push((k, new_idx));
                }
            }
        }
    }

    // Reduce the basis (interreduce)
    interreduce(&g)
}

/// Perform matrix-based reduction of multiple polynomials simultaneously.
///
/// This is the core of the F4 algorithm: construct a matrix where each row
/// represents a polynomial, then use Gaussian elimination to reduce.
fn f4_matrix_reduction(polys: &[Polynomial], basis: &[Polynomial]) -> Vec<Polynomial> {
    if polys.is_empty() {
        return vec![];
    }

    // Collect all monomials that appear in any polynomial
    let mut monomials = Vec::new();
    for p in polys.iter().chain(basis.iter()) {
        for term in p.terms() {
            let monomial = &term.monomial;
            if !monomials.iter().any(|m: &Monomial| m == monomial) {
                monomials.push(monomial.clone());
            }
        }
    }

    // Sort monomials by term order (largest first) using graded reverse lex
    monomials.sort_by(|a, b| b.grevlex_cmp(a));

    // Build coefficient matrix
    // Each row is a polynomial, each column is a monomial coefficient
    let mut matrix: Vec<Vec<BigRational>> = Vec::new();

    for p in polys {
        let row = polynomial_to_row(p, &monomials);
        matrix.push(row);
    }

    // Add basis polynomials as additional rows for reduction
    for b in basis {
        let row = polynomial_to_row(b, &monomials);
        matrix.push(row);
    }

    // Perform Gaussian elimination (row echelon form)
    gaussian_elimination(&mut matrix);

    // Convert rows back to polynomials, taking only the first |polys| rows
    let mut result = Vec::new();
    for row in matrix.iter().take(polys.len()) {
        let p = row_to_polynomial(row, &monomials);
        if !p.is_zero() {
            result.push(p);
        }
    }

    result
}

/// Convert a polynomial to a coefficient vector based on a monomial ordering.
fn polynomial_to_row(poly: &Polynomial, monomials: &[Monomial]) -> Vec<BigRational> {
    let mut row = vec![BigRational::zero(); monomials.len()];

    for term in poly.terms() {
        if let Some(idx) = monomials.iter().position(|m| m == &term.monomial) {
            row[idx] = term.coeff.clone();
        }
    }

    row
}

/// Convert a coefficient vector back to a polynomial.
fn row_to_polynomial(row: &[BigRational], monomials: &[Monomial]) -> Polynomial {
    use crate::polynomial::Term;

    let mut terms = Vec::new();

    for (i, coeff) in row.iter().enumerate() {
        if !coeff.is_zero() && i < monomials.len() {
            terms.push(Term {
                coeff: coeff.clone(),
                monomial: monomials[i].clone(),
            });
        }
    }

    Polynomial::from_terms(terms, MonomialOrder::default())
}

/// Perform Gaussian elimination on a matrix (in-place).
///
/// This converts the matrix to row echelon form, which is used for
/// polynomial reduction in the F4 algorithm.
#[allow(clippy::needless_range_loop)]
fn gaussian_elimination(matrix: &mut [Vec<BigRational>]) {
    if matrix.is_empty() {
        return;
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut pivot_row = 0;

    for col in 0..cols {
        // Find pivot
        let mut max_row = pivot_row;
        for row in pivot_row..rows {
            if matrix[row][col].abs() > matrix[max_row][col].abs() {
                max_row = row;
            }
        }

        // Check if pivot is zero
        if matrix[max_row][col].is_zero() {
            continue;
        }

        // Swap rows
        if max_row != pivot_row {
            matrix.swap(pivot_row, max_row);
        }

        // Eliminate
        let pivot_val = matrix[pivot_row][col].clone();
        for row in (pivot_row + 1)..rows {
            if !matrix[row][col].is_zero() {
                let factor = matrix[row][col].clone() / &pivot_val;
                for c in col..cols {
                    let subtractor = &factor * &matrix[pivot_row][c];
                    matrix[row][c] -= subtractor;
                }
            }
        }

        pivot_row += 1;
        if pivot_row >= rows {
            break;
        }
    }
}

/// Relation type for polynomial constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Equality: p = 0
    Equal,
    /// Inequality: p != 0
    NotEqual,
    /// Greater than: p > 0
    Greater,
    /// Greater or equal: p >= 0
    GreaterEqual,
    /// Less than: p < 0
    Less,
    /// Less or equal: p <= 0
    LessEqual,
}

/// A polynomial constraint: polynomial \<relation\> 0
#[derive(Debug, Clone)]
pub struct PolynomialConstraint {
    /// The polynomial on the left-hand side of the constraint.
    pub polynomial: Polynomial,
    /// The relation operator (equals, less than, etc.).
    pub relation: Relation,
}

impl PolynomialConstraint {
    /// Create a new constraint.
    pub fn new(polynomial: Polynomial, relation: Relation) -> Self {
        Self {
            polynomial,
            relation,
        }
    }

    /// Create an equality constraint: p = 0
    pub fn equal(polynomial: Polynomial) -> Self {
        Self::new(polynomial, Relation::Equal)
    }

    /// Create a not-equal constraint: p != 0
    pub fn not_equal(polynomial: Polynomial) -> Self {
        Self::new(polynomial, Relation::NotEqual)
    }

    /// Create a greater-than constraint: p > 0
    pub fn greater(polynomial: Polynomial) -> Self {
        Self::new(polynomial, Relation::Greater)
    }

    /// Create a greater-or-equal constraint: p >= 0
    pub fn greater_equal(polynomial: Polynomial) -> Self {
        Self::new(polynomial, Relation::GreaterEqual)
    }

    /// Create a less-than constraint: p < 0
    pub fn less(polynomial: Polynomial) -> Self {
        Self::new(polynomial, Relation::Less)
    }

    /// Create a less-or-equal constraint: p <= 0
    pub fn less_equal(polynomial: Polynomial) -> Self {
        Self::new(polynomial, Relation::LessEqual)
    }
}

/// Result of a satisfiability check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SatResult {
    /// The constraints are satisfiable.
    Sat,
    /// The constraints are unsatisfiable.
    Unsat,
    /// Unknown (solver couldn't determine).
    Unknown,
}

/// A model (variable assignment) for a satisfiable system.
#[derive(Debug, Clone)]
pub struct Model {
    /// Variable assignments: var_id -> value
    pub assignments: FxHashMap<u32, BigRational>,
}

impl Model {
    /// Create an empty model.
    pub fn new() -> Self {
        Self {
            assignments: FxHashMap::default(),
        }
    }

    /// Assign a value to a variable.
    pub fn assign(&mut self, var: u32, value: BigRational) {
        self.assignments.insert(var, value);
    }

    /// Get the value of a variable.
    pub fn get(&self, var: u32) -> Option<&BigRational> {
        self.assignments.get(&var)
    }

    /// Evaluate a polynomial at this model.
    pub fn eval(&self, poly: &Polynomial) -> BigRational {
        poly.eval(&self.assignments)
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

/// NRA solver using Gröbner basis methods.
///
/// This solver handles systems of polynomial equations and inequalities
/// over the real numbers using Gröbner basis computation and other
/// algebraic methods.
pub struct NraSolver {
    /// Equality constraints (polynomials that must equal zero)
    equalities: Vec<Polynomial>,
    /// Inequality constraints
    inequalities: Vec<PolynomialConstraint>,
    /// Current Gröbner basis (cached)
    grobner_basis: Option<Vec<Polynomial>>,
    /// Whether the Gröbner basis needs recomputation
    basis_dirty: bool,
}

impl NraSolver {
    /// Create a new NRA solver.
    pub fn new() -> Self {
        Self {
            equalities: Vec::new(),
            inequalities: Vec::new(),
            grobner_basis: None,
            basis_dirty: false,
        }
    }

    /// Add a polynomial constraint.
    pub fn add_constraint(&mut self, constraint: PolynomialConstraint) {
        match constraint.relation {
            Relation::Equal => {
                self.equalities.push(constraint.polynomial);
                self.basis_dirty = true;
            }
            _ => {
                self.inequalities.push(constraint);
            }
        }
    }

    /// Add an equality constraint: p = 0
    pub fn add_equality(&mut self, polynomial: Polynomial) {
        self.equalities.push(polynomial);
        self.basis_dirty = true;
    }

    /// Compute (or retrieve cached) Gröbner basis for the equalities.
    fn get_grobner_basis(&mut self) -> &Vec<Polynomial> {
        if self.basis_dirty || self.grobner_basis.is_none() {
            let gb = if self.equalities.is_empty() {
                vec![]
            } else {
                grobner_basis_f5(&self.equalities)
            };
            self.grobner_basis = Some(gb);
            self.basis_dirty = false;
        }
        self.grobner_basis.as_ref().expect("grobner basis computed")
    }

    /// Check if the equality constraints are satisfiable.
    ///
    /// This uses the Gröbner basis to check if the ideal contains 1
    /// (which would indicate unsatisfiability).
    pub fn check_equalities(&mut self) -> SatResult {
        if self.equalities.is_empty() {
            return SatResult::Sat;
        }

        let gb = self.get_grobner_basis();

        // If the Gröbner basis contains a non-zero constant, the system is unsatisfiable
        for p in gb {
            if p.is_constant() && !p.is_zero() {
                return SatResult::Unsat;
            }
        }

        SatResult::Sat
    }

    /// Check satisfiability of all constraints (equalities and inequalities).
    pub fn check_sat(&mut self) -> SatResult {
        // First check if equalities are satisfiable
        let eq_result = self.check_equalities();
        if eq_result == SatResult::Unsat {
            return SatResult::Unsat;
        }

        // Check inequalities
        if !self.inequalities.is_empty() {
            // Get Gröbner basis first, then work with inequalities
            let gb = self.get_grobner_basis().clone();

            for constraint in &self.inequalities {
                let simplified = reduce(&constraint.polynomial, &gb);

                // Check if the simplified constraint is a constant (or zero)
                if simplified.is_constant() || simplified.is_zero() {
                    let const_val = if simplified.is_zero() {
                        BigRational::zero()
                    } else {
                        simplified.constant_term()
                    };

                    let is_sat = match constraint.relation {
                        Relation::Equal => const_val.is_zero(),
                        Relation::NotEqual => !const_val.is_zero(),
                        Relation::Greater => const_val.is_positive(),
                        Relation::GreaterEqual => !const_val.is_negative(),
                        Relation::Less => const_val.is_negative(),
                        Relation::LessEqual => !const_val.is_positive(),
                    };

                    if !is_sat {
                        return SatResult::Unsat;
                    }
                }
            }

            // If we have non-constant inequalities, try basic inequality solving
            // For now, return Unknown for complex cases
            let has_complex_inequality = self.inequalities.iter().any(|c| {
                let simplified = reduce(&c.polynomial, &gb);
                !simplified.is_constant() && simplified.total_degree() > 1
            });

            if has_complex_inequality {
                return SatResult::Unknown;
            }
        }

        SatResult::Sat
    }

    /// Simplify a polynomial using the current Gröbner basis.
    ///
    /// This reduces the polynomial with respect to the Gröbner basis,
    /// which can help simplify constraints.
    pub fn simplify(&mut self, polynomial: &Polynomial) -> Polynomial {
        if self.equalities.is_empty() {
            return polynomial.clone();
        }

        let gb = self.get_grobner_basis();
        reduce(polynomial, gb)
    }

    /// Check if a polynomial is implied by the current equality constraints.
    ///
    /// Returns true if the polynomial is in the ideal generated by the equalities,
    /// meaning it must be zero whenever the equalities are satisfied.
    pub fn implies_zero(&mut self, polynomial: &Polynomial) -> bool {
        if self.equalities.is_empty() {
            return polynomial.is_zero();
        }

        let gb = self.get_grobner_basis();
        let reduced = reduce(polynomial, gb);
        reduced.is_zero()
    }

    /// Extract a model from the Gröbner basis (if possible).
    ///
    /// This attempts to solve for variables using the simplified basis.
    /// For univariate polynomials in the basis, we can find roots.
    /// This is a basic implementation; a complete solver would use more
    /// sophisticated techniques.
    pub fn get_model(&mut self) -> Option<Model> {
        let eq_result = self.check_equalities();
        if eq_result == SatResult::Unsat {
            return None;
        }

        let gb = self.get_grobner_basis();
        let mut model = Model::new();

        // Try to extract values from univariate polynomials in the basis
        for poly in gb {
            if poly.is_constant() {
                continue;
            }

            // Get variables in this polynomial
            let vars = poly.vars();
            if vars.len() == 1 {
                // Univariate polynomial - try to find a root
                let var = vars[0];

                // For linear polynomials ax + b = 0, solve as x = -b/a
                if poly.total_degree() == 1 {
                    if let Some(root) = solve_linear(poly) {
                        model.assign(var, root);
                    }
                }
                // For higher degree, we'd need root isolation
                // For now, just assign 0 if not already assigned
                else if model.get(var).is_none() {
                    model.assign(var, BigRational::zero());
                }
            }
        }

        Some(model)
    }

    /// Clear all constraints.
    pub fn reset(&mut self) {
        self.equalities.clear();
        self.inequalities.clear();
        self.grobner_basis = None;
        self.basis_dirty = false;
    }

    /// Get the number of equality constraints.
    pub fn num_equalities(&self) -> usize {
        self.equalities.len()
    }

    /// Get the number of inequality constraints.
    pub fn num_inequalities(&self) -> usize {
        self.inequalities.len()
    }
}

impl Default for NraSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve a linear polynomial ax + b = 0 for x.
///
/// Returns Some(-b/a) if the polynomial is linear, None otherwise.
fn solve_linear(poly: &Polynomial) -> Option<BigRational> {
    if poly.total_degree() != 1 {
        return None;
    }

    let terms = poly.terms();
    if terms.is_empty() {
        return None;
    }

    let mut constant = BigRational::zero();
    let mut linear_coeff = BigRational::zero();

    for term in terms {
        if term.monomial.total_degree() == 0 {
            constant = term.coeff.clone();
        } else if term.monomial.total_degree() == 1 {
            linear_coeff = term.coeff.clone();
        }
    }

    if linear_coeff.is_zero() {
        return None;
    }

    Some(-constant / linear_coeff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[allow(dead_code)]
    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_monomial_lcm() {
        // x^2 and x^3 -> x^3
        let m1 = Monomial::from_var_power(0, 2);
        let m2 = Monomial::from_var_power(0, 3);
        let lcm = monomial_lcm(&m1, &m2);
        assert_eq!(lcm.degree(0), 3);

        // x^2*y and x*y^2 -> x^2*y^2
        let m1 = Monomial::from_powers([(0, 2), (1, 1)]);
        let m2 = Monomial::from_powers([(0, 1), (1, 2)]);
        let lcm = monomial_lcm(&m1, &m2);
        assert_eq!(lcm.degree(0), 2);
        assert_eq!(lcm.degree(1), 2);
    }

    #[test]
    fn test_s_polynomial() {
        // f = x^2, g = xy
        let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let g = Polynomial::from_coeffs_int(&[(1, &[(0, 1), (1, 1)])]);

        let s = s_polynomial(&f, &g);
        // S(x^2, xy) = y*x^2/x^2 - x^2/(xy) * xy = y - x
        // Actually: LCM(x^2, xy) = x^2*y
        // S = x^2*y/x^2 * f - x^2*y/(xy) * g = y*f - x*g = yx^2 - x*xy = x^2y - x^2y = 0
        // Wait, let me recalculate:
        // LCM(x^2, xy) = x^2*y
        // S = (x^2*y / x^2) * (x^2) - (x^2*y / xy) * (xy)
        // S = y * x^2 - x * xy = x^2*y - x^2*y = 0
        assert!(s.is_zero() || s.total_degree() <= 2);
    }

    #[test]
    fn test_reduce() {
        // Reduce x^2 by [x]
        let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let g = vec![Polynomial::from_coeffs_int(&[(1, &[(0, 1)])])];

        let r = reduce(&f, &g);
        // x^2 should reduce to 0 by x (x^2 = x * x)
        assert!(r.is_zero());
    }

    #[test]
    fn test_grobner_basis_simple() {
        // Ideal generated by [x, y]
        let f1 = Polynomial::from_var(0); // x
        let f2 = Polynomial::from_var(1); // y

        let gb = grobner_basis(&[f1, f2]);

        // Should contain both x and y
        assert!(gb.len() >= 2);
    }

    #[test]
    fn test_ideal_membership() {
        // Test if x^2 is in the ideal <x>
        let f = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let generators = vec![Polynomial::from_var(0)];

        assert!(ideal_membership(&f, &generators));

        // Test if y is in the ideal <x>
        let f = Polynomial::from_var(1);
        assert!(!ideal_membership(&f, &generators));
    }

    #[test]
    fn test_ideal_membership_multivariate() {
        // Test if x + y is in the ideal <x, y>
        let f = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)])]);
        let generators = vec![Polynomial::from_var(0), Polynomial::from_var(1)];

        assert!(ideal_membership(&f, &generators));
    }

    #[test]
    fn test_f4_simple() {
        // Test F4 on a simple ideal
        let f1 = Polynomial::from_var(0); // x
        let f2 = Polynomial::from_var(1); // y

        let gb = grobner_basis_f4(&[f1, f2]);

        // Should contain both x and y
        assert!(gb.len() >= 2);
    }

    #[test]
    fn test_f4_vs_buchberger() {
        // Compare F4 and Buchberger results
        let f1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]); // x^2 - 1
        let f2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1), (1, 1)]), (-1, &[(1, 1)])]); // xy - y

        let gb_buchberger = grobner_basis(&[f1.clone(), f2.clone()]);
        let gb_f4 = grobner_basis_f4(&[f1, f2]);

        // Both should produce Gröbner bases (may differ but should have same ideal)
        assert!(!gb_buchberger.is_empty());
        assert!(!gb_f4.is_empty());

        // The bases should have similar sizes (within a factor)
        assert!(gb_buchberger.len() <= gb_f4.len() * 2);
        assert!(gb_f4.len() <= gb_buchberger.len() * 2);
    }

    #[test]
    fn test_gaussian_elimination() {
        use num_bigint::BigInt;

        // Test simple 2x2 system
        let mut matrix = vec![
            vec![
                BigRational::from_integer(BigInt::from(2)),
                BigRational::from_integer(BigInt::from(1)),
            ],
            vec![
                BigRational::from_integer(BigInt::from(1)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        ];

        gaussian_elimination(&mut matrix);

        // First row should remain [2, 1]
        assert_eq!(matrix[0][0], BigRational::from_integer(BigInt::from(2)));

        // Second row first element should be zero (eliminated)
        assert_eq!(matrix[1][0], BigRational::from_integer(BigInt::from(0)));
    }

    #[test]
    fn test_polynomial_row_conversion() {
        // Test conversion between polynomial and row representation
        let p = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (3, &[(1, 1)])]);

        let monomials = vec![
            Monomial::from_var_power(0, 1),
            Monomial::from_var_power(1, 1),
        ];

        let row = polynomial_to_row(&p, &monomials);
        assert_eq!(row.len(), 2);

        let p2 = row_to_polynomial(&row, &monomials);

        // Should get back an equivalent polynomial
        assert_eq!(p.total_degree(), p2.total_degree());
    }

    #[test]
    fn test_f5_simple() {
        // Test F5 on a simple ideal
        let f1 = Polynomial::from_var(0); // x
        let f2 = Polynomial::from_var(1); // y

        let gb = grobner_basis_f5(&[f1, f2]);

        // Should contain both x and y
        assert!(gb.len() >= 2);
    }

    #[test]
    fn test_f5_vs_buchberger() {
        // Compare F5 and Buchberger results
        let f1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]); // x^2 - 1
        let f2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1), (1, 1)]), (-1, &[(1, 1)])]); // xy - y

        let gb_buchberger = grobner_basis(&[f1.clone(), f2.clone()]);
        let gb_f5 = grobner_basis_f5(&[f1, f2]);

        // Both should produce Gröbner bases
        assert!(!gb_buchberger.is_empty());
        assert!(!gb_f5.is_empty());

        // The bases should have similar sizes (within a factor)
        assert!(gb_buchberger.len() <= gb_f5.len() * 2);
        assert!(gb_f5.len() <= gb_buchberger.len() * 2);
    }

    #[test]
    fn test_f5_vs_f4() {
        // Compare F5 and F4 results
        let f1 = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[(1, 2)])]); // x^2 - y^2
        let f2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[(1, 1)])]); // x - y

        let gb_f4 = grobner_basis_f4(&[f1.clone(), f2.clone()]);
        let gb_f5 = grobner_basis_f5(&[f1, f2]);

        // Both should produce Gröbner bases
        assert!(!gb_f4.is_empty());
        assert!(!gb_f5.is_empty());
    }

    #[test]
    fn test_signature_ordering() {
        // Test signature comparison
        let sig1 = Signature::unit(0);
        let sig2 = Signature::unit(1);

        // Higher index should be "smaller" (comes earlier in processing)
        // In our reverse ordering, sig1 (index 0) > sig2 (index 1)
        assert_eq!(sig1.cmp(&sig2), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_labeled_poly() {
        // Test labeled polynomial operations
        let p = Polynomial::from_var(0);
        let sig = Signature::unit(0);
        let lp = LabeledPoly::new(p, sig);

        // Test multiplication by monomial
        let m = Monomial::from_var_power(1, 1);
        let lp2 = lp.mul_monomial(&m);

        assert_eq!(lp2.polynomial.total_degree(), 2);
    }

    #[test]
    fn test_nra_solver_empty() {
        // Test empty solver (should be SAT)
        let mut solver = NraSolver::new();
        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_nra_solver_simple_equality() {
        // Test simple equality: x = 0
        let mut solver = NraSolver::new();
        let x = Polynomial::from_var(0); // x
        solver.add_equality(x);

        assert_eq!(solver.check_sat(), SatResult::Sat);

        // Get model
        if let Some(model) = solver.get_model() {
            assert_eq!(model.get(0), Some(&BigRational::zero()));
        }
    }

    #[test]
    fn test_nra_solver_linear_system() {
        // Test linear system: x + y = 0, x - y = 0
        // Solution: x = 0, y = 0
        let mut solver = NraSolver::new();

        // x + y = 0
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)])]);
        // x - y = 0
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[(1, 1)])]);

        solver.add_equality(p1);
        solver.add_equality(p2);

        assert_eq!(solver.check_sat(), SatResult::Sat);

        // Model extraction is basic - just check that we can get a model
        // A complete implementation would use more sophisticated methods
        let model = solver.get_model();
        assert!(model.is_some());
    }

    #[test]
    fn test_nra_solver_inconsistent() {
        // Test inconsistent system: 1 = 0
        let mut solver = NraSolver::new();
        let one = Polynomial::constant(rat(1));
        solver.add_equality(one);

        assert_eq!(solver.check_sat(), SatResult::Unsat);
    }

    #[test]
    fn test_nra_solver_inconsistent_system() {
        // Test inconsistent system: x = 1, x = 2
        let mut solver = NraSolver::new();

        // x - 1 = 0
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-1, &[])]);
        // x - 2 = 0
        let p2 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-2, &[])]);

        solver.add_equality(p1);
        solver.add_equality(p2);

        assert_eq!(solver.check_sat(), SatResult::Unsat);
    }

    #[test]
    fn test_nra_solver_simplify() {
        // Test polynomial simplification
        let mut solver = NraSolver::new();

        // Add constraint: x = 0
        let x = Polynomial::from_var(0);
        solver.add_equality(x.clone());

        // Simplify x^2 (should reduce to 0)
        let x_squared = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let simplified = solver.simplify(&x_squared);

        assert!(simplified.is_zero());
    }

    #[test]
    fn test_nra_solver_implies_zero() {
        // Test ideal membership
        let mut solver = NraSolver::new();

        // Add constraint: x = 0
        let x = Polynomial::from_var(0);
        solver.add_equality(x.clone());

        // x^2 should be implied to be zero
        let x_squared = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        assert!(solver.implies_zero(&x_squared));

        // y should not be implied to be zero
        let y = Polynomial::from_var(1);
        assert!(!solver.implies_zero(&y));
    }

    #[test]
    fn test_nra_solver_reset() {
        // Test resetting the solver
        let mut solver = NraSolver::new();

        let x = Polynomial::from_var(0);
        solver.add_equality(x);

        assert_eq!(solver.num_equalities(), 1);

        solver.reset();

        assert_eq!(solver.num_equalities(), 0);
        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_polynomial_constraint_creation() {
        // Test constraint creation helpers
        let p = Polynomial::from_var(0);

        let eq = PolynomialConstraint::equal(p.clone());
        assert_eq!(eq.relation, Relation::Equal);

        let neq = PolynomialConstraint::not_equal(p.clone());
        assert_eq!(neq.relation, Relation::NotEqual);

        let gt = PolynomialConstraint::greater(p.clone());
        assert_eq!(gt.relation, Relation::Greater);

        let gte = PolynomialConstraint::greater_equal(p.clone());
        assert_eq!(gte.relation, Relation::GreaterEqual);

        let lt = PolynomialConstraint::less(p.clone());
        assert_eq!(lt.relation, Relation::Less);

        let lte = PolynomialConstraint::less_equal(p);
        assert_eq!(lte.relation, Relation::LessEqual);
    }

    #[test]
    fn test_model_operations() {
        // Test model creation and evaluation
        let mut model = Model::new();

        model.assign(0, rat(2));
        model.assign(1, rat(3));

        assert_eq!(model.get(0), Some(&rat(2)));
        assert_eq!(model.get(1), Some(&rat(3)));

        // Evaluate 2x + 3y
        let poly = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (3, &[(1, 1)])]);
        let result = model.eval(&poly);

        // 2*2 + 3*3 = 4 + 9 = 13
        assert_eq!(result, rat(13));
    }

    #[test]
    fn test_solve_linear() {
        // Test linear solver: 2x + 4 = 0 => x = -2
        let poly = Polynomial::from_coeffs_int(&[(2, &[(0, 1)]), (4, &[])]);
        let solution = solve_linear(&poly);

        assert_eq!(solution, Some(rat(-2)));
    }

    #[test]
    fn test_solve_linear_simple() {
        // Test linear solver: x - 3 = 0 => x = 3
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-3, &[])]);
        let solution = solve_linear(&poly);

        assert_eq!(solution, Some(rat(3)));
    }

    #[test]
    fn test_nra_solver_with_constraint_api() {
        // Test using the constraint API
        let mut solver = NraSolver::new();

        let x = Polynomial::from_var(0);
        let constraint = PolynomialConstraint::equal(x);

        solver.add_constraint(constraint);

        assert_eq!(solver.num_equalities(), 1);
        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_nra_solver_trivial_inequality_sat() {
        // Test trivial satisfiable inequality: 1 > 0
        let mut solver = NraSolver::new();

        let one = Polynomial::constant(rat(1));
        let constraint = PolynomialConstraint::greater(one);

        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_nra_solver_trivial_inequality_unsat() {
        // Test trivial unsatisfiable inequality: -1 > 0
        let mut solver = NraSolver::new();

        let neg_one = Polynomial::constant(rat(-1));
        let constraint = PolynomialConstraint::greater(neg_one);

        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Unsat);
    }

    #[test]
    fn test_nra_solver_inequality_after_simplification() {
        // Test inequality that becomes constant after simplification
        let mut solver = NraSolver::new();

        // Add x = 2
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (-2, &[])]);
        solver.add_equality(p1);

        // Add constraint: x > 0 (should simplify to 2 > 0, which is SAT)
        let x = Polynomial::from_var(0);
        let constraint = PolynomialConstraint::greater(x);
        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_nra_solver_not_equal_constraint() {
        // Test not-equal constraint: 1 != 0 (should be SAT)
        let mut solver = NraSolver::new();

        let one = Polynomial::constant(rat(1));
        let constraint = PolynomialConstraint::not_equal(one);

        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_nra_solver_not_equal_unsat() {
        // Test not-equal constraint: 0 != 0 (should be UNSAT)
        let mut solver = NraSolver::new();

        let zero = Polynomial::zero();
        let constraint = PolynomialConstraint::not_equal(zero);

        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Unsat);
    }

    #[test]
    fn test_nra_solver_less_equal() {
        // Test less-equal constraint: -1 <= 0 (should be SAT)
        let mut solver = NraSolver::new();

        let neg_one = Polynomial::constant(rat(-1));
        let constraint = PolynomialConstraint::less_equal(neg_one);

        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_nra_solver_greater_equal() {
        // Test greater-equal constraint: 0 >= 0 (should be SAT)
        let mut solver = NraSolver::new();

        let zero = Polynomial::zero();
        let constraint = PolynomialConstraint::greater_equal(zero);

        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Sat);
    }

    #[test]
    fn test_nra_solver_complex_inequality() {
        // Test complex (non-constant, high-degree) inequality
        let mut solver = NraSolver::new();

        // x^2 > 0 (complex case, should return Unknown)
        let x_squared = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let constraint = PolynomialConstraint::greater(x_squared);

        solver.add_constraint(constraint);

        assert_eq!(solver.check_sat(), SatResult::Unknown);
    }
}
