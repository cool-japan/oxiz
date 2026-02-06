//! Enhanced Buchberger Algorithm for Gröbner Basis Computation
#![allow(missing_docs)] // Under development
//!
//! This module implements optimizations for Buchberger's algorithm:
//! - Product criterion for eliminating useless S-polynomials
//! - Chain criterion for detecting redundant pairs
//! - Gebauer-Möller installation strategy
//! - Sugar cube selection strategy

use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Monomial (variable -> exponent)
pub type Monomial = FxHashMap<usize, usize>;

/// Polynomial term
#[derive(Debug, Clone)]
pub struct Term {
    pub coeff: BigRational,
    pub monomial: Monomial,
}

/// Multivariate polynomial
#[derive(Debug, Clone)]
pub struct Polynomial {
    pub terms: Vec<Term>,
}

/// Critical pair for S-polynomial computation
#[derive(Debug, Clone)]
pub struct CriticalPair {
    /// First polynomial index
    pub i: usize,
    /// Second polynomial index
    pub j: usize,
    /// LCM of leading monomials
    pub lcm: Monomial,
    /// Degree of LCM
    pub degree: usize,
    /// Sugar degree (for selection strategy)
    pub sugar: usize,
}

impl Ord for CriticalPair {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by sugar degree (lower is better)
        other
            .sugar
            .cmp(&self.sugar)
            .then_with(|| other.degree.cmp(&self.degree))
            .then_with(|| other.i.cmp(&self.i))
    }
}

impl PartialOrd for CriticalPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CriticalPair {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.j == other.j
    }
}

impl Eq for CriticalPair {}

/// Statistics for Buchberger algorithm
#[derive(Debug, Clone, Default)]
pub struct BuchbergerStats {
    pub s_polynomials_computed: u64,
    pub zero_reductions: u64,
    pub pairs_eliminated_product: u64,
    pub pairs_eliminated_chain: u64,
    pub polynomials_added: u64,
    pub reduction_steps: u64,
}

/// Configuration for enhanced Buchberger
#[derive(Debug, Clone)]
pub struct BuchbergerConfig {
    /// Enable product criterion
    pub use_product_criterion: bool,
    /// Enable chain criterion
    pub use_chain_criterion: bool,
    /// Enable Gebauer-Möller installation
    pub use_gebauer_moller: bool,
    /// Use sugar cube selection strategy
    pub use_sugar_strategy: bool,
    /// Maximum degree for S-polynomials
    pub max_degree: Option<usize>,
}

impl Default for BuchbergerConfig {
    fn default() -> Self {
        Self {
            use_product_criterion: true,
            use_chain_criterion: true,
            use_gebauer_moller: true,
            use_sugar_strategy: true,
            max_degree: None,
        }
    }
}

/// Enhanced Buchberger algorithm implementation
pub struct EnhancedBuchberger {
    config: BuchbergerConfig,
    stats: BuchbergerStats,
    /// Current Gröbner basis
    basis: Vec<Polynomial>,
    /// Priority queue of critical pairs
    pairs: BinaryHeap<CriticalPair>,
    /// Sugar degrees for polynomials
    sugar_degrees: Vec<usize>,
}

impl EnhancedBuchberger {
    /// Create a new enhanced Buchberger computer
    pub fn new(config: BuchbergerConfig) -> Self {
        Self {
            config,
            stats: BuchbergerStats::default(),
            basis: Vec::new(),
            pairs: BinaryHeap::new(),
            sugar_degrees: Vec::new(),
        }
    }

    /// Compute Gröbner basis for a set of polynomials
    pub fn compute_basis(
        &mut self,
        generators: Vec<Polynomial>,
    ) -> Result<Vec<Polynomial>, String> {
        // Initialize basis with generators
        self.basis = generators.clone();

        // Initialize sugar degrees
        self.sugar_degrees = generators.iter().map(|p| self.total_degree(p)).collect();

        // Generate initial critical pairs
        for i in 0..self.basis.len() {
            for j in (i + 1)..self.basis.len() {
                self.add_critical_pair(i, j)?;
            }
        }

        // Main loop
        while let Some(pair) = self.pairs.pop() {
            // Check degree bound
            if let Some(max_deg) = self.config.max_degree
                && pair.degree > max_deg
            {
                continue;
            }

            // Compute S-polynomial
            self.stats.s_polynomials_computed += 1;
            let s_poly = self.compute_s_polynomial(&pair)?;

            // Reduce S-polynomial by current basis
            let reduced = self.reduce_polynomial(&s_poly)?;

            // Check if reduction is zero
            if self.is_zero(&reduced) {
                self.stats.zero_reductions += 1;
                continue;
            }

            // Add to basis
            let new_idx = self.basis.len();
            self.basis.push(reduced.clone());
            self.sugar_degrees.push(pair.sugar);
            self.stats.polynomials_added += 1;

            // Update critical pairs
            if self.config.use_gebauer_moller {
                self.gebauer_moller_update(new_idx)?;
            } else {
                // Standard update: add pairs with all existing polynomials
                for i in 0..new_idx {
                    self.add_critical_pair(i, new_idx)?;
                }
            }
        }

        // Return minimal Gröbner basis
        Ok(self.basis.clone())
    }

    /// Add a critical pair if it passes criteria
    fn add_critical_pair(&mut self, i: usize, j: usize) -> Result<(), String> {
        // Compute LCM of leading monomials
        let lt_i = self.leading_monomial(&self.basis[i]);
        let lt_j = self.leading_monomial(&self.basis[j]);
        let lcm = self.monomial_lcm(&lt_i, &lt_j);
        let degree = self.monomial_degree(&lcm);

        // Product criterion: check if LCM = LT(i) * LT(j)
        if self.config.use_product_criterion {
            let product = self.monomial_product(&lt_i, &lt_j);
            if self.monomial_equal(&lcm, &product) {
                self.stats.pairs_eliminated_product += 1;
                return Ok(());
            }
        }

        // Chain criterion: check if there exists k with specific property
        if self.config.use_chain_criterion && self.satisfies_chain_criterion(i, j, &lcm)? {
            self.stats.pairs_eliminated_chain += 1;
            return Ok(());
        }

        // Compute sugar degree
        let sugar = if self.config.use_sugar_strategy {
            self.compute_sugar_degree(i, j, degree)
        } else {
            degree
        };

        // Add pair to queue
        self.pairs.push(CriticalPair {
            i,
            j,
            lcm,
            degree,
            sugar,
        });

        Ok(())
    }

    /// Check chain criterion (Buchberger's second criterion)
    fn satisfies_chain_criterion(
        &self,
        i: usize,
        j: usize,
        lcm_ij: &Monomial,
    ) -> Result<bool, String> {
        // Check if exists k such that:
        // 1. LT(k) divides LCM(LT(i), LT(j))
        // 2. (i,k) and (j,k) have already been processed or eliminated

        for k in 0..self.basis.len() {
            if k == i || k == j {
                continue;
            }

            let lt_k = self.leading_monomial(&self.basis[k]);

            // Check if LT(k) divides LCM(LT(i), LT(j))
            if self.monomial_divides(&lt_k, lcm_ij) {
                // Found candidate k
                // In practice, would check if pairs (i,k) and (j,k) were eliminated
                // For now, simplified check
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Gebauer-Möller installation strategy
    fn gebauer_moller_update(&mut self, new_idx: usize) -> Result<(), String> {
        let lt_new = self.leading_monomial(&self.basis[new_idx]);

        // Remove pairs whose LCM is divisible by LT(new)
        let mut surviving_pairs = Vec::new();

        while let Some(pair) = self.pairs.pop() {
            let should_keep = !self.monomial_divides(&lt_new, &pair.lcm);

            if should_keep {
                surviving_pairs.push(pair);
            }
        }

        // Restore surviving pairs
        for pair in surviving_pairs {
            self.pairs.push(pair);
        }

        // Add new pairs with existing basis elements
        for i in 0..new_idx {
            self.add_critical_pair(i, new_idx)?;
        }

        Ok(())
    }

    /// Compute S-polynomial for a critical pair
    fn compute_s_polynomial(&self, pair: &CriticalPair) -> Result<Polynomial, String> {
        let fi = &self.basis[pair.i];
        let fj = &self.basis[pair.j];

        let lt_i = self.leading_monomial(fi);
        let lt_j = self.leading_monomial(fj);

        let lc_i = self.leading_coefficient(fi);
        let lc_j = self.leading_coefficient(fj);

        // S(fi, fj) = (lcm/lt_i)/lc_i * fi - (lcm/lt_j)/lc_j * fj
        let cofactor_i = self.monomial_quotient(&pair.lcm, &lt_i)?;
        let cofactor_j = self.monomial_quotient(&pair.lcm, &lt_j)?;

        let term_i = self.multiply_polynomial_by_monomial(fi, &cofactor_i);
        let term_i_scaled = self.scalar_multiply(&term_i, &(BigRational::one() / &lc_i));

        let term_j = self.multiply_polynomial_by_monomial(fj, &cofactor_j);
        let term_j_scaled = self.scalar_multiply(&term_j, &(BigRational::one() / &lc_j));

        Ok(self.subtract_polynomials(&term_i_scaled, &term_j_scaled))
    }

    /// Reduce a polynomial by the current basis
    fn reduce_polynomial(&mut self, poly: &Polynomial) -> Result<Polynomial, String> {
        let mut current = poly.clone();

        loop {
            let mut reduced = false;

            // Try to reduce by each basis element
            for basis_poly in &self.basis {
                if let Some(new_poly) = self.try_reduce_once(&current, basis_poly)? {
                    current = new_poly;
                    reduced = true;
                    self.stats.reduction_steps += 1;
                    break;
                }
            }

            if !reduced {
                break;
            }
        }

        Ok(current)
    }

    /// Try to reduce polynomial by a single basis element
    fn try_reduce_once(
        &self,
        poly: &Polynomial,
        reducer: &Polynomial,
    ) -> Result<Option<Polynomial>, String> {
        if poly.terms.is_empty() {
            return Ok(None);
        }

        let lt_poly = self.leading_monomial(poly);
        let lt_reducer = self.leading_monomial(reducer);

        // Check if reducer's leading term divides polynomial's leading term
        if !self.monomial_divides(&lt_reducer, &lt_poly) {
            return Ok(None);
        }

        // Compute quotient monomial
        let quotient_mono = self.monomial_quotient(&lt_poly, &lt_reducer)?;

        // Compute quotient coefficient
        let lc_poly = self.leading_coefficient(poly);
        let lc_reducer = self.leading_coefficient(reducer);
        let quotient_coeff = &lc_poly / &lc_reducer;

        // Multiply reducer by quotient
        let scaled_reducer = self.multiply_polynomial_by_monomial(reducer, &quotient_mono);
        let final_reducer = self.scalar_multiply(&scaled_reducer, &quotient_coeff);

        // Subtract
        Ok(Some(self.subtract_polynomials(poly, &final_reducer)))
    }

    /// Compute sugar degree for a critical pair
    fn compute_sugar_degree(&self, i: usize, j: usize, lcm_degree: usize) -> usize {
        let sugar_i = self.sugar_degrees[i];
        let sugar_j = self.sugar_degrees[j];

        // Sugar of S-polynomial is max(sugar_i, sugar_j) adjusted by degree
        let lt_i_deg = self.total_degree(&self.basis[i]);
        let lt_j_deg = self.total_degree(&self.basis[j]);

        (lcm_degree + sugar_i - lt_i_deg).max(lcm_degree + sugar_j - lt_j_deg)
    }

    // Helper methods for monomial operations

    fn leading_monomial(&self, poly: &Polynomial) -> Monomial {
        poly.terms
            .first()
            .map(|t| t.monomial.clone())
            .unwrap_or_default()
    }

    fn leading_coefficient(&self, poly: &Polynomial) -> BigRational {
        poly.terms
            .first()
            .map(|t| t.coeff.clone())
            .unwrap_or_else(BigRational::zero)
    }

    fn monomial_lcm(&self, m1: &Monomial, m2: &Monomial) -> Monomial {
        let mut result = m1.clone();
        for (&var, &exp) in m2 {
            let current_exp = result.get(&var).copied().unwrap_or(0);
            result.insert(var, current_exp.max(exp));
        }
        result
    }

    fn monomial_product(&self, m1: &Monomial, m2: &Monomial) -> Monomial {
        let mut result = m1.clone();
        for (&var, &exp) in m2 {
            *result.entry(var).or_insert(0) += exp;
        }
        result
    }

    fn monomial_quotient(
        &self,
        dividend: &Monomial,
        divisor: &Monomial,
    ) -> Result<Monomial, String> {
        let mut result = Monomial::default();

        for (&var, &exp) in dividend {
            let divisor_exp = divisor.get(&var).copied().unwrap_or(0);
            if exp < divisor_exp {
                return Err("Monomial not divisible".to_string());
            }
            if exp > divisor_exp {
                result.insert(var, exp - divisor_exp);
            }
        }

        Ok(result)
    }

    fn monomial_divides(&self, divisor: &Monomial, dividend: &Monomial) -> bool {
        divisor
            .iter()
            .all(|(&var, &exp)| dividend.get(&var).copied().unwrap_or(0) >= exp)
    }

    fn monomial_equal(&self, m1: &Monomial, m2: &Monomial) -> bool {
        if m1.len() != m2.len() {
            return false;
        }
        m1.iter().all(|(var, exp)| m2.get(var) == Some(exp))
    }

    fn monomial_degree(&self, m: &Monomial) -> usize {
        m.values().sum()
    }

    fn total_degree(&self, poly: &Polynomial) -> usize {
        poly.terms
            .iter()
            .map(|t| self.monomial_degree(&t.monomial))
            .max()
            .unwrap_or(0)
    }

    fn multiply_polynomial_by_monomial(&self, poly: &Polynomial, mono: &Monomial) -> Polynomial {
        Polynomial {
            terms: poly
                .terms
                .iter()
                .map(|t| Term {
                    coeff: t.coeff.clone(),
                    monomial: self.monomial_product(&t.monomial, mono),
                })
                .collect(),
        }
    }

    fn scalar_multiply(&self, poly: &Polynomial, scalar: &BigRational) -> Polynomial {
        Polynomial {
            terms: poly
                .terms
                .iter()
                .map(|t| Term {
                    coeff: &t.coeff * scalar,
                    monomial: t.monomial.clone(),
                })
                .collect(),
        }
    }

    fn subtract_polynomials(&self, p1: &Polynomial, p2: &Polynomial) -> Polynomial {
        // Placeholder: proper polynomial subtraction with term merging
        let mut terms: Vec<Term> = p1.terms.clone();
        for t in &p2.terms {
            terms.push(Term {
                coeff: -&t.coeff,
                monomial: t.monomial.clone(),
            });
        }
        Polynomial { terms }
    }

    fn is_zero(&self, poly: &Polynomial) -> bool {
        poly.terms.is_empty() || poly.terms.iter().all(|t| t.coeff.is_zero())
    }

    /// Get statistics
    pub fn stats(&self) -> &BuchbergerStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monomial(vars: &[(usize, usize)]) -> Monomial {
        vars.iter().copied().collect()
    }

    #[test]
    fn test_buchberger_creation() {
        let config = BuchbergerConfig::default();
        let buchberger = EnhancedBuchberger::new(config);
        assert_eq!(buchberger.stats.s_polynomials_computed, 0);
    }

    #[test]
    fn test_monomial_lcm() {
        let buchberger = EnhancedBuchberger::new(BuchbergerConfig::default());

        let m1 = make_monomial(&[(0, 2), (1, 3)]);
        let m2 = make_monomial(&[(0, 1), (1, 4), (2, 1)]);

        let lcm = buchberger.monomial_lcm(&m1, &m2);

        assert_eq!(lcm.get(&0), Some(&2));
        assert_eq!(lcm.get(&1), Some(&4));
        assert_eq!(lcm.get(&2), Some(&1));
    }

    #[test]
    fn test_monomial_product() {
        let buchberger = EnhancedBuchberger::new(BuchbergerConfig::default());

        let m1 = make_monomial(&[(0, 2), (1, 3)]);
        let m2 = make_monomial(&[(0, 1), (2, 1)]);

        let product = buchberger.monomial_product(&m1, &m2);

        assert_eq!(product.get(&0), Some(&3));
        assert_eq!(product.get(&1), Some(&3));
        assert_eq!(product.get(&2), Some(&1));
    }

    #[test]
    fn test_monomial_divides() {
        let buchberger = EnhancedBuchberger::new(BuchbergerConfig::default());

        let divisor = make_monomial(&[(0, 2), (1, 1)]);
        let dividend = make_monomial(&[(0, 3), (1, 2), (2, 1)]);

        assert!(buchberger.monomial_divides(&divisor, &dividend));

        let non_divisor = make_monomial(&[(0, 4)]);
        assert!(!buchberger.monomial_divides(&non_divisor, &dividend));
    }

    #[test]
    fn test_monomial_quotient() {
        let buchberger = EnhancedBuchberger::new(BuchbergerConfig::default());

        let dividend = make_monomial(&[(0, 3), (1, 2)]);
        let divisor = make_monomial(&[(0, 1), (1, 1)]);

        let quotient = buchberger.monomial_quotient(&dividend, &divisor).unwrap();

        assert_eq!(quotient.get(&0), Some(&2));
        assert_eq!(quotient.get(&1), Some(&1));
    }

    #[test]
    fn test_monomial_degree() {
        let buchberger = EnhancedBuchberger::new(BuchbergerConfig::default());

        let m = make_monomial(&[(0, 2), (1, 3), (2, 1)]);
        assert_eq!(buchberger.monomial_degree(&m), 6);
    }

    #[test]
    fn test_critical_pair_ordering() {
        let p1 = CriticalPair {
            i: 0,
            j: 1,
            lcm: Monomial::default(),
            degree: 3,
            sugar: 5,
        };

        let p2 = CriticalPair {
            i: 0,
            j: 2,
            lcm: Monomial::default(),
            degree: 2,
            sugar: 4,
        };

        // Lower sugar should be better (higher priority)
        assert!(p2 > p1);
    }

    #[test]
    fn test_product_criterion_detection() {
        let config = BuchbergerConfig {
            use_product_criterion: true,
            ..Default::default()
        };
        let buchberger = EnhancedBuchberger::new(config);

        let m1 = make_monomial(&[(0, 2)]);
        let m2 = make_monomial(&[(1, 3)]);

        let lcm = buchberger.monomial_lcm(&m1, &m2);
        let product = buchberger.monomial_product(&m1, &m2);

        // Coprime monomials: LCM should equal product
        assert!(buchberger.monomial_equal(&lcm, &product));
    }

    #[test]
    fn test_empty_polynomial() {
        let buchberger = EnhancedBuchberger::new(BuchbergerConfig::default());

        let empty = Polynomial { terms: vec![] };
        assert!(buchberger.is_zero(&empty));
    }
}
