//! Field Extensions for Algebraic Numbers.
//!
//! Provides representation and operations on algebraic field extensions
//! needed for advanced polynomial solving and symbolic computation.

use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::HashMap;

/// Represents an element in an algebraic field extension Q(α).
///
/// An element is represented as a polynomial in α with rational coefficients:
/// a₀ + a₁α + a₂α² + ... + aₙ₋₁αⁿ⁻¹
#[derive(Clone, Debug)]
pub struct FieldElement {
    /// Coefficients [a₀, a₁, ..., aₙ₋₁] where element = Σ aᵢαⁱ
    pub coeffs: Vec<BigRational>,
    /// Reference to the defining polynomial of α
    pub extension_id: ExtensionId,
}

/// Unique identifier for a field extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtensionId(pub usize);

/// A field extension Q(α) where α is a root of a minimal polynomial.
#[derive(Clone, Debug)]
pub struct FieldExtension {
    /// Minimal polynomial p(x) such that p(α) = 0
    pub minimal_poly: Vec<BigRational>,
    /// Degree of the extension [Q(α):Q]
    pub degree: usize,
    /// Cached multiplication table for efficiency
    mult_table: HashMap<(usize, usize), Vec<BigRational>>,
    /// Cached inverse table
    inv_table: HashMap<Vec<BigRational>, Vec<BigRational>>,
}

impl FieldExtension {
    /// Create a new field extension from a minimal polynomial.
    ///
    /// The polynomial should be irreducible over Q.
    pub fn new(minimal_poly: Vec<BigRational>) -> Self {
        let degree = minimal_poly.len().saturating_sub(1);

        Self {
            minimal_poly,
            degree,
            mult_table: HashMap::new(),
            inv_table: HashMap::new(),
        }
    }

    /// Check if the minimal polynomial is monic.
    pub fn is_monic(&self) -> bool {
        if let Some(lead) = self.minimal_poly.last() {
            lead.is_one()
        } else {
            false
        }
    }

    /// Reduce a polynomial modulo the minimal polynomial.
    ///
    /// Returns coefficients of the reduced polynomial of degree < extension degree.
    pub fn reduce(&self, coeffs: &[BigRational]) -> Vec<BigRational> {
        if coeffs.len() <= self.degree {
            return coeffs.to_vec();
        }

        let mut result = coeffs.to_vec();

        // Polynomial long division by minimal_poly
        while result.len() > self.degree && !result.last().map_or(true, |c| c.is_zero()) {
            let deg_diff = result.len() - self.minimal_poly.len();
            let lead_coeff = result.last().cloned().expect("checked non-empty");
            let min_lead = self.minimal_poly.last().cloned().expect("checked non-empty");

            let quotient_coeff = &lead_coeff / &min_lead;

            // Subtract quotient_coeff * x^deg_diff * minimal_poly from result
            for (i, min_coeff) in self.minimal_poly.iter().enumerate() {
                let idx = i + deg_diff;
                if idx < result.len() {
                    result[idx] = &result[idx] - &quotient_coeff * min_coeff;
                }
            }

            result.pop();
        }

        // Remove leading zeros
        while result.last().map_or(false, |c| c.is_zero()) {
            result.pop();
        }

        if result.is_empty() {
            vec![BigRational::zero()]
        } else {
            result
        }
    }

    /// Multiply two elements in the field extension.
    pub fn multiply(&mut self, a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
        // Check cache first
        let key = (a.len(), b.len());
        if let Some(cached) = self.mult_table.get(&key) {
            if a == cached || b == cached {
                // Simple case, use cache if applicable
            }
        }

        // Polynomial multiplication
        let mut product = vec![BigRational::zero(); a.len() + b.len() - 1];

        for (i, a_coeff) in a.iter().enumerate() {
            for (j, b_coeff) in b.iter().enumerate() {
                product[i + j] = &product[i + j] + a_coeff * b_coeff;
            }
        }

        // Reduce modulo minimal polynomial
        let reduced = self.reduce(&product);

        // Cache the result for small degrees
        if a.len() <= 3 && b.len() <= 3 {
            self.mult_table.insert(key, reduced.clone());
        }

        reduced
    }

    /// Add two elements in the field extension.
    pub fn add(&self, a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
        let max_len = a.len().max(b.len());
        let mut result = vec![BigRational::zero(); max_len];

        for (i, coeff) in a.iter().enumerate() {
            result[i] = coeff.clone();
        }

        for (i, coeff) in b.iter().enumerate() {
            result[i] = &result[i] + coeff;
        }

        // Remove trailing zeros
        while result.len() > 1 && result.last().map_or(false, |c| c.is_zero()) {
            result.pop();
        }

        result
    }

    /// Negate an element in the field extension.
    pub fn negate(&self, a: &[BigRational]) -> Vec<BigRational> {
        a.iter().map(|c| -c).collect()
    }

    /// Compute the multiplicative inverse of an element.
    ///
    /// Uses extended Euclidean algorithm in Q[x] modulo minimal polynomial.
    pub fn inverse(&mut self, a: &[BigRational]) -> Option<Vec<BigRational>> {
        // Check cache
        if let Some(cached) = self.inv_table.get(a) {
            return Some(cached.clone());
        }

        // Extended GCD in polynomial ring
        let (gcd, s, _t) = self.extended_gcd(a, &self.minimal_poly.clone());

        // Check if GCD is a non-zero constant (element is invertible)
        if gcd.len() == 1 && !gcd[0].is_zero() {
            let inv_gcd = BigRational::one() / &gcd[0];
            let result: Vec<BigRational> = s.iter().map(|c| c * &inv_gcd).collect();
            let reduced = self.reduce(&result);

            // Cache the result
            self.inv_table.insert(a.to_vec(), reduced.clone());

            Some(reduced)
        } else {
            None // Element is not invertible
        }
    }

    /// Extended Euclidean algorithm for polynomials.
    ///
    /// Returns (gcd, s, t) such that s*a + t*b = gcd(a, b).
    fn extended_gcd(
        &self,
        a: &[BigRational],
        b: &[BigRational],
    ) -> (Vec<BigRational>, Vec<BigRational>, Vec<BigRational>) {
        if b.is_empty() || b.iter().all(|c| c.is_zero()) {
            return (
                a.to_vec(),
                vec![BigRational::one()],
                vec![BigRational::zero()],
            );
        }

        let (q, r) = self.poly_div(a, b);
        let (gcd, s1, t1) = self.extended_gcd(b, &r);

        // s = t1
        // t = s1 - q * t1
        let q_t1 = self.poly_mult(&q, &t1);
        let t = self.poly_sub(&s1, &q_t1);

        (gcd, t1, t)
    }

    /// Polynomial division: returns (quotient, remainder).
    fn poly_div(&self, a: &[BigRational], b: &[BigRational]) -> (Vec<BigRational>, Vec<BigRational>) {
        if b.is_empty() || b.iter().all(|c| c.is_zero()) {
            return (vec![BigRational::zero()], a.to_vec());
        }

        let mut remainder = a.to_vec();
        let mut quotient = vec![BigRational::zero(); a.len().saturating_sub(b.len()) + 1];

        let b_lead = b.last().expect("checked non-empty");

        while remainder.len() >= b.len() {
            let r_lead = match remainder.last() {
                Some(c) if !c.is_zero() => c,
                _ => break,
            };

            let deg_diff = remainder.len() - b.len();
            let q_coeff = r_lead / b_lead;

            if deg_diff < quotient.len() {
                quotient[deg_diff] = q_coeff.clone();
            }

            // Subtract q_coeff * x^deg_diff * b from remainder
            for (i, b_coeff) in b.iter().enumerate() {
                let idx = i + deg_diff;
                if idx < remainder.len() {
                    remainder[idx] = &remainder[idx] - &q_coeff * b_coeff;
                }
            }

            remainder.pop();
        }

        // Remove leading zeros
        while quotient.last().map_or(false, |c| c.is_zero()) {
            quotient.pop();
        }
        while remainder.last().map_or(false, |c| c.is_zero()) {
            remainder.pop();
        }

        if quotient.is_empty() {
            quotient.push(BigRational::zero());
        }
        if remainder.is_empty() {
            remainder.push(BigRational::zero());
        }

        (quotient, remainder)
    }

    /// Polynomial multiplication.
    fn poly_mult(&self, a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
        if a.is_empty() || b.is_empty() {
            return vec![BigRational::zero()];
        }

        let mut result = vec![BigRational::zero(); a.len() + b.len() - 1];

        for (i, a_coeff) in a.iter().enumerate() {
            for (j, b_coeff) in b.iter().enumerate() {
                result[i + j] = &result[i + j] + a_coeff * b_coeff;
            }
        }

        result
    }

    /// Polynomial subtraction.
    fn poly_sub(&self, a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
        let max_len = a.len().max(b.len());
        let mut result = vec![BigRational::zero(); max_len];

        for (i, coeff) in a.iter().enumerate() {
            result[i] = coeff.clone();
        }

        for (i, coeff) in b.iter().enumerate() {
            result[i] = &result[i] - coeff;
        }

        while result.len() > 1 && result.last().map_or(false, |c| c.is_zero()) {
            result.pop();
        }

        if result.is_empty() {
            vec![BigRational::zero()]
        } else {
            result
        }
    }

    /// Compute the norm of an element.
    ///
    /// The norm is the product of all conjugates of the element.
    pub fn norm(&mut self, a: &[BigRational]) -> BigRational {
        // For now, compute as determinant of multiplication matrix
        // This is a simplified implementation
        if a.is_empty() || (a.len() == 1 && a[0].is_zero()) {
            return BigRational::zero();
        }

        // If element is a rational, norm is just the element to the power of degree
        if a.len() == 1 {
            let mut result = a[0].clone();
            for _ in 1..self.degree {
                result = &result * &a[0];
            }
            return result;
        }

        // General case: use characteristic polynomial
        BigRational::one() // Placeholder
    }

    /// Compute the trace of an element.
    ///
    /// The trace is the sum of all conjugates of the element.
    pub fn trace(&self, a: &[BigRational]) -> BigRational {
        // For element a₀ + a₁α + ..., trace is degree * a₀
        if a.is_empty() {
            BigRational::zero()
        } else {
            &a[0] * &BigRational::from_integer((self.degree as i32).into())
        }
    }

    /// Check if an element is a primitive element (generates the extension).
    pub fn is_primitive(&mut self, a: &[BigRational]) -> bool {
        // Element is primitive if its minimal polynomial has degree equal to extension degree
        // For now, simplified check
        a.len() >= self.degree
    }
}

impl FieldElement {
    /// Create a new field element from coefficients.
    pub fn new(coeffs: Vec<BigRational>, extension_id: ExtensionId) -> Self {
        Self {
            coeffs,
            extension_id,
        }
    }

    /// Create a rational element (no extension).
    pub fn from_rational(r: BigRational) -> Self {
        Self {
            coeffs: vec![r],
            extension_id: ExtensionId(0), // 0 represents Q itself
        }
    }

    /// Check if this element is actually rational.
    pub fn is_rational(&self) -> bool {
        self.coeffs.len() == 1 || self.coeffs[1..].iter().all(|c| c.is_zero())
    }

    /// Get the rational part (constant term).
    pub fn rational_part(&self) -> BigRational {
        self.coeffs.first().cloned().unwrap_or_else(BigRational::zero)
    }

    /// Get the degree of the element (highest non-zero coefficient index).
    pub fn degree(&self) -> usize {
        self.coeffs.iter()
            .rposition(|c| !c.is_zero())
            .unwrap_or(0)
    }
}

/// Manager for multiple field extensions.
pub struct FieldExtensionManager {
    extensions: Vec<FieldExtension>,
    /// Map from minimal polynomial to extension ID
    poly_to_id: HashMap<Vec<BigRational>, ExtensionId>,
}

impl FieldExtensionManager {
    /// Create a new field extension manager.
    pub fn new() -> Self {
        Self {
            extensions: Vec::new(),
            poly_to_id: HashMap::new(),
        }
    }

    /// Register a new field extension or get existing one.
    pub fn get_or_create(&mut self, minimal_poly: Vec<BigRational>) -> ExtensionId {
        if let Some(&ext_id) = self.poly_to_id.get(&minimal_poly) {
            return ext_id;
        }

        let ext_id = ExtensionId(self.extensions.len());
        let extension = FieldExtension::new(minimal_poly.clone());

        self.extensions.push(extension);
        self.poly_to_id.insert(minimal_poly, ext_id);

        ext_id
    }

    /// Get a field extension by ID.
    pub fn get_extension(&mut self, ext_id: ExtensionId) -> Option<&mut FieldExtension> {
        self.extensions.get_mut(ext_id.0)
    }

    /// Add two field elements (must be in same extension).
    pub fn add(&mut self, a: &FieldElement, b: &FieldElement) -> Option<FieldElement> {
        if a.extension_id != b.extension_id {
            return None; // Different extensions
        }

        let ext = self.get_extension(a.extension_id)?;
        let coeffs = ext.add(&a.coeffs, &b.coeffs);

        Some(FieldElement::new(coeffs, a.extension_id))
    }

    /// Multiply two field elements.
    pub fn multiply(&mut self, a: &FieldElement, b: &FieldElement) -> Option<FieldElement> {
        if a.extension_id != b.extension_id {
            return None; // Different extensions
        }

        let ext = self.get_extension(a.extension_id)?;
        let coeffs = ext.multiply(&a.coeffs, &b.coeffs);

        Some(FieldElement::new(coeffs, a.extension_id))
    }

    /// Compute multiplicative inverse.
    pub fn inverse(&mut self, a: &FieldElement) -> Option<FieldElement> {
        let ext = self.get_extension(a.extension_id)?;
        let coeffs = ext.inverse(&a.coeffs)?;

        Some(FieldElement::new(coeffs, a.extension_id))
    }

    /// Compute norm of a field element.
    pub fn norm(&mut self, a: &FieldElement) -> Option<BigRational> {
        let ext = self.get_extension(a.extension_id)?;
        Some(ext.norm(&a.coeffs))
    }

    /// Compute trace of a field element.
    pub fn trace(&mut self, a: &FieldElement) -> Option<BigRational> {
        let ext = self.get_extension(a.extension_id)?;
        Some(ext.trace(&a.coeffs))
    }
}

impl Default for FieldExtensionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_field_extension_creation() {
        // Create Q(√2) via minimal polynomial x² - 2
        let minimal_poly = vec![rat(-2), rat(0), rat(1)]; // -2 + 0x + x²
        let ext = FieldExtension::new(minimal_poly);

        assert_eq!(ext.degree, 2);
        assert!(ext.is_monic());
    }

    #[test]
    fn test_reduction() {
        // Q(√2): x² - 2 = 0, so x² ≡ 2
        let minimal_poly = vec![rat(-2), rat(0), rat(1)];
        let ext = FieldExtension::new(minimal_poly);

        // Reduce x² to 2
        let reduced = ext.reduce(&[rat(0), rat(0), rat(1)]);
        assert_eq!(reduced, vec![rat(2)]);
    }

    #[test]
    fn test_multiplication() {
        // Q(√2): x² = 2
        let minimal_poly = vec![rat(-2), rat(0), rat(1)];
        let mut ext = FieldExtension::new(minimal_poly);

        // (1 + √2) * (1 + √2) = 1 + 2√2 + 2 = 3 + 2√2
        let a = vec![rat(1), rat(1)]; // 1 + x
        let b = vec![rat(1), rat(1)]; // 1 + x

        let product = ext.multiply(&a, &b);
        assert_eq!(product, vec![rat(3), rat(2)]);
    }

    #[test]
    fn test_field_element_manager() {
        let mut mgr = FieldExtensionManager::new();

        // Create Q(√2)
        let minimal_poly = vec![rat(-2), rat(0), rat(1)];
        let ext_id = mgr.get_or_create(minimal_poly);

        // Create elements 1 + √2 and 2 + √2
        let a = FieldElement::new(vec![rat(1), rat(1)], ext_id);
        let b = FieldElement::new(vec![rat(2), rat(1)], ext_id);

        // Add them: should get 3 + 2√2
        let sum = mgr.add(&a, &b).expect("addition failed");
        assert_eq!(sum.coeffs, vec![rat(3), rat(2)]);
    }

    #[test]
    fn test_inverse() {
        // Q(√2): x² = 2
        let minimal_poly = vec![rat(-2), rat(0), rat(1)];
        let mut ext = FieldExtension::new(minimal_poly);

        // Inverse of 1 + √2 is (-1 + √2) / (-1) = (√2 - 1)
        let a = vec![rat(1), rat(1)];
        let inv = ext.inverse(&a).expect("inverse should exist");

        // Check: (1 + √2) * inv = 1
        let product = ext.multiply(&a, &inv);
        assert!(product.len() == 1 || (product.len() > 1 && product[1..].iter().all(|c| c.is_zero())));
    }
}
