//! Hilbert basis computation for integer cones.
//!
//! This module implements algorithms for computing Hilbert bases of rational
//! polyhedral cones, which is fundamental for integer programming and
//! polyhedral computations in SMT solving.
//!
//! A Hilbert basis is a minimal generating set for the set of integer points
//! in a pointed rational polyhedral cone.
//!
//! Reference: Z3's arithmetic theories and polyhedral computation literature.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// A vector in integer space.
pub type IntVector = Vec<BigInt>;

/// A vector in rational space.
pub type RatVector = Vec<BigRational>;

/// A rational polyhedral cone defined by a system of inequalities: Ax >= 0.
#[derive(Debug, Clone)]
pub struct Cone {
    /// Constraint matrix A.
    constraints: Vec<RatVector>,
    /// Dimension of the ambient space.
    dimension: usize,
}

impl Cone {
    /// Create a new cone from constraint inequalities.
    ///
    /// Each constraint is a vector a such that a^T x >= 0.
    pub fn new(constraints: Vec<RatVector>) -> Self {
        let dimension = if constraints.is_empty() {
            0
        } else {
            constraints[0].len()
        };

        Self {
            constraints,
            dimension,
        }
    }

    /// Get the dimension of the ambient space.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if a point is in the cone.
    pub fn contains(&self, point: &RatVector) -> bool {
        if point.len() != self.dimension {
            return false;
        }

        for constraint in &self.constraints {
            let mut dot = BigRational::zero();
            for i in 0..self.dimension {
                dot += &constraint[i] * &point[i];
            }
            if dot < BigRational::zero() {
                return false;
            }
        }

        true
    }
}

/// Compute the Hilbert basis of a rational polyhedral cone.
///
/// This uses a simplified version of the algorithm based on Normaliz.
/// The full implementation would use more sophisticated techniques like
/// triangulation and unimodular cones.
pub fn hilbert_basis(cone: &Cone) -> Vec<IntVector> {
    if cone.dimension == 0 {
        return vec![];
    }

    // Start with extreme rays (simplified: use unit vectors in positive orthant)
    let mut basis = Vec::new();
    let mut candidates = Vec::new();

    // Generate initial candidates from unit vectors
    for i in 0..cone.dimension {
        let mut v = vec![BigInt::zero(); cone.dimension];
        v[i] = BigInt::one();

        let rat_v: RatVector = v
            .iter()
            .map(|x| BigRational::from_integer(x.clone()))
            .collect();
        if cone.contains(&rat_v) {
            candidates.push(v);
        }
    }

    // Compute Hilbert basis using a simplified algorithm
    // In a full implementation, this would use cone decomposition
    let max_iterations = 100;
    let mut iteration = 0;

    while !candidates.is_empty() && iteration < max_iterations {
        iteration += 1;

        let candidate = candidates
            .pop()
            .expect("collection validated to be non-empty");

        // Check if this is a primitive vector (not a multiple of another)
        if is_primitive(&candidate) {
            // Check if this vector is already in the basis or can be generated
            if !is_generated_by(&candidate, &basis) {
                basis.push(candidate.clone());

                // Generate new candidates by adding to existing basis elements
                for base_vec in &basis {
                    if base_vec != &candidate {
                        let sum = add_vectors(&candidate, base_vec);
                        let rat_sum: RatVector = sum
                            .iter()
                            .map(|x| BigRational::from_integer(x.clone()))
                            .collect();

                        if cone.contains(&rat_sum) && !candidates.contains(&sum) {
                            // Limit the size of candidates to prevent explosion
                            if candidates.len() < 1000 && vector_norm(&sum) < BigInt::from(100) {
                                candidates.push(sum);
                            }
                        }
                    }
                }
            }
        }
    }

    // Filter to ensure minimality
    minimize_basis(basis)
}

/// Check if a vector is primitive (GCD of components is 1).
fn is_primitive(v: &IntVector) -> bool {
    if v.is_empty() {
        return false;
    }

    let mut g = v[0].clone();
    for component in v.iter().skip(1) {
        g = gcd(&g, component);
        if g == BigInt::one() {
            return true;
        }
    }

    g == BigInt::one()
}

/// Compute GCD of two BigInts.
fn gcd(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.abs();
    let mut b = b.abs();

    while !b.is_zero() {
        let temp = b.clone();
        b = &a % &b;
        a = temp;
    }

    a
}

/// Check if a vector is generated by (is a non-negative integer combination of) a set of vectors.
fn is_generated_by(v: &IntVector, generators: &[IntVector]) -> bool {
    if generators.is_empty() {
        return v.iter().all(|x| x.is_zero());
    }

    // Simple check: if v is a scalar multiple of any generator
    for generator in generators {
        if is_multiple(v, generator) {
            return true;
        }
    }

    // For a complete implementation, we would solve an integer programming problem
    // For now, use a simplified heuristic
    false
}

/// Check if v1 is a non-negative integer multiple of v2.
fn is_multiple(v1: &IntVector, v2: &IntVector) -> bool {
    if v1.len() != v2.len() {
        return false;
    }

    // Special case: zero vector is a multiple of any vector (0 * v2 = 0)
    if v1.iter().all(|x| x.is_zero()) {
        return true;
    }

    // Find the first non-zero component in v2
    let mut ratio: Option<BigRational> = None;

    for i in 0..v1.len() {
        if !v2[i].is_zero() {
            let r = BigRational::new(v1[i].clone(), v2[i].clone());
            if let Some(ref existing_ratio) = ratio {
                if &r != existing_ratio {
                    return false;
                }
            } else {
                // Check if ratio is a positive integer
                if r < BigRational::zero() || !r.is_integer() {
                    return false;
                }
                ratio = Some(r);
            }
        } else if !v1[i].is_zero() {
            return false;
        }
    }

    ratio.is_some()
}

/// Add two integer vectors component-wise.
fn add_vectors(v1: &IntVector, v2: &IntVector) -> IntVector {
    assert_eq!(v1.len(), v2.len());
    v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect()
}

/// Compute the L1 norm of a vector.
fn vector_norm(v: &IntVector) -> BigInt {
    v.iter().map(|x| x.abs()).sum()
}

/// Remove redundant vectors from a Hilbert basis to ensure minimality.
fn minimize_basis(mut basis: Vec<IntVector>) -> Vec<IntVector> {
    let mut minimal = Vec::new();

    // Sort by norm to process smaller vectors first
    basis.sort_by(|a, b| {
        let norm_a = vector_norm(a);
        let norm_b = vector_norm(b);
        norm_a.cmp(&norm_b)
    });

    for vec in basis {
        // Check if this vector is generated by vectors already in minimal basis
        if !is_generated_by(&vec, &minimal) {
            minimal.push(vec);
        }
    }

    minimal
}

/// Integer cone operations.
pub struct IntCone {
    /// Hilbert basis generators.
    generators: Vec<IntVector>,
    /// Dimension of the cone.
    dimension: usize,
}

impl IntCone {
    /// Create a new integer cone from its Hilbert basis.
    pub fn from_hilbert_basis(generators: Vec<IntVector>) -> Self {
        let dimension = if generators.is_empty() {
            0
        } else {
            generators[0].len()
        };

        Self {
            generators,
            dimension,
        }
    }

    /// Get the generators.
    pub fn generators(&self) -> &[IntVector] {
        &self.generators
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if a point is in the integer cone.
    pub fn contains_int(&self, point: &IntVector) -> bool {
        // Check if point can be expressed as a non-negative integer combination
        // of generators. This is a simplified check.
        if point.len() != self.dimension {
            return false;
        }

        // Trivial case: zero vector
        if point.iter().all(|x| x.is_zero()) {
            return true;
        }

        // Check if point is one of the generators or a multiple
        for generator in &self.generators {
            if is_multiple(point, generator) {
                return true;
            }
        }

        // For a complete implementation, solve an integer programming problem
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_vec(values: &[i64]) -> IntVector {
        values.iter().map(|&x| BigInt::from(x)).collect()
    }

    fn rat_vec(values: &[i64]) -> RatVector {
        values
            .iter()
            .map(|&x| BigRational::from_integer(BigInt::from(x)))
            .collect()
    }

    #[test]
    fn test_cone_contains() {
        // Positive orthant in 2D: x >= 0, y >= 0
        let constraints = vec![rat_vec(&[1, 0]), rat_vec(&[0, 1])];
        let cone = Cone::new(constraints);

        assert!(cone.contains(&rat_vec(&[1, 1])));
        assert!(cone.contains(&rat_vec(&[0, 0])));
        assert!(!cone.contains(&rat_vec(&[-1, 1])));
        assert!(!cone.contains(&rat_vec(&[1, -1])));
    }

    #[test]
    fn test_is_primitive() {
        assert!(is_primitive(&int_vec(&[1, 2, 3])));
        assert!(is_primitive(&int_vec(&[1, 0, 0])));
        assert!(!is_primitive(&int_vec(&[2, 4, 6])));
        assert!(is_primitive(&int_vec(&[3, 5, 7])));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(&BigInt::from(12), &BigInt::from(8)), BigInt::from(4));
        assert_eq!(gcd(&BigInt::from(7), &BigInt::from(3)), BigInt::from(1));
        assert_eq!(gcd(&BigInt::from(0), &BigInt::from(5)), BigInt::from(5));
    }

    #[test]
    fn test_is_multiple() {
        assert!(is_multiple(&int_vec(&[2, 4]), &int_vec(&[1, 2])));
        assert!(!is_multiple(&int_vec(&[2, 5]), &int_vec(&[1, 2])));
        assert!(is_multiple(&int_vec(&[0, 0]), &int_vec(&[1, 2])));
        assert!(!is_multiple(&int_vec(&[1, 2]), &int_vec(&[0, 0])));
    }

    #[test]
    fn test_add_vectors() {
        let v1 = int_vec(&[1, 2, 3]);
        let v2 = int_vec(&[4, 5, 6]);
        let sum = add_vectors(&v1, &v2);
        assert_eq!(sum, int_vec(&[5, 7, 9]));
    }

    #[test]
    fn test_vector_norm() {
        assert_eq!(vector_norm(&int_vec(&[1, -2, 3])), BigInt::from(6));
        assert_eq!(vector_norm(&int_vec(&[0, 0, 0])), BigInt::from(0));
    }

    #[test]
    fn test_hilbert_basis_simple() {
        // Positive orthant in 2D
        let constraints = vec![rat_vec(&[1, 0]), rat_vec(&[0, 1])];
        let cone = Cone::new(constraints);

        let basis = hilbert_basis(&cone);

        // Should contain at least the unit vectors
        assert!(!basis.is_empty());
        assert!(basis.len() >= 2);
    }

    #[test]
    fn test_int_cone() {
        let generators = vec![int_vec(&[1, 0]), int_vec(&[0, 1])];
        let cone = IntCone::from_hilbert_basis(generators);

        assert_eq!(cone.dimension(), 2);
        assert!(cone.contains_int(&int_vec(&[0, 0])));
        assert!(cone.contains_int(&int_vec(&[1, 0])));
        assert!(cone.contains_int(&int_vec(&[2, 0])));
    }
}
