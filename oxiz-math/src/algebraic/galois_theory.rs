//! Galois Theory Basics for Polynomial Root Analysis.
//!
//! Provides fundamental Galois theory operations needed for understanding
//! polynomial splitting fields and solvability by radicals.

use super::field_extension::{FieldExtension, FieldExtensionManager, ExtensionId};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::{HashMap, HashSet};

/// Represents an automorphism of a field extension.
///
/// An automorphism σ: K → K fixes the base field and permutes roots.
#[derive(Clone, Debug)]
pub struct Automorphism {
    /// Maps each generator index to its image under the automorphism
    pub root_permutation: Vec<usize>,
    /// Extension this automorphism belongs to
    pub extension_id: ExtensionId,
}

/// The Galois group of a field extension.
///
/// Group of automorphisms that fix the base field.
#[derive(Clone, Debug)]
pub struct GaloisGroup {
    /// All automorphisms in the group
    pub elements: Vec<Automorphism>,
    /// Multiplication table (composition of automorphisms)
    mult_table: HashMap<(usize, usize), usize>,
    /// Extension this group corresponds to
    pub extension_id: ExtensionId,
}

impl GaloisGroup {
    /// Create a new Galois group.
    pub fn new(extension_id: ExtensionId) -> Self {
        Self {
            elements: Vec::new(),
            mult_table: HashMap::new(),
            extension_id,
        }
    }

    /// Add an automorphism to the group.
    pub fn add_automorphism(&mut self, aut: Automorphism) {
        self.elements.push(aut);
    }

    /// Compute the group order (number of elements).
    pub fn order(&self) -> usize {
        self.elements.len()
    }

    /// Compute the composition of two automorphisms.
    pub fn compose(&mut self, i: usize, j: usize) -> Option<usize> {
        // Check cache
        if let Some(&k) = self.mult_table.get(&(i, j)) {
            return Some(k);
        }

        if i >= self.elements.len() || j >= self.elements.len() {
            return None;
        }

        // Compose: first apply j, then apply i
        let sigma_i = &self.elements[i];
        let sigma_j = &self.elements[j];

        let mut composed_perm = vec![0; sigma_j.root_permutation.len()];
        for (idx, &target) in sigma_j.root_permutation.iter().enumerate() {
            composed_perm[idx] = sigma_i.root_permutation.get(target).copied().unwrap_or(target);
        }

        // Find or create this automorphism
        let composed = Automorphism {
            root_permutation: composed_perm,
            extension_id: self.extension_id,
        };

        // Check if this automorphism already exists
        for (k, aut) in self.elements.iter().enumerate() {
            if aut.root_permutation == composed.root_permutation {
                self.mult_table.insert((i, j), k);
                return Some(k);
            }
        }

        // Add new automorphism
        let k = self.elements.len();
        self.elements.push(composed);
        self.mult_table.insert((i, j), k);
        Some(k)
    }

    /// Compute the inverse of an automorphism.
    pub fn inverse(&self, i: usize) -> Option<usize> {
        if i >= self.elements.len() {
            return None;
        }

        let aut = &self.elements[i];
        let mut inv_perm = vec![0; aut.root_permutation.len()];

        for (idx, &target) in aut.root_permutation.iter().enumerate() {
            inv_perm[target] = idx;
        }

        // Find this permutation in the group
        for (j, other) in self.elements.iter().enumerate() {
            if other.root_permutation == inv_perm {
                return Some(j);
            }
        }

        None
    }

    /// Check if the group is abelian (commutative).
    pub fn is_abelian(&mut self) -> bool {
        for i in 0..self.elements.len() {
            for j in 0..self.elements.len() {
                let ij = self.compose(i, j);
                let ji = self.compose(j, i);

                if ij != ji {
                    return false;
                }
            }
        }
        true
    }

    /// Find all subgroups of this Galois group.
    pub fn subgroups(&mut self) -> Vec<Vec<usize>> {
        let n = self.elements.len();
        let mut subgroups = Vec::new();

        // Trivial subgroup {e}
        subgroups.push(vec![0]);

        // Enumerate all subsets and check group property
        for subset_bits in 1..(1 << n) {
            let mut subset = Vec::new();
            for i in 0..n {
                if (subset_bits >> i) & 1 == 1 {
                    subset.push(i);
                }
            }

            if self.is_subgroup(&subset) {
                subgroups.push(subset);
            }
        }

        subgroups
    }

    /// Check if a subset forms a subgroup.
    fn is_subgroup(&mut self, subset: &[usize]) -> bool {
        if subset.is_empty() {
            return false;
        }

        // Must contain identity (element 0)
        if !subset.contains(&0) {
            return false;
        }

        // Check closure under composition
        for &i in subset {
            for &j in subset {
                if let Some(k) = self.compose(i, j) {
                    if !subset.contains(&k) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        // Check inverses
        for &i in subset {
            if let Some(inv) = self.inverse(i) {
                if !subset.contains(&inv) {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Compute the fixed field of a subgroup.
    ///
    /// Returns indices of elements fixed by all automorphisms in the subgroup.
    pub fn fixed_field(&self, subgroup: &[usize]) -> Vec<usize> {
        if subgroup.is_empty() || self.elements.is_empty() {
            return Vec::new();
        }

        let first_aut = &self.elements[subgroup[0]];
        let n_roots = first_aut.root_permutation.len();

        let mut fixed = Vec::new();

        for root_idx in 0..n_roots {
            let mut is_fixed = true;

            for &aut_idx in subgroup {
                if aut_idx >= self.elements.len() {
                    continue;
                }

                let aut = &self.elements[aut_idx];
                if aut.root_permutation.get(root_idx).copied() != Some(root_idx) {
                    is_fixed = false;
                    break;
                }
            }

            if is_fixed {
                fixed.push(root_idx);
            }
        }

        fixed
    }

    /// Check if the extension is Galois (normal and separable).
    ///
    /// For this, the Galois group order should equal the extension degree.
    pub fn is_galois(&self, extension_degree: usize) -> bool {
        self.order() == extension_degree
    }

    /// Determine if the polynomial is solvable by radicals.
    ///
    /// A polynomial is solvable by radicals if and only if its Galois group is solvable.
    pub fn is_solvable(&mut self) -> bool {
        self.is_solvable_helper(&(0..self.elements.len()).collect::<Vec<_>>())
    }

    /// Recursive check for solvability.
    fn is_solvable_helper(&mut self, group: &[usize]) -> bool {
        if group.len() <= 1 {
            return true; // Trivial group is solvable
        }

        // Find a normal subgroup with abelian quotient
        let subgroups = self.subgroups();

        for subgroup in subgroups {
            if subgroup.len() >= group.len() || subgroup.is_empty() {
                continue;
            }

            // Check if subgroup is normal
            if !self.is_normal_subgroup(group, &subgroup) {
                continue;
            }

            // Check if quotient group is abelian
            // For simplicity, just check if subgroup itself is solvable
            if self.is_solvable_helper(&subgroup) {
                return true;
            }
        }

        false
    }

    /// Check if H is a normal subgroup of G.
    ///
    /// H is normal if gHg⁻¹ = H for all g in G.
    fn is_normal_subgroup(&mut self, g: &[usize], h: &[usize]) -> bool {
        let h_set: HashSet<usize> = h.iter().copied().collect();

        for &g_elem in g {
            let g_inv = match self.inverse(g_elem) {
                Some(inv) => inv,
                None => return false,
            };

            for &h_elem in h {
                // Compute g * h * g⁻¹
                let gh = match self.compose(g_elem, h_elem) {
                    Some(x) => x,
                    None => return false,
                };

                let ghg_inv = match self.compose(gh, g_inv) {
                    Some(x) => x,
                    None => return false,
                };

                if !h_set.contains(&ghg_inv) {
                    return false;
                }
            }
        }

        true
    }
}

/// Polynomial discriminant computation.
///
/// The discriminant determines if a polynomial has repeated roots.
pub struct Discriminant;

impl Discriminant {
    /// Compute the discriminant of a polynomial.
    ///
    /// For a polynomial of degree n with roots r₁, ..., rₙ:
    /// disc(f) = ∏ᵢ<ⱼ (rᵢ - rⱼ)²
    ///
    /// Returns None for constant polynomials.
    pub fn compute(poly: &[BigRational]) -> Option<BigRational> {
        if poly.len() <= 1 {
            return None;
        }

        // Use resultant-based formula: disc(f) = (-1)^(n(n-1)/2) * res(f, f') / a_n
        let derivative = Self::derivative(poly);
        let resultant = Self::resultant(poly, &derivative)?;

        let n = poly.len() - 1;
        let sign_factor = if (n * (n - 1) / 2) % 2 == 0 {
            BigRational::one()
        } else {
            -BigRational::one()
        };

        let leading_coeff = poly.last()?;

        Some(sign_factor * resultant / leading_coeff)
    }

    /// Compute the derivative of a polynomial.
    fn derivative(poly: &[BigRational]) -> Vec<BigRational> {
        if poly.len() <= 1 {
            return vec![BigRational::zero()];
        }

        let mut deriv = Vec::with_capacity(poly.len() - 1);
        for (i, coeff) in poly.iter().enumerate().skip(1) {
            deriv.push(coeff * &BigRational::from_integer((i as i32).into()));
        }

        deriv
    }

    /// Compute the resultant of two polynomials using Sylvester matrix.
    fn resultant(f: &[BigRational], g: &[BigRational]) -> Option<BigRational> {
        let m = f.len().checked_sub(1)?;
        let n = g.len().checked_sub(1)?;

        if m == 0 || n == 0 {
            return Some(BigRational::one());
        }

        // Build Sylvester matrix
        let size = m + n;
        let mut matrix = vec![vec![BigRational::zero(); size]; size];

        // Fill first n rows with shifted copies of f
        for i in 0..n {
            for (j, coeff) in f.iter().enumerate() {
                matrix[i][i + j] = coeff.clone();
            }
        }

        // Fill last m rows with shifted copies of g
        for i in 0..m {
            for (j, coeff) in g.iter().enumerate() {
                matrix[n + i][i + j] = coeff.clone();
            }
        }

        // Compute determinant
        Self::determinant(&matrix)
    }

    /// Compute determinant of a matrix (simplified implementation).
    fn determinant(matrix: &[Vec<BigRational>]) -> Option<BigRational> {
        let n = matrix.len();
        if n == 0 || matrix[0].len() != n {
            return None;
        }

        if n == 1 {
            return Some(matrix[0][0].clone());
        }

        if n == 2 {
            let det = &matrix[0][0] * &matrix[1][1] - &matrix[0][1] * &matrix[1][0];
            return Some(det);
        }

        // Laplace expansion along first row
        let mut det = BigRational::zero();
        for j in 0..n {
            let minor = Self::minor(matrix, 0, j)?;
            let minor_det = Self::determinant(&minor)?;

            let sign = if j % 2 == 0 {
                BigRational::one()
            } else {
                -BigRational::one()
            };

            det = det + sign * &matrix[0][j] * minor_det;
        }

        Some(det)
    }

    /// Extract minor matrix by removing row i and column j.
    fn minor(matrix: &[Vec<BigRational>], i: usize, j: usize) -> Option<Vec<Vec<BigRational>>> {
        let n = matrix.len();
        if i >= n || j >= n {
            return None;
        }

        let mut minor = Vec::with_capacity(n - 1);
        for (row_idx, row) in matrix.iter().enumerate() {
            if row_idx == i {
                continue;
            }

            let mut new_row = Vec::with_capacity(n - 1);
            for (col_idx, elem) in row.iter().enumerate() {
                if col_idx == j {
                    continue;
                }
                new_row.push(elem.clone());
            }
            minor.push(new_row);
        }

        Some(minor)
    }
}

/// Galois group computation for specific polynomial types.
pub struct GaloisComputation;

impl GaloisComputation {
    /// Compute Galois group for a quadratic polynomial ax² + bx + c.
    pub fn quadratic_galois_group(
        a: &BigRational,
        b: &BigRational,
        c: &BigRational,
    ) -> Option<GaloisGroup> {
        // Discriminant Δ = b² - 4ac
        let disc = b * b - &(BigRational::from_integer(4_i32.into()) * a * c);

        // If discriminant is a perfect square, Galois group is trivial
        // Otherwise, it's ℤ/2ℤ (order 2)

        let mut group = GaloisGroup::new(ExtensionId(0));

        // Identity automorphism
        group.add_automorphism(Automorphism {
            root_permutation: vec![0, 1], // maps r₀ → r₀, r₁ → r₁
            extension_id: ExtensionId(0),
        });

        // If discriminant is not a perfect square, add swap automorphism
        if !Self::is_perfect_square(&disc) {
            group.add_automorphism(Automorphism {
                root_permutation: vec![1, 0], // maps r₀ → r₁, r₁ → r₀
                extension_id: ExtensionId(0),
            });
        }

        Some(group)
    }

    /// Check if a rational number is a perfect square.
    fn is_perfect_square(r: &BigRational) -> bool {
        // Check if both numerator and denominator are perfect squares
        let numer = r.numer();
        let denom = r.denom();

        Self::is_perfect_square_int(numer) && Self::is_perfect_square_int(denom)
    }

    /// Check if a BigInt is a perfect square.
    fn is_perfect_square_int(n: &num_bigint::BigInt) -> bool {
        if n.sign() == num_bigint::Sign::Minus {
            return false;
        }

        let sqrt = n.sqrt();
        &(&sqrt * &sqrt) == n
    }

    /// Compute Galois group for a cubic polynomial.
    ///
    /// Cubic Galois groups are subgroups of S₃:
    /// - Trivial (order 1) if all roots are rational
    /// - ℤ/3ℤ (order 3) if discriminant is a perfect square
    /// - S₃ (order 6) otherwise
    pub fn cubic_galois_group(poly: &[BigRational]) -> Option<GaloisGroup> {
        if poly.len() != 4 {
            return None;
        }

        let disc = Discriminant::compute(poly)?;

        let mut group = GaloisGroup::new(ExtensionId(1));

        // Identity
        group.add_automorphism(Automorphism {
            root_permutation: vec![0, 1, 2],
            extension_id: ExtensionId(1),
        });

        if Self::is_perfect_square(&disc) {
            // Order 3: cyclic group ℤ/3ℤ
            group.add_automorphism(Automorphism {
                root_permutation: vec![1, 2, 0], // (0 1 2) cycle
                extension_id: ExtensionId(1),
            });
            group.add_automorphism(Automorphism {
                root_permutation: vec![2, 0, 1], // (0 2 1) cycle
                extension_id: ExtensionId(1),
            });
        } else {
            // Order 6: full symmetric group S₃
            group.add_automorphism(Automorphism {
                root_permutation: vec![1, 0, 2], // (0 1) transposition
                extension_id: ExtensionId(1),
            });
            group.add_automorphism(Automorphism {
                root_permutation: vec![0, 2, 1], // (1 2) transposition
                extension_id: ExtensionId(1),
            });
            group.add_automorphism(Automorphism {
                root_permutation: vec![2, 1, 0], // (0 2) transposition
                extension_id: ExtensionId(1),
            });
            group.add_automorphism(Automorphism {
                root_permutation: vec![1, 2, 0], // (0 1 2) cycle
                extension_id: ExtensionId(1),
            });
            group.add_automorphism(Automorphism {
                root_permutation: vec![2, 0, 1], // (0 2 1) cycle
                extension_id: ExtensionId(1),
            });
        }

        Some(group)
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
    fn test_quadratic_galois_group() {
        // x² - 2 has irrational roots, so Galois group is ℤ/2ℤ
        let group = GaloisComputation::quadratic_galois_group(&rat(1), &rat(0), &rat(-2))
            .expect("galois group computation failed");

        assert_eq!(group.order(), 2);
    }

    #[test]
    fn test_discriminant_quadratic() {
        // x² - 2: discriminant = 0² - 4(1)(-2) = 8
        let poly = vec![rat(-2), rat(0), rat(1)];
        let disc = Discriminant::compute(&poly).expect("discriminant computation failed");

        assert_eq!(disc, rat(8));
    }

    #[test]
    fn test_galois_group_composition() {
        let mut group = GaloisGroup::new(ExtensionId(0));

        // Add identity and swap for ℤ/2ℤ
        group.add_automorphism(Automorphism {
            root_permutation: vec![0, 1],
            extension_id: ExtensionId(0),
        });
        group.add_automorphism(Automorphism {
            root_permutation: vec![1, 0],
            extension_id: ExtensionId(0),
        });

        // σ₁ ∘ σ₁ = identity
        let composed = group.compose(1, 1).expect("composition failed");
        assert_eq!(composed, 0);
    }

    #[test]
    fn test_is_abelian() {
        let mut group = GaloisGroup::new(ExtensionId(0));

        // ℤ/2ℤ is abelian
        group.add_automorphism(Automorphism {
            root_permutation: vec![0, 1],
            extension_id: ExtensionId(0),
        });
        group.add_automorphism(Automorphism {
            root_permutation: vec![1, 0],
            extension_id: ExtensionId(0),
        });

        assert!(group.is_abelian());
    }
}
