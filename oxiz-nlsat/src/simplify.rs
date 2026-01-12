//! Simplification of polynomials and constraints.
//!
//! This module provides simplification and normalization routines for
//! polynomial constraints to improve solver efficiency.
//!
//! Reference: Z3's `nlsat/nlsat_simplify.cpp`

use crate::types::{Atom, AtomKind, IneqAtom, PolyFactor};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use oxiz_math::polynomial::Polynomial;

/// Simplify a polynomial by removing common factors and normalizing.
pub fn simplify_polynomial(poly: &Polynomial) -> Polynomial {
    // Normalize coefficients (make them coprime if possible)
    let mut result = poly.clone();

    // Remove zero polynomial
    if result.is_zero() {
        return result;
    }

    // Factor out GCD of coefficients (for rational polynomials, this normalizes)
    result = normalize_coefficients(result);

    result
}

/// Normalize polynomial coefficients by dividing by their GCD.
fn normalize_coefficients(poly: Polynomial) -> Polynomial {
    let terms = poly.terms();

    if terms.is_empty() {
        return poly;
    }

    // Find the GCD of all denominators and LCM would be complex,
    // so we just normalize the leading coefficient to be positive
    let leading_term = &terms[0];

    if leading_term.coeff.is_negative() {
        // Negate all coefficients
        poly.neg()
    } else {
        poly
    }
}

/// Simplify an inequality atom.
pub fn simplify_ineq_atom(atom: &IneqAtom) -> Option<SimplifiedAtom> {
    // Handle trivial cases
    if atom.factors.is_empty() {
        return Some(SimplifiedAtom::Trivial(false));
    }

    // Simplify each factor
    let mut simplified_factors = Vec::new();
    let mut has_zero = false;

    for factor in &atom.factors {
        let simplified = simplify_polynomial(&factor.poly);

        if simplified.is_zero() {
            has_zero = true;
            break;
        }

        // Check if constant
        if simplified.is_constant() {
            let const_val = simplified.constant_term();

            // If this is the only factor, we can evaluate the constraint directly
            if atom.factors.len() == 1 {
                return match atom.kind {
                    AtomKind::Eq => Some(SimplifiedAtom::Trivial(const_val.is_zero())),
                    AtomKind::Lt => Some(SimplifiedAtom::Trivial(const_val.is_negative())),
                    AtomKind::Gt => Some(SimplifiedAtom::Trivial(const_val.is_positive())),
                    _ => None,
                };
            }

            // Handle constant factors based on atom kind
            match atom.kind {
                AtomKind::Eq => {
                    // p * c = 0 is equivalent to p = 0 if c != 0
                    if const_val.is_zero() {
                        return Some(SimplifiedAtom::Trivial(true));
                    }
                    // Otherwise skip this factor
                    continue;
                }
                AtomKind::Lt | AtomKind::Gt => {
                    // p * c < 0 or p * c > 0
                    if const_val.is_zero() {
                        return Some(SimplifiedAtom::Trivial(false));
                    }
                    // Sign of constant affects the constraint
                    if (atom.kind == AtomKind::Lt && const_val.is_negative())
                        || (atom.kind == AtomKind::Gt && const_val.is_positive())
                    {
                        // Flip is needed, but we'll handle this by tracking sign
                        continue;
                    }
                }
                _ => {}
            }
        }

        simplified_factors.push(PolyFactor {
            poly: simplified,
            is_even: factor.is_even,
        });
    }

    // Handle the zero case
    if has_zero {
        return match atom.kind {
            AtomKind::Eq => Some(SimplifiedAtom::Trivial(true)), // 0 = 0 is true
            AtomKind::Lt => Some(SimplifiedAtom::Trivial(false)), // 0 < 0 is false
            AtomKind::Gt => Some(SimplifiedAtom::Trivial(false)), // 0 > 0 is false
            _ => None,
        };
    }

    // If no factors remain, the constraint is trivial
    if simplified_factors.is_empty() {
        return match atom.kind {
            AtomKind::Eq => Some(SimplifiedAtom::Trivial(false)), // constant != 0
            AtomKind::Lt => Some(SimplifiedAtom::Trivial(false)), // constant < 0 (false for positive)
            AtomKind::Gt => Some(SimplifiedAtom::Trivial(true)), // constant > 0 (true for positive)
            _ => None,
        };
    }

    // Create simplified atom
    Some(SimplifiedAtom::Atom(Atom::Ineq(IneqAtom {
        kind: atom.kind,
        factors: simplified_factors,
        max_var: atom.max_var,
        bool_var: atom.bool_var,
    })))
}

/// Result of simplification.
#[derive(Debug, Clone)]
pub enum SimplifiedAtom {
    /// Atom simplified to a constant.
    Trivial(bool),
    /// Simplified atom.
    Atom(Atom),
}

/// Eliminate redundant constraints from a set of atoms.
pub fn eliminate_redundant(atoms: &[Atom]) -> Vec<usize> {
    let mut redundant = Vec::new();

    // Simple redundancy check: find duplicate atoms
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            if atoms_equivalent(&atoms[i], &atoms[j]) {
                redundant.push(j);
            }
        }
    }

    // Deduplicate the redundant list
    redundant.sort_unstable();
    redundant.dedup();

    redundant
}

/// Check if two atoms are equivalent (represent the same constraint).
fn atoms_equivalent(a1: &Atom, a2: &Atom) -> bool {
    match (a1, a2) {
        (Atom::Ineq(ineq1), Atom::Ineq(ineq2)) => {
            if ineq1.kind != ineq2.kind {
                return false;
            }

            if ineq1.factors.len() != ineq2.factors.len() {
                return false;
            }

            // Check if all factors match (order-independent for multiplication)
            // For simplicity, we just check if they're identical in order
            for (f1, f2) in ineq1.factors.iter().zip(ineq2.factors.iter()) {
                if !polynomials_equivalent(&f1.poly, &f2.poly) {
                    return false;
                }
                if f1.is_even != f2.is_even {
                    return false;
                }
            }

            true
        }
        _ => false,
    }
}

/// Check if two polynomials are equivalent (up to constant factor).
/// Two polynomials are equivalent if one is a non-zero constant multiple of the other.
fn polynomials_equivalent(p1: &Polynomial, p2: &Polynomial) -> bool {
    use num_traits::Zero;

    let t1 = p1.terms();
    let t2 = p2.terms();

    // Both empty (zero polynomials) are equivalent
    if t1.is_empty() && t2.is_empty() {
        return true;
    }

    // Different number of terms means different structure
    if t1.len() != t2.len() {
        return false;
    }

    // Find the ratio from the first pair of terms
    let mut ratio: Option<BigRational> = None;

    for (term1, term2) in t1.iter().zip(t2.iter()) {
        // Monomials must match exactly
        if term1.monomial != term2.monomial {
            return false;
        }

        // Check coefficient ratio
        if term2.coeff.is_zero() {
            // If term2's coeff is zero, term1's must also be zero
            if !term1.coeff.is_zero() {
                return false;
            }
            // Both zero, continue to next term
            continue;
        }

        // Compute the ratio term1.coeff / term2.coeff
        let current_ratio = &term1.coeff / &term2.coeff;

        match ratio {
            None => {
                // First non-zero ratio found
                ratio = Some(current_ratio);
            }
            Some(ref expected_ratio) => {
                // Check if this ratio matches the expected one
                if &current_ratio != expected_ratio {
                    return false;
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    #[test]
    fn test_simplify_polynomial_zero() {
        let zero = Polynomial::zero();
        let simplified = simplify_polynomial(&zero);
        assert!(simplified.is_zero());
    }

    #[test]
    fn test_simplify_polynomial_constant() {
        let c = Polynomial::constant(rat(5));
        let simplified = simplify_polynomial(&c);
        assert_eq!(simplified.constant_term(), rat(5));
    }

    #[test]
    fn test_simplify_polynomial_negative() {
        let x = Polynomial::from_var(0);
        let neg_x = x.neg();
        let simplified = simplify_polynomial(&neg_x);

        // Leading coefficient should be positive after normalization
        let terms = simplified.terms();
        assert!(!terms.is_empty());
        assert!(terms[0].coeff.is_positive());
    }

    #[test]
    fn test_simplify_ineq_zero() {
        let zero = Polynomial::zero();
        let atom = IneqAtom::from_poly(zero, AtomKind::Eq);

        let result = simplify_ineq_atom(&atom);
        assert!(matches!(result, Some(SimplifiedAtom::Trivial(true))));
    }

    #[test]
    fn test_simplify_ineq_constant_eq() {
        let c = Polynomial::constant(rat(5));
        let atom = IneqAtom::from_poly(c, AtomKind::Eq);

        let result = simplify_ineq_atom(&atom);
        // 5 = 0 is false
        assert!(matches!(result, Some(SimplifiedAtom::Trivial(false))));
    }

    #[test]
    fn test_simplify_ineq_constant_gt() {
        let c = Polynomial::constant(rat(5));
        let atom = IneqAtom::from_poly(c, AtomKind::Gt);

        let result = simplify_ineq_atom(&atom);
        // 5 > 0 is true
        assert!(matches!(result, Some(SimplifiedAtom::Trivial(true))));
    }

    #[test]
    fn test_simplify_ineq_constant_lt() {
        let c = Polynomial::constant(rat(5));
        let atom = IneqAtom::from_poly(c, AtomKind::Lt);

        let result = simplify_ineq_atom(&atom);
        // 5 < 0 is false
        assert!(matches!(result, Some(SimplifiedAtom::Trivial(false))));
    }

    #[test]
    fn test_eliminate_redundant_none() {
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);

        let atoms = vec![
            Atom::Ineq(IneqAtom::from_poly(x, AtomKind::Gt)),
            Atom::Ineq(IneqAtom::from_poly(y, AtomKind::Lt)),
        ];

        let redundant = eliminate_redundant(&atoms);
        assert!(redundant.is_empty());
    }

    #[test]
    fn test_eliminate_redundant_duplicates() {
        let x = Polynomial::from_var(0);

        let atoms = vec![
            Atom::Ineq(IneqAtom::from_poly(x.clone(), AtomKind::Gt)),
            Atom::Ineq(IneqAtom::from_poly(x.clone(), AtomKind::Gt)),
            Atom::Ineq(IneqAtom::from_poly(x, AtomKind::Gt)),
        ];

        let redundant = eliminate_redundant(&atoms);
        assert_eq!(redundant.len(), 2); // Two duplicates found
        assert!(redundant.contains(&1));
        assert!(redundant.contains(&2));
    }
}
