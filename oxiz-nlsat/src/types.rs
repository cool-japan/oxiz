//! Type definitions for NLSAT.
//!
//! This module defines the core types used in the NLSAT solver for
//! non-linear arithmetic satisfiability.
//!
//! Reference: Z3's `nlsat/nlsat_types.h`

use oxiz_math::polynomial::{NULL_VAR, Polynomial, Var};
use std::fmt;

/// Boolean variable index.
pub type BoolVar = u32;

/// Null boolean variable.
pub const NULL_BOOL_VAR: BoolVar = u32::MAX;

/// True boolean variable (constant true).
pub const TRUE_BOOL_VAR: BoolVar = 0;

/// A literal is a boolean variable with an optional negation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal {
    /// The encoded literal: `2 * var + sign`.
    data: u32,
}

impl Literal {
    /// Create a positive literal.
    #[inline]
    pub fn positive(var: BoolVar) -> Self {
        Self { data: var << 1 }
    }

    /// Create a negative literal.
    #[inline]
    pub fn negative(var: BoolVar) -> Self {
        Self {
            data: (var << 1) | 1,
        }
    }

    /// Create a literal with the given sign.
    #[inline]
    pub fn new(var: BoolVar, negated: bool) -> Self {
        if negated {
            Self::negative(var)
        } else {
            Self::positive(var)
        }
    }

    /// Get the boolean variable.
    #[inline]
    pub fn var(self) -> BoolVar {
        self.data >> 1
    }

    /// Check if the literal is negated.
    #[inline]
    pub fn is_negated(self) -> bool {
        (self.data & 1) == 1
    }

    /// Check if the literal is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        !self.is_negated()
    }

    /// Get the negation of this literal.
    #[inline]
    pub fn negate(self) -> Self {
        Self {
            data: self.data ^ 1,
        }
    }

    /// Get the raw encoded value.
    #[inline]
    pub fn index(self) -> u32 {
        self.data
    }

    /// Create from raw encoded value.
    #[inline]
    pub fn from_index(index: u32) -> Self {
        Self { data: index }
    }
}

impl fmt::Debug for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_negated() {
            write!(f, "~b{}", self.var())
        } else {
            write!(f, "b{}", self.var())
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// True literal.
pub const TRUE_LITERAL: Literal = Literal {
    data: TRUE_BOOL_VAR << 1,
};

/// False literal.
pub const FALSE_LITERAL: Literal = Literal {
    data: (TRUE_BOOL_VAR << 1) | 1,
};

/// Null literal (invalid/unassigned).
pub const NULL_LITERAL: Literal = Literal {
    data: NULL_BOOL_VAR << 1,
};

/// Kind of atom (comparison operator).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AtomKind {
    /// Equality: p = 0
    Eq = 0,
    /// Less than: p < 0
    Lt = 1,
    /// Greater than: p > 0
    Gt = 2,
    /// Root equality: x = root\[i\](p)
    RootEq = 10,
    /// Root less than: x < root\[i\](p)
    RootLt = 11,
    /// Root greater than: x > root\[i\](p)
    RootGt = 12,
    /// Root less or equal: x <= root\[i\](p)
    RootLe = 13,
    /// Root greater or equal: x >= root\[i\](p)
    RootGe = 14,
}

impl AtomKind {
    /// Check if this is an inequality atom (EQ, LT, GT).
    #[inline]
    pub fn is_ineq(self) -> bool {
        (self as u8) <= 2
    }

    /// Check if this is a root atom.
    #[inline]
    pub fn is_root(self) -> bool {
        (self as u8) >= 10
    }

    /// Check if this is an equality.
    #[inline]
    pub fn is_eq(self) -> bool {
        matches!(self, AtomKind::Eq | AtomKind::RootEq)
    }

    /// Flip the direction (LT <-> GT).
    pub fn flip(self) -> Self {
        match self {
            AtomKind::Lt => AtomKind::Gt,
            AtomKind::Gt => AtomKind::Lt,
            AtomKind::RootLt => AtomKind::RootGt,
            AtomKind::RootGt => AtomKind::RootLt,
            AtomKind::RootLe => AtomKind::RootGe,
            AtomKind::RootGe => AtomKind::RootLe,
            other => other,
        }
    }

    /// Negate the condition.
    pub fn negate(self) -> Self {
        match self {
            AtomKind::Eq => AtomKind::Lt, // Actually can't negate EQ to one kind
            AtomKind::Lt => AtomKind::Gt, // Actually >= 0, but we approximate
            AtomKind::Gt => AtomKind::Lt, // Actually <= 0
            AtomKind::RootEq => AtomKind::RootEq, // Complex
            AtomKind::RootLt => AtomKind::RootGe,
            AtomKind::RootGt => AtomKind::RootLe,
            AtomKind::RootLe => AtomKind::RootGt,
            AtomKind::RootGe => AtomKind::RootLt,
        }
    }
}

impl fmt::Display for AtomKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomKind::Eq => write!(f, "="),
            AtomKind::Lt => write!(f, "<"),
            AtomKind::Gt => write!(f, ">"),
            AtomKind::RootEq => write!(f, "= root"),
            AtomKind::RootLt => write!(f, "< root"),
            AtomKind::RootGt => write!(f, "> root"),
            AtomKind::RootLe => write!(f, "<= root"),
            AtomKind::RootGe => write!(f, ">= root"),
        }
    }
}

/// A polynomial factor with parity information.
#[derive(Debug, Clone)]
pub struct PolyFactor {
    /// The polynomial.
    pub poly: Polynomial,
    /// Whether the factor has even degree in the product.
    pub is_even: bool,
}

/// An inequality atom: p1^k1 * p2^k2 * ... op 0
/// where op is =, <, or >.
#[derive(Debug, Clone)]
pub struct IneqAtom {
    /// Kind of comparison.
    pub kind: AtomKind,
    /// Polynomial factors with parity.
    pub factors: Vec<PolyFactor>,
    /// Maximum variable in any factor.
    pub max_var: Var,
    /// Associated boolean variable.
    pub bool_var: BoolVar,
}

impl IneqAtom {
    /// Create a new inequality atom from a single polynomial.
    pub fn from_poly(poly: Polynomial, kind: AtomKind) -> Self {
        let max_var = poly.max_var();
        Self {
            kind,
            factors: vec![PolyFactor {
                poly,
                is_even: false,
            }],
            max_var,
            bool_var: NULL_BOOL_VAR,
        }
    }

    /// Create a new inequality atom from multiple factors.
    pub fn from_factors(factors: Vec<PolyFactor>, kind: AtomKind) -> Self {
        let max_var = factors
            .iter()
            .map(|f| f.poly.max_var())
            .filter(|&v| v != NULL_VAR)
            .max()
            .unwrap_or(NULL_VAR);
        Self {
            kind,
            factors,
            max_var,
            bool_var: NULL_BOOL_VAR,
        }
    }

    /// Get the number of factors.
    #[inline]
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Evaluate the sign of the atom under an assignment.
    /// Returns true if the constraint is satisfied.
    pub fn evaluate_sign(&self, signs: &[i8]) -> Option<bool> {
        // Compute the product sign
        let mut product_sign: i8 = 1;
        for (i, factor) in self.factors.iter().enumerate() {
            let sign = signs.get(i).copied()?;
            if sign == 0 {
                product_sign = 0;
                break;
            }
            if !factor.is_even {
                product_sign *= sign;
            }
        }

        Some(match self.kind {
            AtomKind::Eq => product_sign == 0,
            AtomKind::Lt => product_sign < 0,
            AtomKind::Gt => product_sign > 0,
            _ => unreachable!("IneqAtom should not have root kind"),
        })
    }
}

/// A root atom: x op root\[i\](p)
/// where op is =, <, >, <=, >=.
#[derive(Debug, Clone)]
pub struct RootAtom {
    /// Kind of comparison.
    pub kind: AtomKind,
    /// The variable being compared.
    pub var: Var,
    /// The root index (1-based).
    pub root_index: u32,
    /// The polynomial whose root we're comparing against.
    pub poly: Polynomial,
    /// Associated boolean variable.
    pub bool_var: BoolVar,
}

impl RootAtom {
    /// Create a new root atom.
    pub fn new(kind: AtomKind, var: Var, root_index: u32, poly: Polynomial) -> Self {
        debug_assert!(kind.is_root());
        Self {
            kind,
            var,
            root_index,
            poly,
            bool_var: NULL_BOOL_VAR,
        }
    }

    /// Get the maximum variable in this atom.
    pub fn max_var(&self) -> Var {
        self.var.max(self.poly.max_var())
    }
}

/// An atom in the NLSAT solver.
#[derive(Debug, Clone)]
pub enum Atom {
    /// Inequality atom.
    Ineq(IneqAtom),
    /// Root atom.
    Root(RootAtom),
}

impl Atom {
    /// Get the kind of the atom.
    pub fn kind(&self) -> AtomKind {
        match self {
            Atom::Ineq(a) => a.kind,
            Atom::Root(a) => a.kind,
        }
    }

    /// Get the boolean variable.
    pub fn bool_var(&self) -> BoolVar {
        match self {
            Atom::Ineq(a) => a.bool_var,
            Atom::Root(a) => a.bool_var,
        }
    }

    /// Set the boolean variable.
    pub fn set_bool_var(&mut self, bvar: BoolVar) {
        match self {
            Atom::Ineq(a) => a.bool_var = bvar,
            Atom::Root(a) => a.bool_var = bvar,
        }
    }

    /// Get the maximum variable.
    pub fn max_var(&self) -> Var {
        match self {
            Atom::Ineq(a) => a.max_var,
            Atom::Root(a) => a.max_var(),
        }
    }

    /// Check if this is an inequality atom.
    pub fn is_ineq(&self) -> bool {
        matches!(self, Atom::Ineq(_))
    }

    /// Check if this is a root atom.
    pub fn is_root(&self) -> bool {
        matches!(self, Atom::Root(_))
    }

    /// Check if this is an equality atom.
    pub fn is_eq(&self) -> bool {
        self.kind().is_eq()
    }
}

/// Lbool (three-valued boolean for SAT solving).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Lbool {
    /// Unknown/unassigned.
    #[default]
    Undef = 0,
    /// True.
    True = 1,
    /// False.
    False = 2,
}

impl Lbool {
    /// Convert from bool.
    #[inline]
    pub fn from_bool(b: bool) -> Self {
        if b { Lbool::True } else { Lbool::False }
    }

    /// Convert to bool (panics if Undef).
    #[inline]
    pub fn to_bool(self) -> Option<bool> {
        match self {
            Lbool::True => Some(true),
            Lbool::False => Some(false),
            Lbool::Undef => None,
        }
    }

    /// Check if this is undefined.
    #[inline]
    pub fn is_undef(self) -> bool {
        self == Lbool::Undef
    }

    /// Check if this is true.
    #[inline]
    pub fn is_true(self) -> bool {
        self == Lbool::True
    }

    /// Check if this is false.
    #[inline]
    pub fn is_false(self) -> bool {
        self == Lbool::False
    }

    /// Negate the value.
    #[inline]
    pub fn negate(self) -> Self {
        match self {
            Lbool::True => Lbool::False,
            Lbool::False => Lbool::True,
            Lbool::Undef => Lbool::Undef,
        }
    }
}

impl From<bool> for Lbool {
    fn from(b: bool) -> Self {
        Lbool::from_bool(b)
    }
}

impl fmt::Display for Lbool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lbool::True => write!(f, "true"),
            Lbool::False => write!(f, "false"),
            Lbool::Undef => write!(f, "undef"),
        }
    }
}

/// Normalize a sign to -1, 0, or 1.
#[inline]
pub fn normalize_sign(s: i32) -> i8 {
    if s < 0 {
        -1
    } else if s > 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal() {
        let l = Literal::positive(5);
        assert_eq!(l.var(), 5);
        assert!(!l.is_negated());
        assert!(l.is_positive());

        let nl = l.negate();
        assert_eq!(nl.var(), 5);
        assert!(nl.is_negated());

        assert_eq!(nl.negate(), l);
    }

    #[test]
    fn test_literal_encoding() {
        for v in 0..100 {
            let pos = Literal::positive(v);
            let neg = Literal::negative(v);

            assert_eq!(pos.var(), v);
            assert_eq!(neg.var(), v);
            assert!(pos.is_positive());
            assert!(neg.is_negated());
            assert_eq!(pos.negate(), neg);
            assert_eq!(neg.negate(), pos);
        }
    }

    #[test]
    fn test_atom_kind() {
        assert!(AtomKind::Eq.is_ineq());
        assert!(AtomKind::Lt.is_ineq());
        assert!(AtomKind::Gt.is_ineq());
        assert!(!AtomKind::RootEq.is_ineq());

        assert!(!AtomKind::Eq.is_root());
        assert!(AtomKind::RootEq.is_root());
        assert!(AtomKind::RootLt.is_root());

        assert!(AtomKind::Eq.is_eq());
        assert!(AtomKind::RootEq.is_eq());
        assert!(!AtomKind::Lt.is_eq());
    }

    #[test]
    fn test_lbool() {
        assert!(Lbool::Undef.is_undef());
        assert!(Lbool::True.is_true());
        assert!(Lbool::False.is_false());

        assert_eq!(Lbool::True.negate(), Lbool::False);
        assert_eq!(Lbool::False.negate(), Lbool::True);
        assert_eq!(Lbool::Undef.negate(), Lbool::Undef);

        assert_eq!(Lbool::from_bool(true), Lbool::True);
        assert_eq!(Lbool::from_bool(false), Lbool::False);
    }

    // Property-based tests
    use proptest::prelude::*;

    proptest! {
        /// Property: Double negation returns original literal
        #[test]
        fn prop_literal_double_negation_is_identity(var in 0u32..1000) {
            let lit = Literal::positive(var);
            prop_assert_eq!(lit.negate().negate(), lit);

            let neg_lit = Literal::negative(var);
            prop_assert_eq!(neg_lit.negate().negate(), neg_lit);
        }

        /// Property: Variable is preserved through negation
        #[test]
        fn prop_literal_var_preserved_by_negation(var in 0u32..1000) {
            let lit = Literal::positive(var);
            prop_assert_eq!(lit.var(), var);
            prop_assert_eq!(lit.negate().var(), var);
        }

        /// Property: Encoding/decoding is bijective
        #[test]
        fn prop_literal_encoding_bijective(var in 0u32..1000, negated: bool) {
            let lit = Literal::new(var, negated);
            let index = lit.index();
            let decoded = Literal::from_index(index);

            prop_assert_eq!(decoded, lit);
            prop_assert_eq!(decoded.var(), var);
            prop_assert_eq!(decoded.is_negated(), negated);
        }

        /// Property: Positive and negative literals are complementary
        #[test]
        fn prop_literal_positive_negative_complement(var in 0u32..1000) {
            let pos = Literal::positive(var);
            let neg = Literal::negative(var);

            prop_assert_eq!(pos.negate(), neg);
            prop_assert_eq!(neg.negate(), pos);
            prop_assert!(pos.is_positive());
            prop_assert!(neg.is_negated());
        }

        /// Property: Literal index uniqueness
        #[test]
        fn prop_literal_index_uniqueness(var1 in 0u32..500, var2 in 500u32..1000, sign1: bool, sign2: bool) {
            let lit1 = Literal::new(var1, sign1);
            let lit2 = Literal::new(var2, sign2);

            // Different variables should have different indices
            if var1 != var2 || sign1 != sign2 {
                prop_assert_ne!(lit1.index(), lit2.index());
            } else {
                prop_assert_eq!(lit1.index(), lit2.index());
            }
        }

        /// Property: Lbool negation is involutive
        #[test]
        fn prop_lbool_negation_involutive(b: bool) {
            let lbool = Lbool::from_bool(b);
            prop_assert_eq!(lbool.negate().negate(), lbool);
        }

        /// Property: Lbool truth values are consistent
        #[test]
        fn prop_lbool_truth_consistency(b: bool) {
            let lbool = Lbool::from_bool(b);
            prop_assert_eq!(lbool.is_true(), b);
            prop_assert_eq!(lbool.is_false(), !b);
            prop_assert!(!lbool.is_undef());
        }
    }
}
