//! Literal Type for Theory Interfaces.
//!
//! Provides a minimal literal representation for use in trait definitions.
//! This avoids circular dependencies with oxiz-sat.

use std::fmt;

/// A Boolean variable identifier.
pub type Var = u32;

/// A literal (signed Boolean variable).
///
/// This is a lightweight newtype wrapper compatible with oxiz-sat::Lit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lit(u32);

impl Lit {
    /// Create a positive literal from a variable.
    #[must_use]
    pub const fn positive(var: Var) -> Self {
        Self(var << 1)
    }

    /// Create a negative literal from a variable.
    #[must_use]
    pub const fn negative(var: Var) -> Self {
        Self((var << 1) | 1)
    }

    /// Get the variable of this literal.
    #[must_use]
    pub const fn var(self) -> Var {
        self.0 >> 1
    }

    /// Check if this literal is positive.
    #[must_use]
    pub const fn is_positive(self) -> bool {
        (self.0 & 1) == 0
    }

    /// Check if this literal is negative.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        (self.0 & 1) != 0
    }

    /// Get the negation of this literal.
    #[must_use]
    pub const fn negate(self) -> Self {
        Self(self.0 ^ 1)
    }

    /// Get the raw value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Create from raw value.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

impl fmt::Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_positive() {
            write!(f, "{}", self.var())
        } else {
            write!(f, "-{}", self.var())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_literal() {
        let lit = Lit::positive(5);
        assert!(lit.is_positive());
        assert!(!lit.is_negative());
        assert_eq!(lit.var(), 5);
    }

    #[test]
    fn test_negative_literal() {
        let lit = Lit::negative(5);
        assert!(!lit.is_positive());
        assert!(lit.is_negative());
        assert_eq!(lit.var(), 5);
    }

    #[test]
    fn test_negation() {
        let pos = Lit::positive(3);
        let neg = pos.negate();

        assert!(pos.is_positive());
        assert!(neg.is_negative());
        assert_eq!(pos.var(), neg.var());
        assert_eq!(pos, neg.negate());
    }

    #[test]
    fn test_raw_value() {
        let lit = Lit::positive(10);
        let raw = lit.raw();
        let restored = Lit::from_raw(raw);

        assert_eq!(lit, restored);
    }
}
