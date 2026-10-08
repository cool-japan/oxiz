//! Literals and Variables

#[allow(unused_imports)]
use crate::prelude::*;
use core::ops::Not;

/// A variable in the SAT formula
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Var(pub u32);

impl Var {
    /// The largest variable index this representation supports.
    ///
    /// Two independent packing invariants meet at the same number, and
    /// `(1 << 31) - 2` is the largest index that satisfies both:
    ///
    /// * the largest index a [`Lit`] can pack — a literal is `index << 1 | sign`
    ///   in a `u32`, so the index must stay below `2^31`; and
    /// * the largest index DIMACS can carry with **both** signs — the external
    ///   form is `index + 1` with a sign, so `index + 1 <= i32::MAX` is needed
    ///   for `-(index + 1)` to exist as an `i32`.
    ///
    /// Above it, `Lit::pos` used to truncate the index silently and
    /// [`Lit::to_dimacs`] used to return a wrong number rather than fail.  The
    /// bound is now checked by `debug_assert!` at every construction site and
    /// enforced unconditionally by `to_dimacs` / [`Lit::try_to_dimacs`].
    pub const MAX_INDEX: u32 = (1 << 31) - 2;

    /// Create a new variable from a raw index
    ///
    /// # Panics
    ///
    /// In debug builds, when `idx` exceeds [`Var::MAX_INDEX`].
    #[must_use]
    pub const fn new(idx: u32) -> Self {
        debug_assert!(idx <= Var::MAX_INDEX);
        Self(idx)
    }

    /// Get the raw index
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A literal (positive or negative occurrence of a variable)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lit(u32);

impl Lit {
    /// Create a positive literal from a variable
    ///
    /// # Panics
    ///
    /// In debug builds, when the variable index exceeds [`Var::MAX_INDEX`] —
    /// above that bound `var.0 << 1` overflows the `u32` packing and the
    /// literal would silently denote a *different* variable.
    #[must_use]
    pub const fn pos(var: Var) -> Self {
        debug_assert!(var.0 <= Var::MAX_INDEX);
        Self(var.0 << 1)
    }

    /// Create a negative literal from a variable
    ///
    /// # Panics
    ///
    /// In debug builds, when the variable index exceeds [`Var::MAX_INDEX`]
    /// (see [`Lit::pos`]).
    #[must_use]
    pub const fn neg(var: Var) -> Self {
        debug_assert!(var.0 <= Var::MAX_INDEX);
        Self((var.0 << 1) | 1)
    }

    /// Create a literal from a signed integer (DIMACS format)
    /// Positive int -> positive literal, negative int -> negative literal
    ///
    /// # Panics
    ///
    /// In debug builds, when `lit` is `0` (DIMACS reserves it as the clause
    /// terminator, and there is no variable `-1`) or [`i32::MIN`] (whose
    /// magnitude is `2^31`, one past what a `Lit` can pack: the resulting
    /// index `2^31 - 1` exceeds [`Var::MAX_INDEX`] and its `to_dimacs` has no
    /// negative counterpart).
    #[must_use]
    pub fn from_dimacs(lit: i32) -> Self {
        debug_assert!(lit != 0 && lit != i32::MIN);
        let var = Var::new(lit.unsigned_abs() - 1);
        if lit > 0 {
            Self::pos(var)
        } else {
            Self::neg(var)
        }
    }

    /// Convert to DIMACS format
    ///
    /// # Panics
    ///
    /// When the variable index exceeds [`Var::MAX_INDEX`], in *every* build
    /// profile.  Such a literal has no DIMACS form at all: `index + 1` does
    /// not fit an `i32`, and the pre-fix code returned a wrong number for it
    /// (`(index + 1) as i32` wrapping to a negative value, then negated a
    /// second time, so a positive literal printed as a negative one).  A
    /// documented panic is the honest answer; use [`Lit::try_to_dimacs`] for a
    /// non-panicking form.
    #[must_use]
    pub fn to_dimacs(self) -> i32 {
        match self.try_to_dimacs() {
            Some(dimacs) => dimacs,
            None => panic!(
                "Lit::to_dimacs: variable index exceeds Var::MAX_INDEX \
                 ((1 << 31) - 2), so index + 1 has no i32 representation"
            ),
        }
    }

    /// Convert to DIMACS format, or `None` when the literal has no DIMACS
    /// form.
    ///
    /// The non-panicking counterpart of [`Lit::to_dimacs`]: it is `None`
    /// exactly where that method panics, i.e. for a variable index above
    /// [`Var::MAX_INDEX`].
    #[must_use]
    pub fn try_to_dimacs(self) -> Option<i32> {
        let magnitude = i32::try_from(self.var().0.checked_add(1)?).ok()?;
        Some(if self.is_pos() { magnitude } else { -magnitude })
    }

    /// Get the variable of this literal
    #[must_use]
    pub const fn var(self) -> Var {
        Var(self.0 >> 1)
    }

    /// Check if this is a positive literal
    #[must_use]
    pub const fn is_pos(self) -> bool {
        (self.0 & 1) == 0
    }

    /// Check if this is a negative literal
    #[must_use]
    pub const fn is_neg(self) -> bool {
        (self.0 & 1) == 1
    }

    /// Get the negation of this literal
    #[must_use]
    pub const fn negate(self) -> Self {
        Self(self.0 ^ 1)
    }

    /// Get the sign (true for positive, false for negative)
    #[must_use]
    pub const fn sign(self) -> bool {
        self.is_pos()
    }

    /// Get the raw encoding
    #[must_use]
    pub const fn code(self) -> u32 {
        self.0
    }

    /// Create from raw encoding
    #[must_use]
    pub const fn from_code(code: u32) -> Self {
        Self(code)
    }

    /// Get the index for array access (2 * var + sign)
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl Not for Lit {
    type Output = Self;

    fn not(self) -> Self::Output {
        self.negate()
    }
}

/// Boolean value or undefined
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LBool {
    /// True
    True,
    /// False
    False,
    /// Undefined
    Undef,
}

impl LBool {
    /// Create from a boolean
    #[must_use]
    pub const fn from_bool(b: bool) -> Self {
        if b { Self::True } else { Self::False }
    }

    /// Check if defined
    #[must_use]
    pub const fn is_defined(self) -> bool {
        !matches!(self, Self::Undef)
    }

    /// Check if true
    #[must_use]
    pub const fn is_true(self) -> bool {
        matches!(self, Self::True)
    }

    /// Check if false
    #[must_use]
    pub const fn is_false(self) -> bool {
        matches!(self, Self::False)
    }

    /// Negate
    #[must_use]
    pub const fn negate(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Undef => Self::Undef,
        }
    }
}

impl From<bool> for LBool {
    fn from(b: bool) -> Self {
        Self::from_bool(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let var = Var::new(0);
        let pos = Lit::pos(var);
        let neg = Lit::neg(var);

        assert!(pos.is_pos());
        assert!(neg.is_neg());
        assert_eq!(pos.var(), var);
        assert_eq!(neg.var(), var);
        assert_eq!(pos.negate(), neg);
        assert_eq!(neg.negate(), pos);
    }

    #[test]
    fn test_dimacs_conversion() {
        let lit = Lit::from_dimacs(1);
        assert!(lit.is_pos());
        assert_eq!(lit.var(), Var::new(0));
        assert_eq!(lit.to_dimacs(), 1);

        let lit = Lit::from_dimacs(-2);
        assert!(lit.is_neg());
        assert_eq!(lit.var(), Var::new(1));
        assert_eq!(lit.to_dimacs(), -2);
    }

    /// Round trips at the two ends of the representable index range.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// `Var::MAX_INDEX` did not exist, so this test could not be written at
    /// all. `Lit::pos(Var::new(Var::MAX_INDEX)).to_dimacs()` evaluated
    /// `(2147483646 + 1) as i32` = `2147483647` and answered `2147483647`,
    /// which is right — the bound is exactly where the old arithmetic stopped
    /// being wrong, which is why it is the bound.
    #[test]
    fn dimacs_round_trip_at_the_index_extremes() {
        for index in [0u32, 1, Var::MAX_INDEX] {
            let var = Var::new(index);
            let pos = Lit::pos(var);
            let neg = Lit::neg(var);
            assert_eq!(pos.var(), var, "positive literal lost its variable");
            assert_eq!(neg.var(), var, "negative literal lost its variable");
            assert!(pos.is_pos());
            assert!(neg.is_neg());
            let dimacs = pos.to_dimacs();
            assert_eq!(dimacs, i32::try_from(index + 1).unwrap_or(i32::MAX));
            assert_eq!(Lit::from_dimacs(dimacs), pos);
            assert_eq!(Lit::from_dimacs(-dimacs), neg);
            assert_eq!(neg.to_dimacs(), -dimacs);
        }
    }

    /// `from_dimacs(i32::MAX)` names variable `i32::MAX - 1`, which is exactly
    /// [`Var::MAX_INDEX`], and comes back unchanged in both polarities.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// The same values — `i32::MAX` was already the largest *well-behaved*
    /// input. The regression this pins is the boundary itself: one more
    /// (`i32::MIN`, magnitude `2^31`) silently produced `Var(2^31 - 1)`, whose
    /// `to_dimacs` computed `(2147483647 + 1) as i32` = `i32::MIN` and then
    /// negated it — an overflow panic in debug and `i32::MIN` again in
    /// release. `Var::MAX_INDEX` is now the documented line between the two.
    #[test]
    fn from_dimacs_i32_max_round_trips() {
        let lit = Lit::from_dimacs(i32::MAX);
        assert_eq!(lit.var(), Var::new(Var::MAX_INDEX));
        assert!(lit.is_pos());
        assert_eq!(lit.to_dimacs(), i32::MAX);

        let neg = Lit::from_dimacs(-i32::MAX);
        assert_eq!(neg.var(), Var::new(Var::MAX_INDEX));
        assert!(neg.is_neg());
        assert_eq!(neg.to_dimacs(), -i32::MAX);
        assert_eq!(neg, lit.negate());
    }

    /// `try_to_dimacs` is `Some` exactly where `to_dimacs` does not panic.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// `try_to_dimacs` did not exist, and `to_dimacs` never panicked: for the
    /// literal packed from index `2^31 - 1` (reachable via
    /// `Lit::from_code`, and via `from_dimacs(i32::MIN)` in release) it
    /// returned `i32::MIN` for *both* polarities — a positive literal
    /// reported as a negative one, and two distinct literals sharing one
    /// DIMACS number.
    #[test]
    fn try_to_dimacs_is_none_exactly_where_to_dimacs_panics() {
        // Representable: agrees with `to_dimacs`.
        for index in [0u32, 1, 12345, Var::MAX_INDEX] {
            let lit = Lit::pos(Var::new(index));
            assert_eq!(lit.try_to_dimacs(), Some(lit.to_dimacs()));
            let lit = Lit::neg(Var::new(index));
            assert_eq!(lit.try_to_dimacs(), Some(lit.to_dimacs()));
        }
        // Not representable. `Var::MAX_INDEX + 1` (= `u32::MAX >> 1`) is the
        // *only* such index a `Lit` can even hold — a larger one does not
        // survive the `index << 1` packing — and `from_code` is the only
        // constructor that reaches it without tripping a debug assertion,
        // which is what makes the non-panicking accessor worth having.
        let index = Var::MAX_INDEX + 1;
        assert_eq!(index, u32::MAX >> 1);
        let code = index << 1;
        assert_eq!(Lit::from_code(code).var().0, index);
        assert_eq!(Lit::from_code(code).try_to_dimacs(), None);
        assert_eq!(Lit::from_code(code | 1).try_to_dimacs(), None);
    }

    /// `to_dimacs` panics rather than answering a wrong number for a literal
    /// whose index has no DIMACS form. Unconditional, so it runs in release
    /// too — this is the one check that is *not* debug-only.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// `i32::MIN` (see `try_to_dimacs_is_none_exactly_where_to_dimacs_panics`).
    #[test]
    #[should_panic(expected = "Lit::to_dimacs")]
    fn to_dimacs_panics_above_max_index() {
        let _ = Lit::from_code((Var::MAX_INDEX + 1) << 1).to_dimacs();
    }

    /// The packing bound is checked where literals are built.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// No panic and no error: `Lit::pos(Var(2^31))` computed `2^31 << 1` = `0`
    /// (the shift overflows the `u32`), so the literal denoted **variable 0**,
    /// positively. Every clause built from it constrained the wrong variable.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn lit_pos_rejects_an_index_above_max_index_in_debug() {
        let _ = Lit::pos(Var(Var::MAX_INDEX + 1));
    }

    /// Same bound, negative polarity.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// `Lit::neg(Var(2^31))` = `Lit(1)`, i.e. the negation of variable 0.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn lit_neg_rejects_an_index_above_max_index_in_debug() {
        let _ = Lit::neg(Var(Var::MAX_INDEX + 1));
    }

    /// `Var::new` rejects the same indices its literals cannot pack.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// `Var::new(u32::MAX)` was accepted silently; the truncation only showed
    /// up later, in whichever literal was built from it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn var_new_rejects_an_index_above_max_index_in_debug() {
        let _ = Var::new(Var::MAX_INDEX + 1);
    }

    /// `from_dimacs` rejects the two inputs that have no literal.
    ///
    /// # What OxiZ 0.3.3 / 0.3.4 answered before this fix
    ///
    /// `from_dimacs(i32::MIN)` returned `Lit` packing `Var(2^31 - 1)` with no
    /// complaint (`0` was already rejected by the existing `debug_assert!`,
    /// via the `unsigned_abs() - 1` underflow).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn from_dimacs_rejects_i32_min_in_debug() {
        let _ = Lit::from_dimacs(i32::MIN);
    }

    #[test]
    fn test_lbool() {
        assert!(LBool::True.is_true());
        assert!(LBool::False.is_false());
        assert!(!LBool::Undef.is_defined());

        assert_eq!(LBool::True.negate(), LBool::False);
        assert_eq!(LBool::False.negate(), LBool::True);
        assert_eq!(LBool::Undef.negate(), LBool::Undef);
    }
}
