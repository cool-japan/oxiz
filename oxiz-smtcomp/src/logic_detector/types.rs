//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)


/// Kind of syntactic construct at the current depth on the nesting stack.
///
/// Used by [`extract_structural_features`] to track which construct each open
/// parenthesis belongs to so that closing parentheses can update the correct
/// depth counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstructKind {
    /// A `forall` or `exists` quantifier binder.
    Quantifier,
    /// An `ite` (if-then-else) expression.
    Ite,
    /// A `let` binder.
    Let,
    /// An `(assert ...)` command body.
    Assert,
    /// A Boolean connective body such as `and` / `or` / `not`.
    Bool,
    /// An `(Array ...)` sort annotation.
    Array,
    /// Any other parenthesised expression.
    Other,
}
/// Theory feature flags detected in an SMT-LIB benchmark.
///
/// Each field records whether the associated theory appears anywhere in the
/// scanned source. `TheoryBits` is a flat, copyable descriptor so it can be
/// combined, compared, and mapped to logic names without additional
/// allocation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TheoryBits {
    /// User-declared uninterpreted functions with non-empty argument lists.
    pub has_uf: bool,
    /// Integer sort or any integer-flavoured operator.
    pub has_int: bool,
    /// Real sort or any real-flavoured operator.
    pub has_real: bool,
    /// `BitVec` sort or any `bv*` operator.
    pub has_bv: bool,
    /// Array sort or any `select`/`store` operator.
    pub has_array: bool,
    /// `String` sort or any `str.*` operator.
    pub has_string: bool,
    /// `FloatingPoint` sort or any `fp.*` operator.
    pub has_fp: bool,
    /// `declare-datatype(s)` declarations.
    pub has_dt: bool,
    /// Nonlinear arithmetic (variable-by-variable `*` / `/` / `mod` / `pow`).
    pub has_nonlinear: bool,
    /// `forall` / `exists` binders.
    pub has_quantifier: bool,
}
impl TheoryBits {
    /// Whether any theory bit is set.
    ///
    /// Used by the logic mapper to distinguish "no signal — probably a pure
    /// Boolean script" from "something interesting was detected".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !(self.has_uf || self.has_int || self.has_real || self.has_bv || self.has_array
            || self.has_string || self.has_fp || self.has_dt || self.has_nonlinear
            || self.has_quantifier)
    }
}
/// Structural features extracted by scanning the raw SMT-LIB source text.
#[derive(Debug, Clone, Default)]
pub struct StructuralFeatures {
    /// Maximum nesting depth of parenthesized terms.
    pub max_term_depth: u32,
    /// Number of predicate applications and comparison atoms.
    pub atom_count: u32,
    /// Number of `(assert ...)` commands.
    pub clause_count: u32,
    /// Maximum nesting depth of `forall` / `exists`.
    pub max_quantifier_nesting: u32,
    /// Histogram of bit-vector widths as `(width, count)`.
    pub bv_width_histogram: Vec<(u32, u32)>,
    /// Histogram of nested array sort dimensions as `(dims, count)`.
    pub array_dim_histogram: Vec<(u32, u32)>,
    /// Maximum nesting depth of `ite`.
    pub max_ite_depth: u32,
    /// Maximum nesting depth of `let`.
    pub max_let_depth: u32,
}
