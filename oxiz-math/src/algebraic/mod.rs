//! Algebraic Numbers.
//!
//! Support for algebraic numbers represented as roots of polynomials.
//! Essential for CAD (Cylindrical Algebraic Decomposition) and
//! non-linear real arithmetic.

pub mod number;
pub mod isolate;
pub mod field_extension;
pub mod galois_theory;

pub use number::{AlgebraicNumber, AlgebraicNumberError};
pub use isolate::{IntervalRefinement, RootIsolator};
pub use field_extension::{
    ExtensionId, FieldElement, FieldExtension, FieldExtensionManager,
};
pub use galois_theory::{
    Automorphism, Discriminant, GaloisComputation, GaloisGroup,
};
