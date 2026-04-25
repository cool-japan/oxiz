//! Algebraic Numbers.
//!
//! Support for algebraic numbers represented as roots of polynomials.
//! Essential for CAD (Cylindrical Algebraic Decomposition) and
//! non-linear real arithmetic.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod field_extension;
pub mod galois_theory;
pub mod isolate;
pub mod number;

pub use field_extension::{ExtensionId, FieldElement, FieldExtension, FieldExtensionManager};
pub use galois_theory::{Automorphism, Discriminant, GaloisComputation, GaloisGroup};
pub use isolate::{IntervalRefinement, RootIsolator};
pub use number::{AlgebraicNumber, AlgebraicNumberError};
