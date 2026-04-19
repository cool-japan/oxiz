//! Arithmetic Quantifier Elimination.
//!
//! This module implements quantifier elimination procedures for linear
//! arithmetic over integers and reals.

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;

pub mod cooper;
pub mod ferrante_rackoff;
pub mod omega_test;
pub mod qe_lite_arith;
pub mod virtual_term;

pub use cooper::{CooperEliminator, CooperStats};
pub use ferrante_rackoff::{
    FerranteRackoffEliminator, FerranteRackoffStats, Inequality, InequalityType,
};
pub use omega_test::{
    LinearConstraint as OmegaLinearConstraint, OmegaResult, OmegaTestConfig as OmegaConfig,
    OmegaTestStats as OmegaStats, OmegaTester as OmegaTest,
};
pub use virtual_term::{VirtualTermEliminator, VirtualTermStats};

/// Eliminate one linear arithmetic variable, using virtual substitution when
/// strict inequalities appear in the formula.
pub fn eliminate_linear(var: TermId, formula: TermId, tm: &mut TermManager) -> TermId {
    if contains_strict_inequality(formula, tm) {
        crate::qe::virtual_substitution::eliminate_quantifier_vs(var, &formula, tm)
    } else {
        formula
    }
}

fn contains_strict_inequality(term: TermId, tm: &TermManager) -> bool {
    let Some(node) = tm.get(term) else {
        return false;
    };
    match &node.kind {
        TermKind::Lt(_, _) | TermKind::Gt(_, _) => true,
        TermKind::And(args) | TermKind::Or(args) | TermKind::Add(args) | TermKind::Mul(args) => {
            args.iter().any(|&arg| contains_strict_inequality(arg, tm))
        }
        TermKind::Not(arg) | TermKind::Neg(arg) => contains_strict_inequality(*arg, tm),
        TermKind::Eq(lhs, rhs)
        | TermKind::Le(lhs, rhs)
        | TermKind::Ge(lhs, rhs)
        | TermKind::Sub(lhs, rhs)
        | TermKind::Div(lhs, rhs)
        | TermKind::Mod(lhs, rhs) => {
            contains_strict_inequality(*lhs, tm) || contains_strict_inequality(*rhs, tm)
        }
        _ => false,
    }
}
