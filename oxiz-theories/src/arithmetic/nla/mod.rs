// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Nonlinear integer arithmetic over a linear relaxation (NIA-over-LP).
//!
//! [`check_assertions`] is the entry point: hand it a conjunction of ground
//! arithmetic assertions and it answers [`NlaVerdict::Unsat`],
//! [`NlaVerdict::Sat`] with a witness, or [`NlaVerdict::Unknown`].
//!
//! # The shape of the engine
//!
//! 1. `linearize` turns the conjunction into a *linear* problem plus a set of
//!    `Monic`s. Every product of two or more non-constant
//!    factors is replaced by a fresh variable, and the monic records
//!    `product = f0^e0 * f1^e1 * ...`. What the LP core then solves is a
//!    *relaxation*: it knows the linear structure and nothing at all about the
//!    multiplication.
//! 2. If the relaxation is infeasible, the original problem is unsat and we are
//!    done — dropping the nonlinear semantics only ever *weakens*, so an
//!    infeasible relaxation is a sound refutation.
//! 3. If it is feasible, the model may violate a monic (`v != x * y` under the
//!    returned assignment). That is where `monomial_bounds` and `lemmas`
//!    come in: interval propagation over the monics either derives a conflict
//!    directly, or the lemma constructors emit linear consequences of the
//!    multiplication that the current model falsifies. Those are added to the
//!    relaxation and the loop repeats — the *lemma cascade*, driven by the
//!    crate-private `engine::NlaEngine`.
//! 4. When neither propagation nor a cut makes progress, the engine splits a
//!    case — on a factor's sign where that is still open, otherwise on a
//!    factor's value — and recurses into every case.
//!
//! # Soundness contract
//!
//! The two verdicts do not carry the same weight, and the difference is
//! deliberate.
//!
//! **`Unsat` is proof-backed.** It is only ever produced by an infeasibility
//! closure inside the LIA solver, over a constraint set every member of which
//! is a consequence over `Z` of the input:
//!
//! * the *input atoms*, translated by `linearize`;
//! * the *monic definitions*, which are definitional and cannot change
//!   satisfiability;
//! * *derived bounds*, each computed by checked interval arithmetic over bounds
//!   already in scope;
//! * *lemmas*, each a linear consequence of a product under premises the engine
//!   has established — global ones (the square tangent `(x-a)² ≥ 0`) at any
//!   scope, branch-local ones only inside the case that establishes them;
//! * *branch atoms* at scopes at or above the node, whose cases are exhaustive
//!   over `Z` (`x ≤ -1 ∨ x = 0 ∨ x ≥ 1`, and `v ≤ k ∨ v ≥ k+1` for integral
//!   `k`).
//!
//! A case split refutes only when *every* case refutes; a case the budget cut
//! short makes the whole node `Unknown`.
//!
//! **`Sat` is advisory.** The witness is assembled from an LP model and is
//! meant to be re-verified by the caller with
//! [`crate::nl_eval::holds_under`] against the *original* assertions.
//! [`check_assertions`] performs that verification itself before returning, so
//! a `Sat` it hands back has already been checked in exact `BigInt` arithmetic
//! — but the contract stays advisory so that a caller which rewrote the goal on
//! the way in still re-checks against what it actually needs to answer about.
//!
//! **Every degradation goes toward `Unknown`.** A coefficient that cannot be
//! represented, a budget that runs out, a resource limit inside the LIA solver,
//! a conjunct outside the linearisation grammar — each of these loses precision
//! or completeness and none of them can produce a wrong verdict. In particular
//! `Linearization::incomplete` records
//! that a conjunct was dropped: the resulting problem is *weaker*, so `Unsat`
//! still refutes the input while `Sat` is suppressed to `Unknown`.
//!
//! # Mapping onto Z3
//!
//! This is Z3's `nlsat`/`nla_solver` split seen from the LP side. The lemma
//! families here correspond to `nla_basics`, `nla_order`, `nla_monotone` and
//! `nla_tangents` in `math/lp/`; the interval layer to `nla_intervals`; the
//! linearisation to `lar_solver`'s term/monic registration. The case split is
//! the integer specialisation of what `nlsat` does with sign conditions over
//! cells, done here on `Z` where the split is finite and exact rather than over
//! `R` where it needs cell decomposition.
//!
//! # Deferred, and honestly so
//!
//! These families are *not* implemented, and their absence costs completeness
//! (more `Unknown`), never soundness:
//!
//! * **Gröbner bases.** Z3 runs a bounded Gröbner/Horner pass over the monics
//!   to derive polynomial consequences that no interval or tangent argument
//!   reaches. Nothing here computes an S-polynomial.
//! * **Horner form / cross-nested intervals.** Interval evaluation is done
//!   factor-by-factor, so a correlated expression such as `x*y - x*z` is
//!   bounded as if the two `x` occurrences were independent.
//! * **Divisions and `mod`.** `linearize` drops any conjunct containing them
//!   and sets `incomplete`.
//! * **Bounded `nlsat` fallback.** Z3 hands a hard sub-goal to the
//!   cell-decomposition core; this engine answers `Unknown` instead. The
//!   crate's [`crate::nlsat`] module is that core, but wiring the
//!   handoff is a separate step.
//! * **`patch_monomials`.** Z3 repairs a model that violates one monic by
//!   moving a single factor, often turning a near-miss into a `Sat` without any
//!   search. The engine here re-solves instead.
//! * **Higher-degree backward propagation with a cofactor.** Inverting `x^e`
//!   is done exactly (see the crate-private `int_root`) only for a monic
//!   that is a single
//!   power; `x^2 * y` does not propagate backward into `x`.
//!
//! # Arithmetic never wraps
//!
//! Every multiplication, addition and division on coefficients goes through the
//! checked helpers re-exported by this module (mostly `arithmetic/simplex`'s).
//! `None` means "cannot represent"; the caller then declines to derive anything
//! rather than deriving something wrong. Monic consistency — the one check that
//! could turn an overflow into a wrong `Sat` — is done in [`num_bigint::BigInt`]
//! instead, where nothing can wrap.

pub(crate) mod engine;
pub(crate) mod int_root;
pub(crate) mod lemmas;
pub(crate) mod linearize;
pub(crate) mod monomial_bounds;

use super::simplex::checked_ratio_i128;
pub(crate) use super::simplex::{checked_add_r64, checked_mul_r64};
use crate::config::{LiaConfig, SimplexConfig};
use crate::nl_eval::{Interpretation, holds_under};
#[allow(unused_imports)]
use crate::prelude::*;
use engine::{NlaEngine, NodeOutcome};
use num_bigint::BigInt;
use num_rational::Rational64;
use oxiz_core::ast::{TermId, TermManager};

// TODO(reconcile): drop these two copies once `arithmetic/simplex` exposes
// them.
//
// The concurrent hoist that made `checked_add_r64`, `checked_mul_r64` and
// `checked_ratio_i128` `pub(crate)` (imported above) left `checked_neg_r64`
// and `checked_recip_r64` private, so those two are still duplicated here,
// byte-identical to their originals. When they follow, delete both bodies and
// add them to the `pub(crate) use` line above -- a pure deletion, no
// behavioural delta to audit. `checked_pow_r64` below has no counterpart in
// `simplex` and stays regardless.

/// Checked rational negation. Only the `i64::MIN` numerator fails.
pub(crate) fn checked_neg_r64(a: Rational64) -> Option<Rational64> {
    let n = (*a.numer() as i128).checked_neg()?;
    if !(i64::MIN as i128..=i64::MAX as i128).contains(&n) {
        return None;
    }
    // `new_raw`: negating the numerator changes neither the gcd nor the sign
    // of the denominator, so canonical form is preserved.
    Some(Rational64::new_raw(n as i64, *a.denom()))
}

/// Checked rational reciprocal. `None` when `a` is zero or the result
/// overflows.
pub(crate) fn checked_recip_r64(a: Rational64) -> Option<Rational64> {
    if a.numer() == &0 {
        return None;
    }
    checked_ratio_i128(*a.denom() as i128, *a.numer() as i128)
}

/// Checked exponentiation, `a^e`, by repeated checked multiplication.
/// `a^0` is `1`. `None` on overflow at any intermediate step.
pub(crate) fn checked_pow_r64(a: Rational64, e: u32) -> Option<Rational64> {
    let mut acc = Rational64::new_raw(1, 1);
    for _ in 0..e {
        acc = checked_mul_r64(acc, a)?;
    }
    Some(acc)
}

/// Budgets for the nonlinear search.
///
/// Every field bounds work, and exhausting any of them degrades the answer to
/// [`NlaVerdict::Unknown`]. None of them affects which answers are *possible* —
/// a larger budget can only turn `Unknown` into a verdict, never one verdict
/// into the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NlaConfig {
    /// Iterations of the propagate / cut / solve loop at a single node.
    pub max_rounds: usize,
    /// Case splits across the whole search.
    pub max_nodes: usize,
    /// Depth of nested case splits.
    pub max_depth: usize,
    /// Tangent and McCormick lemmas emitted across the whole search.
    pub max_tangent_cuts: usize,
    /// Pivot budget handed to each LP solve, forwarded to
    /// [`SimplexConfig::max_pivots`].
    pub max_pivots: usize,
    /// Branch-and-bound depth budget inside a single integer feasibility check,
    /// forwarded to [`LiaConfig::max_depth`].
    ///
    /// Distinct from [`NlaConfig::max_depth`], which bounds *nonlinear* case
    /// splits: one node of the nonlinear search runs a whole branch-and-bound
    /// underneath itself.
    pub max_lia_depth: usize,
}

impl Default for NlaConfig {
    fn default() -> Self {
        Self {
            max_rounds: 16,
            max_nodes: 512,
            max_depth: 48,
            max_tangent_cuts: 32,
            max_pivots: 10_000,
            max_lia_depth: 200,
        }
    }
}

impl NlaConfig {
    /// The LIA configuration the engine's solver is built with.
    ///
    /// Root Gomory cuts stay on: [`LiaSolver::check_balanced`] runs them at the
    /// entry scope, where they are valid inequalities of the constraint set the
    /// caller established, and the caller's own `pop` retracts them.
    fn lia_config(&self) -> LiaConfig {
        LiaConfig {
            max_depth: self.max_lia_depth,
            ..LiaConfig::default()
        }
    }

    /// The simplex configuration the engine's solver is built with.
    fn simplex_config(&self) -> SimplexConfig {
        SimplexConfig {
            max_pivots: self.max_pivots,
            ..SimplexConfig::default()
        }
    }
}

/// What the engine could establish about a goal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NlaVerdict {
    /// Satisfiable, with a witness that has been verified against the input by
    /// [`crate::nl_eval::holds_under`].
    Sat(Interpretation),
    /// Unsatisfiable. Backed by an LP infeasibility closure over valid
    /// consequences and exhaustive case splits; see the module docs.
    Unsat,
    /// Neither could be established. Covers every budget, every overflow, and
    /// every conjunct the linearisation could not translate.
    Unknown,
}

/// Decide `assertions`, read as a conjunction.
///
/// See the module documentation for the soundness contract. In short: `Unsat`
/// is a proof, `Sat` carries a witness this function has already re-verified
/// against `assertions`, and everything else is `Unknown`.
#[must_use]
pub fn check_assertions(
    assertions: &[TermId],
    manager: &TermManager,
    config: &NlaConfig,
) -> NlaVerdict {
    let Some(lin) = linearize::linearize(assertions, manager) else {
        // Nothing arithmetic survived translation: this engine has no business
        // with the goal.
        return NlaVerdict::Unknown;
    };
    let Some(mut engine) = NlaEngine::new(&lin, config) else {
        return NlaVerdict::Unknown;
    };

    match engine.solve() {
        // A refutation of the relaxation refutes the input, whether or not a
        // conjunct was dropped: dropping only weakens, and an unsatisfiable
        // weakening has an unsatisfiable original.
        NodeOutcome::Refuted => NlaVerdict::Unsat,
        NodeOutcome::Unknown => NlaVerdict::Unknown,
        NodeOutcome::Sat(model) => {
            // A `sat` derived from a problem that dropped a conjunct says
            // nothing about the input.
            if lin.incomplete {
                return NlaVerdict::Unknown;
            }
            let Some(interp) = witness(&lin, &model, &engine) else {
                return NlaVerdict::Unknown;
            };
            // Re-verify in exact arithmetic against the untouched input. The
            // engine's own consistency check already ran in `BigInt`, so this
            // should never fail — but "should never" is not a soundness
            // argument, and the check is cheap next to the search.
            if holds_under(assertions, manager, &interp) {
                NlaVerdict::Sat(interp)
            } else {
                NlaVerdict::Unknown
            }
        }
    }
}

/// Turn an LP model into an interpretation, pinning every term the
/// linearisation named a variable for.
///
/// `None` when a value is missing or non-integral, which would make the witness
/// incomplete and so unusable.
fn witness(
    lin: &linearize::Linearization,
    model: &FxHashMap<super::simplex::VarId, Rational64>,
    engine: &NlaEngine<'_>,
) -> Option<Interpretation> {
    let mut interp = Interpretation::empty();
    for (&var, &term) in &lin.term_of_var {
        let solver_var = engine.solver_var(var)?;
        let value = model.get(&solver_var)?;
        if !value.is_integer() {
            return None;
        }
        interp.pin_int(term, BigInt::from(value.to_integer()));
    }
    Some(interp)
}
