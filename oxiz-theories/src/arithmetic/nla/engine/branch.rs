// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Case splitting, and the lemma families a sign context unlocks.
//!
//! When neither interval propagation nor a cut can make progress on a violated
//! monic, the engine splits the problem into cases and solves each. This file
//! holds that half of [`NlaEngine`](super::NlaEngine): choosing what to split
//! on, enumerating the cases, combining their answers, and emitting the lemmas
//! that only become valid once a factor's sign is pinned down.
//!
//! # Exhaustiveness is the whole soundness argument
//!
//! A split may only be used to refute if its cases cover every integer point,
//! and both forms here do:
//!
//! * `x <= -1 ∨ x = 0 ∨ x >= 1` — over `Z` nothing lies strictly between `-1`
//!   and `1` except `0`.
//! * `v <= k ∨ v >= k+1` for integral `k` — the two halves of `Z` at `k`.
//!
//! The combination rule in [`NlaEngine::split`](super::NlaEngine::split) is the
//! matching half: `Sat` from any case wins, `Refuted` demands that *every* case
//! was refuted, and a single `Unknown` case makes the whole node `Unknown`.
//! Treating an unexplored case as refuted is the one mistake here that would
//! produce a wrong `unsat`, so the discipline is a `bool` threaded through the
//! loop rather than something inferred at the end.

use super::super::super::simplex::{LinExpr, VarId};
use super::super::checked_add_r64;
use super::super::lemmas::{self, Lemma, LemmaScope, Sign};
use super::super::linearize::{LinAtom, LinAtomKind, Monic};
use super::{NLA_REASON, NlaEngine, NodeOutcome};
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{One, Signed, Zero};

impl NlaEngine<'_> {
    // --- (d3) branching -----------------------------------------------------

    /// Split on a variable of a violated monic and recurse into every case.
    ///
    /// Consumes one node from the budget. Prefers a sign split on a factor
    /// whose sign the current bounds leave open, because the sign context is
    /// what unlocks the basics / order / monotonicity lemma families; failing
    /// that, splits a factor's value at its model point.
    pub(super) fn branch(
        &mut self,
        violated: &[usize],
        model: &FxHashMap<VarId, Rational64>,
        depth: usize,
    ) -> NodeOutcome {
        self.nodes_used += 1;

        if let Some(x) = self.sign_unfixed_factor(violated) {
            return self.split(&self.sign_cases(x), depth, Some(x));
        }
        let Some((v, k)) = self.widest_factor(violated, model) else {
            return NodeOutcome::Unknown;
        };
        self.split(&self.value_cases(v, k), depth, None)
    }

    /// The three cases of a sign split on `x`, each a conjunction of atoms.
    ///
    /// `x <= -1 ∨ x = 0 ∨ x >= 1` covers `Z`: strictly between `-1` and `1`
    /// the only integer is `0`.
    fn sign_cases(&self, x: VarId) -> Vec<BranchCase> {
        let neg_one = -Rational64::one();
        vec![
            BranchCase {
                atoms: vec![bound_atom(x, LinAtomKind::Le, neg_one)],
                sign: Some(Sign::Neg),
                var: x,
            },
            BranchCase {
                atoms: vec![bound_atom(x, LinAtomKind::Eq, Rational64::zero())],
                sign: None,
                var: x,
            },
            BranchCase {
                atoms: vec![bound_atom(x, LinAtomKind::Ge, Rational64::one())],
                sign: Some(Sign::Pos),
                var: x,
            },
        ]
    }

    /// The two cases of a value split on `v` at integral `k`: `v <= k` and
    /// `v >= k + 1`, which covers `Z`.
    fn value_cases(&self, v: VarId, k: Rational64) -> Vec<BranchCase> {
        vec![
            BranchCase {
                atoms: vec![bound_atom(v, LinAtomKind::Le, k)],
                sign: None,
                var: v,
            },
            BranchCase {
                atoms: vec![bound_atom(v, LinAtomKind::Ge, k + Rational64::one())],
                sign: None,
                var: v,
            },
        ]
    }

    /// Explore every case under its own scope and combine the answers.
    ///
    /// `Sat` in any case wins outright. `Refuted` needs every case refuted —
    /// `saw_unknown` makes sure a case the budget cut short never passes for
    /// one that was explored and closed.
    fn split(
        &mut self,
        cases: &[BranchCase],
        depth: usize,
        splitting: Option<VarId>,
    ) -> NodeOutcome {
        let entry_depth = self.lia.scope_depth();
        let mut saw_unknown = false;

        for case in cases {
            self.lia.push();

            let mut representable = true;
            for atom in &case.atoms {
                if self.assert_atom(atom).is_none() {
                    representable = false;
                }
            }

            let outcome = if representable {
                match (splitting, case.sign) {
                    (Some(x), Some(sign)) => self.emit_branch_lemmas(x, sign),
                    // The `x = 0` case of a sign split. Without a lemma the LP
                    // knows only that `x` is zero and nothing about the
                    // products `x` occurs in, so the case is unrefutable and
                    // the whole split degrades to `Unknown` — the annihilation
                    // lemma is what makes a sign split able to close.
                    (Some(x), None) => self.emit_zero_lemmas(x),
                    _ => {}
                }
                self.node(depth + 1)
            } else {
                // A case whose own atoms could not be asserted has not been
                // explored at all. Its scope holds an arbitrary subset of them,
                // so nothing may be concluded from it.
                NodeOutcome::Unknown
            };

            self.lia.pop();

            match outcome {
                NodeOutcome::Sat(model) => {
                    debug_assert_eq!(
                        self.lia.scope_depth(),
                        entry_depth,
                        "a case must restore the scope depth it was entered at"
                    );
                    return NodeOutcome::Sat(model);
                }
                NodeOutcome::Refuted => {}
                NodeOutcome::Unknown => saw_unknown = true,
            }
        }

        debug_assert_eq!(
            self.lia.scope_depth(),
            entry_depth,
            "every case must restore the scope depth it was entered at"
        );
        if saw_unknown {
            NodeOutcome::Unknown
        } else {
            NodeOutcome::Refuted
        }
    }

    /// Emit the lemma families that the sign context on `x` unlocks.
    ///
    /// All are [`LemmaScope::BranchLocal`]: they are consequences of `x`'s sign,
    /// which holds only inside the case just pushed. They are asserted after
    /// that push and retracted with it.
    fn emit_branch_lemmas(&mut self, x: VarId, x_sign: Sign) {
        // Collect first: the borrow of `self.lin` has to end before the
        // assertions, which take `&mut self`.
        let mut pending: Vec<Lemma> = Vec::new();
        for monic in &self.lin.monics {
            if !monic.factors.iter().any(|(f, _)| *f == x) {
                continue;
            }
            let product = monic.product;

            // Sign of the product, when every factor's sign is known. `x`'s is
            // the case premise; the others must come from the bounds in scope.
            //
            // The bilinear case goes through `sign`, which says what it means —
            // the two factor signs give the product's. Higher degrees have no
            // such pair, so the conclusion `self.product_sign` derived is
            // asserted directly rather than re-encoded as a synthetic pair.
            let sign_lemma = match monic.factors.as_slice() {
                [(a, 1), (b, 1)] => {
                    let other = if *a == x { *b } else { *a };
                    match self.fixed_sign(other) {
                        Some(other_sign) if other != x => {
                            lemmas::sign(product, x_sign, other_sign, &[NLA_REASON])
                        }
                        _ => None,
                    }
                }
                _ => self
                    .product_sign(monic, x, x_sign)
                    .and_then(|result| lemmas::product_sign(product, result, &[NLA_REASON])),
            };
            if let Some(lemma) = sign_lemma {
                pending.push(lemma);
            }

            if let [(a, 1), (b, 1)] = monic.factors.as_slice() {
                let other = if *a == x { *b } else { *a };

                // |x| >= 1 cannot shrink a product: |v| >= |cofactor|. Needs
                // the cofactor's sign as well, to pick which of the four
                // linear forms of that fact holds.
                if other != x
                    && let Some(other_sign) = self.fixed_sign(other)
                    && let Some(lemma) =
                        lemmas::proportion(product, other, x_sign, other_sign, &[NLA_REASON])
                {
                    pending.push(lemma);
                }

                // A factor pinned to exactly 1 makes the product its cofactor.
                // The sign case establishes `x >= 1`; a bound of `x <= 1` on
                // top of it pins the value, and the branch that asserted the
                // upper bound is in scope, so the premise really does hold
                // here.
                if other != x
                    && x_sign == Sign::Pos
                    && self.is_fixed_to_one(x)
                    && let Some(lemma) = lemmas::neutral(product, other, &[NLA_REASON])
                {
                    pending.push(lemma);
                }
            }
        }

        for lemma in &pending {
            debug_assert_eq!(
                lemma.scope,
                LemmaScope::BranchLocal,
                "a lemma emitted under a branch premise must be branch-local"
            );
            let _ = self.assert_lemma(lemma);
        }
    }

    /// Emit `v = 0` for every monic that `x` is a factor of, under the premise
    /// `x = 0` established by the case just pushed.
    ///
    /// A zero factor annihilates whatever it is multiplied by, at any degree
    /// and whatever the other factors are — the one product lemma that needs no
    /// bound on anything else. Branch-local all the same: the premise is the
    /// case's own `x = 0`.
    fn emit_zero_lemmas(&mut self, x: VarId) {
        let mut pending: Vec<Lemma> = Vec::new();
        for monic in &self.lin.monics {
            if monic.factors.iter().any(|(f, _)| *f == x)
                && let Some(lemma) = lemmas::zero(monic.product, &[NLA_REASON])
            {
                pending.push(lemma);
            }
        }
        for lemma in &pending {
            debug_assert_eq!(
                lemma.scope,
                LemmaScope::BranchLocal,
                "a lemma emitted under a branch premise must be branch-local"
            );
            let _ = self.assert_lemma(lemma);
        }
    }

    /// The sign of `monic`'s product, given that `x` has sign `x_sign` and the
    /// bounds fix every other factor's sign. `None` when some factor's sign is
    /// open — an unsigned factor could be zero, and then the product is zero,
    /// so no strict sign lemma follows.
    ///
    /// A factor raised to an even power is positive whatever its own sign, but
    /// only when it is also known non-zero; the sign premise supplies exactly
    /// that, so an even power of a sign-fixed factor contributes `Pos`.
    fn product_sign(&self, monic: &Monic, x: VarId, x_sign: Sign) -> Option<Sign> {
        let mut negatives = 0u32;
        for (factor, exponent) in &monic.factors {
            let sign = if *factor == x {
                x_sign
            } else {
                self.fixed_sign(*factor)?
            };
            if sign == Sign::Neg && !exponent.is_multiple_of(2) {
                negatives += 1;
            }
        }
        Some(if negatives.is_multiple_of(2) {
            Sign::Pos
        } else {
            Sign::Neg
        })
    }

    /// Whether the bounds in scope pin `v` to exactly `1`.
    fn is_fixed_to_one(&self, v: VarId) -> bool {
        let Some(solver_var) = self.solver_var(v) else {
            return false;
        };
        let one = Rational64::one();
        matches!(
            (
                self.lia.bound_lower(solver_var),
                self.lia.bound_upper(solver_var),
            ),
            (Some((lo, _)), Some((hi, _))) if lo == one && hi == one
        )
    }

    /// The sign the current bounds force on `v`, or `None` when they permit
    /// zero.
    fn fixed_sign(&self, v: VarId) -> Option<Sign> {
        let solver_var = self.solver_var(v)?;
        if let Some((lo, _)) = self.lia.bound_lower(solver_var)
            && lo.is_positive()
        {
            return Some(Sign::Pos);
        }
        if let Some((hi, _)) = self.lia.bound_upper(solver_var)
            && hi.is_negative()
        {
            return Some(Sign::Neg);
        }
        None
    }

    /// A factor of a violated monic whose sign the current bounds leave open.
    ///
    /// Scanned in monic order, then factor order, so the choice is a function
    /// of the problem rather than of hash-map iteration order.
    ///
    /// A factor that is itself a product variable is skipped: splitting on it
    /// constrains a *derived* quantity, which tells the search nothing new
    /// about the variables that actually determine it, and the monic defining
    /// it would still be violated in both cases. Splitting on a genuine input
    /// variable is what makes progress.
    fn sign_unfixed_factor(&self, violated: &[usize]) -> Option<VarId> {
        for &index in violated {
            let monic = self.lin.monics.get(index)?;
            for (factor, _) in &monic.factors {
                if self.fixed_sign(*factor).is_none() && !self.is_product_var(*factor) {
                    return Some(*factor);
                }
            }
        }
        None
    }

    /// Whether `v` is a variable the linearisation introduced to stand for a
    /// product, rather than one the input mentions.
    fn is_product_var(&self, v: VarId) -> bool {
        self.solver_var(v)
            .is_some_and(|solver_var| self.monic_of_product.contains_key(&solver_var))
    }

    /// The factor of a violated monic with the widest current interval, and its
    /// model value. An unbounded side counts as maximally wide, since splitting
    /// there is what eventually makes the box finite.
    ///
    /// Ties break toward the lowest variable id, again for determinism.
    fn widest_factor(
        &self,
        violated: &[usize],
        model: &FxHashMap<VarId, Rational64>,
    ) -> Option<(VarId, Rational64)> {
        let mut best: Option<(VarId, Rational64, Width)> = None;
        for &index in violated {
            let Some(monic) = self.lin.monics.get(index) else {
                continue;
            };
            for (factor, _) in &monic.factors {
                // See `sign_unfixed_factor`: a derived product variable is a
                // poor split, for the same reason there.
                if self.is_product_var(*factor) {
                    continue;
                }
                let Some(value) = self.model_value(*factor, model) else {
                    continue;
                };
                if !value.is_integer() {
                    continue;
                }
                let width = self.interval_width(*factor);
                let better = match &best {
                    Some((_, _, current)) => width > *current,
                    None => true,
                };
                if better {
                    best = Some((*factor, value, width));
                }
            }
        }
        best.map(|(v, value, _)| (v, value))
    }

    /// How wide `v`'s current interval is, for branch selection only.
    fn interval_width(&self, v: VarId) -> Width {
        let Some(solver_var) = self.solver_var(v) else {
            return Width::Infinite;
        };
        match (
            self.lia.bound_lower(solver_var),
            self.lia.bound_upper(solver_var),
        ) {
            (Some((lo, _)), Some((hi, _))) => Width::Finite(hi - lo),
            _ => Width::Infinite,
        }
    }
}

/// One case of a split: the atoms that define it, plus the sign premise it
/// establishes (when it establishes one).
struct BranchCase {
    /// Atoms asserted on entering the case, over linearisation variables.
    atoms: Vec<LinAtom>,
    /// The sign this case forces on `var`, if any. `None` for the `= 0` case
    /// and for value splits, neither of which fixes a sign.
    sign: Option<Sign>,
    /// The variable the case constrains. Retained so a case carries its whole
    /// meaning; used by the debug assertions.
    #[allow(dead_code)]
    var: VarId,
}

/// Interval width for branch selection, with an explicit infinity so an
/// unbounded variable sorts above every finite one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Width {
    /// A finite width.
    Finite(Rational64),
    /// Unbounded on at least one side.
    Infinite,
}

/// `v ⋈ value`, as an atom over the expression `v - value`.
fn bound_atom(v: VarId, kind: LinAtomKind, value: Rational64) -> LinAtom {
    let mut expr = LinExpr::var(v);
    expr.add_constant(-value);
    LinAtom { expr, kind }
}

/// `expr` with `delta` added to its constant, or `None` on overflow.
pub(super) fn shift_constant(expr: &LinExpr, delta: Rational64) -> Option<LinExpr> {
    let mut out = expr.clone();
    out.constant = checked_add_r64(out.constant, delta)?;
    Some(out)
}
