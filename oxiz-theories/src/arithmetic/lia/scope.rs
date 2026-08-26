//! Backtracking-scope and bound-inspection delegates for the LIA solver.
//!
//! [`LiaSolver`] owns its [`Simplex`](super::super::simplex::Simplex) as a
//! module-private field, so a solver *layered on top* of the LIA core cannot
//! open a scope of its own, nor read back the bounds the LP currently holds.
//! Branch-and-bound inside this module needs neither — it pushes and pops
//! internally and reads bounds through the tableau — but the nonlinear driver
//! in `arithmetic::nla` needs both: it explores a case split by asserting
//! lemmas valid only inside one case (so it must retract exactly those), and
//! it runs interval propagation over the monics from whatever box the LP has
//! established (so it must read that box).
//!
//! The scope methods are pure delegation, with no LIA-level bookkeeping of
//! their own. That is deliberate and worth stating, because the obvious
//! alternative is wrong: `branch_stack`, `active_cuts` and `pseudo_costs` are
//! *heuristic* state, not semantic state. Whether a pseudo-cost survives a pop
//! changes which variable gets branched on next and nothing else, whereas the
//! constraint set — the only thing soundness depends on — lives entirely in
//! the simplex trail and is restored exactly by [`Simplex::pop`].

use super::super::simplex::VarId;
use super::types::LiaSolver;
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;

impl LiaSolver {
    /// Open a backtracking scope.
    ///
    /// Every constraint, bound and cut asserted after this call is undone by
    /// the matching [`LiaSolver::pop`]. Scopes nest; [`LiaSolver::check`] and
    /// [`LiaSolver::check_balanced`] may push scopes of their own inside this
    /// one, and `check_balanced` is guaranteed to leave the depth unchanged.
    pub fn push(&mut self) {
        self.simplex.push();
    }

    /// Close the innermost backtracking scope opened by [`LiaSolver::push`],
    /// retracting everything asserted inside it.
    ///
    /// Popping more often than pushing is a no-op rather than an error, which
    /// matches the underlying simplex.
    pub fn pop(&mut self) {
        self.simplex.pop();
    }

    /// Number of [`LiaSolver::push`] calls not yet matched by a
    /// [`LiaSolver::pop`].
    ///
    /// Exists for scope-balance assertions in the layers above; forwards
    /// unchanged to the simplex's own notion of scope depth.
    #[must_use]
    pub(crate) fn scope_depth(&self) -> usize {
        self.simplex.scope_depth()
    }

    /// The current lower bound on `var` as an ordinary rational, together with
    /// the reason tag that justifies it, or `None` when `var` is unbounded
    /// below.
    ///
    /// A *strict* lower bound `x > r` is stored as the delta-rational `r + δ`;
    /// this reports `r`, which is the weaker non-strict consequence `x >= r`.
    /// Reporting the weaker bound is the sound direction for every consumer:
    /// anything derived from `x >= r` also follows from `x > r`.
    #[must_use]
    pub(crate) fn bound_lower(&self, var: VarId) -> Option<(Rational64, u32)> {
        self.simplex
            .get_lower(var)
            .map(|b| (b.value.real, b.reason))
    }

    /// The current upper bound on `var`; see [`LiaSolver::bound_lower`], of
    /// which this is the mirror image (a strict `x < r` reports the weaker
    /// `x <= r`).
    #[must_use]
    pub(crate) fn bound_upper(&self, var: VarId) -> Option<(Rational64, u32)> {
        self.simplex
            .get_upper(var)
            .map(|b| (b.value.real, b.reason))
    }

    /// Assert `var >= value` as a *bound*, but only when it is strictly tighter
    /// than the bound already in scope. Returns whether it was written.
    ///
    /// The guard is what makes this safe to call repeatedly and in any order.
    /// [`Simplex::set_lower`] *replaces* a bound rather than intersecting with
    /// it, so an unguarded write could relax a tighter bound already
    /// established — losing a refutation. Writing only strict improvements
    /// makes the sequence of writes monotone, which is exactly the intersection
    /// semantics a constraint would have given.
    ///
    /// Asserting a bound rather than the equivalent single-variable constraint
    /// matters because bounds are what [`LiaSolver::bound_lower`] can read back:
    /// `add_ge` on `x - 3` installs a slack row that constrains the LP
    /// identically but leaves `x` looking unbounded to every consumer that
    /// inspects bounds. The two are interchangeable for feasibility and not at
    /// all interchangeable for propagation.
    pub(crate) fn tighten_lower(&mut self, var: VarId, value: Rational64, reason: u32) -> bool {
        let improves = match self.simplex.get_lower(var) {
            Some(current) => value > current.value.real,
            None => true,
        };
        if improves {
            self.simplex.set_lower(var, value, reason);
        }
        improves
    }

    /// Assert `var <= value` as a bound when it is strictly tighter than the
    /// one in scope; see [`LiaSolver::tighten_lower`].
    pub(crate) fn tighten_upper(&mut self, var: VarId, value: Rational64, reason: u32) -> bool {
        let improves = match self.simplex.get_upper(var) {
            Some(current) => value < current.value.real,
            None => true,
        };
        if improves {
            self.simplex.set_upper(var, value, reason);
        }
        improves
    }

    /// Run the simplex's own tableau bound propagation, so that bounds implied
    /// by a *row* (rather than asserted directly on a variable) become visible
    /// to [`LiaSolver::bound_lower`] and [`LiaSolver::bound_upper`].
    ///
    /// Without this, `x + y <= 10 ∧ x >= 0 ∧ y >= 0` leaves both variables
    /// looking unbounded above even though the LP plainly bounds them, and any
    /// consumer that needs a finite box — interval propagation over products,
    /// a McCormick envelope — comes away with nothing.
    ///
    /// Every bound it writes is routed through the undo trail, so a
    /// [`LiaSolver::pop`] retracts it along with the constraints that implied
    /// it.
    pub(crate) fn propagate_lp_bounds(&mut self) {
        self.simplex.propagate_bounds();
        self.simplex.clear_propagated();
    }
}
