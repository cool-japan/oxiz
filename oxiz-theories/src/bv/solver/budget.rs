//! Resource budgets for the embedded bit-blasting SAT solver (finding U-Z12).
//!
//! # Why this exists
//!
//! [`BvSolver::check`] runs a *full* `oxiz_sat::Solver::solve()` on its own
//! embedded solver, once per asserted bit-vector atom, from inside the enclosing
//! CDCL(T) search's `on_assignment` callback. Every budget poll in the
//! CDCL(T) layer — `TheoryManager::timed_out`, the `max_conflicts` comparison
//! against `Statistics::conflicts`, `check_core`'s round-boundary deadline —
//! sits *outside* that call. So a single bit-blasted `solve()` could run
//! unbounded, and it did: a 64x64 multiplier verification condition under
//! `(set-option :timeout 100)` answered after 17 s, 170x its budget, because
//! 99.95 % of the run was one uninterruptible embedded `solve()`.
//!
//! The fix is this module: the owning solver hands `BvSolver` a conflict
//! allowance and a wall-clock deadline with [`BvSolver::set_budget`], and every
//! embedded `solve()` is armed from them.
//!
//! # The allowance is a total, not a per-probe grant
//!
//! Two things make a naive `sat.set_max_conflicts(Some(n))` wrong here:
//!
//! * `oxiz_sat::SolverStats::conflicts` is *cumulative* across `solve()` calls,
//!   so a fixed ceiling of `n` would only be right for the very first probe;
//! * `BvSolver::reset` calls `sat.reset()`, which zeroes those statistics, and
//!   `oxiz-solver`'s `Solver::rebase_theory_state` calls `BvSolver::reset` once
//!   per check **and again on every repair round** — so a budget expressed
//!   purely in terms of the embedded statistics would silently re-arm in full
//!   several times inside one `(check-sat)`.
//!
//! [`BvSolver::conflicts_spent`] is therefore an accumulator owned by
//! `BvSolver`, deliberately **not** cleared by `reset()`, and each probe's
//! ceiling is computed relative to the embedded solver's current conflict count
//! ([`BvSolver::apply_budget_to_embedded`]). The allowance is re-armed only by
//! [`BvSolver::set_budget`], which the owning solver calls once per
//! `check_core` — so `(set-option :max-conflicts N)` bounds the *total*
//! bit-blasting work of a check-sat, however many probes and repair rounds it
//! takes.

use super::BvSolver;

/// Denominator of the allowance reserved for the `Unsat` re-verification in
/// [`BvSolver::check`]. One quarter of whatever is left when a probe starts is
/// held back, so a first solve that ends `Unsat` still has budget to be
/// re-verified — see [`BvSolver::first_solve_allowance`].
const REVERIFY_RESERVE_DIVISOR: u64 = 4;

impl BvSolver {
    /// Bound every embedded `solve()` this solver runs.
    ///
    /// * `max_conflicts` — a **total** conflict allowance across all probes and
    ///   all repair rounds until the next `set_budget`, not a per-probe grant.
    ///   `None` means unbounded. Calling this re-arms the allowance: the
    ///   accumulator behind [`Self::conflicts_spent`] is reset to zero, so the
    ///   caller decides the budget's period (`oxiz-solver` arms it once per
    ///   `check_core`, making the period one `(check-sat)`).
    /// * `deadline` — a wall-clock instant shared by every embedded solve, so a
    ///   timeout stays a bound on the whole check rather than becoming a
    ///   per-probe bound. `None` means no deadline. On a target with no clock
    ///   (`wasm32-unknown-unknown`, or a `--no-default-features` build) the
    ///   frozen `oxiz_time::Instant` makes this a documented no-op and
    ///   `max_conflicts` is the only working budget; see the `oxiz_time` crate
    ///   docs.
    ///
    /// A budget can only turn a verdict into `TheoryResult::Unknown`, never
    /// into a different definite answer: an exhausted embedded solve reports
    /// `SolverResult::Unknown`, which `check()` maps to `TheoryResult::Unknown`
    /// and the theory manager turns into an overall `unknown`.
    pub fn set_budget(&mut self, max_conflicts: Option<u64>, deadline: Option<oxiz_time::Instant>) {
        self.budget_max_conflicts = max_conflicts;
        self.budget_deadline = deadline;
        self.conflicts_spent = 0;
    }

    /// Embedded SAT conflicts charged against the current allowance so far.
    ///
    /// Survives [`crate::theory::Theory::reset`] (which zeroes the embedded
    /// solver's own statistics), and is cleared only by [`Self::set_budget`].
    /// Exposed so a caller — or a test — can see how much of the third budget
    /// `(set-option :max-conflicts N)` installs has actually been spent; the
    /// other two (outer Boolean conflicts, theory conflicts) are visible
    /// through `oxiz-solver`'s `Solver::stats()` and `Solver::get_statistics()`
    /// respectively.
    #[must_use]
    pub fn conflicts_spent(&self) -> u64 {
        self.conflicts_spent
    }

    /// The conflict allowance still unspent (`None` = unbounded).
    pub(crate) fn remaining_conflict_budget(&self) -> Option<u64> {
        self.budget_max_conflicts
            .map(|total| total.saturating_sub(self.conflicts_spent))
    }

    /// The allowance the *first* solve of a probe may spend, keeping
    /// `1 / REVERIFY_RESERVE_DIVISOR` of the remainder back for the `Unsat`
    /// re-verification.
    ///
    /// Rounded so the first solve gets the larger share: with `r` left it may
    /// spend `r - r / 4`, which is `r` itself for `r < 4`. A tiny allowance
    /// therefore goes entirely to the first solve and the re-verify gets none,
    /// which is exactly the case [`BvSolver::check`] reports as `Unknown` —
    /// an `Unsat` that could not be re-verified is not a verdict this solver
    /// trusts (see `check`'s own comment on why the re-verify exists).
    pub(crate) fn first_solve_allowance(&self) -> Option<u64> {
        self.remaining_conflict_budget()
            .map(|remaining| remaining - remaining / REVERIFY_RESERVE_DIVISOR)
    }

    /// Arm the embedded solver for one `solve()`: install the shared deadline
    /// and a conflict ceiling `allowance` above the conflicts it has already
    /// counted. `None` clears the ceiling.
    pub(crate) fn apply_budget_to_embedded(&mut self, allowance: Option<u64>) {
        self.sat.set_deadline(self.budget_deadline);
        // Computed into a local first: `stats()` borrows `self.sat` immutably
        // and `set_max_conflicts` borrows it mutably, so the ceiling must not be
        // produced inside the argument expression.
        let ceiling = allowance.map(|a| self.sat.stats().conflicts.saturating_add(a));
        self.sat.set_max_conflicts(ceiling);
    }

    /// Charge the conflicts the embedded solver accumulated since
    /// `conflicts_before` (read from `sat.stats().conflicts` immediately before
    /// the `solve()`) against the total allowance.
    pub(crate) fn charge_embedded_conflicts(&mut self, conflicts_before: u64) {
        let spent = self.sat.stats().conflicts.saturating_sub(conflicts_before);
        self.conflicts_spent = self.conflicts_spent.saturating_add(spent);
    }
}
