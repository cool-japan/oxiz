//! The deterministic budget on the **embedded bit-blasted solver**.
//!
//! Split out of `theory_manager.rs` to keep that file under the workspace's
//! 2,000-line ceiling.  One constant and the two accessors that spend it; the
//! places that charge against it are `bv_bridge`'s three `BvSolver::check`
//! call sites and the mid-search exit in `TheoryManager::on_assignment`.

use super::TheoryManager;

/// Complete checks of the embedded bit-blasted solver one `check` may run
/// before it answers `Unknown` (`#P2b-38` strand (b), `#P2b-46`; third
/// deterministic currency).
///
/// # Why a third counter
///
/// [`ARRAY_REFINEMENT_RESOLVE_CONFLICTS`] bounds a refinement loop that
/// *searches*; [`ARRAY_REFINEMENT_LEMMA_BUDGET`] bounds one that only
/// *builds*.  Neither sees a loop that does **one** round whose re-solve is
/// enormous, and that is the shape the enumerated extensionality family
/// produces: `C(n,2) · |D|` bit-vector equality atoms asserted in a single
/// round, after which the outer search runs one complete embedded
/// `BvSolver::check` per bit-vector atom propagation.  Measured on twelve
/// pairwise-distinct arrays over `(Array (_ BitVec 3) (_ BitVec 1))`: 75,740
/// embedded checks and 22 s of wall clock, in **one** refinement round with 66
/// lemma instances and 1,444 conflicts — three orders of magnitude below both
/// ceilings above, so neither fires.  At twenty arrays the same script ran
/// 900.03 s under `/usr/bin/time` with no answer at all and no budget stopping
/// it; only an explicit `:timeout` did, which is precisely the
/// machine-dependence decision (9) exists to remove.
///
/// Counted in [`crate::solver::Statistics::bv_embedded_checks`], advanced by
/// `TheoryManager` once per embedded check and reset at the entry of every
/// `check`.  Exhaustion sets the theory manager's `resource_exhausted` flag,
/// so the verdict is `Unknown` and never a fabricated `sat`/`unsat`.
///
/// # Calibration
///
/// Measured on this tree (2026-09-19), release, peak embedded checks per
/// script:
///
/// * the 217-script `bench/` corpus: **207**
///   (`extended_theories/QF_ABV/02_bv_array_overwrite.smt2`); the next four are
///   170, 167, 155 and 79, and 209 of the 217 are under 10.
/// * the array-cardinality ladder over `(Array (_ BitVec 2) (_ BitVec 1))`,
///   which `round4_recheck_regressions`' `#[ignore]`d cost pin requires to
///   answer `sat` for every `n` up to 16: **36,281** at `n = 16`.
/// * the same family one index bit wider: 34,117 at `n = 11`, 76,860 at
///   `n = 13` (25.5 s).
/// * above the enumeration limit, where the Skolem cascade runs: 785 at index
///   width 4 / `n = 11` and 5,123 at `n = 15`.
///
/// A quarter of a million is 1,200x the `bench/` peak, 6.9x the largest script
/// any gate requires to answer, and 3.3x the most expensive script measured
/// that still decides.  That headroom is thinner than the two ceilings above
/// carry, and deliberately so: those two exist to be *unreachable*, while this
/// one exists to **fire** — a fifteen-line `(distinct a0 … a15)` over
/// `(Array (_ BitVec 3) (_ BitVec 1))` runs at about 2,300 embedded checks per
/// second and had, before this, no answer at all in 900 s.  It now answers
/// `Unknown` after a bounded, deterministic amount of work (about two minutes
/// on this machine, and the *same* count of checks on any machine).
///
/// Decision (10) is not closed by this and is not claimed to be: the budget
/// makes the runaway terminate, it does not make it fast.  See `TODO.md`
/// `#P2b-38` strand (b).
const BV_EMBEDDED_CHECK_CEILING: u64 = 250_000;

impl TheoryManager<'_> {
    /// Charge one embedded bit-blasted check to this `check`'s deterministic
    /// budget and report whether the budget is now spent.
    ///
    /// See [`BV_EMBEDDED_CHECK_CEILING`] for what this bounds and why the two
    /// refinement counters cannot see it.
    pub(super) fn charge_bv_embedded_check(&mut self) -> bool {
        self.statistics.bv_embedded_checks = self.statistics.bv_embedded_checks.saturating_add(1);
        self.statistics.bv_embedded_checks > BV_EMBEDDED_CHECK_CEILING
    }

    /// Whether the embedded-check budget is already spent, without charging
    /// for another one.
    pub(super) fn bv_embedded_budget_spent(&self) -> bool {
        self.statistics.bv_embedded_checks > BV_EMBEDDED_CHECK_CEILING
    }
}
