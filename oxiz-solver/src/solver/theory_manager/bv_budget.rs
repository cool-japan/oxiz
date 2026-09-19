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
///
/// # Widening this constant moves a test
///
/// The two scripts that actually reach the ceiling cost minutes in release and
/// are therefore `#[ignore]`d, release-calibrated cost pins: nothing in the
/// default `cargo nextest` gate could observe a widened ceiling, so a later
/// pass could raise it to make a slow script "answer" and no gate would
/// notice.  [`bv_embedded_check_ceiling`] exists so one can:
/// `the_embedded_check_ceiling_is_the_calibrated_value` reads it and asserts
/// the exact value.  Changing the constant is allowed; changing it silently is
/// not, and the test is where the new calibration has to be argued.
const BV_EMBEDDED_CHECK_CEILING: u64 = 250_000;

/// The live value of [`BV_EMBEDDED_CHECK_CEILING`].
///
/// A read of the constant the solver actually spends, not a copy of its text:
/// this is what lets a default-profile test assert the calibrated value
/// without paying the minutes the two cost pins pay.
pub(crate) fn bv_embedded_check_ceiling() -> u64 {
    BV_EMBEDDED_CHECK_CEILING
}

/// Whether `checks` embedded checks have exhausted the budget.
///
/// The single predicate both accessors below use, so the value
/// [`bv_embedded_check_ceiling`] reports and the point at which the budget
/// actually fires cannot drift apart — which is the one way an accessor-based
/// guard could be satisfied while the solver behaved differently.
pub(crate) fn bv_embedded_budget_is_spent(checks: u64) -> bool {
    checks > bv_embedded_check_ceiling()
}

impl TheoryManager<'_> {
    /// Charge one embedded bit-blasted check to this `check`'s deterministic
    /// budget and report whether the budget is now spent.
    ///
    /// See [`BV_EMBEDDED_CHECK_CEILING`] for what this bounds and why the two
    /// refinement counters cannot see it.
    pub(super) fn charge_bv_embedded_check(&mut self) -> bool {
        self.statistics.bv_embedded_checks = self.statistics.bv_embedded_checks.saturating_add(1);
        bv_embedded_budget_is_spent(self.statistics.bv_embedded_checks)
    }

    /// Whether the embedded-check budget is already spent, without charging
    /// for another one.
    pub(super) fn bv_embedded_budget_spent(&self) -> bool {
        bv_embedded_budget_is_spent(self.statistics.bv_embedded_checks)
    }
}

#[cfg(test)]
mod tests {
    use super::{bv_embedded_budget_is_spent, bv_embedded_check_ceiling};

    /// **The default-profile guard on the ceiling** (decision (26), finding
    /// R5-4).
    ///
    /// The two pins that would otherwise catch a widened ceiling —
    /// `the_index_width_three_cardinality_ladder_terminates` and
    /// `the_store_term_cliff_terminates` in
    /// `oxiz-solver/tests/round4_pass4_recheck_pins.rs` — both return early
    /// under `cfg!(debug_assertions)`, which is the profile
    /// `cargo nextest run --workspace` builds, and the cheapest script that
    /// reaches the ceiling costs over two minutes in release.  So in the
    /// default gate they assert nothing about this number and nothing cheap
    /// could.
    ///
    /// This reads the live constant instead.  It is not an `include_str!`
    /// source-text pin: `bv_embedded_check_ceiling` returns the value the
    /// solver spends, and the second half below ties that value to the point
    /// at which the budget actually fires, so an accessor that drifted from
    /// the charge path would fail here too.
    ///
    /// If a calibration pass changes the ceiling, change this number with it
    /// and say in `TODO.md` `#P2b-46` what the new value was calibrated
    /// against — the discriminating constraint is "no script the **base**
    /// `c4b04b7` decides may become `unknown`", over `bench/`, the `rc2`–`rc5`
    /// corpora and the `rf6/cal` store/cardinality ladder.
    #[test]
    fn the_embedded_check_ceiling_is_the_calibrated_value() {
        assert_eq!(
            bv_embedded_check_ceiling(),
            250_000,
            "the embedded-check ceiling is a calibrated number, not a knob: \
             widening it makes a slow script answer where the base decides it \
             in milliseconds, and the two pins that would otherwise catch that \
             are release-only.  See this test's doc."
        );
    }

    /// The accessor and the budget fire at the same count.
    #[test]
    fn the_budget_fires_exactly_one_check_past_the_ceiling() {
        let ceiling = bv_embedded_check_ceiling();
        assert!(
            !bv_embedded_budget_is_spent(ceiling),
            "the ceiling itself is affordable; the budget is `checks > ceiling`"
        );
        assert!(
            bv_embedded_budget_is_spent(ceiling + 1),
            "one check past the ceiling must exhaust the budget"
        );
        assert!(!bv_embedded_budget_is_spent(0));
    }
}
