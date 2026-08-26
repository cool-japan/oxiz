//! The ledger of committed arithmetic witnesses, and withdrawing from it.
//!
//! ## What goes wrong without one
//!
//! The search hands each arithmetic variable a rational witness taken from
//! whatever region its currently-assigned constraints still allow
//! (`NlsatSolver::pick_arith_value`). When the constraints on that variable
//! mention no other variable, the pick is as good as any other point in the
//! region and never needs revisiting. Coupling breaks that: after `x` is
//! handed a witness, the region left for `y` in `x·y = 12` depends entirely on
//! *which* witness `x` got, and the region a first-choice `x = 0` leaves for
//! `y` is empty even though the equation has infinitely many solutions.
//!
//! An empty region is therefore two very different things wearing the same
//! face. It is a proof of unsatisfiability when nothing about it was chosen,
//! and it is merely a bad guess when something was. Reading the second as the
//! first is how `x·y = 12` came back `unsat`, and the decision level is no
//! help in telling them apart — level 0 says only that no *Boolean* branch is
//! outstanding, which has no bearing on how many arithmetic witnesses were
//! free picks.
//!
//! ## What this module provides instead
//!
//! [`WitnessLedger`] remembers, for every witness the search commits to, where
//! that witness came from and which values that source has already produced.
//! When the search reaches a dead end it cannot certify as a theory lemma, it
//! *withdraws*: the ledger hands back a [`Withdrawal`] naming the variables
//! whose witnesses are being taken away and, when one of them still has an
//! untried point in its recorded region, the replacement to install.
//!
//! Two properties make this safe to do at any point in the search:
//!
//! * Nothing is learned. A withdrawal changes only which rational number an
//!   arithmetic variable holds; no clause is added, no Boolean assignment is
//!   touched, and conflict analysis never sees it. It cannot interact with
//!   CDCL at all, correctly or otherwise.
//! * Running out is not a proof. A region whose untried points this module's
//!   own bounded enumeration has exhausted is not a region without points, so
//!   an exhausted ledger yields `Unknown` and never `Unsat`.
//!
//! A variable whose region was a single real point is recorded as
//! [`CommittedWitness::Pinned`] rather than as a source to draw from — there
//! is nothing to draw. That distinction is also what
//! `NlsatSolver::certify_forced_chain_conflict` reads: a dead end reached
//! through nothing but pinned variables really is a theory conflict, because
//! no other assignment was ever available to reach it differently.

use crate::interval_set::IntervalSet;
use num_rational::BigRational;
use oxiz_math::polynomial::Var;
use smallvec::SmallVec;

use super::NlsatSolver;

/// How many replacement witnesses one `solve()` call may install in total.
///
/// A region can hold unboundedly many untried points, so without a ceiling the
/// withdrawal machinery would itself become the non-termination hazard it was
/// added to avoid.
const RETRY_ALLOWANCE: u32 = 4096;

/// One arithmetic variable's entry in the ledger.
enum CommittedWitness {
    /// The constraints admitted exactly one real point for this variable, so
    /// its value was not a choice and no replacement exists.
    ///
    /// The region is dropped rather than stored: a pinned variable's region
    /// is the closed singleton holding the value already installed, so asking
    /// it for a point that value has not already used answers nothing.
    Pinned {
        /// The variable.
        subject: Var,
    },
    /// The variable's value is an exact real-algebraic point (see
    /// `solver/witness_algebraic.rs`), which this ledger cannot offer a
    /// replacement for.
    ///
    /// It is deliberately **not** [`CommittedWitness::Pinned`]: the point was a
    /// choice among the cells of a decomposition, so it must never license
    /// `NlsatSolver::certify_forced_chain_conflict`, which reads an all-pinned
    /// ledger as "no arithmetic choice was ever made". It is not
    /// [`CommittedWitness::Chosen`] either: the alternatives to `√2` are the
    /// *other roots*, not the rationals crowded around it, every one of which
    /// already violates the equality that produced it. Offering those would
    /// spend the retry allowance on known-failing points. A withdrawal
    /// therefore drops this entry and keeps walking to an earlier genuine
    /// choice; retrying alternative algebraic roots is a later phase.
    Algebraic {
        /// The variable.
        subject: Var,
    },
    /// The variable's value was a free pick out of `region`.
    Chosen {
        /// The variable.
        subject: Var,
        /// Every point drawn for it so far, oldest first. The first entry is
        /// the original pick.
        drawn: Vec<BigRational>,
        /// Where replacements come from — the region as it stood when the
        /// original pick was made.
        region: IntervalSet,
    },
}

impl CommittedWitness {
    /// The variable this entry is about.
    fn subject(&self) -> Var {
        match self {
            CommittedWitness::Pinned { subject }
            | CommittedWitness::Algebraic { subject }
            | CommittedWitness::Chosen { subject, .. } => *subject,
        }
    }
}

/// The plan a [`WitnessLedger::withdraw`] call produces, for the caller to
/// apply to the arithmetic assignment.
///
/// Splitting the decision from its application keeps the ledger a pure record:
/// it never reaches into the solver's assignment, so what it decides can be
/// unit-tested on its own and cannot leave the assignment and the ledger
/// disagreeing about who holds what.
pub(super) struct Withdrawal {
    /// Variables whose witnesses are withdrawn and which must be unassigned,
    /// most recent first. May be empty (nothing was withdrawn).
    pub(super) released: SmallVec<[Var; 4]>,
    /// The variable to re-assign and the untried point to give it, when one
    /// was found. `None` means every recorded region is spent (or the retry
    /// allowance is gone), which the caller must read as incompleteness.
    pub(super) replacement: Option<(Var, BigRational)>,
}

/// Chronological record of the arithmetic witnesses the search has committed
/// to, most recent last.
pub(crate) struct WitnessLedger {
    /// One entry per committed witness.
    entries: Vec<CommittedWitness>,
    /// Replacements still allowed before the search must concede.
    retries_left: u32,
}

impl Default for WitnessLedger {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            retries_left: RETRY_ALLOWANCE,
        }
    }
}

impl WitnessLedger {
    /// Record that `subject` has been given `value`.
    ///
    /// `region` is the set the value was drawn from, or `None` when the
    /// constraints left exactly one real point and nothing was chosen.
    pub(super) fn record(&mut self, subject: Var, value: BigRational, region: Option<IntervalSet>) {
        self.entries.push(match region {
            Some(region) => CommittedWitness::Chosen {
                subject,
                drawn: vec![value],
                region,
            },
            None => CommittedWitness::Pinned { subject },
        });
    }

    /// Record that `subject` has been given an exact algebraic point.
    ///
    /// See [`CommittedWitness::Algebraic`] for why this is neither a pin nor a
    /// replaceable choice.
    pub(super) fn record_algebraic(&mut self, subject: Var) {
        self.entries.push(CommittedWitness::Algebraic { subject });
    }

    /// Take back the most recent witness that still has an alternative, and
    /// say which variables lose theirs along the way.
    ///
    /// The walk runs from the newest entry backwards: a variable whose region
    /// is spent is not the one to blame for the dead end on its own, but an
    /// *earlier* variable's pick may still be, so an exhausted entry is
    /// dropped and the search for an alternative continues behind it.
    ///
    /// With no allowance left this withdraws nothing at all — neither
    /// releasing nor dropping any entry — so a conceded search leaves the
    /// assignment exactly as the caller found it.
    pub(super) fn withdraw(&mut self) -> Withdrawal {
        let mut released: SmallVec<[Var; 4]> = SmallVec::new();
        while self.retries_left > 0 {
            let Some(entry) = self.entries.last_mut() else {
                break;
            };
            released.push(entry.subject());
            if let CommittedWitness::Chosen {
                subject,
                drawn,
                region,
            } = entry
                && let Some(point) = region.sample_avoiding(drawn)
            {
                self.retries_left -= 1;
                drawn.push(point.clone());
                let subject = *subject;
                return Withdrawal {
                    released,
                    replacement: Some((subject, point)),
                };
            }
            self.entries.pop();
        }
        Withdrawal {
            released,
            replacement: None,
        }
    }

    /// Whether every recorded witness was pinned by the constraints rather
    /// than chosen — and there is at least one.
    ///
    /// See `NlsatSolver::certify_forced_chain_conflict`, the only caller: a
    /// dead end reached without ever making an arithmetic choice cannot be
    /// blamed on one.
    pub(super) fn every_witness_pinned(&self) -> bool {
        !self.entries.is_empty()
            && self
                .entries
                .iter()
                .all(|entry| matches!(entry, CommittedWitness::Pinned { .. }))
    }

    /// Forget every recorded witness, keeping the remaining retry allowance.
    ///
    /// Used after a backjump or an incremental reset unassigns the arithmetic
    /// variables: a region computed against Boolean assignments that no longer
    /// hold describes nothing, and a point drawn from it could violate
    /// whatever the search assigns differently this time.
    pub(super) fn forget_all(&mut self) {
        self.entries.clear();
    }

    /// Forget every recorded witness *and* restore the full retry allowance,
    /// as a fresh `solve()` call is entitled to.
    pub(super) fn restart(&mut self) {
        self.entries.clear();
        self.retries_left = RETRY_ALLOWANCE;
    }
}

impl NlsatSolver {
    /// Install `value` for `var` and note it on the ledger.
    ///
    /// Every arithmetic assignment the main loop makes goes through here, so
    /// that every one of them is a candidate for [`Self::rewind_to_untried_witness`].
    /// `region` is `None` exactly when the value was pinned rather than picked.
    pub(super) fn commit_arith_witness(
        &mut self,
        var: Var,
        value: BigRational,
        region: Option<IntervalSet>,
    ) {
        self.assignment.set_arith(var, value.clone());
        self.eval_cache.clear();
        self.arith_witnesses.record(var, value, region);
    }

    /// Ask the ledger for a replacement witness and apply whatever it decides.
    ///
    /// `true` means an arithmetic variable now holds a point it has not held
    /// before and the main loop should carry on from there. `false` means the
    /// ledger had nothing left; the caller must answer `Unknown`, since an
    /// enumeration this module gave up on is not a region proved empty.
    pub(super) fn rewind_to_untried_witness(&mut self) -> bool {
        let Withdrawal {
            released,
            replacement,
        } = self.arith_witnesses.withdraw();
        for var in released {
            self.assignment.unset_arith(var);
        }
        let installed = if let Some((var, point)) = replacement {
            self.assignment.set_arith(var, point);
            true
        } else {
            false
        };
        // One clear covers the whole batch: nothing reads a cached evaluation
        // between the unassignments and the replacement.
        self.eval_cache.clear();
        installed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    fn bounded(lo: i64, hi: i64) -> IntervalSet {
        IntervalSet::from_interval(oxiz_math::interval::Interval::closed(rat(lo), rat(hi)))
    }

    /// A chosen witness is replaced from its own region, and the replacement
    /// is never a point the same entry has already produced.
    #[test]
    fn test_ledger_replaces_a_chosen_witness_without_repeating() {
        let mut ledger = WitnessLedger::default();
        ledger.record(0, rat(0), Some(bounded(0, 3)));

        // Drain the region rather than assuming how many points the
        // enumeration can find in it: what matters is that every point it
        // does hand back is fresh and inside the region, and that it stops.
        let mut seen = vec![rat(0)];
        for _ in 0..128 {
            let withdrawal = ledger.withdraw();
            assert_eq!(withdrawal.released.as_slice(), &[0]);
            let Some((var, point)) = withdrawal.replacement else {
                break;
            };
            assert_eq!(var, 0);
            assert!(!seen.contains(&point), "{point} was handed out twice");
            assert!(
                point >= rat(0) && point <= rat(3),
                "{point} left the region"
            );
            seen.push(point);
        }
        assert!(
            seen.len() >= 4,
            "[0, 3] should yield several distinct witnesses, got {seen:?}"
        );
        assert!(
            ledger.entries.is_empty(),
            "the drained entry must have been dropped"
        );
    }

    /// A region holding one point has nothing to offer once that point is the
    /// one already installed, so the entry is dropped and the ledger empties.
    #[test]
    fn test_a_single_point_region_is_immediately_exhausted() {
        let mut ledger = WitnessLedger::default();
        ledger.record(0, rat(4), Some(bounded(4, 4)));

        let withdrawal = ledger.withdraw();
        assert!(withdrawal.replacement.is_none());
        assert_eq!(
            withdrawal.released.as_slice(),
            &[0],
            "the variable still loses its witness on the way out"
        );
        assert!(ledger.entries.is_empty());
    }

    /// A pinned witness offers no replacement, so a withdrawal walks past it
    /// to an earlier entry that does — releasing both on the way.
    #[test]
    fn test_ledger_walks_past_pinned_entries_to_an_earlier_choice() {
        let mut ledger = WitnessLedger::default();
        ledger.record(0, rat(1), Some(bounded(1, 9)));
        ledger.record(1, rat(4), None);

        let withdrawal = ledger.withdraw();
        let Some((var, point)) = withdrawal.replacement else {
            panic!("variable 0's region [1, 9] still had untried points");
        };
        assert_eq!(var, 0, "the pinned entry cannot be the one retried");
        assert_ne!(point, rat(1));
        assert_eq!(
            withdrawal.released.as_slice(),
            &[1, 0],
            "both the pinned variable and the retried one lose their witnesses"
        );
        assert!(!ledger.every_witness_pinned());
    }

    /// An algebraic witness offers no rational replacement, and — critically —
    /// must not read as "pinned", which would license a forced-chain conflict
    /// certificate over a point that was in fact chosen.
    #[test]
    fn test_algebraic_entry_offers_nothing_and_is_not_pinned() {
        let mut ledger = WitnessLedger::default();
        ledger.record_algebraic(0);
        assert!(
            !ledger.every_witness_pinned(),
            "an algebraic pick is a choice, not a pin"
        );

        let withdrawal = ledger.withdraw();
        assert!(withdrawal.replacement.is_none());
        assert_eq!(withdrawal.released.as_slice(), &[0]);
        assert!(ledger.entries.is_empty());
    }

    /// A withdrawal walks *past* an algebraic entry to an earlier free choice,
    /// releasing both, exactly as it does for a pinned one.
    #[test]
    fn test_withdrawal_walks_past_an_algebraic_entry() {
        let mut ledger = WitnessLedger::default();
        ledger.record(0, rat(1), Some(bounded(1, 9)));
        ledger.record_algebraic(1);

        let withdrawal = ledger.withdraw();
        let Some((var, point)) = withdrawal.replacement else {
            panic!("variable 0's region [1, 9] still had untried points");
        };
        assert_eq!(var, 0);
        assert_ne!(point, rat(1));
        assert_eq!(withdrawal.released.as_slice(), &[1, 0]);
    }

    /// An all-pinned ledger is what licenses the forced-chain conflict
    /// certificate, and an empty one is not.
    #[test]
    fn test_every_witness_pinned_requires_a_nonempty_all_pinned_ledger() {
        let mut ledger = WitnessLedger::default();
        assert!(
            !ledger.every_witness_pinned(),
            "an empty ledger licenses nothing"
        );
        ledger.record(0, rat(4), None);
        assert!(ledger.every_witness_pinned());
        ledger.record(1, rat(0), Some(bounded(0, 5)));
        assert!(!ledger.every_witness_pinned());
    }

    /// With the allowance spent, a withdrawal must leave the ledger and the
    /// caller's assignment completely untouched rather than unwinding it.
    #[test]
    fn test_a_spent_allowance_withdraws_nothing() {
        let mut ledger = WitnessLedger {
            retries_left: 0,
            ..Default::default()
        };
        ledger.record(0, rat(0), Some(bounded(0, 100)));

        let withdrawal = ledger.withdraw();
        assert!(withdrawal.replacement.is_none());
        assert!(
            withdrawal.released.is_empty(),
            "a conceded search must not release witnesses it is about to report on"
        );
        assert_eq!(ledger.entries.len(), 1, "no entry may be dropped either");
    }

    /// `forget_all` drops the record without refunding the allowance;
    /// `restart` does both.
    #[test]
    fn test_forget_all_keeps_the_allowance_and_restart_refunds_it() {
        let mut ledger = WitnessLedger::default();
        ledger.record(0, rat(0), Some(bounded(0, 100)));
        assert!(ledger.withdraw().replacement.is_some());
        let spent_to = ledger.retries_left;
        assert!(spent_to < RETRY_ALLOWANCE);

        ledger.forget_all();
        assert!(ledger.entries.is_empty());
        assert_eq!(ledger.retries_left, spent_to);

        ledger.record(0, rat(0), Some(bounded(0, 100)));
        ledger.restart();
        assert!(ledger.entries.is_empty());
        assert_eq!(ledger.retries_left, RETRY_ALLOWANCE);
    }
}
