//! Decision (48): a candidate model that already satisfies every assertion is
//! **kept**; the completion replaces only a falsifying one.
//!
//! On a `Sat` no honesty gate takes away, decision (40) runs the completion
//! and installs the certified interpretation in place of the candidate's
//! arrays (`#P2b-51`: the candidate could falsify the very universal the
//! verdict rests on).  But a candidate that is already a model must not be
//! swapped for a different one: `bench/extended_theories/AUFLIRA/06` printed
//! `(store (store (store ((as const (Array Int Real)) 0.0) 0 0.0) 1 0.0) 42
//! 0.0)` — correct — and re-fix pass 12 replaced it by the certified constant
//! `((as const (Array Int Real)) 0.0)`, a response change that corrected
//! nothing.
//!
//! # Whose interpretation is judged
//!
//! The one `(get-model)` *prints*.  The solver-level candidate is a partial
//! model — reads, a class's constant, a `store`'s own writes — and pass 7's
//! reverted `Sat`-exit gate failed exactly by judging it instead of the
//! rendered arrays.  So the judgement is made by `Context` (which owns the
//! renderer): it renders each completed array from the recorded candidate the
//! same way `(get-model)` would, parses the printed value back into a term,
//! and hands the interpretations to
//! [`Solver::candidate_interpretation_certifies`] — the very certificate the
//! completion itself passed (a validity query per universal, one over the
//! assertions), discharged over the candidate's interpretation.  Only a
//! passing certificate puts the candidate back; a failing, declined or
//! unparseable one leaves the certified completion installed, so a model is
//! never swapped for one nobody verified.

use oxiz_core::ast::{TermId, TermManager};
use rustc_hash::FxHashMap;

use super::super::Solver;
use super::super::types::Model;
use super::Goal;

/// The candidate model a completion replaced on a `Sat` no gate took away.
#[derive(Debug, Clone)]
pub(crate) struct ReplacedCandidate {
    /// The candidate model as the search left it.
    pub(crate) model: Model,
}

impl Solver {
    /// Take the candidate the last `check` replaced, if any (decision (48)).
    pub(crate) fn take_replaced_candidate(&mut self) -> Option<ReplacedCandidate> {
        self.replaced_candidate.take()
    }

    /// The arrays the installed completion interprets: every model key the
    /// current model binds to an installed array value.
    #[must_use]
    pub(crate) fn completed_arrays(&self, manager: &TermManager) -> Vec<TermId> {
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        let mut arrays: Vec<TermId> = model
            .assignments()
            .iter()
            .filter(|&(_, &value)| super::is_array_value(value, manager))
            .map(|(&array, _)| array)
            .collect();
        arrays.sort_unstable_by_key(|term| term.raw());
        arrays
    }

    /// Whether `rendered` — the candidate's arrays as `(get-model)` prints
    /// them — together with the candidate's scalar values satisfies every
    /// assertion, by the completion's own certificate.  `false` whenever the
    /// certificate cannot say so (it fails, declines, or runs out of budget).
    pub(crate) fn candidate_interpretation_certifies(
        &mut self,
        candidate: &ReplacedCandidate,
        rendered: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> bool {
        let assertions = self.assertions.clone();
        let Some(goal) = Goal::build(&assertions, candidate.model.assignments(), manager) else {
            return false;
        };
        if goal
            .arrays
            .iter()
            .any(|array| !rendered.contains_key(array))
        {
            return false;
        }
        let logic = self.logic.clone();
        let mut queries = 0usize;
        goal.certificate_passes(self, rendered, manager, logic.as_deref(), &mut queries)
    }

    /// Put the candidate model back in place of the completion (decision
    /// (48)); the caller has certified that it satisfies every assertion.
    pub(crate) fn restore_candidate_model(&mut self, candidate: ReplacedCandidate) {
        self.model = Some(candidate.model);
        self.certified_array_model = None;
        self.candidate_certified_as_printed = true;
    }
}
