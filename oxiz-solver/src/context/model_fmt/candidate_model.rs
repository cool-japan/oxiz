//! Decision (48) at the `Context`: keep a candidate model that already
//! satisfies every assertion **as printed**.
//!
//! The solver half is `solver::array_completion_certify::candidate`, whose
//! docs carry the rationale.  This half owns the renderer, so it is the one
//! that can say what the candidate's arrays *print as*: each array the
//! completion interpreted is rendered from the recorded candidate model
//! exactly as `(get-model)` would render it, the printed value is parsed back
//! into a term, and the solver certifies those interpretations with the
//! completion's own certificate.  A value that does not print as a closed term
//! (an uninterpreted-sort witness `@uc_S_n`, the `?` placeholder) cannot be
//! parsed back and leaves the completion in place.

use super::*;

impl Context {
    /// Put the candidate model back when the completion replaced one that
    /// already satisfied every assertion (decision (48)); otherwise leave the
    /// certified completion installed.  Called after every `Sat`.
    pub(in crate::context) fn keep_a_correct_candidate_model(&mut self) {
        let Some(candidate) = self.solver.take_replaced_candidate() else {
            return;
        };
        let arrays = self.solver.completed_arrays(&self.terms);
        if arrays.is_empty() {
            return;
        }
        // Rendered with the fresh values of `#P2b-71`, as `(get-model)`
        // would print it.
        let candidate_model = self.with_fresh_values(&candidate.model);
        let class_values = self.build_class_values(&candidate_model);
        let mut printed: Vec<(TermId, String)> = Vec::with_capacity(arrays.len());
        for array in arrays {
            let Some(sort) = self.terms.get(array).map(|t| t.sort) else {
                return;
            };
            // The same fallbacks, in the same order, as `get_model`.
            let text = if let Some(value) = candidate_model.get(array) {
                self.format_value(value)
            } else if let Some(chain) =
                self.array_model_value(array, sort, &candidate_model, &class_values)
            {
                chain
            } else {
                self.default_value(sort)
            };
            printed.push((array, text));
        }
        let mut rendered: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (array, text) in printed {
            let Ok(term) = oxiz_core::smtlib::parse_term(&text, &mut self.terms) else {
                return;
            };
            rendered.insert(array, term);
        }
        if self
            .solver
            .candidate_interpretation_certifies(&candidate, &rendered, &mut self.terms)
        {
            self.solver.restore_candidate_model(candidate);
        }
    }
}
