//! The published-model certificate: **a model is certified or absent, never
//! falsifying** (`#P2b-51` reopened, decision (54)(i)).
//!
//! The completion (decision (40)) replaces a candidate's arrays by a certified
//! interpretation where it can, and decision (48) keeps a candidate that
//! already satisfies every assertion as printed.  Everywhere else — a goal
//! the completion declines — the `sat` used to publish the candidate
//! unchecked, and recheck 13 measured 141 falsifying models in 2,574 `sat`s
//! of one fuzz campaign.  This is the net under all of it: at a quantified
//! array `Sat` whose model no certificate has covered, `Context` renders every
//! declared constant exactly as `(get-model)` would, parses the printed values
//! back into terms, and hands them here; the completion's own certificate
//! (a validity query per universal, a witness per `exists`, one check over
//! the assertions — `certificate_passes`) is discharged over that
//! interpretation.  A model that does not certify is withheld: `(get-model)`
//! answers an error naming the first assertion that fails and, where an exact
//! evaluation finds one, the point it fails at.  The verdict is untouched
//! (decision (40)): the search's own `sat` claim stands.

use oxiz_core::ast::{TermId, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    Combinations, Goal, completion_points, declined, evaluate_closed, is_array_sort,
    is_small_enough_to_certify, peel_universal_with_vars, polarity,
};
use crate::solver::Solver;

/// Instances one universal's falsifying-point search may evaluate.
const MAX_EXPLAIN_INSTANCES: usize = 256;

/// Characters of an assertion quoted in the error message.
const MAX_QUOTED_ASSERTION: usize = 160;

impl Solver {
    /// Whether a certificate of this check already covered exactly the
    /// model the last `check` left: a completion it installed, or a
    /// candidate decision (48) certified as printed.
    #[must_use]
    pub(crate) fn model_already_certified(&self) -> bool {
        let Some(model) = self.model.as_ref() else {
            return false;
        };
        self.candidate_certified_as_printed
            || self
                .certified_array_model
                .as_ref()
                .is_some_and(|certified| certified == model.assignments())
    }

    /// Certify `printed` — every declared constant mapped to its printed
    /// value, parsed back into a term — against `goal`: the asserted terms
    /// with every uninterpreted function replaced by its printed table
    /// (`context::model_fmt::printed_check`).  `originals` are the asserted
    /// terms as written, quoted by the error.  `Err` carries the reason, for
    /// the error `(get-model)` answers instead.
    pub(crate) fn certify_printed_model(
        &mut self,
        goal: &[TermId],
        originals: &[TermId],
        printed: &FxHashMap<TermId, TermId>,
        hypotheses: &[TermId],
        manager: &mut TermManager,
    ) -> Result<(), String> {
        let assertions = goal.to_vec();
        let assignments = self
            .model
            .as_ref()
            .map(|model| model.assignments().clone())
            .unwrap_or_default();
        let goal =
            Goal::for_interpretation(&assertions, &assignments, printed, hypotheses, manager)?;
        let completion: FxHashMap<TermId, TermId> = goal
            .arrays
            .iter()
            .filter_map(|&array| printed.get(&array).map(|&value| (array, value)))
            .collect();
        let logic = self.logic.clone();
        let mut queries = 0usize;
        if goal.certificate_passes(self, &completion, manager, logic.as_deref(), &mut queries) {
            return Ok(());
        }
        Err(goal.explain_failure(self, &completion, originals, manager, logic.as_deref()))
    }
}

impl Solver {
    /// Where the printed model fails its certificate, complete it around what
    /// it got right: each independent group of arrays (`groups`) keeps its
    /// printed interpretation when that certifies on the group's own
    /// sub-goal, and is completed by the ordinary searches otherwise; the
    /// union, with the printed scalars, is installed when it certifies over
    /// every assertion and passes the ground-assertion gate.  `true` when it
    /// installed.
    ///
    /// This is what completes a goal whose groups need different treatment:
    /// `¬∃i. a0[i] = #b0` beside `∃i. a1[i] = #b1` and `∃i. a1[i] = #b0`
    /// (a fuzzed script, `fuzz_qc` seed 29093020) has a constant completion
    /// for `a0` and none for `a1`, whose printed candidate — the two Skolem
    /// witnesses over `#b0` — is right as it stands.
    pub(crate) fn complete_around_printed(
        &mut self,
        goal: &[TermId],
        printed: &FxHashMap<TermId, TermId>,
        hypotheses: &[TermId],
        manager: &mut TermManager,
    ) -> bool {
        let assertions = goal.to_vec();
        let assignments = self
            .model
            .as_ref()
            .map(|model| model.assignments().clone())
            .unwrap_or_default();
        let Ok(goal) =
            Goal::for_interpretation(&assertions, &assignments, printed, hypotheses, manager)
        else {
            return false;
        };
        let logic = self.logic.clone();
        let mut merged: FxHashMap<TermId, TermId> = FxHashMap::default();
        for group in super::groups::array_groups(&goal.arrays, &goal.assertions, manager) {
            let printed_group: FxHashMap<TermId, TermId> = group
                .iter()
                .filter_map(|&array| printed.get(&array).map(|&value| (array, value)))
                .collect();
            let Some(sub_goal) = goal.restricted_to(&group, manager) else {
                return false;
            };
            let mut queries = 0usize;
            if sub_goal.certificate_passes(
                self,
                &printed_group,
                manager,
                logic.as_deref(),
                &mut queries,
            ) {
                merged.extend(printed_group);
                continue;
            }
            if group.len() > super::MAX_COMPLETED_ARRAYS || sub_goal.universals.is_empty() {
                return false;
            }
            let Some(completion) =
                self.search_goal_completion(&sub_goal, &assignments, manager, logic.as_deref())
            else {
                return false;
            };
            merged.extend(completion);
        }
        let mut queries = 0usize;
        if !goal.certificate_passes(self, &merged, manager, logic.as_deref(), &mut queries) {
            return false;
        }
        let original = self.model.clone();
        self.install_completed_model(&goal, &merged, manager);
        if let Some(model) = self.model.as_mut() {
            let mut scalars: Vec<(TermId, TermId)> = printed
                .iter()
                .filter(|(term, _)| !merged.contains_key(term))
                .map(|(&term, &value)| (term, value))
                .collect();
            scalars.sort_unstable_by_key(|&(term, _)| term.raw());
            for (term, value) in scalars {
                model.set(term, value);
            }
            self.certified_array_model = Some(model.assignments().clone());
        }
        if self.quantified_model_refutes_ground_assertions(manager) {
            self.model = original;
            self.certified_array_model = None;
            return false;
        }
        true
    }
}

impl Goal {
    /// The goal of the published-model certificate: every array is
    /// interpreted by its printed value (no search, so no bound on how many),
    /// every scalar by its printed value where it has one and by the
    /// candidate's otherwise, and ground applications of uninterpreted
    /// functions by the candidate (`declined`).  `Err` names why the model
    /// cannot be certified at all.
    fn for_interpretation(
        assertions: &[TermId],
        assignments: &FxHashMap<TermId, TermId>,
        printed: &FxHashMap<TermId, TermId>,
        hypotheses: &[TermId],
        manager: &mut TermManager,
    ) -> Result<Self, String> {
        if !is_small_enough_to_certify(assertions, manager) {
            return Err("the assertions are too large to certify".to_string());
        }
        let quantifiers = polarity::collect_quantifiers(assertions, manager)
            .ok_or_else(|| "a quantifier the certificate cannot read".to_string())?;
        let mut arrays: Vec<TermId> = Vec::new();
        let mut scalar_pins: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        for &assertion in assertions {
            let mut free = manager.free_vars_including_patterns(assertion);
            free.sort_unstable_by_key(|term| term.raw());
            for var in free {
                if !seen.insert(var) {
                    continue;
                }
                let Some(sort) = manager.get(var).map(|d| d.sort) else {
                    continue;
                };
                if is_array_sort(sort, manager) {
                    if !printed.contains_key(&var) {
                        return Err(format!(
                            "no printed value for the array {}",
                            print_quoted(var, manager)
                        ));
                    }
                    arrays.push(var);
                } else if let Some(&value) = printed.get(&var).or_else(|| assignments.get(&var)) {
                    scalar_pins.insert(var, value);
                }
            }
        }
        let mut ground_values: FxHashMap<TermId, TermId> = FxHashMap::default();
        if declined::has_uninterpreted_application(assertions, manager) {
            let (_, values) = declined::interpret_from_candidate(
                assertions,
                &arrays,
                assignments,
                &scalar_pins,
                false,
                manager,
            )
            .ok_or_else(|| {
                "an uninterpreted function the certificate cannot interpret".to_string()
            })?;
            ground_values = values;
        }
        // No bound on the number of arrays: nothing is searched here.
        Ok(Self {
            assertions: assertions.to_vec(),
            universals: quantifiers.universals,
            arrays,
            scalar_pins,
            existentials: quantifiers.existentials,
            ground_values,
            negations: quantifiers.negations,
            hypotheses: hypotheses.to_vec(),
        })
    }

    /// Why `completion` fails: the first assertion that does not certify on
    /// its own, and — where an exact evaluation finds one — a point at which
    /// one of its universals is false.
    fn explain_failure(
        &self,
        solver: &Solver,
        completion: &FxHashMap<TermId, TermId>,
        originals: &[TermId],
        manager: &mut TermManager,
        logic: Option<&str>,
    ) -> String {
        // One query budget for the whole explanation: it only produces text,
        // and it runs before `check-sat` returns.
        let mut queries = 0usize;
        for (number, &assertion) in self.assertions.iter().enumerate() {
            let Some(quantifiers) = polarity::collect_quantifiers(&[assertion], manager) else {
                continue;
            };
            let single = Goal {
                assertions: vec![assertion],
                universals: quantifiers.universals,
                arrays: self.arrays.clone(),
                scalar_pins: self.scalar_pins.clone(),
                existentials: quantifiers.existentials,
                ground_values: self.ground_values.clone(),
                negations: quantifiers.negations,
                hypotheses: self.hypotheses.clone(),
            };
            if single.certificate_passes(solver, completion, manager, logic, &mut queries) {
                continue;
            }
            if queries >= super::MAX_CERTIFICATE_QUERIES {
                // Out of budget: naming this assertion would be a guess.
                return "the assertions could not be certified within the certificate's query budget"
                    .to_string();
            }
            let quoted = print_quoted(originals.get(number).copied().unwrap_or(assertion), manager);
            return match single.falsifying_point(solver, completion, manager) {
                Some(point) => format!("assertion {} {quoted} is false at {point}", number + 1),
                None => format!("assertion {} {quoted} could not be certified", number + 1),
            };
        }
        "the assertions could not be certified together".to_string()
    }

    /// A point at which one of the goal's universals (or the dual of a
    /// negated `exists`) evaluates to `false` under `completion`, printed as
    /// `((x v) …)`; `None` when the sampled points find none.
    fn falsifying_point(
        &self,
        solver: &Solver,
        completion: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> Option<String> {
        let substitution = self.full_substitution(completion);
        let mut points = self.points_by_sort(manager);
        for (sort, extra) in completion_points(completion, manager) {
            let list = points.entry(sort).or_default();
            for point in extra {
                if !list.contains(&point) {
                    list.push(point);
                }
            }
        }
        for (position, &universal) in self.universals.iter().enumerate() {
            let (body, vars) = peel_universal_with_vars(universal, position, manager)?;
            let mut lists: Vec<Vec<TermId>> = Vec::with_capacity(vars.len());
            for &var in &vars {
                let sort: SortId = manager.get(var)?.sort;
                let mut list = points.get(&sort).cloned().unwrap_or_default();
                list.extend(super::pinned::gap_representatives(sort, &list, manager));
                lists.push(list);
            }
            for combination in Combinations::with_cap(&lists, MAX_EXPLAIN_INSTANCES) {
                let map: FxHashMap<TermId, TermId> = vars
                    .iter()
                    .copied()
                    .zip(combination.iter().copied())
                    .collect();
                let instance = manager.substitute(body, &map);
                let instance = manager.substitute(instance, &substitution);
                if evaluate_closed(solver, instance, manager) == Some(false) {
                    let names = binder_names(universal, manager);
                    let parts: Vec<String> = names
                        .iter()
                        .zip(combination.iter())
                        .map(|(name, &value)| format!("({name} {})", print_quoted(value, manager)))
                        .collect();
                    return Some(format!("({})", parts.join(" ")));
                }
            }
        }
        // A closed assertion that evaluates false has no point to name.
        None
    }
}

/// The bound variable names of a chain of consecutive quantifiers of one
/// kind, in binder order.
fn binder_names(term: TermId, manager: &TermManager) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut current = term;
    while let Some(data) = manager.get(current) {
        match &data.kind {
            oxiz_core::ast::TermKind::Forall { vars, body, .. }
            | oxiz_core::ast::TermKind::Exists { vars, body, .. } => {
                names.extend(
                    vars.iter()
                        .map(|&(name, _)| manager.resolve_str(name).to_string()),
                );
                current = *body;
            }
            _ => break,
        }
    }
    names
}

/// `term` as SMT-LIB text on one line, cut at [`MAX_QUOTED_ASSERTION`]
/// characters, with every `"` doubled so it can sit inside an SMT-LIB string.
fn print_quoted(term: TermId, manager: &TermManager) -> String {
    let text = oxiz_core::smtlib::Printer::new(manager).print_term(term);
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut cut: String = one_line.chars().take(MAX_QUOTED_ASSERTION).collect();
    if one_line.chars().count() > MAX_QUOTED_ASSERTION {
        cut.push_str(" ...");
    }
    cut.replace('"', "\"\"")
}
