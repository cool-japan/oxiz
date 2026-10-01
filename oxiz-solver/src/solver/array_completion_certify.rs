//! Model **completion** for an array default under a binder, and the
//! quantifier-free **certificate** that has to pass before the `sat` it
//! licenses may be published (`#P2b-58`, decision (36)).
//!
//! # The gap this closes
//!
//! `finite_expand` decides a universal over a finite sort by writing it out at
//! every point, and it declines above a 64-point budget.  One bit above that
//! budget — `(_ BitVec 7)`, 128 points — a *satisfiable* quantified array
//! script stops being decided: MBQI reaches its fixpoint without ever being
//! able to say what the array is at the indices no ground term names, so the
//! candidate model leaves the array's **default** free and nothing may be
//! published from it.  `rk6/corpus/qmbqi120/q0074` is the minimal shape: its
//! quantifier reduces to `a1[i] = #b1` for every `i` over `(_ BitVec 7)`, which
//! is satisfiable exactly by the *constant* array `#b1` — an interpretation the
//! candidate model has no way to name.
//!
//! # What this module does
//!
//! Given the candidate model, it *completes* every array-sorted free variable
//! to a total interpretation — a constant array over a searched default, in the
//! spirit of Ge & de Moura's pins-plus-default model construction (CAV 2009)
//! and of [`crate::mbqi::model_certify`], which does the same thing for
//! uninterpreted functions over `Int` and `Real` — and then **certifies** the
//! completion before anything is published.
//!
//! # Why the certificate is sound, and why it is only ever one-directional
//!
//! Every query this module runs is a *validity* query, discharged by refuting
//! its negation with an ordinary quantifier-free solve:
//!
//! * a maximal `forall` sub-term `∀x⃗. ψ` is accepted as **true** only when
//!   `¬ψ[completion]`, with `x⃗` replaced by fresh reserved constants, is
//!   `Unsat`.  `Unsat` means `ψ[completion]` holds for *every* interpretation
//!   of whatever symbols are left in it, hence in particular under ours — so
//!   the conclusion survives any symbol this module failed to interpret.  The
//!   converse is **not** sound for the same reason (a `Sat` says some
//!   interpretation falsifies it, not that ours does), so a `forall` that
//!   cannot be certified true makes the whole attempt decline rather than be
//!   recorded as false.  An `exists` is certified true only by a witness — an
//!   instance at one of the goal's points that is itself valid (`declined`) —
//!   and a quantifier at negative polarity through its dual (`polarity`).
//! * the assertions themselves are accepted only when
//!   `(or (not A₁[completion]) … (not Aₙ[completion]))` is `Unsat`, which is
//!   the same one-directional argument taken over all of them at once — one
//!   quantifier-free query, exactly as decision (36) asks.
//!
//! So a `sat` is published only behind a passing certificate, and a *wrong*
//! completion can never produce one: `rk11/atk/m4_body_false_at_a_point.smt2`
//! and `m4b_stored_point_conflict.smt2` are refuted rather than published.
//!
//! # Where it runs, and what it may change (decision (40))
//!
//! [`Solver::array_completion_at_exit`] is called once per `check`, after
//! `check_core` and before the honesty gates, whatever `check_core`'s `Sat`
//! exit was:
//!
//! * where a verdict would otherwise be given up — `Unknown`, or a `Sat` a
//!   gate is about to take away — a passing certificate licenses `Sat`
//!   (`#P2b-58`);
//! * on a `Sat` no gate takes away, the **verdict is not touched**: a passing
//!   certificate replaces the candidate model's arrays by the certified
//!   interpretation, and a failed or declined one leaves the model exactly as
//!   it was (`#P2b-51`: the candidate model left every index the search did
//!   not pin to the sort default, which could falsify the very quantifier
//!   the verdict rests on — `(forall ((i (_ BitVec 7))) (= (select a i) #b1)))`
//!   was published with `a = ((as const …) #b0)`).
//!
//! In both cases the installed model then passes the same ground-assertion
//! gate every quantified `Sat` exit of `check_core` passes
//! (`quantified_model_refutes_ground_assertions`), and a model that fails it
//! is put back as it was.  Installing *replaces* an interpretation, so every
//! model entry derived from the old one — a read of a completed array, an
//! atom over such a read — is dropped, and `(get-value)` re-derives it from
//! the certified interpretation (the evaluator reads through an installed
//! array value; see `model_eval::open::store_chain`).
//!
//! # Pins plus a default (decision (41))
//!
//! The first search tries a constant array per array.  Where no constant
//! certifies, the second one — `array_completion_certify::pinned` — tries a
//! default **plus finitely many pinned points**, the points being the ground
//! index terms the goal already names (the literals of the index sort, and
//! the values of the index-sort constants the candidate model fixes), and the
//! values at them found by one quantifier-free *fill* query.  The fill query
//! only proposes; the interpretation it proposes goes through the very same
//! certificate, so a wrong proposal costs a query and never a verdict.
//!
//! # Decision (54): negated quantifiers, groups, infinite indices, the net
//!
//! A quantifier sub-term is certified at the value its polarity needs, a
//! negated `exists` through its dual universal (`polarity`); arrays no
//! assertion links are searched one group at a time, the three-array bound
//! applying to a group (`groups`); over `Int` / `Real` the candidate's own
//! points over a pooled default are proposed (`infinite`) and a universal is
//! certified pointwise (`split`); an equality between two literal array values
//! is decided exactly (`array_values`).  Where the completion still declines,
//! `Context` certifies the model it is about to print through the same
//! certificate and withholds one that fails, completing it around the groups
//! it gets right where it can (`printed`).
//!
//! # What it is not
//!
//! It is not a decision procedure for the fragment and does not pretend to be:
//! the default pool is finite, the pins are the ground index terms the goal
//! names and nothing else, and the combination count is capped, so a script
//! whose satisfying interpretation differs from every pooled default at an
//! index the goal does not name keeps its `unknown`.  Declining costs only
//! completeness.  `#P2b-50` — a *declared* sort whose cardinality nothing
//! pins — stays separate and open; this module never invents a domain, it
//! only interprets an array whose index and element sorts are both already
//! pinned.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::smtlib::reserved_name;
use oxiz_core::sort::{SortId, SortKind};
use rustc_hash::{FxHashMap, FxHashSet};

use super::array_axioms::CONST_ARRAY_FUNC;
use super::model_eval::EvalOutcome;
use super::types::Model;
use super::{EvalVal, Solver, SolverResult};

mod array_values;
mod candidate;
mod declined;
mod groups;
mod infinite;
mod pinned;
mod polarity;
mod printed;
mod split;
#[cfg(test)]
mod tests;

pub(crate) use candidate::ReplacedCandidate;

/// Array-sorted free variables one attempt may complete.
///
/// The search is a product over the pool below, so this is what keeps it
/// finite.  Three covers every script `#P2b-58` names (`q0106` has three
/// arrays under one binder) and bounds the product at `POOL^3`.
const MAX_COMPLETED_ARRAYS: usize = 3;

/// Default values tried per array.
const MAX_DEFAULT_CANDIDATES: usize = 6;

/// Default *combinations* across all completed arrays.
const MAX_COMBINATIONS: usize = 24;

/// Assertion-DAG nodes above which an attempt declines before it starts.
///
/// The certificate is a *validity* query per universal plus one for the
/// assertions, and a goal this module could actually discharge is small by
/// construction — the whole `#P2b-58` family is two to five assertions.  The
/// cap is what keeps a large benchmark from paying for a search that would
/// decline anyway.
const MAX_GOAL_NODES: usize = 4_096;

/// Quantifier-free certificate queries one attempt may spend in total.
///
/// Each query is a fresh, budgeted, quantifier-free solve; this bounds the
/// cost of the whole attempt independently of how the search goes.
const MAX_CERTIFICATE_QUERIES: usize = 96;

/// Boolean conflicts one certificate query may spend.
///
/// A deterministic budget, not a clock: the certificate's verdict must be a
/// property of the tree and not of the machine (decision (9)).  A query that
/// runs out answers `Unknown`, which this module reads as "not certified".
const CERTIFICATE_CONFLICT_BUDGET: u64 = 20_000;

/// Maximal quantifier sub-terms, plus the free symbols an attempt must pin.
struct Goal {
    /// The assertions, unchanged.
    assertions: Vec<TermId>,
    /// Every maximal `forall` sub-term of the assertion set, deduplicated and
    /// in first-encounter order so the attempt is deterministic.
    universals: Vec<TermId>,
    /// The array-sorted free variables to complete, in first-encounter order.
    arrays: Vec<TermId>,
    /// `scalar -> value`, the pins taken from the candidate model.
    scalar_pins: FxHashMap<TermId, TermId>,
    /// Every maximal `exists` sub-term at positive polarity, certified by a
    /// witness among the goal's points (`#P2b-63`, `declined`).
    existentials: Vec<TermId>,
    /// Ground terms left at the candidate model's value rather than
    /// completed — ground applications of uninterpreted functions and ground
    /// reads of arrays only ever read at ground indices (`#P2b-63`,
    /// `declined`); empty on every goal the completion took before.
    ground_values: FxHashMap<TermId, TermId>,
    /// `(original, dual)` for every quantifier sub-term at negative polarity
    /// (`polarity`): the dual is listed in `universals` / `existentials`, and
    /// the original is `false` when the dual is certified.
    negations: Vec<(TermId, TermId)>,
    /// Facts every query of the certificate is read under: for a printed
    /// model, that its uninterpreted-sort witnesses are pairwise distinct
    /// (`printed`); empty for the completion's own searches.
    hypotheses: Vec<TermId>,
}

impl Solver {
    /// The completion hook at `check`'s exit (decisions (36), (40)).
    ///
    /// `result` is what `check_core` answered and `gate_pending` whether an
    /// honesty gate is about to take a `Sat` away.  Returns `true` only where
    /// the caller must now answer `Sat` in place of a verdict it would have
    /// given up; on a `Sat` that survives the gates it returns `false` and at
    /// most replaces the published model (the verdict is already `Sat`).
    pub(super) fn array_completion_at_exit(
        &mut self,
        result: SolverResult,
        gate_pending: bool,
        manager: &mut TermManager,
    ) -> bool {
        self.replaced_candidate = None;
        self.candidate_certified_as_printed = false;
        match result {
            SolverResult::Unknown => self.certify_sat_by_array_completion(manager),
            SolverResult::Sat if gate_pending => self.certify_sat_by_array_completion(manager),
            SolverResult::Sat => {
                // Decision (40): only the model may change here, so whether
                // a completion was installed is deliberately not the verdict.
                // Decision (48): the candidate it replaced is recorded, and
                // `Context` puts it back when the model it prints already
                // satisfies every assertion (`candidate`).
                let original = self.model.clone();
                let already = original
                    .as_ref()
                    .is_some_and(|model| self.completion_already_installed(model));
                if self.certify_sat_by_array_completion(manager)
                    && !already
                    && let Some(model) = original
                {
                    self.replaced_candidate = Some(ReplacedCandidate { model });
                }
                false
            }
            SolverResult::Unsat => false,
        }
    }

    /// Try to publish `Sat` for a quantified array goal by completing the
    /// arrays the candidate model leaves partial and certifying the result.
    ///
    /// Returns `true` only with a *certified* completion in hand, and installs
    /// it into [`Solver::model`] so `(get-model)` prints the interpretation the
    /// certificate verified rather than the partial candidate.  A `false`
    /// leaves the solver exactly as it was found.
    pub(super) fn certify_sat_by_array_completion(&mut self, manager: &mut TermManager) -> bool {
        let Some(original) = self.model.clone() else {
            return false;
        };
        // Already completed in this check (at the MBQI saturation point, see
        // `check_core`): the model carries the certified interpretation.
        if self.completion_already_installed(&original) {
            return true;
        }
        let Some((goal, completion)) = self.search_certified_completion(manager) else {
            return false;
        };
        self.install_completed_model(&goal, &completion, manager);
        // The gate every quantified `Sat` exit of `check_core` passes, run on
        // the model that will actually be published.  The certificate already
        // implies it; a disagreement means the two readings differ, and then
        // the original model is what stays.
        if self.quantified_model_refutes_ground_assertions(manager) {
            self.model = Some(original);
            self.certified_array_model = None;
            return false;
        }
        true
    }

    /// The completion tried at MBQI's saturation point, where the only fresh
    /// instances left are the unnamed-region representatives of `#P2b-60`
    /// (`check_core`).  A certified completion concludes `sat` without them.
    ///
    /// Tried for finite (bit-vector or `Bool`) index sorts only: that is
    /// where the representatives' extra index term costs the array search
    /// most (`qeq120/q0033`, seed 20260931: no answer in 60 s with them, `sat`
    /// in 12 ms before and with this), and over `Int` the constant search is
    /// all the completion has (`pinned` declines), so an attempt there would
    /// only intern its queries' terms into the running search.
    pub(super) fn certify_at_mbqi_saturation(&mut self, manager: &mut TermManager) -> bool {
        let finite = self.assertions.iter().all(|&assertion| {
            manager
                .free_vars_including_patterns(assertion)
                .into_iter()
                .filter_map(|var| manager.get(var).map(|d| d.sort))
                .filter(|&sort| is_array_sort(sort, manager))
                .all(|sort| {
                    matches!(
                        manager.sorts.get(sort).map(|s| &s.kind),
                        Some(SortKind::Array { domain, .. })
                            if matches!(
                                manager.sorts.get(*domain).map(|d| &d.kind),
                                Some(SortKind::BitVec(_) | SortKind::Bool)
                            )
                    )
                })
        });
        finite && self.certify_sat_by_array_completion(manager)
    }

    /// Whether `model` is exactly the model a completion installed earlier in
    /// this `check` (at MBQI's saturation point, then again at the exit).
    /// Equality of the whole assignment map, not a look at the arrays' values:
    /// a model that merely binds an array to a value term was not certified.
    fn completion_already_installed(&self, model: &Model) -> bool {
        self.certified_array_model
            .as_ref()
            .is_some_and(|certified| certified == model.assignments())
    }

    /// The search behind [`Self::certify_sat_by_array_completion`]: the goal
    /// and a certified completion of its arrays, or `None`.
    ///
    /// Two searches, cheapest first: one constant array per array over the
    /// pooled defaults (`#P2b-58`), then a default plus pinned points
    /// (`pinned`, decision (41)).  Every candidate is first *evaluated* on a
    /// finite sample — the assertions, each universal replaced by its
    /// instances at the goal's own points and at one point they do not name —
    /// and only a candidate the evaluation cannot refute costs a certificate
    /// query.  The evaluation only ever discards (a definite `false` at a
    /// sampled point is a counterexample the certificate would find), so it
    /// changes which candidates are *tried*, never what is accepted.
    fn search_certified_completion(
        &mut self,
        manager: &mut TermManager,
    ) -> Option<(Goal, FxHashMap<TermId, TermId>)> {
        if !self.has_array_ops || !self.has_quantifiers || self.assertions.is_empty() {
            return None;
        }
        let assignments = self.model.as_ref()?.assignments().clone();
        let assertions = self.assertions.clone();
        if !is_small_enough_to_certify(&assertions, manager) {
            return None;
        }
        let goal = Goal::build(&assertions, &assignments, manager)?;
        let logic = self.logic.clone();
        // The joint search first, exactly as before whenever it is within its
        // bound, so a goal it certified keeps the interpretation it had.
        if goal.arrays.len() <= MAX_COMPLETED_ARRAYS
            && let Some(completion) =
                self.search_goal_completion(&goal, &assignments, manager, logic.as_deref())
        {
            return Some((goal, completion));
        }
        // Then one search per independent group (`groups`), and the one
        // certificate over every assertion on the union of what they found.
        let groups = groups::array_groups(&goal.arrays, &goal.assertions, manager);
        if groups.len() < 2 {
            return None;
        }
        let mut merged: FxHashMap<TermId, TermId> = FxHashMap::default();
        for group in &groups {
            let sub_goal = goal.restricted_to(group, manager)?;
            let completion =
                self.search_goal_completion(&sub_goal, &assignments, manager, logic.as_deref())?;
            merged.extend(completion);
        }
        let mut queries = 0usize;
        goal.certificate_passes(self, &merged, manager, logic.as_deref(), &mut queries)
            .then_some((goal, merged))
    }

    /// One search over `goal`'s arrays: a constant array per array over the
    /// pooled defaults, then (over an `Int` / `Real` index) the candidate's
    /// own points over a pooled default (`infinite`), then a default plus
    /// pinned points (`pinned`).  `None` when none certifies.
    fn search_goal_completion(
        &self,
        goal: &Goal,
        assignments: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
        logic: Option<&str>,
    ) -> Option<FxHashMap<TermId, TermId>> {
        let pools = goal.default_pools(assignments, manager);
        if pools.iter().any(Vec::is_empty) {
            return None;
        }
        let points = goal.points_by_sort(manager);
        let samples = goal.sample_instances(&points, manager)?;
        let mut queries = 0usize;
        for combination in Combinations::new(&pools) {
            let mut completion: FxHashMap<TermId, TermId> = FxHashMap::default();
            for (&array, &default) in goal.arrays.iter().zip(combination.iter()) {
                let sort = manager.get(array).map(|d| d.sort)?;
                let interpretation = const_array(sort, default, manager);
                completion.insert(array, interpretation);
            }
            if self.completion_refuted_by_evaluation(goal, &completion, &samples, manager) {
                continue;
            }
            if goal.certificate_passes(self, &completion, manager, logic, &mut queries) {
                return Some(completion);
            }
            if queries >= MAX_CERTIFICATE_QUERIES {
                break;
            }
        }
        if let Some(completion) =
            self.search_candidate_point_completion(goal, assignments, &samples, manager, logic)
        {
            return Some(completion);
        }
        self.search_pinned_completion(goal, &samples, &points, manager, logic)
    }

    /// Whether `completion` (with the scalar pins) makes some assertion
    /// definitely `false` once every universal is replaced by its sampled
    /// instances (`samples`).  See [`Self::search_certified_completion`].
    fn completion_refuted_by_evaluation(
        &self,
        goal: &Goal,
        completion: &FxHashMap<TermId, TermId>,
        samples: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> bool {
        let mut interpretation = Model::new();
        for (&symbol, &value) in goal.scalar_pins.iter().chain(completion.iter()) {
            interpretation.set(symbol, value);
        }
        goal.assertions.iter().any(|&assertion| {
            let sampled = manager.substitute(assertion, samples);
            let sampled = manager.substitute(sampled, &goal.ground_values);
            matches!(
                self.eval_under_interpretation(sampled, &interpretation, manager),
                EvalOutcome::Value(EvalVal::Bool(false))
            )
        })
    }

    /// Record the certified interpretation in the published model.
    ///
    /// Each array variable is bound to the term the certificate was
    /// discharged over — `((as const A) d)`, or a `store` chain over it — so
    /// what `(get-model)` prints and what was verified are the same object.
    ///
    /// Installing *replaces* an interpretation, so every other entry whose
    /// term mentions a completed array is dropped first: a read
    /// `(select a k) ↦ v` or an atom over one describes the candidate model,
    /// not the certified one, and `Solver::model_value_in` answers
    /// `(get-value)` from an entry before it evaluates anything.  With the
    /// entry gone the evaluator reads the term through the installed value.
    fn install_completed_model(
        &mut self,
        goal: &Goal,
        completion: &FxHashMap<TermId, TermId>,
        manager: &TermManager,
    ) {
        let Some(model) = self.model.as_mut() else {
            return;
        };
        let completed: FxHashSet<TermId> = goal.arrays.iter().copied().collect();
        let mut stale: Vec<TermId> = model
            .assignments()
            .keys()
            .copied()
            .filter(|&term| !completed.contains(&term) && mentions_any(term, &completed, manager))
            .collect();
        stale.sort_unstable_by_key(|term| term.raw());
        for term in stale {
            model.remove(term);
        }
        for &array in &goal.arrays {
            if let Some(&value) = completion.get(&array) {
                model.set(array, value);
            }
        }
        self.certified_array_model = Some(model.assignments().clone());
    }
}

/// Whether `term` has one of `targets` as a sub-term.
fn mentions_any(term: TermId, targets: &FxHashSet<TermId>, manager: &TermManager) -> bool {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if targets.contains(&current) {
            return true;
        }
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    false
}

/// The array-sorted free variables of `assertions` in first-encounter order,
/// and the candidate model's value of every other free symbol it gives one.
///
/// A scalar with no value in the candidate model is left out of the pins:
/// the certificate is a *validity* query, so a symbol it does not interpret
/// can only make the query harder to discharge, never the conclusion weaker.
fn free_symbols(
    assertions: &[TermId],
    assignments: &FxHashMap<TermId, TermId>,
    manager: &TermManager,
) -> Option<(Vec<TermId>, FxHashMap<TermId, TermId>)> {
    let mut arrays: Vec<TermId> = Vec::new();
    let mut scalar_pins: FxHashMap<TermId, TermId> = FxHashMap::default();
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    for &assertion in assertions {
        let mut free: Vec<TermId> = manager.free_vars_including_patterns(assertion);
        // `free_vars_including_patterns` answers from a set, so order it by
        // the id the term manager already assigned: the attempt's combination
        // order — and hence its verdict — must not depend on a hash iteration
        // order.
        free.sort_unstable_by_key(|term| term.raw());
        for var in free {
            if !seen.insert(var) {
                continue;
            }
            let data = manager.get(var)?;
            if is_array_sort(data.sort, manager) {
                arrays.push(var);
            } else if let Some(&value) = assignments.get(&var) {
                scalar_pins.insert(var, value);
            }
        }
    }
    Some((arrays, scalar_pins))
}

impl Goal {
    /// Classify the assertion set for the completion's search, or decline.
    fn build(
        assertions: &[TermId],
        assignments: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> Option<Self> {
        let quantifiers = polarity::collect_quantifiers(assertions, manager)?;
        if quantifiers.universals.is_empty() {
            // Nothing to complete under a binder: an `exists`-only goal keeps
            // the candidate's Skolem witness, which the published-model
            // certificate (`printed`) then checks as printed.
            return None;
        }
        let (mut arrays, scalar_pins) = free_symbols(assertions, assignments, manager)?;
        // Where the completion used to decline — an uninterpreted function
        // in the goal, or more arrays than it searches over — the symbols the
        // universals never read through a binder are interpreted from the
        // candidate model instead (`#P2b-63`); every other goal is classified
        // exactly as before.
        let mut ground_values: FxHashMap<TermId, TermId> = FxHashMap::default();
        let fix_arrays = arrays.len() > MAX_COMPLETED_ARRAYS;
        if fix_arrays || declined::has_uninterpreted_application(assertions, manager) {
            let (completed, values) = declined::interpret_from_candidate(
                assertions,
                &arrays,
                assignments,
                &scalar_pins,
                fix_arrays,
                manager,
            )?;
            arrays = completed;
            ground_values = values;
        }
        // The bound is on a *group* of arrays some assertion mentions
        // together (`groups`): four arrays each under its own binder are four
        // one-array searches.
        if arrays.is_empty()
            || groups::largest_group(&arrays, assertions, manager) > MAX_COMPLETED_ARRAYS
        {
            return None;
        }
        Some(Self {
            assertions: assertions.to_vec(),
            universals: quantifiers.universals,
            arrays,
            scalar_pins,
            existentials: quantifiers.existentials,
            ground_values,
            negations: quantifiers.negations,
            hypotheses: Vec::new(),
        })
    }

    /// `query` read under the goal's hypotheses.
    fn under_hypotheses(&self, query: TermId, manager: &mut TermManager) -> TermId {
        if self.hypotheses.is_empty() {
            return query;
        }
        let mut parts = self.hypotheses.clone();
        parts.push(query);
        manager.mk_and(parts)
    }

    /// The default values tried for each array, in a deterministic order.
    ///
    /// The pool is drawn from what the goal itself makes plausible — the
    /// element-sort values the candidate model already committed to, then the
    /// element-sort literals the script spells out, then the two ends of the
    /// sort — which is the same "values the goal makes plausible" rule
    /// [`crate::mbqi::model_certify`] uses for a function default.
    fn default_pools(
        &self,
        assignments: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> Vec<Vec<TermId>> {
        let mut literals: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        for &assertion in &self.assertions {
            collect_constants(assertion, manager, &mut literals, &mut seen);
        }
        let mut model_values: Vec<TermId> = assignments.values().copied().collect();
        model_values.sort_unstable_by_key(|term| term.raw());
        model_values.dedup();

        self.arrays
            .iter()
            .map(|&array| {
                let element = element_sort(array, manager);
                let mut pool: Vec<TermId> = Vec::new();
                let mut pool_keys: FxHashSet<TermId> = FxHashSet::default();
                let push = |candidate: TermId,
                            pool: &mut Vec<TermId>,
                            keys: &mut FxHashSet<TermId>,
                            manager: &TermManager| {
                    if pool.len() >= MAX_DEFAULT_CANDIDATES {
                        return;
                    }
                    if !manager
                        .get(candidate)
                        .is_some_and(|d| Some(d.sort) == element)
                    {
                        return;
                    }
                    if keys.insert(candidate) {
                        pool.push(candidate);
                    }
                };
                for &value in &model_values {
                    push(value, &mut pool, &mut pool_keys, manager);
                }
                for &literal in &literals {
                    push(literal, &mut pool, &mut pool_keys, manager);
                }
                if let Some(element) = element {
                    for extreme in sort_extremes(element, manager) {
                        push(extreme, &mut pool, &mut pool_keys, manager);
                    }
                }
                pool
            })
            .collect()
    }

    /// The quantifier-free certificate for one completion.
    ///
    /// Every step is a validity query discharged by refuting its negation; see
    /// the module docs for why only that direction is taken.  A closed
    /// instance is *evaluated* exactly instead of queried: `true` is what the
    /// query's `Unsat` would conclude and `false` what its `Sat` would, so the
    /// evaluation changes which queries are spent, never what is accepted.
    fn certificate_passes(
        &self,
        solver: &Solver,
        completion: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
        logic: Option<&str>,
        queries: &mut usize,
    ) -> bool {
        let substitution = self.full_substitution(completion);
        let dual_of: FxHashMap<TermId, TermId> = self
            .negations
            .iter()
            .map(|&(original, dual)| (dual, original))
            .collect();
        let extra_points = completion_points(completion, manager);

        // Step 1: every maximal `forall` is certified *true* under the
        // completion where it can be (and one that cannot is left
        // undetermined for step 2, `undetermined`).  The bound variables become fresh reserved constants, so
        // the query is quantifier-free and the bound name can never be
        // confused with a free one of the same name — which is exactly the
        // collision shape of `rk8/atk/f5_binder_collide_index.smt2`.  The dual
        // `∀x⃗. ¬φ` of a negated `exists` certifies that `exists` *false*
        // (`polarity`); where it does not, the `exists` itself may still be
        // certified true by a witness, and the assertions then decide.
        let mut truths: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (position, &universal) in self.universals.iter().enumerate() {
            let original = dual_of.get(&universal).copied();
            if self.universal_certified(
                solver,
                universal,
                position,
                &substitution,
                manager,
                logic,
                queries,
            ) {
                truths.insert(universal, manager.mk_true());
                if let Some(original) = original {
                    truths.insert(original, manager.mk_false());
                }
                continue;
            }
            let Some(original) = original else {
                truths.insert(universal, undetermined(manager, "u", position));
                continue;
            };
            if !self.certify_existential(
                solver,
                original,
                position,
                &substitution,
                &extra_points,
                manager,
                logic,
                queries,
            ) {
                truths.insert(original, undetermined(manager, "u", position));
                continue;
            }
            truths.insert(original, manager.mk_true());
        }
        // Step 1b: every maximal `exists` must be certified true by a witness
        // (`#P2b-63`, `declined`); the dual `∃x⃗. ¬φ` of a negated `forall`
        // certifies that `forall` false, and where it does not, the `forall`
        // itself may still be certified true.
        for (position, &existential) in self.existentials.iter().enumerate() {
            let original = dual_of.get(&existential).copied();
            if self.certify_existential(
                solver,
                existential,
                position,
                &substitution,
                &extra_points,
                manager,
                logic,
                queries,
            ) {
                truths.insert(existential, manager.mk_true());
                if let Some(original) = original {
                    truths.insert(original, manager.mk_false());
                }
                continue;
            }
            let Some(original) = original else {
                truths.insert(existential, undetermined(manager, "e", position));
                continue;
            };
            if !self.universal_certified(
                solver,
                original,
                self.universals.len() + position,
                &substitution,
                manager,
                logic,
                queries,
            ) {
                truths.insert(original, undetermined(manager, "e", position));
                continue;
            }
            truths.insert(original, manager.mk_true());
        }

        // Step 2: the assertions themselves.  Evaluated first; where every
        // one evaluates, no query is spent.  Otherwise one query:
        // `(or ¬A₁ … ¬Aₙ)` is `Unsat` exactly when every `Aᵢ` holds under the
        // completion.
        let mut disjuncts: Vec<TermId> = Vec::with_capacity(self.assertions.len());
        let mut all_true = true;
        for &assertion in &self.assertions {
            let grounded = manager.substitute(assertion, &truths);
            let grounded = manager.substitute(grounded, &substitution);
            match evaluate_closed(solver, grounded, manager) {
                Some(false) => return false,
                Some(true) => {}
                None => all_true = false,
            }
            disjuncts.push(manager.mk_not(grounded));
        }
        if all_true {
            return true;
        }
        let refutation = manager.mk_or(disjuncts);
        let refutation = self.under_hypotheses(refutation, manager);
        matches!(
            run_query(refutation, manager, logic, queries),
            Some(SolverResult::Unsat)
        )
    }

    /// The scalar pins, the fixed ground values and `completion`, as one map.
    fn full_substitution(
        &self,
        completion: &FxHashMap<TermId, TermId>,
    ) -> FxHashMap<TermId, TermId> {
        let mut substitution = self.scalar_pins.clone();
        substitution.extend(self.ground_values.iter().map(|(&k, &v)| (k, v)));
        substitution.extend(completion.iter().map(|(&k, &v)| (k, v)));
        substitution
    }

    /// Whether the universal `universal` (listed at `position`) is certified
    /// true under `substitution`: by the point split over an infinite index
    /// sort where it applies (`split`), else by the validity query.
    #[allow(clippy::too_many_arguments)]
    fn universal_certified(
        &self,
        solver: &Solver,
        universal: TermId,
        position: usize,
        substitution: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
        logic: Option<&str>,
        queries: &mut usize,
    ) -> bool {
        if let Some(verdict) = split::universal_by_points(
            solver,
            universal,
            position,
            substitution,
            &self.hypotheses,
            manager,
            logic,
            queries,
        ) {
            return verdict;
        }
        let Some(body) = peel_universal(universal, position, manager) else {
            return false;
        };
        let body = manager.substitute(body, substitution);
        let negated = manager.mk_not(body);
        let negated = self.under_hypotheses(negated, manager);
        matches!(
            run_query(negated, manager, logic, queries),
            Some(SolverResult::Unsat)
        )
    }
}

/// A fresh Boolean constant standing for a quantifier sub-term neither of
/// whose values could be certified.  The assertions' validity query then has
/// to hold for *both* of its values, so an assertion that does not depend on
/// the sub-term — `(=> p (forall …))` under `p = false` — still certifies, and
/// one that does cannot: sound for the same reason every other symbol the
/// query leaves free is.
fn undetermined(manager: &mut TermManager, kind: &str, position: usize) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    manager.mk_var(
        &reserved_name("qunk", &format!("{kind}{position}")),
        bool_sort,
    )
}

/// The value of a closed term, read exactly (every `select` over its
/// `store` chain): what `(get-value)` prints for a term of the printed model
/// (`context::model_fmt::printed_check`).  `None` when the evaluator cannot
/// fold it.
pub(crate) fn evaluate_closed_value(
    solver: &Solver,
    term: TermId,
    manager: &mut TermManager,
) -> Option<crate::solver::EvalVal> {
    let term = array_values::fold_array_value_equalities(term, manager);
    match solver.eval_under_interpretation(term, &Model::new(), manager) {
        EvalOutcome::Value(value) => Some(value),
        _ => None,
    }
}

/// [`evaluate_closed_value`] without the array-value fold: the term read
/// exactly, through a shared borrow, so it interns nothing.  The printed-model
/// net's value reading (`context::model_fmt::printed_eval::ground_values`)
/// calls it on every scalar sub-term of a check's assertions, and the term
/// arena a later check's search reads must not move for it.
pub(crate) fn evaluate_closed_value_pure(
    solver: &Solver,
    term: TermId,
    manager: &TermManager,
) -> Option<crate::solver::EvalVal> {
    match solver.eval_under_interpretation(term, &Model::new(), manager) {
        EvalOutcome::Value(value) => Some(value),
        _ => None,
    }
}

/// The exact value of a closed Boolean term, or `None` when the evaluator
/// cannot fold it to one (a symbol it does not interpret, an array equality).
pub(crate) fn evaluate_closed(
    solver: &Solver,
    term: TermId,
    manager: &mut TermManager,
) -> Option<bool> {
    let term = array_values::fold_array_value_equalities(term, manager);
    match solver.eval_under_interpretation(term, &Model::new(), manager) {
        EvalOutcome::Value(EvalVal::Bool(value)) => Some(value),
        _ => None,
    }
}

/// Every literal `store` index of the completion's interpretations, by sort:
/// the points the interpretation names, which a witness search must try
/// (an `exists` witnessed where the candidate's Skolem constant was read).
fn completion_points(
    completion: &FxHashMap<TermId, TermId>,
    manager: &TermManager,
) -> FxHashMap<SortId, Vec<TermId>> {
    let mut out: FxHashMap<SortId, Vec<TermId>> = FxHashMap::default();
    let mut values: Vec<TermId> = completion.values().copied().collect();
    values.sort_unstable_by_key(|term| term.raw());
    for value in values {
        let mut current = value;
        while let Some(TermKind::Store(inner, index, _)) = manager.get(current).map(|d| &d.kind) {
            if is_literal(*index, manager)
                && let Some(sort) = manager.get(*index).map(|d| d.sort)
            {
                let list = out.entry(sort).or_default();
                if !list.contains(index) {
                    list.push(*index);
                }
            }
            current = *inner;
        }
    }
    out
}

/// Run one quantifier-free certificate query, or `None` when the query budget
/// is spent.
///
/// The sub-solver is budgeted deterministically and carries no clock, so the
/// certificate reproduces on an idle and on a loaded machine alike.  It cannot
/// re-enter this module: the goal handed to it is quantifier-free by
/// construction, and the entry point above requires a quantifier.
fn run_query(
    goal: TermId,
    manager: &mut TermManager,
    logic: Option<&str>,
    queries: &mut usize,
) -> Option<SolverResult> {
    if *queries >= MAX_CERTIFICATE_QUERIES {
        return None;
    }
    *queries += 1;
    // An equality between two literal array values is decided exactly here
    // rather than left to extensionality in the sub-solver (`array_values`).
    let goal = array_values::fold_array_value_equalities(goal, manager);
    let mut solver = Solver::new();
    solver.set_logic(logic.unwrap_or("ALL"));
    solver.config.max_conflicts = CERTIFICATE_CONFLICT_BUDGET;
    solver.assert(goal, manager);
    Some(solver.check(manager))
}

/// The body of a chain of consecutive `forall`s, with every bound variable
/// replaced by a fresh reserved constant, or `None` when the sub-term is not a
/// universal this module handles.
///
/// A chain is peeled rather than a single binder, so
/// `(forall ((k …)) (forall ((k …)) …))` — decision (36)'s "the collision shape
/// with the array name bound **twice**" — is one multi-binder universal and not
/// a nested one.  A quantifier that survives the peel is a genuine
/// alternation and is declined, because the certificate below would not be
/// quantifier-free.
fn peel_universal(term: TermId, position: usize, manager: &mut TermManager) -> Option<TermId> {
    peel_universal_with_vars(term, position, manager).map(|(body, _)| body)
}

/// [`peel_universal`], also returning the fresh constants the bound variables
/// became, in binder order, so a caller can instantiate the body at chosen
/// points (`pinned::Goal::fill_query`).
fn peel_universal_with_vars(
    term: TermId,
    position: usize,
    manager: &mut TermManager,
) -> Option<(TermId, Vec<TermId>)> {
    let mut current = term;
    let mut bound: Vec<(Spur, SortId)> = Vec::new();
    loop {
        let kind = manager.get(current).map(|d| d.kind.clone())?;
        match kind {
            TermKind::Forall { vars, body, .. } => {
                bound.extend(vars.iter().copied());
                current = body;
            }
            _ => break,
        }
    }
    if contains_quantifier(current, manager) {
        return None;
    }
    let mut rename: FxHashMap<TermId, TermId> = FxHashMap::default();
    let mut fresh_vars: Vec<TermId> = Vec::with_capacity(bound.len());
    for (index, (name, sort)) in bound.into_iter().enumerate() {
        let bound_name = manager.resolve_str(name).to_string();
        let bound_var = manager.mk_var(&bound_name, sort);
        let fresh = manager.mk_var(
            &reserved_name("qcert", &format!("{position}_{index}")),
            sort,
        );
        // A chain that binds one name twice keeps the innermost binding,
        // which is the one the body sees; the constant list keeps one entry
        // per *distinct* variable so an instantiation never assigns it twice.
        match rename.insert(bound_var, fresh) {
            None => fresh_vars.push(fresh),
            Some(previous) => {
                if let Some(slot) = fresh_vars.iter_mut().find(|var| **var == previous) {
                    *slot = fresh;
                }
            }
        }
    }
    Some((manager.substitute(current, &rename), fresh_vars))
}

/// Every constant (literal) sub-term of `term`, in first-encounter order.
fn collect_constants(
    term: TermId,
    manager: &TermManager,
    out: &mut Vec<TermId>,
    seen: &mut FxHashSet<TermId>,
) {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    let mut found: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if matches!(
            data.kind,
            TermKind::BitVecConst { .. }
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::True
                | TermKind::False
        ) {
            found.push(current);
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    found.sort_unstable_by_key(|term| term.raw());
    for term in found {
        if seen.insert(term) {
            out.push(term);
        }
    }
}

/// Whether `term` carries a quantifier anywhere in it.
fn contains_quantifier(term: TermId, manager: &TermManager) -> bool {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if matches!(data.kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
            return true;
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    false
}

/// Whether the goal is one an attempt can afford to try: at most
/// [`MAX_GOAL_NODES`] assertion-DAG nodes.
///
/// An uninterpreted application used to decline here as well (it survives
/// into the validity query, where the sub-solver may interpret it any way it
/// likes, so a satisfiable-but-not-valid goal never certifies).  It is now
/// interpreted from the candidate model where it is ground, and declines in
/// `Goal::build` where it is not (`#P2b-63`, `declined`).
fn is_small_enough_to_certify(assertions: &[TermId], manager: &TermManager) -> bool {
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = assertions.to_vec();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        if visited.len() > MAX_GOAL_NODES {
            return false;
        }
        let Some(data) = manager.get(current) else {
            return false;
        };
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    true
}

/// Whether `sort` is an array sort.
fn is_array_sort(sort: SortId, manager: &TermManager) -> bool {
    manager
        .sorts
        .get(sort)
        .is_some_and(|s| matches!(s.kind, SortKind::Array { .. }))
}

/// The element sort of an array-sorted term.
fn element_sort(term: TermId, manager: &TermManager) -> Option<SortId> {
    let data = manager.get(term)?;
    match manager.sorts.get(data.sort)?.kind {
        SortKind::Array { range, .. } => Some(range),
        _ => None,
    }
}

/// The two ends of a sort, as a last resort for the default pool.
fn sort_extremes(sort: SortId, manager: &mut TermManager) -> Vec<TermId> {
    let kind = manager.sorts.get(sort).map(|s| s.kind.clone());
    match kind {
        Some(SortKind::BitVec(width)) => {
            let zero = manager.mk_bitvec(num_bigint::BigInt::from(0u8), width);
            let ones = manager.mk_bitvec(
                (num_bigint::BigInt::from(1u8) << width) - num_bigint::BigInt::from(1u8),
                width,
            );
            vec![zero, ones]
        }
        Some(SortKind::Bool) => vec![manager.mk_false(), manager.mk_true()],
        _ => Vec::new(),
    }
}

/// A bounded product over per-position pools (the per-array default pools,
/// or the per-variable instantiation points of `pinned::Goal::fill_query`).
struct Combinations<'a> {
    pools: &'a [Vec<TermId>],
    cursor: Vec<usize>,
    emitted: usize,
    cap: usize,
    done: bool,
}

impl<'a> Combinations<'a> {
    /// The product over `pools`, at most [`MAX_COMBINATIONS`] items.
    fn new(pools: &'a [Vec<TermId>]) -> Self {
        Self::with_cap(pools, MAX_COMBINATIONS)
    }

    /// The product over `pools`, at most `cap` items.
    fn with_cap(pools: &'a [Vec<TermId>], cap: usize) -> Self {
        Self {
            pools,
            cursor: vec![0; pools.len()],
            emitted: 0,
            cap,
            done: pools.is_empty(),
        }
    }
}

impl Iterator for Combinations<'_> {
    type Item = Vec<TermId>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.emitted >= self.cap {
            return None;
        }
        let mut item: Vec<TermId> = Vec::with_capacity(self.pools.len());
        for (pool, &index) in self.pools.iter().zip(self.cursor.iter()) {
            item.push(*pool.get(index)?);
        }
        self.emitted += 1;
        // Odometer, least significant position first.
        let mut position = self.cursor.len();
        loop {
            if position == 0 {
                self.done = true;
                break;
            }
            position -= 1;
            let limit = self.pools.get(position).map_or(0, Vec::len);
            let next = self.cursor.get_mut(position)?;
            *next += 1;
            if *next < limit {
                break;
            }
            *next = 0;
        }
        Some(item)
    }
}

/// The interpretation a completed array is published under: `((as const A) d)`.
pub(super) fn const_array(
    array_sort: SortId,
    default: TermId,
    manager: &mut TermManager,
) -> TermId {
    manager.mk_apply(CONST_ARRAY_FUNC, [default], array_sort)
}

/// Whether `term` is a closed array *value* this module installs: the array
/// constant [`const_array`] builds over a literal default, or a `store` chain
/// over one whose every index and stored value is a literal.
///
/// The renderer prints such a term verbatim and the model evaluator reads a
/// `select` through it; nothing else in the solver assigns an array a value
/// term, so the predicate is exactly "an interpretation the certificate
/// verified".
pub(crate) fn is_array_value(term: TermId, manager: &TermManager) -> bool {
    let mut current = term;
    loop {
        let Some(data) = manager.get(current) else {
            return false;
        };
        match &data.kind {
            TermKind::Store(inner, index, value) => {
                if !is_literal(*index, manager) || !is_literal(*value, manager) {
                    return false;
                }
                current = *inner;
            }
            _ => {
                return super::array_axioms::const_array_default(current, manager)
                    .is_some_and(|default| is_literal(default, manager));
            }
        }
    }
}

/// Whether `term` is a literal value of a scalar sort.
fn is_literal(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|data| {
        matches!(
            data.kind,
            TermKind::BitVecConst { .. }
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::True
                | TermKind::False
        )
    })
}

/// Whether `term` is the array constant [`const_array`] builds.
///
/// The model formatter asks this before it prefers an explicit model entry
/// over the congruence class it would otherwise render an array from; see
/// `Context::array_model_value`.
pub(crate) fn is_const_array(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|data| match &data.kind {
        TermKind::Apply { func, args } => {
            args.len() == 1 && manager.resolve_str(*func) == CONST_ARRAY_FUNC
        }
        _ => false,
    })
}
