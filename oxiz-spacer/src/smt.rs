//! SMT solver integration for Spacer
//!
//! Provides incremental SMT queries, model extraction, and interpolation support.
//!
//! Reference: Z3's `muz/spacer/spacer_context.cpp` solver integration

use crate::chc::{ChcSystem, PredId, Rule};
use crate::frames::FrameManager;
use crate::interp::Interpolator;
use oxiz_core::ast::TermKind;
use oxiz_core::{TermId, TermManager};
use oxiz_solver::{Solver, SolverResult};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{debug, trace};

/// Build the frame formula `F_level(pred)`: the conjunction of every
/// active lemma for `pred` at or above `level`, or `true` if there are
/// none. Shared by any caller that needs to check consecution/
/// inductiveness for a specific `(pred, level)` (e.g. [`SmtSolver::
/// is_lemma_inductive`] callers in `pdr.rs` and `parallel.rs`).
pub(crate) fn build_frame_formula(
    terms: &mut TermManager,
    frames: &FrameManager,
    pred: PredId,
    level: u32,
) -> TermId {
    match frames.get(pred) {
        Some(pred_frames) => {
            let lemmas: Vec<TermId> = pred_frames
                .lemmas_geq_level(level)
                .map(|l| l.formula)
                .collect();
            if lemmas.is_empty() {
                terms.mk_true()
            } else if lemmas.len() == 1 {
                lemmas[0]
            } else {
                terms.mk_and(lemmas)
            }
        }
        None => terms.mk_true(),
    }
}

/// Build the canonical *current-state* variables for a predicate.
///
/// Spacer normalizes every state formula (init facts, POB cubes, lemmas,
/// frames) so that a predicate `P`'s arguments are represented by the fixed
/// variables `__sp_c#<pred>#<j>`.  This gives a single, consistent variable
/// namespace across rules that use different local argument names, which is
/// what makes primed-state renaming (consecution) and reachability queries
/// sound.
pub(crate) fn canon_cur_vars(
    terms: &mut TermManager,
    system: &ChcSystem,
    pred: PredId,
) -> SmallVec<[TermId; 4]> {
    match system.get_predicate(pred) {
        Some(decl) => decl
            .params
            .iter()
            .enumerate()
            .map(|(j, &sort)| {
                let name = format!("__sp_c#{}#{}", pred.raw(), j);
                terms.mk_var(&name, sort)
            })
            .collect(),
        None => SmallVec::new(),
    }
}

/// Build a substitution mapping each `arg` term to the corresponding `target`.
///
/// Returns `None` unless every `arg` is a plain (interned) variable, because
/// substituting a compound term is not a sound state renaming.  This is the
/// guard that keeps Spacer restricted to the linear fragment it can handle
/// soundly; callers treat `None` as "unsupported, do not claim a result".
pub(crate) fn var_subst(
    terms: &TermManager,
    args: &[TermId],
    targets: &[TermId],
) -> Option<FxHashMap<TermId, TermId>> {
    if args.len() != targets.len() {
        return None;
    }
    let mut map = FxHashMap::default();
    for (&arg, &target) in args.iter().zip(targets.iter()) {
        match terms.get(arg).map(|d| &d.kind) {
            Some(TermKind::Var(_)) => {}
            _ => return None,
        }
        map.insert(arg, target);
    }
    Some(map)
}

/// Assert `formula` onto `solver`, splitting a top-level `And` into separate
/// assertions.  See [`SmtSolver::assert`] for why the split matters.
///
/// `And`-nesting is attacker-controlled (it comes straight from parsed
/// CHC/SMT input), and this function returns `()`, so there is nowhere to
/// report a depth failure: the flattening therefore uses an explicit heap
/// stack ([`crate::walk::flatten_conjuncts`]) rather than recursion.
fn assert_flat(solver: &mut Solver, terms: &mut TermManager, formula: TermId) {
    for conjunct in crate::walk::flatten_conjuncts(terms, formula) {
        solver.assert(conjunct, terms);
    }
}

/// Errors from SMT queries
#[derive(Error, Debug)]
pub enum SmtError {
    /// Solver returned unknown
    #[error("solver returned unknown")]
    Unknown,
    /// Internal solver error
    #[error("internal solver error: {0}")]
    Internal(String),
}

/// SMT solver interface for Spacer.
///
/// Uses a single canonical `TermManager` (borrowed from the caller) so that
/// all `TermId`s produced by e.g. `mk_not` / `mk_and` are valid when
/// asserted.  The underlying `Solver` is kept separate and accepts the same
/// manager on every call.
pub struct SmtSolver<'a> {
    /// The underlying SAT/SMT solver (does NOT own a TermManager)
    solver: Solver,
    /// Canonical term manager — the same arena used by the CHC system
    terms: &'a mut TermManager,
    /// CHC system
    system: &'a ChcSystem,
    /// Current assertion level (for push/pop tracking)
    level: u32,
    /// Cache of predicate frame formulas
    frame_cache: FxHashMap<(PredId, u32), TermId>,
    /// Interpolator for Craig interpolation
    interpolator: Interpolator,
    /// Statistics
    stats: SmtStats,
}

/// Statistics for SMT queries
#[derive(Debug, Clone, Default)]
pub struct SmtStats {
    /// Number of check-sat queries
    pub num_queries: u64,
    /// Number of SAT results
    pub num_sat: u64,
    /// Number of UNSAT results
    pub num_unsat: u64,
    /// Number of UNKNOWN results
    pub num_unknown: u64,
    /// Number of push operations
    pub num_push: u64,
    /// Number of pop operations
    pub num_pop: u64,
    /// Total time spent in check-sat (microseconds)
    pub total_check_sat_time_us: u64,
    /// Total time spent in model extraction (microseconds)
    pub total_model_extraction_time_us: u64,
    /// Frame cache hits
    pub frame_cache_hits: u64,
    /// Frame cache misses
    pub frame_cache_misses: u64,
}

impl<'a> SmtSolver<'a> {
    /// Create a new SMT solver for Spacer.
    ///
    /// `terms` is the **single canonical** arena — all `TermId`s produced by
    /// callers and by methods on this struct must belong to this arena.
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        let mut solver = Solver::new();
        solver.set_logic("HORN");

        Self {
            solver,
            terms,
            system,
            level: 0,
            frame_cache: FxHashMap::default(),
            interpolator: Interpolator::new(),
            stats: SmtStats::default(),
        }
    }

    /// Borrow the canonical term manager.
    ///
    /// Every `TermId` built via this reference is valid for use in `assert`.
    #[inline]
    pub fn terms(&mut self) -> &mut TermManager {
        self.terms
    }

    /// Push a solver context
    pub fn push(&mut self) {
        self.solver.push();
        self.level += 1;
        self.stats.num_push += 1;
        trace!("SMT push to level {}", self.level);
    }

    /// Pop a solver context
    pub fn pop(&mut self) {
        if self.level > 0 {
            self.solver.pop();
            self.level -= 1;
            self.stats.num_pop += 1;
            trace!("SMT pop to level {}", self.level);
        }
    }

    /// Assert a formula.
    ///
    /// `formula` **must** be a `TermId` that belongs to the canonical
    /// `TermManager` passed to `SmtSolver::new`.
    pub fn assert(&mut self, formula: TermId) {
        // Assert each top-level conjunct separately.  This is logically
        // identical to asserting the conjunction, but avoids a solver
        // incompleteness where a single `And` term containing disequalities
        // (`¬(x = k)`) can be answered SAT when the individually-asserted
        // conjuncts are correctly UNSAT.  Spacer's blocking lemmas are exactly
        // such disequalities, so this normalization is essential for soundness.
        //
        // The flattening walks an explicit heap stack
        // ([`crate::walk::flatten_conjuncts`]) instead of recursing: the
        // `And`-nesting depth comes from parsed input, and `assert` returns
        // `()`, so a depth limit would have no honest way to report that it
        // dropped assertions.
        let conjuncts = crate::walk::flatten_conjuncts(self.terms, formula);
        for conjunct in conjuncts {
            self.solver.assert(conjunct, self.terms);
        }
        trace!("SMT assert formula");
    }

    /// Check satisfiability
    pub fn check_sat(&mut self) -> Result<bool, SmtError> {
        use oxiz_time::Instant;

        self.stats.num_queries += 1;
        let start = Instant::now();
        let result = self.solver.check(self.terms);
        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.total_check_sat_time_us += elapsed;

        match result {
            SolverResult::Sat => {
                self.stats.num_sat += 1;
                debug!("SMT query: SAT ({}µs)", elapsed);
                Ok(true)
            }
            SolverResult::Unsat => {
                self.stats.num_unsat += 1;
                debug!("SMT query: UNSAT ({}µs)", elapsed);
                Ok(false)
            }
            SolverResult::Unknown => {
                self.stats.num_unknown += 1;
                debug!("SMT query: UNKNOWN ({}µs)", elapsed);
                Err(SmtError::Unknown)
            }
        }
    }

    /// Check satisfiability of the conjunction of `assertions` on a **fresh**
    /// underlying solver.
    ///
    /// The backend's incremental `push`/`pop` interface can return a stale
    /// (wrong) result on a solver that has already answered and rolled back a
    /// query.  Any independent query whose result must be trusted for
    /// soundness (e.g. per-rule consecution) therefore runs on its own fresh
    /// solver instance instead of reusing this one across `pop`.
    ///
    /// Top-level `And` conjuncts are asserted separately (see [`Self::assert`]).
    pub fn check_sat_fresh(&mut self, assertions: &[TermId]) -> Result<bool, SmtError> {
        let mut solver = Solver::new();
        solver.set_logic("HORN");
        for &a in assertions {
            assert_flat(&mut solver, self.terms, a);
        }

        self.stats.num_queries += 1;
        match solver.check(self.terms) {
            SolverResult::Sat => {
                self.stats.num_sat += 1;
                Ok(true)
            }
            SolverResult::Unsat => {
                self.stats.num_unsat += 1;
                Ok(false)
            }
            SolverResult::Unknown => {
                self.stats.num_unknown += 1;
                Err(SmtError::Unknown)
            }
        }
    }

    /// Evaluate a term in the current model (only valid after a SAT result).
    ///
    /// Returns `None` if no model is available or the last result was not SAT.
    pub fn eval_in_model(&mut self, term: TermId) -> Option<TermId> {
        let model = self.solver.model()?;
        Some(model.eval(term, self.terms))
    }

    /// Check if a state is reachable: F_level(pred) ∧ state is SAT?
    pub fn is_state_reachable(
        &mut self,
        pred: PredId,
        state: TermId,
        level: u32,
        frame_formula: TermId,
    ) -> Result<Option<Model>, SmtError> {
        // Check if frame formula is cached
        if self.frame_cache.contains_key(&(pred, level)) {
            self.stats.frame_cache_hits += 1;
            trace!("Frame cache hit for predicate {:?} level {}", pred, level);
        } else {
            self.stats.frame_cache_misses += 1;
            self.frame_cache.insert((pred, level), frame_formula);
            trace!("Frame cache miss for predicate {:?} level {}", pred, level);
        }

        self.push();

        // Assert frame formula: F_level(pred)
        self.assert(frame_formula);

        // Assert the state
        self.assert(state);

        let is_sat = self.check_sat()?;
        let result = if is_sat {
            // Extract model
            Some(self.extract_model(pred))
        } else {
            None
        };

        self.pop();
        Ok(result)
    }

    /// Check if a transition is feasible:
    /// F_level(body_preds) ∧ transition_constraint ∧ post is SAT?
    pub fn is_transition_feasible(
        &mut self,
        rule: &Rule,
        body_frames: &[(PredId, TermId)],
        post: TermId,
    ) -> Result<Option<Model>, SmtError> {
        self.push();

        // Assert frames for body predicates
        for (_pred, frame_formula) in body_frames {
            self.assert(*frame_formula);
        }

        // Assert transition constraint
        self.assert(rule.body.constraint);

        // Assert post-condition
        self.assert(post);

        let is_sat = self.check_sat()?;
        let result = if is_sat {
            Some(self.extract_model(PredId::new(0))) // Will be refined
        } else {
            None
        };

        self.pop();
        Ok(result)
    }

    /// Check whether `lemma` (expressed over predicate `pred`'s canonical
    /// current-state variables) is inductive relative to the frame
    /// `frame_formula = F_level`.
    ///
    /// Consecution is checked **per rule** (the transition relation is a
    /// disjunction of rules, so each rule must independently preserve the
    /// lemma) with proper primed-state renaming:
    ///
    /// For each self-loop rule `P(body_args) ∧ C ⇒ P(head_args)` we test
    /// satisfiability of
    ///   `F_level ∧ C[body_args ↦ C_vars] ∧ ¬lemma[C_vars ↦ head_args]`.
    /// `head_args` are the rule's next-state variables, so `¬lemma'` ranges
    /// over the next state while `F_level` ranges over the current state.  If
    /// any rule makes this SAT the lemma is **not** inductive and cannot be
    /// pushed.  UNSAT for every rule ⇒ inductive.
    ///
    /// Only the single-predicate linear (self-loop) fragment is supported
    /// soundly; any rule outside it makes this return `Ok(false)`
    /// (conservatively "not inductive"), which merely prevents a fixpoint
    /// rather than fabricating one.
    pub fn is_lemma_inductive(
        &mut self,
        pred: PredId,
        lemma: TermId,
        _level: u32,
        frame_formula: TermId,
    ) -> Result<bool, SmtError> {
        // Canonical current-state variables for `pred`.
        let cur = canon_cur_vars(self.terms, self.system, pred);

        // Collect the (body_args, head_args, constraint, linear?) of every rule
        // whose head is `pred`, cloned up-front to avoid borrowing the system
        // while mutating `self`.
        #[allow(clippy::type_complexity)]
        let rules: Vec<(SmallVec<[TermId; 4]>, SmallVec<[TermId; 4]>, TermId, bool)> = self
            .system
            .rules_by_head(pred)
            .map(|rule| {
                let body_args = rule
                    .body
                    .predicates
                    .first()
                    .map(|app| app.args.clone())
                    .unwrap_or_default();
                let head_args = rule
                    .head
                    .as_predicate()
                    .map(|app| app.args.clone())
                    .unwrap_or_default();
                let linear = rule.body.predicates.len() <= 1
                    && rule.body.predicates.iter().all(|app| app.pred == pred);
                (body_args, head_args, rule.body.constraint, linear)
            })
            .collect();

        for (body_args, head_args, constraint, linear) in rules {
            if !linear {
                // Non-linear / cross-predicate rule: cannot check soundly here.
                return Ok(false);
            }

            // Renaming of the lemma to the next state (head args).
            let Some(next_subst) = var_subst(self.terms, &cur, &head_args) else {
                return Ok(false);
            };
            let lemma_next = self.terms.substitute(lemma, &next_subst);
            let not_lemma_next = self.terms.mk_not(lemma_next);

            // Transition constraint in canonical current-state space.
            let trans = if body_args.is_empty() {
                // Init rule (no body predicate): the constraint already defines
                // the head state; the current state is unconstrained.
                constraint
            } else {
                let Some(cur_subst) = var_subst(self.terms, &body_args, &cur) else {
                    return Ok(false);
                };
                self.terms.substitute(constraint, &cur_subst)
            };

            let is_sat = self.check_sat_fresh(&[frame_formula, trans, not_lemma_next])?;

            if is_sat {
                // A transition out of F_level violates lemma' ⇒ not inductive.
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if state is blocked by a lemma:
    /// lemma ∧ state is UNSAT?
    pub fn is_blocked_by(&mut self, lemma: TermId, state: TermId) -> Result<bool, SmtError> {
        self.push();

        self.assert(lemma);
        self.assert(state);

        let is_sat = self.check_sat()?;
        let is_blocked = !is_sat;

        self.pop();
        Ok(is_blocked)
    }

    /// Extract model from current satisfying assignment.
    ///
    /// Evaluates the predicate's canonical *current-state* variables
    /// (`__sp_c#<pred>#<j>`, see [`canon_cur_vars`]) -- the exact names
    /// every asserted formula (init facts, transition constraints,
    /// lemmas, POB cubes) actually uses for this predicate's arguments.
    ///
    /// This previously invented its own `"{pred_name}_{idx}"` variable
    /// names via a fresh `mk_var` call. Those names never occurred in any
    /// asserted formula, so evaluating them in the model just read back
    /// the solver's arbitrary default for a totally unconstrained
    /// variable -- not the real value of the predicate's argument in the
    /// reachable state the model actually describes.
    fn extract_model(&mut self, pred: PredId) -> Model {
        use oxiz_time::Instant;

        let start = Instant::now();

        let cur_vars = canon_cur_vars(self.terms, self.system, pred);
        let model = match self.solver.model() {
            Some(m) => Model {
                assignments: cur_vars.iter().map(|&v| m.eval(v, self.terms)).collect(),
            },
            None => Model {
                assignments: Vec::new(),
            },
        };

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.total_model_extraction_time_us += elapsed;
        trace!("Model extraction: {}µs", elapsed);

        model
    }

    /// Generalize a cube using model-based projection
    /// Given a model M that satisfies cube C, find a minimal generalization
    pub fn generalize_cube(
        &mut self,
        cube: &[TermId],
        _pred: PredId,
        _model: &Model,
    ) -> Vec<TermId> {
        // MBP: Model-Based Projection
        // Try to drop literals from the cube while maintaining unsatisfiability

        let mut generalized = cube.to_vec();
        let mut i = 0;

        while i < generalized.len() {
            // Try removing literal i
            let removed = generalized.remove(i);

            // Check if the remaining cube is still sufficient
            self.push();

            // Assert all remaining literals
            let remaining = generalized.clone();
            for &lit in &remaining {
                self.assert(lit);
            }

            // Check if the cube is still unsatisfiable with the bad state
            // (This is a simplified version - real MBP is more sophisticated)
            let is_sat = self.check_sat().unwrap_or(true);

            self.pop();

            if is_sat {
                // Need this literal, put it back
                generalized.insert(i, removed);
                i += 1;
            }
            // else: successfully removed, continue with same index
        }

        generalized
    }

    /// Compute interpolant between A and B where A ∧ B is UNSAT
    /// Returns a formula I such that:
    /// - A => I
    /// - I ∧ B is UNSAT
    /// - I only uses common variables
    pub fn interpolate(&mut self, a: TermId, b: TermId) -> Result<TermId, SmtError> {
        // Check that A ∧ B is UNSAT
        self.push();
        self.assert(a);
        self.assert(b);
        let is_sat = self.check_sat()?;
        self.pop();

        if is_sat {
            return Err(SmtError::Internal(
                "Cannot interpolate SAT formula".to_string(),
            ));
        }

        // Use the Interpolator to compute Craig interpolant
        // This projects A onto common variables with B
        let interp = self
            .interpolator
            .interpolate(self.terms, a, b)
            .map_err(|e| SmtError::Internal(format!("Interpolation failed: {}", e)))?;

        Ok(interp.formula)
    }

    /// Get the current statistics
    pub fn stats(&self) -> &SmtStats {
        &self.stats
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.solver.reset();
        self.level = 0;
        self.frame_cache.clear();
        self.stats = SmtStats::default();
    }
}

/// A model (satisfying assignment)
#[derive(Debug, Clone)]
pub struct Model {
    /// Variable assignments (term values)
    pub assignments: Vec<TermId>,
}

impl Model {
    /// Create an empty model
    pub fn new() -> Self {
        Self {
            assignments: Vec::new(),
        }
    }

    /// Get the value assigned to a variable (by index)
    pub fn get(&self, index: usize) -> Option<TermId> {
        self.assignments.get(index).copied()
    }

    /// Get all assignments
    pub fn assignments(&self) -> &[TermId] {
        &self.assignments
    }

    /// Check if model is empty
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

/// Model-based generalization result
#[derive(Debug, Clone)]
pub struct MbpResult {
    /// Generalized formula (cube without unnecessary literals)
    pub cube: SmallVec<[TermId; 8]>,
    /// Literals that were eliminated
    pub eliminated: SmallVec<[TermId; 4]>,
    /// Whether the generalization is inductive
    pub is_inductive: bool,
}

impl MbpResult {
    /// Create a new MBP result
    pub fn new(cube: impl IntoIterator<Item = TermId>) -> Self {
        Self {
            cube: cube.into_iter().collect(),
            eliminated: SmallVec::new(),
            is_inductive: false,
        }
    }

    /// Mark as inductive
    pub fn set_inductive(&mut self, inductive: bool) {
        self.is_inductive = inductive;
    }

    /// Add an eliminated literal
    pub fn add_eliminated(&mut self, lit: TermId) {
        self.eliminated.push(lit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stack size and nesting depth for the deep-recursion test below.
    ///
    /// The two are scaled together on purpose: what the test actually pins is
    /// the *ratio* -- about 21 bytes of stack per nesting level
    /// (128 KiB / 6_250). A natively recursive `assert` needs far more than
    /// that per frame and still overflows, so the regression keeps every bit
    /// of its detection power. The pair used to be 1 MiB / 50_000 -- the same
    /// 21 bytes. `mk_and` flattens its arguments (so `acc = mk_and([acc,
    /// lit])` never actually nests, and is quadratic to boot), so the deep
    /// term below is built with `TermManager::intern_term` directly, which
    /// neither flattens nor re-interns already-built prefixes. Never raise
    /// `DEEP_DEPTH` without raising `DEEP_STACK` by the same factor.
    const DEEP_STACK: usize = 1 << 17;
    const DEEP_DEPTH: u32 = 6_250;

    #[test]
    fn test_smt_solver_creation() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();

        let solver = SmtSolver::new(&mut terms, &system);
        assert_eq!(solver.level, 0);
        assert_eq!(solver.stats.num_queries, 0);
    }

    #[test]
    fn test_smt_push_pop() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();

        let mut solver = SmtSolver::new(&mut terms, &system);

        assert_eq!(solver.level, 0);
        solver.push();
        assert_eq!(solver.level, 1);
        solver.push();
        assert_eq!(solver.level, 2);
        solver.pop();
        assert_eq!(solver.level, 1);
        solver.pop();
        assert_eq!(solver.level, 0);
    }

    #[test]
    fn test_smt_basic_sat() {
        let mut terms = TermManager::new();
        let t = terms.mk_true();
        let system = ChcSystem::new();

        let mut solver = SmtSolver::new(&mut terms, &system);
        solver.assert(t);

        let result = solver.check_sat();
        assert!(result.is_ok());
        assert!(result.expect("test operation should succeed"));
    }

    #[test]
    fn test_smt_basic_unsat() {
        let mut terms = TermManager::new();
        let f = terms.mk_false();
        let system = ChcSystem::new();

        let mut solver = SmtSolver::new(&mut terms, &system);
        solver.assert(f);

        let result = solver.check_sat();
        assert!(result.is_ok());
        assert!(!result.expect("test operation should succeed"));
    }

    #[test]
    fn test_model_creation() {
        let model = Model::new();
        assert!(model.is_empty());
        assert_eq!(model.assignments().len(), 0);
    }

    #[test]
    fn test_mbp_result() {
        let cube = [TermId::new(1), TermId::new(2), TermId::new(3)];
        let mut mbp = MbpResult::new(cube);

        assert_eq!(mbp.cube.len(), 3);
        assert!(!mbp.is_inductive);

        mbp.set_inductive(true);
        assert!(mbp.is_inductive);

        mbp.add_eliminated(TermId::new(2));
        assert_eq!(mbp.eliminated.len(), 1);
    }

    #[test]
    fn test_eval_in_model_sat() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();

        // Build: x >= 5 and x <= 5 => x = 5
        let x = terms.mk_var("x_eval_test", terms.sorts.int_sort);
        let five = terms.mk_int(5i64);
        let ge = terms.mk_ge(x, five);
        let le = terms.mk_le(x, five);

        let mut solver = SmtSolver::new(&mut terms, &system);
        solver.assert(ge);
        solver.assert(le);

        let result = solver.check_sat().expect("should be SAT");
        assert!(result, "x >= 5 and x <= 5 should be SAT");

        // Evaluate x in the model — should yield 5
        let val = solver.eval_in_model(x);
        assert!(val.is_some(), "eval_in_model should return Some for x");
    }

    /// Regression test for the `sweep-backend-misc` triage sweep:
    /// `extract_model` used to invent its own `"{pred_name}_{idx}"`
    /// variable names, which never occur in any asserted formula, so the
    /// "extracted" values were arbitrary solver defaults rather than the
    /// real reachable-state values. This verifies the model returned by
    /// `is_state_reachable` now evaluates the predicate's *actual*
    /// canonical current-state variable (`canon_cur_vars`) to the value
    /// pinned down by the asserted state.
    #[test]
    fn test_extract_model_uses_canonical_predicate_variables() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();
        let pred = system.declare_predicate("ExtrModelInv", [terms.sorts.int_sort]);

        // The state formula, expressed the way real Spacer code builds
        // state formulas: over the predicate's canonical current-state
        // variable, not some ad hoc local name.
        let cur_vars = canon_cur_vars(&mut terms, &system, pred);
        assert_eq!(cur_vars.len(), 1);
        let seven = terms.mk_int(7i64);
        let state = terms.mk_eq(cur_vars[0], seven);
        let frame_true = terms.mk_true();

        let mut solver = SmtSolver::new(&mut terms, &system);
        let model = solver
            .is_state_reachable(pred, state, 0, frame_true)
            .expect("SMT query should not error")
            .expect("x=7 with a trivially true frame must be SAT");

        assert_eq!(
            model.assignments.len(),
            1,
            "one assignment per predicate parameter"
        );
        let evaluated = model.assignments[0];
        let seven_again = terms.mk_int(7i64);
        assert_eq!(
            evaluated, seven_again,
            "extract_model must evaluate the predicate's real canonical \
             variable (pinned to 7 by the asserted state), not an \
             unconstrained fabricated variable"
        );
    }

    /// `SmtSolver::assert` splits top-level `And`s, and the `And` nesting
    /// depth comes from parsed input: a [`DEEP_DEPTH`]-deep conjunction must
    /// be asserted without overflowing the stack, and must still be solved
    /// correctly (here: contradictory, hence UNSAT).
    #[test]
    fn assert_flattens_deeply_nested_conjunction() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut terms = TermManager::new();
                let system = ChcSystem::new();
                let bool_sort = terms.sorts.bool_sort;
                let int_sort = terms.sorts.int_sort;
                let x = terms.mk_var("x", int_sort);
                let zero = terms.mk_int(0);
                let one = terms.mk_int(1);

                // (x = 0) /\ (x = 1) /\ b0 /\ b1 /\ ... -- UNSAT. Interned
                // directly (not via `mk_and`, which would flatten this back
                // into one wide `And`) so the tree really is `DEEP_DEPTH`
                // deep.
                let mut formula = terms.mk_eq(x, zero);
                let contradiction = terms.mk_eq(x, one);
                formula = terms.intern_term(
                    TermKind::And(vec![formula, contradiction].into()),
                    bool_sort,
                );
                for i in 0..DEEP_DEPTH {
                    let b = terms.mk_var(&format!("b{i}"), bool_sort);
                    formula = terms.intern_term(TermKind::And(vec![formula, b].into()), bool_sort);
                }

                let mut solver = SmtSolver::new(&mut terms, &system);
                solver.assert(formula);
                assert_eq!(
                    solver.check_sat().ok(),
                    Some(false),
                    "a contradictory deep conjunction must be UNSAT"
                );
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep assert must return");
    }

    /// `SmtSolver::assert` also normalizes a single, genuinely *wide*
    /// top-level `And` -- the form `mk_and` actually produces -- into
    /// `DEEP_DEPTH + 2` separate `solver.assert` calls. That is the case
    /// `assert`'s own soundness comment is about (a single wide `And`
    /// containing disequalities can be answered SAT when the individually
    /// asserted conjuncts are correctly UNSAT), so it is pinned alongside
    /// the deep case rather than only implied by it.
    #[test]
    fn assert_flattens_a_wide_conjunction() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();
        let bool_sort = terms.sorts.bool_sort;
        let int_sort = terms.sorts.int_sort;
        let x = terms.mk_var("x", int_sort);
        let zero = terms.mk_int(0);
        let one = terms.mk_int(1);

        // (x = 0) /\ (x = 1) /\ b0 /\ b1 /\ ... -- UNSAT, as one flat `And`.
        let mut conjuncts = vec![terms.mk_eq(x, zero), terms.mk_eq(x, one)];
        for i in 0..DEEP_DEPTH {
            conjuncts.push(terms.mk_var(&format!("b{i}"), bool_sort));
        }
        let formula = terms.mk_and(conjuncts);

        let mut solver = SmtSolver::new(&mut terms, &system);
        solver.assert(formula);
        assert_eq!(
            solver.check_sat().ok(),
            Some(false),
            "a contradictory wide conjunction must be UNSAT"
        );
    }
}
