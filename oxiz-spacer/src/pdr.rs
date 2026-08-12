//! Property Directed Reachability (PDR/IC3) algorithm.
//!
//! This implements the Spacer algorithm for solving Constrained Horn Clauses.
//!
//! Reference: Z3's `muz/spacer/spacer_context.cpp`
//!
//! ## Algorithm Overview
//!
//! 1. Initialize: F_0 = Init, F_i = True for i > 0
//! 2. Main loop:
//!    a. Check if Bad is reachable from F_N
//!    b. If reachable: create POB and try to block
//!    c. If blocked: propagate lemmas, check for fixpoint
//!    d. If fixpoint: SAFE
//!    e. If counterexample: UNSAFE

use crate::chc::{ChcSystem, PredId, Rule};
use crate::frames::{FrameManager, LemmaId};
use crate::generalize::Generalizer;
use crate::pob::{PobId, PobManager};
use crate::reach::{CexState, Counterexample, ReachFactStore};
use crate::smt::{SmtError, SmtSolver, canon_cur_vars, var_subst};
use oxiz_core::ast::TermKind;
use oxiz_core::{TermId, TermManager};
use smallvec::SmallVec;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;
use tracing::{debug, trace};

/// Errors that can occur during Spacer solving
#[derive(Error, Debug)]
pub enum SpacerError {
    /// The CHC system is empty
    #[error("empty CHC system")]
    EmptySystem,
    /// No query found in the system
    #[error("no query found in CHC system")]
    NoQuery,
    /// SMT solver error
    #[error("SMT solver error: {0}")]
    SolverError(String),
    /// SMT error from solver
    #[error("SMT error: {0}")]
    Smt(#[from] SmtError),
    /// Resource limit exceeded
    #[error("resource limit exceeded")]
    ResourceLimit,
    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

/// Maximum depth of the recursive POB-blocking search.
///
/// Blocking descends one frame level per step, so this is an upper bound on
/// `SpacerConfig::max_level`'s effect on native stack usage. It is generous
/// enough never to be hit by a realistic configuration and small enough that
/// the frames fit comfortably; exceeding it is reported as
/// [`SpacerError::ResourceLimit`], never silently ignored.
const BLOCK_RECURSION_LIMIT: usize = 100_000;

/// Result of Spacer solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpacerResult {
    /// Property holds - system is safe
    /// Contains inductive invariants for each predicate
    Safe,
    /// Counterexample found - system is unsafe
    Unsafe,
    /// Could not determine within resource limits
    Unknown,
}

/// Configuration for Spacer
#[derive(Debug, Clone)]
pub struct SpacerConfig {
    /// Maximum number of frames
    pub max_level: u32,
    /// Maximum number of POBs to process
    pub max_pobs: u32,
    /// Maximum number of SMT queries
    pub max_smt_queries: u32,
    /// Enable inductive generalization
    pub use_inductive_gen: bool,
    /// Enable counterexample-guided abstraction refinement
    pub use_cegar: bool,
    /// Verbosity level (0 = quiet, 1 = normal, 2 = verbose)
    pub verbosity: u32,
}

impl Default for SpacerConfig {
    fn default() -> Self {
        Self {
            max_level: 1000,
            max_pobs: 100000,
            max_smt_queries: 1_000_000,
            use_inductive_gen: true,
            use_cegar: true,
            verbosity: 0,
        }
    }
}

/// Statistics from Spacer solving
#[derive(Debug, Clone, Default)]
pub struct SpacerStats {
    /// Number of frames created
    pub num_frames: u32,
    /// Number of lemmas learned
    pub num_lemmas: u32,
    /// Number of inductive lemmas
    pub num_inductive: u32,
    /// Number of POBs processed
    pub num_pobs: u32,
    /// Number of POBs blocked
    pub num_blocked: u32,
    /// Number of SMT queries
    pub num_smt_queries: u32,
    /// Number of propagation attempts
    pub num_propagations: u32,
    /// Number of POBs subsumed
    pub num_subsumed: u32,
    /// Number of MIC (minimal inductive core) attempts
    pub num_mic_attempts: u32,
    /// Number of CTG (counterexample-guided) strengthenings
    pub num_ctg_strengthenings: u32,
    /// Number of lazy model extractions deferred
    pub num_lazy_models_deferred: u32,
    /// Number of lazy generalizations deferred
    pub num_lazy_generalizations_deferred: u32,
    /// Number of under-approximation states tracked
    pub num_under_approx_states: u32,
    /// Number of under-approximation cache hits
    pub num_under_approx_hits: u32,
    /// Number of SMT queries avoided via under-approximation
    pub num_under_approx_avoided_queries: u32,
    /// Total solving time (microseconds)
    pub total_time_us: u64,
    /// Time spent in reachability checks (microseconds)
    pub reachability_time_us: u64,
    /// Time spent in blocking (microseconds)
    pub blocking_time_us: u64,
    /// Time spent in propagation (microseconds)
    pub propagation_time_us: u64,
    /// Time spent in generalization (microseconds)
    pub generalization_time_us: u64,
}

/// The Spacer solver for Constrained Horn Clauses
pub struct Spacer<'a> {
    /// Term manager for creating formulas
    terms: &'a mut TermManager,
    /// The CHC system to solve
    system: &'a ChcSystem,
    /// Configuration
    config: SpacerConfig,
    /// Frame manager
    frames: FrameManager,
    /// POB manager
    pobs: PobManager,
    /// Reach facts
    reach_facts: ReachFactStore,
    /// Statistics
    stats: SpacerStats,
    /// Current counterexample (if found)
    counterexample: Option<Counterexample>,
    /// Optional cooperative cancellation token. When set and observed `true`
    /// (e.g. by a parallel portfolio once a peer has decided), `solve` stops
    /// at the next main-loop iteration and returns [`SpacerResult::Unknown`]
    /// rather than an unsound answer.
    cancel: Option<Arc<AtomicBool>>,
}

impl<'a> Spacer<'a> {
    /// Create a new Spacer solver
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        Self::with_config(terms, system, SpacerConfig::default())
    }

    /// Create a new Spacer solver with configuration
    pub fn with_config(
        terms: &'a mut TermManager,
        system: &'a ChcSystem,
        config: SpacerConfig,
    ) -> Self {
        Self {
            terms,
            system,
            config,
            frames: FrameManager::new(),
            pobs: PobManager::new(),
            reach_facts: ReachFactStore::new(),
            stats: SpacerStats::default(),
            counterexample: None,
            cancel: None,
        }
    }

    /// Attach a cooperative cancellation token.
    ///
    /// Once the token is set to `true`, the next iteration of the main PDR
    /// loop returns [`SpacerResult::Unknown`]. This lets a parallel portfolio
    /// stop the losing engines as soon as one worker reaches a verdict,
    /// without ever fabricating a Safe/Unsafe answer for a cancelled run.
    #[must_use]
    pub fn with_cancel(mut self, token: Arc<AtomicBool>) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Solve the CHC system.
    ///
    /// Any SMT `Unknown` or resource-limit encountered while solving is
    /// surfaced honestly as [`SpacerResult::Unknown`] rather than being
    /// collapsed into a (potentially unsound) Safe/Unsafe answer.
    pub fn solve(&mut self) -> Result<SpacerResult, SpacerError> {
        match self.solve_inner() {
            Err(SpacerError::Smt(SmtError::Unknown)) | Err(SpacerError::ResourceLimit) => {
                Ok(SpacerResult::Unknown)
            }
            other => other,
        }
    }

    fn solve_inner(&mut self) -> Result<SpacerResult, SpacerError> {
        // Validate system
        if self.system.is_empty() {
            // Empty system is trivially safe - nothing can go wrong
            return Ok(SpacerResult::Safe);
        }

        if self.system.queries().next().is_none() {
            return Err(SpacerError::NoQuery);
        }

        // Spacer's PDR engine here is sound only for the single-predicate
        // linear fragment (one predicate, at most one body predicate per rule,
        // and predicate arguments that are plain variables).  For anything
        // outside it we return `Unknown` rather than risk an unsound answer.
        if !self.is_supported_fragment() {
            debug!("Spacer: unsupported CHC fragment — returning Unknown");
            return Ok(SpacerResult::Unknown);
        }

        // Initialize frames for all predicates
        self.initialize()?;

        // Main PDR loop
        loop {
            // Cooperative cancellation (e.g. a portfolio peer already decided).
            if self
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Relaxed))
            {
                return Ok(SpacerResult::Unknown);
            }

            // Check resource limits
            if self.stats.num_frames > self.config.max_level {
                return Ok(SpacerResult::Unknown);
            }
            if self.stats.num_pobs > self.config.max_pobs {
                return Ok(SpacerResult::Unknown);
            }
            if self.stats.num_smt_queries > self.config.max_smt_queries {
                return Ok(SpacerResult::Unknown);
            }

            // Try to find a counterexample at the current level
            match self.check_reachability()? {
                ReachabilityResult::Unreachable => {
                    // Try to propagate lemmas
                    if self.propagate()? {
                        // Fixpoint found - system is safe
                        return Ok(SpacerResult::Safe);
                    }
                    // Move to next level
                    self.frames.next_level();
                    self.stats.num_frames = self.stats.num_frames.saturating_add(1);
                }
                ReachabilityResult::Reachable(pob_id) => {
                    // Try to block the POB
                    match self.block(pob_id)? {
                        BlockResult::Blocked => {
                            // Continue processing POBs
                        }
                        BlockResult::Counterexample => {
                            // Real counterexample found
                            return Ok(SpacerResult::Unsafe);
                        }
                    }
                }
            }
        }
    }

    /// Check whether the system is in the single-predicate linear fragment
    /// that this PDR engine can handle soundly.
    fn is_supported_fragment(&self) -> bool {
        // Exactly one declared predicate.
        if self.system.predicates().count() != 1 {
            return false;
        }
        let the_pred = match self.system.predicates().next() {
            Some(p) => p.id,
            None => return false,
        };

        for rule in self.system.rules() {
            // At most one body predicate, and it must be `the_pred`.
            if rule.body.predicates.len() > 1 {
                return false;
            }
            for app in &rule.body.predicates {
                if app.pred != the_pred || !self.args_are_distinct_vars(&app.args) {
                    return false;
                }
            }
            // Head predicate arguments must be distinct plain variables.
            if let Some(app) = rule.head.as_predicate()
                && (app.pred != the_pred || !self.args_are_distinct_vars(&app.args))
            {
                return false;
            }
        }
        true
    }

    /// True iff every argument is a distinct plain variable term.
    fn args_are_distinct_vars(&self, args: &[TermId]) -> bool {
        for (i, &a) in args.iter().enumerate() {
            match self.terms.get(a).map(|d| &d.kind) {
                Some(TermKind::Var(_)) => {}
                _ => return false,
            }
            if args[..i].contains(&a) {
                return false;
            }
        }
        true
    }

    /// Canonicalize a constraint expressed over the argument terms `app_args`
    /// of predicate `pred` into `pred`'s canonical current-state variables.
    /// Returns `None` if the args are not plain variables.
    fn canonicalize(
        &mut self,
        constraint: TermId,
        app_args: &[TermId],
        pred: PredId,
    ) -> Option<TermId> {
        let cur = canon_cur_vars(self.terms, self.system, pred);
        let subst = var_subst(self.terms, app_args, &cur)?;
        Some(self.terms.substitute(constraint, &subst))
    }

    /// Initialize the solver
    fn initialize(&mut self) -> Result<(), SpacerError> {
        // Initialize frames for all predicates
        for pred in self.system.predicates() {
            self.frames.get_or_create(pred.id);
        }

        // Process init rules to establish initial reach facts
        for rule in self.system.entries() {
            self.process_init_rule(rule)?;
        }

        Ok(())
    }

    /// Process an init rule.
    ///
    /// The init constraint is normalized into the head predicate's canonical
    /// current-state variables so that it can be intersected with POB cubes
    /// (which live in the same namespace) during `is_init_reachable`.
    fn process_init_rule(&mut self, rule: &Rule) -> Result<(), SpacerError> {
        if let Some(head_app) = rule.head.as_predicate() {
            let head_pred = head_app.pred;
            let args = head_app.args.clone();
            let rule_id = rule.id;
            let constraint = rule.body.constraint;
            if let Some(init_fact) = self.canonicalize(constraint, &args, head_pred) {
                self.reach_facts.add(head_pred, init_fact, rule_id, true);
            }
        }
        Ok(())
    }

    /// Check reachability of bad states at the current frame level.
    fn check_reachability(&mut self) -> Result<ReachabilityResult, SpacerError> {
        let level = self.frames.current_level();

        // Collect (pred, args, constraint) for every query body predicate.  The
        // query constraint is normalized into the predicate's canonical
        // current-state variables so that it can be checked against frames and
        // intersected with init facts consistently.
        let mut targets: Vec<(PredId, SmallVec<[TermId; 4]>, TermId)> = Vec::new();
        for query in self.system.queries() {
            for body_app in &query.body.predicates {
                targets.push((body_app.pred, body_app.args.clone(), query.body.constraint));
            }
        }

        for (pred, args, constraint) in targets {
            let Some(post) = self.canonicalize(constraint, &args, pred) else {
                continue;
            };
            if self.is_bad_reachable(pred, post, level)? {
                let pob_id = self.pobs.create(pred, post, level, 0);
                self.stats.num_pobs = self.stats.num_pobs.saturating_add(1);
                return Ok(ReachabilityResult::Reachable(pob_id));
            }
        }

        Ok(ReachabilityResult::Unreachable)
    }

    /// Check whether the (already canonicalized) bad state `post` intersects
    /// the current over-approximation `F_level(pred)`.
    ///
    /// Query: is `F_level(pred) ∧ post` SAT?
    fn is_bad_reachable(
        &mut self,
        pred: PredId,
        post: TermId,
        level: u32,
    ) -> Result<bool, SpacerError> {
        let frame_formula = self.build_frame_formula(pred, level);

        let mut smt = SmtSolver::new(self.terms, self.system);
        smt.push();
        smt.assert(frame_formula);
        smt.assert(post);
        let is_sat = smt.check_sat()?;
        smt.pop();

        self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
        if is_sat {
            debug!("Bad state reachable at level {}", level);
        }
        Ok(is_sat)
    }

    /// Block a proof obligation.
    ///
    /// Standard IC3/PDR recursive blocking:
    ///
    /// * If the POB is already excluded by a frame lemma, close it.
    /// * At level 0, if the (canonical) bad state intersects Init, a real
    ///   counterexample has been found.
    /// * Otherwise repeatedly look for a concrete predecessor at `level-1`.
    ///   Each predecessor found is recursively blocked; if it turns out to be
    ///   init-reachable the counterexample propagates up.  When no further
    ///   predecessor exists, the POB is generalized into a blocking lemma at
    ///   `level` and closed.
    fn block(&mut self, pob_id: PobId) -> Result<BlockResult, SpacerError> {
        self.block_at_depth(pob_id, 0)
    }

    /// [`Self::block`] with the current blocking-recursion depth.
    ///
    /// Blocking recurses on a POB one level lower each time, so the depth is
    /// bounded by `SpacerConfig::max_level` -- which is a *user-supplied*
    /// number, and a legitimate-looking `max_level: 100_000` would overflow
    /// the native stack. Because this function already returns a `Result`,
    /// the bound is enforced honestly: exceeding it reports
    /// [`SpacerError::ResourceLimit`], which the caller surfaces as
    /// `Unknown`, rather than truncating the search and reporting `Safe` on
    /// an unexplored obligation.
    fn block_at_depth(&mut self, pob_id: PobId, depth: usize) -> Result<BlockResult, SpacerError> {
        if depth > BLOCK_RECURSION_LIMIT {
            return Err(SpacerError::ResourceLimit);
        }

        let (level, pred, post) = {
            let pob = self
                .pobs
                .get(pob_id)
                .ok_or_else(|| SpacerError::Internal("POB not found".to_string()))?;
            (pob.level(), pob.pred, pob.post)
        };

        // Check if already blocked by an existing lemma.
        if self.is_blocked_by_lemma(pred, post, level)? {
            if let Some(lemma_id) = self.find_blocking_lemma(pred, post, level)? {
                self.pobs.close(pob_id, lemma_id);
                self.stats.num_blocked = self.stats.num_blocked.saturating_add(1);
            }
            return Ok(BlockResult::Blocked);
        }

        // Level 0: the frame is exactly Init, so the only way to block is to
        // prove the bad state is not an initial state.
        if level == 0 {
            if self.is_init_reachable(pred, post)? {
                self.build_counterexample(pob_id)?;
                return Ok(BlockResult::Counterexample);
            }
            let lemma = self.generalize_blocking_lemma(pob_id)?;
            let lemma_id = self.frames.add_lemma(pred, lemma, level);
            self.pobs.close(pob_id, lemma_id);
            self.stats.num_blocked = self.stats.num_blocked.saturating_add(1);
            self.stats.num_lemmas = self.stats.num_lemmas.saturating_add(1);
            return Ok(BlockResult::Blocked);
        }

        // Level > 0: search for predecessors until none remain.
        loop {
            // Bound the search so a diverging predecessor chain surfaces as
            // Unknown rather than looping forever.
            if self.stats.num_pobs > self.config.max_pobs {
                return Err(SpacerError::ResourceLimit);
            }

            match self.find_predecessor(pob_id)? {
                Some(pred_pob_id) => match self.block_at_depth(pred_pob_id, depth + 1)? {
                    BlockResult::Counterexample => return Ok(BlockResult::Counterexample),
                    BlockResult::Blocked => {
                        // Predecessor blocked; loop to look for another one.
                        continue;
                    }
                },
                None => {
                    // No predecessor: generalize into a blocking lemma.
                    let lemma = self.generalize_blocking_lemma(pob_id)?;
                    let lemma_id = self.frames.add_lemma(pred, lemma, level);
                    self.pobs.close(pob_id, lemma_id);
                    self.stats.num_blocked = self.stats.num_blocked.saturating_add(1);
                    self.stats.num_lemmas = self.stats.num_lemmas.saturating_add(1);
                    return Ok(BlockResult::Blocked);
                }
            }
        }
    }

    /// Check if a state is blocked by an existing lemma
    fn is_blocked_by_lemma(
        &mut self,
        pred: PredId,
        state: TermId,
        level: u32,
    ) -> Result<bool, SpacerError> {
        // Check if any lemma at this level or higher blocks the state
        if let Some(pred_frames) = self.frames.get(pred) {
            // Collect lemma formulas to check
            let lemmas: Vec<TermId> = pred_frames
                .lemmas_geq_level(level)
                .map(|l| l.formula)
                .collect();

            // Check each lemma
            for lemma in lemmas {
                let mut smt = SmtSolver::new(self.terms, self.system);
                if smt.is_blocked_by(lemma, state)? {
                    self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
                    return Ok(true);
                }
                self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
            }
        }
        Ok(false)
    }

    /// Find the specific lemma that actually blocks `state` at `level`
    /// (i.e. the one `is_blocked_by_lemma` found), rather than blindly
    /// returning whatever lemma happens to be first in the level's lemma
    /// list. Returning an unrelated lemma here made `self.pobs.close`
    /// record the wrong lemma as the reason a POB was closed whenever
    /// more than one lemma existed at `level` -- corrupting lemma
    /// provenance/usage accounting (and any downstream proof/certificate
    /// construction that trusts it) for every such POB.
    fn find_blocking_lemma(
        &mut self,
        pred: PredId,
        state: TermId,
        level: u32,
    ) -> Result<Option<LemmaId>, SpacerError> {
        let lemmas: Vec<(LemmaId, TermId)> = match self.frames.get(pred) {
            Some(pred_frames) => pred_frames
                .lemmas_geq_level(level)
                .map(|lemma| (lemma.id, lemma.formula))
                .collect(),
            None => return Ok(None),
        };

        for (id, formula) in lemmas {
            let mut smt = SmtSolver::new(self.terms, self.system);
            let blocks = smt.is_blocked_by(formula, state)?;
            self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
            if blocks {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    /// Check whether the (canonical) state `state` intersects the initial
    /// states of `pred`: is `init_fact ∧ state` SAT for any init fact?
    fn is_init_reachable(&mut self, pred: PredId, state: TermId) -> Result<bool, SpacerError> {
        let facts: Vec<TermId> = self
            .reach_facts
            .for_pred(pred)
            .filter(|f| f.is_init())
            .map(|f| f.fact)
            .collect();

        for fact in facts {
            let mut smt = SmtSolver::new(self.terms, self.system);
            smt.push();
            smt.assert(fact);
            smt.assert(state);
            let is_sat = smt.check_sat()?;
            smt.pop();
            self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
            if is_sat {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Find a concrete predecessor state for a POB via a model-based SMT query.
    ///
    /// For each transition rule `Q(body_args) ∧ C ⇒ P(head_args)` we ask:
    ///   `F_{level-1}(Q) ∧ C[body_args ↦ cur] ∧ post[cur_P ↦ head_args]`
    /// where `cur` are `Q`'s canonical current-state variables.  A SAT result
    /// exhibits a concrete state (read from the model as a cube of equalities)
    /// that transitions in one step into the POB's bad state — a genuine
    /// predecessor.  This is what makes `SpacerResult::Unsafe` reachable.
    fn find_predecessor(&mut self, pob_id: PobId) -> Result<Option<PobId>, SpacerError> {
        let (pred, level, depth, post) = {
            let pob = self
                .pobs
                .get(pob_id)
                .ok_or_else(|| SpacerError::Internal("POB not found".to_string()))?;
            (pob.pred, pob.level(), pob.depth(), pob.post)
        };

        if level == 0 {
            return Ok(None);
        }

        // Collect transition-rule data (skip init rules, which have no body
        // predicate) up-front to release the borrow on `self.system`.
        #[allow(clippy::type_complexity)]
        let rules: Vec<(PredId, SmallVec<[TermId; 4]>, SmallVec<[TermId; 4]>, TermId)> = self
            .system
            .rules_by_head(pred)
            .filter_map(|rule| {
                let body_app = rule.body.predicates.first()?;
                let head_app = rule.head.as_predicate()?;
                Some((
                    body_app.pred,
                    body_app.args.clone(),
                    head_app.args.clone(),
                    rule.body.constraint,
                ))
            })
            .collect();

        for (body_pred, body_args, head_args, constraint) in rules {
            // Canonical current-state variables of the predecessor predicate.
            let cur = canon_cur_vars(self.terms, self.system, body_pred);
            let frame = self.build_frame_formula(body_pred, level - 1);

            // Transition constraint with the body args mapped to canonical
            // current-state variables; the next state stays as the head args.
            let Some(cur_subst) = var_subst(self.terms, &body_args, &cur) else {
                continue;
            };
            let trans = self.terms.substitute(constraint, &cur_subst);

            // Move `post` (over `pred`'s canonical current vars) onto the
            // next-state (head arg) variables.
            let pred_cur = canon_cur_vars(self.terms, self.system, pred);
            let Some(next_subst) = var_subst(self.terms, &pred_cur, &head_args) else {
                continue;
            };
            let post_next = self.terms.substitute(post, &next_subst);

            let mut smt = SmtSolver::new(self.terms, self.system);
            smt.push();
            smt.assert(frame);
            smt.assert(trans);
            smt.assert(post_next);
            let is_sat = smt.check_sat()?;
            self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);

            if !is_sat {
                smt.pop();
                continue;
            }

            // Extract a concrete predecessor cube from the model.  Every state
            // variable must be pinned to a concrete value, otherwise the "cube"
            // would be an over-approximation and claiming Unsafe from it would
            // be unsound — in that case we surface Unknown.
            let mut lits: Vec<TermId> = Vec::new();
            let mut underdetermined = false;
            for &v in &cur {
                match smt.eval_in_model(v) {
                    Some(val) if is_concrete_value(smt.terms(), val) => {
                        let eq = smt.terms().mk_eq(v, val);
                        lits.push(eq);
                    }
                    _ => {
                        underdetermined = true;
                        break;
                    }
                }
            }
            smt.pop();

            if underdetermined {
                return Err(SpacerError::Smt(SmtError::Unknown));
            }

            let cube = match lits.len() {
                0 => self.terms.mk_true(),
                1 => lits[0],
                _ => self.terms.mk_and(lits),
            };

            let pred_pob = self
                .pobs
                .create_derived(body_pred, cube, level - 1, depth + 1, pob_id);
            self.stats.num_pobs = self.stats.num_pobs.saturating_add(1);
            return Ok(Some(pred_pob));
        }

        Ok(None)
    }

    /// Generalize a blocking lemma via MIC-style inductive generalization.
    ///
    /// The ungeneralized blocking lemma is `¬post`. When inductive
    /// generalization is enabled we compute a minimal subcube `c ⊆ post` such
    /// that the *stronger* blocking clause `¬c` is still a sound frame lemma —
    /// it excludes every initial state and (for `level > 0`) is inductive
    /// relative to `F_{level-1}` — and return `¬c`. Because `post ⇒ c` we have
    /// `¬c ⇒ ¬post`, so the generalized clause still blocks the bad state while
    /// excluding a strictly larger region. Without this, frames accumulate only
    /// exact single-cube exclusions and the engine cannot converge on
    /// non-trivial inductive invariants.
    ///
    /// Reference: Z3's `muz/spacer/spacer_generalizers.cpp` (inductive
    /// generalization / MIC).
    fn generalize_blocking_lemma(&mut self, pob_id: PobId) -> Result<TermId, SpacerError> {
        let (pred, level, post) = {
            let pob = self
                .pobs
                .get(pob_id)
                .ok_or_else(|| SpacerError::Internal("POB not found".to_string()))?;
            (pob.pred, pob.level(), pob.post)
        };

        if !self.config.use_inductive_gen {
            return Ok(self.terms.mk_not(post));
        }

        let start = oxiz_time::Instant::now();
        let cube = self.mic_generalize_cube(pred, post, level)?;
        self.stats.generalization_time_us = self
            .stats
            .generalization_time_us
            .saturating_add(start.elapsed().as_micros() as u64);

        let generalized = match cube.len() {
            // MIC keeps the cube non-empty; guard defensively so we never emit
            // `¬true = false`, which would unsoundly collapse the frame to UNSAT.
            0 => return Ok(self.terms.mk_not(post)),
            1 => cube[0],
            _ => self.terms.mk_and(cube),
        };
        Ok(self.terms.mk_not(generalized))
    }

    /// MIC (Minimal Inductive Clause) generalization of a blocking cube.
    ///
    /// Starting from the literals of `post`, greedily drop literals as long as
    /// the blocking clause `¬(∧ remaining)` stays a sound frame lemma — see
    /// [`Self::subcube_is_inductive_blocker`]. The returned subcube is a
    /// non-empty subset of `post`'s literals. Increments
    /// [`SpacerStats::num_mic_attempts`] whenever a real (multi-literal)
    /// generalization is attempted.
    fn mic_generalize_cube(
        &mut self,
        pred: PredId,
        post: TermId,
        level: u32,
    ) -> Result<Vec<TermId>, SpacerError> {
        let mut cube = Generalizer::extract_cube(self.terms, post);
        if cube.len() <= 1 {
            // A single-literal (or empty) cube cannot be generalized further.
            return Ok(cube);
        }

        self.stats.num_mic_attempts = self.stats.num_mic_attempts.saturating_add(1);

        let mut i = 0;
        while i < cube.len() {
            // Never drop the last remaining literal: `¬true = false` is not a
            // sound frame lemma.
            if cube.len() <= 1 {
                break;
            }
            let removed = cube.remove(i);
            if self.subcube_is_inductive_blocker(pred, &cube, level)? {
                // Literal dropped successfully; keep `i` (the next literal has
                // shifted into this slot).
                continue;
            }
            // Restore: this literal is required for soundness.
            cube.insert(i, removed);
            i += 1;
        }

        Ok(cube)
    }

    /// Check that the blocking clause `¬(∧ cube)` is a sound frame lemma at
    /// `level`:
    ///
    /// * **Initiation** — `Init ∧ (∧ cube)` is UNSAT, so `¬(∧ cube)` holds in
    ///   every initial state. Checked at all levels.
    /// * **Consecution** (`level > 0` only) — `¬(∧ cube)` is inductive relative
    ///   to `F_{level-1}`: `F_{level-1} ∧ ¬(∧ cube) ∧ T ⇒ ¬(∧ cube)'` for every
    ///   transition rule, with the clause assumed on the current state
    ///   (standard Bradley relative inductive generalization).
    ///
    /// A `false` return means dropping the corresponding literal would break
    /// one of these properties, so the literal must be kept.
    fn subcube_is_inductive_blocker(
        &mut self,
        pred: PredId,
        cube: &[TermId],
        level: u32,
    ) -> Result<bool, SpacerError> {
        if cube.is_empty() {
            return Ok(false);
        }
        let cube_conj = match cube.len() {
            1 => cube[0],
            _ => self.terms.mk_and(cube.iter().copied()),
        };

        // Initiation: the clause must still exclude every initial state.
        if self.is_init_reachable(pred, cube_conj)? {
            return Ok(false);
        }

        if level == 0 {
            // F_0 is exactly Init, so initiation is the only requirement.
            return Ok(true);
        }

        // Consecution relative to F_{level-1}, with the clause assumed on the
        // current state. `is_lemma_inductive` checks, per rule, that
        // `frame ∧ T ∧ ¬lemma'` is UNSAT; conjoining `¬cube` into the frame
        // yields `F_{level-1} ∧ ¬cube ∧ T ∧ cube'` UNSAT.
        let neg_cube = self.terms.mk_not(cube_conj);
        let frame = self.build_frame_formula(pred, level - 1);
        let frame_with_lemma = self.terms.mk_and([frame, neg_cube]);
        let mut smt = SmtSolver::new(self.terms, self.system);
        let inductive = smt.is_lemma_inductive(pred, neg_cube, level, frame_with_lemma)?;
        self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
        Ok(inductive)
    }

    /// Build a counterexample trace
    fn build_counterexample(&mut self, pob_id: PobId) -> Result<(), SpacerError> {
        let mut cex = Counterexample::new();

        // Trace back from POB to initial state
        let mut current = Some(pob_id);
        while let Some(id) = current {
            if let Some(pob) = self.pobs.get(id) {
                cex.push(CexState {
                    pred: pob.pred,
                    state: pob.post,
                    rule: None,
                    assignments: SmallVec::new(),
                });
                current = pob.parent();
            } else {
                break;
            }
        }

        cex.reverse();
        self.counterexample = Some(cex);
        Ok(())
    }

    /// Propagate lemmas to higher levels
    fn propagate(&mut self) -> Result<bool, SpacerError> {
        self.stats.num_propagations = self.stats.num_propagations.saturating_add(1);

        // Try to push lemmas to higher levels
        let current_level = self.frames.current_level();

        for level in 1..=current_level {
            let mut all_pushed = true;

            // Collect all predicates to process
            let pred_ids: Vec<_> = self.system.predicates().map(|p| p.id).collect();

            for pred_id in pred_ids {
                // Collect lemmas to push (immutable borrow)
                let lemmas_to_push: Vec<_> = if let Some(pred_frames) = self.frames.get(pred_id) {
                    pred_frames.lemmas_at_level(level).map(|l| l.id).collect()
                } else {
                    Vec::new()
                };

                // Check and propagate each lemma
                for lemma_id in lemmas_to_push {
                    // Check if lemma can be pushed: F_level /\ T => lemma'
                    let can_push = self.can_push_lemma(pred_id, lemma_id, level)?;

                    if can_push {
                        if let Some(pred_frames) = self.frames.get_mut(pred_id) {
                            pred_frames.propagate(lemma_id, level + 1);
                        }
                    } else {
                        all_pushed = false;
                    }
                }
            }

            // If all lemmas at this level were pushed, we found a fixpoint
            if all_pushed && level == current_level {
                // Mark all pushed lemmas as inductive
                let pred_ids: Vec<_> = self.system.predicates().map(|p| p.id).collect();
                for pred_id in pred_ids {
                    if let Some(pred_frames) = self.frames.get_mut(pred_id) {
                        pred_frames.propagate_to_infinity(level);
                    }
                }
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check if a lemma can be pushed to the next level
    fn can_push_lemma(
        &mut self,
        pred: PredId,
        lemma_id: LemmaId,
        level: u32,
    ) -> Result<bool, SpacerError> {
        // Get the lemma formula
        let lemma = if let Some(pred_frames) = self.frames.get(pred) {
            if let Some(lemma_data) = pred_frames.get_lemma(lemma_id) {
                lemma_data.formula
            } else {
                return Ok(false);
            }
        } else {
            return Ok(false);
        };

        // Build frame formula at current level
        let frame_formula = self.build_frame_formula(pred, level);

        // Check if lemma is inductive: F_level /\ T => lemma'
        // This is checked by verifying UNSAT of: F_level /\ T /\ ¬lemma'
        let mut smt = SmtSolver::new(self.terms, self.system);
        let can_push = smt.is_lemma_inductive(pred, lemma, level, frame_formula)?;

        self.stats.num_smt_queries = self.stats.num_smt_queries.saturating_add(1);
        trace!(
            "Lemma {:?} at level {} can_push: {}",
            lemma_id, level, can_push
        );
        Ok(can_push)
    }

    /// Get the counterexample (if found)
    #[must_use]
    pub fn counterexample(&self) -> Option<&Counterexample> {
        self.counterexample.as_ref()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SpacerStats {
        &self.stats
    }

    /// Get inductive invariants for all predicates
    pub fn invariants(&self) -> Vec<(PredId, Vec<TermId>)> {
        let mut result = Vec::new();

        for pred in self.system.predicates() {
            if let Some(pred_frames) = self.frames.get(pred.id) {
                let invs: Vec<TermId> = pred_frames.inductive_lemmas().map(|l| l.formula).collect();
                if !invs.is_empty() {
                    result.push((pred.id, invs));
                }
            }
        }

        result
    }

    /// Reset the solver for a new run
    pub fn reset(&mut self) {
        self.frames.reset();
        self.pobs.clear();
        self.reach_facts.clear();
        self.stats = SpacerStats::default();
        self.counterexample = None;
    }

    /// Build a frame formula for a predicate at a given level
    /// Returns the conjunction of all lemmas at level or higher
    fn build_frame_formula(&mut self, pred: PredId, level: u32) -> TermId {
        if let Some(pred_frames) = self.frames.get(pred) {
            let lemmas: Vec<TermId> = pred_frames
                .lemmas_geq_level(level)
                .map(|l| l.formula)
                .collect();

            if lemmas.is_empty() {
                // No lemmas, frame is true
                self.terms.mk_true()
            } else if lemmas.len() == 1 {
                lemmas[0]
            } else {
                // Conjunction of all lemmas
                self.terms.mk_and(lemmas)
            }
        } else {
            // No frames for this predicate, return true
            self.terms.mk_true()
        }
    }
}

/// True iff `term` is a concrete value (a model literal), not a variable or
/// compound term.  Used to confirm that a predecessor cube extracted from a
/// model pins every state variable to a definite value.
fn is_concrete_value(terms: &TermManager, term: TermId) -> bool {
    matches!(
        terms.get(term).map(|d| &d.kind),
        Some(
            TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::True
                | TermKind::False
                | TermKind::StringLit(_)
        )
    )
}

/// Result of reachability check
enum ReachabilityResult {
    /// Bad state is unreachable at current level
    Unreachable,
    /// Bad state is reachable, POB created
    Reachable(PobId),
}

/// Result of blocking a POB
enum BlockResult {
    /// POB was successfully blocked
    Blocked,
    /// A real counterexample was found
    Counterexample,
}

/// Legacy interface for backward compatibility
pub struct LegacySpacer {
    result: SpacerResult,
}

impl LegacySpacer {
    /// Create a new legacy Spacer solver
    pub fn new() -> Self {
        Self {
            result: SpacerResult::Unknown,
        }
    }

    /// Solve (placeholder for legacy interface)
    pub fn solve(&mut self) -> SpacerResult {
        self.result.clone()
    }
}

impl Default for LegacySpacer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chc::PredicateApp;

    #[test]
    fn test_spacer_creation() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            constraint,
            inv,
            [x],
        );

        let spacer = Spacer::new(&mut terms, &system);
        assert_eq!(spacer.stats().num_frames, 0);
    }

    #[test]
    fn test_spacer_config() {
        let config = SpacerConfig {
            max_level: 100,
            max_pobs: 1000,
            max_smt_queries: 10000,
            use_inductive_gen: true,
            use_cegar: false,
            verbosity: 1,
        };

        assert_eq!(config.max_level, 100);
        assert_eq!(config.max_smt_queries, 10000);
        assert!(config.use_inductive_gen);
        assert!(!config.use_cegar);
    }

    #[test]
    fn test_spacer_empty_system() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();

        let mut spacer = Spacer::new(&mut terms, &system);
        let result = spacer.solve();

        // Empty system is trivially safe - nothing can go wrong
        assert!(matches!(result, Ok(SpacerResult::Safe)));
    }

    #[test]
    fn test_spacer_no_query() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let constraint = terms.mk_true();

        // Only init rule, no query
        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            constraint,
            inv,
            [x],
        );

        let mut spacer = Spacer::new(&mut terms, &system);
        let result = spacer.solve();

        assert!(matches!(result, Err(SpacerError::NoQuery)));
    }

    #[test]
    fn test_spacer_simple_safe() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);

        // Init: x = 0 => Inv(x)
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let init_constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            init_constraint,
            inv,
            [x],
        );

        // Trans: Inv(x) /\ x' = x + 1 /\ x' < 10 => Inv(x')
        let x_prime = terms.mk_var("x'", terms.sorts.int_sort);
        let one = terms.mk_int(1);
        let ten = terms.mk_int(10);
        let x_plus_one = terms.mk_add([x, one]);
        let trans_eq = terms.mk_eq(x_prime, x_plus_one);
        let bound = terms.mk_lt(x_prime, ten);
        let trans_constraint = terms.mk_and([trans_eq, bound]);

        system.add_transition_rule(
            [
                ("x".to_string(), terms.sorts.int_sort),
                ("x'".to_string(), terms.sorts.int_sort),
            ],
            [PredicateApp::new(inv, [x])],
            trans_constraint,
            inv,
            [x_prime],
        );

        // Query: Inv(x) /\ x < 0 => false
        let neg_constraint = terms.mk_lt(x, zero);
        system.add_query(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(inv, [x])],
            neg_constraint,
        );

        let mut spacer = Spacer::new(&mut terms, &system);
        let result = spacer.solve();

        // The system is safe: x >= 0 is an inductive invariant, proved via the
        // consecution check with primed-state renaming.
        assert!(matches!(result, Ok(SpacerResult::Safe)));
    }

    #[test]
    fn test_legacy_spacer() {
        let spacer = LegacySpacer::new();
        assert!(matches!(spacer.result, SpacerResult::Unknown));
    }

    /// Regression test for the `sweep-backend-misc` triage sweep:
    /// `find_blocking_lemma` used to return whatever lemma happened to be
    /// first in the level's lemma list, without checking that it actually
    /// blocks the queried state. Set up several lemmas at the same level
    /// where only *one* actually blocks the state in question, and verify
    /// `find_blocking_lemma` returns that specific lemma (not merely "a"
    /// lemma that happens to be first).
    #[test]
    fn test_find_blocking_lemma_returns_actual_blocker() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();
        let pred = system.declare_predicate("FblInv", [terms.sorts.int_sort]);
        // A query is required for `Spacer::new` to accept the system.
        let x = terms.mk_var("fbl_x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let neg = terms.mk_lt(x, zero);
        system.add_query(
            [("fbl_x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(pred, [x])],
            neg,
        );

        let mut spacer = Spacer::new(&mut terms, &system);

        // The state under test: x = 3.
        let three = spacer.terms.mk_int(3);
        let state = spacer.terms.mk_eq(x, three);

        // Several decoy lemmas that do NOT block x=3 (x != 5, x != 7, x !=
        // 9 are all satisfiable together with x=3), inserted first.
        for excluded in [5i64, 7, 9] {
            let val = spacer.terms.mk_int(excluded);
            let eq = spacer.terms.mk_eq(x, val);
            let ne = spacer.terms.mk_not(eq);
            spacer.frames.add_lemma(pred, ne, 1);
        }

        // The one lemma that DOES block x=3: x != 3.
        let three_again = spacer.terms.mk_int(3);
        let eq3 = spacer.terms.mk_eq(x, three_again);
        let blocking_formula = spacer.terms.mk_not(eq3);
        let blocking_id = spacer.frames.add_lemma(pred, blocking_formula, 1);

        // More decoys after it, so the true blocker isn't simply "last"
        // either.
        for excluded in [11i64, 13] {
            let val = spacer.terms.mk_int(excluded);
            let eq = spacer.terms.mk_eq(x, val);
            let ne = spacer.terms.mk_not(eq);
            spacer.frames.add_lemma(pred, ne, 1);
        }

        let found = spacer
            .find_blocking_lemma(pred, state, 1)
            .expect("SMT queries should not error")
            .expect("at least one lemma blocks x=3");

        assert_eq!(
            found, blocking_id,
            "find_blocking_lemma must return the lemma that actually \
             blocks the state, not just the first lemma at the level"
        );
    }

    /// SP-01 regression: MIC-style inductive generalization must actually drop
    /// irrelevant literals from a blocking cube. The transition keeps `x`
    /// constant and only increments `y`, so blocking the bad cube
    /// `(x = 5 ∧ y = 3)` should generalize to the single-literal clause
    /// `¬(x = 5)` — the `y = 3` literal is dropped because `¬(x = 5)` is
    /// inductive on its own, whereas `¬(y = 3)` is not. This proves the
    /// generalized lemma is strictly shorter than the raw cube.
    #[test]
    fn test_mic_generalize_drops_irrelevant_literal() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();
        let inv = system.declare_predicate("MicInv", [terms.sorts.int_sort, terms.sorts.int_sort]);

        // Init: x = 0 ∧ y = 0 => Inv(x, y)
        let x = terms.mk_var("mic_x", terms.sorts.int_sort);
        let y = terms.mk_var("mic_y", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let ix = terms.mk_eq(x, zero);
        let iy = terms.mk_eq(y, zero);
        let init = terms.mk_and([ix, iy]);
        system.add_init_rule(
            [
                ("mic_x".to_string(), terms.sorts.int_sort),
                ("mic_y".to_string(), terms.sorts.int_sort),
            ],
            init,
            inv,
            [x, y],
        );

        // Trans: Inv(x, y) ∧ x' = x ∧ y' = y + 1 => Inv(x', y')
        let xp = terms.mk_var("mic_xp", terms.sorts.int_sort);
        let yp = terms.mk_var("mic_yp", terms.sorts.int_sort);
        let one = terms.mk_int(1);
        let keep_x = terms.mk_eq(xp, x);
        let y_plus_one = terms.mk_add([y, one]);
        let step_y = terms.mk_eq(yp, y_plus_one);
        let trans = terms.mk_and([keep_x, step_y]);
        system.add_transition_rule(
            [
                ("mic_x".to_string(), terms.sorts.int_sort),
                ("mic_y".to_string(), terms.sorts.int_sort),
                ("mic_xp".to_string(), terms.sorts.int_sort),
                ("mic_yp".to_string(), terms.sorts.int_sort),
            ],
            [PredicateApp::new(inv, [x, y])],
            trans,
            inv,
            [xp, yp],
        );

        // A query so the system is well-formed (unused by this direct test).
        let neg = terms.mk_lt(x, zero);
        system.add_query(
            [
                ("mic_x".to_string(), terms.sorts.int_sort),
                ("mic_y".to_string(), terms.sorts.int_sort),
            ],
            [PredicateApp::new(inv, [x, y])],
            neg,
        );

        // Build the bad cube over the predicate's canonical current-state
        // variables, the same namespace real POB cubes use.
        let cur = canon_cur_vars(&mut terms, &system, inv);
        assert_eq!(cur.len(), 2);
        let five = terms.mk_int(5);
        let three = terms.mk_int(3);
        let lit_x = terms.mk_eq(cur[0], five);
        let lit_y = terms.mk_eq(cur[1], three);
        let post = terms.mk_and([lit_x, lit_y]);

        let mut spacer = Spacer::new(&mut terms, &system);
        spacer
            .initialize()
            .expect("initialization populates init reach facts");

        let raw_cube = Generalizer::extract_cube(spacer.terms, post);
        assert_eq!(raw_cube.len(), 2, "raw blocking cube has two literals");

        let generalized = spacer
            .mic_generalize_cube(inv, post, 1)
            .expect("MIC generalization should not error");

        assert_eq!(
            generalized.len(),
            1,
            "MIC must drop the irrelevant y-literal, shrinking the cube 2 -> 1"
        );
        assert_eq!(
            generalized[0], lit_x,
            "the retained literal must be the inductive one (x = 5)"
        );
        assert!(
            generalized.len() < raw_cube.len(),
            "generalized lemma must be strictly shorter than the raw cube"
        );
    }
}
