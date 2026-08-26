//! CDCL SAT Solver

mod add_clause;
mod bve;
mod config;
mod conflict;
mod congruence;
mod decide;
mod equiv;
pub mod heuristic;
mod incremental;
mod learn;
mod lrat_trace;
mod lucky;
mod probe;
mod propagate;
mod search_ext;
mod self_subsumption;

pub use config::{RestartStrategy, SolverConfig};
pub use heuristic::{BoxedBranchingHeuristic, BranchingHeuristic};

use crate::chb::CHB;
use crate::chrono::ChronoBacktrack;
use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{LBool, Lit, Var};
use crate::lrb::LRB;
use crate::memory_opt::{MemoryAction, MemoryOptimizer};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::restart_model::{LubyClock, ModeLbdTrackers};
use crate::trail::{Reason, Trail};
use crate::vmtf::VMTF;
use crate::vsids::VSIDS;
use crate::watched::{WatchLists, Watcher};
use core::sync::atomic::{AtomicBool, Ordering};
use smallvec::SmallVec;

/// Binary implication graph for efficient binary clause propagation
/// For each literal L, stores the list of literals that are implied when L is false
/// (i.e., for binary clause (~L v M), when L is assigned false, M must be true)
#[derive(Debug, Clone)]
pub(super) struct BinaryImplicationGraph {
    /// implications[lit] = list of (implied_lit, clause_id) pairs
    implications: Vec<Vec<(Lit, ClauseId)>>,
}

impl BinaryImplicationGraph {
    fn new(num_vars: usize) -> Self {
        Self {
            implications: vec![Vec::new(); num_vars * 2],
        }
    }

    fn resize(&mut self, num_vars: usize) {
        self.implications.resize(num_vars * 2, Vec::new());
    }

    fn add(&mut self, lit: Lit, implied: Lit, clause_id: ClauseId) {
        self.implications[lit.code() as usize].push((implied, clause_id));
    }

    fn get(&self, lit: Lit) -> &[(Lit, ClauseId)] {
        &self.implications[lit.code() as usize]
    }

    fn clear(&mut self) {
        for implications in &mut self.implications {
            implications.clear();
        }
    }

    /// Remove every edge belonging to `clause_id` that is keyed under `trigger`.
    /// Used to purge binary implications when a clause is retracted so the graph
    /// does not accumulate stale (and, after slot reuse, misleading) edges.
    fn remove_clause_edges(&mut self, trigger: Lit, clause_id: ClauseId) {
        let idx = trigger.code() as usize;
        if idx < self.implications.len() {
            self.implications[idx].retain(|(_, cid)| *cid != clause_id);
        }
    }
}

/// Result from a theory check
#[derive(Debug, Clone)]
pub enum TheoryCheckResult {
    /// Theory is satisfied under current assignment
    Sat,
    /// Theory detected a conflict, returns conflict clause literals
    Conflict(SmallVec<[Lit; 8]>),
    /// Theory propagated new literals (lit, reason clause)
    Propagated(Vec<(Lit, SmallVec<[Lit; 8]>)>),
}

/// Callback trait for theory solvers
/// The CDCL(T) solver implements this to receive theory callbacks
pub trait TheoryCallback {
    /// Called when a literal is assigned
    /// Returns a theory check result
    fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult;

    /// Called after propagation is complete to do a full theory check
    fn final_check(&mut self) -> TheoryCheckResult;

    /// Called when the decision level increases
    fn on_new_level(&mut self, _level: u32) {}

    /// Called when backtracking
    fn on_backtrack(&mut self, level: u32);
}

/// Result of SAT solving
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown (e.g., timeout, resource limit)
    Unknown,
}

/// A condition [`Solver`] cannot recover from soundly, distinct from an
/// ordinary `Unsat`/`Unknown` verdict.
///
/// Recorded on the solver (see [`Solver::error`]) rather than threaded
/// through [`Solver::add_clause`]'s or [`Solver::solve_with_assumptions`]'s
/// existing return types: both are long-established public signatures used
/// well beyond this crate, and every legitimate caller of either one hits
/// this only in the one narrow situation each variant documents — changing
/// their shape for that would be a far larger blast radius than one more
/// field to check. Once set, [`Solver::solve`], [`Solver::solve_with_assumptions`],
/// and [`Solver::solve_with_theory`] all refuse to report `Sat` or `Unsat`
/// (either could be wrong) and answer [`SolverResult::Unknown`] instead,
/// until [`Solver::reset`] clears it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverError {
    /// A clause or assumption named a variable that [`Solver::var_eliminated`]
    /// reports as bounded-variable-eliminated: its defining clauses were
    /// resolved away and replaced by their resolvents (see
    /// `bounded_variable_elimination`, a private method not part of this
    /// crate's public API), and this port does not reintroduce such a
    /// variable's original definition back into the live search on demand
    /// the way CaDiCaL's extension-stack replay does.
    /// Equivalent-literal-substituted variables do not hit this: they are
    /// rewritten through the substitution map instead (still logically the
    /// same variable, just expressed via its class representative), which is
    /// sound without any reconstruction at all.
    EliminatedVariableReintroduction {
        /// The variable a later clause or assumption tried to reintroduce.
        var: Var,
    },
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EliminatedVariableReintroduction { var } => write!(
                f,
                "variable {var:?} was eliminated by bounded variable elimination and cannot be \
                 reintroduced by a later add_clause/assumption in this port; avoid mentioning it \
                 again, or disable SolverConfig::enable_bve for incremental use"
            ),
        }
    }
}

impl std::error::Error for SolverError {}

/// Statistics for the solver
#[derive(Debug, Default, Clone)]
pub struct SolverStats {
    /// Number of decisions made
    pub decisions: u64,
    /// Number of propagations
    pub propagations: u64,
    /// Number of conflicts
    pub conflicts: u64,
    /// Number of restarts
    pub restarts: u64,
    /// Number of learned clauses
    pub learned_clauses: u64,
    /// Number of deleted clauses
    pub deleted_clauses: u64,
    /// Number of binary clauses learned
    pub binary_clauses: u64,
    /// Number of unit clauses learned
    pub unit_clauses: u64,
    /// Total LBD of learned clauses
    pub total_lbd: u64,
    /// Number of clause minimizations
    pub minimizations: u64,
    /// Literals removed by minimization
    pub literals_removed: u64,
    /// Number of chronological backtracks
    pub chrono_backtracks: u64,
    /// Number of non-chronological backtracks
    pub non_chrono_backtracks: u64,
    /// Literals removed from live clauses by self-subsuming resolution during
    /// inprocessing (see `Solver::self_subsuming_resolution`).
    pub self_subsumed_literals: u64,
    /// Number of pre-search *lucky* scans attempted (see `solver/lucky.rs`,
    /// a private module). Cumulative across `solve()` calls, like every other
    /// counter here: a single call runs each scan at most once and stops at
    /// the first that succeeds, contributing its wasted guesses plus (on a
    /// hit) one.
    pub lucky_attempts: u64,
    /// Number of lucky scans that produced a verified model, ending the solve
    /// before the CDCL loop ran. At most one per `solve()` call, and — like
    /// [`SolverStats::lucky_attempts`] — summed over all of them.
    pub lucky_successes: u64,
    /// Number of times the on-the-fly lazy hyper-binary resolution pass
    /// (`Solver::check_hyper_binary_resolution`, gated by
    /// [`SolverConfig::enable_lazy_hyper_binary`]) reached the point where it
    /// does real work: the propagation happened at decision level >= 2 with no
    /// proof being traced, *and* the reason clause turned out to be 2 to 4
    /// literals wide. A wider reason is looked at and rejected without being
    /// counted, so this is the number of scans performed rather than the
    /// number of propagations the pass was offered.
    /// Counts *work done*, not clauses produced; pair it with
    /// [`SolverStats::hyper_binary_learned`] to see the hit rate. Cumulative
    /// across `solve()` calls like every other counter here.
    pub hyper_binary_attempts: u64,
    /// Number of binary clauses the pass above actually learned. The
    /// difference against [`SolverStats::hyper_binary_attempts`] is the pass's
    /// wasted work: a reason clause whose literals do not fit the
    /// "exactly one other current-level literal, all the rest false at level
    /// 0" shape, or whose resolvent the binary implication graph already
    /// contains. These clauses are also counted in
    /// [`SolverStats::learned_clauses`].
    pub hyper_binary_learned: u64,
}

impl SolverStats {
    /// Get average LBD of learned clauses
    #[must_use]
    pub fn avg_lbd(&self) -> f64 {
        if self.learned_clauses == 0 {
            0.0
        } else {
            self.total_lbd as f64 / self.learned_clauses as f64
        }
    }

    /// Get average decisions per conflict
    #[must_use]
    pub fn avg_decisions_per_conflict(&self) -> f64 {
        if self.conflicts == 0 {
            0.0
        } else {
            self.decisions as f64 / self.conflicts as f64
        }
    }

    /// Get propagations per conflict
    #[must_use]
    pub fn propagations_per_conflict(&self) -> f64 {
        if self.conflicts == 0 {
            0.0
        } else {
            self.propagations as f64 / self.conflicts as f64
        }
    }

    /// Get clause deletion ratio
    #[must_use]
    pub fn deletion_ratio(&self) -> f64 {
        if self.learned_clauses == 0 {
            0.0
        } else {
            self.deleted_clauses as f64 / self.learned_clauses as f64
        }
    }

    /// Get chronological backtrack ratio
    #[must_use]
    pub fn chrono_backtrack_ratio(&self) -> f64 {
        let total = self.chrono_backtracks + self.non_chrono_backtracks;
        if total == 0 {
            0.0
        } else {
            self.chrono_backtracks as f64 / total as f64
        }
    }

    /// Display formatted statistics
    pub fn display(&self) {
        println!("========== Solver Statistics ==========");
        println!("Decisions:              {:>12}", self.decisions);
        println!("Propagations:           {:>12}", self.propagations);
        println!("Conflicts:              {:>12}", self.conflicts);
        println!("Restarts:               {:>12}", self.restarts);
        println!("Learned clauses:        {:>12}", self.learned_clauses);
        println!("  - Unit clauses:       {:>12}", self.unit_clauses);
        println!("  - Binary clauses:     {:>12}", self.binary_clauses);
        println!("Deleted clauses:        {:>12}", self.deleted_clauses);
        println!("Minimizations:          {:>12}", self.minimizations);
        println!("Literals removed:       {:>12}", self.literals_removed);
        println!("Lucky scans:            {:>12}", self.lucky_attempts);
        println!("  - Lucky hits:         {:>12}", self.lucky_successes);
        println!("Chrono backtracks:      {:>12}", self.chrono_backtracks);
        println!("Non-chrono backtracks:  {:>12}", self.non_chrono_backtracks);
        println!("---------------------------------------");
        println!("Avg LBD:                {:>12.2}", self.avg_lbd());
        println!(
            "Avg decisions/conflict: {:>12.2}",
            self.avg_decisions_per_conflict()
        );
        println!(
            "Propagations/conflict:  {:>12.2}",
            self.propagations_per_conflict()
        );
        println!(
            "Deletion ratio:         {:>12.2}%",
            self.deletion_ratio() * 100.0
        );
        println!(
            "Chrono backtrack ratio: {:>12.2}%",
            self.chrono_backtrack_ratio() * 100.0
        );
        println!("=======================================");
    }
}

/// CDCL SAT Solver
#[derive(Debug)]
pub struct Solver {
    /// Configuration
    pub(super) config: SolverConfig,
    /// Number of variables
    pub(super) num_vars: usize,
    /// Clause database
    pub(super) clauses: ClauseDatabase,
    /// Assignment trail
    pub(super) trail: Trail,
    /// Watch lists
    pub(super) watches: WatchLists,
    /// VSIDS branching heuristic
    pub(super) vsids: VSIDS,
    /// VMTF move-to-front decision queue, used in focused mode when
    /// [`SolverConfig::use_vmtf`] is set (see [`Solver::pick_branch_var`]).
    pub(super) vmtf: VMTF,
    /// CHB branching heuristic
    pub(super) chb: CHB,
    /// LRB branching heuristic
    pub(super) lrb: LRB,
    /// Statistics
    pub(super) stats: SolverStats,
    /// Learnt clause for conflict analysis
    pub(super) learnt: SmallVec<[Lit; 16]>,
    /// Seen flags for conflict analysis
    pub(super) seen: Vec<bool>,
    /// Analyze stack
    pub(super) analyze_stack: Vec<Lit>,
    /// Current restart threshold
    pub(super) restart_threshold: u64,
    /// Assertions stack for incremental solving (number of original clauses)
    pub(super) assertion_levels: Vec<usize>,
    /// Trail sizes at each assertion level (for proper pop backtracking)
    pub(super) assertion_trail_sizes: Vec<usize>,
    /// Clause IDs added at each assertion level (for proper pop)
    pub(super) assertion_clause_ids: Vec<Vec<ClauseId>>,
    /// Model (if sat)
    pub(super) model: Vec<LBool>,
    /// Whether formula is trivially unsatisfiable
    pub(super) trivially_unsat: bool,
    /// Phase saving: last polarity assigned to each variable
    pub(super) phase: Vec<bool>,
    /// Global polarity inversion applied on top of `phase` by rephasing.
    /// Toggled periodically (see [`Solver::restart`]) so a restart explores
    /// the complementary phase region instead of re-deriving the trail it
    /// just abandoned.
    pub(super) phase_inverted: bool,
    /// Polarities captured from the longest trail reached so far (without a
    /// conflict at its final assignment), restored by some rephase rounds to
    /// refocus the search on the best region found instead of a blind flip.
    pub(super) best_phase: Vec<bool>,
    /// Length of the trail that produced `best_phase` (0 until captured).
    pub(super) best_trail_size: usize,
    /// Rephase round counter; alternates which rephase action fires (restore
    /// best vs. invert) on successive rounds.
    pub(super) rephase_count: u64,
    /// Luby sequence index for restarts
    pub(super) luby_index: u64,
    /// Stable/focused search mode. `false` (focused) favors frequent
    /// Glucose-EMA restarts and VMTF decisions; `true` (stable) favors rare
    /// reluctant-doubling restarts, VSIDS decisions, and rephasing. See
    /// [`Solver::check_stabilize`].
    pub(super) stable: bool,
    /// Number of stable/focused switches performed so far; drives the
    /// quadratic growth of each phase's tick budget.
    pub(super) stabphases: u64,
    /// Per-mode LBD trackers for the currently-active mode, swapped with
    /// `glue_saved` on every stable/focused switch so each mode judges
    /// restarts only against samples gathered while it was active.
    pub(super) glue_current: ModeLbdTrackers,
    /// Per-mode LBD trackers for the *other* mode, held here while it is
    /// inactive.
    pub(super) glue_saved: ModeLbdTrackers,
    /// Reluctant-doubling (Luby) restart clock, armed only in stable mode.
    pub(super) reluctant: LubyClock,
    /// Accumulated propagation "ticks" (a cache-line-count proxy for work
    /// done, not wall-clock time) while in focused mode.
    pub(super) ticks_focused: u64,
    /// Accumulated propagation ticks while in stable mode.
    pub(super) ticks_stable: u64,
    /// Next conflict count at which the focused-mode Glucose restart
    /// condition is re-checked (checked every couple of conflicts, not every
    /// single one, to keep the check itself cheap).
    pub(super) lim_restart: u64,
    /// Tick count (in whichever mode is about to become active) at which the
    /// next stable/focused switch is due.
    pub(super) lim_stabilize: u64,
    /// Level marks for LBD computation
    pub(super) level_marks: Vec<u32>,
    /// Current mark counter for LBD computation
    pub(super) lbd_mark: u32,
    /// Learned clause IDs for deletion
    pub(super) learned_clause_ids: Vec<ClauseId>,
    /// Number of conflicts since last clause deletion
    pub(super) conflicts_since_deletion: u64,
    /// PRNG state (xorshift64)
    pub(super) rng_state: u64,
    /// For Glucose-style restarts: average LBD of recent conflicts
    pub(super) recent_lbd_sum: u64,
    /// Number of conflicts contributing to recent_lbd_sum
    pub(super) recent_lbd_count: u64,
    /// Binary implication graph for fast binary clause propagation
    pub(super) binary_graph: BinaryImplicationGraph,
    /// Global average LBD for local restarts
    pub(super) global_lbd_sum: u64,
    /// Number of conflicts contributing to global LBD
    pub(super) global_lbd_count: u64,
    /// Conflicts since last local restart
    pub(super) conflicts_since_local_restart: u64,
    /// Conflicts since last inprocessing
    pub(super) conflicts_since_inprocessing: u64,
    /// Chronological backtracking helper
    pub(super) chrono_backtrack: ChronoBacktrack,
    /// Clause activity bump increment (for MapleSAT-style clause bumping)
    pub(super) clause_bump_increment: f64,
    /// Memory optimizer with size-class pools for clause allocation
    pub(super) memory_optimizer: MemoryOptimizer,
    /// Model-reconstruction stack for pure literals eliminated during
    /// inprocessing. Pure-literal elimination deletes clauses that are only
    /// satisfiable *if* the pure literal is fixed to its polarity; the search
    /// itself may assign the variable the opposite phase, so each recorded
    /// literal is forced to `true` in the reconstructed model (see
    /// [`Solver::save_model`]). At most one polarity per variable is recorded.
    pub(super) pure_literal_reconstruction: Vec<Lit>,
    /// Optional cooperative interrupt flag. When set to `true` by another thread
    /// (e.g. a portfolio coordinator on timeout), the search loop stops at the
    /// next check and returns [`SolverResult::Unknown`]. `None` means no external
    /// interrupt is wired.
    pub(super) interrupt: Option<Arc<AtomicBool>>,
    /// Optional conflict budget. When `Some(n)`, the search loop returns
    /// [`SolverResult::Unknown`] once `n` conflicts have been reached instead of
    /// running unbounded. `None` (the default) means no conflict limit. This is
    /// the resource budget consulted by the CDCL loop and drives, e.g.,
    /// `oxiz-cli --timeout`-style bounded solving.
    pub(super) max_conflicts: Option<u64>,
    /// Optional DRAT proof logger. When `Some`, the CDCL loop emits a DRAT
    /// addition line for every learned clause, a deletion line for every clause
    /// dropped by clause-database reduction / subsumption / vivification /
    /// incremental forgetting, and the empty clause when unconditional UNSAT is
    /// derived. `None` (the default) means no proof is produced and every DRAT
    /// hook is a no-op, so proof logging costs nothing when unused.
    pub(super) drat: Option<crate::proof::DratWriter>,
    /// Optional LRAT proof tracer. When `Some`, every original clause and
    /// every clause the main CDCL loop learns is registered with an LRAT id
    /// and (for learned clauses) an explicit RUP hint chain; every clause
    /// this crate deletes is retracted by id. See `solver/lrat_trace.rs` for
    /// the hint-chain construction and the mechanisms that gate themselves
    /// off rather than run without sound hint coverage while this is set.
    pub(super) lrat: Option<crate::proof::LratWriter>,
    /// `ClauseId -> LRAT id` for every clause LRAT tracing has registered,
    /// indexed by `ClauseId.0`. `0` (never a real `LratWriter` id) marks an
    /// unregistered slot. Empty and untouched whenever `lrat` is `None`.
    pub(super) clause_lrat_id: Vec<u64>,
    /// `Var::index() -> LRAT id` of the original unit clause that justifies
    /// a variable's level-0 value when [`Solver::add_clause`]'s unit-clause
    /// path installed it via a plain decision (no [`ClauseId`] exists for
    /// such a fact — see `solver/lrat_trace.rs`'s module doc comment for why
    /// this separate table is necessary). `0` marks "no such justification
    /// recorded" (a genuine branching decision, or LRAT tracing is off).
    pub(super) lrat_unit_id: Vec<u64>,
    /// One-shot latch: this incarnation's LRAT proof has already emitted (or
    /// been given, as a literal empty original clause) its conclusion.
    /// Prevents every `solve()` UNSAT exit path from independently trying to
    /// emit a second, redundant empty-clause line.
    pub(super) lrat_unsat_finalized: bool,
    /// One-shot latch: the pre-search inprocessing toolkit (probing / BVE /
    /// equivalent-literal substitution) has already run for this incarnation
    /// of the solver. Cleared by [`Solver::reset`] and by popping back to the
    /// base assertion level, so a reused incremental solver re-runs the
    /// toolkit against its new formula rather than silently skipping it.
    pub(super) bve_latched: bool,
    /// Same one-shot latch as `bve_latched`, for equivalent-literal substitution.
    pub(super) equiv_fold_latched: bool,
    /// Per-literal representative map built by equivalent-literal
    /// substitution, indexed by [`Lit::code`]. Identity (`sub[l] == l`) for
    /// every literal not folded into another's equivalence class. See
    /// [`Solver::fold_equivalent_literals`].
    pub(super) equiv_substitution: Vec<Lit>,
    /// Whether `equiv_substitution` has been sized/initialized to identity for
    /// the current `num_vars`. Distinguishes "never ran" from "ran and found
    /// no equivalences" (both leave the map's *content* looking like the
    /// identity map, but only the former should skip the reconstruction pass
    /// entirely).
    pub(super) equiv_substitution_sized: bool,
    /// Per-eliminated-variable definitional clauses recorded by bounded
    /// variable elimination, keyed by [`Var::index`]: the clauses that
    /// contained the variable positively, with that literal stripped off,
    /// captured *before* the originals were deleted. Empty for a variable
    /// that was never eliminated. Used by [`Solver::save_model`] to derive
    /// the eliminated variable's correct value — see
    /// [`Solver::bounded_variable_elimination`].
    pub(super) bve_def: Vec<Vec<SmallVec<[Lit; 4]>>>,
    /// Variables eliminated by bounded variable elimination, in elimination
    /// order. Reconstructed in reverse (see [`Solver::save_model`]) because a
    /// variable eliminated earlier in the pass can only be defined in terms of
    /// variables eliminated later (once removed, a variable can no longer
    /// appear in a subsequent resolvent).
    pub(super) bve_order: Vec<Var>,
    /// Set once and never cleared except by [`Solver::reset`]: a later
    /// `add_clause`/assumption tried to reintroduce a bounded-variable-
    /// eliminated variable with no sound way to honor it (see
    /// [`SolverError::EliminatedVariableReintroduction`]). Every `solve*`
    /// entry point checks this before doing any work and answers
    /// [`SolverResult::Unknown`] rather than a verdict that might be wrong.
    pub(super) fatal_error: Option<SolverError>,
    /// Propagation step budget consulted by [`Solver::propagate`] when
    /// `Some`; decremented per watch-list/binary-edge visit and, on
    /// exhaustion, aborts the in-flight propagation early (see
    /// [`Solver::propagate_bounded`]). `None` means unbounded (the default
    /// search path). Used by probing so a single pathological cascade cannot
    /// dominate the whole pass.
    pub(super) propagate_step_limit: Option<u64>,
    /// Set by [`Solver::propagate`] when a bounded propagation
    /// ([`Solver::propagate_bounded`]) hit `propagate_step_limit` before
    /// reaching a fixpoint. The caller must treat the probe as inconclusive —
    /// neither a genuine conflict nor a complete model.
    pub(super) propagate_aborted: bool,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    /// Create a new solver
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new solver with configuration
    #[must_use]
    pub fn with_config(config: SolverConfig) -> Self {
        let chrono_enabled = config.enable_chronological_backtrack;
        let chrono_threshold = config.chrono_backtrack_threshold;

        Self {
            restart_threshold: config.restart_interval,
            config,
            num_vars: 0,
            clauses: ClauseDatabase::new(),
            trail: Trail::new(0),
            watches: WatchLists::new(0),
            vsids: VSIDS::new(0),
            vmtf: VMTF::new(0),
            chb: CHB::new(0),
            lrb: LRB::new(0),
            stats: SolverStats::default(),
            learnt: SmallVec::new(),
            seen: Vec::new(),
            analyze_stack: Vec::new(),
            assertion_levels: vec![0],
            assertion_trail_sizes: vec![0],
            assertion_clause_ids: vec![Vec::new()],
            model: Vec::new(),
            trivially_unsat: false,
            phase: Vec::new(),
            phase_inverted: false,
            best_phase: Vec::new(),
            best_trail_size: 0,
            rephase_count: 0,
            luby_index: 0,
            stable: false,
            stabphases: 0,
            glue_current: ModeLbdTrackers::new(),
            glue_saved: ModeLbdTrackers::new(),
            reluctant: LubyClock::default(),
            ticks_focused: 0,
            ticks_stable: 0,
            lim_restart: 0,
            lim_stabilize: 0,
            level_marks: Vec::new(),
            lbd_mark: 0,
            learned_clause_ids: Vec::new(),
            conflicts_since_deletion: 0,
            rng_state: 0x853c_49e6_748f_ea9b, // Random seed
            recent_lbd_sum: 0,
            recent_lbd_count: 0,
            binary_graph: BinaryImplicationGraph::new(0),
            global_lbd_sum: 0,
            global_lbd_count: 0,
            conflicts_since_local_restart: 0,
            conflicts_since_inprocessing: 0,
            chrono_backtrack: ChronoBacktrack::new(chrono_enabled, chrono_threshold),
            clause_bump_increment: 1.0,
            memory_optimizer: MemoryOptimizer::new(),
            pure_literal_reconstruction: Vec::new(),
            interrupt: None,
            max_conflicts: None,
            drat: None,
            lrat: None,
            clause_lrat_id: Vec::new(),
            lrat_unit_id: Vec::new(),
            lrat_unsat_finalized: false,
            bve_latched: false,
            equiv_fold_latched: false,
            equiv_substitution: Vec::new(),
            equiv_substitution_sized: false,
            bve_def: Vec::new(),
            bve_order: Vec::new(),
            fatal_error: None,
            propagate_step_limit: None,
            propagate_aborted: false,
        }
    }

    /// Enable DRAT proof logging to `path`.
    ///
    /// While enabled, the CDCL search emits a DRAT proof: one addition line per
    /// learned clause, one deletion line per clause removed by database
    /// reduction / subsumption / vivification / incremental forgetting, and the
    /// empty clause when unconditional UNSAT is derived. The resulting file can
    /// be checked by any DRAT proof checker. Enabling it does not change the
    /// search itself — only whether the trace is recorded.
    pub fn enable_drat_proof(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut writer = crate::proof::DratWriter::new();
        writer.enable(path)?;
        self.drat = Some(writer);
        Ok(())
    }

    /// Disable DRAT proof logging (flushing any buffered output).
    pub fn disable_drat_proof(&mut self) {
        if let Some(mut writer) = self.drat.take() {
            writer.disable();
        }
    }

    /// Returns `true` when DRAT proof logging is currently enabled.
    #[must_use]
    pub fn drat_proof_enabled(&self) -> bool {
        self.drat.is_some()
    }

    /// Emit a DRAT addition line for `lits` (no-op when proof logging is off).
    pub(super) fn drat_add(&mut self, lits: &[Lit]) {
        if let Some(writer) = &mut self.drat {
            let _ = writer.add_clause(lits);
        }
    }

    /// Emit a DRAT deletion line for the clause `clause_id`, reading its literals
    /// before it is removed from the database (no-op when proof logging is off or
    /// the clause is already gone).
    pub(super) fn drat_delete(&mut self, clause_id: ClauseId) {
        if self.drat.is_none() {
            return;
        }
        let lits: Option<SmallVec<[Lit; 8]>> = self.clauses.get(clause_id).and_then(|c| {
            if c.deleted {
                None
            } else {
                Some(c.lits.iter().copied().collect())
            }
        });
        if let (Some(writer), Some(lits)) = (&mut self.drat, lits) {
            let _ = writer.delete_clause(&lits);
        }
    }

    /// Emit a DRAT deletion line for an explicit literal set (used when a clause
    /// is strengthened in place and its pre-strengthening form must be retired).
    pub(super) fn drat_delete_lits(&mut self, lits: &[Lit]) {
        if let Some(writer) = &mut self.drat {
            let _ = writer.delete_clause(lits);
        }
    }

    /// Emit the empty clause (the DRAT proof of unconditional UNSAT).
    pub(super) fn drat_emit_empty(&mut self) {
        if let Some(writer) = &mut self.drat {
            let _ = writer.add_clause(&[]);
        }
    }

    /// Purge every binary-implication-graph edge belonging to `clause_id`.
    ///
    /// The binary graph is a direct-index fast path over binary clauses; unlike
    /// the watch lists (which lazily skip deleted clauses) it is consulted
    /// without a liveness check at its call sites' hot loop, so a retracted
    /// binary clause must have its edges physically removed. Reads the clause's
    /// literals, so it must run *before* the clause is removed from the database.
    pub(super) fn purge_binary_edges(&mut self, clause_id: ClauseId) {
        let binary_lits = self.clauses.get(clause_id).and_then(|c| {
            if c.lits.len() == 2 && !c.deleted {
                Some((c.lits[0], c.lits[1]))
            } else {
                None
            }
        });
        if let Some((a, b)) = binary_lits {
            self.binary_graph.remove_clause_edges(a.negate(), clause_id);
            self.binary_graph.remove_clause_edges(b.negate(), clause_id);
        }
    }

    /// Attach a cooperative interrupt flag.
    ///
    /// While solving, the CDCL loop periodically checks this flag; if another
    /// thread sets it to `true`, the current `solve*` call abandons the search
    /// and returns [`SolverResult::Unknown`]. Combined with
    /// [`Solver::set_max_conflicts`], this lets callers bound solving by both
    /// wall-clock time (via an external timer that sets the flag) and work.
    pub fn set_interrupt(&mut self, flag: Arc<AtomicBool>) {
        self.interrupt = Some(flag);
    }

    /// Set the conflict budget (`None` clears it). When set, the CDCL search
    /// loop returns [`SolverResult::Unknown`] once the budget is reached.
    pub fn set_max_conflicts(&mut self, max_conflicts: Option<u64>) {
        self.max_conflicts = max_conflicts;
    }

    /// Returns `true` when the search must stop early: the conflict budget has
    /// been reached or an external interrupt flag has been raised.
    #[inline]
    pub(super) fn should_stop_search(&self) -> bool {
        if let Some(max) = self.max_conflicts
            && self.stats.conflicts >= max
        {
            return true;
        }
        if let Some(flag) = &self.interrupt
            && flag.load(Ordering::Relaxed)
        {
            return true;
        }
        false
    }

    /// Create a new variable
    pub fn new_var(&mut self) -> Var {
        let var = Var::new(self.num_vars as u32);
        self.num_vars += 1;
        self.trail.resize(self.num_vars);
        self.watches.resize(self.num_vars);
        self.binary_graph.resize(self.num_vars);
        self.vsids.insert(var);
        self.vmtf.resize(self.num_vars);
        self.chb.insert(var);
        self.lrb.resize(self.num_vars);
        self.seen.resize(self.num_vars, false);
        self.model.resize(self.num_vars, LBool::Undef);
        self.phase.resize(self.num_vars, false); // Default phase: negative
        self.best_phase.resize(self.num_vars, false);
        // Resize level_marks to at least num_vars (enough for decision levels)
        if self.level_marks.len() < self.num_vars {
            self.level_marks.resize(self.num_vars, 0);
        }
        var
    }

    /// Ensure we have at least n variables
    pub fn ensure_vars(&mut self, n: usize) {
        while self.num_vars < n {
            self.new_var();
        }
    }

    /// Decay every learned clause's activity and grow the bump increment
    /// (MiniSat-style: cheap "bump" via a growing increment instead of
    /// touching every clause on every conflict), then guard against the
    /// increment overflowing `f64`.
    ///
    /// The increment is divided by `clause_decay` (< 1) on every conflict, so
    /// after enough conflicts on a long run it would approach `f64::MAX` and
    /// start producing `inf`/`NaN` activities — silently breaking the
    /// tier/reduction heuristic's ordering (never soundness: activity only
    /// picks which learned clauses survive deletion). Rescaling every
    /// activity and the increment itself by a small constant factor resets
    /// the growth without changing any clause's *relative* activity order.
    /// The rescale threshold is far below `f64::MAX` so this always fires
    /// before precision is actually lost, and is rare enough (roughly every
    /// ~10^5-10^6 conflicts at the default 0.999 decay) that its O(n) cost is
    /// amortized away.
    pub(super) fn decay_clause_activity(&mut self) {
        self.clauses.decay_activity(self.config.clause_decay);
        self.clause_bump_increment /= self.config.clause_decay;
        const RESCALE_THRESHOLD: f64 = 1e100;
        const RESCALE_FACTOR: f64 = 1e-100;
        if self.clause_bump_increment > RESCALE_THRESHOLD {
            self.clauses.rescale_activity(RESCALE_FACTOR);
            self.clause_bump_increment *= RESCALE_FACTOR;
        }
    }

    /// Solve the SAT problem
    pub fn solve(&mut self) -> SolverResult {
        // A prior `add_clause` tried to reintroduce a bounded-variable-
        // eliminated variable with no sound way to honor it (see
        // `SolverError::EliminatedVariableReintroduction`). Answering `Sat`
        // or `Unsat` here could be wrong in either direction — refuse rather
        // than guess. `self.error()` explains why to a caller that checks.
        if self.fatal_error.is_some() {
            return SolverResult::Unknown;
        }

        // Check if trivially unsatisfiable
        if self.trivially_unsat {
            self.drat_emit_empty();
            // Always a no-op in practice by this point: every path that can
            // set `trivially_unsat` before `solve()` is even called (every
            // branch of `add_clause`) already finalized the LRAT proof of
            // its own accord — see `lrat_emit_empty_from`'s idempotency
            // guard. Called anyway for defensive completeness, mirroring
            // `drat_emit_empty` immediately above.
            self.lrat_emit_empty_from(&[], 0);
            return SolverResult::Unsat;
        }

        // Initial propagation
        if let Some(conflict) = self.propagate() {
            self.drat_emit_empty();
            let seed_lits: SmallVec<[Lit; 8]> = self
                .clauses
                .get(conflict)
                .map(|c| c.lits.iter().copied().collect())
                .unwrap_or_default();
            let final_id = self.lrat_id_of(conflict).unwrap_or(0);
            self.lrat_emit_empty_from(&seed_lits, final_id);
            return SolverResult::Unsat;
        }

        // Cheapest thing in the pre-search pipeline, and therefore first: a
        // handful of structural guesses at a full assignment (all-true,
        // all-false, greedy clause-ordered passes, the Horn/dual-Horn
        // closures), each verified against the original clauses before it can
        // be reported. See `solver/lucky.rs` — in particular for why this is
        // sound to run unconditionally, ahead of the mechanisms below, and
        // even while a proof is being traced: it touches nothing, and a `Sat`
        // verdict carries no proof obligation. On a hit the inprocessing
        // toolkit below never runs, which is the entire point (the instances
        // this catches are solved before probing would have finished its
        // first variable) but is also why a caller that depends on those
        // mechanisms having run must set `enable_lucky_phase: false`.
        if self.try_lucky_phase(&[]).is_some() {
            return SolverResult::Sat;
        }

        // One-shot inprocessing toolkit, run once before search proper
        // starts. Each mechanism below is individually opt-in (see the
        // `SolverConfig` fields it checks) and latches itself so calling it
        // here unconditionally is safe even when disabled or already run —
        // see each one's own doc comment for its guard conditions. Bundled
        // together in one place because their guards and effects compose:
        // probing can only add facts (never removes a variable, so it is
        // safe to run regardless of what the other two decide), while
        // equivalent-literal substitution and bounded variable elimination
        // are mutually exclusive with each other by construction (see
        // `Solver::fold_equivalent_literals`'s doc comment). Bails out
        // immediately on any UNSAT verdict rather than running the remaining
        // mechanisms against an already-contradictory formula.
        if self.config.enable_failed_literal_probing {
            self.failed_literal_probing();
            self.probe_hyper_binaries();
            if self.trivially_unsat {
                self.drat_emit_empty();
                // Unreachable while any proof is being traced: probing
                // gates itself off entirely in that case (see
                // `Solver::failed_literal_probing`). Present for defensive
                // completeness, not because this path is expected to run.
                self.lrat_emit_empty_from(&[], 0);
                return SolverResult::Unsat;
            }
        }
        if matches!(
            self.fold_equivalent_literals(),
            equiv::PreprocessOutcome::Unsat
        ) {
            self.trivially_unsat = true;
            self.drat_emit_empty();
            // Unreachable while any proof is being traced (ELS gates itself
            // off — see `Solver::fold_equivalent_literals`); present
            // for defensive completeness only.
            self.lrat_emit_empty_from(&[], 0);
            return SolverResult::Unsat;
        }
        self.bounded_variable_elimination();
        if self.trivially_unsat {
            self.drat_emit_empty();
            // Unreachable while any proof is being traced (BVE gates itself
            // off — see `Solver::bounded_variable_elimination`); present for
            // defensive completeness only.
            self.lrat_emit_empty_from(&[], 0);
            return SolverResult::Unsat;
        }
        // BVE's resolvents commonly subsume an original clause they were
        // derived from (or each other); sweep them with the existing
        // subsumption pass so the pass actually shrinks the database instead
        // of just replacing clauses one-for-one. Investigated whether this
        // needs its own reason-clause guard the way the PR's equivalent pass
        // has one: it does not, here — `ClauseDatabase::add` never reuses a
        // freed slot and a clause marked `deleted` keeps its literals intact
        // forever (see its doc comment), so a reason lookup for an
        // already-made trail assignment stays valid even after the clause it
        // points to is marked deleted. Only run when BVE has eliminated at
        // least one variable (in this call or an earlier one on the same
        // solver), skipping the `Preprocessor` allocation and clause scan on
        // every ordinary solve where BVE is disabled or found nothing to do.
        if !self.bve_order.is_empty() {
            let mut preprocessor = crate::Preprocessor::new(self.num_vars);
            preprocessor.subsumption_elimination(&mut self.clauses);
            self.rebuild_propagation_index();
        }

        loop {
            // Resource budget / interrupt check: honor a configured conflict
            // limit or an external interrupt by returning Unknown rather than
            // spinning forever on a hard instance.
            if self.should_stop_search() {
                return SolverResult::Unknown;
            }

            // Propagate
            if let Some(conflict) = self.propagate() {
                self.debug_check_conflict_clause(conflict);
                self.stats.conflicts += 1;
                self.conflicts_since_inprocessing += 1;

                if self.trail.decision_level() == 0 {
                    // Conflict under only level-0 (unconditional) facts: the empty
                    // clause is derivable, completing the DRAT proof of UNSAT.
                    self.drat_emit_empty();
                    let seed_lits: SmallVec<[Lit; 8]> = self
                        .clauses
                        .get(conflict)
                        .map(|c| c.lits.iter().copied().collect())
                        .unwrap_or_default();
                    let final_id = self.lrat_id_of(conflict).unwrap_or(0);
                    self.lrat_emit_empty_from(&seed_lits, final_id);
                    return SolverResult::Unsat;
                }

                // Analyze conflict
                let (backtrack_level, learnt_clause) = self.analyze(conflict);

                // Empty learned clause = genuine root-level (level-0) refutation:
                // every conflict literal is false under unconditional facts, so
                // the instance is UNSAT and the empty clause completes the DRAT
                // proof. `analyze` can report this even above decision level 0
                // when a clause is falsified purely at the root.
                if learnt_clause.is_empty() {
                    self.trivially_unsat = true;
                    self.drat_emit_empty();
                    let seed_lits: SmallVec<[Lit; 8]> = self
                        .clauses
                        .get(conflict)
                        .map(|c| c.lits.iter().copied().collect())
                        .unwrap_or_default();
                    let final_id = self.lrat_id_of(conflict).unwrap_or(0);
                    self.lrat_emit_empty_from(&seed_lits, final_id);
                    return SolverResult::Unsat;
                }

                // Compute the LRAT hint chain *before* backtracking: it
                // walks the trail as it stands right now, and backtracking
                // is about to unassign exactly the literals a non-trivial
                // backtrack level was computed to discard — often the same
                // literals this chain needs to cite. `None` when LRAT
                // tracing is off.
                let lrat_hints = self.lrat_hints_for_conflict(conflict, &learnt_clause);

                // Backtrack with phase saving
                self.backtrack_with_phase_saving(backtrack_level);
                self.debug_check_invariants("after backtrack");

                // Emit the learned clause as a DRAT addition (RUP-derivable from
                // the current database by construction of 1-UIP learning). Covers
                // both the unit and general learned-clause branches below.
                self.drat_add(&learnt_clause);
                // Same clause, registered with LRAT using the hint chain
                // captured above — a no-op returning `None` when LRAT
                // tracing is off. The returned id is stashed here and
                // attached to whichever `clause_id` ClauseDatabase assigns
                // below (unit or general branch).
                let learned_lrat_id =
                    lrat_hints.and_then(|hints| self.lrat_finish_learn(&learnt_clause, &hints));

                // Learn clause
                if learnt_clause.len() == 1 {
                    // Store unit learned clause in database for persistence
                    let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
                    if let Some(id) = learned_lrat_id {
                        self.lrat_set_clause_id(clause_id, id);
                        // `assert_learned_clause` below installs this fact via
                        // `Trail::assign_unit_fact`, which — like
                        // `add_clause`'s own unit-clause path — records
                        // `Reason::Decision` rather than
                        // `Reason::Propagation(clause_id)` (a unit fact has
                        // no watched literals to propagate through). Without
                        // this, `lrat_hint_chain`'s trail replay would see a
                        // plain decision here and cite no justification at
                        // all for a variable that is very much not free.
                        self.lrat_set_unit_justification(learnt_clause[0].var(), id);
                    }
                    self.stats.learned_clauses += 1;
                    self.stats.unit_clauses += 1;
                    self.learned_clause_ids.push(clause_id);

                    // Track for incremental solving
                    if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                        current_level_clauses.push(clause_id);
                    }

                    self.assert_learned_clause(&learnt_clause, clause_id);
                } else {
                    // Compute LBD for the learned clause
                    let lbd = self.compute_lbd(&learnt_clause);

                    // Track recent LBD for Glucose-style and local restarts
                    self.recent_lbd_sum += u64::from(lbd);
                    self.recent_lbd_count += 1;
                    self.global_lbd_sum += u64::from(lbd);
                    self.global_lbd_count += 1;

                    // Feed the active mode's glue trackers (the stable/focused
                    // schedule's own quality signal, independent of the
                    // legacy `recent_lbd_sum`/`global_lbd_sum` counters above)
                    // and advance the stable-mode reluctant-doubling clock —
                    // it ticks once per conflict regardless of which mode is
                    // active so its sequence position stays meaningful across
                    // a focused stretch, ready to resume the instant stable
                    // mode is re-entered.
                    let glue = f64::from(lbd);
                    self.glue_current.recent.observe(glue);
                    self.glue_current.baseline.observe(glue);
                    self.reluctant.tick();

                    // Reset recent LBD tracking periodically
                    if self.recent_lbd_count >= 5000 {
                        self.recent_lbd_sum /= 2;
                        self.recent_lbd_count /= 2;
                    }

                    let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
                    if let Some(id) = learned_lrat_id {
                        self.lrat_set_clause_id(clause_id, id);
                    }
                    self.stats.learned_clauses += 1;

                    // Set LBD score for the clause
                    if let Some(clause) = self.clauses.get_mut(clause_id) {
                        clause.lbd = lbd;
                    }
                    self.debug_check_learned_clause_lbd(clause_id);

                    // Track learned clause for potential deletion
                    self.learned_clause_ids.push(clause_id);

                    // Track clause for incremental solving
                    if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                        current_level_clauses.push(clause_id);
                    }

                    // Watch first two literals
                    let lit0 = learnt_clause[0];
                    let lit1 = learnt_clause[1];
                    self.watches
                        .add(lit0.negate(), Watcher::new(clause_id, lit1));
                    self.watches
                        .add(lit1.negate(), Watcher::new(clause_id, lit0));

                    // Propagate the asserting literal at its true implication
                    // level (see `Solver::assert_learned_clause`).
                    self.assert_learned_clause(&learnt_clause, clause_id);
                }

                // Decay activities
                self.vsids.decay();
                self.chb.decay();
                self.lrb.decay();
                self.lrb.on_conflict();
                self.decay_clause_activity();

                // Track conflicts for clause deletion
                self.conflicts_since_deletion += 1;

                // Periodic clause database reduction
                if self.conflicts_since_deletion >= self.config.clause_deletion_threshold as u64 {
                    self.reduce_clause_database();
                    self.debug_check_invariants("after clause database reduction");
                    self.conflicts_since_deletion = 0;

                    // Vivification after clause database reduction (at level 0 after restart)
                    if self.stats.restarts.is_multiple_of(10) {
                        let saved_level = self.trail.decision_level();
                        if saved_level == 0 {
                            self.vivify_clauses();
                        }
                    }
                }

                // Stable/focused mode switch (see `Solver::check_stabilize`).
                // Checked before the restart decision so the *new* mode (if a
                // switch just happened) governs this restart's condition.
                self.check_stabilize();

                // Restart decision: the stable/focused schedule, when enabled,
                // replaces the plain threshold check with a per-mode condition
                // (focused: Glucose EMA divergence; stable: reluctant-doubling
                // clock). With the schedule disabled, fall back to the legacy
                // per-strategy threshold (`restart_threshold`, computed by
                // `Solver::restart`'s `match` on `restart_strategy`).
                let should_restart = if self.config.enable_stabilize {
                    if self.stable {
                        self.reluctant.due()
                    } else {
                        // The Glucose EMA condition is cheap but not free;
                        // check it only once every couple of conflicts rather
                        // than on every single one.
                        if self.stats.conflicts < self.lim_restart {
                            false
                        } else {
                            self.lim_restart = self.stats.conflicts.saturating_add(2);
                            let slow = self.glue_current.baseline.get();
                            let fast = self.glue_current.recent.get();
                            // 10% margin against the baseline; guard the
                            // all-zero startup state (no samples yet).
                            slow > 0.0 && fast >= 1.10 * slow
                        }
                    }
                } else {
                    self.stats.conflicts >= self.restart_threshold
                };
                if should_restart {
                    self.restart();
                    // Reuse-trail restarts stop above level 0 by design, so
                    // the level-0 invariant `debug_check_restart_consistency`
                    // checks does not apply to them.
                    if !self.config.reuse_trail {
                        self.debug_check_restart_consistency();
                    }
                }

                // Periodic inprocessing
                if self.config.enable_inprocessing
                    && self.conflicts_since_inprocessing >= self.config.inprocessing_interval
                {
                    self.inprocess();
                    self.conflicts_since_inprocessing = 0;
                }
            } else {
                // No conflict - try to decide. `propagate()` just returned `None`,
                // i.e. reached a fixpoint, which is exactly where the watched-literal
                // and unit-propagation-completeness invariants become meaningful.
                self.debug_check_fixpoint_invariants("after propagation fixpoint");
                if let Some(var) = self.pick_branch_var() {
                    self.stats.decisions += 1;
                    self.trail.new_decision_level();

                    // Use phase saving with random polarity, flipped by the
                    // rephase inversion flag so a restart following a rephase
                    // round explores the complementary region instead of
                    // re-deriving the trail it just abandoned.
                    let polarity = if self.rand_bool(self.config.random_polarity_prob) {
                        // Random polarity
                        self.rand_bool(0.5)
                    } else {
                        // Saved phase, optionally inverted by rephasing
                        self.phase[var.index()] ^ self.phase_inverted
                    };
                    let lit = if polarity {
                        Lit::pos(var)
                    } else {
                        Lit::neg(var)
                    };
                    self.trail.assign_decision(lit);
                } else {
                    // All variables assigned - SAT
                    self.save_model();
                    self.debug_verify_model();
                    self.debug_check_invariants("at SAT");
                    return SolverResult::Sat;
                }
            }
        }
    }

    /// Solve with assumptions and return unsat core if UNSAT
    ///
    /// This is the key method for MaxSAT: it solves under assumptions and
    /// if the result is UNSAT, returns the subset of assumptions in the core.
    ///
    /// # Arguments
    /// * `assumptions` - Literals that must be true
    ///
    /// # Returns
    /// * `(SolverResult, Option<Vec<Lit>>)` - Result and unsat core (if UNSAT)
    ///
    /// # LRAT tracing is unsupported here
    ///
    /// This entry point's clause-learning goes through `Solver::learn_clause`
    /// (a private method, not part of this crate's public API) rather than
    /// the plain [`Solver::solve`] loop's hint-chain-aware inline
    /// path, and an assumption literal is installed without going through
    /// [`Solver::add_clause`] (so it has no original-clause LRAT id to be
    /// justified by regardless). Rather than emit an LRAT proof this port
    /// cannot back with a real hint chain, LRAT tracing is force-disabled the
    /// instant this entry point runs (DRAT is unaffected — `learn_clause`
    /// already emits it correctly, self-justifying, independent of this gap).
    ///
    /// # Unsat core is expressed in the caller's own literals
    ///
    /// Internally, an assumption may be rewritten before it is decided on
    /// (see `resolve_reintroduced_literal`, a private method not part of
    /// this crate's public API: an equivalent-literal-substituted variable
    /// becomes its class representative). A core drawn from those *resolved*
    /// literals is translated back to the caller's originals (via
    /// `translate_core_to_original`, likewise private) before this method
    /// returns, so `core ⊆ assumptions` — a genuine subset of exactly what
    /// was passed in — always holds, matching what a MaxSAT-style caller
    /// keying relaxations on its own selector literals expects.
    pub fn solve_with_assumptions(
        &mut self,
        assumptions: &[Lit],
    ) -> (SolverResult, Option<Vec<Lit>>) {
        self.disable_lrat_proof();
        // See `Solver::solve`'s identical guard: a prior `add_clause` may
        // have tried to reintroduce a bounded-variable-eliminated variable.
        if self.fatal_error.is_some() {
            return (SolverResult::Unknown, None);
        }
        if self.trivially_unsat {
            return (SolverResult::Unsat, Some(Vec::new()));
        }

        // Ensure all assumption variables exist
        for &lit in assumptions {
            while self.num_vars <= lit.var().index() {
                self.new_var();
            }
        }

        // Resolve each assumption literal exactly like a fresh `add_clause`
        // literal (see `Self::resolve_reintroduced_literal`): an
        // equivalent-literal-substituted variable is rewritten to its class
        // representative — still the same constraint, since the map exists
        // only because that equivalence was already proven — and a
        // bounded-variable-eliminated one has no sound rewrite available and
        // poisons the solver instead of guessing (see `SolverError`).
        // Everything below decides on and analyzes these *resolved*
        // literals, but any unsat core handed back to the caller is
        // translated back to `original_assumptions` (see
        // `Self::translate_core_to_original`) before it is returned: a
        // caller of this MaxSAT-style API expects the core to be a genuine
        // subset of exactly what it passed in, not a class representative it
        // never mentioned.
        let original_assumptions = assumptions;
        let mut resolved_assumptions: Vec<Lit> = Vec::with_capacity(assumptions.len());
        for &lit in assumptions {
            match self.resolve_reintroduced_literal(lit) {
                Some(resolved) => resolved_assumptions.push(resolved),
                None => return (SolverResult::Unknown, None),
            }
        }
        let assumptions: &[Lit] = &resolved_assumptions;

        // A prior solve() may have returned Sat while leaving its full model on the
        // trail (decisions at levels > 0). Fully restart the search state by
        // backtracking to the root BEFORE capturing `assumption_level_start` and
        // testing the assumptions. Otherwise leftover model decisions masquerade as
        // fixed level-0 facts: an assumption that merely disagrees with the previous
        // arbitrary model would hit `value.is_false()` below and be reported as a
        // false UNSAT (e.g. (a∨b); solve() picks ¬a,b; then assumptions=[a] must be
        // SAT, not UNSAT). This is the standard incremental / MaxSAT entry protocol.
        self.backtrack_with_phase_saving(0);

        // Clear conflict-analysis marks so a stale `seen` array left by a previous
        // solve cannot pollute the extracted assumption core.
        for s in &mut self.seen {
            *s = false;
        }

        // Initial propagation at level 0
        if self.propagate().is_some() {
            return (SolverResult::Unsat, Some(Vec::new()));
        }

        // Lucky phase, the assumption-aware variant: the resolved assumption
        // literals are seeded into the candidate as frozen values, so a model
        // it reports satisfies them by construction (see `solver/lucky.rs`).
        // Placed after the level-0 propagation that establishes the facts it
        // freezes and before the first assumption decision below, so a hit
        // returns without ever touching the trail — there is nothing to
        // backtrack, and no core is owed because the phase never concludes
        // UNSAT. It declines outright when an assumption contradicts a level-0
        // fact, leaving that verdict (and its core) to the loop below.
        //
        // Unlike `solve()` this entry point runs no inprocessing at all, so
        // nothing downstream is skipped by a hit.
        if self.try_lucky_phase(assumptions).is_some() {
            return (SolverResult::Sat, None);
        }

        // Create a new decision level for assumptions
        let assumption_level_start = self.trail.decision_level();

        // Assign assumptions as decisions
        for (i, &lit) in assumptions.iter().enumerate() {
            // Check if already assigned
            let value = self.trail.lit_value(lit);
            if value.is_true() {
                continue; // Already satisfied
            }
            if value.is_false() {
                // Conflict with assumption - extract core from conflicting assumptions
                let core = self.extract_assumption_core(assumptions, i);
                self.backtrack(assumption_level_start);
                let core =
                    Self::translate_core_to_original(core, assumptions, original_assumptions);
                return (SolverResult::Unsat, Some(core));
            }

            // Make decision for assumption
            self.trail.new_decision_level();
            self.trail.assign_decision(lit);

            // Propagate after each assumption
            if let Some(conflict) = self.propagate() {
                // Conflict during assumption propagation: collect the full set of
                // contributing assumptions from the conflict clause.
                let core = self.analyze_assumption_conflict(assumptions, conflict);
                self.backtrack(assumption_level_start);
                let core =
                    Self::translate_core_to_original(core, assumptions, original_assumptions);
                return (SolverResult::Unsat, Some(core));
            }
        }

        // Now solve normally
        loop {
            // Resource budget / interrupt check: abandon under-assumption search
            // and report Unknown when the conflict budget or interrupt fires.
            if self.should_stop_search() {
                self.backtrack(assumption_level_start);
                return (SolverResult::Unknown, None);
            }

            if let Some(conflict) = self.propagate() {
                self.debug_check_conflict_clause(conflict);
                self.stats.conflicts += 1;

                // Check if conflict involves assumptions
                let backtrack_level = self.analyze_conflict_level(conflict);

                if backtrack_level <= assumption_level_start {
                    // Conflict forces backtracking past assumptions - UNSAT
                    let core = self.analyze_assumption_conflict(assumptions, conflict);
                    self.backtrack(assumption_level_start);
                    let core =
                        Self::translate_core_to_original(core, assumptions, original_assumptions);
                    return (SolverResult::Unsat, Some(core));
                }

                let (bt_level, learnt_clause) = self.analyze(conflict);

                // Empty learned clause = genuine root-level (level-0) refutation.
                // The `backtrack_level <= assumption_level_start` guard above
                // already routes all-level-0 conflicts to the UNSAT-core path, so
                // this is a belt-and-braces guard that also avoids an empty-clause
                // index panic in `learn_clause`.
                if learnt_clause.is_empty() {
                    let core = self.analyze_assumption_conflict(assumptions, conflict);
                    self.backtrack(assumption_level_start);
                    let core =
                        Self::translate_core_to_original(core, assumptions, original_assumptions);
                    return (SolverResult::Unsat, Some(core));
                }

                self.backtrack_with_phase_saving(bt_level.max(assumption_level_start + 1));
                self.debug_check_invariants("after backtrack (assumptions)");
                self.learn_clause(learnt_clause);

                self.vsids.decay();
                self.clauses.decay_activity(self.config.clause_decay);
                self.handle_clause_deletion_and_restart_limited(assumption_level_start);
            } else {
                // No conflict - try to decide. `propagate()` just returned `None`,
                // i.e. reached a fixpoint.
                self.debug_check_fixpoint_invariants("after propagation fixpoint (assumptions)");
                if let Some(var) = self.pick_branch_var() {
                    self.stats.decisions += 1;
                    self.trail.new_decision_level();

                    let polarity = if self.rand_bool(self.config.random_polarity_prob) {
                        self.rand_bool(0.5)
                    } else {
                        self.phase.get(var.index()).copied().unwrap_or(false) ^ self.phase_inverted
                    };
                    let lit = if polarity {
                        Lit::pos(var)
                    } else {
                        Lit::neg(var)
                    };
                    self.trail.assign_decision(lit);
                } else {
                    // All variables assigned - SAT
                    self.save_model();
                    self.debug_verify_model();
                    self.debug_check_invariants("at SAT (assumptions)");
                    self.backtrack(assumption_level_start);
                    return (SolverResult::Sat, None);
                }
            }
        }
    }

    /// Translate a core drawn from `resolved` (the literals
    /// [`Self::solve_with_assumptions`] actually decided on) back to the
    /// caller's `original` assumption literals it corresponds to.
    ///
    /// `resolved[i]` is what `original[i]` became after
    /// [`Self::resolve_reintroduced_literal`]; a core literal that does not
    /// match any position in `resolved` (should not happen — every core
    /// literal comes from `resolved` in the first place) is passed through
    /// unchanged rather than dropped, so a coding error here fails toward
    /// "core has an unexpected literal" rather than silently shrinking the
    /// core. First occurrence wins on a duplicate resolved literal, matching
    /// how `analyze_final_core`'s own `assumption_of` map is built from this
    /// same `resolved` list.
    fn translate_core_to_original(core: Vec<Lit>, resolved: &[Lit], original: &[Lit]) -> Vec<Lit> {
        core.into_iter()
            .map(|lit| {
                resolved
                    .iter()
                    .position(|&r| r == lit)
                    .map_or(lit, |i| original[i])
            })
            .collect()
    }

    /// Get the model (if sat)
    #[must_use]
    pub fn model(&self) -> &[LBool] {
        &self.model
    }

    /// Get the value of a variable in the model
    #[must_use]
    pub fn model_value(&self, var: Var) -> LBool {
        self.model.get(var.index()).copied().unwrap_or(LBool::Undef)
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Get memory optimizer statistics
    #[must_use]
    pub fn memory_opt_stats(&self) -> &crate::memory_opt::MemoryOptStats {
        self.memory_optimizer.stats()
    }

    /// Get number of variables
    #[must_use]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get number of clauses
    #[must_use]
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Number of *original* (asserted, non-learned) clauses in the database.
    ///
    /// This is the ground truth for "how much did the caller's encoding grow",
    /// and it is **not** the same as `num_clauses() - learned_clause_count()`:
    /// [`Self::learned_clause_count`] reports the size of the
    /// `learned_clause_ids` registry, which is a *subset* of the clauses the
    /// database itself flags as learned (the registry exists so an incremental
    /// caller can forget a probe's learned clauses again).  Subtracting it from
    /// the total therefore silently counts every unregistered learned clause as
    /// "original".  Callers that want to pin encoder growth must use this.
    #[must_use]
    pub fn num_original_clauses(&self) -> usize {
        self.clauses.num_original()
    }

    /// Number of clauses the database flags as learned.
    ///
    /// See [`Self::num_original_clauses`] for why this can exceed
    /// [`Self::learned_clause_count`].
    #[must_use]
    pub fn num_learned_clauses(&self) -> usize {
        self.clauses.num_learned()
    }

    /// Push a new assertion level (for incremental solving)
    ///
    /// This saves the current state so that clauses added after this point
    /// can be removed with pop(). Automatically backtracks to decision level 0
    /// to ensure a clean state for adding new constraints.
    pub fn push(&mut self) {
        // Backtrack to level 0 to ensure clean state
        // This is necessary because solve() may leave assignments on the trail
        // Use phase-saving backtrack to properly re-insert variables into decision heaps
        self.backtrack_with_phase_saving(0);

        self.assertion_levels.push(self.clauses.num_original());
        self.assertion_trail_sizes.push(self.trail.size());
        self.assertion_clause_ids.push(Vec::new());
    }

    /// Pop to previous assertion level
    pub fn pop(&mut self) {
        if self.assertion_levels.len() > 1 {
            self.assertion_levels.pop();

            // Get the trail size to backtrack to
            let trail_size = self.assertion_trail_sizes.pop().unwrap_or(0);

            // Remove all clauses added at this assertion level
            if let Some(clause_ids_to_remove) = self.assertion_clause_ids.pop() {
                for clause_id in clause_ids_to_remove {
                    // Purge any binary-implication-graph edges for this clause
                    // before removing it. Unlike the watch lists (which lazily
                    // skip deleted clauses during propagation), the binary graph
                    // is consulted directly, so leaving stale edges behind would
                    // let a retracted binary clause keep propagating after pop().
                    self.purge_binary_edges(clause_id);

                    // Record the retraction in the DRAT proof (if enabled) before
                    // the clause's literals become inaccessible.
                    self.drat_delete(clause_id);
                    self.lrat_delete(clause_id);

                    // Remove from clause database
                    self.clauses.remove(clause_id);

                    // Remove from learned clause tracking if it's a learned clause
                    self.learned_clause_ids.retain(|&id| id != clause_id);

                    // Note: Watch lists will be cleaned up naturally during propagation
                    // as they check if clauses are deleted before using them
                }
            }

            // Backtrack trail to the exact size it was at push()
            // This properly handles unit clauses that were added after push
            // Note: backtrack_to_size clears values but doesn't re-insert into heaps,
            // so we need to manually re-insert unassigned variables.
            let current_size = self.trail.size();
            if current_size > trail_size {
                // Collect variables that will be unassigned
                let mut unassigned_vars = Vec::new();
                for i in trail_size..current_size {
                    let lit = self.trail.assignments()[i];
                    unassigned_vars.push(lit.var());
                }

                self.trail.backtrack_to_size(trail_size);

                // Re-insert unassigned variables into decision heaps
                for var in unassigned_vars {
                    if !self.vsids.contains(var) {
                        self.vsids.insert(var);
                    }
                    if !self.chb.contains(var) {
                        self.chb.insert(var);
                    }
                    self.lrb.unassign(var);
                    // A stale unit-justification id for a now-unassigned
                    // variable must not survive to (wrongly) back whatever
                    // this variable is reassigned to next — see
                    // `Self::lrat_clear_unit_justification`.
                    self.lrat_clear_unit_justification(var);
                }
            }

            // Ensure we're at decision level 0 with proper heap re-insertion
            self.backtrack_with_phase_saving(0);

            // Re-arm unit propagation over the retained prefix.
            //
            // `backtrack_to_size` parks the propagation head at the end of the
            // surviving trail, declaring that prefix fully propagated. That is
            // false here for two independent reasons: the discarded suffix held
            // level-0 *consequences* of the retained prefix, and this pop has
            // just removed clauses the prefix was propagated against. The
            // surviving literals are therefore assigned but no longer followed by
            // their implications, and nothing would ever recompute them —
            // `backtrack_with_phase_saving(0)` above only clamps the head when it
            // actually rolls a level back, and after `backtrack_to_size` the
            // solver already sits at level 0.
            //
            // A clause left falsified that way is never revisited: its watched
            // literals were assigned before the head and so are never
            // re-propagated, so the conflict is silently lost and the next
            // `solve()` reports `Sat` on a model violating it. Rewinding costs
            // one extra pass over the retained watch lists; re-propagating an
            // already-assigned literal is a no-op, so it has no semantic effect
            // beyond restoring the facts the pop erased. Mirrors the same rewind
            // in `Solver::restore_to_trail_size`.
            self.trail.reset_propagation_head();

            // Clear the trivially_unsat flag as we've removed problematic clauses
            self.trivially_unsat = false;
        }
    }

    /// Backtrack to decision level 0 (for AllSAT enumeration)
    ///
    /// This is necessary after a SAT result before adding blocking clauses
    /// to ensure the new clauses can trigger propagation correctly.
    /// Uses phase-saving backtrack to properly re-insert unassigned variables
    /// into the decision heaps (VSIDS, CHB, LRB).
    pub fn backtrack_to_root(&mut self) {
        self.backtrack_with_phase_saving(0);
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.clauses = ClauseDatabase::new();
        self.trail.clear();
        self.watches.clear();
        self.vsids.clear();
        self.chb.clear();
        // `VMTF` (like `LRB`) only ever *grows* on `resize` — it has no
        // in-place `clear()`, because a shrink-then-regrow would leave stale
        // links pointing at variable indices the fresh problem may never
        // recreate. Rebuilding it empty here is the only safe way to make
        // `new_var` repopulate it from scratch after `num_vars` is reset to 0
        // below; otherwise a decision could return a variable index beyond
        // the new (smaller) problem's range, which every var-indexed array
        // downstream would then index out of bounds.
        self.vmtf = VMTF::new(0);
        self.stats = SolverStats::default();
        self.learnt.clear();
        self.seen.clear();
        self.analyze_stack.clear();
        self.assertion_levels.clear();
        self.assertion_levels.push(0);
        self.assertion_trail_sizes.clear();
        self.assertion_trail_sizes.push(0);
        self.assertion_clause_ids.clear();
        self.assertion_clause_ids.push(Vec::new());
        self.model.clear();
        self.num_vars = 0;
        self.restart_threshold = self.config.restart_interval;
        self.trivially_unsat = false;
        self.phase.clear();
        self.phase_inverted = false;
        self.best_phase.clear();
        self.best_trail_size = 0;
        self.rephase_count = 0;
        self.luby_index = 0;
        self.stable = false;
        self.stabphases = 0;
        self.glue_current = ModeLbdTrackers::new();
        self.glue_saved = ModeLbdTrackers::new();
        self.reluctant = LubyClock::default();
        self.ticks_focused = 0;
        self.ticks_stable = 0;
        self.lim_restart = 0;
        self.lim_stabilize = 0;
        self.level_marks.clear();
        self.lbd_mark = 0;
        self.learned_clause_ids.clear();
        self.conflicts_since_deletion = 0;
        self.rng_state = 0x853c_49e6_748f_ea9b;
        self.recent_lbd_sum = 0;
        self.recent_lbd_count = 0;
        self.binary_graph.clear();
        self.global_lbd_sum = 0;
        self.global_lbd_count = 0;
        self.conflicts_since_local_restart = 0;
        self.pure_literal_reconstruction.clear();
        // Drop any proof logger: its clause ids refer to the now-cleared database,
        // so continuing to emit against it would produce a meaningless proof.
        self.drat = None;
        self.lrat = None;
        self.clause_lrat_id.clear();
        self.lrat_unit_id.clear();
        self.lrat_unsat_finalized = false;
        // Drop the inprocessing toolkit's one-shot latches and reconstruction
        // maps along with everything else. Without this a solver reused via
        // `reset()` would keep `bve_latched`/`equiv_fold_latched` set from the
        // *previous* problem, so the toolkit would silently never run again,
        // and any surviving `equiv_substitution`/`bve_def` entries — indexed
        // by the old problem's variables — would overwrite the new problem's
        // live variables with garbage during `save_model`.
        self.bve_latched = false;
        self.equiv_fold_latched = false;
        self.equiv_substitution.clear();
        self.equiv_substitution_sized = false;
        self.bve_def.clear();
        self.bve_order.clear();
        self.fatal_error = None;
        self.propagate_step_limit = None;
        self.propagate_aborted = false;
    }

    /// Get the current trail (for theory solvers)
    #[must_use]
    pub fn trail(&self) -> &Trail {
        &self.trail
    }

    /// Get the current decision level
    #[must_use]
    pub fn decision_level(&self) -> u32 {
        self.trail.decision_level()
    }

    /// Debug method: print all learned clauses
    pub fn debug_print_learned_clauses(&self) {
        println!(
            "=== Learned Clauses ({}) ===",
            self.learned_clause_ids.len()
        );
        for (i, &cid) in self.learned_clause_ids.iter().enumerate() {
            if let Some(clause) = self.clauses.get(cid)
                && !clause.deleted
            {
                let lits: Vec<String> = clause
                    .lits
                    .iter()
                    .map(|lit| {
                        let var = lit.var().index();
                        if lit.is_pos() {
                            format!("v{}", var)
                        } else {
                            format!("~v{}", var)
                        }
                    })
                    .collect();
                println!(
                    "  Learned {}: ({}), LBD={}",
                    i,
                    lits.join(" | "),
                    clause.lbd
                );
            }
        }
    }

    /// Debug method: print binary implication graph entries
    pub fn debug_print_binary_graph(&self) {
        println!("=== Binary Implication Graph ===");
        for lit_code in 0..(self.num_vars * 2) {
            let lit = Lit::from_code(lit_code as u32);
            let implications = self.binary_graph.get(lit);
            if !implications.is_empty() {
                let lit_str = if lit.is_pos() {
                    format!("v{}", lit.var().index())
                } else {
                    format!("~v{}", lit.var().index())
                };
                for &(implied, _cid) in implications {
                    let impl_str = if implied.is_pos() {
                        format!("v{}", implied.var().index())
                    } else {
                        format!("~v{}", implied.var().index())
                    };
                    println!("  {} -> {}", lit_str, impl_str);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
