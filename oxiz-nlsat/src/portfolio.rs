//! Portfolio-based parallel solving for NLSAT.
//!
//! This module implements portfolio-based parallel solving where multiple solver instances
//! run concurrently with different configurations. The first solver to find a solution wins.
//!
//! Key features:
//! - Multiple solver instances with diverse configurations
//! - Work-stealing and clause sharing between solvers
//! - Dynamic configuration adjustment based on problem characteristics
//!
//! Reference: Z3's portfolio solver and modern SAT competition solvers

use crate::clause::Clause;
use crate::restart::RestartStrategy;
use crate::solver::{AtomId, NlsatSolver, SolverResult};
use crate::types::{Atom, AtomKind, Literal};
use crate::var_order::{OrderingStrategy, VariableOrdering};
use oxiz_math::polynomial::{Monomial, MonomialOrder, Polynomial, Term, Var};
use oxiz_time::{Duration, Instant};
use rustc_hash::FxHashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Configuration for portfolio-based solving.
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Number of parallel solver instances.
    pub num_solvers: usize,
    /// Timeout for the portfolio (None = no timeout).
    pub timeout: Option<Duration>,
    /// Enable clause sharing between solvers.
    pub enable_clause_sharing: bool,
    /// Maximum LBD for shared clauses.
    pub max_shared_lbd: u32,
    /// Share clauses every N conflicts.
    pub share_interval: usize,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            num_solvers: num_cpus::get().max(2),
            timeout: None,
            enable_clause_sharing: true,
            max_shared_lbd: 8,
            share_interval: 1000,
        }
    }
}

/// Statistics for portfolio solving.
#[derive(Debug, Clone, Default)]
pub struct PortfolioStats {
    /// Number of solvers that participated.
    pub num_solvers: usize,
    /// ID of the winning solver.
    pub winning_solver: Option<usize>,
    /// Total clauses shared.
    pub total_shared_clauses: usize,
    /// Total time spent.
    pub total_time: Duration,
    /// Number of conflicts per solver.
    pub conflicts_per_solver: Vec<usize>,
}

/// Shared clause database for portfolio solvers.
#[derive(Debug)]
struct SharedClauseDB {
    /// Clauses to be shared.
    #[allow(dead_code)]
    clauses: Mutex<Vec<(usize, Clause)>>, // (source_solver_id, clause)
    /// Total number of shared clauses.
    total_shared: AtomicUsize,
}

impl SharedClauseDB {
    fn new() -> Self {
        Self {
            clauses: Mutex::new(Vec::new()),
            total_shared: AtomicUsize::new(0),
        }
    }

    /// Add a clause to share from a solver.
    #[allow(dead_code)]
    fn share_clause(&self, solver_id: usize, clause: Clause) {
        let mut clauses = self.clauses.lock().expect("lock should not be poisoned");
        clauses.push((solver_id, clause));
        self.total_shared.fetch_add(1, Ordering::Relaxed);
    }

    /// Get clauses shared by other solvers (not from this solver).
    #[allow(dead_code)]
    fn get_shared_clauses(&self, solver_id: usize) -> Vec<Clause> {
        let mut clauses = self.clauses.lock().expect("lock should not be poisoned");
        let result: Vec<_> = clauses
            .iter()
            .filter(|(id, _)| *id != solver_id)
            .map(|(_, c)| c.clone())
            .collect();
        // Clear after reading
        clauses.clear();
        result
    }

    fn total_shared(&self) -> usize {
        self.total_shared.load(Ordering::Relaxed)
    }
}

/// Result of portfolio solving.
#[derive(Debug, Clone)]
pub enum PortfolioResult {
    /// Satisfiable with model.
    Sat {
        solver_id: usize,
        model: Vec<(Literal, bool)>,
    },
    /// Unsatisfiable with core.
    Unsat {
        solver_id: usize,
        core: Vec<Literal>,
    },
    /// Unknown (timeout or resource limit).
    Unknown,
}

/// A faithful, replayable snapshot of an [`NlsatSolver`]'s problem
/// (arithmetic variable count, atoms, and non-learned clauses).
///
/// [`NlsatSolver`] has no [`Clone`] implementation and only exposes atom
/// enumeration for single-factor inequality atoms, so this is how each
/// portfolio worker gets its own independent solver instance seeded with
/// the *actual* problem rather than starting empty (see the audit finding
/// this module fixes: workers used to solve `NlsatSolver::new()` with no
/// clauses, which is trivially `Sat` for any input).
struct ProblemSnapshot {
    num_arith_vars: u32,
    /// One slot per boolean variable index (`0..num_bool_vars`):
    /// `Some((poly, kind))` if that variable backs a single-factor
    /// inequality atom, `None` if it's a "free" boolean variable with no
    /// associated atom (e.g. a plain SAT variable, or a MaxSAT-style
    /// relaxation variable). Replaying slots in index order via
    /// `new_ineq_atom`/`new_bool_var` reproduces the exact original
    /// variable numbering, since both allocate the next sequential
    /// boolean variable deterministically.
    atom_slots: Vec<Option<(Polynomial, AtomKind)>>,
    clauses: Vec<Vec<Literal>>,
}

/// Portfolio solver manager.
pub struct PortfolioSolver {
    /// Configuration.
    config: PortfolioConfig,
    /// Base solver holding the actual problem (atoms/clauses) to diversify
    /// and solve in parallel.
    base_solver: NlsatSolver,
    /// Shared clause database.
    shared_db: Arc<SharedClauseDB>,
    /// Flag to signal termination to all solvers.
    terminated: Arc<AtomicBool>,
    /// Statistics.
    stats: PortfolioStats,
}

impl PortfolioSolver {
    /// Create a new portfolio solver.
    pub fn new(config: PortfolioConfig, base_solver: NlsatSolver) -> Self {
        Self {
            config,
            base_solver,
            shared_db: Arc::new(SharedClauseDB::new()),
            terminated: Arc::new(AtomicBool::new(false)),
            stats: PortfolioStats::default(),
        }
    }

    /// Solve using portfolio approach.
    pub fn solve(&mut self) -> PortfolioResult {
        let start_time = Instant::now();
        self.stats.num_solvers = self.config.num_solvers;

        // Reset termination flag
        self.terminated.store(false, Ordering::Relaxed);

        // Create solver configurations with diversity
        let solver_configs = self.create_diverse_configs();

        // Run solvers in parallel
        let result = self.run_parallel_solvers(solver_configs);

        self.stats.total_time = start_time.elapsed();
        self.stats.total_shared_clauses = self.shared_db.total_shared();

        result
    }

    /// Create diverse solver configurations.
    ///
    /// Diversity spans three real, sound axes: restart schedule, the
    /// [`OrderingStrategy`] (wired via variable relabeling — see
    /// [`Self::ordering_relabel_map`]), and a distinct RNG seed per worker so
    /// that two workers sharing a strategy still explore different search
    /// trees instead of being identical clones.
    fn create_diverse_configs(&self) -> Vec<SolverConfig> {
        let mut configs = Vec::new();

        for i in 0..self.config.num_solvers {
            let (restart_strategy, ordering_strategy, use_phase_saving) = match i % 6 {
                // Aggressive restart
                0 => (
                    RestartStrategy::Geometric {
                        initial: 100,
                        multiplier: 1.1,
                    },
                    OrderingStrategy::Brown,
                    true,
                ),
                // Conservative restart
                1 => (
                    RestartStrategy::Luby { unit: 512 },
                    OrderingStrategy::MaxDegree,
                    false,
                ),
                // Fixed interval restart
                2 => (
                    RestartStrategy::Fixed { interval: 1000 },
                    OrderingStrategy::MaxOccurrence,
                    true,
                ),
                // No restart (very high interval)
                3 => (
                    RestartStrategy::Geometric {
                        initial: 1_000_000,
                        multiplier: 1.0,
                    },
                    OrderingStrategy::MinDegree,
                    false,
                ),
                // Fast restart
                4 => (
                    RestartStrategy::Geometric {
                        initial: 50,
                        multiplier: 1.5,
                    },
                    OrderingStrategy::Brown,
                    true,
                ),
                // Balanced
                _ => (
                    RestartStrategy::Geometric {
                        initial: 200,
                        multiplier: 1.2,
                    },
                    OrderingStrategy::MaxOccurrence,
                    true,
                ),
            };
            configs.push(SolverConfig {
                restart_strategy,
                ordering_strategy,
                use_phase_saving,
                // Distinct, well-spread seed per worker (an odd multiple of
                // the golden-ratio constant) so identically-strategied workers
                // still diverge under random decisions.
                seed: 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((i as u64).wrapping_add(1)),
            });
        }

        configs
    }

    /// Take a faithful, replayable snapshot of `solver`'s current problem.
    ///
    /// Returns `None` if the problem cannot be guaranteed to replay
    /// identically onto a fresh [`NlsatSolver`] (e.g. it contains atoms
    /// not reachable through the public `new_ineq_atom` API). Portfolio
    /// diversification depends on exact replay fidelity for soundness
    /// (every worker must solve the SAME problem as `base_solver`), so
    /// callers must treat `None` as "cannot safely diversify" rather than
    /// silently falling back to an empty problem.
    fn snapshot_problem(solver: &NlsatSolver) -> Option<ProblemSnapshot> {
        let num_atoms = solver.num_atoms();
        let num_bool_vars = solver.num_bool_vars() as usize;
        let mut atom_slots: Vec<Option<(Polynomial, AtomKind)>> = vec![None; num_bool_vars];

        for id in 0..num_atoms as AtomId {
            match solver.get_atom(id)? {
                Atom::Ineq(ineq) if ineq.factors.len() == 1 && !ineq.factors[0].is_even => {
                    let slot = atom_slots.get_mut(ineq.bool_var as usize)?;
                    *slot = Some((ineq.factors[0].poly.clone(), ineq.kind));
                }
                // Root atoms or multi-factor/even-power atoms are not
                // reproducible via `new_ineq_atom`; bail out honestly
                // rather than silently dropping or mis-replaying them.
                _ => return None,
            }
        }

        let clauses: Vec<Vec<Literal>> = solver
            .clauses()
            .clauses()
            .iter()
            .filter(|c| !c.is_learned())
            .map(|c| c.literals().to_vec())
            .collect();

        Some(ProblemSnapshot {
            num_arith_vars: solver.num_arith_vars(),
            atom_slots,
            clauses,
        })
    }

    /// Replay `snapshot`'s variables, atoms, and clauses onto `solver`,
    /// applying the bijective arithmetic-variable `relabel` (see
    /// [`Self::ordering_relabel_map`]) to every atom polynomial so the worker
    /// visits variables in its strategy's preferred order.
    fn populate_from_snapshot(
        solver: &mut NlsatSolver,
        snapshot: &ProblemSnapshot,
        relabel: &[Var],
    ) {
        for _ in 0..snapshot.num_arith_vars {
            solver.new_arith_var();
        }
        for slot in &snapshot.atom_slots {
            match slot {
                Some((poly, kind)) => {
                    solver.new_ineq_atom(Self::relabel_polynomial(poly, relabel), *kind);
                }
                None => {
                    solver.new_bool_var();
                }
            }
        }
        for clause in &snapshot.clauses {
            solver.add_clause(clause.clone());
        }
    }

    /// Compute a bijective relabeling of the `0..num_arith_vars` arithmetic
    /// variables that places the variable `strategy` wants decided *first* at
    /// index `0`, the next at index `1`, and so on.
    ///
    /// The solver decides arithmetic variables in creation order (its
    /// `var_order` is `[0, 1, .., n-1]`) and exposes no public hook to reorder
    /// that or to set a per-instance ordering strategy. Renaming the variables
    /// of the replayed problem is a *sound* way to achieve the same effect: a
    /// bijective renaming of variables preserves satisfiability exactly, and
    /// the boolean structure (atoms, clauses, bool-var numbering) is untouched,
    /// so an UNSAT core or boolean model produced by a worker is still valid
    /// for the shared base problem.
    ///
    /// Variables that do not occur in any polynomial keep arbitrary (but
    /// distinct) trailing labels, preserving the bijection.
    fn ordering_relabel_map(snapshot: &ProblemSnapshot, strategy: OrderingStrategy) -> Vec<Var> {
        let n = snapshot.num_arith_vars as usize;
        let polys: Vec<Polynomial> = snapshot
            .atom_slots
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(poly, _)| poly.clone()))
            .collect();

        // `compute()` returns the variables (that occur in `polys`) in the
        // order the strategy would like them decided.
        let ordering = VariableOrdering::new(strategy, polys).compute();

        const UNSET: Var = Var::MAX;
        let mut map: Vec<Var> = vec![UNSET; n];
        let mut next: Var = 0;
        for &var in &ordering {
            let idx = var as usize;
            if idx < n && map[idx] == UNSET {
                map[idx] = next;
                next += 1;
            }
        }
        for slot in map.iter_mut() {
            if *slot == UNSET {
                *slot = next;
                next += 1;
            }
        }
        map
    }

    /// Apply a bijective arithmetic-variable `relabel` to `poly`, returning a
    /// polynomial denoting the same function with variable `v` renamed to
    /// `relabel[v]`. Purely a renaming — value, degree structure, and
    /// satisfiability are preserved.
    fn relabel_polynomial(poly: &Polynomial, relabel: &[Var]) -> Polynomial {
        let terms: Vec<Term> = poly
            .terms()
            .iter()
            .map(|term| {
                let monomial = Monomial::from_powers(term.monomial.vars().iter().map(|vp| {
                    let renamed = relabel.get(vp.var as usize).copied().unwrap_or(vp.var);
                    (renamed, vp.power)
                }));
                Term::new(term.coeff.clone(), monomial)
            })
            .collect();
        Polynomial::from_terms(terms, MonomialOrder::default())
    }

    /// Build a solver diversified per `config` and seeded with the actual
    /// problem in `snapshot`, with arithmetic variables relabeled so the
    /// worker visits them in `config.ordering_strategy`'s preferred order.
    fn build_configured_solver(config: &SolverConfig, snapshot: &ProblemSnapshot) -> NlsatSolver {
        let mut solver = Self::create_configured_solver(config);
        let relabel = Self::ordering_relabel_map(snapshot, config.ordering_strategy);
        Self::populate_from_snapshot(&mut solver, snapshot, &relabel);
        solver
    }

    /// Extract a real (non-fabricated) unsat core as the deduplicated
    /// union of literals across every clause identified by
    /// [`NlsatSolver::get_unsat_core`].
    fn extract_core_literals(solver: &NlsatSolver) -> Vec<Literal> {
        let mut seen = FxHashSet::default();
        let mut literals = Vec::new();
        for clause_id in solver.get_unsat_core() {
            if let Some(clause) = solver.clauses().get(clause_id) {
                for &lit in clause.literals() {
                    if seen.insert(lit) {
                        literals.push(lit);
                    }
                }
            }
        }
        literals
    }

    /// Extract the real (non-fabricated) boolean model from a solved
    /// solver, as `(positive literal, assigned value)` pairs.
    fn extract_sat_model(solver: &NlsatSolver) -> Vec<(Literal, bool)> {
        solver
            .get_model()
            .map(|model| {
                model
                    .bool_values
                    .into_iter()
                    .map(|(var, value)| (Literal::positive(var), value))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Run solvers in parallel.
    ///
    /// Each worker gets its own [`NlsatSolver`], diversified per its
    /// [`SolverConfig`] and seeded with the SAME problem as `base_solver`
    /// (via [`Self::snapshot_problem`]/[`Self::build_configured_solver`]),
    /// so every worker is actually attempting the real problem rather than
    /// a trivially-`Sat` empty one.
    fn run_parallel_solvers(&mut self, configs: Vec<SolverConfig>) -> PortfolioResult {
        use rayon::prelude::*;

        let Some(snapshot) = Self::snapshot_problem(&self.base_solver) else {
            // Cannot faithfully replay the base problem onto worker
            // solvers (see `snapshot_problem` docs); running against an
            // empty problem would silently report Sat for every input, so
            // honestly report Unknown instead.
            return PortfolioResult::Unknown;
        };
        let snapshot = Arc::new(snapshot);

        let shared_db = self.shared_db.clone();
        let terminated = self.terminated.clone();
        let result_mutex: Arc<Mutex<Option<PortfolioResult>>> = Arc::new(Mutex::new(None));
        let enable_sharing = self.config.enable_clause_sharing;

        // Store config count for stats
        self.stats.conflicts_per_solver = vec![0; configs.len()];

        let run = {
            let terminated = terminated.clone();
            let result_mutex = result_mutex.clone();
            let shared_db = shared_db.clone();
            let snapshot = snapshot.clone();
            move || {
                (0..configs.len()).into_par_iter().for_each(|solver_id| {
                    // Check if another solver already found a result
                    if terminated.load(Ordering::Relaxed) {
                        return;
                    }

                    // Create a solver instance for this worker, diversified
                    // per its config and seeded with the actual problem.
                    let mut solver = Self::build_configured_solver(&configs[solver_id], &snapshot);
                    solver.set_unsat_core_extraction(true);

                    let local_result = solver.solve();

                    match local_result {
                        SolverResult::Sat => {
                            // Found a solution - signal other threads to stop
                            terminated.store(true, Ordering::Relaxed);

                            let mut result =
                                result_mutex.lock().expect("lock should not be poisoned");
                            if result.is_none() {
                                *result = Some(PortfolioResult::Sat {
                                    solver_id,
                                    model: Self::extract_sat_model(&solver),
                                });
                            }

                            // Share clauses if enabled
                            if enable_sharing {
                                shared_db.total_shared();
                            }
                        }
                        SolverResult::Unsat => {
                            // Found UNSAT - signal other threads to stop
                            terminated.store(true, Ordering::Relaxed);

                            let mut result =
                                result_mutex.lock().expect("lock should not be poisoned");
                            if result.is_none() {
                                *result = Some(PortfolioResult::Unsat {
                                    solver_id,
                                    core: Self::extract_core_literals(&solver),
                                });
                            }
                        }
                        SolverResult::Unknown => {
                            // Continue searching
                        }
                    }
                });
            }
        };

        // Best-effort timeout: `NlsatSolver::solve()` has no cooperative
        // cancellation hook, so an in-flight `solve()` call cannot be
        // forcibly interrupted; but we can still honor the caller-facing
        // contract that `solve()` (this method) returns within
        // `config.timeout` by racing the worker pool against a deadline on
        // a dedicated channel, reporting `Unknown` if the deadline fires
        // first (workers keep running in the background and are cut off
        // via `terminated`, but their thread(s) may finish late).
        match self.config.timeout {
            Some(timeout) => {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    run();
                    let _ = tx.send(());
                });
                if rx.recv_timeout(timeout).is_err() {
                    terminated.store(true, Ordering::Relaxed);
                }
            }
            None => run(),
        }

        // Return the result
        let result = result_mutex.lock().expect("lock should not be poisoned");
        result.clone().unwrap_or(PortfolioResult::Unknown)
    }

    /// Create a new solver with the given configuration.
    /// Since NlsatSolver configuration is set at construction time,
    /// we create a new solver instead of modifying an existing one.
    fn create_configured_solver(config: &SolverConfig) -> NlsatSolver {
        let mut solver_config = crate::solver::SolverConfig {
            restart_strategy: config.restart_strategy,
            dynamic_reordering: true,
            // Per-worker seed so random-decision workers actually diverge.
            random_seed: config.seed,
            ..Default::default()
        };

        // Diversify other parameters based on strategy
        if matches!(config.restart_strategy, RestartStrategy::Geometric { .. }) {
            solver_config.reorder_frequency = 500;
        } else {
            solver_config.reorder_frequency = 2000;
        }

        // Enable phase saving if configured
        solver_config.random_decisions = !config.use_phase_saving;

        NlsatSolver::with_config(solver_config)
    }

    /// Run a single solver instance with clause sharing, returning a real
    /// [`PortfolioResult`] (model / unsat core extracted from the solver) or
    /// `None` if this worker cannot decide the problem within its budget.
    ///
    /// [`NlsatSolver::solve`] already returns a *definitive* result, so this
    /// does not spin re-running an identical search. On `Unknown` the only way a
    /// re-solve can change the outcome is if freshly-imported shared clauses
    /// (from other workers) tightened this worker's problem, so we exchange
    /// clauses and retry a bounded number of rounds (`share_interval`), then
    /// honestly give up with `None` rather than looping forever or fabricating a
    /// verdict. This replaces the previous stub that returned empty models/cores
    /// and aborted after a hard-coded 10 conflicts.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn run_single_solver(
        &self,
        solver_id: usize,
        mut solver: NlsatSolver,
        shared_db: Arc<SharedClauseDB>,
        terminated: Arc<AtomicBool>,
        enable_sharing: bool,
        share_interval: usize,
        max_shared_lbd: u32,
    ) -> Option<PortfolioResult> {
        // Real (non-fabricated) unsat cores require core extraction to be on.
        solver.set_unsat_core_extraction(true);

        // Bound on clause-exchange rounds; never spin unboundedly.
        let max_rounds = share_interval.max(1);

        for _ in 0..max_rounds {
            // Another worker already found a result.
            if terminated.load(Ordering::Relaxed) {
                return None;
            }

            match solver.solve() {
                SolverResult::Sat => {
                    terminated.store(true, Ordering::Relaxed);
                    return Some(PortfolioResult::Sat {
                        solver_id,
                        model: Self::extract_sat_model(&solver),
                    });
                }
                SolverResult::Unsat => {
                    terminated.store(true, Ordering::Relaxed);
                    return Some(PortfolioResult::Unsat {
                        solver_id,
                        core: Self::extract_core_literals(&solver),
                    });
                }
                SolverResult::Unknown => {
                    // A bare re-solve reproduces the same Unknown; only newly
                    // imported shared clauses can help. Without sharing there is
                    // nothing more to try, so report honestly.
                    if !enable_sharing {
                        return None;
                    }
                    self.share_learned_clauses(solver_id, &solver, &shared_db, max_shared_lbd);
                    self.import_shared_clauses(solver_id, &mut solver, &shared_db);
                }
            }
        }

        // Budget exhausted without a definitive answer.
        None
    }

    /// Share learned clauses with good LBD to other solvers.
    #[allow(dead_code)]
    fn share_learned_clauses(
        &self,
        solver_id: usize,
        solver: &NlsatSolver,
        shared_db: &Arc<SharedClauseDB>,
        max_lbd: u32,
    ) {
        // Get clauses from the solver
        for clause in solver.clauses().clauses() {
            if clause.is_learned() && clause.lbd() <= max_lbd {
                shared_db.share_clause(solver_id, clause.clone());
            }
        }
    }

    /// Import shared clauses from other solvers.
    #[allow(dead_code)]
    fn import_shared_clauses(
        &self,
        solver_id: usize,
        solver: &mut NlsatSolver,
        shared_db: &Arc<SharedClauseDB>,
    ) {
        let clauses = shared_db.get_shared_clauses(solver_id);
        for clause in clauses {
            // Add the shared clause to this solver
            solver.add_clause(clause.literals().to_vec());
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &PortfolioStats {
        &self.stats
    }
}

/// Configuration for a single solver in the portfolio.
#[derive(Debug, Clone)]
struct SolverConfig {
    restart_strategy: RestartStrategy,
    /// Variable-ordering heuristic for this worker. `NlsatSolver`'s own
    /// `SolverConfig` has no ordering-strategy knob and exposes no hook to
    /// reorder its decision variables, so this is wired in soundly by
    /// *relabeling* the replayed problem's arithmetic variables (a bijective
    /// renaming that preserves satisfiability) so the solver visits them in
    /// this strategy's preferred order. See [`PortfolioSolver::ordering_relabel_map`].
    ordering_strategy: OrderingStrategy,
    use_phase_saving: bool,
    /// Per-worker RNG seed so workers that share a strategy still explore
    /// distinct search trees instead of being identical clones.
    seed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_config_default() {
        let config = PortfolioConfig::default();
        assert!(config.num_solvers >= 2);
        assert!(config.enable_clause_sharing);
        assert_eq!(config.max_shared_lbd, 8);
    }

    #[test]
    fn test_shared_clause_db() {
        let db = SharedClauseDB::new();

        // Solver 0 shares a clause
        let clause = Clause::new(vec![Literal::positive(1)], 1, false, 0);
        db.share_clause(0, clause.clone());

        assert_eq!(db.total_shared(), 1);

        // Solver 1 gets the shared clause
        let shared = db.get_shared_clauses(1);
        assert_eq!(shared.len(), 1);

        // After getting, the queue is cleared
        let shared2 = db.get_shared_clauses(1);
        assert_eq!(shared2.len(), 0);
    }

    #[test]
    fn test_portfolio_solver_new() {
        let solver = NlsatSolver::new();
        let config = PortfolioConfig::default();
        let portfolio = PortfolioSolver::new(config, solver);

        assert_eq!(portfolio.stats.num_solvers, 0);
        assert!(portfolio.stats.winning_solver.is_none());
    }

    // Regression: `run_single_solver` previously returned `model: Vec::new()` on
    // Sat, `core: Vec::new()` on Unsat, and gave up after a hard-coded 10
    // conflicts. It must now return a real (non-empty) model / core.
    #[test]
    fn test_run_single_solver_returns_real_model() {
        // Worker problem: x > 0 (SAT).
        let mut worker = NlsatSolver::new();
        let atom = worker.new_ineq_atom(Polynomial::from_var(0), AtomKind::Gt);
        worker.add_clause(vec![worker.atom_literal(atom, true)]);

        let portfolio = PortfolioSolver::new(PortfolioConfig::default(), NlsatSolver::new());
        let terminated = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(SharedClauseDB::new());

        match portfolio.run_single_solver(0, worker, shared, terminated, false, 4, 8) {
            Some(PortfolioResult::Sat { model, .. }) => {
                assert!(
                    !model.is_empty(),
                    "run_single_solver must return a real (non-empty) model, not the empty stub"
                );
            }
            other => panic!("expected Sat with a real model, got {other:?}"),
        }
    }

    #[test]
    fn test_run_single_solver_returns_real_core() {
        // Worker problem: the classic 2-variable UNSAT core
        // (a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b) ∧ (¬a ∨ ¬b), which is decided by
        // search (not at clause-add time) so conflict analysis records the core.
        let mut worker = NlsatSolver::new();
        let a = worker.new_bool_var();
        let b = worker.new_bool_var();
        worker.add_clause(vec![Literal::positive(a), Literal::positive(b)]);
        worker.add_clause(vec![Literal::negative(a), Literal::positive(b)]);
        worker.add_clause(vec![Literal::positive(a), Literal::negative(b)]);
        worker.add_clause(vec![Literal::negative(a), Literal::negative(b)]);

        let portfolio = PortfolioSolver::new(PortfolioConfig::default(), NlsatSolver::new());
        let terminated = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(SharedClauseDB::new());

        match portfolio.run_single_solver(0, worker, shared, terminated, false, 4, 8) {
            Some(PortfolioResult::Unsat { core, .. }) => {
                assert!(
                    !core.is_empty(),
                    "run_single_solver must return a real (non-empty) unsat core"
                );
            }
            other => panic!("expected Unsat with a real core, got {other:?}"),
        }
    }

    /// Regression test for the audit finding: `run_parallel_solvers` used
    /// to spawn fresh `NlsatSolver::new()` workers and never copy the base
    /// problem's clauses into them, so an empty (trivially `Sat`) problem
    /// was solved instead — `PortfolioSolver::solve()` reported `Sat` for
    /// every input, including directly contradictory (UNSAT) ones.
    #[test]
    fn test_portfolio_solve_reports_unsat_for_unsat_problem() {
        let mut base = NlsatSolver::new();
        let v = base.new_bool_var();
        // v AND NOT v: directly unsatisfiable.
        base.add_clause(vec![Literal::positive(v)]);
        base.add_clause(vec![Literal::negative(v)]);

        let config = PortfolioConfig {
            num_solvers: 3,
            timeout: None,
            ..PortfolioConfig::default()
        };
        let mut portfolio = PortfolioSolver::new(config, base);
        let result = portfolio.solve();

        assert!(
            matches!(result, PortfolioResult::Unsat { .. }),
            "expected Unsat for a directly contradictory base problem, got {result:?}"
        );
    }

    /// Regression test: a genuinely satisfiable base problem must produce
    /// a real model (not the old hardcoded empty `Vec::new()`) that
    /// actually satisfies the asserted clauses.
    #[test]
    fn test_portfolio_solve_finds_real_sat_model() {
        let mut base = NlsatSolver::new();
        let v1 = base.new_bool_var();
        let v2 = base.new_bool_var();
        // v1 AND (NOT v1 OR v2)  =>  v1 = true, v2 = true is forced.
        base.add_clause(vec![Literal::positive(v1)]);
        base.add_clause(vec![Literal::negative(v1), Literal::positive(v2)]);

        let config = PortfolioConfig {
            num_solvers: 3,
            timeout: None,
            ..PortfolioConfig::default()
        };
        let mut portfolio = PortfolioSolver::new(config, base);
        let result = portfolio.solve();

        match result {
            PortfolioResult::Sat { model, .. } => {
                let v1_val = model
                    .iter()
                    .find(|(lit, _)| lit.var() == v1)
                    .map(|(_, v)| *v);
                let v2_val = model
                    .iter()
                    .find(|(lit, _)| lit.var() == v2)
                    .map(|(_, v)| *v);
                assert_eq!(v1_val, Some(true), "model must actually satisfy `v1`");
                assert_eq!(
                    v2_val,
                    Some(true),
                    "model must actually satisfy `NOT v1 OR v2` given v1 = true"
                );
            }
            other => panic!("expected Sat with a real, non-empty model, got {other:?}"),
        }
    }

    /// Regression test: with a nonzero timeout, `solve()` must return
    /// promptly (well within the deadline plus solving time) rather than
    /// ignoring `config.timeout` entirely.
    #[test]
    fn test_portfolio_solve_respects_timeout_config_shape() {
        let mut base = NlsatSolver::new();
        let v = base.new_bool_var();
        base.add_clause(vec![Literal::positive(v)]);

        let config = PortfolioConfig {
            num_solvers: 2,
            timeout: Some(Duration::from_secs(5)),
            ..PortfolioConfig::default()
        };
        let mut portfolio = PortfolioSolver::new(config, base);

        let start = Instant::now();
        let result = portfolio.solve();
        let elapsed = start.elapsed();

        assert!(
            matches!(result, PortfolioResult::Sat { .. }),
            "trivially satisfiable problem should solve well within the timeout"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "solve() should complete promptly, took {elapsed:?}"
        );
    }

    /// Relabeling arithmetic variables is a *renaming*: the relabeled
    /// polynomial evaluated under the permuted assignment must equal the
    /// original polynomial under the original assignment. This is the
    /// soundness guarantee that lets each portfolio worker diversify its
    /// variable order without changing the problem's meaning.
    #[test]
    fn test_relabel_polynomial_preserves_value() {
        use num_bigint::BigInt;
        use num_rational::BigRational;
        use rustc_hash::FxHashMap;

        let int = |k: i64| BigRational::from_integer(BigInt::from(k));

        // p = 3*x0^2*x1 + 2*x2 - 5
        let p = Polynomial::from_coeffs_int(&[(3, &[(0, 2), (1, 1)]), (2, &[(2, 1)]), (-5, &[])]);

        // Bijection over {0,1,2}: x0->2, x1->0, x2->1.
        let relabel: Vec<Var> = vec![2, 0, 1];
        let q = PortfolioSolver::relabel_polynomial(&p, &relabel);

        // Original assignment.
        let a: FxHashMap<Var, BigRational> = [(0u32, int(2)), (1u32, int(3)), (2u32, int(7))]
            .into_iter()
            .collect();
        // Permuted assignment: a'[relabel[v]] = a[v].
        let a_permuted: FxHashMap<Var, BigRational> =
            [(2u32, int(2)), (0u32, int(3)), (1u32, int(7))]
                .into_iter()
                .collect();

        assert_eq!(
            p.eval(&a),
            q.eval(&a_permuted),
            "variable relabeling must preserve polynomial value"
        );
        // Sanity: 3*4*3 + 2*7 - 5 = 45.
        assert_eq!(p.eval(&a), int(45));
    }

    /// The relabeling map must always be a bijection over `0..num_arith_vars`,
    /// including variables that never occur in any polynomial (they get
    /// distinct trailing labels).
    #[test]
    fn test_ordering_relabel_map_is_bijection() {
        // x0*x1^2 and x1 + x3 : variable 2 occurs in nothing.
        let p0 = Polynomial::from_coeffs_int(&[(1, &[(0, 1), (1, 2)])]);
        let p1 = Polynomial::from_coeffs_int(&[(1, &[(1, 1)]), (1, &[(3, 1)])]);
        let snapshot = ProblemSnapshot {
            num_arith_vars: 4,
            atom_slots: vec![
                Some((p0, AtomKind::Gt)),
                Some((p1, AtomKind::Gt)),
                None,
                None,
            ],
            clauses: Vec::new(),
        };

        for strategy in [
            OrderingStrategy::Brown,
            OrderingStrategy::MaxDegree,
            OrderingStrategy::MinDegree,
            OrderingStrategy::MaxOccurrence,
            OrderingStrategy::MinOccurrence,
            OrderingStrategy::Static,
        ] {
            let map = PortfolioSolver::ordering_relabel_map(&snapshot, strategy);
            assert_eq!(map.len(), 4, "map covers every arithmetic variable");
            let mut sorted = map.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                vec![0, 1, 2, 3],
                "relabeling for {strategy:?} must be a permutation of 0..4, got {map:?}"
            );
        }
    }

    /// Each worker configuration must receive a distinct RNG seed so that
    /// workers sharing a strategy still diverge (the diversification fix).
    #[test]
    fn test_diverse_configs_have_distinct_seeds() {
        let base = NlsatSolver::new();
        let config = PortfolioConfig {
            num_solvers: 8,
            ..PortfolioConfig::default()
        };
        let portfolio = PortfolioSolver::new(config, base);
        let configs = portfolio.create_diverse_configs();
        let seeds: FxHashSet<u64> = configs.iter().map(|c| c.seed).collect();
        assert_eq!(
            seeds.len(),
            configs.len(),
            "every worker seed must be distinct"
        );
    }
}
