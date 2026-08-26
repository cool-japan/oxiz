//! Solver configuration: [`SolverConfig`], its [`Default`] impl, and
//! [`RestartStrategy`]. Split out of `solver/mod.rs` to keep that file under
//! the project's line-count limit as the inprocessing toolkit
//! (`enable_failed_literal_probing`/`enable_bve`/`enable_equiv_substitution`/
//! `enable_gate_congruence`) grew the struct; re-exported from `solver/mod.rs`
//! so `oxiz_sat::SolverConfig`/`oxiz_sat::RestartStrategy` are unaffected.

use super::*;

/// Solver configuration
#[derive(Clone)]
pub struct SolverConfig {
    /// Restart interval (number of conflicts)
    pub restart_interval: u64,
    /// Restart multiplier for geometric restarts
    pub restart_multiplier: f64,
    /// Clause deletion threshold
    pub clause_deletion_threshold: usize,
    /// Variable decay factor
    pub var_decay: f64,
    /// Clause decay factor
    pub clause_decay: f64,
    /// Random polarity probability (0.0 to 1.0)
    pub random_polarity_prob: f64,
    /// Restart strategy: "luby" or "geometric"
    pub restart_strategy: RestartStrategy,
    /// Enable lazy hyper-binary resolution: while a clause of 3 or 4 literals
    /// propagates, try to turn it into a shortcut binary clause by resolving
    /// away the literals that are false at level 0 (see
    /// `Solver::check_hyper_binary_resolution`). Independent of the
    /// probing-time hyper-binary resolution reached through
    /// [`SolverConfig::enable_failed_literal_probing`], which this flag does
    /// not affect.
    ///
    /// # Default: `true`, decided by measurement (issue #38, 2026-08-25)
    ///
    /// Issue #38 reported a ~12x conflict blowup on a quasigroup family. The
    /// flag was re-measured with `tests/issue38_hyper_binary_measurement.rs`
    /// (an `#[ignore]`d harness, every instance generated in-process from a
    /// fixed seed, so the table is re-derivable without a corpus): both arms
    /// identical except this flag, [`SolverConfig::enable_lucky_phase`] off in
    /// both so the pre-search scans cannot decide an instance before the CDCL
    /// loop, the same PRNG seed, and a 25 000-conflict cap. `median` is the
    /// per-family median of the on/off **conflict** ratio over the pairs both
    /// arms decided, so **above 1.0 means the pass made the search worse**:
    ///
    /// | family                              | rated | median | range        | scans -> learned |
    /// |-------------------------------------|-------|--------|--------------|------------------|
    /// | guarded chains (built to favour it) | 5     | 0.98   | 0.81 .. 1.16 | 13k-104k -> 104-214 |
    /// | random 3-SAT, n=150, r=4.26         | 5     | 0.97   | 0.89 .. 1.26 | 82k-211k -> 67-87   |
    /// | random 3-SAT, n=200, r=4.26         | 3     | 1.00   | 0.98 .. 2.88 | 1.0M-1.2M -> 0-101  |
    /// | PHP(6,5), PHP(7,6), PHP(8,7)        | 3     | 1.00   | 1.00 .. 1.01 | 4-905 -> 0-5        |
    /// | quasigroup completion, order 7-15   | 0     | n/a    | all identical | 2.8k -> 0          |
    /// | quasigroup QG3 existence, order 7-8 | 2     | 1.06   | 1.00 .. 1.12 | 4.6k-140k -> 56-166 |
    ///
    /// Two rows need a footnote. Pigeonhole is *structurally* out of reach of
    /// the pass rather than merely unhelpful: its "some hole holds this
    /// pigeon" clauses are `holes` literals wide, which the reason-width guard
    /// rejects outright, so only the width-2 at-most-one constraints ever
    /// reach it — 905 scans against 200k propagations on PHP(8,7), and none
    /// at all within the first 3 000 conflicts of PHP(9,8). And quasigroup
    /// *completion*, the shape
    /// closest to the original report, is solved by propagation alone at every
    /// order this port can afford (7 through 15, fills 35%-65%): 30 of 30
    /// pairs came out bit-identical in conflicts, propagations and decisions
    /// with zero clauses learned. The quasigroup family that genuinely
    /// searches is QG3 existence, which is why it has a row of its own.
    ///
    /// The sizes above that — QG3 at orders 9 and 10, random 3-SAT at n=200
    /// and n=250, PHP(9,8) — take hours each to *finish*, so they were
    /// measured on a fixed 3 000-conflict **budget** instead: both arms stop
    /// at the same point, which forfeits the conflict ratio and leaves the
    /// propagation ratio, which says whether the two arms' searches *diverged*
    /// over that budget. Note what it does not say: scanning is read-only and
    /// never propagates, so the pass's own cost is invisible to this column —
    /// a 1.00 here means "same search", not "free". Per the evidence rule in
    /// the harness, a family measured only this way cannot move the default on
    /// its own:
    ///
    /// | family                    | pairs | props ratio (divergence) | scans -> learned |
    /// |---------------------------|-------|--------------------------|------------------|
    /// | QG3 existence, order 9    | 1     | 0.98                     | 58k -> 252       |
    /// | QG3 existence, order 10   | 1     | 1.03                     | 72k -> 360       |
    /// | random 3-SAT, n=200       | 5     | 1.00                     | 131k-135k -> 0-1 |
    /// | random 3-SAT, n=250       | 5     | 1.00                     | 130k-161k -> 0   |
    /// | PHP(9,8)                  | 1     | 1.00                     | 0 -> 0           |
    ///
    /// Ten of those thirteen pairs were bit-identical in conflicts,
    /// propagations *and* decisions, and the scan counts show why: over the
    /// first 3 000 conflicts the pass performed 130k-160k scans at n=200 and
    /// n=250 and came away with one clause or none, so there was nothing to
    /// perturb. That is the clearest statement of the whole measurement — at
    /// these sizes the pass is a large amount of read-only work with no effect
    /// either way. The 2.88x outlier at n=200 in the first table happens later
    /// than this budget reaches.
    ///
    /// Readings:
    ///
    /// * The reported ~12x **did not reproduce** — not on the quasigroup
    ///   shapes it was reported on, and not on anything else measured. No
    ///   family's median reaches 2x in either direction, so the default stays
    ///   `true`.
    /// * Nor does the pass earn its keep. The most telling row is the family
    ///   built specifically so that hyper-binary resolution *should* pay —
    ///   long implication chains encoded as guarded ternaries over level-0
    ///   units, which is exactly the shape it looks for — and even there the
    ///   median is 0.98. The default is `true` because nothing justifies
    ///   changing it, not because the pass was shown to be worth having.
    /// * What is real is the *tail*: the pass never helped by more than 1.23x
    ///   on any single instance, but cost 2.88x on one (a satisfiable random
    ///   3-SAT at n=200) and >=1.47x on another. Turn it off if reproducible
    ///   worst-case behaviour matters more than the median.
    /// * What is also real is the wasted work: only 0.02%-0.2% of the pass's
    ///   scans produced a clause, and it scans more often than the solver
    ///   dequeues a literal to propagate (e.g. 211k scans against 180k
    ///   dequeues on one n=150 instance), because one dequeued literal can
    ///   drive many watch-list visits.
    /// * One slice of that was pure waste and is now gone: propagation along
    ///   the binary implication graph used to enter the pass, where the only
    ///   resolvent formable is the reason clause itself and the graph lookup
    ///   always rejected it. Removing that call site left conflicts,
    ///   propagations and decisions bit-identical on every instance of the
    ///   sweep — the search is provably untouched — while removing a share of
    ///   the scans that tracks how binary-clause-rich the family is: about
    ///   6%-8% on random 3-SAT, 12%-31% on the guarded chains, and 83% on QG3
    ///   at order 7. (Measured by comparing the pass's own scan counter across
    ///   builds on identical search trajectories.)
    ///
    /// The obvious next step — refusing a binary reason inside the pass as
    /// well — is deliberately *not* taken: a binary clause can also propagate
    /// from the watch lists, and there the resolvent is not always a
    /// duplicate, so that filter was measured to change which clauses are
    /// learned rather than merely saving work.
    ///
    /// # Why no adaptive guard
    ///
    /// A cheap runtime signal that switched the pass off once it stopped
    /// paying would be the obvious refinement, and the data says it would not
    /// work: the pass's effect on the search comes entirely from the handful
    /// of clauses it *does* learn, and their sign is not predictable from how
    /// many it has learned. 206 clauses bought 0.81x on one guarded-chain
    /// instance; 184 cost 1.16x on the next. 33 clauses cost 2.88x at n=200,
    /// while 101 on a sibling instance came out at 0.98x. The scan rate, the
    /// hit rate and the clause count all fail to separate the cases, so any
    /// threshold built on them would be fitted to the sample rather than
    /// measured.
    ///
    /// Off in the `Random`, `Aggressive`, and `MiniSat` presets; on in the
    /// other seven, each of which pins the field explicitly.
    pub enable_lazy_hyper_binary: bool,
    /// Use CHB instead of VSIDS for branching
    pub use_chb_branching: bool,
    /// Use LRB (Learning Rate Branching) for branching
    pub use_lrb_branching: bool,
    /// Enable inprocessing (periodic preprocessing during search)
    pub enable_inprocessing: bool,
    /// Inprocessing interval (number of conflicts between inprocessing)
    pub inprocessing_interval: u64,
    /// Enable chronological backtracking
    pub enable_chronological_backtrack: bool,
    /// Chronological backtracking threshold (max distance from assertion level)
    pub chrono_backtrack_threshold: u32,
    /// Use the VMTF move-to-front queue instead of VSIDS for decisions while
    /// the search is in *focused* mode (see [`SolverConfig::enable_stabilize`]).
    /// While in *stable* mode — or always, when `enable_stabilize` is off —
    /// VSIDS is used. Ignored when `use_chb_branching`/`use_lrb_branching`
    /// select a different heuristic outright.
    pub use_vmtf: bool,
    /// Cap on the Luby restart multiplier so the sequence's `2^k` growth
    /// cannot inflate the restart interval into a multi-thousand-conflict
    /// grind on long runs. `0` means uncapped. Only consulted by
    /// [`RestartStrategy::Luby`] when [`SolverConfig::enable_stabilize`] is
    /// off; the stable/focused schedule uses [`SolverConfig::focused_luby_cap`]
    /// instead.
    pub luby_cap: u64,
    /// Enable the stable/focused restart schedule: alternate a *focused*
    /// phase (frequent Glucose-EMA-triggered restarts, capped Luby length)
    /// with a *stable* phase (rare reluctant-doubling restarts, eligible for
    /// rephasing) on a quadratically-growing tick budget per phase. Off
    /// falls back to the legacy single restart strategy selected by
    /// [`SolverConfig::restart_strategy`].
    pub enable_stabilize: bool,
    /// Tick budget for the first stable/focused switch; each subsequent
    /// switch's budget grows quadratically in the number of switches so far.
    pub stabilize_base: u64,
    /// Luby restart cap used specifically during *focused* mode (`0` =
    /// uncapped). Stable mode's restarts are driven by the reluctant-doubling
    /// clock instead and are not capped here.
    pub focused_luby_cap: u64,
    /// Restart count between rephase rounds (periodic saved-polarity flips
    /// meant to let a restart explore a genuinely different region instead of
    /// re-deriving the trail it just abandoned). `0` disables rephasing.
    /// Rephasing only fires while the search is in stable mode — see
    /// `Solver::restart`'s internals for why (a private method, not part of
    /// this crate's public API).
    pub rephase_interval: u32,
    /// Reuse-trail restarts (Heule/Möhle & Biere): instead of always
    /// backtracking to the root, keep the longest decision prefix whose
    /// variables are still at least as "important" (by VSIDS activity) as the
    /// next variable the search would decide anyway — that prefix would
    /// simply be re-derived, so throwing it away is pure waste.
    pub reuse_trail: bool,
    /// Optional external branching heuristic. When `Some`, called before built-in
    /// VSIDS/LRB/CHB; returning `None` from the heuristic falls back to built-in.
    /// Default: `None` (pure built-in strategy).
    pub external_branching: Option<BoxedBranchingHeuristic>,
    /// Run failed-literal probing (with on-the-fly hyper-binary resolution)
    /// once before search starts. For each still-unassigned variable, both
    /// polarities are tentatively propagated at decision level 0; a polarity
    /// that conflicts is a *failed literal* and its negation is forced as a
    /// permanent unit. Bounded by an internal propagation budget, so it never
    /// dominates on any instance size, and unlike bounded variable
    /// elimination / equivalent-literal substitution it never removes a
    /// variable (only forces facts), so it carries none of their
    /// incremental-scope caveat. Off by default anyway: on some instances a
    /// probing-only pass is enough to settle the verdict without the main
    /// CDCL loop ever running, which is sound but changes observable solve
    /// behavior (e.g. how many times conflict-analysis hooks fire) — opt-in
    /// until a caller has confirmed that shape is acceptable for their use.
    pub enable_failed_literal_probing: bool,
    /// Bounded variable elimination (SatELite-style): resolve away a variable
    /// whose defining clauses are cheap to fold together, replacing them with
    /// their resolvents. Off by default — unlike probing this *removes*
    /// variables from the live formula, which is unsound across an
    /// incremental scope (a later `push`/`add_clause` could reintroduce the
    /// eliminated polarity) and is therefore only ever run at the base
    /// assertion level. Mutually exclusive with
    /// [`SolverConfig::enable_equiv_substitution`] in this implementation —
    /// see `Solver::bounded_variable_elimination`'s internals (a private
    /// method, not part of this crate's public API) for why combining the
    /// two reconstruction maps in one pass is not yet supported.
    ///
    /// Known limitation shared with `enable_equiv_substitution`: neither is
    /// consulted by [`Solver::solve_with_assumptions`], which assigns
    /// assumption literals without checking whether the toolkit already
    /// eliminated that variable. Assuming a literal on a variable this pass
    /// (or substitution) removed leaves it unconstrained by any live clause,
    /// so the assumption trivially "succeeds" and `save_model`'s
    /// unconditional reconstruction then overwrites it — a model that can
    /// violate the caller's own assumption. Narrow (opt-in BVE/substitution
    /// *and* assumption-based solving on the same solver instance) and not
    /// addressed in this pass; do not combine them.
    pub enable_bve: bool,
    /// Equivalent-literal substitution: find literals proven equivalent by a
    /// cycle in the binary implication graph (Tarjan SCC) and rewrite every
    /// clause through a single representative per class. Off by default for
    /// the same incremental-scope reason as [`SolverConfig::enable_bve`], and
    /// mutually exclusive with it (see there) — including the same
    /// `solve_with_assumptions` limitation documented on that field.
    pub enable_equiv_substitution: bool,
    /// When [`SolverConfig::enable_equiv_substitution`] is set, also run
    /// AND/XOR gate congruence detection first and fold the detected
    /// equivalences into the binary implication graph before the SCC pass —
    /// this is what lets equivalent-literal substitution collapse structural
    /// duplication (e.g. repeated partial-product/full-adder gates in a
    /// multiplier) that plain binary clauses do not expose. Ignored when
    /// `enable_equiv_substitution` is off.
    pub enable_gate_congruence: bool,
    /// Self-subsuming resolution during inprocessing: when a clause `C`
    /// resolves against a clause `D` to produce a clause that subsumes `D`,
    /// strengthen `D` in place by dropping the resolved literal (see
    /// `Solver::self_subsuming_resolution`, a private method).
    ///
    /// **On by default.** Unlike [`SolverConfig::enable_bve`] and
    /// [`SolverConfig::enable_equiv_substitution`] this never removes a
    /// variable and needs no model reconstruction: the strengthened clause is
    /// a resolvent of two clauses already entailed by the formula, so it is
    /// entailed too, and it only ever *shrinks* the model set's complement —
    /// every model of the original formula still satisfies it. It is
    /// therefore safe across incremental `push`/`pop` scopes and with
    /// assumptions, and carries none of those two fields' caveats.
    ///
    /// Only ever runs from `Solver::inprocess`, so it is additionally gated by
    /// [`SolverConfig::enable_inprocessing`], and — like the rest of
    /// inprocessing — steps aside entirely while LRAT tracing is active. DRAT
    /// tracing is fully supported (each strengthening emits the shortened
    /// clause followed by a deletion of the original).
    pub enable_self_subsumption: bool,
    /// Run the CaDiCaL-style *lucky phase* before search starts: a handful of
    /// structurally-motivated full assignments (all-true, all-false, a
    /// forward and a backward clause-ordered pass, and the Horn / dual-Horn
    /// least-model closures) are built in scratch buffers and each checked
    /// against the original clauses. The first one that satisfies them all is
    /// installed as the model and `Sat` is returned without the CDCL loop
    /// ever running. See `solver/lucky.rs` for what each scan is and why the
    /// set is the one it is.
    ///
    /// **On by default**, unlike the rest of the pre-search toolkit. The
    /// whole phase is a small constant number of `O(vars + literals)` sweeps
    /// over scratch buffers with *no* effect on solver state — no clause is
    /// added, removed or strengthened, no variable is eliminated, nothing is
    /// assigned on the trail — so it is sound in every configuration
    /// (incremental scopes, assumptions, DRAT/LRAT tracing) and cannot change
    /// any verdict: it only ever answers `Sat` with a verified model, or
    /// declines and lets the search proceed exactly as it would have. On the
    /// instances it does not solve, the cost is a few passes over the clause
    /// database, which any instance hard enough to reach the CDCL loop repays
    /// within its first handful of conflicts.
    ///
    /// Turn it off to measure the search itself (a "before" number in a
    /// benchmark), or when a caller depends on the pre-search inprocessing
    /// toolkit having run — a lucky hit returns *before* probing, BVE,
    /// equivalent-literal substitution and subsumption, so on an instance
    /// lucky solves, their statistics stay at zero.
    pub enable_lucky_phase: bool,
}

impl core::fmt::Debug for SolverConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SolverConfig")
            .field("restart_interval", &self.restart_interval)
            .field("restart_multiplier", &self.restart_multiplier)
            .field("clause_deletion_threshold", &self.clause_deletion_threshold)
            .field("var_decay", &self.var_decay)
            .field("clause_decay", &self.clause_decay)
            .field("random_polarity_prob", &self.random_polarity_prob)
            .field("restart_strategy", &self.restart_strategy)
            .field("enable_lazy_hyper_binary", &self.enable_lazy_hyper_binary)
            .field("use_chb_branching", &self.use_chb_branching)
            .field("use_lrb_branching", &self.use_lrb_branching)
            .field("enable_inprocessing", &self.enable_inprocessing)
            .field("inprocessing_interval", &self.inprocessing_interval)
            .field(
                "enable_chronological_backtrack",
                &self.enable_chronological_backtrack,
            )
            .field(
                "chrono_backtrack_threshold",
                &self.chrono_backtrack_threshold,
            )
            .field("use_vmtf", &self.use_vmtf)
            .field("luby_cap", &self.luby_cap)
            .field("enable_stabilize", &self.enable_stabilize)
            .field("stabilize_base", &self.stabilize_base)
            .field("focused_luby_cap", &self.focused_luby_cap)
            .field("rephase_interval", &self.rephase_interval)
            .field("reuse_trail", &self.reuse_trail)
            .field(
                "external_branching",
                &self
                    .external_branching
                    .as_ref()
                    .map(|_| "<BranchingHeuristic>"),
            )
            .field(
                "enable_failed_literal_probing",
                &self.enable_failed_literal_probing,
            )
            .field("enable_bve", &self.enable_bve)
            .field("enable_equiv_substitution", &self.enable_equiv_substitution)
            .field("enable_gate_congruence", &self.enable_gate_congruence)
            .field("enable_self_subsumption", &self.enable_self_subsumption)
            .field("enable_lucky_phase", &self.enable_lucky_phase)
            .finish()
    }
}

/// Restart strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Luby sequence restarts
    Luby,
    /// Geometric restarts
    Geometric,
    /// Glucose-style dynamic restarts based on LBD
    Glucose,
    /// Local restarts based on LBD trail
    LocalLbd,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            restart_interval: 100,
            restart_multiplier: 1.5,
            clause_deletion_threshold: 10000,
            var_decay: 0.95,
            clause_decay: 0.999,
            random_polarity_prob: 0.02,
            restart_strategy: RestartStrategy::Luby,
            enable_lazy_hyper_binary: true,
            use_chb_branching: false,
            use_lrb_branching: false,
            enable_inprocessing: false,
            inprocessing_interval: 5000,
            enable_chronological_backtrack: true,
            chrono_backtrack_threshold: 100,
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            // Rephasing only pays off once other benchmarking has tuned an
            // interval for a given workload; off by default so a freshly
            // created solver behaves exactly like `enable_stabilize` alone
            // predicts. Presets opt into a tuned interval explicitly.
            rephase_interval: 0,
            reuse_trail: true,
            external_branching: None,
            // Off by default: although sound (it only ever forces facts, never
            // removes a variable), a probing-only pass can fully settle a
            // small/dense instance's verdict before the main CDCL loop ever
            // runs, changing observable solve behavior (conflict-analysis
            // hooks firing zero times on an UNSAT instance, for one) even
            // though the reported verdict itself never changes. Opt-in until
            // a caller has confirmed that shape.
            enable_failed_literal_probing: false,
            // BVE and equivalent-literal substitution both delete variables
            // from the live formula (recording reconstruction data to fix
            // their model value back up afterward), which is only sound at
            // the base assertion level with no active incremental `push` in
            // scope. Left opt-in until a caller has confirmed that shape.
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On by default: no variable is removed, no reconstruction data is
            // needed, and the strengthened clause is a resolvent of two
            // clauses the formula already entails — so unlike the two fields
            // above this is sound in every configuration, including
            // incremental and assumption-based solving. Still only reachable
            // when `enable_inprocessing` is set.
            enable_self_subsumption: true,
            // On by default — the only pre-search mechanism that is. It
            // touches no solver state at all (scratch buffers only), verifies
            // every candidate against the original clauses before reporting
            // it, and can therefore never change a verdict; the worst case is
            // a few wasted linear sweeps. See the field's doc comment.
            enable_lucky_phase: true,
        }
    }
}
