//! Main CDCL(T) SMT Solver module

pub(super) mod arith_axioms;
pub(super) mod array_axioms;
pub(super) mod branch_priority;
pub(super) mod candidates;
pub(super) mod check_array;
pub(super) mod check_bv;
pub(super) mod check_core;
pub(super) mod check_dt;
pub(super) mod check_fp;
pub(super) mod check_fp_model;
pub(super) mod check_nlsat;
pub(super) mod check_string;
pub(super) mod config;
pub(super) mod dt_axioms;
pub(super) mod encode;
pub(super) mod encode_guards;
pub(super) mod eq_skeleton;
pub(super) mod int_case_split;
pub(super) mod int_range_lp;
pub(super) mod model_blocking;
pub(super) mod model_builder;
pub(super) mod model_eval;
pub(super) mod pigeonhole;
pub(super) mod term_walk;
pub(super) mod theory_bv_encode;
pub(super) mod theory_manager;
pub(super) mod trail;
pub(super) mod types;
pub(super) mod verdict_cache;

pub use types::{
    FpConstraintData, Model, NamedAssertion, Proof, ProofStep, SolverConfig, SolverResult,
    Statistics, TheoryMode, UnsatCore,
};

use crate::mbqi::{MBQIIntegration, MBQIResult};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::simplify::Simplifier;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::ematching::{EmatchingConfig, EmatchingEngine};
use oxiz_core::sort::SortId;
#[cfg(test)]
use oxiz_sat::RestartStrategy;
use oxiz_sat::{
    Lit, Solver as SatSolver, SolverConfig as SatConfig, SolverResult as SatResult, Var,
};
use oxiz_theories::Theory;
use oxiz_theories::arithmetic::ArithSolver;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::euf::EufSolver;

use theory_manager::TheoryManager;
use trail::{ContextState, TrailOp};
use types::{Constraint, ParsedArithConstraint, Polarity};
use verdict_cache::GoalFingerprint;

/// Main CDCL(T) SMT Solver
#[derive(Debug)]
pub struct Solver {
    /// Configuration
    pub(super) config: SolverConfig,
    /// SAT solver core
    pub(super) sat: SatSolver,
    /// EUF theory solver
    pub(super) euf: EufSolver,
    /// Arithmetic theory solver
    pub(super) arith: ArithSolver,
    /// Bitvector theory solver
    pub(super) bv: BvSolver,
    /// Explanations for theory assertions tagged with a *derived* equality.
    ///
    /// Owned here rather than by the `TheoryManager` because the theory solvers
    /// above are: a fresh manager is built for every MBQI round while their
    /// state is deliberately kept, so an explanation stored in the manager would
    /// vanish in front of the assertion it explains.  See
    /// [`DerivedReasons`](theory_manager::DerivedReasons).
    pub(super) derived_reasons: theory_manager::DerivedReasons,
    /// NLSAT solver for nonlinear arithmetic (QF_NIA/QF_NRA).
    ///
    /// Present only under the `nlsat` feature: without it `oxiz_theories::nlsat`
    /// -- and the `oxiz-nlsat` crate behind it -- is not compiled, and this
    /// solver answers `unknown` on goals it would have decided (see
    /// `check_nlsat::dispatch_nl_solver`).
    #[cfg(feature = "nlsat")]
    pub(super) nlsat: Option<oxiz_theories::nlsat::NlsatTheory>,
    /// MBQI solver for quantified formulas
    pub(super) mbqi: MBQIIntegration,
    /// E-matching engine for quantifier instantiation via trigger patterns
    pub(super) ematch_engine: EmatchingEngine,
    /// Whether the formula contains quantifiers
    pub(super) has_quantifiers: bool,
    /// Uninterpreted-function symbols (`Spur`s) that occur as the head of
    /// some `Apply` reachable from a quantifier's body, as registered by
    /// `register_asserted_quantifiers`/`register_asserted_forall`.
    ///
    /// `purify_numeric_uf_args` consults this to scope its MBQI-avoidance
    /// gate to exactly the functions that interact with e-matching, rather
    /// than `has_quantifiers` wholesale: gating on `has_quantifiers` alone
    /// made purification's soundness fix order-dependent on *when* a
    /// quantifier happened to be asserted relative to an unrelated ground
    /// UF fact -- `(assert (forall ((z Int)) (> (g z) 0))) (assert (not (=
    /// (f y) (f 3))))` reintroduced the exact false-`sat` this module
    /// exists to close, because `g`'s quantifier flips `has_quantifiers` to
    /// `true` before `f`'s ground disequality is ever purified, even though
    /// `f` never appears under a binder. Scoping per function symbol keeps
    /// the fix sound for every function that is not itself a quantifier
    /// trigger, in any assertion order, at the cost of the same
    /// conservatism `has_quantifiers` already accepted for functions that
    /// genuinely are.
    pub(super) quantifier_uf_funcs: FxHashSet<oxiz_core::interner::Spur>,
    /// Next unused Skolem symbol id.
    ///
    /// Skolem symbols are named positionally (`sk!N` / `skf!N`) and names are
    /// interned, so two [`SkolemizationContext`](crate::skolemization::SkolemizationContext)s
    /// that both start at zero mint the *same* symbols.  Reusing one witness
    /// symbol for two unrelated existentials strengthens the assertion set and
    /// can turn a satisfiable problem unsatisfiable, so every Skolemization the
    /// solver performs threads this monotone counter (never reset by `pop`: a
    /// popped symbol's name must not come back attached to a different
    /// existential).
    pub(super) next_skolem_id: u64,
    /// Term to SAT variable mapping
    pub(super) term_to_var: FxHashMap<TermId, Var>,
    /// SAT variable to term mapping
    pub(super) var_to_term: Vec<TermId>,
    /// SAT variable to theory constraint mapping
    pub(super) var_to_constraint: FxHashMap<Var, Constraint>,
    /// SAT variable to parsed arithmetic constraint mapping
    pub(super) var_to_parsed_arith: FxHashMap<Var, ParsedArithConstraint>,
    /// Current logic
    pub(super) logic: Option<String>,
    /// Assertions
    pub(super) assertions: Vec<TermId>,
    /// Named assertions for unsat core tracking
    pub(super) named_assertions: Vec<NamedAssertion>,
    /// Assumption literals for unsat core tracking (maps assertion index to assumption var)
    /// Reserved for future use with assumption-based unsat core extraction
    #[allow(dead_code)]
    pub(super) assumption_vars: FxHashMap<u32, Var>,
    /// Model (if sat)
    pub(super) model: Option<Model>,
    /// Exact model values for the nonlinear-real variables that [`Model`]
    /// cannot hold — the `(get-model)` side-channel for algebraic witnesses.
    ///
    /// # Why a side-channel and not a `Model` entry
    ///
    /// [`Model`] maps a term to a *term*, and the value language of the term
    /// graph is `Rational64`. `√2` — the only value satisfying `x² = 2` — has
    /// no term, and inventing a `TermKind` for it would ripple through every
    /// exhaustive match over `TermKind` in the workspace for the sake of a
    /// value that is never computed with, only printed. More importantly it
    /// would put a non-rational into the value language that
    /// `Solver::model_refutes_assertions` and
    /// `oxiz_theories::nl_eval::holds_under` evaluate in, both of which are
    /// soundness gates. So the ordinary `Model` stays exactly as rational as
    /// it was, and the exact value travels beside it.
    ///
    /// Nothing re-checks these values here, and nothing needs to: the cell
    /// decomposition that produced them already verified the point against
    /// every assigned atom inside `oxiz-nlsat`, and no gate in this crate
    /// consumes them — `(get-model)` / `(get-value ..)` are the only readers.
    ///
    /// # Populated all-or-nothing
    ///
    /// Non-empty only when a `Sat` from the real cell decomposition needed the
    /// algebraic channel, and then it covers **every** variable of that
    /// problem, rationals included (see
    /// [`oxiz_theories::nlsat::NlDispatchResult`]). A partial map would let
    /// `Context::get_model` fill the rest in from sort defaults and print an
    /// assignment satisfying nothing.
    ///
    /// # Lifetime
    ///
    /// A `RESULT`: it describes the last `check`'s model exactly as `model`
    /// does, and is dropped by the same [`Self::invalidate_results`] hook, so
    /// a value cannot survive the `assert`/`push`/`pop` that invalidated the
    /// verdict it belongs to. `dispatch_nl_solver` additionally clears it on
    /// entry, so a `check` that reaches a *different* nonlinear procedure
    /// cannot inherit the previous one's values.
    pub(super) nl_algebraic_values: FxHashMap<TermId, oxiz_theories::nl_witness::NlWitnessValue>,
    /// Unsat core (if unsat)
    pub(super) unsat_core: Option<UnsatCore>,
    /// Context stack for push/pop
    pub(super) context_stack: Vec<ContextState>,
    /// Trail of operations for efficient undo
    pub(super) trail: Vec<TrailOp>,
    /// Tracking which literals have been processed by theories
    pub(super) theory_processed_up_to: usize,
    /// Whether to produce unsat cores
    pub(super) produce_unsat_cores: bool,
    /// Track if we've asserted False (for immediate unsat)
    pub(super) has_false_assertion: bool,
    /// Polarity tracking for optimization
    pub(super) polarities: FxHashMap<TermId, Polarity>,
    /// Whether polarity-aware encoding is enabled
    pub(super) polarity_aware: bool,
    /// Whether theory-aware branching is enabled
    pub(super) theory_aware_branching: bool,
    /// Proof of unsatisfiability (if proof generation is enabled)
    pub(super) proof: Option<Proof>,
    /// Formula simplifier
    pub(super) simplifier: Simplifier,
    /// Solver statistics
    pub(super) statistics: Statistics,
    /// Bitvector terms (for model extraction)
    pub(super) bv_terms: FxHashSet<TermId>,
    /// Whether we've seen arithmetic BV operations (division/remainder)
    /// Used to decide when to run eager BV checking
    pub(super) has_bv_arith_ops: bool,
    /// Arithmetic terms (Int/Real variables for model extraction)
    pub(super) arith_terms: FxHashSet<TermId>,
    /// Datatype constructor constraints: variable -> constructor name
    /// Used to detect mutual exclusivity conflicts (var = C1 AND var = C2 where C1 != C2)
    pub(super) dt_var_constructors: FxHashMap<TermId, oxiz_core::interner::Spur>,
    /// Cache for parsed arithmetic constraints, keyed by the comparison term id.
    /// `ParsedArithConstraint` is purely structural (depends only on the term graph),
    /// so it is safe to reuse across CDCL backtracks.
    pub(super) arith_parse_cache: FxHashMap<TermId, Option<ParsedArithConstraint>>,
    /// Set of compound term ids whose theory-variable sub-graph has been fully
    /// traversed by `track_theory_vars`.  Avoids redundant O(depth) re-walks
    /// when the same sub-expression appears in multiple parent constraints.
    pub(super) tracked_compound_terms: FxHashSet<TermId>,
    /// Bool-sorted `Var` terms known to appear in a UF-application argument
    /// position, populated by `abstract_compound_bool_args`. The
    /// `TermKind::Var` arm of `encode_depth_uncached` consults this to decide
    /// whether to register the variable for Bool completion
    /// (`Constraint::BoolApp`) -- see [`Solver::mark_bool_uf_arg`] for why
    /// this is gated rather than unconditional.
    pub(super) bool_uf_arg_terms: FxHashSet<TermId>,
    /// Every numeric (Int/Real) term known to appear as a direct argument of
    /// some uninterpreted-function application, populated by
    /// `purify_numeric_uf_args` — either a fresh purification proxy or a
    /// term `track_theory_vars` already shares on its own (typically a bare
    /// `Var`). This is the candidate pool for the non-convex integer
    /// case-split refinement (`int_case_split`): a UF argument tightly
    /// bounded to a small finite domain is exactly the shape that needs an
    /// explicit `(or (= t v0) ...)` lemma when Nelson-Oppen equality sharing
    /// alone cannot resolve which value the search should pick.
    pub(super) numeric_uf_arg_terms: FxHashSet<TermId>,
    /// Alias map from a pre-purification uninterpreted-function-application
    /// term (still what `self.assertions` and any external `get-value`
    /// query name) to the purified counterpart actually interned into
    /// EUF/arithmetic, populated by `purify_numeric_uf_args`.
    ///
    /// `Model::eval`/`build_model` resolve an `Apply` term only by direct
    /// `TermId` identity (an uninterpreted function has no structural
    /// evaluation rule), so without this alias a satisfiable model could
    /// never report a value for the original, unpurified application shape.
    pub(super) numeric_purify_aliases: FxHashMap<TermId, TermId>,
    /// Tseitin-encoding memo: term id -> (literal returned by `encode_depth`,
    /// polarity the term's clauses were emitted under).
    ///
    /// `get_or_create_var` caches only the SAT *variable*; without this map the
    /// encoder re-descends into every occurrence of a shared sub-term, which is
    /// exponential on a hash-consed DAG (each level referencing the previous
    /// twice gives `2^n` re-encodes and `2^n` duplicate clauses).
    ///
    /// The polarity component exists because `And`/`Or` are the only arms whose
    /// emitted clauses depend on more than the term itself: under
    /// `polarity_aware` they emit just one implication direction.  A hit is
    /// therefore only valid when the cached polarity covers the one the arm
    /// would use now (`Both` covers everything).  `collect_polarities` only
    /// ever *widens* a term's polarity, so on a widening miss the term is
    /// re-encoded — emitting the missing direction (plus harmless duplicates
    /// of the old one) — and the entry settles at `Both`.  Every other arm's
    /// clause set is polarity-independent, so those entries are stored as
    /// `Both` directly.
    ///
    /// Lifetime: every write is journalled as
    /// [`TrailOp::EncodedTermAdded`](super::solver::trail::TrailOp), so `pop`
    /// retracts exactly the entries whose clauses the matching `sat.pop()`
    /// retracts and keeps the ones written at an outer level (whose clauses
    /// survive).  Cleared wholesale only by `reset`.
    pub(super) encoded_terms: FxHashMap<TermId, (Lit, Polarity)>,
    /// Cache for FP constraint checking results.
    pub(super) fp_constraint_cache: FxHashMap<TermId, FpConstraintData>,
    /// Set to `true` when `encode` aborted a branch because the term nesting
    /// depth exceeded [`ENCODE_DEPTH_LIMIT`].  A truncated encoding leaves the
    /// affected sub-formula under-constrained, so the solver must answer
    /// `Unknown` rather than trust a model built over an incomplete encoding.
    pub(super) encode_depth_exceeded: bool,
    /// Set to `true` when any array `select`/`store` operation is encoded.  Gates
    /// the lazy array-axiom instantiation refinement (see
    /// [`Solver::instantiate_array_axioms`]) so non-array problems pay no cost.
    pub(super) has_array_ops: bool,
    /// Ground array-axiom instances (read-over-write / extensionality /
    /// select-congruence) already added to the SAT core as lemmas, keyed by the
    /// interned lemma term id.  Guarantees each valid instance is asserted at
    /// most once, which makes the in-loop refinement in `check` terminate: every
    /// refinement round either adds a strictly new instance or reports `Sat`.
    pub(super) array_axiom_instances: FxHashSet<TermId>,
    /// `div` / `mod` / numeric-`ite` terms whose defining axioms have already
    /// been asserted (see [`Solver::instantiate_arith_axioms`]).  The linear
    /// solver treats those terms as opaque atoms, so this set is what tells the
    /// honesty gate whether an atom's meaning is actually present: an atom
    /// mentioning a term that is *not* in here has no theory semantics and
    /// `check` must answer `Unknown`.
    pub(super) arith_defined_terms: FxHashSet<TermId>,
    /// Numeric `Eq` atoms that have already been given their trichotomy clause
    /// `(a = b) OR (a < b) OR (a > b)` by [`Solver::add_numeric_trichotomy`].
    ///
    /// The clause is what makes an arithmetic *disequality* reach the theory
    /// solver at all (see that method), and it must be emitted exactly once per
    /// atom. The Tseitin memo (`encoded_terms`) cannot be relied on for that:
    /// it is retracted per entry on `pop` and is also *bypassed* whenever a
    /// term is re-encoded under a widened polarity, so the `TermKind::Eq` arm
    /// runs again for an atom whose trichotomy is already in the database. Left
    /// unguarded that re-emitted a literal-identical clause over identical
    /// variables — which `oxiz_sat::Solver::add_clause` has no duplicate
    /// detection for — once per `(push)(pop)` pair, unbounded (exactly the
    /// growth `scope_rebase_tests::a_no_op_push_pop_between_checks_does_not_
    /// re_encode_the_goal` exists to catch).
    ///
    /// Journalled on the trail as [`TrailOp::NumericTrichotomyAdded`] so a
    /// `pop` drops the mark together with the clause it stands for, and the
    /// next encode re-emits both.
    pub(super) numeric_trichotomy_atoms: FxHashSet<TermId>,
    /// Ground datatype-axiom instances (exhaustiveness, exclusivity,
    /// reconstruction, selector-over-constructor, congruence, acyclicity)
    /// already added to the SAT core as lemmas, keyed by the interned lemma
    /// term id.  See [`Solver::instantiate_dt_axioms`].
    pub(super) dt_axiom_instances: FxHashSet<TermId>,
    /// Set to `true` when the datatype axiom budget ran out before every
    /// instance had been asserted.  The remaining axiomatisation is then a
    /// strict subset of the theory, so `Unsat` is still trustworthy but a `Sat`
    /// must be reported as `Unknown`.
    pub(super) dt_axioms_incomplete: bool,
    /// Set to `true` when the *array* axiom budget
    /// ([`array_axioms::MAX_ARRAY_AXIOM_INSTANCES`]) ran out before every
    /// applicable instance had been asserted, exactly mirroring
    /// [`Solver::dt_axioms_incomplete`].
    ///
    /// Hitting the cap means `instantiate_array_axioms` returned `false` — its
    /// "the candidate model satisfies every axiom" answer — for a reason that
    /// has nothing to do with the model.  `check_core` reads that `false` as
    /// permission to report `Sat`, so without this flag an exhausted budget
    /// turns straight into a fabricated verdict.  `Unsat` derived from a subset
    /// of the axioms is still `Unsat`; a `Sat` is a guess and is reported as
    /// `Unknown`.
    pub(super) array_axioms_incomplete: bool,
    /// Terms that the *current* assertion stack pins to a concrete integer,
    /// i.e. `t` appears in some top-level `(assert (= t <literal>))`.
    ///
    /// Consumed by [`Solver::finite_expand_assertion`] so that a quantifier
    /// guard such as `(< i n)` counts as a concrete bound once `(= n 5)` has
    /// been asserted.  Every entry is a consequence of the live assertion set,
    /// so substituting it preserves both `sat` and `unsat`.
    ///
    /// Folded in incrementally (see `entailed_int_consts_upto`) to keep the
    /// scan linear in the number of assertions over a whole run instead of
    /// quadratic, and dropped wholesale by `pop` — an entry justified by a
    /// retracted assertion must never survive it.
    pub(super) entailed_int_consts: FxHashMap<TermId, num_bigint::BigInt>,
    /// How many entries of `assertions` have been folded into
    /// `entailed_int_consts`.  Reset to `0` by `pop`, which invalidates the
    /// whole map.
    pub(super) entailed_int_consts_upto: usize,
    /// Test-only: the SAT clause count sampled at each MBQI **round boundary**
    /// this solver has crossed since it was created, cumulative over every
    /// `check`.
    ///
    /// A round boundary is the one place the quantifier loop re-enters the
    /// search: it encodes the round's instantiation / e-matching lemmas, calls
    /// [`Self::rebase_theory_state`] and builds a fresh `TheoryManager`.  Both
    /// facts a "repeated `check-sat` costs no clauses" claim needs are in this
    /// one vector — `len()` says how many boundaries were crossed (a plateau
    /// measured over calls that cross none measures nothing), and the values say
    /// what each crossing cost.  See `solver::scope_rebase_tests`.
    #[cfg(test)]
    pub(crate) mbqi_round_clauses: Vec<usize>,
    /// The most recent [`Self::check`]'s verdict, with the goal fingerprint it
    /// was computed from; see [`verdict_cache`].  Dropped by
    /// [`Self::invalidate_results`], the same hook that drops `model`.
    pub(super) last_check: Option<(GoalFingerprint, SolverResult)>,
    /// Monotone counter of *settings* changes — see [`Self::settings_changed`].
    ///
    /// Bumped by every mutator of something a [`Self::check`] reads that is not
    /// the assertion stack (the logic, the SAT engine's random seed, the
    /// configuration, the unsat-core and branching switches), and carried in the
    /// goal fingerprint so a cached verdict cannot survive one of them.
    pub(super) settings_epoch: u64,
    /// Terms already given an explicit integer case-split lemma, so the same
    /// term is never split twice. Trail-scoped like `numeric_uf_arg_terms`
    /// (see [`int_case_split`]) — **not** reset per `check`: the lemma clause
    /// [`Solver::split_narrow_int_domains`] asserts survives at the live SAT
    /// scope across a repeated `check-sat` on an unchanged goal, so the dedup
    /// entry must too, or a no-op repeated `check` would re-emit the same
    /// clause every time (unbounded clause-database growth — see
    /// `scope_rebase_tests`). A `pop` retracts both together.
    pub(super) case_split_terms: FxHashSet<TermId>,
    /// Number of reset-and-re-solve non-convex-LIA refinement rounds spent
    /// within the *current* `check`. Reset at the start of every
    /// `check_core`, unlike `case_split_terms`: it bounds how many extra
    /// solves this particular search is willing to pay for, not which terms
    /// are already split, so a later `check` on a still-unresolved goal must
    /// be able to spend its own round budget again.
    pub(super) case_split_rounds: u32,
    /// Set when a candidate `Sat` had case-split targets that were never
    /// actually branched on, because the affordability gate in `check_core`
    /// declined the refinement round.
    ///
    /// The gate compares the first solve's wall clock against
    /// [`int_case_split::REFINEMENT_TIME_CEILING_MS`], so without this flag the
    /// *verdict* depends on how fast the machine happened to be: the same
    /// query answered `unsat` on an idle box and `sat` under load, because the
    /// refinement that would have refuted the candidate was skipped. A budget
    /// may cost completeness, never soundness — so a skipped round downgrades
    /// `Sat` to `Unknown` exactly as [`Solver::dt_axioms_incomplete`] and
    /// [`Solver::array_axioms_incomplete`] do.
    ///
    /// Per-search like `case_split_rounds`, and cleared by the round that
    /// actually runs.
    pub(super) case_split_skipped_targets: bool,
    /// How many refuted candidate models have been excluded by
    /// [`model_blocking::Solver::block_refuted_model`] at the *current* context
    /// scope, and equivalently how many model-blocking clauses are live in the
    /// SAT database.
    ///
    /// Deliberately **not** per-search (unlike `case_split_rounds`): the
    /// clauses are permanent in `self.sat` until `Solver::pop` retracts them
    /// with `self.sat.pop()`, so a counter reset at `check_core` entry would
    /// let a second `check` report `Unsat` off a database still restricted by
    /// the first one's blocking clauses — a wrong `unsat`, strictly worse than
    /// the spurious `Unknown` the whole mechanism exists to remove. It is
    /// therefore snapshotted in [`trail::ContextState`] (restored by `pop` in
    /// lockstep with `sat.pop()`), zeroed only by `Solver::reset` (which calls
    /// `sat.reset()`), and never touched by `invalidate_results`.
    ///
    /// Nonzero means "the search space is restricted": see
    /// [`Solver::blocking_clauses_present`].
    pub(super) model_blocking_active: usize,
    /// Test-only event log: one entry per ground candidate model, recording
    /// whether `self.model` was still populated when `check_core` reached the
    /// case-split / array-axiom repair paths.
    ///
    /// The repairs read the candidate model to decide what needs repairing
    /// (`instantiate_array_axioms` reads it to skip axiom instances the
    /// candidate already satisfies), and the refutation gate used to run
    /// *before* them and clear it. Nothing observable from outside distinguishes
    /// "the repairs ran on a live model" from "the repairs ran on `None` and
    /// degenerated into eager instantiation", so the ordering is pinned here
    /// instead — same device, and same rationale, as `mbqi_round_clauses`.
    #[cfg(test)]
    pub(super) repair_paths_saw_model: Vec<bool>,
    /// Index terms of finite-map lookup spines flattened by
    /// [`encode::finite_map_ite`]. Fed into
    /// [`int_case_split::Solver::split_narrow_int_domains`]'s candidate set
    /// alongside `numeric_uf_arg_terms`, so a table index gets the same
    /// entailed-bound case-split treatment a non-convex-LIA UF argument
    /// already does, with no separate refinement loop to keep in sync.
    pub(super) lookup_index_terms: FxHashSet<TermId>,
    /// Triangles of the pure-equality fast path's chordal graph whose
    /// transitivity clauses are already in the SAT database, keyed by their
    /// three constants in ascending order. Trail-scoped exactly like
    /// `case_split_terms`: the clauses live at the SAT scope they were added
    /// at and survive a repeated `check-sat` on an unchanged goal, so without
    /// this a no-op repeat would re-assert every triangle again. See
    /// [`eq_skeleton`].
    pub(super) eq_transitivity_triangles: FxHashSet<[TermId; 3]>,
    /// Equivalence classes the pure-equality fast path proved, for the
    /// uninterpreted-sort constants it decided. Consumed by
    /// [`Self::euf_class_representative`] so `(get-model)` groups those
    /// constants into `@uc_S_n` witnesses correctly even though EUF never ran.
    /// Cleared at every fast-path attempt, so it only ever describes the most
    /// recent verdict.
    pub(super) equality_skeleton_classes: FxHashMap<TermId, u32>,
    /// The shared queue backing the opt-in branch-priority heuristic (see
    /// [`branch_priority`]). Always present; only ever read by the SAT
    /// engine when [`SolverConfig::enable_domain_first_branching`] was set
    /// before construction, but always safe to push into.
    pub(super) branch_priority: branch_priority::PriorityQueue,
}

/// Maximum term-nesting depth the recursive Tseitin encoder will descend
/// before bailing out.  Adversarially deep formulas would otherwise overflow
/// the native call stack (a hard crash / DoS); instead we stop, flag
/// [`Solver::encode_depth_exceeded`], and let `check` answer `Unknown`.
///
/// # Measured, not guessed
///
/// The historical value 2000 was refuted by measurement: an at-cap
/// `encode` descent on a 1 MiB worker thread (the smallest stack an embedder
/// plausibly hands us) **aborted the process before the cap could fire**, so
/// the guard protected nothing there.  Measured on this workspace's dev
/// profile (`opt-level = 1`, the profile the regression test runs under),
/// with an `Implies` chain exactly at the cap
/// (`encode_at_cap_depth_survives_a_one_mib_stack` in `encode/tests.rs`):
///
/// | cap  | thread stack | outcome |
/// |------|--------------|---------|
/// | 2000 | 1 MiB        | SIGABRT (stack overflow) |
/// | 512  | 256 KiB      | SIGABRT (stack overflow) |
/// | 512  | 384 KiB      | returns |
/// | 512  | 1 MiB        | returns |
///
/// i.e. the encoder costs 512-768 bytes of native stack per nesting level at
/// `opt-level = 1`, so 512 levels fit in 384 KiB and the committed 1 MiB
/// test carries a measured >= 2.7x head-room (release `opt-level = "z"`
/// frames are smaller still; unoptimised `opt-level = 0` frames of other
/// walks in this crate measured ~2.8x larger, which this margin also
/// absorbs).
///
/// Lowering the cap is *honest*: the flag routes to `Unknown`, never to a
/// wrong answer.  The completeness cost is confined to formulas nested
/// deeper than 512 — the SMT-LIB parser already refuses raw nesting past
/// 1024 (`MAX_PARSE_DEPTH`), so only the 513..=1024 band of parseable
/// scripts (and arbitrarily deep API-built terms) moves from "encoded" to
/// "honest `Unknown`", whereas the old 2000 admitted terms whose encoding
/// died on the native stack.  Every other recursive pass that runs behind
/// the assert-time gate (`simplify`, `collect_polarities`, Skolemization,
/// `eval_in_model`) inherits the same tightened bound.
pub(super) const ENCODE_DEPTH_LIMIT: u32 = 512;

/// A fully-evaluated ground value used by the model-verification soundness gate
/// ([`Solver::model_refutes_assertions`]).  Integers and reals are unified as an
/// exact rational so mixed Int/Real arithmetic and comparisons fold without loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum EvalVal {
    Bool(bool),
    Num(num_rational::Rational64),
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
        let proof_enabled = config.proof;

        // The branch-priority queue is always created so `Solver` methods
        // can push into it unconditionally; whether the SAT engine actually
        // consults it depends on `enable_domain_first_branching` below (see
        // `branch_priority`'s module doc for why that has to be decided
        // here, at construction, rather than later).
        let (branch_priority, priority_heuristic) = branch_priority::new_priority_branching();

        // Build SAT solver configuration from our config
        let sat_config = SatConfig {
            restart_strategy: config.restart_strategy,
            enable_inprocessing: config.enable_inprocessing,
            inprocessing_interval: config.inprocessing_interval,
            external_branching: if config.enable_domain_first_branching {
                Some(priority_heuristic)
            } else {
                None
            },
            // Bounded variable elimination. Without this line the SAT
            // engine's BVE pass was unreachable from this crate at any
            // setting: `oxiz_sat::SolverConfig::enable_bve` is `false` in
            // `Default` and in every one of the nine `ConfigPreset` bodies,
            // and nothing here overrode it. See the field's doc comment on
            // `SolverConfig` for why `thorough()` opts in and `balanced()`
            // does not.
            enable_bve: config.enable_bve,
            ..SatConfig::default()
        };

        // Note: The following features are controlled by the SAT solver's preprocessor
        // and clause management systems. We pass the configuration but the actual
        // implementation is in oxiz-sat:
        // - Clause minimization (via RecursiveMinimizer)
        // - Clause subsumption: NOT via oxiz-sat's `SubsumptionChecker` /
        //   `DynamicSubsumption` types, which are exported but never
        //   constructed anywhere in the workspace and are retained as
        //   reference formulations (see their own doc comments). What
        //   actually runs is inprocessing-driven subsumption inside oxiz-sat
        //   itself (`Preprocessor::subsumption_elimination`, invoked from
        //   `Solver::inprocess`), reachable because `enable_inprocessing`
        //   defaults to `true` in `balanced()`.
        // - Self-subsuming resolution (clause strengthening): likewise runs
        //   from `Solver::inprocess`, as `Solver::self_subsuming_resolution`,
        //   gated by oxiz-sat's `enable_self_subsumption` (on by default).
        // - Variable elimination: via the SAT engine's own
        //   `Solver::bounded_variable_elimination`, which keeps the model
        //   reconstruction data the `Preprocessor::variable_elimination`
        //   reference implementation does not. Reached through the
        //   `enable_bve` pass-through above, not through
        //   `enable_variable_elimination`.
        // - Blocked clause elimination (via Preprocessor::blocked_clause_elimination)
        // - Symmetry breaking (via SymmetryBreaker)

        Self {
            config,
            sat: SatSolver::with_config(sat_config),
            euf: EufSolver::new(),
            arith: ArithSolver::lra(),
            bv: BvSolver::new(),
            derived_reasons: theory_manager::DerivedReasons::default(),
            #[cfg(feature = "nlsat")]
            nlsat: None,
            mbqi: MBQIIntegration::new(),
            ematch_engine: EmatchingEngine::new(EmatchingConfig::default()),
            has_quantifiers: false,
            quantifier_uf_funcs: FxHashSet::default(),
            next_skolem_id: 0,
            term_to_var: FxHashMap::default(),
            var_to_term: Vec::new(),
            var_to_constraint: FxHashMap::default(),
            var_to_parsed_arith: FxHashMap::default(),
            logic: None,
            assertions: Vec::new(),
            named_assertions: Vec::new(),
            assumption_vars: FxHashMap::default(),
            model: None,
            nl_algebraic_values: FxHashMap::default(),
            unsat_core: None,
            context_stack: Vec::new(),
            trail: Vec::new(),
            theory_processed_up_to: 0,
            produce_unsat_cores: false,
            has_false_assertion: false,
            polarities: FxHashMap::default(),
            polarity_aware: true, // Enable polarity-aware encoding by default
            theory_aware_branching: true, // Enable theory-aware branching by default
            proof: if proof_enabled {
                Some(Proof::new())
            } else {
                None
            },
            simplifier: Simplifier::new(),
            statistics: Statistics::new(),
            bv_terms: FxHashSet::default(),
            has_bv_arith_ops: false,
            arith_terms: FxHashSet::default(),
            dt_var_constructors: FxHashMap::default(),
            arith_parse_cache: FxHashMap::default(),
            tracked_compound_terms: FxHashSet::default(),
            bool_uf_arg_terms: FxHashSet::default(),
            numeric_uf_arg_terms: FxHashSet::default(),
            numeric_purify_aliases: FxHashMap::default(),
            encoded_terms: FxHashMap::default(),
            fp_constraint_cache: FxHashMap::default(),
            encode_depth_exceeded: false,
            has_array_ops: false,
            array_axiom_instances: FxHashSet::default(),
            arith_defined_terms: FxHashSet::default(),
            numeric_trichotomy_atoms: FxHashSet::default(),
            dt_axiom_instances: FxHashSet::default(),
            dt_axioms_incomplete: false,
            array_axioms_incomplete: false,
            entailed_int_consts: FxHashMap::default(),
            entailed_int_consts_upto: 0,
            #[cfg(test)]
            mbqi_round_clauses: Vec::new(),
            last_check: None,
            settings_epoch: 0,
            case_split_terms: FxHashSet::default(),
            case_split_rounds: 0,
            case_split_skipped_targets: false,
            model_blocking_active: 0,
            #[cfg(test)]
            repair_paths_saw_model: Vec::new(),
            lookup_index_terms: FxHashSet::default(),
            eq_transitivity_triangles: FxHashSet::default(),
            equality_skeleton_classes: FxHashMap::default(),
            branch_priority,
        }
    }

    /// Get the proof (if proof generation is enabled and the result is unsat)
    #[must_use]
    pub fn get_proof(&self) -> Option<&Proof> {
        self.proof.as_ref()
    }

    /// Get the solver statistics
    #[must_use]
    pub fn get_statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Reset the solver statistics
    pub fn reset_statistics(&mut self) {
        self.statistics.reset();
    }

    /// Check if theory-aware branching is enabled
    #[must_use]
    pub fn theory_aware_branching(&self) -> bool {
        self.theory_aware_branching
    }

    /// Register a declared constant as an MBQI ground instantiation candidate.
    ///
    /// This must be called from the context layer whenever a `declare-const`
    /// command is processed, so that trigger-free quantifiers can be
    /// instantiated with constants that exist in scope.
    pub fn register_declared_const(&mut self, term: TermId, sort: SortId) {
        self.mbqi.register_declared_const(term, sort);
    }

    /// Check satisfiability of the asserted formulas.
    ///
    /// Wraps the private `Solver::check_core` with the arithmetic-definition
    /// fixpoint.
    /// `div`/`mod`/numeric-`ite` terms are opaque atoms to the linear solver and
    /// only carry meaning once `Solver::instantiate_arith_axioms` has asserted
    /// their defining axioms.  `check_core` axiomatises everything reachable
    /// from the assertions up front, but a lemma generated *during* the search
    /// can internalise a fresh such term afterwards — an MBQI instantiation, or
    /// an array read-over-write lemma whose `ite` is Int-sorted.  Each round
    /// here supplies the missing definitions and re-solves; when no definition
    /// can be supplied (a symbolic divisor) the verdict degrades to `Unknown`
    /// rather than trusting an atom the SAT core treated as a free Boolean.
    ///
    /// # Repeating a check on an unchanged goal costs nothing
    ///
    /// One `check` is one MBQI search, and it leaves no trace of itself behind:
    /// the candidate pool, dedup filter, blind-instantiation guard and round
    /// budget are snapshotted on entry and restored on exit — see
    /// [`MBQIIntegration::restore_search_state`] for the field-by-field split
    /// between goal state and search state.
    ///
    /// Above that, a repeated query on a goal the caller has not touched is
    /// answered from the previous verdict rather than run again.  Re-running is
    /// not idempotent — the SAT solver keeps what it learned, so the second
    /// search ends on a different model, and a model-based instantiator handed a
    /// different model emits lemmas over ground terms that did not exist before
    /// — and those clauses can never be retracted, because the assertion stack
    /// never moved (task #28).  The cache is dropped by
    /// `Solver::invalidate_results`, the same hook that drops the model and
    /// unsat core beside it, and is additionally gated on a structural
    /// fingerprint of the goal; the `solver::verdict_cache` module carries the
    /// full argument for why it cannot go stale.  Everything the caller is
    /// entitled to keep across
    /// checks is kept: the asserted and lemma clauses, the Tseitin memo, the
    /// registered quantifiers, and any candidate registered outside a check.
    pub fn check(&mut self, manager: &mut TermManager) -> SolverResult {
        let fingerprint = self.goal_fingerprint();
        if let Some(cached) = self.cached_verdict(&fingerprint) {
            return cached;
        }

        let mbqi_checkpoint = self.mbqi.search_checkpoint();
        let result = self.check_with_arith_refinement(manager);
        self.mbqi.restore_search_state(&mbqi_checkpoint);
        self.remember_verdict(result);
        result
    }

    /// The body of [`Self::check`], minus the caching and the MBQI search-state
    /// restore that wrap it.  Split out so both cover every exit, including the
    /// honesty gates' early returns.
    fn check_with_arith_refinement(&mut self, manager: &mut TermManager) -> SolverResult {
        /// Refinement rounds allowed before conceding.  Each round strictly
        /// grows `arith_defined_terms`, and the loop exits as soon as it stops
        /// growing, so this is only a guard against pathological growth.
        const MAX_ARITH_DEFINITION_ROUNDS: usize = 4;

        self.debug_check_invariants("check: entry");
        let mut result = self.check_core(manager);
        for _ in 0..MAX_ARITH_DEFINITION_ROUNDS {
            // Nothing left to refine.  `break` rather than `return`: the gates
            // below are the single exit for a `Sat` verdict, and returning here
            // would skip them.
            if result != SolverResult::Sat || !self.arith_defs_incomplete(manager) {
                break;
            }
            let defined_before = self.arith_defined_terms.len();
            self.instantiate_arith_axioms(manager);
            if self.arith_defined_terms.len() == defined_before {
                // Nothing new could be defined: the offending term has no linear
                // axiomatisation (a symbolic divisor).  Fall through to the gate.
                break;
            }
            result = self.check_core(manager);
        }
        // Honesty gate (soundness): the Tseitin encoder can refuse a sub-term
        // *during* the search as well.  MBQI instantiation results and
        // E-matching lemmas are encoded mid-loop, never pass the assert-time
        // depth pre-check, and `encode_depth`'s own cap then fires *after*
        // `check_core` consulted `encode_depth_exceeded` at its top.  A
        // truncated encoding only ever drops clauses, so an `Unsat` stays
        // sound, but a `Sat` may rest on constraints the truncation lost and
        // must degrade to `Unknown`.
        if result == SolverResult::Sat && self.encode_depth_exceeded {
            self.model = None;
            self.unsat_core = None;
            return SolverResult::Unknown;
        }
        if result == SolverResult::Sat && self.arith_defs_incomplete(manager) {
            self.model = None;
            self.unsat_core = None;
            return SolverResult::Unknown;
        }
        // Honesty gate (soundness): the datatype axiom budget ran out, so the
        // formula was solved against a strict subset of the datatype theory.
        // `Unsat` from a subset of the axioms is still `Unsat`; a `Sat` is a
        // guess and is reported as `Unknown`.
        if result == SolverResult::Sat && self.dt_axioms_incomplete {
            self.model = None;
            self.unsat_core = None;
            return SolverResult::Unknown;
        }
        // Same honesty gate for the array axiom budget: an exhausted cap makes
        // `instantiate_array_axioms` report "no new lemma" for a reason that is
        // about the budget, not about the model, and `check_core` reads that as
        // permission to answer `Sat`.
        if result == SolverResult::Sat && self.array_axioms_incomplete {
            self.model = None;
            self.unsat_core = None;
            return SolverResult::Unknown;
        }
        // Same honesty gate for a case-split round the affordability ceiling
        // declined: the candidate has shared terms whose domains were never
        // branched on, so this `Sat` is exactly the unverified kind
        // `int_case_split` exists to refute -- and which of the two verdicts
        // comes back must not depend on machine speed.
        if result == SolverResult::Sat && self.case_split_skipped_targets {
            self.model = None;
            self.unsat_core = None;
            return SolverResult::Unknown;
        }
        self.debug_check_invariants("check: exit");
        result
    }

    /// Return the incremental theory solvers to their base scope, keeping every
    /// fact the next search round is entitled to and dropping every fact that
    /// belonged to a search *branch*.
    ///
    /// # The leak this closes
    ///
    /// [`TheoryManager`] opens exactly one theory scope per SAT decision level
    /// (`push_theory_scope`) and unwinds them from its own `level_stack` on
    /// backtrack.  A CDCL(T) search that ends `Sat` **never backtracks**, so it
    /// returns with the EUF / arithmetic / bit-vector solvers several scopes
    /// deep, holding every assertion the winning branch made.  A fresh manager —
    /// built for the next MBQI round, or for the next `check` — starts with
    /// `level_stack == vec![0]`, and from that instant those scopes are
    /// *unreachable*: `on_backtrack(l)` pops only while `level_stack.len() > l +
    /// 1`, which is already false.  The branch's facts are committed for the
    /// lifetime of the `Solver`.
    ///
    /// That is a soundness hazard, not merely a leak.  The MBQI round that
    /// follows exists precisely to *retract* the branch the previous round chose:
    /// it adds an instantiation lemma such as `f(1) = 100 ⇒ x ≠ 1`, whose unit
    /// form drops the SAT trail to the root.  The next round then asserts `x ≠ 1`
    /// into a tableau that still contains the leaked `x = 1`, the arithmetic
    /// solver refutes it at decision level 0, and `solve_with_theory` answers
    /// `Unsat` on a satisfiable goal — before conflict analysis runs, so no
    /// honesty net is even consulted.
    ///
    /// # What the next round is entitled to
    ///
    /// Exactly the facts implied by the **root-level** SAT trail: the original
    /// ground assertions, plus every quantifier instance / e-matching lemma MBQI
    /// chose to keep.  The kept lemmas are universally justified instantiations,
    /// so they are sound at the base scope — and they are not stored in the
    /// theory solvers at all.  They live in the SAT clause database, where
    /// [`Self::encode`] put them and `oxiz_sat::Solver::add_clause` committed
    /// their unit consequences at the root.  What must *not* survive is the
    /// third category: assertions that were merely a search-branch decision.
    ///
    /// # Why reset-and-replay rather than popping the leaked scopes
    ///
    /// Popping down to the base scope looks more surgical, and it was tried: it
    /// makes the MBQI loop diverge on `tests/mbqi_sat_certification.rs`.  The
    /// reason is that popping addresses only half of the desynchronisation.  The
    /// SAT trail and the theory scope stack must be re-aligned *together*, and
    /// the SAT side is already at the root (`add_clause` calls
    /// `backtrack_to_root` for a unit lemma without notifying the theory layer).
    /// Unwinding the theory scopes alone leaves the next round replaying the
    /// whole trail — decisions included — into what it believes is the base
    /// scope, which commits the *new* branch's facts permanently instead of the
    /// old one's: the same defect, one round later, and now unrecoverable.
    ///
    /// Dropping the trail to the root and re-deriving the theory state from it is
    /// the alignment that actually holds.  `solve_with_theory` restarts with
    /// `theory_processed == 0` and replays the entire trail through
    /// `TheoryManager::on_assignment`, so the three solvers are repopulated —
    /// through the ordinary assert path, re-interning terms and re-registering
    /// theory variables as they go — from exactly the constraints that are active
    /// now.  Nothing is re-encoded, so the clause database and the
    /// [`Self::encoded_terms`] memo are untouched and the clause count plateaus
    /// across rounds (`scope_rebase_tests::clause_count_plateaus_across_rounds`).
    ///
    /// This is the same reset-and-replay strategy the array-axiom refinement
    /// rounds and [`TheoryManager::resync_theory_state`] already rely on; the
    /// solvers support only level-scoped `pop`, never point removal of a single
    /// mid-scope assertion, so it is also the only available way to retract a
    /// fact that a leaked scope has put out of reach.
    ///
    /// # The one seam for theory-state teardown
    ///
    /// Four sites need the theory solvers returned to a state that reflects only
    /// what is currently asserted, and all four call this function: `check`'s
    /// entry, the array-axiom refinement boundary, the MBQI round boundary, and
    /// [`Self::pop`].  They are deliberately *not* allowed to hand-roll the set
    /// — see the comment in `pop`, where an open-coded copy of it drifted (it
    /// omitted `bv.reset()`).  [`Self::reset`] is the one exception: it performs
    /// a strictly larger teardown that also drops the quantifier engines, the
    /// term/variable tables and every side cache, so it does not reduce to this.
    ///
    /// # Invariant: theory-variable registrations are re-derived, never carried
    ///
    /// The encoder registers theory variables directly into these long-lived
    /// solvers as it internalises a term — `Solver::register_arith_atom` calls
    /// `ArithSolver::intern`, the `Var` arm of `encode_depth` calls it and
    /// `BvSolver::new_bv`.  Those registrations are wiped by the resets here, so
    /// the invariant this function relies on is:
    ///
    /// > every theory-variable registration a later round needs is re-performed
    /// > by the trail replay, not inherited from before the rebase.
    ///
    /// It holds because the theory solvers are told about a term only through
    /// an assertion, and every assertion path re-registers what it mentions:
    /// `ArithSolver::assert_{eq,le,ge,lt,gt}` call `intern` on each term of the
    /// linear expression they are given, `TheoryManager::process_constraint`
    /// re-interns EUF nodes through `intern_term_for_congruence` (plus
    /// `intern_arith_shared_terms` for the arithmetic operands), and
    /// `bit_blast_bv_pair` re-blasts both operands — falling back to
    /// `BvSolver::new_bv` for a leaf — before every BV check.  The replay runs
    /// `on_assignment` over the *whole* trail, so each of those paths is taken
    /// again for every atom that is still asserted.
    ///
    /// The encoder-side memos (`Solver::arith_terms`, `Solver::bv_terms`,
    /// `Solver::tracked_compound_terms`) are *not* cleared, and deliberately so:
    /// they gate assert-time work — journalling a `TrailOp`, walking a compound
    /// term — and not the theory-solver registration, which the replay owns.
    /// Clearing them would re-journal trail operations for assertions that are
    /// already on the trail.  The consequence to be aware of is the converse: a
    /// term in `arith_terms` that appears in *no* asserted constraint (so the
    /// replay never re-interns it) loses its `ArithSolver` variable and hence
    /// its model value.  That is a model-completeness matter, not a soundness
    /// one — such a term is unconstrained by construction, so any value
    /// satisfies it, `Model` completion supplies one, and the candidate model is
    /// still put through `Solver::model_refutes_assertions` before any `Sat`.
    fn rebase_theory_state(&mut self) {
        // Root-level facts stay on the trail; only the decisions go.
        self.sat.backtrack_to_root();
        self.euf.reset();
        self.arith.reset();
        // The bit-vector solver additionally accumulates unit facts at its own
        // base level (`assert_const` pinning `x = 5`), which are not wired into
        // `Solver::push` / `pop` at all and would otherwise outlive both the
        // check and a user `pop` — a stale `x = 5` refuting a later `(= x 6)`.
        self.bv.reset();
        // The tableau these explanations were read out of is gone, and so are
        // the equalities they justified; keeping them would let a later conflict
        // cite literals belonging to a retracted scope.  This also returns the
        // absolute scope-depth counter to zero, matching the solvers.
        self.derived_reasons.clear();
    }

    /// Check satisfiability under assumptions
    /// Assumptions are temporary constraints that don't modify the assertion stack
    ///
    /// # Why the verdict survives the internal `pop`
    ///
    /// The assumption scope is realised as `push` / `assert` / `check` / `pop`,
    /// and [`Self::pop`] discards the results of the last check (see the private
    /// `Solver::invalidate_results`).  That rule is right for a *user* `pop`,
    /// which replaces the assertion stack, and wrong here: SMT-LIB 2.6 leaves
    /// `check-sat-assuming` in `sat` / `unsat` mode, so a following
    /// `(get-value ...)` / `(get-model)` must still read this check's model.
    ///
    /// Carrying the model across is sound because the assumption scope is a
    /// strict *extension* of the assertion stack: a model of
    /// `assertions ∧ assumptions` satisfies `assertions`, which is exactly the
    /// stack that survives the `pop`.
    ///
    /// The unsat core is carried under one condition, which is a soundness
    /// condition and not merely a bounds check.  With `:produce-unsat-cores` on
    /// every assumption is itself a tracked assertion, and `build_unsat_core`
    /// lists *every* tracked assertion — so a core computed here names an
    /// assumption index whenever any assumption was in play.  Such an index is
    /// past the end of the post-`pop` `assertions` vector: it dangles, exactly
    /// the failure the rule exists to prevent, and `invariants::check_unsat_core`
    /// rejects it.  Conversely a core that survives the range check is one that
    /// named no assumption at all, which makes it a core of the surviving stack.
    /// So: keep it when every index still resolves, drop it otherwise — and
    /// `(get-unsat-core)` then answers the standard "not available" error rather
    /// than a set that is not unsatisfiable on its own.
    ///
    /// The proof is deliberately *not* carried.  A refutation of
    /// `assertions ∧ assumptions` is not a refutation of `assertions`, so
    /// letting `(get-proof)` report it would be the same error the unsat-core
    /// condition above rules out, minus the index that makes it detectable.
    pub fn check_with_assumptions(
        &mut self,
        assumptions: &[TermId],
        manager: &mut TermManager,
    ) -> SolverResult {
        // Save current state
        self.push();

        // Assert all assumptions
        for &assumption in assumptions {
            self.assert(assumption, manager);
        }

        // Check satisfiability
        let result = self.check(manager);

        // Take the verdict out before the `pop` drops it, then restore it (see
        // the doc comment for why each half is or is not allowed to survive).
        let model = self.model.take();
        let unsat_core = self.unsat_core.take();

        // Restore state
        self.pop();

        self.model = model;
        let num_assertions = self.assertions.len() as u32;
        self.unsat_core =
            unsat_core.filter(|core| core.indices.iter().all(|&i| i < num_assertions));
        self.debug_check_invariants("after check_with_assumptions");

        result
    }

    /// Whether a complete interpretation satisfying *every* assertion can be
    /// built from the current candidate model and verified.
    ///
    /// This is the honest route from "the ground search is satisfied" to `sat`
    /// on a quantified goal: [`crate::mbqi::model_certify::certify`] answers
    /// `true` only with a total interpretation in hand that it has checked
    /// against every assertion, quantifiers included, over the whole of their
    /// domain.  A `false` answer claims nothing and leaves the caller's
    /// existing behaviour — ultimately `unknown` — in place.
    fn certify_quantified_sat(&mut self, manager: &TermManager) -> bool {
        let Some(model) = self.model.as_ref() else {
            return false;
        };
        let assignments = model.assignments().clone();
        crate::mbqi::model_certify::certify(&self.assertions, &assignments, manager)
    }

    /// Sound sufficient check used only at the MBQI incompleteness fallback.
    ///
    /// Returns `true` iff every assertion that carries a quantifier is
    /// *trivially valid* — i.e. it simplifies to `True` in every model.  In
    /// that case the quantifiers add no constraint and the model already found
    /// by the SAT/theory layer satisfies the whole formula, so answering `Sat`
    /// is sound.  Any quantified assertion we cannot prove trivially valid
    /// makes this return `false`, so the solver conservatively answers
    /// `Unknown` instead of fabricating an unverified `Sat`.
    fn quantifiers_trivially_valid(&mut self, manager: &mut TermManager) -> bool {
        let assertions = self.assertions.clone();
        for assertion in assertions {
            // Quantifier-free assertions are already satisfied by the model the
            // SAT/theory search produced (that is why we reached the Sat
            // branch); only quantified assertions need a validity proof.
            if oxiz_core::tactic::contains_quantifier(assertion, manager)
                && !self.term_is_valid(assertion, manager)
            {
                return false;
            }
        }
        true
    }

    /// Returns `true` only when `term` is *valid* (True in every model).
    ///
    /// This is a sound (never over-claiming) syntactic check: a term is valid
    /// when it simplifies to `True`, when it is `forall x. body` with a valid
    /// body, or when it is a conjunction of valid terms.  Every other shape —
    /// including a universal whose body is merely satisfiable — yields `false`.
    fn term_is_valid(&mut self, term: TermId, manager: &mut TermManager) -> bool {
        let simplified = self.mbqi.deep_simplify(term, manager);
        match manager.get(simplified).map(|t| t.kind.clone()) {
            Some(TermKind::True) => true,
            Some(TermKind::Forall { body, .. }) => self.term_is_valid(body, manager),
            Some(TermKind::And(args)) => args.iter().all(|&conj| self.term_is_valid(conj, manager)),
            _ => false,
        }
    }

    /// Check satisfiability (pure SAT, no theory integration)
    /// Useful for benchmarking or when theories are not needed
    pub fn check_sat_only(&mut self, manager: &mut TermManager) -> SolverResult {
        // Trivial verdicts first, mirroring `check_core`: an asserted `False`
        // never reaches the SAT core as a clause (`assert` records the flag
        // and returns), so solving the clause set alone would miss it and
        // report a wrong `Sat` for e.g. `{false}`.
        if self.has_false_assertion {
            return SolverResult::Unsat;
        }
        if self.assertions.is_empty() {
            return SolverResult::Sat;
        }
        // Honesty gate (soundness): an assertion the encoder refused because
        // it nests deeper than `ENCODE_DEPTH_LIMIT` contributed *no clauses at
        // all*, so even the pure-SAT view of the problem is incomplete — a
        // verdict over the remaining clauses would be a guess about a formula
        // this solver never saw.  Same rule as `check_core`'s top gate.
        if self.encode_depth_exceeded {
            return SolverResult::Unknown;
        }

        // Soundness gate: an earlier `check` may have left model-blocking
        // clauses in this same SAT core (see `solver::model_blocking`). They
        // are a restriction of the search space, not consequences of the
        // assertions, so an `Unsat` over them is not a refutation of the goal.
        // The `Sat` side needs no gate: a surviving assignment is a genuine
        // assignment of the unrestricted clause set.
        match self.sat.solve() {
            SatResult::Sat => {
                self.build_model(manager);
                SolverResult::Sat
            }
            SatResult::Unsat if self.blocking_clauses_present() => SolverResult::Unknown,
            SatResult::Unsat => SolverResult::Unsat,
            SatResult::Unknown => SolverResult::Unknown,
        }
    }

    /// Build the model after SAT solving, which can be used to efficiently extract minimal unsat cores
    pub fn enable_assumption_based_cores(&mut self) {
        self.produce_unsat_cores = true;
        // Assumption variables would be created during assertion
        // to enable fine-grained core extraction
    }

    /// Minimize an unsat core using greedy deletion
    /// This creates a minimal (but not necessarily minimum) unsatisfiable subset
    pub fn minimize_unsat_core(&mut self, manager: &mut TermManager) -> Option<UnsatCore> {
        if !self.produce_unsat_cores {
            return None;
        }

        // Get the current unsat core
        let core = self.unsat_core.as_ref()?;
        if core.is_empty() {
            return Some(core.clone());
        }

        // Extract the assertions in the core
        let mut core_assertions: Vec<_> = core
            .indices
            .iter()
            .map(|&idx| {
                let assertion = self.assertions[idx as usize];
                let name = self
                    .named_assertions
                    .iter()
                    .find(|na| na.index == idx)
                    .and_then(|na| na.name.clone());
                (idx, assertion, name)
            })
            .collect();

        // Try to remove each assertion one by one
        let mut i = 0;
        while i < core_assertions.len() {
            // Create a temporary solver with all assertions except the i-th one
            let mut temp_solver = Solver::new();
            temp_solver.set_logic(self.logic.as_deref().unwrap_or("ALL"));

            // Add all assertions except the i-th one
            for (j, &(_, assertion, _)) in core_assertions.iter().enumerate() {
                if i != j {
                    temp_solver.assert(assertion, manager);
                }
            }

            // Check if still unsat
            if temp_solver.check(manager) == SolverResult::Unsat {
                // Still unsat without this assertion - remove it
                core_assertions.remove(i);
                // Don't increment i, check the next element which is now at position i
            } else {
                // This assertion is needed
                i += 1;
            }
        }

        // Build the minimized core
        let mut minimized = UnsatCore::new();
        for (idx, _, name) in core_assertions {
            minimized.indices.push(idx);
            if let Some(n) = name {
                minimized.names.push(n);
            }
        }

        Some(minimized)
    }

    /// Get the model (if sat)
    #[must_use]
    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// Exact model values for nonlinear-real variables whose witness is not a
    /// rational — see [`Solver::nl_algebraic_values`](Self::nl_algebraic_values)
    /// (the field) for the whole contract.
    ///
    /// Empty for every model that fits in [`Model`], which is all of them
    /// except a real cell decomposition that had to answer with an algebraic
    /// number. Non-empty implies [`Self::model`] is `None`: the two are
    /// alternatives, not layers.
    #[must_use]
    pub fn nl_algebraic_values(
        &self,
    ) -> &FxHashMap<TermId, oxiz_theories::nl_witness::NlWitnessValue> {
        &self.nl_algebraic_values
    }

    /// The exact value this solver holds for `term`, if the last `check`
    /// answered it with a nonlinear-real witness. See
    /// [`Self::nl_algebraic_values`].
    #[must_use]
    pub fn nl_algebraic_value(
        &self,
        term: TermId,
    ) -> Option<&oxiz_theories::nl_witness::NlWitnessValue> {
        self.nl_algebraic_values.get(&term)
    }

    /// Congruence-closed function-application entries from the EUF solver for
    /// the given function symbol id (crate-internal use only).
    ///
    /// Each entry's argument and result classes have already been canonicalized
    /// through the union-find, so callers building a `FuncInterp` get congruence
    /// applied for free (e.g. `f(a)` and `f(b)` collapse when `a = b`).  The
    /// `func_id` is the EUF function symbol id, which for an `Apply` term is the
    /// underlying value of the function-name `Spur` (`spur.into_inner().get()`).
    #[must_use]
    pub(crate) fn euf_function_entries(
        &self,
        func_id: u32,
    ) -> Vec<oxiz_theories::euf::FuncAppEntry> {
        self.euf.function_application_entries(func_id)
    }

    /// Check satisfiability with resource limits.
    pub fn check_with_limits(
        &mut self,
        manager: &mut TermManager,
        limits: &crate::resource_limits::ResourceLimits,
    ) -> core::result::Result<SolverResult, crate::resource_limits::ResourceExhausted> {
        use crate::resource_limits::ResourceMonitor;
        let mut monitor = ResourceMonitor::new(limits.clone());
        if let Some(reason) = monitor.check() {
            return Err(reason);
        }
        let orig_max_conflicts = self.config.max_conflicts;
        let orig_max_decisions = self.config.max_decisions;
        if let Some(max_c) = limits.max_conflicts {
            if self.config.max_conflicts == 0 || max_c < self.config.max_conflicts {
                self.config.max_conflicts = max_c;
            }
        }
        if let Some(max_d) = limits.max_decisions {
            if self.config.max_decisions == 0 || max_d < self.config.max_decisions {
                self.config.max_decisions = max_d;
            }
        }
        let result = self.check(manager);
        self.config.max_conflicts = orig_max_conflicts;
        self.config.max_decisions = orig_max_decisions;
        monitor.conflicts = self.statistics.conflicts;
        monitor.decisions = self.statistics.decisions;
        monitor.restarts = self.statistics.restarts;
        monitor.theory_checks =
            self.statistics.theory_propagations + self.statistics.theory_conflicts;
        if result == SolverResult::Unknown {
            if let Some(reason) = monitor.check() {
                return Err(reason);
            }
        }
        Ok(result)
    }

    /// Assert multiple terms at once
    /// This is more efficient than calling assert() multiple times
    pub fn assert_many(&mut self, terms: &[TermId], manager: &mut TermManager) {
        for &term in terms {
            self.assert(term, manager);
        }
    }

    /// Get the number of assertions in the solver
    #[must_use]
    pub fn num_assertions(&self) -> usize {
        self.assertions.len()
    }

    /// Get the number of variables in the SAT solver
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.term_to_var.len()
    }

    /// Check if the solver has any assertions
    #[must_use]
    pub fn has_assertions(&self) -> bool {
        !self.assertions.is_empty()
    }

    /// Get the current context level (push/pop depth)
    #[must_use]
    pub fn context_level(&self) -> usize {
        self.context_stack.len()
    }

    /// Push a context level
    pub fn push(&mut self) {
        // A `push` opens a scope the previous verdict knew nothing about.  It
        // adds no assertion by itself, so the old model would still satisfy the
        // stack *at this instant* — but the only way to observe it is to ask
        // after the `assert`s that follow, by which time it is stale.  Dropping
        // it here rather than at the first following `assert` keeps the rule
        // stated once ("the verdict belongs to the stack it was computed on")
        // instead of depending on what the caller does next.
        self.invalidate_results();
        self.context_stack.push(ContextState {
            num_assertions: self.assertions.len(),
            num_vars: self.var_to_term.len(),
            has_false_assertion: self.has_false_assertion,
            trail_position: self.trail.len(),
            num_mbqi_quantifiers: self.mbqi.num_quantifiers(),
            num_ematch_quantifiers: self.ematch_engine.num_quantifiers(),
            has_quantifiers: self.has_quantifiers,
            quantifier_uf_funcs: self.quantifier_uf_funcs.clone(),
            has_bv_arith_ops: self.has_bv_arith_ops,
            has_array_ops: self.has_array_ops,
            encode_depth_exceeded: self.encode_depth_exceeded,
            dt_axioms_incomplete: self.dt_axioms_incomplete,
            array_axioms_incomplete: self.array_axioms_incomplete,
            model_blocking_active: self.model_blocking_active,
        });
        self.sat.push();
        // No EUF / arithmetic scope is opened here on purpose.
        //
        // It used to be, and it was an *untracked* scope: it bypassed
        // `TheoryManager::push_theory_scope`, so the absolute depth counter in
        // `derived_reasons` did not see it, and the matching `Solver::pop`
        // resets the two solvers wholesale rather than popping (see the comment
        // there for why a single pop retracted an arbitrary search scope).  A
        // push with no pop, invisible to the one counter that tracks the true
        // depth, is exactly the shape of defect `rebase_theory_state` exists to
        // remove.
        //
        // Dropping it is behaviour-preserving: every `check` now rebases the
        // three theory solvers at its entry and re-derives their state from the
        // SAT trail, so nothing between this `push` and the next verdict can
        // observe the scope, and `pop` resets regardless.
        #[cfg(feature = "nlsat")]
        if let Some(nlsat) = &mut self.nlsat {
            nlsat.push();
        }
        self.debug_check_invariants("after push");
    }

    /// Pop a context level using trail-based undo
    pub fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // The verdict on the table was computed against the scope being
            // retracted.  Its unsat core indexes an `assertions` vector this
            // call is about to truncate, so keeping it hands
            // `minimize_unsat_core` a dangling index (it panicked outright on
            // `(push)(assert)(check-sat)(pop)(get-unsat-core)`) and trips
            // `invariants::check_unsat_core` in the debug net below.  See
            // `invalidate_results` for the rule.
            self.invalidate_results();

            // Undo all operations in the trail since the push
            while self.trail.len() > state.trail_position {
                if let Some(op) = self.trail.pop() {
                    match op {
                        TrailOp::AssertionAdded { index } => {
                            if self.assertions.len() > index {
                                self.assertions.truncate(index);
                            }
                        }
                        TrailOp::VarCreated { var: _, term } => {
                            // Remove the term-to-var mapping
                            self.term_to_var.remove(&term);
                        }
                        TrailOp::ConstraintAdded { var } => {
                            // Remove the constraint.  Every site that records a
                            // parsed linear constraint for a variable also
                            // records this op for the same variable, so the two
                            // maps are retracted together — a surviving
                            // `var_to_parsed_arith` entry would keep feeding a
                            // retracted inequality to the arithmetic solver.
                            self.var_to_constraint.remove(&var);
                            self.var_to_parsed_arith.remove(&var);
                        }
                        TrailOp::FalseAssertionSet => {
                            // Reset the flag
                            self.has_false_assertion = false;
                        }
                        TrailOp::NamedAssertionAdded { index } => {
                            // Remove the named assertion
                            if self.named_assertions.len() > index {
                                self.named_assertions.truncate(index);
                            }
                        }
                        TrailOp::BvTermAdded { term } => {
                            // Remove the bitvector term
                            self.bv_terms.remove(&term);
                        }
                        TrailOp::ArithTermAdded { term } => {
                            // Remove the arithmetic term
                            self.arith_terms.remove(&term);
                        }
                        TrailOp::DtVarConstructorAdded { term } => {
                            // Forget the constructor this variable was pinned
                            // to; keeping it would make a later, unrelated
                            // constructor equality look mutually exclusive with
                            // a retracted one and force a wrong `unsat`.
                            self.dt_var_constructors.remove(&term);
                        }
                        TrailOp::TrackedCompoundAdded { term } => {
                            // Re-open this compound term for traversal: the
                            // theory-variable registrations the walk performed
                            // are themselves undone above, so the memo must go
                            // too or the sub-terms would never be re-registered.
                            self.tracked_compound_terms.remove(&term);
                        }
                        TrailOp::BoolUfArgAdded { term } => {
                            // The assertion that put this Bool variable in a
                            // UF-argument position is retracted with the
                            // scope's clauses, so it must stop being
                            // completed too -- otherwise a later, unrelated
                            // scope would pay EUF bookkeeping for a variable
                            // that no longer appears near any UF application.
                            self.bool_uf_arg_terms.remove(&term);
                        }
                        TrailOp::NumericUfArgAdded { term } => {
                            // Mirrors `BoolUfArgAdded`: the assertion that put
                            // this numeric term in a UF-argument position is
                            // gone, so it must stop being an int-case-split
                            // candidate for it too.
                            self.numeric_uf_arg_terms.remove(&term);
                        }
                        TrailOp::NumericPurifyAliasAdded { term } => {
                            // The purification that produced this alias is
                            // retracted with the scope's clauses; a later,
                            // unrelated scope must not resolve `term`'s model
                            // value through a purified twin that no longer
                            // exists.
                            self.numeric_purify_aliases.remove(&term);
                        }
                        TrailOp::EqTransitivityTriangleAdded { triangle } => {
                            // The three clauses this entry stands for lived at
                            // the scope now retracted, so the dedup entry has
                            // to go with them — otherwise a later search would
                            // believe the triangle is still constrained when
                            // its clauses are gone.
                            self.eq_transitivity_triangles.remove(&triangle);
                        }
                        TrailOp::CaseSplitTermAdded { term } => {
                            // The case-split lemma clause for this term lived
                            // at the scope now retracted, so the dedup entry
                            // must go too — otherwise a later, still-live
                            // search would believe the term already has a
                            // case-split disjunction in the clause database
                            // when it does not.
                            self.case_split_terms.remove(&term);
                        }
                        TrailOp::ArrayAxiomInstanceAdded { term } => {
                            // The lemma clause for this instance is retracted
                            // with the scope's clauses, so its dedup entry must
                            // go as well — otherwise a later scope would never
                            // re-assert an axiom it still needs.
                            self.array_axiom_instances.remove(&term);
                        }
                        TrailOp::ArithDefinedTermAdded { term } => {
                            // The defining lemmas for this `div`/`mod`/`ite`
                            // term are retracted with the scope's clauses, so
                            // the mark must go too: keeping it would let the
                            // honesty gate trust an atom whose meaning has just
                            // been dropped from the SAT core.
                            self.arith_defined_terms.remove(&term);
                        }
                        TrailOp::NumericTrichotomyAdded { term } => {
                            // The trichotomy clause is retracted with the
                            // scope's clauses, so the mark must go too — or the
                            // atom would keep looking "already split" while the
                            // clause that split it is gone, and the arithmetic
                            // disequality would stop reaching the tableau.
                            self.numeric_trichotomy_atoms.remove(&term);
                        }
                        TrailOp::EncodedTermAdded { term, previous } => {
                            // Take back exactly this one memo write.  `None`
                            // means the term's whole encoding was emitted inside
                            // this scope (forget it, so the next `encode`
                            // re-emits); `Some` means only the implication
                            // direction that *widened* the coverage was (restore
                            // the narrower coverage, whose clauses predate the
                            // push and survive `sat.pop()`).
                            match previous {
                                Some(entry) => {
                                    self.encoded_terms.insert(term, entry);
                                }
                                None => {
                                    self.encoded_terms.remove(&term);
                                }
                            }
                        }
                        TrailOp::DtAxiomInstanceAdded { term } => {
                            // Same reasoning as the array axioms: the lemma
                            // clause is retracted with the scope's clauses, so
                            // its dedup entry must go too or a later scope would
                            // never re-assert a datatype axiom it still needs.
                            self.dt_axiom_instances.remove(&term);
                        }
                    }
                }
            }

            // Use state to restore other fields
            self.assertions.truncate(state.num_assertions);

            // Every entailed-constant entry was justified by *some* assertion,
            // and the truncation above may have retracted it.  Dropping the map
            // wholesale (rather than journalling each entry) keeps a popped
            // `(assert (= n 5))` from making a later quantifier's `(< i n)`
            // look like the concrete bound `i <= 4` — an unsound expansion.
            // The next quantified assertion re-folds the surviving assertions,
            // so the only cost is one linear re-scan per `pop`.
            self.entailed_int_consts.clear();
            self.entailed_int_consts_upto = 0;
            self.var_to_term.truncate(state.num_vars);
            self.has_false_assertion = state.has_false_assertion;

            // Quantifier reasoning state: MBQI and the e-matching engine turn a
            // registered quantifier into hard ground lemmas, so a quantifier
            // whose only asserting scope has been popped must stop being
            // instantiated (it produced a false `unsat` otherwise).  Both
            // registries are append-only, so the push-time counts restore them
            // exactly.
            self.mbqi.truncate_quantifiers(state.num_mbqi_quantifiers);
            self.ematch_engine
                .truncate_quantifiers(state.num_ematch_quantifiers);
            self.has_quantifiers = state.has_quantifiers;
            self.quantifier_uf_funcs = state.quantifier_uf_funcs.clone();

            // Sticky encoder flags derived from the retracted assertions.
            self.has_bv_arith_ops = state.has_bv_arith_ops;
            self.has_array_ops = state.has_array_ops;
            self.encode_depth_exceeded = state.encode_depth_exceeded;
            self.dt_axioms_incomplete = state.dt_axioms_incomplete;
            self.array_axioms_incomplete = state.array_axioms_incomplete;

            // Model-blocking clauses added inside the retracted scope go away
            // with the `self.sat.pop()` below, so the count of live ones has to
            // roll back with them.  Restoring the push-time value (rather than
            // zeroing) is what keeps the `Unsat`-downgrade honest across nested
            // scopes: a clause added *outside* this scope survives the pop and
            // must keep the verdict downgraded.
            self.model_blocking_active = state.model_blocking_active;

            // The Tseitin memo is retracted per entry, by the
            // `TrailOp::EncodedTermAdded` arm of the undo loop above — not
            // cleared wholesale, as it once was.  The clear was justified by
            // "`sat.pop()` retracts the definitional clauses of everything in
            // the memo", which holds only for entries written *inside* the
            // scope: an entry written outside keeps its clauses and (because
            // `TrailOp::VarCreated` is journalled) its SAT variable, so dropping
            // it made the next `encode` walk the term again and re-emit
            // byte-identical clauses that `oxiz_sat::Solver::add_clause` cannot
            // recognise as duplicates — one extra copy of the encoding per
            // `(push)(pop)` pair, growing without bound (task #28).  See
            // `Solver::memoize_encoding` for why the journalled `previous` value
            // makes the per-entry retraction exact.
            self.sat.pop();

            // Drop the incremental theory state instead of popping one scope off
            // it, through the *same seam* the search-time rebase uses.
            //
            // Their scope stacks are NOT aligned with the assertion scopes: the
            // CDCL(T) search pushes one theory scope per *decision level* and
            // `TheoryManager::resync_theory_state` rebuilds the stack from
            // scratch, so after a `check` the depth bears no relation to the
            // depth `Solver::push` left behind.  A single `pop` here therefore
            // retracted an arbitrary search scope and left facts derived from the
            // retracted assertions committed — an MBQI instantiation lemma
            // (`f(7) = 1`) survived its scope and refuted the satisfiable
            // `(= (f k) 2)` that followed the `pop`.
            //
            // Resetting is safe because the theory state is fully re-derived on
            // every `check`: `Solver::check` builds a fresh `TheoryManager` and
            // `solve_with_theory` replays the *entire* SAT trail through it, so
            // each solver is repopulated from exactly the constraints that are
            // active in the current context — which is precisely what
            // [`Self::rebase_theory_state`] documents and does.
            //
            // Routing through it (rather than repeating three of its five lines
            // here) is what keeps the two teardowns from drifting: this site used
            // to reset `euf` and `arith` and clear `derived_reasons` but *not*
            // reset `bv`, so a bit-vector unit fact pinned at the BV solver's own
            // base level (`assert_const` for `x = 5`) outlived the scope that
            // asserted it, until the next `check`'s rebase happened to clear it.
            // The `sat.backtrack_to_root()` the rebase performs first is a no-op
            // here: `sat.pop()` above already ends at decision level 0, and
            // `Trail::backtrack_to_with_callback` returns immediately when asked
            // for a level it is already at — in particular it does not disturb
            // the propagation head that `sat.pop()` deliberately rewound.
            self.rebase_theory_state();
            #[cfg(feature = "nlsat")]
            if let Some(nlsat) = &mut self.nlsat {
                nlsat.pop();
            }

            #[cfg(debug_assertions)]
            self.debug_assert_scope_restored(&state);
            // Structural counterpart to the scope check above: the undo trail
            // must have left the term/variable maps, the side tables and the
            // remaining context snapshots mutually consistent.
            self.debug_check_invariants("after pop");
        }
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.sat.reset();
        self.euf.reset();
        self.arith.reset();
        self.bv.reset();
        self.derived_reasons.clear();
        // Quantifier reasoning state must be cleared too: leaving the previous
        // problem's quantifiers, e-matching triggers, Skolem candidates, and the
        // `has_quantifiers` flag in place would make a subsequent `check` apply
        // stale quantifiers (and take the MBQI path) for a brand-new formula —
        // a correctness defect.  Rebuild the engines from scratch.
        self.mbqi = MBQIIntegration::new();
        self.ematch_engine = EmatchingEngine::new(EmatchingConfig::default());
        self.has_quantifiers = false;
        self.quantifier_uf_funcs.clear();
        #[cfg(feature = "nlsat")]
        {
            self.nlsat = None;
        }
        self.term_to_var.clear();
        self.var_to_term.clear();
        self.var_to_constraint.clear();
        self.var_to_parsed_arith.clear();
        self.assertions.clear();
        self.named_assertions.clear();
        self.invalidate_results();
        self.context_stack.clear();
        self.trail.clear();
        self.logic = None;
        // `reset` clears the logic without going through `set_logic`, so it owes
        // the same announcement every setter in `solver::config` makes.
        self.settings_changed();
        self.theory_processed_up_to = 0;
        self.has_false_assertion = false;
        self.has_bv_arith_ops = false;
        self.polarities.clear();
        self.bv_terms.clear();
        self.arith_terms.clear();
        self.dt_var_constructors.clear();
        self.arith_parse_cache.clear();
        self.tracked_compound_terms.clear();
        self.bool_uf_arg_terms.clear();
        self.numeric_uf_arg_terms.clear();
        self.numeric_purify_aliases.clear();
        self.encoded_terms.clear();
        self.fp_constraint_cache.clear();
        self.encode_depth_exceeded = false;
        self.has_array_ops = false;
        self.array_axiom_instances.clear();
        self.array_axioms_incomplete = false;
        self.arith_defined_terms.clear();
        self.numeric_trichotomy_atoms.clear();
        // The assertions that entailed these constants are gone; a survivor
        // would license a finite-range expansion the new formula never asserts.
        self.entailed_int_consts.clear();
        self.entailed_int_consts_upto = 0;
        self.case_split_terms.clear();
        self.case_split_rounds = 0;
        self.case_split_skipped_targets = false;
        // `self.sat.reset()` above discarded every clause, the model-blocking
        // ones included, so the search is unrestricted again.
        self.model_blocking_active = 0;
        self.lookup_index_terms.clear();
        // The clauses those triangles stand for went with the SAT core the
        // reset discarded, and the classes describe a formula that no longer
        // exists.
        self.eq_transitivity_triangles.clear();
        self.equality_skeleton_classes.clear();
        // `branch_priority` is a live handle the installed SAT-engine
        // heuristic (if any) shares with this `Solver`; clearing its
        // contents (not replacing the `Arc`) drops stale hints from the
        // formula `reset` just discarded without disturbing that wiring.
        if let Ok(mut queue) = self.branch_priority.lock() {
            queue.clear();
        }
        self.debug_check_invariants("after reset");
    }

    /// Get solver statistics
    #[must_use]
    pub fn stats(&self) -> &oxiz_sat::SolverStats {
        self.sat.stats()
    }
}

#[cfg(test)]
mod scope_rebase_tests;
#[cfg(test)]
mod tests;
