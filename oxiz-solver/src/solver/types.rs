//! Types and data structures for the SMT solver

#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{CheckedEuclid, Zero};
use oxiz_core::ast::{RoundingMode, TermId, TermKind, TermManager};
use oxiz_sat::{Lit, RestartStrategy, Var};
use smallvec::SmallVec;

/// Proof step for resolution-based proofs
#[derive(Debug, Clone)]
pub enum ProofStep {
    /// Input clause (from the original formula)
    Input {
        /// Clause index
        index: u32,
        /// The clause (as a disjunction of literals)
        clause: Vec<Lit>,
    },
    /// Resolution step
    Resolution {
        /// Index of this proof step
        index: u32,
        /// Left parent clause index
        left: u32,
        /// Right parent clause index
        right: u32,
        /// Pivot variable (the variable resolved on)
        pivot: Var,
        /// Resulting clause
        clause: Vec<Lit>,
    },
    /// Theory lemma (from a theory solver)
    TheoryLemma {
        /// Index of this proof step
        index: u32,
        /// The theory that produced this lemma
        theory: String,
        /// The lemma clause
        clause: Vec<Lit>,
        /// Explanation terms
        explanation: Vec<TermId>,
    },
}

/// A proof of unsatisfiability
#[derive(Debug, Clone)]
pub struct Proof {
    /// Sequence of proof steps leading to the empty clause
    steps: Vec<ProofStep>,
    /// Index of the final empty clause (proving unsat)
    empty_clause_index: Option<u32>,
}

impl Proof {
    /// Create a new empty proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            empty_clause_index: None,
        }
    }

    /// Add a proof step
    pub fn add_step(&mut self, step: ProofStep) {
        self.steps.push(step);
    }

    /// Set the index of the empty clause (final step proving unsat)
    pub fn set_empty_clause(&mut self, index: u32) {
        self.empty_clause_index = Some(index);
    }

    /// Check if the proof is complete (has an empty clause)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.empty_clause_index.is_some()
    }

    /// Get the number of proof steps
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Iterate over all proof steps
    pub fn steps(&self) -> impl Iterator<Item = &ProofStep> {
        self.steps.iter()
    }

    /// Format the proof as a string (for debugging or output)
    #[must_use]
    pub fn format(&self) -> String {
        let mut result = String::from("(proof\n");
        for step in &self.steps {
            match step {
                ProofStep::Input { index, clause } => {
                    result.push_str(&format!("  (input {} {:?})\n", index, clause));
                }
                ProofStep::Resolution {
                    index,
                    left,
                    right,
                    pivot,
                    clause,
                } => {
                    result.push_str(&format!(
                        "  (resolution {} {} {} {:?} {:?})\n",
                        index, left, right, pivot, clause
                    ));
                }
                ProofStep::TheoryLemma {
                    index,
                    theory,
                    clause,
                    ..
                } => {
                    result.push_str(&format!(
                        "  (theory-lemma {} {} {:?})\n",
                        index, theory, clause
                    ));
                }
            }
        }
        if let Some(idx) = self.empty_clause_index {
            result.push_str(&format!("  (empty-clause {})\n", idx));
        }
        result.push_str(")\n");
        result
    }
}

impl Default for Proof {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a theory constraint associated with a boolean variable
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum Constraint {
    /// Equality constraint: lhs = rhs
    Eq(TermId, TermId),
    /// Disequality constraint: lhs != rhs (negation of equality)
    Diseq(TermId, TermId),
    /// Less-than constraint: lhs < rhs
    Lt(TermId, TermId),
    /// Less-than-or-equal constraint: lhs <= rhs
    Le(TermId, TermId),
    /// Greater-than constraint: lhs > rhs
    Gt(TermId, TermId),
    /// Greater-than-or-equal constraint: lhs >= rhs
    Ge(TermId, TermId),
    /// Boolean-valued uninterpreted function application.
    /// When the SAT solver assigns this variable true/false, we must inform
    /// the EUF solver so that congruence closure can detect conflicts
    /// (e.g., `t(m) = true` and `t(co) = false` but `m = co`).
    BoolApp(TermId),
}

/// Type of arithmetic constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArithConstraintType {
    /// Less than (<)
    Lt,
    /// Less than or equal (<=)
    Le,
    /// Greater than (>)
    Gt,
    /// Greater than or equal (>=)
    Ge,
}

/// Parsed arithmetic constraint with extracted linear expression
/// Represents: sum of (term, coefficient) <= constant OR < constant (if strict)
#[derive(Debug, Clone)]
pub(crate) struct ParsedArithConstraint {
    /// Linear terms: (variable_term, coefficient)
    pub(crate) terms: SmallVec<[(TermId, Rational64); 4]>,
    /// Constant bound (RHS)
    pub(crate) constant: Rational64,
    /// Type of constraint
    pub(crate) constraint_type: ArithConstraintType,
    /// The original term (for conflict explanation)
    pub(crate) reason_term: TermId,
}

/// Polarity of a term in the formula
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Polarity {
    /// Term appears only positively
    Positive,
    /// Term appears only negatively
    Negative,
    /// Term appears in both polarities
    Both,
}

/// Result of SMT solving
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout, incomplete, etc.)
    Unknown,
}

/// Theory checking mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TheoryMode {
    /// Eager theory checking (check on every assignment)
    Eager,
    /// Lazy theory checking (check only on complete assignments)
    Lazy,
}

/// Solver configuration
///
/// Every field is a scalar or a field-less enum, so this is cheap to clone and
/// cheap to compare.  `PartialEq` is part of the contract: the solver's
/// repeated-`check` cache compares two of these to decide whether a previous
/// verdict still answers the caller's question, and a field added here without
/// an equality of its own would silently widen what counts as "the same query".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverConfig {
    /// Timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,
    /// Enable parallel solving
    pub parallel: bool,
    /// Number of threads for parallel solving
    pub num_threads: usize,
    /// Enable proof generation
    pub proof: bool,
    /// Enable model generation
    pub model: bool,
    /// Theory checking mode
    pub theory_mode: TheoryMode,
    /// Enable preprocessing/simplification
    pub simplify: bool,
    /// Maximum number of conflicts before giving up (0 = unlimited)
    pub max_conflicts: u64,
    /// Maximum number of decisions before giving up (0 = unlimited)
    pub max_decisions: u64,
    /// Restart strategy for SAT solver
    pub restart_strategy: RestartStrategy,
    /// Enable clause minimization (recursive minimization of learned clauses)
    pub enable_clause_minimization: bool,
    /// Enable learned clause subsumption
    pub enable_clause_subsumption: bool,
    /// Enable variable elimination during preprocessing
    pub enable_variable_elimination: bool,
    /// Variable elimination limit (max clauses to produce)
    pub variable_elimination_limit: usize,
    /// Enable SatELite-style bounded variable elimination inside the SAT core.
    ///
    /// Distinct from [`SolverConfig::enable_variable_elimination`], which is a
    /// coarser "do preprocessing at all" knob: this one maps directly onto
    /// `oxiz_sat::SolverConfig::enable_bve` and is the only thing that makes
    /// the SAT engine's BVE pass reachable at all.
    ///
    /// Off in [`SolverConfig::balanced`] and on in [`SolverConfig::thorough`].
    /// Two properties make that split the right one:
    ///
    /// * BVE runs only from `oxiz_sat::Solver::solve`, i.e. only on this
    ///   crate's pure-Boolean paths (`check_sat_only` and the equality
    ///   skeleton). The CDCL(T) path used by `check` goes through
    ///   `oxiz_sat::Solver::solve_with_theory`, which never invokes it — so no
    ///   theory lemma can ever be blocked by an eliminated variable on the
    ///   normal SMT route.
    /// * The SAT engine additionally declines to run BVE whenever an
    ///   incremental `push` is in scope or a proof is being traced, so an
    ///   incremental workload silently keeps the old behavior.
    ///
    /// Left off in `balanced` (the default) all the same, because a caller who
    /// mixes `check_sat_only` with later incremental assertions on the same
    /// solver can still reach the one documented rough edge:
    /// `oxiz_sat::SolverError::EliminatedVariableReintroduction`, which turns
    /// subsequent verdicts into `Unknown` rather than risking a wrong answer.
    /// `thorough` opts into that trade for the extra preprocessing power.
    pub enable_bve: bool,
    /// Enable blocked clause elimination during preprocessing
    pub enable_blocked_clause_elimination: bool,
    /// Enable symmetry breaking predicates
    pub enable_symmetry_breaking: bool,
    /// Enable inprocessing (periodic preprocessing during search)
    pub enable_inprocessing: bool,
    /// Inprocessing interval (number of conflicts between inprocessing)
    pub inprocessing_interval: u64,
    /// Run the model-repair and grammar-reduction searches on nonlinear
    /// problems the cell-decomposition core leaves undecided (see
    /// `oxiz_theories::nl_repair_search` and
    /// `oxiz_theories::nl_ground_reduce`).
    ///
    /// These searches can only ever turn an `unknown` into a `sat`: they have
    /// no way to derive `unsat`, and every `sat` they produce carries a witness
    /// that is re-checked against the original assertions in exact arithmetic
    /// before the verdict leaves the solver. So the flag is not a soundness
    /// switch — it is a *budget* switch, for a caller who would rather have a
    /// fast `unknown` than spend the search budget.
    ///
    /// Default: `true`.
    pub nonlinear_model_search: bool,
    /// Run the NIA-over-LP relaxation engine
    /// (`oxiz_theories::arithmetic::nla`) on QF_NIA problems the
    /// cell-decomposition core leaves undecided.
    ///
    /// Unlike [`Self::nonlinear_model_search`] this engine *can* derive
    /// `unsat`, so the two halves of the flag are not symmetric — but it is
    /// still a budget switch rather than a soundness switch, because both of
    /// its verdicts are backed rather than guessed:
    ///
    /// * `unsat` is proof-backed. The engine produces it only from an LP
    ///   infeasibility closure over a constraint set every member of which is a
    ///   consequence over `Z` of the input, with case splits that are
    ///   exhaustive over `Z`. Its linearisation may *drop* a conjunct it cannot
    ///   translate, which only weakens the problem — and an unsatisfiable
    ///   weakening still refutes the original.
    /// * `sat` is advisory by that module's own contract, and this solver does
    ///   not take it on trust: the witness must pass `adopt_nl_witness`, which
    ///   re-evaluates it with `oxiz_theories::nl_eval::holds_under` against the
    ///   untouched assertions in exact `BigRational` arithmetic. A witness that
    ///   fails yields no verdict at all.
    ///
    /// The engine is dispatched *after* the cell-decomposition core, so no
    /// verdict that core reaches can move; what the flag converts is goals that
    /// were previously answered `unknown`. Turning it off restores the earlier
    /// *verdicts* at the price of those answers.
    ///
    /// It does not restore the earlier *models* in every case. The engine also
    /// sits above the two searches [`Self::nonlinear_model_search`] gates, so a
    /// goal that both can solve is now answered by the engine and reports the
    /// engine's witness instead. Every such witness is re-verified against the
    /// assertions, so both are correct — but on a goal with several solutions
    /// they need not be the same one, and a caller pinning an exact
    /// `(get-value ...)` result for a multi-solution QF_NIA goal may see it
    /// change.
    ///
    /// Gated on `std` rather than on the `nlsat` feature, so it is present in
    /// both builds.
    ///
    /// Default: `true`.
    pub nonlinear_relaxation_engine: bool,
    /// Cap on the number of ground instances a single bounded-integer
    /// quantifier may be expanded into at assert time (`0` disables the
    /// expansion entirely).
    ///
    /// A quantifier whose own guard pins every bound Int variable to a
    /// concrete finite interval is rewritten into the *logically equivalent*
    /// finite conjunction (`forall`) / disjunction (`exists`) over that
    /// interval, so the ground solver decides it directly instead of MBQI
    /// having to certify it (see the `solver::encode::finite_expand` module).
    /// The product of the per-variable interval widths must not exceed this
    /// cap; a wider box falls through to MBQI unchanged.
    ///
    /// Default: 64 (`DEFAULT_FINITE_EXPANSION_BUDGET`).
    pub finite_expansion_budget: usize,
    /// Install the branch-priority heuristic (see
    /// `crate::solver::branch_priority`) that decides a flattened
    /// lookup-table index's key equalities before falling back to
    /// VSIDS/LRB/CHB, once `flatten_lookup_spines` has populated it.
    ///
    /// Only takes effect through [`Solver::with_config`] — `oxiz-sat`'s
    /// external-branching slot is set once at construction and cannot be
    /// retargeted by a later [`Solver::set_config`] call (see
    /// `branch_priority`'s module doc).
    ///
    /// Default: `false`. Turning this on makes `oxiz-sat`'s
    /// `pick_branch_var` build its full unassigned-variable candidate list
    /// on *every* decision for the lifetime of this `Solver` — the price of
    /// using its pre-existing external-branching hook at all, independent of
    /// whether any lookup table ever actually appears — so it is not safe to
    /// default on for formulas in general; enable it for workloads that are
    /// specifically table-index-heavy.
    ///
    /// [`Solver::with_config`]: crate::solver::Solver::with_config
    /// [`Solver::set_config`]: crate::solver::Solver::set_config
    pub enable_domain_first_branching: bool,
    /// Re-solve after excluding a candidate model the soundness gate refuted,
    /// instead of conceding `unknown` on the first unlucky candidate (see
    /// `crate::solver::model_blocking`).
    ///
    /// Like [`Self::nonlinear_model_search`] this is a *budget* switch, not a
    /// soundness switch: blocking can only ever turn an `unknown` into a `sat`,
    /// every `sat` it produces is a surviving assignment re-checked by the same
    /// gate, and while any blocking clause is live an `unsat` from the SAT core
    /// is surfaced as `unknown` rather than trusted. Turning it off buys a fast
    /// `unknown` in exchange for the models it would have found.
    ///
    /// Note that it gates only the blocking loop. The ordering fix that goes
    /// with it — running the case-split and array-axiom repairs *before* the
    /// refutation gate, so a repairable candidate is repaired rather than
    /// discarded — is a bug fix and applies regardless. Turning the flag off
    /// therefore does not restore the pre-issue-#40 solver exactly: because the
    /// gate now sits below the array-axiom loop, that loop's round-budget exit
    /// also drops the candidate model it was holding, which it did not have to
    /// do while the gate ran first.
    ///
    /// Default: `true`.
    pub enable_model_blocking: bool,
    /// How many refuted candidate models one context scope may exclude before
    /// the solver concedes `unknown` (`0` disables blocking as surely as
    /// [`Self::enable_model_blocking`] does).
    ///
    /// Default: 64 (`MAX_MODEL_BLOCKING_ROUNDS`).
    pub max_model_blocking_rounds: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl SolverConfig {
    /// Create a configuration optimized for speed (minimal preprocessing)
    /// Best for easy problems or when quick results are needed
    #[must_use]
    pub fn fast() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 4,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Eager,
            simplify: true, // Keep basic simplification
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Geometric, // Faster than Glucose
            enable_clause_minimization: true,             // Keep this, it's fast
            enable_clause_subsumption: false,             // Skip for speed
            enable_variable_elimination: false,           // Skip preprocessing
            variable_elimination_limit: 0,
            enable_bve: false,                        // Skip preprocessing
            enable_blocked_clause_elimination: false, // Skip preprocessing
            enable_symmetry_breaking: false,
            enable_inprocessing: false, // No inprocessing for speed
            inprocessing_interval: 0,
            finite_expansion_budget:
                crate::solver::encode::finite_expand::DEFAULT_FINITE_EXPANSION_BUDGET,
            nonlinear_model_search: true,
            nonlinear_relaxation_engine: true,
            enable_domain_first_branching: false,
            enable_model_blocking: true,
            max_model_blocking_rounds: crate::solver::model_blocking::MAX_MODEL_BLOCKING_ROUNDS,
        }
    }

    /// Create a balanced configuration (default)
    /// Good balance between preprocessing and solving speed
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 4,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Eager,
            simplify: true,
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Glucose, // Adaptive restarts
            enable_clause_minimization: true,
            enable_clause_subsumption: true,
            enable_variable_elimination: true,
            variable_elimination_limit: 1000, // Conservative limit
            // Off: see the field doc -- BVE is sound here, but a caller that
            // mixes `check_sat_only` with later incremental assertions can
            // reach `EliminatedVariableReintroduction`, and the default
            // preset should not surprise such a caller.
            enable_bve: false,
            enable_blocked_clause_elimination: true,
            enable_symmetry_breaking: false, // Still expensive
            enable_inprocessing: true,
            inprocessing_interval: 10000,
            finite_expansion_budget:
                crate::solver::encode::finite_expand::DEFAULT_FINITE_EXPANSION_BUDGET,
            nonlinear_model_search: true,
            nonlinear_relaxation_engine: true,
            enable_domain_first_branching: false,
            enable_model_blocking: true,
            max_model_blocking_rounds: crate::solver::model_blocking::MAX_MODEL_BLOCKING_ROUNDS,
        }
    }

    /// Create a configuration optimized for hard problems
    /// Uses aggressive preprocessing and symmetry breaking
    #[must_use]
    pub fn thorough() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 4,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Eager,
            simplify: true,
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Glucose,
            enable_clause_minimization: true,
            enable_clause_subsumption: true,
            enable_variable_elimination: true,
            variable_elimination_limit: 5000, // More aggressive
            // On: this preset exists to trade solve-shape surprises for
            // preprocessing power on hard instances (it is the one that also
            // turns symmetry breaking on).
            enable_bve: true,
            enable_blocked_clause_elimination: true,
            enable_symmetry_breaking: true, // Enable for hard problems
            enable_inprocessing: true,
            inprocessing_interval: 5000, // More frequent inprocessing
            finite_expansion_budget:
                crate::solver::encode::finite_expand::DEFAULT_FINITE_EXPANSION_BUDGET,
            nonlinear_model_search: true,
            nonlinear_relaxation_engine: true,
            enable_domain_first_branching: false,
            enable_model_blocking: true,
            max_model_blocking_rounds: crate::solver::model_blocking::MAX_MODEL_BLOCKING_ROUNDS,
        }
    }

    /// Create a minimal configuration (almost all features disabled)
    /// Useful for debugging or when you want full control
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 1,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Lazy, // Lazy for minimal overhead
            simplify: false,
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Geometric,
            enable_clause_minimization: false,
            enable_clause_subsumption: false,
            enable_variable_elimination: false,
            variable_elimination_limit: 0,
            enable_bve: false,
            enable_blocked_clause_elimination: false,
            enable_symmetry_breaking: false,
            enable_inprocessing: false,
            inprocessing_interval: 0,
            // `minimal()` disables every optional rewrite; the caller asked for
            // full control, so quantifiers keep their plain MBQI path and the
            // nonlinear searches do not spend their budget either.
            finite_expansion_budget: 0,
            nonlinear_model_search: false,
            // Off for the same reason, and mirroring `nonlinear_model_search`
            // above: `minimal()` spends no optional search budget on the
            // caller's behalf. Note this one costs `unsat` answers as well as
            // `sat` ones — the relaxation engine is the only procedure on the
            // QF_NIA path below the cell decomposition that can refute — so a
            // `minimal()` caller trades away more completeness here than the
            // searches alone would.
            nonlinear_relaxation_engine: false,
            // Same reasoning as the two lines above: `minimal()` hands the
            // caller full control, so the extra re-solves the blocking loop
            // spends are not taken on their behalf. A refuted candidate is
            // conceded as `unknown` exactly as it was before issue #40.
            enable_model_blocking: false,
            max_model_blocking_rounds: 0,
            enable_domain_first_branching: false,
        }
    }

    /// Enable proof generation
    #[must_use]
    pub fn with_proof(mut self) -> Self {
        self.proof = true;
        self
    }

    /// Set timeout in milliseconds
    #[must_use]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set maximum number of conflicts
    #[must_use]
    pub fn with_max_conflicts(mut self, max_conflicts: u64) -> Self {
        self.max_conflicts = max_conflicts;
        self
    }

    /// Set maximum number of decisions
    #[must_use]
    pub fn with_max_decisions(mut self, max_decisions: u64) -> Self {
        self.max_decisions = max_decisions;
        self
    }

    /// Enable parallel solving
    #[must_use]
    pub fn with_parallel(mut self, num_threads: usize) -> Self {
        self.parallel = true;
        self.num_threads = num_threads;
        self
    }

    /// Set restart strategy
    #[must_use]
    pub fn with_restart_strategy(mut self, strategy: RestartStrategy) -> Self {
        self.restart_strategy = strategy;
        self
    }

    /// Set theory mode
    #[must_use]
    pub fn with_theory_mode(mut self, mode: TheoryMode) -> Self {
        self.theory_mode = mode;
        self
    }
}

/// Solver statistics
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Number of decisions made
    pub decisions: u64,
    /// Number of conflicts encountered
    pub conflicts: u64,
    /// Number of propagations performed
    pub propagations: u64,
    /// Number of restarts performed
    pub restarts: u64,
    /// Number of learned clauses
    pub learned_clauses: u64,
    /// Number of theory propagations
    pub theory_propagations: u64,
    /// Number of theory conflicts
    pub theory_conflicts: u64,
    /// Number of model-blocking clauses added: one per candidate model the
    /// soundness gate refuted and `crate::solver::model_blocking` excluded so
    /// the search could look for another (issue #40).
    ///
    /// A nonzero count means the last verdict was reached under a *restricted*
    /// search: a `sat` is a genuine model all the same, and an `unsat` was
    /// downgraded to `unknown` rather than reported.
    pub model_blocking_clauses: u64,
}

impl Statistics {
    /// Create new statistics with all counters set to zero
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// A model (assignment to variables)
#[derive(Debug, Clone)]
pub struct Model {
    /// Variable assignments
    assignments: FxHashMap<TermId, TermId>,
}

impl Model {
    /// Create a new empty model
    #[must_use]
    pub fn new() -> Self {
        Self {
            assignments: FxHashMap::default(),
        }
    }

    /// Get the value of a term in the model
    #[must_use]
    pub fn get(&self, term: TermId) -> Option<TermId> {
        self.assignments.get(&term).copied()
    }

    /// Set a value in the model
    pub fn set(&mut self, term: TermId, value: TermId) {
        self.assignments.insert(term, value);
    }

    /// Minimize the model by removing redundant assignments
    /// Returns a new minimized model containing only essential assignments
    pub fn minimize(&self, essential_vars: &[TermId]) -> Model {
        let mut minimized = Model::new();

        // Only keep assignments for essential variables
        for &var in essential_vars {
            if let Some(&value) = self.assignments.get(&var) {
                minimized.set(var, value);
            }
        }

        minimized
    }

    /// Get the number of assignments in the model
    #[must_use]
    pub fn size(&self) -> usize {
        self.assignments.len()
    }

    /// Get the assignments map (for MBQI integration)
    #[must_use]
    pub fn assignments(&self) -> &FxHashMap<TermId, TermId> {
        &self.assignments
    }

    /// Evaluate a term in this model.
    /// Returns the simplified/evaluated term.
    ///
    /// Runs on an explicit heap-allocated frame stack, so nesting depth is
    /// bounded by memory rather than by the native call stack — this is a
    /// public entry point (the `(get-value ...)` path) and callers control
    /// the depth of the terms they hand in.  Shared sub-terms of the
    /// hash-consed DAG are evaluated once per call through a per-call memo
    /// keyed on `TermId` (no binders are descended into, so the memo needs no
    /// binding context); cross-call caching remains `ModelCache`'s job.
    ///
    /// Short-circuit behaviour matches the recursive original exactly:
    /// `and` stops at the first `false` operand, `or` at the first `true`,
    /// and `ite` evaluates only the branch a constant condition selects,
    /// while `=>`, `=`, `-` and the n-ary arithmetic operators evaluate all
    /// of their operands.
    pub fn eval(&self, term: TermId, manager: &mut TermManager) -> TermId {
        let mut memo: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut frames: Vec<EvalFrame> = Vec::new();
        let mut current = term;

        'open: loop {
            // Open `current`, descending through operators until some term
            // produces a value.
            let mut value = loop {
                // Direct model assignment — checked before anything else,
                // exactly as the recursive version did.
                if let Some(val) = self.get(current) {
                    break val;
                }
                if let Some(&val) = memo.get(&current) {
                    break val;
                }
                let Some(t) = manager.get(current).cloned() else {
                    break current;
                };
                match t.kind {
                    // Constants evaluate to themselves.
                    TermKind::True
                    | TermKind::False
                    | TermKind::IntConst(_)
                    | TermKind::RealConst(_)
                    | TermKind::BitVecConst { .. } => break current,

                    // Variables: the model was already consulted above, so an
                    // unassigned variable evaluates to itself.
                    TermKind::Var(_) => break current,

                    TermKind::Not(arg) => {
                        frames.push(EvalFrame::Not { term: current });
                        current = arg;
                    }
                    TermKind::And(args) => match args.first() {
                        Some(&first) => {
                            frames.push(EvalFrame::Connective {
                                term: current,
                                conjunction: true,
                                args,
                                next: 1,
                                acc: Vec::new(),
                            });
                            current = first;
                        }
                        None => break manager.mk_true(),
                    },
                    TermKind::Or(args) => match args.first() {
                        Some(&first) => {
                            frames.push(EvalFrame::Connective {
                                term: current,
                                conjunction: false,
                                args,
                                next: 1,
                                acc: Vec::new(),
                            });
                            current = first;
                        }
                        None => break manager.mk_false(),
                    },
                    TermKind::Implies(lhs, rhs) => {
                        frames.push(EvalFrame::ImpliesLhs { term: current, rhs });
                        current = lhs;
                    }
                    TermKind::Ite(cond, then_br, else_br) => {
                        frames.push(EvalFrame::IteCond {
                            term: current,
                            then_br,
                            else_br,
                        });
                        current = cond;
                    }
                    TermKind::Eq(lhs, rhs) => {
                        frames.push(EvalFrame::EqLhs { term: current, rhs });
                        current = lhs;
                    }
                    TermKind::Neg(arg) => {
                        frames.push(EvalFrame::Neg { term: current });
                        current = arg;
                    }
                    TermKind::Add(args) => match args.first() {
                        Some(&first) => {
                            frames.push(EvalFrame::Nary {
                                term: current,
                                product: false,
                                args,
                                next: 1,
                                acc: Vec::new(),
                            });
                            current = first;
                        }
                        None => break manager.mk_add(Vec::<TermId>::new()),
                    },
                    TermKind::Sub(lhs, rhs) => {
                        frames.push(EvalFrame::SubLhs { term: current, rhs });
                        current = lhs;
                    }
                    TermKind::Mul(args) => match args.first() {
                        Some(&first) => {
                            frames.push(EvalFrame::Nary {
                                term: current,
                                product: true,
                                args,
                                next: 1,
                                acc: Vec::new(),
                            });
                            current = first;
                        }
                        None => break manager.mk_mul(Vec::<TermId>::new()),
                    },
                    TermKind::Div(lhs, rhs) => {
                        frames.push(EvalFrame::DivModLhs {
                            term: current,
                            rhs,
                            is_div: true,
                        });
                        current = lhs;
                    }
                    TermKind::Mod(lhs, rhs) => {
                        frames.push(EvalFrame::DivModLhs {
                            term: current,
                            rhs,
                            is_div: false,
                        });
                        current = lhs;
                    }

                    // For other operations, just return the term (the model
                    // was already consulted above).
                    _ => break current,
                }
            };

            // Fold the finished value into the pending frames; a frame that
            // still needs another operand re-enters the open loop.
            loop {
                let Some(frame) = frames.pop() else {
                    return value;
                };
                match frame {
                    EvalFrame::Not { term } => {
                        let arg_is_true = term_kind_is_true(value, manager);
                        let arg_is_false = term_kind_is_false(value, manager);
                        let v = if arg_is_true {
                            manager.mk_false()
                        } else if arg_is_false {
                            manager.mk_true()
                        } else {
                            manager.mk_not(value)
                        };
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::Connective {
                        term,
                        conjunction,
                        args,
                        next,
                        mut acc,
                    } => {
                        let is_true = term_kind_is_true(value, manager);
                        let is_false = term_kind_is_false(value, manager);
                        if conjunction && is_false {
                            // `and` short-circuit: remaining operands are not
                            // evaluated, matching the recursive version.
                            let v = manager.mk_false();
                            memo.insert(term, v);
                            value = v;
                            continue;
                        }
                        if !conjunction && is_true {
                            let v = manager.mk_true();
                            memo.insert(term, v);
                            value = v;
                            continue;
                        }
                        // `true` operands of `and` and `false` operands of
                        // `or` are dropped; everything else is kept.
                        let is_neutral = if conjunction { is_true } else { is_false };
                        if !is_neutral {
                            acc.push(value);
                        }
                        if let Some(&child) = args.get(next) {
                            frames.push(EvalFrame::Connective {
                                term,
                                conjunction,
                                args,
                                next: next + 1,
                                acc,
                            });
                            current = child;
                            continue 'open;
                        }
                        let v = match (acc.len(), conjunction) {
                            (0, true) => manager.mk_true(),
                            (0, false) => manager.mk_false(),
                            (1, _) => acc[0],
                            (_, true) => manager.mk_and(acc),
                            (_, false) => manager.mk_or(acc),
                        };
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::ImpliesLhs { term, rhs } => {
                        frames.push(EvalFrame::ImpliesRhs {
                            term,
                            lhs_val: value,
                        });
                        current = rhs;
                        continue 'open;
                    }
                    EvalFrame::ImpliesRhs { term, lhs_val } => {
                        let rhs_val = value;
                        let v = if term_kind_is_false(lhs_val, manager) {
                            manager.mk_true()
                        } else if term_kind_is_true(lhs_val, manager) {
                            rhs_val
                        } else if term_kind_is_true(rhs_val, manager) {
                            manager.mk_true()
                        } else {
                            manager.mk_implies(lhs_val, rhs_val)
                        };
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::IteCond {
                        term,
                        then_br,
                        else_br,
                    } => {
                        if term_kind_is_true(value, manager) {
                            frames.push(EvalFrame::Forward { term });
                            current = then_br;
                            continue 'open;
                        }
                        if term_kind_is_false(value, manager) {
                            frames.push(EvalFrame::Forward { term });
                            current = else_br;
                            continue 'open;
                        }
                        frames.push(EvalFrame::IteThen {
                            term,
                            cond_val: value,
                            else_br,
                        });
                        current = then_br;
                        continue 'open;
                    }
                    EvalFrame::IteThen {
                        term,
                        cond_val,
                        else_br,
                    } => {
                        frames.push(EvalFrame::IteElse {
                            term,
                            cond_val,
                            then_val: value,
                        });
                        current = else_br;
                        continue 'open;
                    }
                    EvalFrame::IteElse {
                        term,
                        cond_val,
                        then_val,
                    } => {
                        let v = manager.mk_ite(cond_val, then_val, value);
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::Forward { term } => {
                        // The selected branch's value is the `ite`'s value.
                        memo.insert(term, value);
                    }
                    EvalFrame::EqLhs { term, rhs } => {
                        frames.push(EvalFrame::EqRhs {
                            term,
                            lhs_val: value,
                        });
                        current = rhs;
                        continue 'open;
                    }
                    EvalFrame::EqRhs { term, lhs_val } => {
                        let rhs_val = value;
                        // Simplify boolean equalities with constants:
                        // x = true  => x
                        // x = false => NOT x
                        // true = x  => x
                        // false = x => NOT x
                        let lhs_is_bool = manager
                            .get(lhs_val)
                            .is_some_and(|t| t.sort == manager.sorts.bool_sort);
                        let v = if lhs_val == rhs_val {
                            manager.mk_true()
                        } else if lhs_is_bool && term_kind_is_true(rhs_val, manager) {
                            lhs_val
                        } else if lhs_is_bool && term_kind_is_false(rhs_val, manager) {
                            manager.mk_not(lhs_val)
                        } else if lhs_is_bool && term_kind_is_true(lhs_val, manager) {
                            rhs_val
                        } else if lhs_is_bool && term_kind_is_false(lhs_val, manager) {
                            manager.mk_not(rhs_val)
                        } else {
                            manager.mk_eq(lhs_val, rhs_val)
                        };
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::Neg { term } => {
                        // Arithmetic constant folding; the residual case is a
                        // *negation* term.  (The recursive version built a
                        // boolean `not` here — a wrong-operator defect that
                        // produced an ill-sorted term for an unassigned
                        // arithmetic operand.)
                        let folded = match manager.get(value).map(|t| t.kind.clone()) {
                            Some(TermKind::IntConst(n)) => Some(manager.mk_int(-n)),
                            Some(TermKind::RealConst(r)) => Some(manager.mk_real(-r)),
                            _ => None,
                        };
                        let v = match folded {
                            Some(v) => v,
                            None => manager.mk_neg(value),
                        };
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::Nary {
                        term,
                        product,
                        args,
                        next,
                        mut acc,
                    } => {
                        acc.push(value);
                        if let Some(&child) = args.get(next) {
                            frames.push(EvalFrame::Nary {
                                term,
                                product,
                                args,
                                next: next + 1,
                                acc,
                            });
                            current = child;
                            continue 'open;
                        }
                        let v = if product {
                            manager.mk_mul(acc)
                        } else {
                            manager.mk_add(acc)
                        };
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::SubLhs { term, rhs } => {
                        frames.push(EvalFrame::SubRhs {
                            term,
                            lhs_val: value,
                        });
                        current = rhs;
                        continue 'open;
                    }
                    EvalFrame::SubRhs { term, lhs_val } => {
                        let v = manager.mk_sub(lhs_val, value);
                        memo.insert(term, v);
                        value = v;
                    }
                    EvalFrame::DivModLhs { term, rhs, is_div } => {
                        frames.push(EvalFrame::DivModRhs {
                            term,
                            lhs_val: value,
                            is_div,
                        });
                        current = rhs;
                        continue 'open;
                    }
                    EvalFrame::DivModRhs {
                        term,
                        lhs_val,
                        is_div,
                    } => {
                        let v = fold_div_mod(lhs_val, value, is_div, manager);
                        memo.insert(term, v);
                        value = v;
                    }
                }
            }
        }
    }
}

/// Whether `term` is interned and its kind is the boolean constant `true`.
fn term_kind_is_true(term: TermId, manager: &TermManager) -> bool {
    manager
        .get(term)
        .is_some_and(|t| matches!(t.kind, TermKind::True))
}

/// Whether `term` is interned and its kind is the boolean constant `false`.
fn term_kind_is_false(term: TermId, manager: &TermManager) -> bool {
    manager
        .get(term)
        .is_some_and(|t| matches!(t.kind, TermKind::False))
}

/// One pending operator of [`Model::eval`]'s explicit evaluation stack.
///
/// Each variant carries the original term it evaluates (the memo key for its
/// finished value) plus per-operator progress; a frame that still needs an
/// operand pushes itself back and re-enters the open loop.
enum EvalFrame {
    /// `not` — one operand.
    Not { term: TermId },
    /// `and` (`conjunction`) or `or`: operands evaluated left to right with
    /// the recursive version's short-circuiting; `acc` holds the operands
    /// kept so far.
    Connective {
        term: TermId,
        conjunction: bool,
        args: SmallVec<[TermId; 4]>,
        next: usize,
        acc: Vec<TermId>,
    },
    /// `=>` — waiting on the antecedent.
    ImpliesLhs { term: TermId, rhs: TermId },
    /// `=>` — waiting on the consequent.
    ImpliesRhs { term: TermId, lhs_val: TermId },
    /// `ite` — waiting on the condition.
    IteCond {
        term: TermId,
        then_br: TermId,
        else_br: TermId,
    },
    /// `ite` with a non-constant condition — waiting on the then-branch.
    IteThen {
        term: TermId,
        cond_val: TermId,
        else_br: TermId,
    },
    /// `ite` with a non-constant condition — waiting on the else-branch.
    IteElse {
        term: TermId,
        cond_val: TermId,
        then_val: TermId,
    },
    /// The child's value *is* this term's value (`ite` on a constant
    /// condition evaluating only the selected branch).
    Forward { term: TermId },
    /// `=` — waiting on the left operand.
    EqLhs { term: TermId, rhs: TermId },
    /// `=` — waiting on the right operand.
    EqRhs { term: TermId, lhs_val: TermId },
    /// Unary arithmetic negation.
    Neg { term: TermId },
    /// n-ary `+` (`product = false`) or `*`; all operands are evaluated.
    Nary {
        term: TermId,
        product: bool,
        args: SmallVec<[TermId; 4]>,
        next: usize,
        acc: Vec<TermId>,
    },
    /// Binary `-` — waiting on the left operand.
    SubLhs { term: TermId, rhs: TermId },
    /// Binary `-` — waiting on the right operand.
    SubRhs { term: TermId, lhs_val: TermId },
    /// `div` (`is_div`) or `mod` — waiting on the dividend.
    DivModLhs {
        term: TermId,
        rhs: TermId,
        is_div: bool,
    },
    /// `div` (`is_div`) or `mod` — waiting on the divisor.
    DivModRhs {
        term: TermId,
        lhs_val: TermId,
        is_div: bool,
    },
}

/// Fold a `div`/`mod` (`is_div` selects which) node from its already-evaluated
/// operands.
///
/// SMT-LIB's `Ints` theory defines `div`/`mod` *Euclidean*-style: for a
/// nonzero divisor `n`, `(div m n)` and `(mod m n)` are the unique `q`, `r`
/// satisfying `m = n·q + r` with `0 ≤ r < |n|` — the remainder is never
/// negative regardless of either operand's sign (`(div 7 (- 2)) = -3`,
/// `(mod 7 (- 2)) = 1`; `(div (- 7) 2) = -4`, `(mod (- 7) 2) = 1`). This is
/// exactly [`num_traits::CheckedEuclid`]'s convention, already the shared
/// vocabulary for this fact across the workspace — see
/// `oxiz_core::rewrite::arith`'s constant folder, `oxiz_core`'s model
/// evaluator, and this crate's own `check_array::eval_int` (which cites the
/// same two others). `checked_div_euclid`/`checked_rem_euclid` return `None`
/// at a zero divisor, which is also the right answer here: division and
/// modulo by zero are left *uninterpreted* by SMT-LIB (any value is
/// admissible, so this evaluator must not invent one), and folding is
/// skipped in favour of rebuilding the (by-then operand-evaluated) node.
///
/// A `Real`-sorted `Div` (SMT-LIB's `/`) is exact rational division instead —
/// selected by the *operand's* sort, since the shared [`TermKind::Div`] node
/// carries both meanings. `Mod` is Int-only in SMT-LIB, so no such split
/// applies to it.
fn fold_div_mod(lhs: TermId, rhs: TermId, is_div: bool, manager: &mut TermManager) -> TermId {
    // `Model::eval`'s own `Add`/`Sub`/`Mul`/`Neg` handling above rebuilds a
    // compound arithmetic operand structurally (e.g. `(* 2 3)` stays
    // `(* 2 3)`, not `6`) rather than numerically folding it — that is
    // `TermManager::simplify`'s job, run by the caller *after* `eval`
    // returns. A divisor written as an expression rather than a bare
    // literal (`(mod x (- (* 2 3) 1))`) would otherwise reach here still
    // unfolded and always take the `rebuild` fallback below, even though it
    // is perfectly constant. Simplifying both operands first — cheap and
    // idempotent for the already-constant case this function exists for —
    // closes that gap without waiting for the caller's later pass.
    let lhs = manager.simplify(lhs);
    let rhs = manager.simplify(rhs);
    match (
        manager.get(lhs).map(|t| t.kind.clone()),
        manager.get(rhs).map(|t| t.kind.clone()),
    ) {
        (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) if !b.is_zero() => {
            let folded = if is_div {
                a.checked_div_euclid(&b)
            } else {
                a.checked_rem_euclid(&b)
            };
            match folded {
                Some(v) => manager.mk_int(v),
                // Only reachable at the one overflowing edge case
                // (`BigInt` never actually overflows, but `CheckedEuclid`'s
                // contract allows `None` generically); rebuild rather than
                // fabricate a value.
                None => rebuild_div_mod(lhs, rhs, is_div, manager),
            }
        }
        (Some(TermKind::RealConst(a)), Some(TermKind::RealConst(b))) if is_div && !b.is_zero() => {
            manager.mk_real(a / b)
        }
        _ => rebuild_div_mod(lhs, rhs, is_div, manager),
    }
}

/// Rebuild a `div`/`mod` node from its (possibly already-evaluated) operands
/// without folding — the divisor is zero, symbolic, or the operands are not
/// both constants of the same evaluable shape.
fn rebuild_div_mod(lhs: TermId, rhs: TermId, is_div: bool, manager: &mut TermManager) -> TermId {
    if is_div {
        manager.mk_div(lhs, rhs)
    } else {
        manager.mk_mod(lhs, rhs)
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    /// Pretty print the model in SMT-LIB2 format
    #[cfg(feature = "std")]
    pub fn pretty_print(&self, manager: &TermManager) -> String {
        if self.assignments.is_empty() {
            return "(model)".to_string();
        }

        let mut lines = vec!["(model".to_string()];
        let printer = oxiz_core::smtlib::Printer::new(manager);

        for (&var, &value) in &self.assignments {
            if let Some(term) = manager.get(var) {
                // Only print top-level variables, not internal encoding variables
                if let TermKind::Var(name) = &term.kind {
                    let sort_str = Self::format_sort(term.sort, manager);
                    let value_str = printer.print_term(value);
                    // Use Debug format for the symbol name
                    let name_str = format!("{:?}", name);
                    lines.push(format!(
                        "  (define-fun {} () {} {})",
                        name_str, sort_str, value_str
                    ));
                }
            }
        }
        lines.push(")".to_string());
        lines.join("\n")
    }

    /// Format a sort ID to its SMT-LIB2 representation
    fn format_sort(sort: oxiz_core::sort::SortId, manager: &TermManager) -> String {
        if sort == manager.sorts.bool_sort {
            "Bool".to_string()
        } else if sort == manager.sorts.int_sort {
            "Int".to_string()
        } else if sort == manager.sorts.real_sort {
            "Real".to_string()
        } else if let Some(s) = manager.sorts.get(sort) {
            if let Some(w) = s.bitvec_width() {
                format!("(_ BitVec {})", w)
            } else {
                "Unknown".to_string()
            }
        } else {
            "Unknown".to_string()
        }
    }
}

/// A named assertion for unsat core tracking
#[derive(Debug, Clone)]
pub struct NamedAssertion {
    /// The assertion term (kept for potential future use in minimization)
    #[allow(dead_code)]
    pub term: TermId,
    /// The name (if any)
    pub name: Option<String>,
    /// Index of this assertion
    pub index: u32,
}

/// An unsat core - a minimal set of assertions that are unsatisfiable
#[derive(Debug, Clone)]
pub struct UnsatCore {
    /// The names of assertions in the core
    pub names: Vec<String>,
    /// The indices of assertions in the core
    pub indices: Vec<u32>,
}

impl UnsatCore {
    /// Create a new empty unsat core
    #[must_use]
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Check if the core is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get the number of assertions in the core
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

impl Default for UnsatCore {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached FP constraint data for a single assertion term.
#[derive(Debug, Clone)]
pub struct FpConstraintData {
    pub additions: Vec<(TermId, TermId, TermId, TermId, RoundingMode)>,
    pub divisions: Vec<(TermId, TermId, TermId, TermId, RoundingMode)>,
    pub multiplications: Vec<(TermId, TermId, TermId, TermId, RoundingMode)>,
    pub comparisons: Vec<(TermId, TermId, bool)>,
    pub equalities: Vec<(TermId, TermId)>,
    pub literals: FxHashMap<TermId, f64>,
    pub rounding_add_results: FxHashMap<(TermId, TermId, RoundingMode), TermId>,
    pub is_zero: FxHashSet<TermId>,
    pub is_positive: FxHashSet<TermId>,
    pub is_negative: FxHashSet<TermId>,
    pub not_nan: FxHashSet<TermId>,
    pub gt_comparisons: Vec<(TermId, TermId)>,
    pub lt_comparisons: Vec<(TermId, TermId)>,
    pub conversions: Vec<(TermId, u32, u32, TermId)>,
    pub real_to_fp_conversions: Vec<(TermId, u32, u32, TermId)>,
    pub subtractions: Vec<(TermId, TermId, TermId)>,
}

impl FpConstraintData {
    #[must_use]
    pub fn new() -> Self {
        Self {
            additions: Vec::new(),
            divisions: Vec::new(),
            multiplications: Vec::new(),
            comparisons: Vec::new(),
            equalities: Vec::new(),
            literals: FxHashMap::default(),
            rounding_add_results: FxHashMap::default(),
            is_zero: FxHashSet::default(),
            is_positive: FxHashSet::default(),
            is_negative: FxHashSet::default(),
            not_nan: FxHashSet::default(),
            gt_comparisons: Vec::new(),
            lt_comparisons: Vec::new(),
            conversions: Vec::new(),
            real_to_fp_conversions: Vec::new(),
            subtractions: Vec::new(),
        }
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.additions.is_empty()
            && self.divisions.is_empty()
            && self.multiplications.is_empty()
            && self.comparisons.is_empty()
            && self.equalities.is_empty()
    }

    pub fn merge(&mut self, other: &FpConstraintData) {
        self.additions.extend_from_slice(&other.additions);
        self.divisions.extend_from_slice(&other.divisions);
        self.multiplications
            .extend_from_slice(&other.multiplications);
        self.comparisons.extend_from_slice(&other.comparisons);
        self.equalities.extend_from_slice(&other.equalities);
        for (&k, &v) in &other.literals {
            self.literals.insert(k, v);
        }
        for (&k, &v) in &other.rounding_add_results {
            self.rounding_add_results.insert(k, v);
        }
        self.is_zero.extend(other.is_zero.iter().copied());
        self.is_positive.extend(other.is_positive.iter().copied());
        self.is_negative.extend(other.is_negative.iter().copied());
        self.not_nan.extend(other.not_nan.iter().copied());
        self.gt_comparisons.extend_from_slice(&other.gt_comparisons);
        self.lt_comparisons.extend_from_slice(&other.lt_comparisons);
        self.conversions.extend_from_slice(&other.conversions);
        self.real_to_fp_conversions
            .extend_from_slice(&other.real_to_fp_conversions);
        self.subtractions.extend_from_slice(&other.subtractions);
    }
}

impl Default for FpConstraintData {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy model evaluation cache.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ModelCache {
    model: Model,
    eval_cache: FxHashMap<TermId, TermId>,
    cache_hits: u64,
    cache_misses: u64,
}

#[allow(dead_code)]
impl ModelCache {
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            eval_cache: FxHashMap::default(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[must_use]
    pub fn get_direct(&self, term: TermId) -> Option<TermId> {
        self.model.get(term)
    }

    pub fn eval_lazy(&mut self, term: TermId, manager: &mut TermManager) -> TermId {
        if let Some(&cached) = self.eval_cache.get(&term) {
            self.cache_hits += 1;
            return cached;
        }
        self.cache_misses += 1;
        let result = self.model.eval(term, manager);
        self.eval_cache.insert(term, result);
        result
    }

    pub fn eval_batch(
        &mut self,
        terms: &[TermId],
        manager: &mut TermManager,
    ) -> SmallVec<[TermId; 8]> {
        terms
            .iter()
            .map(|&t| {
                if let Some(&cached) = self.eval_cache.get(&t) {
                    self.cache_hits += 1;
                    cached
                } else {
                    self.cache_misses += 1;
                    let result = self.model.eval(t, manager);
                    self.eval_cache.insert(t, result);
                    result
                }
            })
            .collect()
    }

    pub fn invalidate(&mut self) {
        self.eval_cache.clear();
    }

    pub fn invalidate_term(&mut self, term: TermId) {
        self.eval_cache.remove(&term);
    }

    #[must_use]
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.cache_hits, self.cache_misses)
    }

    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.eval_cache.len()
    }

    #[must_use]
    pub fn model_size(&self) -> usize {
        self.model.size()
    }

    #[must_use]
    pub fn is_cached(&self, term: TermId) -> bool {
        self.eval_cache.contains_key(&term)
    }

    #[must_use]
    pub fn into_model(self) -> Model {
        self.model
    }
}
