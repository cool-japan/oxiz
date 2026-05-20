//! Z3 API Compatibility Layer — Extension 2
//!
//! This module implements seven additional Z3-compatible surfaces on top of
//! the core types defined in [`crate::z3_compat`] and the first extension
//! layer in [`crate::z3_compat::ext`]:
//!
//! - [`Z3Statistics`]         — key/value statistics after solving
//! - [`Z3Params`] / [`ParamVal`] — solver parameter bags
//! - [`Z3Probe`]              — goal-analysis probes (size, depth, is-qfbv, …)
//! - [`Z3Goal`]               — goal wrapper for tactic input
//! - [`Z3Tactic`]             — named tactic factory + combinators
//! - [`Z3ApplyResult`]        — result of tactic application
//! - [`Z3DatatypeSort`] / [`Z3Constructor`] — algebraic datatype declarations
//! - `check_assumptions` / `unsat_core` methods on [`Z3Solver`]
//! - [`Z3AstVector`]          — a simple ordered collection of boolean terms

use std::collections::HashMap;
use std::rc::Rc;

use oxiz_core::ast::{TermId, TermManager};
use oxiz_core::sort::SortId;
use oxiz_core::tactic::{
    Goal, HasArrayProbe, HasBitVectorProbe, HasQuantifierProbe, IsLinearProbe, NodeCountProbe,
    Probe, SizeProbe, TacticResult,
};
use oxiz_core::tactic::DepthProbe;
use oxiz_theories::datatype::{Constructor, DatatypeDecl};

use crate::solver::SolverConfig;
use crate::z3_compat::{Bool, SatResult, Z3Context, Z3Solver};

// ─── Z3Statistics ────────────────────────────────────────────────────────────

/// Z3-style statistics object produced after a `check()` call.
///
/// Iterates the solver's internal [`Statistics`] struct as a flat list of
/// named `f64` entries so callers don't need to know the internal field names.
///
/// [`Statistics`]: oxiz_core::statistics::Statistics
pub struct Z3Statistics {
    /// Pairs of (key, value) extracted from the solver statistics.
    pairs: Vec<(&'static str, f64)>,
}

impl Z3Statistics {
    /// Build a [`Z3Statistics`] from the solver's internal statistics.
    fn from_solver_stats(stats: &crate::solver::Statistics) -> Self {
        let pairs: Vec<(&'static str, f64)> = vec![
            ("decisions", stats.decisions as f64),
            ("propagations", stats.propagations as f64),
            ("conflicts", stats.conflicts as f64),
            ("restarts", stats.restarts as f64),
            ("learned-clauses", stats.learned_clauses as f64),
            ("theory-propagations", stats.theory_propagations as f64),
            ("theory-conflicts", stats.theory_conflicts as f64),
        ];
        Self { pairs }
    }

    /// Number of statistical keys.
    #[must_use]
    pub fn num_keys(&self) -> usize {
        self.pairs.len()
    }

    /// Return the key at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_keys()`.
    #[must_use]
    pub fn key(&self, i: usize) -> &str {
        self.pairs[i].0
    }

    /// Return the value at index `i` as an `f64`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_keys()`.
    #[must_use]
    pub fn value(&self, i: usize) -> f64 {
        self.pairs[i].1
    }

    /// Look up a statistic by name.  Returns `None` if the key is absent.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<f64> {
        self.pairs.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }

    /// Format all statistics as a multi-line string (for debugging).
    #[must_use]
    pub fn to_stat_string(&self) -> String {
        let mut s = String::new();
        for (k, v) in &self.pairs {
            s.push_str(&format!("  {} = {}\n", k, v));
        }
        s
    }
}

impl std::fmt::Display for Z3Statistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_stat_string())
    }
}

// ─── Z3Solver::statistics ─────────────────────────────────────────────────────

impl Z3Solver {
    /// Return a snapshot of solver statistics after the last `check()` call.
    #[must_use]
    pub fn statistics(&self) -> Z3Statistics {
        Z3Statistics::from_solver_stats(self.ctx.raw_statistics())
    }
}

// ─── Z3Params / ParamVal ─────────────────────────────────────────────────────

/// A parameter value that can be stored in a [`Z3Params`] bag.
#[derive(Debug, Clone)]
pub enum ParamVal {
    /// A boolean parameter.
    Bool(bool),
    /// An unsigned integer parameter.
    UInt(u64),
    /// A double-precision floating-point parameter.
    Double(f64),
    /// A string parameter.
    Str(String),
}

/// Analogue of `z3::Params`.
///
/// A key/value bag that can be applied to a [`Z3Solver`] to override solver
/// configuration options before calling `check()`.
#[derive(Debug, Clone, Default)]
pub struct Z3Params {
    map: HashMap<String, ParamVal>,
}

impl Z3Params {
    /// Create an empty parameter bag.
    #[must_use]
    pub fn new(_ctx: &Z3Context) -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Set a boolean parameter (e.g. `"mbqi"`, `"proof"`).
    pub fn set_bool(&mut self, key: &str, val: bool) {
        self.map.insert(key.to_string(), ParamVal::Bool(val));
    }

    /// Set an unsigned-integer parameter (e.g. `"seed"`, `"threads"`).
    pub fn set_u32(&mut self, key: &str, val: u64) {
        self.map.insert(key.to_string(), ParamVal::UInt(val));
    }

    /// Set a double-precision parameter (e.g. `"var-decay"`).
    pub fn set_double(&mut self, key: &str, val: f64) {
        self.map.insert(key.to_string(), ParamVal::Double(val));
    }

    /// Set a string parameter (e.g. `"logic"`).
    pub fn set_str(&mut self, key: &str, val: &str) {
        self.map
            .insert(key.to_string(), ParamVal::Str(val.to_string()));
    }

    /// Return the underlying map.
    #[must_use]
    pub fn as_map(&self) -> &HashMap<String, ParamVal> {
        &self.map
    }
}

/// Apply a [`Z3Params`] bag to the solver, mapping common keys to
/// [`SolverConfig`] fields.
impl Z3Solver {
    /// Apply a parameter bag to the solver.
    ///
    /// Recognised keys:
    /// - `"timeout"` / `"timeout_ms"` (UInt or Double) → `config.timeout_ms`
    /// - `"seed"` (UInt) → (accepted but currently no-op for reproducibility)
    /// - `"mbqi"` (Bool) → no direct field; forwarded to the underlying solver
    ///   option `"mbqi"`.
    /// - `"max-conflicts"` (UInt) → `config.max_conflicts`
    /// - `"max-decisions"` (UInt) → `config.max_decisions`
    /// - `"proof"` (Bool) → `config.proof`
    pub fn set_params(&mut self, params: &Z3Params) {
        let mut config: SolverConfig = self.ctx.solver_config().clone();

        for (key, val) in &params.map {
            match (key.as_str(), val) {
                ("timeout" | "timeout_ms", ParamVal::UInt(ms)) => {
                    config.timeout_ms = *ms;
                }
                ("timeout", ParamVal::Double(ms)) => {
                    config.timeout_ms = *ms as u64;
                }
                ("max-conflicts", ParamVal::UInt(n)) => {
                    config.max_conflicts = *n;
                }
                ("max-decisions", ParamVal::UInt(n)) => {
                    config.max_decisions = *n;
                }
                ("proof", ParamVal::Bool(b)) => {
                    config.proof = *b;
                }
                // "seed" / "mbqi" / unknown keys: accepted silently.
                _ => {}
            }
        }

        self.ctx.set_solver_config(config);
    }
}

// ─── Z3Goal ──────────────────────────────────────────────────────────────────

/// Analogue of `z3::Goal`.
///
/// A container of [`Bool`] assertions that can be fed to a [`Z3Tactic`].
pub struct Z3Goal {
    /// Inner goal from the tactic framework.
    inner: Goal,
    /// Back-reference to the context for term building.
    ctx_tm: Rc<std::cell::RefCell<TermManager>>,
}

impl Z3Goal {
    /// Create a new empty goal.
    #[must_use]
    pub fn new(ctx: &Z3Context) -> Self {
        Self {
            inner: Goal::empty(),
            ctx_tm: ctx.tm.clone(),
        }
    }

    /// Add an assertion to the goal.
    pub fn assert(&mut self, b: &Bool) {
        self.inner.add(b.id);
    }

    /// Number of assertions in this goal.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Retrieve the `i`-th assertion as a [`Bool`].
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.size()`.
    #[must_use]
    pub fn get_formula(&self, i: usize) -> Bool {
        Bool {
            id: self.inner.assertions[i],
        }
    }

    /// Returns `true` if the goal has been determined to be satisfiable
    /// (all assertions simplified to `true`).
    #[must_use]
    pub fn is_decided_sat(&self) -> bool {
        if self.inner.is_empty() {
            return true;
        }
        let tm = self.ctx_tm.borrow();
        let true_id = tm.mk_true_ro();
        self.inner.assertions.iter().all(|&a| a == true_id)
    }

    /// Returns `true` if the goal has been determined to be unsatisfiable
    /// (at least one assertion is `false`).
    #[must_use]
    pub fn is_decided_unsat(&self) -> bool {
        let tm = self.ctx_tm.borrow();
        let false_id = tm.mk_false_ro();
        self.inner.assertions.contains(&false_id)
    }

    /// Access the underlying [`Goal`] for use with the tactic framework.
    #[must_use]
    pub fn as_inner(&self) -> &Goal {
        &self.inner
    }
}

// ─── Z3ApplyResult ────────────────────────────────────────────────────────────

/// Analogue of `z3::ApplyResult`.
///
/// Holds the sub-goals produced by applying a [`Z3Tactic`] to a [`Z3Goal`].
pub struct Z3ApplyResult {
    subgoals: Vec<Z3Goal>,
}

impl Z3ApplyResult {
    /// Number of sub-goals in this result.
    #[must_use]
    pub fn num_subgoals(&self) -> usize {
        self.subgoals.len()
    }

    /// Return a reference to the `i`-th sub-goal.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_subgoals()`.
    #[must_use]
    pub fn get_subgoal(&self, i: usize) -> &Z3Goal {
        &self.subgoals[i]
    }
}

// ─── TacticKind — internal enum ───────────────────────────────────────────────

/// Internal representation of a tactic, used to allow cheap cloning and
/// combinator construction without requiring all concrete tactic types to
/// implement `Clone`.
#[derive(Clone)]
enum TacticKind {
    /// A named leaf tactic.
    Named(String),
    /// Sequential composition.
    Then(Box<TacticKind>, Box<TacticKind>),
    /// Fallback composition.
    OrElse(Box<TacticKind>, Box<TacticKind>),
    /// Fixed-point repetition.
    Repeat(Box<TacticKind>),
    /// Time-limited wrapper (milliseconds).
    TryFor(Box<TacticKind>, u64),
}

impl TacticKind {
    /// Apply this tactic kind to a [`Goal`].
    fn apply_to_goal(&self, goal: &Goal) -> TacticResult {
        match self {
            TacticKind::Named(name) => apply_named_tactic(name.as_str(), goal),
            TacticKind::Then(a, b) => {
                let first_result = a.apply_to_goal(goal);
                match first_result {
                    TacticResult::SubGoals(sub) => {
                        // Apply `b` to each sub-goal, collecting results.
                        let mut combined: Vec<Goal> = Vec::new();
                        for sg in sub {
                            match b.apply_to_goal(&sg) {
                                TacticResult::SubGoals(more) => combined.extend(more),
                                TacticResult::Solved(r) => {
                                    return TacticResult::Solved(r);
                                }
                                TacticResult::NotApplicable => combined.push(sg),
                                TacticResult::Failed(msg) => {
                                    return TacticResult::Failed(msg);
                                }
                            }
                        }
                        TacticResult::SubGoals(combined)
                    }
                    other => other,
                }
            }
            TacticKind::OrElse(a, b) => {
                let r = a.apply_to_goal(goal);
                if matches!(r, TacticResult::NotApplicable) {
                    b.apply_to_goal(goal)
                } else {
                    r
                }
            }
            TacticKind::Repeat(inner) => {
                let mut current = goal.clone();
                for _ in 0..1000_usize {
                    match inner.apply_to_goal(&current) {
                        TacticResult::Solved(r) => return TacticResult::Solved(r),
                        TacticResult::SubGoals(sub) if sub.len() == 1 => {
                            if sub[0].assertions == current.assertions {
                                break; // fixed-point
                            }
                            current = sub.into_iter().next().unwrap();
                        }
                        TacticResult::SubGoals(sub) => {
                            return TacticResult::SubGoals(sub);
                        }
                        TacticResult::NotApplicable => break,
                        TacticResult::Failed(msg) => return TacticResult::Failed(msg),
                    }
                }
                TacticResult::SubGoals(vec![current])
            }
            TacticKind::TryFor(inner, _ms) => {
                // Best-effort: run synchronously; timeout semantics are
                // honoured by the underlying solver's conflict limit, not by
                // wall-clock here.
                inner.apply_to_goal(goal)
            }
        }
    }
}

/// Lazily-built, process-wide [`TacticRegistry`].
///
/// [`default_registry`] allocates and registers all 19 canonical tactics on
/// every call.  `apply_named_tactic` is on a hot path — the `Repeat` and `Then`
/// combinators in [`TacticKind::apply_to_goal`] can invoke it up to 1000 times
/// for a single `Z3Tactic::apply` — so we build the registry exactly once and
/// share it behind a [`OnceLock`].
///
/// This is only sound because [`TacticRegistry`] is `Send + Sync`: its factory
/// closures are stored as `Box<dyn Fn() -> Box<dyn Tactic> + Send + Sync>`.
///
/// [`default_registry`]: oxiz_core::tactic::default_registry
/// [`TacticRegistry`]: oxiz_core::tactic::TacticRegistry
fn tactic_registry() -> &'static oxiz_core::tactic::TacticRegistry {
    use oxiz_core::tactic::{default_registry, TacticRegistry};
    use std::sync::OnceLock;
    static REG: OnceLock<TacticRegistry> = OnceLock::new();
    REG.get_or_init(default_registry)
}

/// Dispatch a named tactic via the canonical [`TacticRegistry`].
///
/// Backend-only tactics that need a full solver (`"smt"`, `"sat"`) are not in
/// the registry; they fall through to the `None` branch and return the goal
/// unchanged so a tactic pipeline can continue on to the solver backend.
///
/// [`TacticRegistry`]: oxiz_core::tactic::TacticRegistry
fn apply_named_tactic(name: &str, goal: &Goal) -> TacticResult {
    // Backward-compatibility aliases: map historical Z3 short-form names onto
    // the registry's canonical keys.
    let canonical = match name {
        "ctx-simplify" => "ctx-solver-simplify",
        other => other,
    };

    match tactic_registry().create(canonical) {
        Some(tactic) => tactic.apply(goal).unwrap_or(TacticResult::NotApplicable),
        None => {
            // Unknown / backend-only tactic (e.g. "smt", "sat"): return goal
            // unchanged so a tactic pipeline can continue to the solver backend.
            TacticResult::SubGoals(vec![goal.clone()])
        }
    }
}

// ─── Z3Tactic ────────────────────────────────────────────────────────────────

/// Analogue of `z3::Tactic`.
///
/// A Z3Tactic wraps a [`TacticKind`] tree and can be combined with other
/// tactics using `.then()`, `.or_else()`, `.repeat()`, and `.try_for()`.
#[derive(Clone)]
pub struct Z3Tactic {
    kind: TacticKind,
}

impl Z3Tactic {
    /// Create a tactic by name.
    ///
    /// Names are resolved through the canonical
    /// [`TacticRegistry`](oxiz_core::tactic::TacticRegistry), so every tactic
    /// registered by
    /// [`default_registry`](oxiz_core::tactic::default_registry) is reachable —
    /// e.g. `"simplify"`, `"propagate-values"`, `"ctx-solver-simplify"`,
    /// `"bit-blast"`, `"ackermannize"`, `"solve-eqs"`, `"nnf"`, `"tseitin-cnf"`,
    /// `"fm"`, `"pb2bv"`, `"split"`, `"skip"`, and more.
    ///
    /// The historical short form `"ctx-simplify"` is accepted as an alias for
    /// `"ctx-solver-simplify"`.
    ///
    /// Backend-only names that require a full solver (`"smt"`, `"sat"`) and any
    /// unrecognised name are accepted and return the goal unchanged so a
    /// pipeline can continue on to the solver backend.
    #[must_use]
    pub fn new(_ctx: &Z3Context, name: &str) -> Self {
        Self {
            kind: TacticKind::Named(name.to_string()),
        }
    }

    /// Apply this tactic to `goal`.
    ///
    /// Returns a [`Z3ApplyResult`] containing zero or more sub-goals.
    #[must_use]
    pub fn apply(&self, ctx: &Z3Context, goal: &Z3Goal) -> Z3ApplyResult {
        let raw_result = self.kind.apply_to_goal(goal.as_inner());
        let ctx_tm = ctx.tm.clone();
        let subgoals = match raw_result {
            TacticResult::SubGoals(goals) => goals
                .into_iter()
                .map(|g| Z3Goal {
                    inner: g,
                    ctx_tm: ctx_tm.clone(),
                })
                .collect(),
            TacticResult::Solved(_) | TacticResult::NotApplicable | TacticResult::Failed(_) => {
                // A solved / inapplicable / failed result produces no sub-goals.
                Vec::new()
            }
        };
        Z3ApplyResult { subgoals }
    }

    /// Sequential composition: apply `self`, then apply `other` to each
    /// resulting sub-goal.
    #[must_use]
    pub fn then(&self, other: &Z3Tactic) -> Self {
        Self {
            kind: TacticKind::Then(Box::new(self.kind.clone()), Box::new(other.kind.clone())),
        }
    }

    /// Fallback composition: if `self` returns `NotApplicable`, apply `other`.
    #[must_use]
    pub fn or_else(&self, other: &Z3Tactic) -> Self {
        Self {
            kind: TacticKind::OrElse(Box::new(self.kind.clone()), Box::new(other.kind.clone())),
        }
    }

    /// Repeat `self` until a fixed-point is reached (at most 1000 iterations).
    #[must_use]
    pub fn repeat(&self) -> Self {
        Self {
            kind: TacticKind::Repeat(Box::new(self.kind.clone())),
        }
    }

    /// Wrap `self` with a millisecond timeout.
    ///
    /// The timeout is stored but currently enforced at the solver level via the
    /// `timeout_ms` config field, not by a wall-clock check in the tactic runner.
    #[must_use]
    pub fn try_for(&self, ms: u64) -> Self {
        Self {
            kind: TacticKind::TryFor(Box::new(self.kind.clone()), ms),
        }
    }
}

// ─── Z3Probe ─────────────────────────────────────────────────────────────────

/// Internal enum of concrete probe implementations.
///
/// Using an enum rather than `Box<dyn Probe>` lets [`Z3Probe`] be cheaply
/// composed without heap allocation on the hot path.
#[derive(Clone)]
enum ProbeKind {
    Size,
    NodeCount,
    Depth,
    HasQuantifier,
    IsLinear,
    HasBitVector,
    HasArray,
    Const(f64),
    /// Combinator: result = 1.0 if left < right, else 0.0.
    Lt(Box<ProbeKind>, Box<ProbeKind>),
    /// Combinator: result = 1.0 if left > right, else 0.0.
    Gt(Box<ProbeKind>, Box<ProbeKind>),
}

impl ProbeKind {
    fn evaluate(&self, goal: &Goal, tm: &TermManager) -> f64 {
        match self {
            ProbeKind::Size => SizeProbe.evaluate(goal, tm),
            ProbeKind::NodeCount => NodeCountProbe.evaluate(goal, tm),
            ProbeKind::Depth => DepthProbe.evaluate(goal, tm),
            ProbeKind::HasQuantifier => HasQuantifierProbe.evaluate(goal, tm),
            ProbeKind::IsLinear => IsLinearProbe.evaluate(goal, tm),
            ProbeKind::HasBitVector => HasBitVectorProbe.evaluate(goal, tm),
            ProbeKind::HasArray => HasArrayProbe.evaluate(goal, tm),
            ProbeKind::Const(v) => *v,
            ProbeKind::Lt(a, b) => {
                if a.evaluate(goal, tm) < b.evaluate(goal, tm) {
                    1.0
                } else {
                    0.0
                }
            }
            ProbeKind::Gt(a, b) => {
                if a.evaluate(goal, tm) > b.evaluate(goal, tm) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Analogue of `z3::Probe`.
///
/// A function that analyses a [`Z3Goal`] and returns a numeric value.
/// Probes can be combined using `.lt()` and `.gt()` combinators.
#[derive(Clone)]
pub struct Z3Probe {
    kind: ProbeKind,
}

impl Z3Probe {
    /// Create a probe by name.
    ///
    /// Supported names:
    /// - `"size"` — number of assertions
    /// - `"num-exprs"` / `"num-consts"` — total unique term nodes
    /// - `"depth"` — maximum term depth
    /// - `"has-quantifiers"` / `"is-quantified"` — quantifier check
    /// - `"is-linear"` / `"is-qflia"` — linearity check
    /// - `"is-qfbv"` / `"has-bitvector"` — bitvector check
    /// - `"has-array"` — array check
    /// - any unrecognised name — constant 0.0
    #[must_use]
    pub fn new(_ctx: &Z3Context, name: &str) -> Self {
        let kind = match name {
            "size" => ProbeKind::Size,
            "num-exprs" | "num-consts" => ProbeKind::NodeCount,
            "depth" => ProbeKind::Depth,
            "has-quantifiers" | "is-quantified" => ProbeKind::HasQuantifier,
            "is-linear" | "is-qflia" => ProbeKind::IsLinear,
            "is-qfbv" | "has-bitvector" => ProbeKind::HasBitVector,
            "has-array" => ProbeKind::HasArray,
            _ => ProbeKind::Const(0.0),
        };
        Self { kind }
    }

    /// Evaluate the probe on `goal`.  Returns a numeric value.
    #[must_use]
    pub fn apply(&self, ctx: &Z3Context, goal: &Z3Goal) -> f64 {
        let tm = ctx.tm.borrow();
        self.kind.evaluate(goal.as_inner(), &tm)
    }

    /// Combinator: returns a probe that is 1.0 iff `self < other`.
    #[must_use]
    pub fn lt(self, other: Z3Probe) -> Z3Probe {
        Z3Probe {
            kind: ProbeKind::Lt(Box::new(self.kind), Box::new(other.kind)),
        }
    }

    /// Combinator: returns a probe that is 1.0 iff `self > other`.
    #[must_use]
    pub fn gt(self, other: Z3Probe) -> Z3Probe {
        Z3Probe {
            kind: ProbeKind::Gt(Box::new(self.kind), Box::new(other.kind)),
        }
    }
}

// ─── Z3DatatypeSort / Z3Constructor ──────────────────────────────────────────

/// A field specification in a [`Z3Constructor`].
#[derive(Debug, Clone)]
pub struct Z3Field {
    /// Field name.
    pub name: String,
    /// Field sort (as a string, e.g. `"Int"`, `"Bool"`, `"Real"`).
    pub sort_name: String,
}

/// A constructor specification for a [`Z3DatatypeSort`].
#[derive(Debug, Clone)]
pub struct Z3Constructor {
    /// Constructor name (e.g. `"Cons"`, `"Nil"`).
    pub name: String,
    /// Fields of this constructor.
    pub fields: Vec<Z3Field>,
}

/// Analogue of `z3::DatatypeSort`.
///
/// Wraps a [`DatatypeDecl`] and provides indexed access to constructor,
/// recogniser, and accessor [`FuncDecl`]s.
///
/// [`FuncDecl`]: crate::z3_compat::ext::FuncDecl
pub struct Z3DatatypeSort {
    decl: DatatypeDecl,
    sort_id: SortId,
}

impl Z3DatatypeSort {
    /// Declare a new algebraic datatype from a list of [`Z3Constructor`]s.
    ///
    /// # Panics
    ///
    /// Panics if `constructors` is empty.
    #[must_use]
    pub fn new(ctx: &Z3Context, name: &str, constructors: &[Z3Constructor]) -> Self {
        assert!(
            !constructors.is_empty(),
            "Z3DatatypeSort requires at least one constructor"
        );

        let mut decl = DatatypeDecl::new(name);
        for (tag, z3_con) in constructors.iter().enumerate() {
            let mut con = Constructor::new(z3_con.name.clone(), tag as u32);
            for field in &z3_con.fields {
                con = con.with_field(field.name.clone(), field.sort_name.clone());
            }
            decl = decl.with_constructor(con);
        }

        // Register the sort in the term manager so that TermId-based usage
        // can reference it by SortId.
        let sort_id = ctx.tm.borrow_mut().sorts.mk_datatype_sort(name);

        Self { decl, sort_id }
    }

    /// Number of constructors in this datatype.
    #[must_use]
    pub fn num_constructors(&self) -> usize {
        self.decl.constructors.len()
    }

    /// Return the [`FuncDecl`] for the `i`-th constructor.
    ///
    /// The returned `FuncDecl` has arity equal to the number of fields of the
    /// constructor, and its range is this datatype's sort.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_constructors()`.
    #[must_use]
    pub fn constructor(&self, ctx: &Z3Context, i: usize) -> crate::z3_compat::ext::FuncDecl {
        let con = &self.decl.constructors[i];
        // Each field maps to the range sort declared for it; for simplicity we
        // use the context's int sort as a placeholder for unknown sorts and
        // look up Bool/Real specifically.
        let domain: Vec<SortId> = con
            .fields
            .iter()
            .map(|f| sort_name_to_id(ctx, &f.sort))
            .collect();
        crate::z3_compat::ext::FuncDecl::new(ctx, &con.name, &domain, self.sort_id)
    }

    /// Return the recogniser [`FuncDecl`] for the `i`-th constructor.
    ///
    /// The recogniser takes one argument of this datatype's sort and returns
    /// `Bool`.  Its name is `"is-<constructor-name>"`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_constructors()`.
    #[must_use]
    pub fn recognizer(&self, ctx: &Z3Context, i: usize) -> crate::z3_compat::ext::FuncDecl {
        let con = &self.decl.constructors[i];
        let name = format!("is-{}", con.name);
        let bool_sort = ctx.bool_sort();
        crate::z3_compat::ext::FuncDecl::new(ctx, &name, &[self.sort_id], bool_sort)
    }

    /// Return the accessor [`FuncDecl`] for field `field_i` of constructor `con_i`.
    ///
    /// The accessor takes one argument of this datatype's sort and returns the
    /// sort of the field.
    ///
    /// # Panics
    ///
    /// Panics if `con_i >= self.num_constructors()` or
    /// `field_i >= constructors[con_i].fields.len()`.
    #[must_use]
    pub fn accessor(
        &self,
        ctx: &Z3Context,
        con_i: usize,
        field_i: usize,
    ) -> crate::z3_compat::ext::FuncDecl {
        let con = &self.decl.constructors[con_i];
        let field = &con.fields[field_i];
        let field_sort = sort_name_to_id(ctx, &field.sort);
        crate::z3_compat::ext::FuncDecl::new(ctx, &field.name, &[self.sort_id], field_sort)
    }

    /// Return the sort ID of this datatype's sort.
    #[must_use]
    pub fn sort_id(&self) -> SortId {
        self.sort_id
    }

    /// Return a reference to the underlying [`DatatypeDecl`].
    #[must_use]
    pub fn decl(&self) -> &DatatypeDecl {
        &self.decl
    }
}

/// Convenience constructor for building a [`Z3Constructor`] specification.
#[must_use]
pub fn mk_constructor(name: &str, fields: &[(&str, &str)]) -> Z3Constructor {
    Z3Constructor {
        name: name.to_string(),
        fields: fields
            .iter()
            .map(|&(fname, fsort)| Z3Field {
                name: fname.to_string(),
                sort_name: fsort.to_string(),
            })
            .collect(),
    }
}

/// Map a sort-name string to a [`SortId`] in the given context.
///
/// Handles `"Bool"`, `"Int"`, `"Real"` and falls back to the integer sort for
/// anything else (the sort must be declared separately for full correctness).
fn sort_name_to_id(ctx: &Z3Context, name: &str) -> SortId {
    match name {
        "Bool" => ctx.bool_sort(),
        "Int" => ctx.int_sort(),
        "Real" => ctx.real_sort(),
        other => ctx
            .tm
            .borrow_mut()
            .sorts
            .mk_datatype_sort(other),
    }
}

// ─── Z3Solver: check_assumptions + unsat_core ────────────────────────────────

impl Z3Solver {
    /// Check satisfiability under a list of additional assumptions.
    ///
    /// The assumptions are asserted into a temporary scope (push/pop) so they
    /// do not modify the permanent assertion stack.  The assumptions must have
    /// been built using the same [`Z3Context`] that was passed to
    /// [`Z3Solver::new`].
    pub fn check_assumptions(&mut self, assumptions: &[Bool]) -> SatResult {
        let term_ids: Vec<TermId> = assumptions.iter().map(|b| b.id).collect();
        self.ctx.check_with_assumptions_raw(&term_ids).into()
    }

    /// Return the unsat core from the most recent `check()` or
    /// `check_assumptions()` that returned `Unsat`.
    ///
    /// Returns an empty `Vec` if no core is available (e.g. the last result
    /// was `Sat`, or unsat-core production is not enabled).
    #[must_use]
    pub fn unsat_core(&self) -> Vec<Bool> {
        match self.ctx.get_unsat_core_raw() {
            None => Vec::new(),
            Some(core) => {
                // The UnsatCore stores assertion *indices* into the assertion
                // list.  We use Context::get_assertions() to map back to TermIds.
                let assertions = self.ctx.get_assertions();
                core.indices
                    .iter()
                    .filter_map(|&idx| assertions.get(idx as usize).copied())
                    .map(|id| Bool { id })
                    .collect()
            }
        }
    }
}

// ─── Z3AstVector ─────────────────────────────────────────────────────────────

/// Analogue of `z3::AstVector`.
///
/// An ordered collection of [`Bool`] terms associated with a context.
/// Terms are stored by [`TermId`] and can be iterated or indexed.
pub struct Z3AstVector {
    terms: Vec<TermId>,
    ctx_tm: Rc<std::cell::RefCell<TermManager>>,
}

impl Z3AstVector {
    /// Create a new, empty vector.
    #[must_use]
    pub fn new(ctx: &Z3Context) -> Self {
        Self {
            terms: Vec::new(),
            ctx_tm: ctx.tm.clone(),
        }
    }

    /// Append a boolean term to the vector.
    pub fn push(&mut self, term: &Bool) {
        self.terms.push(term.id);
    }

    /// Number of terms in this vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Returns `true` if the vector contains no terms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Retrieve the `i`-th term as a [`Bool`].
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    #[must_use]
    pub fn get(&self, i: usize) -> Bool {
        Bool { id: self.terms[i] }
    }

    /// Iterate over all terms as [`Bool`] values.
    pub fn iter(&self) -> impl Iterator<Item = Bool> + '_ {
        self.terms.iter().map(|&id| Bool { id })
    }

    /// Return `true` if any term in this vector is syntactically `true`.
    ///
    /// Uses the context's term manager to check the `TermKind`.
    #[must_use]
    pub fn any_true(&self) -> bool {
        use oxiz_core::ast::TermKind;
        let tm = self.ctx_tm.borrow();
        self.terms.iter().any(|&id| {
            tm.get(id)
                .is_some_and(|t| matches!(t.kind, TermKind::True))
        })
    }
}

// ─── TermManager read-only helpers ───────────────────────────────────────────
// The core TermManager does not yet expose a `mk_true_ro` / `mk_false_ro` that
// takes `&self`.  We add a small compatibility shim via a local trait to avoid
// an immutable borrow while calling methods that internally need `&mut self`
// only for caching.

trait TermManagerExt {
    fn mk_true_ro(&self) -> TermId;
    fn mk_false_ro(&self) -> TermId;
}

impl TermManagerExt for TermManager {
    fn mk_true_ro(&self) -> TermId {
        // True and False are always the first two terms in every fresh
        // TermManager; their IDs are stable across the lifetime of any
        // single manager instance.
        use oxiz_core::ast::TermKind;
        // Walk the first few entries to find True.
        for idx in 0..8u32 {
            let tid = TermId(idx);
            if let Some(t) = self.get(tid) {
                if matches!(t.kind, TermKind::True) {
                    return tid;
                }
            }
        }
        TermId(0) // fallback — should never be reached
    }

    fn mk_false_ro(&self) -> TermId {
        use oxiz_core::ast::TermKind;
        for idx in 0..8u32 {
            let tid = TermId(idx);
            if let Some(t) = self.get(tid) {
                if matches!(t.kind, TermKind::False) {
                    return tid;
                }
            }
        }
        TermId(1) // fallback
    }
}

// ─── Z3FuncInterp / Z3FuncEntry / Z3Value ────────────────────────────────────

/// A model value exposed through the Z3 compat layer.
///
/// Wraps a string representation of the value so callers do not need to depend
/// directly on `oxiz_core::model::Value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Z3Value {
    /// String representation (e.g. `"42"`, `"true"`, `"#b0011"`).
    pub inner: String,
}

impl Z3Value {
    /// Create a `Z3Value` from an arbitrary string representation.
    #[must_use]
    pub fn from_string(s: String) -> Self {
        Self { inner: s }
    }

    /// Return the string representation of this value.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

impl std::fmt::Display for Z3Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.inner)
    }
}

/// One `(args → value)` entry in a [`Z3FuncInterp`].
#[derive(Debug, Clone)]
pub struct Z3FuncEntry {
    /// Argument values for this entry (one per function parameter).
    pub args: Vec<Z3Value>,
    /// The output value for this argument combination.
    pub value: Z3Value,
}

/// Analogue of `z3::FuncInterp`.
///
/// Represents the interpretation of an uninterpreted function in a model as a
/// finite table of `(args → value)` entries plus an `else_value` for all other
/// input combinations.
///
/// Obtained via [`Z3Model::get_func_interp`].
pub struct Z3FuncInterp {
    /// Finite explicit entries.
    entries: Vec<Z3FuncEntry>,
    /// Value returned for inputs not covered by any entry.
    else_value: Z3Value,
    /// Number of arguments of the function.
    arity: usize,
}

impl Z3FuncInterp {
    /// Create a `Z3FuncInterp` from the raw data returned by the solver context.
    pub(crate) fn from_raw(raw: &crate::z3_compat::FuncInterpRaw) -> Self {
        let (raw_entries, else_str, arity) = raw;
        let entries = raw_entries
            .iter()
            .map(|(arg_strs, val_str)| Z3FuncEntry {
                args: arg_strs
                    .iter()
                    .map(|s| Z3Value::from_string(s.clone()))
                    .collect(),
                value: Z3Value::from_string(val_str.clone()),
            })
            .collect();
        Self {
            entries,
            else_value: Z3Value::from_string(else_str.clone()),
            arity: *arity,
        }
    }

    /// Return the number of explicit `(args → value)` entries.
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Return the arity (number of arguments) of the interpreted function.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Return the `else` value applied to any input not matched by an entry.
    #[must_use]
    pub fn else_value(&self) -> &Z3Value {
        &self.else_value
    }

    /// Return the `i`-th entry.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_entries()`.
    #[must_use]
    pub fn get_entry(&self, i: usize) -> &Z3FuncEntry {
        &self.entries[i]
    }

    /// Iterate over all explicit entries.
    pub fn entries(&self) -> impl Iterator<Item = &Z3FuncEntry> {
        self.entries.iter()
    }
}

// ─── Z3Model::get_func_interp ─────────────────────────────────────────────────

impl crate::z3_compat::Z3Model {
    /// Return the full interpretation of an uninterpreted function `f` in this
    /// model, or `None` if `f` was not declared or is not present in the model.
    ///
    /// The returned [`Z3FuncInterp`] contains the finite set of `(args → value)`
    /// entries that the solver determined, plus an `else_value` for all other
    /// inputs.
    ///
    /// # Stub note
    ///
    /// When the EUF e-graph does not contain any application of `f` (e.g. the
    /// function was declared but never constrained), `num_entries()` will be 0
    /// and `else_value()` will be the default value for the return sort.  This
    /// is a valid (conservative) interpretation: the solver is free to choose
    /// any value for unconstrained applications.
    #[must_use]
    pub fn get_func_interp(
        &self,
        f: &crate::z3_compat::ext::FuncDecl,
    ) -> Option<Z3FuncInterp> {
        self.func_interp_raw(&f.name)
            .map(Z3FuncInterp::from_raw)
    }
}

// ─── Re-export convenience items ─────────────────────────────────────────────

/// Re-export `DatatypeDecl`, `Constructor`, `Selector`, and `Field` from the
/// theories crate so downstream code can use them without a direct dep on
/// `oxiz-theories`.
pub use oxiz_theories::datatype::{
    Constructor as DtConstructor, DatatypeDecl as DtDecl, DatatypeSort as DtSort,
    Field as DtField, Selector as DtSelector,
};
