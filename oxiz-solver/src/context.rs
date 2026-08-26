//! Solver context

#[allow(unused_imports)]
use crate::prelude::*;
use crate::solver::{Solver, SolverResult};
use oxiz_core::ast::{TermId, TermKind, TermManager};
#[cfg(feature = "std")]
use oxiz_core::error::Result;
#[cfg(feature = "std")]
use oxiz_core::smtlib::{Command, parse_script};
use oxiz_core::sort::SortId;
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};

/// Model / value / sort output formatting for `(get-model)`, `(get-value ..)`,
/// and the function-interpretation extensions. Split into a child module so
/// this file stays under the 2000-line policy limit.
mod model_fmt;

/// Sort-expression string resolution (`parse_sort_name`) for the
/// declaration-shaped script commands. Split into a child module so this
/// file stays under the 2000-line policy limit.
mod sort_name;

/// The cardinality axioms that give the reserved `RoundingMode` sort its
/// exactly-five-element domain. Split into a child module so this file stays
/// under the 2000-line policy limit.
mod rounding_mode;

/// Constructor/selector registration for `(declare-datatype(s) ..)`. Split into
/// a child module so this file stays under the 2000-line policy limit.
#[cfg(feature = "std")]
mod declare_datatype;

/// Binary proof-log serialisation for `check_sat`. Split into a child module so
/// this file stays under the 2000-line policy limit.
#[cfg(feature = "std")]
mod proof_log;

/// `define-fun-rec` / `define-funs-rec`: the recursive-definition registry and
/// the fuel-bounded unfolding driver that discharges it. Split into a child
/// module so this file stays under the 2000-line policy limit.
mod recfun;

/// Regression tests for the closing restore check of `(get-consequences ..)`.
/// Split into a child module so this file stays under the 2000-line policy
/// limit.
#[cfg(test)]
mod consequences_tests;

/// Raw function interpretation: a list of `(arg_strings, value_string)` entries
/// together with an `else_value` string and the function arity.
///
/// Used as the return type of [`Context::get_func_interp_raw`] to avoid pulling
/// `oxiz_core::model` types into the public API of this file.
pub type RawFuncInterp = (Vec<(Vec<String>, String)>, String, usize);

/// A declared constant
#[derive(Debug, Clone)]
struct DeclaredConst {
    /// The term ID for this constant
    term: TermId,
    /// The sort of this constant
    sort: SortId,
    /// The name of this constant
    name: String,
}

/// A declared function
#[derive(Debug, Clone)]
struct DeclaredFun {
    /// The function name
    name: String,
    /// Argument sorts
    arg_sorts: Vec<SortId>,
    /// Return sort
    ret_sort: SortId,
}

/// Solver context for managing the solving process
///
/// The `Context` provides a high-level API for SMT solving, similar to
/// the SMT-LIB2 standard. It manages declarations, assertions, and solver state.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use oxiz_solver::Context;
///
/// let mut ctx = Context::new();
/// ctx.set_logic("QF_UF");
///
/// // Declare boolean constants
/// let p = ctx.declare_const("p", ctx.terms.sorts.bool_sort);
/// let q = ctx.declare_const("q", ctx.terms.sorts.bool_sort);
///
/// // Assert p AND q
/// let formula = ctx.terms.mk_and(vec![p, q]);
/// ctx.assert(formula);
///
/// // Check satisfiability
/// ctx.check_sat();
/// ```
///
/// ## SMT-LIB2 Script Execution
///
/// ```
/// use oxiz_solver::Context;
///
/// let mut ctx = Context::new();
///
/// let script = r#"
/// (set-logic QF_LIA)
/// (declare-const x Int)
/// (assert (>= x 0))
/// (assert (<= x 10))
/// (check-sat)
/// "#;
///
/// let _ = ctx.execute_script(script);
/// ```
#[derive(Debug)]
pub struct Context {
    /// Term manager
    pub terms: TermManager,
    /// Solver instance
    solver: Solver,
    /// Current logic
    logic: Option<String>,
    /// Assertions
    assertions: Vec<TermId>,
    /// Assertion stack for push/pop
    assertion_stack: Vec<usize>,
    /// Declared constants
    declared_consts: Vec<DeclaredConst>,
    /// Declared constants stack for push/pop
    const_stack: Vec<usize>,
    /// Mapping from constant names to indices (for efficient removal)
    const_name_to_index: crate::prelude::HashMap<String, usize>,
    /// Declared functions
    declared_funs: Vec<DeclaredFun>,
    /// Declared functions stack for push/pop
    fun_stack: Vec<usize>,
    /// Mapping from function names to indices
    fun_name_to_index: crate::prelude::HashMap<String, usize>,
    /// Last check-sat result
    last_result: Option<SolverResult>,
    /// The assumption terms passed to the most recent `check-sat-assuming`
    /// (empty for a plain `check-sat`).  Retained so `get-unsat-assumptions`
    /// can report an unsatisfiable subset after an `unsat` verdict.
    last_assumptions: Vec<TermId>,
    /// Options
    options: crate::prelude::HashMap<String, String>,
    /// Sorts declared via `(declare-sort name arity)`, keyed by name.
    ///
    /// The `SortId` itself lives in `self.terms.sorts` (interned lazily,
    /// on first reference, exactly like the SMT-LIB parser does); this
    /// map exists purely for script-level introspection of which names
    /// were declared and with what arity.
    declared_sorts: crate::prelude::HashMap<String, u32>,
    /// Recursive function definitions in scope, plus the bookkeeping for the
    /// scratch solver scope the unfolding driver leaves open (see
    /// [`context::recfun`]).
    recfun: recfun::RecFunState,
    /// Optional path for binary proof logging.
    ///
    /// When set, `check_sat` creates a `ProofLogger` at this path, records
    /// proof steps derived from the solver result, and flushes/closes the log
    /// before returning.
    #[cfg(feature = "std")]
    proof_log_path: Option<PathBuf>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Create a new context
    #[must_use]
    pub fn new() -> Self {
        Self {
            terms: TermManager::new(),
            solver: Solver::new(),
            logic: None,
            assertions: Vec::new(),
            assertion_stack: Vec::new(),
            declared_consts: Vec::new(),
            const_stack: Vec::new(),
            const_name_to_index: crate::prelude::HashMap::new(),
            declared_funs: Vec::new(),
            fun_stack: Vec::new(),
            fun_name_to_index: crate::prelude::HashMap::new(),
            last_result: None,
            last_assumptions: Vec::new(),
            options: crate::prelude::HashMap::new(),
            declared_sorts: crate::prelude::HashMap::new(),
            recfun: recfun::RecFunState::default(),
            #[cfg(feature = "std")]
            proof_log_path: None,
        }
    }

    /// Configure a path for binary proof logging.
    ///
    /// When a path is configured, every subsequent call to `check_sat` opens a
    /// [`oxiz_proof::logging::ProofLogger`] at that path, writes a structural
    /// summary of the proof, and flushes/closes the log before returning.
    /// Pass `None` to disable proof logging.
    #[cfg(feature = "std")]
    pub fn set_proof_log_path(&mut self, path: Option<PathBuf>) {
        self.proof_log_path = path;
    }

    /// Return the currently configured proof log path, if any.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn proof_log_path(&self) -> Option<&Path> {
        self.proof_log_path.as_deref()
    }

    /// Verify a binary proof log produced by a previous `check_sat` call with
    /// proof logging enabled.
    ///
    /// Delegates to [`oxiz_proof::replay::ProofReplayer::replay_from_file`].
    ///
    /// # Errors
    ///
    /// Returns `Err` only for hard I/O or binary-format failures; logical
    /// invalidity is encoded as `Ok(VerificationResult::Invalid(_))`.
    #[cfg(feature = "std")]
    pub fn verify_proof_log(
        path: &Path,
    ) -> std::result::Result<oxiz_proof::replay::VerificationResult, oxiz_proof::replay::ProofError>
    {
        oxiz_proof::replay::ProofReplayer::replay_from_file(path)
    }

    /// Declare a constant
    pub fn declare_const(&mut self, name: &str, sort: SortId) -> TermId {
        let term = self.terms.mk_var(name, sort);
        let index = self.declared_consts.len();
        self.declared_consts.push(DeclaredConst {
            term,
            sort,
            name: name.to_string(),
        });
        self.const_name_to_index.insert(name.to_string(), index);
        // Seed the constant as an MBQI ground-instantiation candidate.  This is
        // the call site [`crate::solver::Solver::register_declared_const`]
        // documents ("must be called from the context layer whenever a
        // `declare-const` command is processed"): without it, a trigger-free
        // quantifier has no in-scope constant to instantiate with.
        self.solver.register_declared_const(term, sort);
        // A `RoundingMode` constant needs the closure axiom that pins it to
        // one of the five real modes; see `context::rounding_mode`.
        if self.is_rounding_mode_sort(sort) {
            self.assert_rounding_mode_closure(term);
        }
        term
    }

    /// Declare a function
    ///
    /// Registers a function signature in the context. For nullary functions (constants),
    /// use `declare_const` instead.
    pub fn declare_fun(&mut self, name: &str, arg_sorts: Vec<SortId>, ret_sort: SortId) {
        let index = self.declared_funs.len();
        self.declared_funs.push(DeclaredFun {
            name: name.to_string(),
            arg_sorts,
            ret_sort,
        });
        self.fun_name_to_index.insert(name.to_string(), index);
    }

    /// Get function signature if it exists
    pub fn get_fun_signature(&self, name: &str) -> Option<(Vec<SortId>, SortId)> {
        self.fun_name_to_index.get(name).and_then(|&idx| {
            self.declared_funs
                .get(idx)
                .map(|f| (f.arg_sorts.clone(), f.ret_sort))
        })
    }

    /// Iterate over the names of all currently declared uninterpreted functions.
    pub fn declared_function_names(&self) -> impl Iterator<Item = &str> {
        self.declared_funs.iter().map(|d| d.name.as_str())
    }

    /// Iterate over `(name, arity)` for every sort declared via
    /// `(declare-sort name arity)` (through [`Context::execute_script`]).
    pub fn declared_sort_names(&self) -> impl Iterator<Item = (&str, u32)> {
        self.declared_sorts.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// Set the logic
    ///
    /// SMT-LIB 2.6 Fig. 4.1 puts `set-logic` in `start` mode and leaves the
    /// solver in `assert` mode, so it can never legally follow a `check-sat`.
    /// It is accepted here anyway (leniently, like most solvers), but it must
    /// still invalidate the cached verdict: [`crate::solver::Solver::set_logic`]
    /// swaps the arithmetic engine and can install NLSAT, which would leave any
    /// cached model describing a solver configuration that no longer exists.
    pub fn set_logic(&mut self, logic: &str) {
        self.logic = Some(logic.to_string());
        self.solver.set_logic(logic);
        self.invalidate_last_check();
    }

    /// Get the current logic
    #[must_use]
    pub fn logic(&self) -> Option<&str> {
        self.logic.as_deref()
    }

    /// Drop the cached `check-sat` verdict and its assumption context.
    ///
    /// SMT-LIB 2.6 §4.1.1 (solver modes, Fig. 4.1): `assert`, `push`, `pop`,
    /// `reset-assertions` and `reset` all take the solver back to `assert`
    /// mode, and `get-model`, `get-value`, `get-assignment`, `get-unsat-core`,
    /// `get-proof` and `get-unsat-assumptions` are available only in `sat` /
    /// `unsat` mode.  Every one of those queries is gated on `last_result`, so
    /// clearing it *is* that mode transition: they answer with the standard
    /// "not available" error instead of reporting a model, core or proof that
    /// belongs to a superseded assertion stack.
    ///
    /// It is also where the recursive-definition driver's scratch solver scope
    /// is discharged: that scope holds unfolded definitional instances and is
    /// deliberately left *open* after an accepted verdict so `(get-model)` /
    /// `(get-value ..)` can read the model built with them.  Every caller
    /// therefore invokes this **before** touching the solver's scope stack —
    /// popping the scratch scope after a `solver.push()` would retract the
    /// freshly pushed scope, and after a `solver.assert(..)` would throw the
    /// user's assertion away with it.
    fn invalidate_last_check(&mut self) {
        self.discharge_recfun_scope();
        self.last_result = None;
        self.last_assumptions.clear();
    }

    /// Add an assertion
    pub fn assert(&mut self, term: TermId) {
        self.invalidate_last_check();
        self.assertions.push(term);
        self.solver.assert(term, &mut self.terms);
    }

    /// Add a named assertion (from `(assert (! phi :named name))`).
    ///
    /// The name is threaded into the solver's named-assertion tracking so that,
    /// with `:produce-unsat-cores` enabled, `(get-unsat-core)` can report the
    /// user labels of the assertions that participate in an `unsat` refutation.
    /// The name is recorded unconditionally at assert time, so enabling
    /// `:produce-unsat-cores` mid-session still yields a labelled core.
    pub fn assert_named(&mut self, term: TermId, name: &str) {
        self.invalidate_last_check();
        self.assertions.push(term);
        self.solver.assert_named(term, name, &mut self.terms);
    }

    /// One solver check, with the two verdict gates that belong to *every*
    /// check-sat: the rounding-mode closure axioms on the way in and the array
    /// honesty gate on the way out.
    ///
    /// Factored out so the recursive-definition driver
    /// ([`Context::check_sat_recfun`]) runs each of its fuel rounds through
    /// exactly these gates.  A driver that called `solver.check` directly would
    /// silently lose them for any script using `define-fun-rec`.
    pub(super) fn check_sat_core(&mut self) -> SolverResult {
        // The five rounding modes must be pairwise distinct for any solve that
        // mentions one; see `context::rounding_mode` for why this is asserted
        // here rather than once at declaration time.
        if self.terms.rounding_mode_used() {
            self.assert_rounding_mode_distinctness();
        }
        let mut result = self.solver.check(&mut self.terms);

        // Array soundness honesty gate: the syntactic array checks and the EUF
        // congruence core do not implement full array extensionality.  If a
        // positive equality between two store terms survived to a `Sat` verdict
        // without being refuted as a conflict, the assignment is not certified —
        // the core may have merged the two store terms into one class without
        // enforcing element-wise agreement of their bases.  Answer `Unknown`
        // rather than a possibly-spurious `Sat` (never a silent wrong result).
        if result == SolverResult::Sat && self.solver.array_atoms_need_theory(&self.terms) {
            result = SolverResult::Unknown;
        }
        result
    }

    /// Check satisfiability
    pub fn check_sat(&mut self) -> SolverResult {
        // With recursive definitions in scope, the plain check would solve a
        // strictly weaker problem (every `f(x)` unconstrained), so the
        // fuel-bounded unfolding driver takes over.
        let result = if self.recfun.is_empty() {
            self.check_sat_core()
        } else {
            self.check_sat_recfun()
        };

        // A plain check-sat clears any assumption context from a prior
        // check-sat-assuming, so a following get-unsat-assumptions does not
        // report stale assumptions.
        self.last_assumptions.clear();
        self.last_result = Some(result);

        // Write a binary proof log if a path is configured (std-only).
        #[cfg(feature = "std")]
        if let Some(ref path) = self.proof_log_path.clone() {
            if let Err(e) = self.write_proof_log(path, result) {
                // Non-fatal: warn but do not abort the solve.
                #[cfg(feature = "tracing")]
                tracing::warn!("proof log write failed for {:?}: {}", path, e);
                let _ = e;
            }
        }

        result
    }

    /// Evaluate a `term` in the current model.
    ///
    /// Returns `None` if no model is available (i.e. the last `check_sat` did
    /// not return `Sat`).  Otherwise, calls `Model::eval` which traverses the
    /// term structure, substituting variables with their model values, and
    /// returns the simplified/concrete `TermId`.
    ///
    /// The returned `TermId` belongs to `self.terms` — the same `TermManager`
    /// owned by this `Context`.
    pub fn eval_in_model(&mut self, term: TermId) -> Option<TermId> {
        if self.last_result != Some(SolverResult::Sat) {
            return None;
        }
        let value = self.solver.model()?.eval(term, &mut self.terms);
        Some(value)
    }

    /// Push a context level
    pub fn push(&mut self) {
        // Must precede `solver.push()`: it discharges the recfun scratch scope,
        // which would otherwise retract the scope opened just below.
        self.invalidate_last_check();
        self.assertion_stack.push(self.assertions.len());
        self.const_stack.push(self.declared_consts.len());
        self.fun_stack.push(self.declared_funs.len());
        self.recfun.push_scope();
        self.solver.push();
    }

    /// Pop a context level with incremental declaration removal
    pub fn pop(&mut self) {
        self.invalidate_last_check();
        if let Some(len) = self.assertion_stack.pop() {
            self.assertions.truncate(len);
            if let Some(const_len) = self.const_stack.pop() {
                // Remove constants from the name-to-index mapping
                while self.declared_consts.len() > const_len {
                    if let Some(decl) = self.declared_consts.pop() {
                        self.const_name_to_index.remove(&decl.name);
                    }
                }
            }
            if let Some(fun_len) = self.fun_stack.pop() {
                // Remove functions from the name-to-index mapping
                while self.declared_funs.len() > fun_len {
                    if let Some(decl) = self.declared_funs.pop() {
                        self.fun_name_to_index.remove(&decl.name);
                    }
                }
            }
            // Inside the same guard as the other stacks: an unbalanced `pop`
            // must not desynchronize the recursive-definition scope stack from
            // the assertion stack.
            self.recfun.pop_scope();
            self.solver.pop();
        }
    }

    /// Reset the context
    pub fn reset(&mut self) {
        // Before `solver.reset()` wipes the scope stack: the scratch scope
        // recorded by the recfun driver must be forgotten, not popped, or the
        // next discharge would pop a scope that no longer exists.
        self.recfun.clear_all();
        self.solver.reset();
        self.assertions.clear();
        self.assertion_stack.clear();
        self.declared_consts.clear();
        self.const_stack.clear();
        self.const_name_to_index.clear();
        self.declared_funs.clear();
        self.fun_stack.clear();
        self.fun_name_to_index.clear();
        self.logic = None;
        self.options.clear();
        self.invalidate_last_check();
    }

    /// Reset assertions (keep declarations and options)
    ///
    /// SMT-LIB 2.6 §4.2.5: `reset-assertions` empties the assertion stack but —
    /// unlike `reset` — keeps the current logic, all declarations/definitions
    /// and every option.  [`crate::solver::Solver::reset`] is a *total* reset,
    /// so anything the context still owns has to be re-established on the fresh
    /// solver afterwards, otherwise the solver silently loses configuration the
    /// script never retracted:
    ///
    /// - the **logic**, which selects the arithmetic engine (`QF_NRA`/`QF_NIA`
    ///   install NLSAT, `QF_LRA` an LRA simplex, …).  Without this a `QF_NRA`
    ///   problem that answered `sat` before the reset answered `unknown` after.
    /// - the **options**, replayed through [`Context::set_option`] so wired keys
    ///   (`:random-seed`, `:timeout`, `:produce-proofs`, …) survive; the SAT
    ///   engine's PRNG in particular is re-seeded to its default by a reset.
    /// - the **declared constants** as MBQI ground-instantiation candidates, so
    ///   trigger-free quantifiers keep the in-scope constants they had before.
    pub fn reset_assertions(&mut self) {
        // `reset-assertions` keeps definitions but pops every level, so the
        // recursive definitions introduced inside a `push` go away with it.
        // Like `reset`, the scratch scope is *forgotten* rather than popped —
        // `solver.reset()` below empties the scope stack outright.
        self.recfun.retract_to_base();
        self.solver.reset();
        self.assertions.clear();
        self.assertion_stack.clear();
        // Keep declared_consts, const_stack, const_name_to_index,
        // declared_funs, fun_stack, and fun_name_to_index
        // Re-assert nothing - solver is fresh
        self.invalidate_last_check();

        if let Some(logic) = self.logic.clone() {
            self.solver.set_logic(&logic);
        }
        for (key, value) in self
            .options
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
        {
            self.set_option(&key, &value);
        }
        for decl in &self.declared_consts {
            self.solver.register_declared_const(decl.term, decl.sort);
        }
    }

    /// Get all current assertions
    #[must_use]
    pub fn get_assertions(&self) -> &[TermId] {
        &self.assertions
    }

    /// Format assertions as SMT-LIB2
    #[cfg(feature = "std")]
    pub fn format_assertions(&self) -> String {
        if self.assertions.is_empty() {
            return "()".to_string();
        }
        let printer = oxiz_core::smtlib::Printer::new(&self.terms);
        let mut parts = Vec::new();
        for &term in &self.assertions {
            parts.push(printer.print_term(term));
        }
        format!("({})", parts.join("\n "))
    }

    /// Set an option.
    ///
    /// Recognised keys are wired into the underlying [`crate::SolverConfig`] and take
    /// effect on the next `check_sat`.  All keys (recognised or not) are recorded
    /// so that `(get-option ...)` reflects the last value set.  A leading `:` is
    /// stripped so both `:timeout` and `timeout` resolve identically.
    ///
    /// Wired keys (each consumed by the solve loop, so setting them actually
    /// changes behaviour):
    ///
    /// - `produce-proofs` (`true`/`false`) — enable proof generation.
    /// - `produce-unsat-cores` (`true`/`false`) — enable unsat-core tracking.
    /// - `timeout` (milliseconds) — wall-clock budget for the search; `0`
    ///   disables it.  Maps to [`crate::SolverConfig::timeout_ms`], enforced between
    ///   MBQI rounds and inside the theory callbacks.
    /// - `max-conflicts` / `max-decisions` (non-negative integer) — resource
    ///   limits; `0` means unlimited.
    /// - `theory-mode` (`eager`/`lazy`) — theory propagation eagerness.
    /// - `simplify` (`true`/`false`) — pre-solve simplification of asserted
    ///   formulas.
    /// - `random-seed` / `random_seed` (non-negative integer) — seed for the SAT
    ///   engine's phase-randomization PRNG.  It is threaded straight into the SAT
    ///   solver via [`crate::solver::Solver::set_random_seed`], so it perturbs the
    ///   decision order (and hence which model a satisfiable problem yields)
    ///   without ever changing the sat/unsat verdict.  A seed of `0` reproduces
    ///   the default behaviour.
    ///
    /// Keys such as `restarts`, `branching`, and memory limits are *recorded but
    /// not enforced*: the corresponding levers are fixed at solver construction
    /// time (or have no wiring in this crate yet), so honouring them would require
    /// an `oxiz-solver` core change.  They are intentionally left as no-ops rather
    /// than silently pretending to take effect.
    pub fn set_option(&mut self, key: &str, value: &str) {
        let key = key.trim_start_matches(':');
        self.options.insert(key.to_string(), value.to_string());

        // Handle special options that affect the solver.
        match key {
            "produce-proofs" => {
                let mut config = self.solver.config().clone();
                config.proof = value == "true";
                self.solver.set_config(config);
            }
            "produce-unsat-cores" => {
                self.solver.set_produce_unsat_cores(value == "true");
            }
            "timeout" => {
                if let Ok(ms) = value.trim().parse::<u64>() {
                    let mut config = self.solver.config().clone();
                    config.timeout_ms = ms;
                    self.solver.set_config(config);
                }
            }
            "max-conflicts" | "max_conflicts" => {
                if let Ok(n) = value.trim().parse::<u64>() {
                    let mut config = self.solver.config().clone();
                    config.max_conflicts = n;
                    self.solver.set_config(config);
                }
            }
            "max-decisions" | "max_decisions" => {
                if let Ok(n) = value.trim().parse::<u64>() {
                    let mut config = self.solver.config().clone();
                    config.max_decisions = n;
                    self.solver.set_config(config);
                }
            }
            "theory-mode" | "theory_mode" => {
                let mode = match value.trim().to_ascii_lowercase().as_str() {
                    "lazy" => Some(crate::solver::TheoryMode::Lazy),
                    "eager" => Some(crate::solver::TheoryMode::Eager),
                    _ => None,
                };
                if let Some(mode) = mode {
                    let mut config = self.solver.config().clone();
                    config.theory_mode = mode;
                    self.solver.set_config(config);
                }
            }
            "simplify" => {
                let mut config = self.solver.config().clone();
                config.simplify = value == "true";
                self.solver.set_config(config);
            }
            "random-seed" | "random_seed" => {
                // Thread the seed into the SAT engine's phase-randomization PRNG.
                // Only enforce a well-formed non-negative integer; a malformed
                // value is still recorded (above) so `(get-option ...)` reflects
                // exactly what the user set, but it does not silently corrupt the
                // RNG state.
                if let Ok(seed) = value.trim().parse::<u64>() {
                    self.solver.set_random_seed(seed);
                }
            }
            _ => {}
        }
    }

    /// Get an option
    #[must_use]
    pub fn get_option(&self, key: &str) -> Option<&str> {
        self.options.get(key).map(String::as_str)
    }

    /// Whether SMT-LIB `:print-success` mode is currently enabled.
    ///
    /// When on, `execute_script` emits a `success` acknowledgement after every
    /// command that succeeds without producing its own response (per SMT-LIB
    /// 2.6).  Defaults to off until `(set-option :print-success true)` is seen.
    fn print_success_enabled(&self) -> bool {
        self.get_option("print-success") == Some("true")
    }

    /// Format an option value
    fn format_option(&self, key: &str) -> String {
        match self.get_option(key) {
            Some(val) => val.to_string(),
            None => {
                // Return default values for well-known options
                match key {
                    "produce-models" => "false".to_string(),
                    "produce-unsat-cores" => "false".to_string(),
                    "produce-proofs" => "false".to_string(),
                    "produce-assignments" => "false".to_string(),
                    // print-success is honored by `execute_script` (a `success`
                    // line is emitted after each silently-succeeding command
                    // once enabled), but defaults to off — matching common
                    // solver behavior — so that scripts that never opt in keep
                    // their existing terse output.  Once `(set-option
                    // :print-success true)` is issued, the `Some(val)` branch
                    // above reports the real `true`.
                    "print-success" => "false".to_string(),
                    _ => "unsupported".to_string(),
                }
            }
        }
    }

    /// Answer a `(get-info <keyword>)` request.
    ///
    /// The SMT-LIB lexer strips the leading `:` from an info flag, so a request
    /// for `:all-statistics` arrives here as `all-statistics`; we normalize by
    /// stripping any leading colon so both spellings resolve identically
    /// (previously the handler compared against `":all-statistics"` and could
    /// never match, making *every* `get-info` an error).  The mandatory
    /// standard flags (`:name`, `:version`, `:authors`, `:error-behavior`,
    /// `:reason-unknown`) are answered per SMT-LIB 2.6; `:all-statistics`
    /// returns the solver statistics.
    pub fn get_info(&self, keyword: &str) -> String {
        let key = keyword.trim_start_matches(':');
        match key {
            "all-statistics" => self.get_statistics(),
            "name" => "(:name \"oxiz\")".to_string(),
            "version" => format!("(:version \"{}\")", env!("CARGO_PKG_VERSION")),
            "authors" => "(:authors \"COOLJAPAN OU (Team Kitasan)\")".to_string(),
            "error-behavior" => "(:error-behavior continued-execution)".to_string(),
            "reason-unknown" => {
                // Report why the last check returned `unknown`, or `unsupported`
                // when the last result was decided (sat/unsat) or absent.
                match self.last_result {
                    Some(SolverResult::Unknown) => "(:reason-unknown incomplete)".to_string(),
                    _ => "(:reason-unknown \"not applicable\")".to_string(),
                }
            }
            _ => format!(
                "(error {})",
                oxiz_core::smtlib::format_string_literal(&format!(
                    "unsupported info keyword: :{}",
                    key
                ))
            ),
        }
    }

    /// Answer a `(get-assignment)` request.
    ///
    /// Per SMT-LIB, `get-assignment` reports the truth values that the current
    /// model assigns to Boolean-sorted terms.  This implementation returns a
    /// `(name value)` pair for every declared Boolean constant that the model
    /// assigns (`true`/`false`), which covers the labelled propositional
    /// variables users query in practice.  It returns `()` when the last check
    /// did not produce a model (not `sat`, or no model available).
    ///
    /// Boolean constants that never entered a constraint — and therefore carry no
    /// forced value — are reported as `false`, matching the default-completion
    /// convention used by [`Context::get_model`].
    pub fn get_assignment(&self) -> String {
        if self.last_result != Some(SolverResult::Sat) {
            return "()".to_string();
        }
        let Some(model) = self.solver.model() else {
            return "()".to_string();
        };
        let bool_sort = self.terms.sorts.bool_sort;
        let mut parts = Vec::new();
        for decl in &self.declared_consts {
            if decl.sort != bool_sort {
                continue;
            }
            let value = match model.get(decl.term).and_then(|v| self.terms.get(v)) {
                Some(t) if matches!(t.kind, TermKind::True) => "true",
                Some(t) if matches!(t.kind, TermKind::False) => "false",
                // No forced value: complete to `false` (see doc comment).
                _ => "false",
            };
            parts.push(format!("({} {})", decl.name, value));
        }
        format!("({})", parts.join(" "))
    }

    /// Answer a `(get-unsat-assumptions)` request.
    ///
    /// After a `check-sat-assuming` that returned `unsat`, this returns a subset
    /// of the supplied assumptions whose conjunction with the current assertions
    /// is unsatisfiable.  The reported set is the full assumption list — a valid,
    /// though not necessarily minimal, unsatisfiable set (a superset of a minimal
    /// core is still unsatisfiable).  Returns an error S-expression when the last
    /// result was not `unsat`, and `()` when the last check used no assumptions.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn get_unsat_assumptions(&self) -> String {
        if self.last_result != Some(SolverResult::Unsat) {
            return "(error \"unsat assumptions are only available after an unsat check-sat-assuming\")"
                .to_string();
        }
        if self.last_assumptions.is_empty() {
            return "()".to_string();
        }
        let printer = oxiz_core::smtlib::Printer::new(&self.terms);
        let parts: Vec<String> = self
            .last_assumptions
            .iter()
            .map(|&t| printer.print_term(t))
            .collect();
        format!("({})", parts.join(" "))
    }

    /// Get proof (if proof generation is enabled and result is unsat)
    pub fn get_proof(&self) -> String {
        if self.last_result != Some(SolverResult::Unsat) {
            return "(error \"Proof is only available after unsat result\")".to_string();
        }

        match self.solver.get_proof() {
            Some(proof) => proof.format(),
            None => {
                "(error \"Proof generation not enabled. Set :produce-proofs to true\")".to_string()
            }
        }
    }

    /// Get solver statistics
    /// Returns statistics about the last solving run
    pub fn get_statistics(&self) -> String {
        let stats = self.solver.get_statistics();
        format!(
            "(:decisions {} :conflicts {} :propagations {} :restarts {} :learned-clauses {} :theory-propagations {} :theory-conflicts {})",
            stats.decisions,
            stats.conflicts,
            stats.propagations,
            stats.restarts,
            stats.learned_clauses,
            stats.theory_propagations,
            stats.theory_conflicts
        )
    }

    /// Return the raw solver statistics (crate-internal use only).
    #[must_use]
    pub(crate) fn raw_statistics(&self) -> &crate::solver::Statistics {
        self.solver.get_statistics()
    }

    /// Borrow the embedded solver so a crate-internal test can inspect state
    /// that has no place in the public API — currently the SAT clause count,
    /// which `crate::solver::scope_rebase_tests` watches across MBQI rounds.
    ///
    /// Test-only on purpose: exposing the solver publicly would make every
    /// internal field part of the compatibility surface.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn solver(&self) -> &crate::solver::Solver {
        &self.solver
    }

    /// Mutably borrow the embedded solver, for the same crate-internal tests as
    /// [`Context::solver`].
    ///
    /// This exists so a regression pin can reach
    /// [`crate::solver::Solver::forget_cached_verdict`] and force a `check` to
    /// run a real search: the repeated-`check` pins for task #28 have to observe
    /// what the search machinery does on the second and twelfth run, which is
    /// exactly what the verdict cache above it is there to avoid.
    #[cfg(test)]
    pub(crate) fn solver_mut(&mut self) -> &mut crate::solver::Solver {
        &mut self.solver
    }

    /// Return the current solver configuration.
    ///
    /// Callers that build diverse configurations (e.g. an external portfolio
    /// driver) can clone this, mutate the fields they want to vary, and hand it
    /// back via [`Context::set_solver_config`].
    #[must_use]
    pub fn solver_config(&self) -> &crate::solver::SolverConfig {
        self.solver.config()
    }

    /// Replace the entire solver configuration.
    ///
    /// Fields consumed during the solve loop — `timeout_ms`, `max_conflicts`,
    /// `max_decisions`, `theory_mode`, and `simplify` — take effect on the next
    /// `check_sat`.  Fields that the embedded SAT solver only reads at
    /// construction time (notably `restart_strategy` and the inprocessing
    /// toggles) are stored but do not retroactively reconfigure an already-built
    /// SAT engine; vary those before the first solve.
    pub fn set_solver_config(&mut self, config: crate::solver::SolverConfig) {
        self.solver.set_config(config);
    }

    /// Set the wall-clock timeout in milliseconds (`0` disables it).
    pub fn set_timeout_ms(&mut self, timeout_ms: u64) {
        let mut config = self.solver.config().clone();
        config.timeout_ms = timeout_ms;
        self.solver.set_config(config);
    }

    /// Set the maximum number of conflicts before answering `unknown`
    /// (`0` = unlimited).
    pub fn set_max_conflicts(&mut self, max_conflicts: u64) {
        let mut config = self.solver.config().clone();
        config.max_conflicts = max_conflicts;
        self.solver.set_config(config);
    }

    /// Set the maximum number of decisions before answering `unknown`
    /// (`0` = unlimited).
    pub fn set_max_decisions(&mut self, max_decisions: u64) {
        let mut config = self.solver.config().clone();
        config.max_decisions = max_decisions;
        self.solver.set_config(config);
    }

    /// Select the theory propagation eagerness.
    pub fn set_theory_mode(&mut self, mode: crate::solver::TheoryMode) {
        let mut config = self.solver.config().clone();
        config.theory_mode = mode;
        self.solver.set_config(config);
    }

    /// Enable or disable pre-solve simplification of asserted formulas.
    pub fn set_simplify(&mut self, enabled: bool) {
        let mut config = self.solver.config().clone();
        config.simplify = enabled;
        self.solver.set_config(config);
    }

    /// Check satisfiability under temporary assumptions (crate-internal use only).
    pub(crate) fn check_with_assumptions_raw(
        &mut self,
        assumptions: &[oxiz_core::ast::TermId],
    ) -> crate::solver::SolverResult {
        // The same cardinality axiom `check_sat` asserts.  This is the single
        // funnel for every assumption-guarded solve — `(check-sat-assuming ..)`
        // and `(get-consequences ..)` both come through here — so without it
        // those two commands would decide rounding modes over an unconstrained
        // sort while plain `(check-sat)` did not.
        // With recursive definitions in scope this would solve a strictly
        // weaker problem, exactly as a plain `check_sat` would; the driver
        // takes over and threads the assumptions through its rounds.
        if !self.recfun.is_empty() {
            return self.check_sat_recfun_with(assumptions);
        }
        if self.terms.rounding_mode_used() {
            self.assert_rounding_mode_distinctness();
        }
        self.solver
            .check_with_assumptions(assumptions, &mut self.terms)
    }

    /// Return the unsat core from the last check (crate-internal use only).
    #[must_use]
    pub(crate) fn get_unsat_core_raw(&self) -> Option<&crate::solver::UnsatCore> {
        self.solver.get_unsat_core()
    }

    /// Compute the unit consequences of the current assertions together with
    /// `assumptions`, restricted to literals over `variables` (the SMT-LIB /
    /// Z3 `(get-consequences (A..) (V..))` query).
    ///
    /// Model-guided algorithm (mirrors Z3's `solver::get_consequences`):
    ///
    ///   1. Check satisfiability of the assertions under `assumptions`.  If it
    ///      is `unsat`/`unknown`, report exactly that — there is no model to
    ///      seed candidate consequences from.
    ///   2. Otherwise read each queried variable's polarity from the model
    ///      (the candidate consequences).  Every polarity is extracted **up
    ///      front**, because each certification check in step 3 rebuilds and
    ///      overwrites the solver's model.
    ///   3. Certify each candidate literal `lit` by checking whether
    ///      `A ∧ ¬lit` is unsatisfiable; if so `A ⊨ lit`, so `lit` is a genuine
    ///      consequence.
    ///
    /// Returns the output lines: a status (`sat`, or `unknown` when the
    /// closing restore check could not reproduce the model — see
    /// [`consequences_restore_state`]) followed by the Z3-shaped
    /// `((=> (and A) lit) ...)` implication list, or a single
    /// `unsat`/`unknown`/error line.
    #[cfg(feature = "std")]
    fn get_consequences(&mut self, assumptions: &[TermId], variables: &[TermId]) -> Vec<String> {
        let bool_sort = self.terms.sorts.bool_sort;
        // Every assumption and queried variable must be Boolean-sorted.
        let all_bool = assumptions
            .iter()
            .chain(variables.iter())
            .all(|&t| self.terms.get(t).map(|term| term.sort) == Some(bool_sort));
        if !all_bool {
            return vec!["(error \"get-consequences expects Boolean terms\")".to_string()];
        }

        // Base satisfiability under the assumptions.
        match self.check_with_assumptions_raw(assumptions) {
            SolverResult::Unsat => return vec!["unsat".to_string()],
            SolverResult::Unknown => return vec!["unknown".to_string()],
            SolverResult::Sat => {}
        }

        // PHASE 1: extract each variable's model polarity BEFORE any further
        // check (each certification check below overwrites the model).
        let mut candidates: Vec<(TermId, bool)> = Vec::new();
        for &v in variables {
            // Re-fetch the model each iteration; the immutable borrow ends
            // within the statement so `self.terms` can be borrowed mutably by
            // `eval` (disjoint fields — the same pattern as `eval_in_model`).
            let value = match self.solver.model() {
                Some(m) => m.eval(v, &mut self.terms),
                None => continue,
            };
            match self.terms.get(value).map(|t| &t.kind) {
                Some(TermKind::True) => candidates.push((v, true)),
                Some(TermKind::False) => candidates.push((v, false)),
                // Not pinned to a Boolean constant in the model: not a forced
                // consequence, so skip it.
                _ => {}
            }
        }

        // PHASE 2: certify each candidate literal by refuting its negation
        // under the assumptions.
        let mut implied: Vec<TermId> = Vec::new();
        for (v, pol) in candidates {
            let lit = if pol { v } else { self.terms.mk_not(v) };
            let neg = self.terms.mk_not(lit);
            let mut asm2 = assumptions.to_vec();
            asm2.push(neg);
            if self.check_with_assumptions_raw(&asm2) == SolverResult::Unsat {
                implied.push(lit);
            }
        }

        // Restore a consistent `sat` state so a following get-model/get-value
        // reads a model built under exactly `assumptions` (the last
        // certification check left the solver in an arbitrary state).
        //
        // The restore check can fail to reproduce the opening `sat`: a
        // wall-clock timeout or conflict budget consumed by the certification
        // checks above can leave it `unknown`.  Its verdict used to be
        // discarded (`let _ = ...`) and `last_result` forced to `Sat`, which
        // put the context in `sat` mode with no model behind it — a following
        // `(get-model)` then served a stale or absent interpretation while the
        // session claimed `sat`.  Report the restore verdict honestly instead.
        let restore = self.check_with_assumptions_raw(assumptions);
        let (status, cached) = consequences_restore_state(restore);
        match cached {
            Some(result) => {
                self.last_result = Some(result);
                self.last_assumptions = assumptions.to_vec();
            }
            None => self.invalidate_last_check(),
        }

        // Build the Z3-shaped implication list.  The antecedent is `(and A)`
        // rendered directly — NOT via `mk_implies`, which simplifies away the
        // antecedent for an empty/singleton assumption set.
        let ante = self.terms.mk_and(assumptions.iter().copied());
        let printer = oxiz_core::smtlib::Printer::new(&self.terms);
        let ante_str = printer.print_term(ante);
        let impls: Vec<String> = implied
            .iter()
            .map(|&lit| format!("(=> {} {})", ante_str, printer.print_term(lit)))
            .collect();

        // The implication list survives a degraded `status`: every literal in
        // it was certified by its own `unsat` refutation above, independently
        // of whether the closing restore check reproduced the model.
        vec![status.to_string(), format!("({})", impls.join(" "))]
    }

    /// Execute an SMT-LIB2 script
    ///
    /// # Errors
    ///
    /// Returns an error when the script fails to parse, or when a
    /// declaration-shaped command (`declare-const`, `declare-fun`,
    /// `declare-sort`, `define-sort`, `define-fun`) carries a malformed or
    /// unsupported sort expression.  The latter used to resolve silently to
    /// `Bool`, mis-sorting the declared symbol and corrupting every
    /// model/value answer that mentioned it.
    #[cfg(feature = "std")]
    pub fn execute_script(&mut self, script: &str) -> Result<Vec<String>> {
        let commands = parse_script(script, &mut self.terms)?;
        let mut output = Vec::new();

        for cmd in commands {
            // A command that produces its own SMT-LIB response must not
            // additionally emit the `:print-success` acknowledgement.  Compute
            // this before `cmd` is consumed by the match below (the `matches!`
            // patterns bind nothing, so `cmd` is not moved).
            let emits_own_response = matches!(
                cmd,
                Command::CheckSat
                    | Command::CheckSatAssuming(_)
                    | Command::GetConsequences(_, _)
                    | Command::GetModel
                    | Command::GetAssertions
                    | Command::GetAssignment
                    | Command::GetProof
                    | Command::GetOption(_)
                    | Command::GetUnsatCore
                    | Command::GetUnsatAssumptions
                    | Command::GetValue(_)
                    | Command::GetInfo(_)
                    | Command::Echo(_)
                    | Command::Simplify(_)
            );
            let output_len_before = output.len();
            match cmd {
                Command::SetLogic(logic) => {
                    self.set_logic(&logic);
                }
                Command::DeclareConst(name, sort_name) => {
                    let sort = self.parse_sort_name(&sort_name)?;
                    self.declare_const(&name, sort);
                }
                Command::DeclareFun(name, arg_sorts, ret_sort) => {
                    // Treat nullary functions as constants
                    if arg_sorts.is_empty() {
                        let sort = self.parse_sort_name(&ret_sort)?;
                        self.declare_const(&name, sort);
                    } else {
                        // Parse argument sorts and return sort
                        let parsed_arg_sorts: Vec<SortId> = arg_sorts
                            .iter()
                            .map(|s| self.parse_sort_name(s))
                            .collect::<Result<_>>()?;
                        let parsed_ret_sort = self.parse_sort_name(&ret_sort)?;
                        self.declare_fun(&name, parsed_arg_sorts, parsed_ret_sort);
                    }
                }
                Command::Assert(term) => {
                    self.assert(term);
                }
                Command::AssertNamed(term, name) => {
                    // Register the assertion under its `:named` label so that,
                    // with `:produce-unsat-cores` enabled, `(get-unsat-core)`
                    // reports the user label when this assertion participates
                    // in an `unsat` refutation.
                    self.assert_named(term, &name);
                }
                Command::CheckSat => {
                    let result = self.check_sat();
                    output.push(match result {
                        SolverResult::Sat => "sat".to_string(),
                        SolverResult::Unsat => "unsat".to_string(),
                        SolverResult::Unknown => "unknown".to_string(),
                    });
                }
                Command::Push(n) => {
                    for _ in 0..n {
                        self.push();
                    }
                }
                Command::Pop(n) => {
                    for _ in 0..n {
                        self.pop();
                    }
                }
                Command::Reset => {
                    self.reset();
                }
                Command::ResetAssertions => {
                    self.reset_assertions();
                }
                Command::Exit => {
                    // Per SMT-LIB, a successful `exit` is acknowledged before
                    // the interpreter terminates.
                    if self.print_success_enabled() {
                        output.push("success".to_string());
                    }
                    break;
                }
                Command::Echo(msg) => {
                    output.push(msg);
                }
                Command::GetModel => {
                    output.push(self.format_model());
                }
                Command::GetAssertions => {
                    output.push(self.format_assertions());
                }
                Command::GetAssignment => {
                    output.push(self.get_assignment());
                }
                Command::GetProof => {
                    output.push(self.get_proof());
                }
                Command::GetOption(key) => {
                    output.push(self.format_option(&key));
                }
                Command::SetOption(key, value) => {
                    self.set_option(&key, &value);
                }
                Command::CheckSatAssuming(assumptions) => {
                    // Check under temporary assumptions WITHOUT push/assert/pop.
                    // A pop() would discard the model / unsat core built by the
                    // check, leaving `last_result == Sat` but no state for a
                    // following `(get-value ...)` / `(get-model)` to read.
                    // `check_with_assumptions` keeps the solver state produced by
                    // the assumption-guarded solve, so post-check queries observe
                    // the correct model.
                    self.last_assumptions = assumptions.clone();
                    let mut result = self.check_with_assumptions_raw(&assumptions);
                    // Same array soundness honesty gate as `check_sat`.
                    if result == SolverResult::Sat
                        && self.solver.array_atoms_need_theory(&self.terms)
                    {
                        result = SolverResult::Unknown;
                    }
                    self.last_result = Some(result);
                    output.push(match result {
                        SolverResult::Sat => "sat".to_string(),
                        SolverResult::Unsat => "unsat".to_string(),
                        SolverResult::Unknown => "unknown".to_string(),
                    });
                }
                Command::GetConsequences(assumptions, variables) => {
                    let out = self.get_consequences(&assumptions, &variables);
                    output.extend(out);
                }
                Command::Simplify(term) => {
                    // Simplify and output the term
                    let simplified = self.terms.simplify(term);
                    let printer = oxiz_core::smtlib::Printer::new(&self.terms);
                    output.push(printer.print_term(simplified));
                }
                Command::GetUnsatCore => {
                    // SMT-LIB 2.6 §4.1.1: an unsat core exists only in `unsat`
                    // mode.  Without this gate a core computed before a `pop` /
                    // `assert` was reported against the *current* assertion
                    // list, whose indices no longer match — `minimize_unsat_core`
                    // indexes `Solver::assertions` by the stale core indices and
                    // panicked outright on a `(push)(assert)(check-sat)(pop)
                    // (get-unsat-core)` script.
                    //
                    // Minimize the conservative core (greedy deletion-based) so
                    // the reported set contains only assertions actually needed
                    // for the refutation, not every named assertion.  Fall back
                    // to the raw core when minimization is unavailable
                    // (unsat-core production disabled).
                    let core = if self.last_result == Some(SolverResult::Unsat) {
                        self.solver
                            .minimize_unsat_core(&mut self.terms)
                            .or_else(|| self.solver.get_unsat_core().cloned())
                    } else {
                        None
                    };
                    match core {
                        Some(core) if !core.names.is_empty() => {
                            output.push(format!("({})", core.names.join(" ")));
                        }
                        Some(_) => output.push("()".to_string()),
                        None => {
                            output.push("(error \"No unsat core available\")".to_string());
                        }
                    }
                }
                Command::GetUnsatAssumptions => {
                    // Report the failed assumptions from the most recent
                    // `check-sat-assuming` that returned `unsat`.  The printer
                    // used by `get_unsat_assumptions` is `std`-only, so under
                    // `no_std` we answer with an honest error S-expression
                    // rather than silently emitting nothing.
                    #[cfg(feature = "std")]
                    {
                        output.push(self.get_unsat_assumptions());
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        output.push(
                            "(error \"get-unsat-assumptions requires the std feature\")"
                                .to_string(),
                        );
                    }
                }
                Command::GetValue(terms) => {
                    output.push(self.format_get_value(&terms));
                }
                Command::GetInfo(keyword) => {
                    output.push(self.get_info(&keyword));
                }
                Command::SetInfo(_, _) => {
                    // Purely descriptive metadata (`:source`, `:license`,
                    // ...); it has no effect on declarations or solving.
                }
                Command::DeclareSort(name, arity) => {
                    if arity == 0 {
                        // Eagerly materialize the sort so `declared_sort_names`
                        // reflects it immediately, matching what the parser
                        // already did (lazily, on first reference) internally.
                        self.parse_sort_name(&name)?;
                    }
                    // Arity > 0 parametric sorts are recorded for
                    // introspection; applying them with type arguments is
                    // not yet supported anywhere in this crate, matching
                    // the parser's own documented limitation.
                    self.declared_sorts.insert(name, arity);
                }
                Command::DefineSort(name, params, sort_expr) => {
                    if params.is_empty() {
                        let resolved = self.parse_sort_name(&sort_expr)?;
                        self.terms.sorts.define_alias(&name, resolved);
                    }
                    // Parametric aliases (non-empty `params`) are not
                    // resolved: the SMT-LIB parser itself only substitutes
                    // 0-arity `define-sort` aliases in-script (see
                    // `oxiz_core`'s `Parser::parse_sort_name`), so there is
                    // no sound target to register here either.
                }
                Command::DefineFun(name, params, ret_sort, body) => {
                    let sort = self.parse_sort_name(&ret_sort)?;
                    if params.is_empty() {
                        // The parser already inlined every in-script
                        // reference to `name` directly as `body` (see
                        // `oxiz_core`'s `define-fun` handling), so this
                        // doesn't change what gets solved. Declaring a real
                        // constant provably equal to `body` -- rather than
                        // doing nothing -- makes `name` show up correctly
                        // (with its actual value) in `get-model`/`get-value`
                        // output instead of silently vanishing, without
                        // introducing any constraint that could change
                        // satisfiability (the equality is trivially
                        // satisfiable for any assignment to `body`'s free
                        // variables).
                        let const_term = self.declare_const(&name, sort);
                        let eq = self.terms.mk_eq(const_term, body);
                        self.assert(eq);
                    } else {
                        // Functions with parameters are macros: call sites
                        // are meant to be substituted with `body` at parse
                        // time (see `oxiz_core`'s defined-function handling
                        // in `smtlib/parser/terms.rs`), which is outside
                        // this file's ownership -- so no further wiring for
                        // *solving* belongs here. Still register the
                        // signature so introspection (`get_fun_signature`,
                        // `declared_function_names`) reflects the
                        // definition, like `declare-fun` does.
                        let arg_sorts: Vec<SortId> = params
                            .iter()
                            .map(|(_, sort_name)| self.parse_sort_name(sort_name))
                            .collect::<Result<_>>()?;
                        self.declare_fun(&name, arg_sorts, sort);
                    }
                }
                Command::DeclareDatatype { name, .. } => {
                    self.register_datatype_functions(&name);
                }
                Command::DefineFunsRec(defs) => {
                    self.define_funs_rec(defs)?;
                }
            }

            // Emit the `:print-success` acknowledgement for a command that
            // succeeded *silently* — no response of its own and no error it
            // pushed (a pushed error is left as the command's response).  `exit`
            // handles its own acknowledgement before breaking out of the loop.
            if self.print_success_enabled()
                && !emits_own_response
                && output.len() == output_len_before
            {
                output.push("success".to_string());
            }
        }

        Ok(output)
    }

    /// Get solver statistics
    #[must_use]
    pub fn stats(&self) -> &oxiz_sat::SolverStats {
        self.solver.stats()
    }
}

/// Decide what `(get-consequences ...)` reports, and what verdict the context
/// may keep cached, from the verdict of its closing *restore* check.
///
/// After certifying the consequences (each by its own `unsat` refutation), the
/// query re-checks the original assumptions so that a following `(get-model)` /
/// `(get-value)` reads a model built under exactly those assumptions.  That
/// re-check is a full solve and can therefore fail to reproduce the opening
/// `sat`, e.g. when the certification checks have exhausted the wall-clock
/// budget or the conflict limit.
///
/// Returns the status line to emit, and the verdict to cache in
/// `Context::last_result` (`None` = cache nothing, i.e. leave `assert` mode, so
/// the model queries answer "not available" instead of serving whatever the
/// last certification check happened to leave behind).
///
/// * `Sat` — the model is back; stay in `sat` mode and report `sat`.
/// * `Unknown` — no certified model; report `unknown`, cache nothing.
/// * `Unsat` — contradicts the opening `sat` of the same assumption set, so
///   the two checks disagree and neither may be published as a verdict.  That
///   is precisely `unknown`, and nothing is cached.
#[cfg(feature = "std")]
fn consequences_restore_state(restore: SolverResult) -> (&'static str, Option<SolverResult>) {
    match restore {
        SolverResult::Sat => ("sat", Some(SolverResult::Sat)),
        SolverResult::Unsat | SolverResult::Unknown => ("unknown", None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `get_info`'s catch-all arm used to interpolate the (attacker- or
    // merely awkward-input-controlled) keyword straight into
    // `(error "unsupported info keyword: :{key}")` with no escaping, so a
    // `"` in the keyword ended the string literal early and corrupted the
    // rest of the SMT-LIB response. It now routes the whole message through
    // `format_string_literal`, exactly like every other SMT-LIB string
    // value in the workspace.

    /// A keyword containing a `"`, a `\`, a `\u`-prefixed literal
    /// substring, a non-ASCII code point, and a control character must all
    /// come back as a single, well-formed, properly escaped SMT-LIB string
    /// literal -- never a bare, unescaped quote that ends the literal
    /// early.
    #[test]
    fn test_get_info_unknown_keyword_with_special_chars_is_escaped() {
        for keyword in [
            "weird\"keyword",
            "weird\\keyword",
            "weird\\u0041keyword",
            "weird\u{e9}keyword",
            "weird\u{0}keyword",
        ] {
            let ctx = Context::new();
            let response = ctx.get_info(keyword);
            assert!(
                response.starts_with("(error "),
                "expected an `(error ...)` response, got {response}"
            );
            // The response must be exactly one well-formed SMT-LIB string
            // literal: scanning past the first `"`, the first *unescaped*
            // `"` reached must be the very last character before the
            // closing `)`.
            let open = response.find('"').expect("response contains a quote");
            let body = &response[open + 1..];
            let mut chars = body.char_indices();
            let mut closed_at = None;
            while let Some((i, c)) = chars.next() {
                if c == '"' {
                    // SMT-LIB doubles an embedded quote (`""`); only a
                    // *single* trailing quote is the real terminator.
                    if body[i + 1..].starts_with('"') {
                        chars.next();
                        continue;
                    }
                    closed_at = Some(i);
                    break;
                }
            }
            let closed_at = closed_at.expect("literal must have an unescaped closing quote");
            assert_eq!(
                &body[closed_at..],
                "\")",
                "the closing quote must be the last thing before `)` in {response}"
            );
        }
    }

    /// Control: a keyword with no special characters renders exactly as
    /// before, with no gratuitous escaping.
    #[test]
    fn test_get_info_unknown_keyword_plain_ascii_unchanged() {
        let ctx = Context::new();
        assert_eq!(
            ctx.get_info("totally-unknown-keyword"),
            "(error \"unsupported info keyword: :totally-unknown-keyword\")"
        );
    }

    #[test]
    fn test_context_basic() {
        let mut ctx = Context::new();

        ctx.set_logic("QF_UF");
        assert_eq!(ctx.logic(), Some("QF_UF"));

        let t = ctx.terms.mk_true();
        ctx.assert(t);

        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_context_push_pop() {
        let mut ctx = Context::new();

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        ctx.push();

        let f = ctx.terms.mk_false();
        ctx.assert(f);

        // Should be unsat with false asserted
        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Unsat);

        ctx.pop();

        // After pop, should be sat again
        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_execute_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (check-sat)
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output, vec!["sat"]);
    }

    #[test]
    fn test_declare_const() {
        let mut ctx = Context::new();

        let bool_sort = ctx.terms.sorts.bool_sort;
        let int_sort = ctx.terms.sorts.int_sort;

        ctx.declare_const("x", bool_sort);
        ctx.declare_const("y", int_sort);

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Sat);

        // Model should include both constants
        let model = ctx.get_model();
        assert!(model.is_some());
        let model = model.expect("test operation should succeed");
        assert_eq!(model.len(), 2);
    }

    #[test]
    fn test_format_model() {
        let mut ctx = Context::new();

        let bool_sort = ctx.terms.sorts.bool_sort;
        ctx.declare_const("p", bool_sort);

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        let _ = ctx.check_sat();

        let model_str = ctx.format_model();
        assert!(model_str.contains("(model"));
        assert!(model_str.contains("define-fun p () Bool"));
    }

    #[test]
    fn test_get_model_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Bool)
            (assert true)
            (check-sat)
            (get-model)
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "sat");
        assert!(
            output[1].contains("(model"),
            "Expected '(model' in: {}",
            output[1]
        );
        // Note: Sorts may not always appear in model output if values are default
        // The model format is: (define-fun name () Sort value)
    }

    #[test]
    fn test_push_pop_consts() {
        let mut ctx = Context::new();

        let bool_sort = ctx.terms.sorts.bool_sort;
        ctx.declare_const("a", bool_sort);
        ctx.push();
        ctx.declare_const("b", bool_sort);

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        let _ = ctx.check_sat();

        let model = ctx.get_model().expect("test operation should succeed");
        assert_eq!(model.len(), 2);

        ctx.pop();
        let _ = ctx.check_sat();

        let model = ctx.get_model().expect("test operation should succeed");
        assert_eq!(model.len(), 1);
        assert_eq!(model[0].0, "a");
    }

    #[test]
    fn test_get_assertions() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (assert (not p))
            (get-assertions)
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 1);
        assert!(output[0].starts_with('('));
        // Should contain both assertions
        assert!(output[0].contains("p"));
    }

    #[test]
    fn test_check_sat_assuming_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (declare-const q Bool)
            (assert p)
            (check-sat-assuming (q))
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], "sat");
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_get_unsat_assumptions_script() {
        // Regression: `(get-unsat-assumptions)` must be reachable from the
        // SMT-LIB command path (previously the parser rejected it outright).
        // After an `unsat` `check-sat-assuming`, it reports the failed
        // assumptions.
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (check-sat-assuming ((not p)))
            (get-unsat-assumptions)
        "#;

        let output = ctx
            .execute_script(script)
            .expect("script with get-unsat-assumptions should parse and run");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "unsat");
        // The reported set is a non-empty unsatisfiable subset of the
        // assumptions, mentioning the failed literal `p`.
        assert!(output[1].starts_with('('));
        assert!(output[1].contains('p'), "got: {}", output[1]);
        assert_ne!(output[1], "()");
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_get_unsat_assumptions_no_assumptions_is_empty() {
        // A plain (unsat) `check-sat` used no assumptions, so
        // `(get-unsat-assumptions)` reports the empty set rather than an error.
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (assert (not p))
            (check-sat)
            (get-unsat-assumptions)
        "#;

        let output = ctx
            .execute_script(script)
            .expect("script should parse and run");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "unsat");
        assert_eq!(output[1], "()");
    }

    #[test]
    fn test_get_option_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-option :produce-models true)
            (get-option :produce-models)
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], "true");
    }

    #[test]
    fn test_random_seed_option_is_enforced_and_recorded() {
        // Regression: `:random-seed` used to be a documented no-op ("recorded
        // but not enforced").  It is now threaded into the SAT engine's phase
        // PRNG via Solver::set_random_seed.  The observable contract here is
        // two-fold: (1) `(get-option :random-seed)` reflects exactly the value
        // the user set (recording preserved), and (2) setting a seed keeps the
        // sat/unsat verdict sound — seeding must never change a decidable
        // answer.  A previous silent no-op would still pass (1); the point of
        // this test is that the plumbing is now wired without regressing (2).
        let mut ctx = Context::new();

        ctx.set_option(":random-seed", "42");
        assert_eq!(ctx.get_option("random-seed"), Some("42"));

        // A satisfiable BV problem must still be SAT under a non-default seed.
        let script = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (bvult x #x0a))
            (check-sat)
        "#;
        let output = ctx
            .execute_script(script)
            .expect("seeded script should parse and run");
        assert_eq!(output, vec!["sat"]);
    }

    #[test]
    fn test_random_seed_zero_and_malformed_are_safe() {
        // Seed `0` is the degenerate xorshift fixed point; the seed-mixing must
        // map it to the historical default rather than freezing the PRNG.  A
        // malformed seed must not corrupt the RNG (it is still recorded so
        // get-option is faithful), and neither must panic.
        let mut ctx = Context::new();

        ctx.set_option(":random-seed", "0");
        assert_eq!(ctx.get_option("random-seed"), Some("0"));

        ctx.set_option(":random-seed", "not-a-number");
        assert_eq!(ctx.get_option("random-seed"), Some("not-a-number"));

        // Solving remains correct after both.
        let script = r#"
            (set-logic QF_LIA)
            (declare-const y Int)
            (assert (> y 5))
            (assert (< y 8))
            (check-sat)
        "#;
        let output = ctx
            .execute_script(script)
            .expect("script should parse and run");
        assert_eq!(output, vec!["sat"]);
    }

    #[test]
    fn test_reset_assertions() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (reset-assertions)
            (get-assertions)
            (check-sat)
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "()"); // No assertions after reset
        assert_eq!(output[1], "sat"); // Empty formula is SAT
    }

    #[test]
    fn test_simplify_command() {
        let mut ctx = Context::new();

        let script = r#"
            (simplify (+ 1 2))
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 1);
        // Should simplify to 3
        assert_eq!(output[0], "3");
    }

    #[test]
    fn test_simplify_complex() {
        let mut ctx = Context::new();

        let script = r#"
            (simplify (* 2 3 4))
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 1);
        // Should simplify to 24
        assert_eq!(output[0], "24");
    }

    #[test]
    fn test_get_value() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (declare-const q Bool)
            (assert p)
            (assert (not q))
            (check-sat)
            (get-value (p q (and p q) (or p q)))
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "sat");

        // Parse the get-value output
        let value_output = &output[1];
        assert!(value_output.contains("p"));
        assert!(value_output.contains("q"));
        // p should evaluate to true
        assert!(value_output.contains("true"));
        // q should evaluate to false
        assert!(value_output.contains("false"));
    }

    #[test]
    fn test_get_value_no_model() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (get-value (p))
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 1);
        assert!(output[0].contains("error") || output[0].contains("No model"));
    }

    #[test]
    fn test_get_value_after_unsat() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (assert (not p))
            (check-sat)
            (get-value (p))
        "#;

        let output = ctx
            .execute_script(script)
            .expect("test operation should succeed");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "unsat");
        assert!(output[1].contains("error") || output[1].contains("No model"));
    }
}
