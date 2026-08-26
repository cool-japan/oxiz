//! Optimization, MaxSMT, Craig interpolation, and quantifier elimination.
//!
//! # Why `minimize`/`maximize`/`assertSoft` need a side-table
//!
//! `Command` (`oxiz_core::smtlib::Command`) has no `Minimize`/`Maximize`/
//! `AssertSoft` variant, so those SMT-LIB extension commands cannot be routed
//! through [`oxiz_solver::Context::execute_script`] at all -- an unrecognized
//! command is silently balanced-paren-skipped by the parser (lenient
//! interoperability for genuinely vendor-specific commands), which previously
//! made `minimize()`/`maximize()`/`assertSoft()` complete successfully while
//! doing *nothing*, and `optimize()` label any plain `sat` result `"optimal"`
//! regardless of whether anything was actually optimized.
//!
//! The real fix wires these through [`oxiz_solver::Optimizer`] (a genuine
//! lexicographic-objective / lower-bound-search optimizer that already lives
//! in `oxiz-solver`, already a dependency of this crate) instead. That
//! requires accumulating objectives/soft-constraints across separate
//! `minimize`/`maximize`/`assertSoft` calls, ending at a later `optimize()`
//! call -- state that would naturally live on `WasmSolver` (`lib.rs`) or
//! `oxiz_solver::Context`, except this package's owned files are only
//! `oxiz-wasm/src/js_api/*` and `oxiz-wasm/tests/*`. `Context` already
//! exposes a generic, per-instance `set_option`/`get_option(key) ->
//! Option<&str>` string store (used elsewhere for real SMT-LIB options such
//! as `produce-proofs`); this reuses it as a namespaced (`__oxiz_wasm_*`
//! keys, which cannot collide with any real SMT-LIB option name) key-value
//! side-channel, entirely through `Context`'s already-public API and
//! confined to this file. `Context::reset()` clears `options` (and hence
//! this side-table) along with declarations and assertions -- the
//! semantically correct behavior for a full reset -- while
//! `reset_assertions()`/`push()`/`pop()` intentionally leave `options`
//! alone, so (like real solver options) objectives/soft-constraints and the
//! declared-symbol-sort table currently survive a `push`/`pop`/
//! `resetAssertions`. This is a known, documented deviation from strictly
//! push/pop-scoped objectives.
//!
//! A second, related gap this works around: parsing an objective/soft
//! formula in isolation needs to resolve previously-declared symbols (e.g.
//! `x` in `minimize("(+ x y)")`) to their *true* declared sort, not a fresh
//! `Bool`-sorted placeholder. `oxiz_core::smtlib::Parser::with_context` (the
//! "parser seeding" API) exists for exactly this, but `Parser` itself is not
//! re-exported from `oxiz_core::smtlib` (only `Command`/`parse_script`/
//! `parse_term` are), so it is unreachable from this crate. Instead,
//! `WasmSolver::parse_objective_term` rebuilds a `(declare-const ...)`
//! prefix from the side-table and feeds `{prefix}(simplify <formula>)` to
//! the *public* [`oxiz_core::smtlib::parse_script`] in one call: since that
//! function's single `Parser` instance accumulates declarations as it walks
//! the script, and `TermManager::mk_var` hash-conses `Var` terms by
//! `(name, sort)`, this resolves each symbol to the exact same `TermId`
//! already used elsewhere in this context -- without needing `Parser`
//! itself to be exported.

use crate::WasmSolver;
use crate::string_utils;
use crate::{WasmError, WasmErrorKind};
use oxiz_core::ast::TermId;
use oxiz_core::smtlib::{Command, Printer, parse_script, parse_term};
use wasm_bindgen::prelude::*;

/// Newline-joined list of every symbol name ever declared through
/// `declareConst`/`declareFun` (0-ary), keyed into [`oxiz_solver::Context`]'s
/// generic option store (see the module-level docs above).
const DECL_NAMES_KEY: &str = "__oxiz_wasm_decl_names";
/// Prefix for the per-symbol sort-name entry: `{DECL_SORT_PREFIX}{name}` ->
/// the sort name string (e.g. `"Int"`) originally passed to `declareConst`.
const DECL_SORT_PREFIX: &str = "__oxiz_wasm_decl_sort::";
/// Count of registered `minimize`/`maximize` objectives.
const OBJ_COUNT_KEY: &str = "__oxiz_wasm_obj_count";
/// Prefix for objective entry `{OBJ_PREFIX}{i}` -> `"min:{term_id}"` /
/// `"max:{term_id}"`, in registration order.
const OBJ_PREFIX: &str = "__oxiz_wasm_obj::";
/// Count of registered `assertSoft` soft constraints.
const SOFT_COUNT_KEY: &str = "__oxiz_wasm_soft_count";
/// Prefix for soft-constraint entry `{SOFT_PREFIX}{i}` ->
/// `"{term_id}:{weight}"`.
const SOFT_PREFIX: &str = "__oxiz_wasm_soft::";

#[wasm_bindgen]
impl WasmSolver {
    /// Add a minimization objective
    ///
    /// Adds an objective function to minimize. The formula must evaluate to an
    /// integer or real value. When `optimize()` is called, the solver will find
    /// a model that minimizes this objective while satisfying all assertions.
    ///
    /// For multiple objectives, they are optimized lexicographically in the order
    /// they were added (first objective has highest priority). If any
    /// `assertSoft()` soft constraints have also been added, the combined
    /// MaxSMT penalty is treated as the highest-priority objective (see
    /// `optimize()`), with `minimize`/`maximize` objectives breaking ties
    /// among remaining solutions.
    ///
    /// # Parameters
    ///
    /// * `formula` - An SMT-LIB2 arithmetic expression to minimize (must be Int or Real type)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The formula is empty or malformed
    /// - The formula contains syntax errors
    /// - The formula references symbols not declared via `declareConst`/`declareFun`
    /// - The formula is not Int- or Real-sorted
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("y", "Int");
    /// solver.assertFormula("(> x 0)");
    /// solver.assertFormula("(> y 0)");
    /// // Minimize x + y
    /// solver.minimize("(+ x y)");
    /// const result = solver.optimize();
    /// console.log(result); // { status: "optimal", value: "2", model: {...} }
    /// ```
    #[wasm_bindgen]
    pub fn minimize(&mut self, formula: &str) -> Result<(), JsValue> {
        if string_utils::is_effectively_empty(formula) {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Objective formula cannot be empty",
            )
            .into());
        }

        let term = self.parse_objective_term(formula)?;
        self.require_arith_sort(term, "minimize")?;
        self.push_objective(term, false);
        Ok(())
    }

    /// Add a maximization objective
    ///
    /// Adds an objective function to maximize. The formula must evaluate to an
    /// integer or real value. When `optimize()` is called, the solver will find
    /// a model that maximizes this objective while satisfying all assertions.
    ///
    /// For multiple objectives, they are optimized lexicographically in the order
    /// they were added (first objective has highest priority). Objectives are
    /// wired to the real `oxiz_solver::Optimizer` — they are never silently
    /// dropped.
    ///
    /// # Parameters
    ///
    /// * `formula` - An SMT-LIB2 arithmetic expression to maximize (must be Int or Real type)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The formula is empty or malformed
    /// - The formula contains syntax errors
    /// - The formula references symbols not declared via `declareConst`/`declareFun`
    /// - The formula is not Int- or Real-sorted
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("y", "Int");
    /// solver.assertFormula("(< x 10)");
    /// solver.assertFormula("(< y 10)");
    /// // Maximize x + y
    /// solver.maximize("(+ x y)");
    /// const result = solver.optimize();
    /// console.log(result); // { status: "optimal", value: "18", model: {...} }
    /// ```
    #[wasm_bindgen]
    pub fn maximize(&mut self, formula: &str) -> Result<(), JsValue> {
        if string_utils::is_effectively_empty(formula) {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Objective formula cannot be empty",
            )
            .into());
        }

        let term = self.parse_objective_term(formula)?;
        self.require_arith_sort(term, "maximize")?;
        self.push_objective(term, true);
        Ok(())
    }

    /// Run optimization on the current assertions and objectives
    ///
    /// Solves the optimization problem defined by the current assertions,
    /// `minimize`/`maximize` objectives, and `assertSoft` soft constraints
    /// via [`oxiz_solver::Optimizer`], a real lexicographic-objective
    /// optimizer -- not merely a plain `check-sat` mislabeled `"optimal"`.
    ///
    /// # Returns
    ///
    /// A JavaScript object with the following structure:
    /// ```javascript
    /// {
    ///   status: "optimal" | "unbounded" | "unsat" | "unknown",
    ///   value: "42",           // The optimal value (if optimal and objectives were set)
    ///   model: { x: {...} }    // The satisfying model (if optimal)
    /// }
    /// ```
    ///
    /// With no objectives and no soft constraints at all, this degrades to
    /// a plain satisfiability check (`"optimal"` for a vacuous, unconstrained
    /// "optimum" over a satisfiable problem, `"unsat"` otherwise) -- the same
    /// convention `oxiz_solver::Optimizer::optimize` itself uses.
    ///
    /// # A note on `getMinimalModel()`/`getModel()` after `optimize()`
    ///
    /// `Optimizer` solves against its own freshly-built internal solver (see
    /// `oxiz_solver::optimization::Optimizer::build_solver`), never touching
    /// `self.ctx`'s own solver/last-check-sat state. Read the model from
    /// this method's own `.model` field; `getMinimalModel()`/`getModel()`
    /// reflect only a plain `checkSat()` on `self.ctx`; they will honestly
    /// report "no model" if called after `optimize()` without an
    /// intervening `checkSat()`, rather than returning a stale or
    /// mismatched model.
    ///
    /// # Errors
    ///
    /// Returns an error only if building the result object itself fails
    /// (e.g. an internal `js_sys::Reflect` failure); an unsatisfiable or
    /// unbounded problem is reported via `status`, not as a JS exception.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("y", "Int");
    /// solver.assertFormula("(>= x 0)");
    /// solver.assertFormula("(>= y 0)");
    /// solver.assertFormula("(<= (+ x y) 10)");
    /// solver.maximize("(+ (* 3 x) (* 2 y))"); // Maximize 3x + 2y
    ///
    /// const result = solver.optimize();
    /// if (result.status === "optimal") {
    ///   console.log("Optimal value:", result.value);
    ///   console.log("x =", result.model.x.value);
    ///   console.log("y =", result.model.y.value);
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn optimize(&mut self) -> Result<JsValue, JsValue> {
        let hard_assertions: Vec<TermId> = self.ctx.get_assertions().to_vec();
        let logic = self.ctx.logic().map(str::to_string);
        let soft_terms = self.read_soft_constraints();
        let objectives = self.read_objectives();

        let mut opt = oxiz_solver::Optimizer::new();
        if let Some(logic) = logic.as_deref() {
            opt.set_logic(logic);
        }
        for term in &hard_assertions {
            opt.assert(*term);
        }

        if !soft_terms.is_empty() {
            // MaxSMT via a standard relaxation-variable encoding: for each
            // soft constraint `f_i` with weight `w_i`, a fresh Boolean
            // `relax_i` "buys" a way out (`(or f_i relax_i)` is a *hard*
            // constraint on the optimizer's own solver), and the objective
            // becomes minimizing the total weight of constraints that had
            // to be relaxed (`sum(ite(relax_i, w_i, 0))`).
            let mut cost_terms: Vec<TermId> = Vec::with_capacity(soft_terms.len());
            for (idx, (term, weight)) in soft_terms.iter().enumerate() {
                let bool_sort = self.ctx.terms.sorts.bool_sort;
                let relax_var = self
                    .ctx
                    .terms
                    .mk_var(&format!("__oxiz_wasm_soft_relax_{idx}"), bool_sort);
                let relaxed = self.ctx.terms.mk_or([*term, relax_var]);
                opt.assert(relaxed);

                let weight_term = self.ctx.terms.mk_int(*weight);
                let zero_term = self.ctx.terms.mk_int(0u64);
                cost_terms.push(self.ctx.terms.mk_ite(relax_var, weight_term, zero_term));
            }
            let total_cost = self.ctx.terms.mk_add(cost_terms);
            // The MaxSMT penalty is the primary objective -- satisfying soft
            // constraints (weighted by importance) takes lexicographic
            // precedence over any explicit `minimize`/`maximize` objectives,
            // which then break remaining ties.
            opt.minimize(total_cost);
        }

        for (is_max, term) in &objectives {
            if *is_max {
                opt.maximize(*term);
            } else {
                opt.minimize(*term);
            }
        }

        let opt_result = opt.optimize(&mut self.ctx.terms);
        let out = js_sys::Object::new();

        match opt_result {
            oxiz_solver::OptimizationResult::Optimal { value, model } => {
                self.last_result = Some("sat".to_string());
                js_sys::Reflect::set(&out, &"status".into(), &"optimal".into())
                    .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set status"))?;

                let value_str = Printer::new(&self.ctx.terms).print_term(value);
                js_sys::Reflect::set(&out, &"value".into(), &value_str.into())
                    .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set value"))?;

                match self.build_optimizer_model_object(&model) {
                    Ok(model_obj) => {
                        js_sys::Reflect::set(&out, &"model".into(), &model_obj).map_err(|_| {
                            WasmError::new(WasmErrorKind::Unknown, "Failed to set model")
                        })?;
                    }
                    Err(_) => {
                        let _ = js_sys::Reflect::set(&out, &"model".into(), &JsValue::NULL);
                    }
                }
            }
            oxiz_solver::OptimizationResult::Unsat => {
                self.last_result = Some("unsat".to_string());
                js_sys::Reflect::set(&out, &"status".into(), &"unsat".into())
                    .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set status"))?;
            }
            oxiz_solver::OptimizationResult::Unbounded => {
                js_sys::Reflect::set(&out, &"status".into(), &"unbounded".into())
                    .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set status"))?;
            }
            oxiz_solver::OptimizationResult::Unknown => {
                self.last_result = Some("unknown".to_string());
                js_sys::Reflect::set(&out, &"status".into(), &"unknown".into())
                    .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set status"))?;
            }
        }

        Ok(out.into())
    }

    /// Add a soft constraint with weight (for MaxSMT)
    ///
    /// Soft constraints are constraints that the solver will try to satisfy,
    /// but may violate if necessary. Each soft constraint has a weight indicating
    /// its importance. The solver minimizes the total weight of violated constraints.
    ///
    /// This enables MaxSMT (Maximum Satisfiability Modulo Theories) solving, where
    /// you want to satisfy as many constraints as possible (weighted by importance).
    /// Implemented via a relaxation-variable encoding over
    /// [`oxiz_solver::Optimizer`] -- see the `optimize()` docs for the exact
    /// construction.
    ///
    /// # Parameters
    ///
    /// * `formula` - An SMT-LIB2 boolean formula (soft constraint)
    /// * `weight` - The weight/importance of this constraint (positive integer)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The formula is empty or malformed
    /// - The weight is not a positive integer
    /// - The formula contains syntax errors, or references undeclared symbols
    /// - The formula is not Bool-sorted
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    /// solver.declareConst("p", "Bool");
    /// solver.declareConst("q", "Bool");
    /// solver.declareConst("r", "Bool");
    ///
    /// // Hard constraint: at least one must be true
    /// solver.assertFormula("(or p q r)");
    ///
    /// // Soft constraints with weights
    /// solver.assertSoft("p", "10");        // Prefer p=true (weight 10)
    /// solver.assertSoft("(not q)", "5");   // Prefer q=false (weight 5)
    /// solver.assertSoft("r", "3");         // Prefer r=true (weight 3)
    ///
    /// const result = solver.optimize();    // Finds assignment minimizing violated weight
    /// ```
    #[wasm_bindgen(js_name = assertSoft)]
    pub fn assert_soft(&mut self, formula: &str, weight: &str) -> Result<(), JsValue> {
        if string_utils::is_effectively_empty(formula) {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Soft constraint formula cannot be empty",
            )
            .into());
        }

        if string_utils::is_effectively_empty(weight) {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Weight cannot be empty").into(),
            );
        }

        // Validate weight is a positive integer
        let weight_val: u64 = weight.parse().map_err(|_| -> JsValue {
            WasmError::new(
                WasmErrorKind::InvalidInput,
                format!("Weight must be a positive integer, got: {}", weight),
            )
            .into()
        })?;

        let term = self.parse_objective_term(formula)?;
        self.require_bool_sort(term)?;
        self.push_soft(term, weight_val);
        Ok(())
    }

    /// Get a minimal model containing only specified variables
    ///
    /// Returns a model that includes only the specified variables, excluding
    /// all auxiliary variables or other variables created during solving.
    /// This is useful when you have a large problem with many variables but
    /// only care about a subset of them in the solution.
    ///
    /// If no variables are specified (empty array), returns all declared variables
    /// (but still excludes internal auxiliary variables).
    ///
    /// # Parameters
    ///
    /// * `variables` - Array of variable names to include in the minimal model.
    ///                 If empty, includes all user-declared variables.
    ///
    /// # Returns
    ///
    /// A JavaScript object containing only the specified variables and their values.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `checkSat()` has not returned "sat"
    /// - No model is available
    /// - Any specified variable doesn't exist in the model
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("y", "Int");
    /// solver.declareConst("z", "Int");
    /// solver.assertFormula("(= (+ x y z) 10)");
    /// solver.assertFormula("(> x 0)");
    /// solver.assertFormula("(> y 0)");
    /// solver.checkSat(); // "sat"
    ///
    /// // Get minimal model with only x and y
    /// const minimalModel = solver.getMinimalModel(["x", "y"]);
    /// // { x: {...}, y: {...} }
    /// ```
    #[wasm_bindgen(js_name = getMinimalModel)]
    pub fn get_minimal_model(&self, variables: Vec<String>) -> Result<JsValue, JsValue> {
        if self.last_result.as_deref() != Some("sat") {
            return Err(WasmError::new(
                WasmErrorKind::NoModel,
                "checkSat() must return 'sat' before getting model",
            )
            .into());
        }

        match self.ctx.get_model() {
            Some(model) => {
                let obj = js_sys::Object::new();

                if variables.is_empty() {
                    // If no variables specified, return all declared variables
                    for (name, sort, value) in model {
                        let entry = js_sys::Object::new();
                        js_sys::Reflect::set(&entry, &"sort".into(), &sort.into()).map_err(
                            |_| {
                                WasmError::new(
                                    WasmErrorKind::Unknown,
                                    "Failed to set sort property",
                                )
                            },
                        )?;
                        js_sys::Reflect::set(&entry, &"value".into(), &value.into()).map_err(
                            |_| {
                                WasmError::new(
                                    WasmErrorKind::Unknown,
                                    "Failed to set value property",
                                )
                            },
                        )?;
                        js_sys::Reflect::set(&obj, &name.into(), &entry).map_err(|_| {
                            WasmError::new(WasmErrorKind::Unknown, "Failed to set model entry")
                        })?;
                    }
                } else {
                    // Build a map for quick lookup
                    let model_map: std::collections::HashMap<String, (String, String)> = model
                        .into_iter()
                        .map(|(name, sort, value)| (name, (sort, value)))
                        .collect();

                    // Only include specified variables
                    for var_name in &variables {
                        if let Some((sort, value)) = model_map.get(var_name) {
                            let entry = js_sys::Object::new();
                            js_sys::Reflect::set(&entry, &"sort".into(), &sort.as_str().into())
                                .map_err(|_| {
                                    WasmError::new(
                                        WasmErrorKind::Unknown,
                                        "Failed to set sort property",
                                    )
                                })?;
                            js_sys::Reflect::set(&entry, &"value".into(), &value.as_str().into())
                                .map_err(|_| {
                                WasmError::new(
                                    WasmErrorKind::Unknown,
                                    "Failed to set value property",
                                )
                            })?;
                            js_sys::Reflect::set(&obj, &var_name.as_str().into(), &entry).map_err(
                                |_| {
                                    WasmError::new(
                                        WasmErrorKind::Unknown,
                                        "Failed to set model entry",
                                    )
                                },
                            )?;
                        } else {
                            return Err(WasmError::new(
                                WasmErrorKind::InvalidInput,
                                format!("Variable '{}' not found in model", var_name),
                            )
                            .into());
                        }
                    }
                }

                Ok(obj.into())
            }
            None => {
                Err(WasmError::new(WasmErrorKind::NoModel, "No model available from solver").into())
            }
        }
    }

    /// Compute Craig interpolant for an UNSAT problem
    ///
    /// Given an UNSAT formula partitioned into A and B, computes an interpolant I such that:
    /// - A implies I
    /// - I and B is UNSAT
    /// - I only contains symbols common to A and B
    ///
    /// This is useful for modular verification, abstraction refinement, and invariant generation.
    ///
    /// # Parameters
    ///
    /// * `partition_a` - Formulas in partition A (as SMT-LIB2 strings)
    /// * `partition_b` - Formulas in partition B (as SMT-LIB2 strings)
    ///
    /// # Returns
    ///
    /// An interpolant formula as an SMT-LIB2 string, or an error if:
    /// - The partitions are empty
    /// - The combined formula is not UNSAT
    /// - Proof production is not enabled
    ///
    /// # Errors
    ///
    /// A real Craig interpolant requires running the Pudlák algorithm
    /// (`oxiz_core::ast::interpolation::InterpolationContext`) over the
    /// structured proof DAG produced by the check. `oxiz_solver::Context`
    /// only exposes the resolved proof as a formatted string, not the
    /// structured object the algorithm needs, and this binding has no way to
    /// reconstruct one. Rather than return a value that merely looks like an
    /// interpolant (an earlier version of this function returned
    /// `(and <partition A>)`, which is not a valid Craig interpolant in
    /// general and can silently corrupt any verification pipeline consuming
    /// it), this currently always returns `WasmErrorKind::NotSupported` once
    /// the UNSAT/proof preconditions are confirmed.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setOption("produce-proofs", "true");
    /// solver.setLogic("QF_UF");
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("y", "Int");
    ///
    /// // A: x > 0
    /// // B: y < 0, x = y
    /// try {
    ///     solver.computeInterpolant(["(> x 0)"], ["(< y 0)", "(= x y)"]);
    /// } catch (e) {
    ///     // Currently always throws: real interpolation is not wired up yet.
    /// }
    /// ```
    ///
    /// # Reference
    ///
    /// Craig interpolation: W. Craig, "Linear reasoning. A new form of the Herbrand-Gentzen theorem", 1957
    /// Pudlák algorithm: P. Pudlák, "Lower bounds for resolution and cutting plane proofs", 1997
    #[wasm_bindgen(js_name = computeInterpolant)]
    pub fn compute_interpolant(
        &mut self,
        partition_a: Vec<String>,
        partition_b: Vec<String>,
    ) -> Result<String, JsValue> {
        // Validate inputs
        if partition_a.is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Partition A cannot be empty").into(),
            );
        }

        if partition_b.is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Partition B cannot be empty").into(),
            );
        }

        // Ensure proof production is enabled
        if self.ctx.get_option("produce-proofs") != Some("true") {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Proof production must be enabled. Call setOption('produce-proofs', 'true') first.",
            )
            .into());
        }

        // Validate formulas in partitions
        for formula_str in &partition_a {
            if formula_str.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    "Formula in partition A cannot be empty or whitespace",
                )
                .into());
            }
        }

        for formula_str in &partition_b {
            if formula_str.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    "Formula in partition B cannot be empty or whitespace",
                )
                .into());
            }
        }

        // Assert all formulas and check UNSAT
        self.push();
        for formula_str in partition_a.iter().chain(partition_b.iter()) {
            self.assert_formula(formula_str)?;
        }

        let result = self.check_sat();
        self.pop();

        if result != "unsat" {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                format!(
                    "Combined formula must be UNSAT for interpolation, but got: {}",
                    result
                ),
            )
            .into());
        }

        // Get proof from the last check-sat. This only confirms that a proof
        // was actually produced (not merely that the option was set) — the
        // structured proof itself is not exposed by `Context`, see the
        // `# Errors` section above.
        let proof_str = self.ctx.get_proof();

        if proof_str.is_empty() || proof_str.contains("not available") {
            return Err(WasmError::new(
                WasmErrorKind::NoProof,
                "No proof available. Ensure proof production is enabled.",
            )
            .into());
        }

        // Honest failure: never fabricate an interpolant. See the `# Errors`
        // section on this function for why real interpolation is not yet
        // reachable from this binding.
        Err(WasmError::new(
            WasmErrorKind::NotSupported,
            "Craig interpolation is not yet available from the WASM bindings: the \
             solver context does not expose the structured proof object required to \
             run Pudlák interpolation (oxiz_core::ast::interpolation). Returning an \
             unsound approximation instead of a real interpolant is not acceptable; \
             this is planned for a future release once the structured proof is \
             exposed at this API boundary.",
        )
        .into())
    }

    /// Eliminate quantifiers from a formula
    ///
    /// Given a formula with existential or universal quantifiers, attempts to eliminate
    /// them and return an equivalent quantifier-free formula.
    ///
    /// This is useful for:
    /// - Simplifying verification conditions
    /// - Extracting program invariants
    /// - Abstracting away implementation details
    ///
    /// # Parameters
    ///
    /// * `formula` - A formula with quantifiers (SMT-LIB2 string)
    ///
    /// # Returns
    ///
    /// A quantifier-free formula as an SMT-LIB2 string, or an error if:
    /// - The formula is invalid
    /// - The top-level quantifier could not be eliminated by the fast-path
    ///   solver below (most formulas beyond the trivial cases it handles)
    ///
    /// # Implementation status (honest disclosure)
    ///
    /// This calls `oxiz_core::qe::QeLiteSolver`, a real (not fabricated)
    /// quantifier-elimination pass -- but it is currently wired only for
    /// the *trivial* case where the quantifier body is already a tautology
    /// (e.g. `(exists ((y Int)) true)` eliminates to `true`, and dually for
    /// `forall`). `QeLiteSolver` also contains an equality-substitution
    /// path (the "one-point rule": `exists y. (y = e /\ phi)` -> `phi[y :=
    /// e]`), but that path operates on a different internal variable
    /// representation than the one `Forall`/`Exists` terms actually use, so
    /// it is not reachable from parsed quantifiers -- calling it would
    /// require a fix inside `oxiz-core` (out of scope for this binding).
    /// Full quantifier elimination (Cooper's algorithm for LIA, CAD for
    /// NRA) is not implemented at all yet.
    ///
    /// Only a quantifier this pass can *actually* reduce to a
    /// quantifier-free formula is ever returned; everything else fails
    /// with `NotSupported` rather than returning an unreduced or
    /// unsound formula.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    ///
    /// // The trivial case actually works today:
    /// const qfree = solver.eliminateQuantifiers("(exists ((y Int)) true)");
    /// console.log(qfree); // "true"
    ///
    /// // A formula requiring real substitution/Cooper's algorithm still
    /// // honestly reports NotSupported rather than guessing:
    /// try {
    ///   solver.eliminateQuantifiers("(exists ((y Int)) (= x (+ y 1)))");
    /// } catch (e) {
    ///   // e.message explains this specific formula isn't reducible yet.
    /// }
    /// ```
    ///
    /// # Reference
    ///
    /// Quantifier elimination: G. E. Collins, "Quantifier elimination for real closed fields", 1975
    #[wasm_bindgen(js_name = eliminateQuantifiers)]
    pub fn eliminate_quantifiers(&mut self, formula: &str) -> Result<String, JsValue> {
        // Validate input
        if formula.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Formula cannot be empty or whitespace",
            )
            .into());
        }

        // Check if formula contains quantifiers (cheap syntactic
        // pre-check before paying for a full parse).
        let has_quantifiers = formula.contains("exists") || formula.contains("forall");

        if !has_quantifiers {
            // Already quantifier-free, return as-is
            return Ok(formula.to_string());
        }

        let term = parse_term(formula, &mut self.ctx.terms).map_err(|e| -> JsValue {
            WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to parse formula: {}", e),
            )
            .into()
        })?;

        // Only a top-level quantifier is attempted: `QeLiteSolver::eliminate`
        // inspects exactly the outermost term-kind, so a quantifier nested
        // inside a larger boolean structure (e.g. `(and (exists ...) p)`)
        // is honestly reported as unsupported below rather than silently
        // left half-eliminated.
        let mut qe = oxiz_core::qe::QeLiteSolver::new();
        let qe_result = qe.eliminate(term, &mut self.ctx.terms);

        let result_term = match qe_result {
            oxiz_core::qe::QeLiteResult::Eliminated(t) => t,
            oxiz_core::qe::QeLiteResult::Simplified(t) => t,
            oxiz_core::qe::QeLiteResult::Unchanged => {
                return Err(WasmError::new(
                    WasmErrorKind::NotSupported,
                    "Quantifier elimination could not reduce this formula to a \
                     quantifier-free one. The available fast path only handles \
                     trivially-true/false quantifier bodies and top-level \
                     quantifiers (not ones nested inside a larger formula); \
                     general elimination (Cooper's algorithm for LIA, CAD for \
                     NRA) is not implemented yet.",
                )
                .into());
            }
            oxiz_core::qe::QeLiteResult::Error(msg) => {
                return Err(WasmError::new(
                    WasmErrorKind::Unknown,
                    format!("Quantifier elimination failed: {}", msg),
                )
                .into());
            }
        };

        let printer = Printer::new(&self.ctx.terms);
        Ok(printer.print_term(result_term))
    }
}

// Private helper methods (no wasm_bindgen — not exported to JS) supporting
// `minimize`/`maximize`/`assertSoft`/`optimize` above, plus the declared-
// symbol side-table other `js_api` modules (e.g. `declarations::declare_const`,
// `assertions::assert_formula`) feed into and read from. See the
// module-level docs at the top of this file for why this exists.
impl WasmSolver {
    /// Record a declared 0-ary symbol's name and sort-name string so a
    /// later objective/soft-constraint/assertion formula can resolve it
    /// with its true sort in an isolated parse (see the module docs).
    pub(crate) fn record_declared_symbol(&mut self, name: &str, sort_name: &str) {
        let existing = self
            .ctx
            .get_option(DECL_NAMES_KEY)
            .unwrap_or("")
            .to_string();
        let already_present = existing.split('\n').any(|n| n == name);
        if !already_present {
            let updated = if existing.is_empty() {
                name.to_string()
            } else {
                format!("{existing}\n{name}")
            };
            self.ctx.set_option(DECL_NAMES_KEY, &updated);
        }
        self.ctx
            .set_option(&format!("{DECL_SORT_PREFIX}{name}"), sort_name);
    }

    /// Build the `(declare-const ...)` prefix script text re-declaring every
    /// symbol recorded by `record_declared_symbol`, so a subsequent command
    /// parsed in the *same* [`parse_script`]/`execute_script` call resolves
    /// those symbols to their true sorts (and, via `TermManager`'s
    /// hash-consing of `Var` terms by `(name, sort)`, to the exact same
    /// `TermId`s already used in this context's assertions/declarations).
    ///
    /// `pub(crate)` so other `js_api` modules whose script-based operations
    /// suffer the same cross-call declared-symbol-visibility gap (see the
    /// module docs) can reuse it.
    pub(crate) fn declared_symbols_script_prefix(&self) -> String {
        let names = self
            .ctx
            .get_option(DECL_NAMES_KEY)
            .unwrap_or("")
            .to_string();
        let mut script = String::new();
        for name in names.split('\n').filter(|n| !n.is_empty()) {
            if let Some(sort_name) = self.ctx.get_option(&format!("{DECL_SORT_PREFIX}{name}")) {
                script.push_str("(declare-const ");
                script.push_str(name);
                script.push(' ');
                script.push_str(sort_name);
                script.push_str(")\n");
            }
        }
        script
    }

    /// Parse an objective/soft-constraint formula in isolation, but with
    /// every previously-declared symbol visible at its true sort (see the
    /// module docs). Returns the resulting `TermId`.
    fn parse_objective_term(&mut self, formula: &str) -> Result<TermId, JsValue> {
        let script = format!(
            "{}(simplify {formula})\n",
            self.declared_symbols_script_prefix()
        );

        let mut commands = parse_script(&script, &mut self.ctx.terms).map_err(|e| -> JsValue {
            WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to parse formula '{}': {}", formula, e),
            )
            .into()
        })?;

        match commands.pop() {
            Some(Command::Simplify(term)) => Ok(term),
            _ => Err(WasmError::new(
                WasmErrorKind::Unknown,
                "internal error: objective/soft-constraint parse did not yield the expected term",
            )
            .into()),
        }
    }

    /// Reject `term` unless it is Int- or Real-sorted (required for
    /// `minimize`/`maximize` objectives).
    fn require_arith_sort(&self, term: TermId, op: &str) -> Result<(), JsValue> {
        let sort = self.ctx.terms.get(term).map(|t| t.sort);
        let int_sort = self.ctx.terms.sorts.int_sort;
        let real_sort = self.ctx.terms.sorts.real_sort;
        if sort == Some(int_sort) || sort == Some(real_sort) {
            Ok(())
        } else {
            Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                format!("{op}() objective must be an Int- or Real-sorted expression"),
            )
            .into())
        }
    }

    /// Reject `term` unless it is Bool-sorted (required for `assertSoft`).
    fn require_bool_sort(&self, term: TermId) -> Result<(), JsValue> {
        let sort = self.ctx.terms.get(term).map(|t| t.sort);
        if sort == Some(self.ctx.terms.sorts.bool_sort) {
            Ok(())
        } else {
            Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "assertSoft() formula must be a Bool-sorted expression",
            )
            .into())
        }
    }

    /// Read a `_count`-suffixed side-table counter (defaults to `0` if
    /// absent or unparsable, which only happens before the first
    /// `minimize`/`maximize`/`assertSoft` call).
    fn read_count(&self, key: &str) -> usize {
        self.ctx
            .get_option(key)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    }

    /// Append a `minimize`/`maximize` objective to the side-table, in call
    /// order (their relative order becomes the `Optimizer`'s lexicographic
    /// priority order at `optimize()` time).
    fn push_objective(&mut self, term: TermId, is_max: bool) {
        let count = self.read_count(OBJ_COUNT_KEY);
        let kind = if is_max { "max" } else { "min" };
        self.ctx.set_option(
            &format!("{OBJ_PREFIX}{count}"),
            &format!("{kind}:{}", term.0),
        );
        self.ctx.set_option(OBJ_COUNT_KEY, &(count + 1).to_string());
    }

    /// Append an `assertSoft` soft constraint to the side-table.
    fn push_soft(&mut self, term: TermId, weight: u64) {
        let count = self.read_count(SOFT_COUNT_KEY);
        self.ctx.set_option(
            &format!("{SOFT_PREFIX}{count}"),
            &format!("{}:{weight}", term.0),
        );
        self.ctx
            .set_option(SOFT_COUNT_KEY, &(count + 1).to_string());
    }

    /// Read back all `minimize`/`maximize` objectives in registration
    /// order, as `(is_max, term)` pairs.
    fn read_objectives(&self) -> Vec<(bool, TermId)> {
        let count = self.read_count(OBJ_COUNT_KEY);
        let mut objectives = Vec::with_capacity(count);
        for i in 0..count {
            let Some(entry) = self.ctx.get_option(&format!("{OBJ_PREFIX}{i}")) else {
                continue;
            };
            let Some((kind, term_str)) = entry.split_once(':') else {
                continue;
            };
            let Ok(raw) = term_str.parse::<u32>() else {
                continue;
            };
            objectives.push((kind == "max", TermId(raw)));
        }
        objectives
    }

    /// Read back all `assertSoft` soft constraints in registration order,
    /// as `(term, weight)` pairs.
    fn read_soft_constraints(&self) -> Vec<(TermId, u64)> {
        let count = self.read_count(SOFT_COUNT_KEY);
        let mut soft = Vec::with_capacity(count);
        for i in 0..count {
            let Some(entry) = self.ctx.get_option(&format!("{SOFT_PREFIX}{i}")) else {
                continue;
            };
            let Some((term_str, weight_str)) = entry.split_once(':') else {
                continue;
            };
            let Ok(raw) = term_str.parse::<u32>() else {
                continue;
            };
            let Ok(weight) = weight_str.parse::<u64>() else {
                continue;
            };
            soft.push((TermId(raw), weight));
        }
        soft
    }

    /// Build the JS model object for an `Optimizer::optimize()` result,
    /// covering every symbol recorded by `record_declared_symbol` (i.e.
    /// every constant declared via `declareConst`/`declareFun`).
    ///
    /// The `Optimizer` solves against its own, freshly-built internal
    /// `Solver` (see `oxiz_solver::optimization::Optimizer::build_solver`),
    /// so its `Model` is a distinct object from `self.ctx`'s own solver
    /// state; this reconstructs each declared constant's `TermId` via
    /// `TermManager::mk_var`, which (being hash-consed on `(name, sort)`)
    /// returns the exact same `TermId` originally produced by
    /// `declare_const`, then reads that term's value from the optimizer's
    /// model directly.
    fn build_optimizer_model_object(
        &mut self,
        model: &oxiz_solver::Model,
    ) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();
        let names = self
            .ctx
            .get_option(DECL_NAMES_KEY)
            .unwrap_or("")
            .to_string();

        for name in names.split('\n').filter(|n| !n.is_empty()) {
            let Some(sort_name) = self
                .ctx
                .get_option(&format!("{DECL_SORT_PREFIX}{name}"))
                .map(str::to_string)
            else {
                continue;
            };
            let Ok(sort) = self.parse_sort(&sort_name) else {
                continue;
            };
            let term = self.ctx.terms.mk_var(name, sort);
            let value_str = match model.get(term) {
                Some(value_term) => Printer::new(&self.ctx.terms).print_term(value_term),
                // No assignment for this symbol (e.g. it never influenced
                // satisfiability/the objective): report a sort-appropriate
                // default rather than silently omitting it, matching
                // `Context::get_model()`'s own model-completion behavior.
                None => Self::default_value_for_sort(&sort_name),
            };

            let entry = js_sys::Object::new();
            js_sys::Reflect::set(&entry, &"sort".into(), &sort_name.as_str().into()).map_err(
                |_| WasmError::new(WasmErrorKind::Unknown, "Failed to set sort property"),
            )?;
            js_sys::Reflect::set(&entry, &"value".into(), &value_str.as_str().into()).map_err(
                |_| WasmError::new(WasmErrorKind::Unknown, "Failed to set value property"),
            )?;
            js_sys::Reflect::set(&obj, &name.into(), &entry)
                .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set model entry"))?;
        }

        Ok(obj.into())
    }

    /// A sort-appropriate default value string for a declared symbol with
    /// no explicit assignment in an optimizer model.
    fn default_value_for_sort(sort_name: &str) -> String {
        match sort_name {
            "Bool" => "false".to_string(),
            "Real" => "0.0".to_string(),
            s if s.starts_with("BitVec") => "#b0".to_string(),
            _ => "0".to_string(),
        }
    }
}

/// Tests for `eliminateQuantifiers`'s success paths only. Its error paths
/// construct a `wasm_bindgen::JsValue` (via `WasmError::into()`), which
/// aborts the process when actually invoked outside a real wasm32/JS
/// engine (see e.g. `js_api::streaming`'s test-setup notes for the same
/// constraint) -- so only inputs expected to return `Ok` are exercised
/// here. The `NotSupported`/parse-error paths are covered by the
/// wasm32-gated integration tests instead.
#[cfg(test)]
mod eliminate_quantifiers_tests {
    use crate::WasmSolver;

    #[test]
    fn eliminates_trivial_exists_true_to_true() {
        let mut solver = WasmSolver::new();
        let result = solver
            .eliminate_quantifiers("(exists ((y Int)) true)")
            .expect("trivial exists-true should eliminate to true");
        assert_eq!(result, "true");
    }

    #[test]
    fn eliminates_trivial_forall_true_to_true() {
        let mut solver = WasmSolver::new();
        let result = solver
            .eliminate_quantifiers("(forall ((y Int)) true)")
            .expect("trivial forall-true should eliminate to true");
        assert_eq!(result, "true");
    }

    #[test]
    fn quantifier_free_formula_passes_through_unchanged() {
        let mut solver = WasmSolver::new();
        let result = solver
            .eliminate_quantifiers("(> 1 0)")
            .expect("quantifier-free formula should pass through as-is");
        assert_eq!(result, "(> 1 0)");
    }
}

/// Native (non-wasm32) regression tests for `minimize`/`maximize`/
/// `assertSoft`'s success paths -- these methods return `Result<(), JsValue>`
/// whose `Ok(())` arm never touches `js_sys`/`JsValue` construction, so
/// (like `eliminate_quantifiers_tests` above) they can run outside a real
/// wasm32/JS engine as long as the call is expected to succeed. `optimize()`
/// itself always builds a `js_sys::Object` (even on success), so it cannot
/// be exercised here at all -- see `oxiz-wasm/tests/audit_wasm_p2.rs` for
/// the wasm32-gated coverage of `optimize()`'s actual results.
/// The precondition [`WasmSolver::parse_objective_term`] relies on: an
/// objective/soft-constraint string is spliced into `(simplify <formula>)` and
/// only a trailing [`Command::Simplify`] is accepted, everything else being
/// rejected outright.
///
/// The rejection itself cannot be exercised natively — it builds a `JsValue`,
/// which aborts outside a wasm32/JS engine — so what is pinned here is that a
/// recursive definition really does parse to something *other* than `Simplify`.
/// That is what routes it to the rejection instead of letting it through as an
/// objective term with the definition silently dropped.
#[cfg(test)]
mod objective_term_guard_tests {
    use oxiz_core::ast::TermManager;
    use oxiz_core::smtlib::{Command, parse_script};

    #[test]
    fn a_recursive_definition_never_parses_as_an_objective_term() {
        let mut terms = TermManager::new();
        let script = "(define-fun-rec f ((n Int)) Int (ite (<= n 0) 0 (f (- n 1))))";
        let commands = parse_script(script, &mut terms).expect("define-fun-rec parses");
        assert!(
            matches!(commands.last(), Some(Command::DefineFunsRec(_))),
            "a recursive definition must surface as its own command"
        );
        assert!(
            !matches!(commands.last(), Some(Command::Simplify(_))),
            "it must never reach parse_objective_term's accepted arm"
        );
    }
}

#[cfg(test)]
mod objective_registration_tests {
    use crate::WasmSolver;

    #[test]
    fn minimize_accepts_declared_int_constant() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("declare_const should succeed");
        solver
            .assert_formula("(>= x 0)")
            .expect("assert_formula should succeed");
        solver
            .minimize("x")
            .expect("minimize on a declared Int constant should succeed");
    }

    #[test]
    fn maximize_accepts_arithmetic_expression_over_declared_constants() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.assert_formula("(>= x 0)").unwrap();
        solver.assert_formula("(>= y 0)").unwrap();
        solver
            .maximize("(+ x y)")
            .expect("maximize over declared Int constants should succeed");
    }

    #[test]
    fn assert_soft_accepts_declared_bool_constant() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver
            .assert_soft("p", "10")
            .expect("assertSoft on a declared Bool constant should succeed");
    }

    #[test]
    fn multiple_objectives_accumulate_without_error() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.assert_formula("(>= x 0)").unwrap();
        solver.assert_formula("(>= y 0)").unwrap();
        solver.maximize("x").unwrap();
        solver.maximize("y").unwrap();
        solver.minimize("(+ x y)").unwrap();
    }
}
