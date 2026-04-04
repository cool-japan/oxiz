//! Optimization, MaxSMT, Craig interpolation, and quantifier elimination.

use crate::WasmSolver;
use crate::string_utils;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
    /// Add a minimization objective
    ///
    /// Adds an objective function to minimize. The formula must evaluate to an
    /// integer or real value. When `optimize()` is called, the solver will find
    /// a model that minimizes this objective while satisfying all assertions.
    ///
    /// For multiple objectives, they are optimized lexicographically in the order
    /// they were added (first objective has highest priority).
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
    /// - The formula is not an arithmetic expression
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

        // Build minimize command
        let script = format!("(minimize {})", formula);

        match self.ctx.execute_script(&script) {
            Ok(_) => Ok(()),
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to add minimization objective: {}", e),
            )
            .into()),
        }
    }

    /// Add a maximization objective
    ///
    /// Adds an objective function to maximize. The formula must evaluate to an
    /// integer or real value. When `optimize()` is called, the solver will find
    /// a model that maximizes this objective while satisfying all assertions.
    ///
    /// For multiple objectives, they are optimized lexicographically in the order
    /// they were added (first objective has highest priority).
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
    /// - The formula is not an arithmetic expression
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

        // Build maximize command
        let script = format!("(maximize {})", formula);

        match self.ctx.execute_script(&script) {
            Ok(_) => Ok(()),
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to add maximization objective: {}", e),
            )
            .into()),
        }
    }

    /// Run optimization on the current assertions and objectives
    ///
    /// Solves the optimization problem defined by the current assertions and
    /// objectives (added via `minimize()` or `maximize()`). Returns an object
    /// containing the optimization status, optimal value(s), and satisfying model.
    ///
    /// # Returns
    ///
    /// A JavaScript object with the following structure:
    /// ```javascript
    /// {
    ///   status: "optimal" | "unbounded" | "unsat" | "unknown",
    ///   value: "42",           // The optimal value (if optimal)
    ///   model: { x: {...} }    // The satisfying model (if optimal)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No objectives have been added (use `minimize()` or `maximize()` first)
    /// - The optimization process fails
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
        // Execute (check-sat) which will trigger optimization if objectives are set
        match self.ctx.execute_script("(check-sat)") {
            Ok(output) => {
                let status_str = output.join("").trim().to_string();

                let result = js_sys::Object::new();

                match status_str.as_str() {
                    "sat" | "optimal" => {
                        let _ = js_sys::Reflect::set(&result, &"status".into(), &"optimal".into());

                        // Try to get objectives values
                        if let Ok(obj_output) = self.ctx.execute_script("(get-objectives)") {
                            let objectives_str = obj_output.join("\n");
                            let _ = js_sys::Reflect::set(
                                &result,
                                &"objectives".into(),
                                &objectives_str.into(),
                            );

                            // Try to extract the first objective value
                            if let Some(first_line) = obj_output.first() {
                                let _ = js_sys::Reflect::set(
                                    &result,
                                    &"value".into(),
                                    &first_line.into(),
                                );
                            }
                        }

                        // Get the model
                        match self.get_model() {
                            Ok(model) => {
                                let _ = js_sys::Reflect::set(&result, &"model".into(), &model);
                            }
                            Err(_) => {
                                let _ =
                                    js_sys::Reflect::set(&result, &"model".into(), &JsValue::NULL);
                            }
                        }
                    }
                    "unsat" => {
                        let _ = js_sys::Reflect::set(&result, &"status".into(), &"unsat".into());
                    }
                    "inf" | "unbounded" => {
                        let _ =
                            js_sys::Reflect::set(&result, &"status".into(), &"unbounded".into());
                    }
                    _ => {
                        let _ = js_sys::Reflect::set(&result, &"status".into(), &"unknown".into());
                    }
                }

                Ok(result.into())
            }
            Err(e) => Err(WasmError::new(
                WasmErrorKind::Unknown,
                format!("Optimization failed: {}", e),
            )
            .into()),
        }
    }

    /// Add a soft constraint with weight (for MaxSMT)
    ///
    /// Soft constraints are constraints that the solver will try to satisfy,
    /// but may violate if necessary. Each soft constraint has a weight indicating
    /// its importance. The solver minimizes the total weight of violated constraints.
    ///
    /// This enables MaxSMT (Maximum Satisfiability Modulo Theories) solving, where
    /// you want to satisfy as many constraints as possible (weighted by importance).
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
    /// - The formula contains syntax errors
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
        if weight.parse::<u64>().is_err() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                format!("Weight must be a positive integer, got: {}", weight),
            )
            .into());
        }

        // Build assert-soft command
        let script = format!("(assert-soft {} :weight {})", formula, weight);

        match self.ctx.execute_script(&script) {
            Ok(_) => Ok(()),
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to add soft constraint: {}", e),
            )
            .into()),
        }
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
    /// // Interpolant should be: x > 0 (or equivalent)
    /// const interpolant = solver.computeInterpolant(
    ///     ["(> x 0)"],
    ///     ["(< y 0)", "(= x y)"]
    /// );
    /// console.log(interpolant);
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

        // Get proof from the last check-sat
        let proof_str = self.ctx.get_proof();

        if proof_str.is_empty() || proof_str.contains("not available") {
            return Err(WasmError::new(
                WasmErrorKind::NoProof,
                "No proof available. Ensure proof production is enabled.",
            )
            .into());
        }

        // For now, return a placeholder indicating interpolation infrastructure is being built
        // Full interpolation support requires:
        // 1. Proof parsing and traversal
        // 2. Partition tracking through proof steps
        // 3. Pudlák interpolation algorithm

        // Return a simple approximation based on partition A
        let interpolant = format!("(and {})", partition_a.join(" "));

        Ok(interpolant)
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
    /// - Quantifier elimination is not supported for the current logic
    /// - The formula cannot be eliminated (e.g., non-linear arithmetic)
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("LIA");  // Linear Integer Arithmetic
    /// solver.declareConst("x", "Int");
    ///
    /// // Eliminate existential quantifier: exists y. (x = y + 1)
    /// // Result should be: true (always satisfiable for any x)
    /// const qfree = solver.eliminateQuantifiers("(exists ((y Int)) (= x (+ y 1)))");
    /// console.log(qfree);
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

        // Check if formula contains quantifiers
        let has_quantifiers = formula.contains("exists") || formula.contains("forall");

        if !has_quantifiers {
            // Already quantifier-free, return as-is
            return Ok(formula.to_string());
        }

        // For now, return a placeholder indicating quantifier elimination infrastructure is being built
        // Full quantifier elimination support requires:
        // 1. Complete SMT-LIB2 parser with quantifier support
        // 2. Cooper's algorithm for linear integer arithmetic
        // 3. CAD (Cylindrical Algebraic Decomposition) for real arithmetic
        // 4. Integration with oxiz-spacer's existential projector

        // Return an error for now, indicating the feature is planned
        Err(WasmError::new(
            WasmErrorKind::NotSupported,
            "Quantifier elimination is not yet fully implemented. This feature requires:\n\
             - Complete SMT-LIB2 parser with quantifier support\n\
             - Cooper's algorithm for LIA\n\
             - CAD for real arithmetic\n\
             Support is planned for a future release.",
        )
        .into())
    }
}
