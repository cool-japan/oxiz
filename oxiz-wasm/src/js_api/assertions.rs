//! Formula assertion operations.

use crate::WasmSolver;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
    /// Validate a formula without asserting it
    ///
    /// Checks if a formula is well-formed and type-correct without adding it
    /// to the assertion stack. This is useful for validating user input before
    /// asserting it.
    ///
    /// # Parameters
    ///
    /// * `formula` - An SMT-LIB2 expression string to validate
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the formula is valid, or an error with details about
    /// what's wrong with the formula.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    ///
    /// // Validate before asserting
    /// try {
    ///   solver.validateFormula("(> x 0)");
    ///   solver.assertFormula("(> x 0)"); // Safe to assert
    /// } catch (e) {
    ///   console.error("Invalid formula:", e.message);
    /// }
    /// ```
    #[wasm_bindgen(js_name = validateFormula)]
    pub fn validate_formula(&mut self, formula: &str) -> Result<bool, JsValue> {
        if formula.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Formula cannot be empty").into(),
            );
        }

        // Use push/pop to validate without modifying the assertion stack
        self.ctx.push();
        let script = format!("(assert {})", formula);
        let result = self.ctx.execute_script(&script);
        self.ctx.pop();

        match result {
            Ok(_) => Ok(true),
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Invalid formula: {}", e),
            )
            .into()),
        }
    }

    /// Assert a formula from an SMT-LIB2 expression string
    ///
    /// This adds an assertion to the solver. The formula must be a valid SMT-LIB2
    /// boolean expression. All variables referenced in the formula must have been
    /// previously declared using `declareConst` or `declareFun`.
    ///
    /// # Parameters
    ///
    /// * `formula` - An SMT-LIB2 boolean expression string
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The formula is empty or malformed
    /// - The formula contains syntax errors
    /// - The formula references undeclared variables
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("y", "Int");
    /// solver.assertFormula("(> x 0)");
    /// solver.assertFormula("(< y x)");
    /// solver.assertFormula("(and (> x 5) (<= y 10))");
    /// ```
    #[wasm_bindgen(js_name = assertFormula)]
    pub fn assert_formula(&mut self, formula: &str) -> Result<(), JsValue> {
        if formula.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Formula cannot be empty").into(),
            );
        }

        let script = format!("(assert {})", formula);
        self.ctx.execute_script(&script).map_err(|e| -> JsValue {
            WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to assert formula: {}", e),
            )
            .into()
        })?;
        Ok(())
    }

    /// Assert a formula with error recovery
    ///
    /// Tries to assert a formula, but if it fails, provides detailed error
    /// information and suggestions for fixing the issue. Unlike `assertFormula`,
    /// this method attempts to give more helpful error messages.
    ///
    /// # Parameters
    ///
    /// * `formula` - An SMT-LIB2 boolean expression string
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or an error with detailed diagnostics
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    ///
    /// try {
    ///   solver.assertFormulaSafe("(> x y)"); // y is not declared
    /// } catch (e) {
    ///   console.error(e.message); // Will suggest declaring y
    /// }
    /// ```
    #[wasm_bindgen(js_name = assertFormulaSafe)]
    pub fn assert_formula_safe(&mut self, formula: &str) -> Result<(), JsValue> {
        if formula.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Formula cannot be empty").into(),
            );
        }

        // First try to validate the formula
        match self.validate_formula(formula) {
            Ok(_) => {
                // If validation succeeds, assert it
                self.assert_formula(formula)
            }
            Err(e) => {
                // Provide more helpful error message
                let error_msg = e.as_string().unwrap_or_else(|| "Unknown error".to_string());

                // Check for common issues and provide suggestions
                let suggestion = if error_msg.contains("undeclared") {
                    " Hint: Make sure all variables are declared with declareConst() or declareFun() before use."
                } else if error_msg.contains("type") || error_msg.contains("sort") {
                    " Hint: Check that operations are applied to compatible types (e.g., arithmetic on Int/Real, logic on Bool)."
                } else if error_msg.contains("syntax") || error_msg.contains("parse") {
                    " Hint: Verify the formula uses correct SMT-LIB2 syntax with balanced parentheses."
                } else {
                    ""
                };

                Err(WasmError::new(
                    WasmErrorKind::ParseError,
                    format!("{}{}", error_msg, suggestion),
                )
                .into())
            }
        }
    }

    /// Assert multiple formulas at once (batch operation)
    ///
    /// This is a convenience method that asserts multiple formulas in a single call.
    /// It's more efficient than calling `assertFormula` multiple times.
    /// If any formula fails to assert, the method stops and returns an error.
    ///
    /// # Parameters
    ///
    /// * `formulas` - Array of SMT-LIB2 boolean expression strings
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all assertions succeed, or an error with details about the first
    /// failed assertion.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareFuns(["x Int", "y Int", "z Int"]);
    /// solver.assertFormulas([
    ///   "(> x 0)",
    ///   "(< y x)",
    ///   "(= z (+ x y))",
    ///   "(< z 100)"
    /// ]);
    /// ```
    #[wasm_bindgen(js_name = assertFormulas)]
    pub fn assert_formulas(&mut self, formulas: Vec<String>) -> Result<(), JsValue> {
        if formulas.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Formulas list cannot be empty",
            )
            .into());
        }

        for (idx, formula) in formulas.iter().enumerate() {
            self.assert_formula(formula).map_err(|e| -> JsValue {
                let error_msg = e.as_string().unwrap_or_else(|| "Unknown error".to_string());
                WasmError::new(
                    WasmErrorKind::ParseError,
                    format!("Failed to assert formula at index {}: {}", idx, error_msg),
                )
                .into()
            })?;
        }

        Ok(())
    }

    /// Assert a formula with a name/label
    ///
    /// This asserts a formula with an associated name that can be used to identify it
    /// in unsat cores. Named assertions are useful for tracking which assumptions led
    /// to unsatisfiability.
    ///
    /// # Parameters
    ///
    /// * `name` - A unique name/label for this assertion
    /// * `formula` - An SMT-LIB2 boolean expression string
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the assertion succeeds, or an error if it fails.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.setOption("produce-unsat-cores", "true");
    /// solver.declareConst("x", "Int");
    ///
    /// solver.assertNamed("positive", "(> x 0)");
    /// solver.assertNamed("negative", "(< x 0)");
    ///
    /// if (solver.checkSat() === "unsat") {
    ///   const core = solver.getUnsatCore();
    ///   console.log(core); // Shows which named assertions are in conflict
    /// }
    /// ```
    #[wasm_bindgen(js_name = assertNamed)]
    pub fn assert_named(&mut self, name: &str, formula: &str) -> Result<(), JsValue> {
        if name.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Assertion name cannot be empty",
            )
            .into());
        }

        if formula.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Formula cannot be empty").into(),
            );
        }

        let script = format!("(assert (! {} :named {}))", formula, name);
        self.ctx.execute_script(&script).map_err(|e| -> JsValue {
            WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to assert named formula '{}': {}", name, e),
            )
            .into()
        })?;

        Ok(())
    }
}
