//! Model retrieval and inspection operations.

use crate::WasmSolver;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
    /// Get the model as a JavaScript object
    ///
    /// Returns a JavaScript object where keys are variable names and values are
    /// objects containing the sort and value of each variable in the satisfying
    /// assignment.
    ///
    /// This method can only be called after `checkSat()` returns "sat".
    ///
    /// # Returns
    ///
    /// A JavaScript object mapping variable names to `{sort: string, value: string}` objects
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `checkSat()` has not been called yet
    /// - The last `checkSat()` result was not "sat"
    /// - No model is available
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("y", "Bool");
    /// solver.assertFormula("(> x 5)");
    /// solver.assertFormula("y");
    ///
    /// if (solver.checkSat() === "sat") {
    ///   const model = solver.getModel();
    ///   console.log(model.x.value); // e.g., "6"
    ///   console.log(model.x.sort);  // "Int"
    ///   console.log(model.y.value); // "true"
    /// }
    /// ```
    #[wasm_bindgen(js_name = getModel)]
    pub fn get_model(&self) -> Result<JsValue, JsValue> {
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
                for (name, sort, value) in model {
                    let entry = js_sys::Object::new();
                    js_sys::Reflect::set(&entry, &"sort".into(), &sort.into()).map_err(|_| {
                        WasmError::new(WasmErrorKind::Unknown, "Failed to set sort property")
                    })?;
                    js_sys::Reflect::set(&entry, &"value".into(), &value.into()).map_err(|_| {
                        WasmError::new(WasmErrorKind::Unknown, "Failed to set value property")
                    })?;
                    js_sys::Reflect::set(&obj, &name.into(), &entry).map_err(|_| {
                        WasmError::new(WasmErrorKind::Unknown, "Failed to set model entry")
                    })?;
                }
                Ok(obj.into())
            }
            None => {
                Err(WasmError::new(WasmErrorKind::NoModel, "No model available from solver").into())
            }
        }
    }

    /// Get the model as an SMT-LIB2 formatted string
    ///
    /// Returns a human-readable string representation of the model in SMT-LIB2 format.
    /// This method can only be called after `checkSat()` returns "sat".
    ///
    /// # Returns
    ///
    /// An SMT-LIB2 formatted string representing the model
    ///
    /// # Errors
    ///
    /// Returns an error if `checkSat()` has not returned "sat"
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    /// solver.declareConst("p", "Bool");
    /// solver.assertFormula("p");
    ///
    /// if (solver.checkSat() === "sat") {
    ///   const modelStr = solver.getModelString();
    ///   console.log(modelStr); // prints model in SMT-LIB2 format
    /// }
    /// ```
    #[wasm_bindgen(js_name = getModelString)]
    pub fn get_model_string(&self) -> Result<String, JsValue> {
        if self.last_result.as_deref() != Some("sat") {
            return Err(WasmError::new(
                WasmErrorKind::NoModel,
                "checkSat() must return 'sat' before getting model",
            )
            .into());
        }
        Ok(self.ctx.format_model())
    }

    /// Get the value of specific terms/expressions
    ///
    /// Evaluates one or more SMT-LIB2 expressions in the current model and returns
    /// their values. This method can only be called after `checkSat()` returns "sat".
    ///
    /// # Parameters
    ///
    /// * `terms` - An array of SMT-LIB2 expression strings to evaluate
    ///
    /// # Returns
    ///
    /// A string containing the SMT-LIB2 representation of the values
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `checkSat()` has not returned "sat"
    /// - Any term is malformed or contains syntax errors
    /// - Any term references undeclared variables
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 5)");
    ///
    /// if (solver.checkSat() === "sat") {
    ///   const values = solver.getValue(["x", "(+ x 1)"]);
    ///   console.log(values); // prints values of x and (x+1)
    /// }
    /// ```
    #[wasm_bindgen(js_name = getValue)]
    pub fn get_value(&mut self, terms: Vec<String>) -> Result<JsValue, JsValue> {
        if self.last_result.as_deref() != Some("sat") {
            return Err(WasmError::new(
                WasmErrorKind::NoModel,
                "checkSat() must return 'sat' before getting values",
            )
            .into());
        }

        if terms.is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Terms array cannot be empty").into(),
            );
        }

        // Build a get-value command
        let terms_str = terms.join(" ");
        let script = format!("(get-value ({}))", terms_str);

        match self.ctx.execute_script(&script) {
            Ok(output) => {
                let result = output.join("\n");
                Ok(JsValue::from_str(&result))
            }
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to get values: {}", e),
            )
            .into()),
        }
    }

    /// Get the unsat core (set of assertions contributing to unsatisfiability)
    ///
    /// Returns the unsatisfiable core, which is a (usually minimal) subset of
    /// assertions that are sufficient to cause unsatisfiability. This is useful
    /// for debugging why a set of constraints is unsatisfiable.
    ///
    /// This method can only be called after `checkSat()` returns "unsat".
    ///
    /// # Returns
    ///
    /// A string representation of the unsat core in SMT-LIB2 format
    ///
    /// # Errors
    ///
    /// Returns an error if `checkSat()` has not returned "unsat"
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    /// solver.declareConst("p", "Bool");
    /// solver.assertFormula("p");
    /// solver.assertFormula("(not p)");
    ///
    /// if (solver.checkSat() === "unsat") {
    ///   const core = solver.getUnsatCore();
    ///   console.log(core); // prints the conflicting assertions
    /// }
    /// ```
    #[wasm_bindgen(js_name = getUnsatCore)]
    pub fn get_unsat_core(&self) -> Result<JsValue, JsValue> {
        if self.last_result.as_deref() != Some("unsat") {
            return Err(WasmError::new(
                WasmErrorKind::NoUnsatCore,
                "checkSat() must return 'unsat' before getting unsat core",
            )
            .into());
        }

        // Get the formatted unsat core from context
        let core_str = self.ctx.format_assertions();
        Ok(JsValue::from_str(&core_str))
    }

    /// Get a proof of unsatisfiability
    ///
    /// Returns a proof object that demonstrates why the current set of assertions
    /// is unsatisfiable. Proof production must be enabled by setting the
    /// `produce-proofs` option to `true` before calling `checkSat()`.
    ///
    /// This method can only be called after `checkSat()` returns "unsat".
    ///
    /// # Returns
    ///
    /// A string representation of the proof in SMT-LIB2 format
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `checkSat()` has not returned "unsat"
    /// - Proof production is not enabled
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    /// solver.setOption("produce-proofs", "true");
    /// solver.declareConst("p", "Bool");
    /// solver.assertFormula("p");
    /// solver.assertFormula("(not p)");
    ///
    /// if (solver.checkSat() === "unsat") {
    ///   const proof = solver.getProof();
    ///   console.log(proof); // prints the proof object
    /// }
    /// ```
    #[wasm_bindgen(js_name = getProof)]
    pub fn get_proof(&self) -> Result<String, JsValue> {
        if self.last_result.as_deref() != Some("unsat") {
            return Err(WasmError::new(
                WasmErrorKind::InvalidState,
                "checkSat() must return 'unsat' before getting proof",
            )
            .into());
        }

        let proof_str = self.ctx.get_proof();

        // Check if proof generation was enabled
        if proof_str.contains("Proof generation not enabled") {
            return Err(WasmError::new(
                WasmErrorKind::InvalidState,
                "Proof generation not enabled. Set option 'produce-proofs' to 'true' before calling checkSat()",
            )
            .into());
        }

        Ok(proof_str)
    }

    /// Get all current assertions as an SMT-LIB2 formatted string
    ///
    /// Returns a string representation of all currently asserted formulas
    /// in SMT-LIB2 format.
    ///
    /// # Returns
    ///
    /// An SMT-LIB2 formatted string of all assertions
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    /// const assertions = solver.getAssertions();
    /// console.log(assertions); // prints: ((> x 0))
    /// ```
    #[wasm_bindgen(js_name = getAssertions)]
    pub fn get_assertions(&self) -> String {
        self.ctx.format_assertions()
    }
}
