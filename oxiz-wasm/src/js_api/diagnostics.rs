//! Diagnostics, debugging, statistics, and the private `parse_sort` helper.

use crate::WasmSolver;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
    /// Simplify an SMT-LIB2 expression and return the result
    ///
    /// Applies simplification rules to the given expression and returns
    /// the simplified form. This can be useful for debugging or understanding
    /// how the solver interprets expressions.
    ///
    /// # Parameters
    ///
    /// * `expr` - An SMT-LIB2 expression string to simplify
    ///
    /// # Returns
    ///
    /// The simplified expression as a string
    ///
    /// # Errors
    ///
    /// Returns an error if the expression is malformed or contains syntax errors
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const simplified = solver.simplify("(+ 1 2)");
    /// console.log(simplified); // "3"
    /// const simplified2 = solver.simplify("(and true false)");
    /// console.log(simplified2); // "false"
    /// ```
    #[wasm_bindgen]
    pub fn simplify(&mut self, expr: &str) -> Result<String, JsValue> {
        if expr.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Expression cannot be empty").into(),
            );
        }

        let script = format!("(simplify {})", expr);
        match self.ctx.execute_script(&script) {
            Ok(output) => Ok(output.join("")),
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to simplify expression: {}", e),
            )
            .into()),
        }
    }

    /// Get a debug representation of the solver state
    ///
    /// Returns a comprehensive dump of the current solver state in human-readable
    /// format. This is useful for debugging and understanding what the solver knows.
    ///
    /// # Returns
    ///
    /// A string containing the solver state
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    /// solver.checkSat();
    ///
    /// const debug = solver.debugDump();
    /// console.log(debug);
    /// // Output includes: logic, assertions, last result, options, etc.
    /// ```
    #[wasm_bindgen(js_name = debugDump)]
    pub fn debug_dump(&self) -> String {
        let mut output = String::new();
        output.push_str("=== OxiZ Solver Debug Dump ===\n\n");

        // Version info
        output.push_str(&format!("Version: {}\n", env!("CARGO_PKG_VERSION")));

        // Logic
        if let Some(logic) = self.ctx.get_option("logic") {
            output.push_str(&format!("Logic: {}\n", logic));
        } else {
            output.push_str("Logic: <not set>\n");
        }

        // Last result
        if let Some(ref result) = self.last_result {
            output.push_str(&format!("Last Result: {}\n", result));
        } else {
            output.push_str("Last Result: <none>\n");
        }

        // Cancelled status
        output.push_str(&format!("Cancelled: {}\n", self.cancelled));

        // Options
        output.push_str("\nOptions:\n");
        if let Some(produce_models) = self.ctx.get_option("produce-models") {
            output.push_str(&format!("  produce-models: {}\n", produce_models));
        }
        if let Some(produce_cores) = self.ctx.get_option("produce-unsat-cores") {
            output.push_str(&format!("  produce-unsat-cores: {}\n", produce_cores));
        }
        if let Some(incremental) = self.ctx.get_option("incremental") {
            output.push_str(&format!("  incremental: {}\n", incremental));
        }

        // Assertions
        output.push_str("\nAssertions:\n");
        let assertions = self.ctx.format_assertions();
        output.push_str(&assertions);
        output.push('\n');

        // Model (if available)
        if self.last_result.as_deref() == Some("sat") {
            output.push_str("\nModel:\n");
            output.push_str(&self.ctx.format_model());
            output.push('\n');
        }

        output.push_str("\n=== End Debug Dump ===\n");
        output
    }

    /// Get solver statistics as a JavaScript object
    ///
    /// Returns various statistics about the solver's operation, useful for
    /// performance monitoring and debugging. Statistics include information
    /// about the number of assertions, solver calls, and other metrics.
    ///
    /// # Returns
    ///
    /// A JavaScript object containing solver statistics
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    /// solver.checkSat();
    ///
    /// const stats = solver.getStatistics();
    /// console.log("Assertions:", stats.num_assertions);
    /// console.log("Check-sat calls:", stats.num_check_sat);
    /// console.log("Result:", stats.last_result);
    /// ```
    #[wasm_bindgen(js_name = getStatistics)]
    pub fn get_statistics(&self) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();

        // Get assertion count
        let assertions = self.ctx.format_assertions();
        let num_assertions = assertions.matches("(assert").count();
        js_sys::Reflect::set(&obj, &"num_assertions".into(), &num_assertions.into())
            .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set num_assertions"))?;

        // Last result
        if let Some(ref result) = self.last_result {
            js_sys::Reflect::set(&obj, &"last_result".into(), &result.as_str().into())
                .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set last_result"))?;
        }

        // Cancelled status
        js_sys::Reflect::set(&obj, &"cancelled".into(), &self.cancelled.into()).map_err(|_| {
            WasmError::new(WasmErrorKind::Unknown, "Failed to set cancelled status")
        })?;

        // Logic setting
        if let Some(logic) = self.ctx.get_option("logic") {
            js_sys::Reflect::set(&obj, &"logic".into(), &logic.into())
                .map_err(|_| WasmError::new(WasmErrorKind::Unknown, "Failed to set logic"))?;
        }

        Ok(obj.into())
    }

    /// Get detailed solver information
    ///
    /// Returns metadata about the solver, including version, capabilities,
    /// and configuration information.
    ///
    /// # Parameters
    ///
    /// * `key` - The information key to retrieve. Supported keys:
    ///   - `"name"` - Solver name
    ///   - `"version"` - Solver version
    ///   - `"authors"` - Solver authors
    ///   - `"all-statistics"` - Enable all statistics
    ///   - `"capabilities"` - List of supported features
    ///
    /// # Returns
    ///
    /// The requested information as a string, or an error if the key is unknown
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const name = solver.getInfo("name");
    /// const version = solver.getInfo("version");
    /// console.log(`${name} v${version}`);
    ///
    /// const caps = solver.getInfo("capabilities");
    /// console.log("Capabilities:", caps);
    /// ```
    #[wasm_bindgen(js_name = getInfo)]
    pub fn get_info(&self, key: &str) -> Result<String, JsValue> {
        match key {
            "name" | ":name" => Ok("OxiZ".to_string()),
            "version" | ":version" => Ok(env!("CARGO_PKG_VERSION").to_string()),
            "authors" | ":authors" => Ok(env!("CARGO_PKG_AUTHORS").to_string()),
            "all-statistics" | ":all-statistics" => Ok("true".to_string()),
            "error-behavior" | ":error-behavior" => Ok("immediate-exit".to_string()),
            "reason-unknown" | ":reason-unknown" => {
                if self.last_result.as_deref() == Some("unknown") {
                    Ok("incomplete".to_string())
                } else {
                    Ok("".to_string())
                }
            }
            "capabilities" | ":capabilities" => {
                Ok("incremental push-pop check-sat-assuming proofs unsat-cores models simplification function-declarations formula-builders".to_string())
            }
            _ => Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                format!(
                    "Unknown info key '{}'. Supported keys: name, version, authors, all-statistics, error-behavior, reason-unknown, capabilities",
                    key
                ),
            )
            .into()),
        }
    }

    /// Get diagnostic warnings about solver usage
    ///
    /// Analyzes the current solver state and returns an array of warning messages
    /// about potentially suboptimal usage patterns. This can help users identify
    /// performance issues or incorrect API usage.
    ///
    /// # Returns
    ///
    /// An array of warning messages (empty if no issues detected)
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    ///
    /// const warnings = solver.getDiagnostics();
    /// if (warnings.length > 0) {
    ///   console.warn("Solver usage warnings:");
    ///   warnings.forEach(w => console.warn("  -", w));
    /// }
    /// ```
    #[wasm_bindgen(js_name = getDiagnostics)]
    pub fn get_diagnostics(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check if logic has been set
        if self.ctx.get_option("logic").is_none() {
            warnings.push(
                "No logic set. Consider calling setLogic() to enable optimizations for your problem domain.".to_string()
            );
        }

        // Check if model generation is enabled when it might be needed
        if self.ctx.get_option("produce-models").is_none() {
            warnings.push(
                "Model production not explicitly enabled. If you need models, call setOption('produce-models', 'true').".to_string()
            );
        }

        // Check if unsat core production is enabled when it might be needed
        if self.ctx.get_option("produce-unsat-cores").is_none() {
            warnings.push(
                "Unsat core production not explicitly enabled. If you need unsat cores, call setOption('produce-unsat-cores', 'true').".to_string()
            );
        }

        // Warn about using simplify without a logic
        if self.ctx.get_option("logic").is_none() && !warnings.is_empty() {
            warnings.push(
                "Using solver without setting logic may result in suboptimal performance. Set logic with setLogic() for better results.".to_string()
            );
        }

        warnings
    }

    /// Check if a given usage pattern is recommended
    ///
    /// This method checks whether a specific usage pattern is recommended for
    /// the current solver state. It's useful for validating workflow decisions.
    ///
    /// # Parameters
    ///
    /// * `pattern` - The pattern to check. Supported patterns:
    ///   - `"incremental"` - Check if incremental solving (push/pop) is recommended
    ///   - `"assumptions"` - Check if check-sat-assuming is better than push/pop
    ///   - `"async"` - Check if async operations should be used
    ///
    /// # Returns
    ///
    /// A recommendation message, or empty string if the pattern is fine
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const rec = solver.checkPattern("incremental");
    /// if (rec) {
    ///   console.log("Recommendation:", rec);
    /// }
    /// ```
    #[wasm_bindgen(js_name = checkPattern)]
    pub fn check_pattern(&self, pattern: &str) -> String {
        match pattern {
            "incremental" => {
                "Incremental solving with push/pop is useful for exploring different scenarios. \
                Consider using checkSatAssuming() for temporary constraints instead of push/pop \
                for better performance."
                    .to_string()
            }
            "assumptions" => {
                "Using checkSatAssuming() is more efficient than push/assert/checkSat/pop for \
                temporary constraints, as it avoids modifying the assertion stack."
                    .to_string()
            }
            "async" => {
                "For long-running solver operations in web browsers, use checkSatAsync() or \
                executeAsync() to avoid blocking the UI thread. Consider using a WebWorker \
                for better responsiveness."
                    .to_string()
            }
            "validation" => {
                "Use validateFormula() to check formulas before asserting them, especially when \
                dealing with user input. This provides better error messages than waiting for \
                assertion to fail."
                    .to_string()
            }
            _ => format!(
                "Unknown pattern '{}'. Supported patterns: incremental, assumptions, async, validation",
                pattern
            ),
        }
    }
}

// Private helper methods (no wasm_bindgen — not exported to JS)
impl WasmSolver {
    /// Parse a sort name string into a SortId
    pub(crate) fn parse_sort(
        &mut self,
        sort_name: &str,
    ) -> Result<oxiz_core::sort::SortId, JsValue> {
        match sort_name {
            "Bool" => Ok(self.ctx.terms.sorts.bool_sort),
            "Int" => Ok(self.ctx.terms.sorts.int_sort),
            "Real" => Ok(self.ctx.terms.sorts.real_sort),
            s if s.starts_with("BitVec") => {
                let width_str = s.strip_prefix("BitVec").ok_or_else(|| {
                    WasmError::new(WasmErrorKind::InvalidSort, "Failed to parse BitVec sort")
                })?;

                if width_str.is_empty() {
                    return Err(WasmError::new(
                        WasmErrorKind::InvalidSort,
                        "BitVec sort requires a width (e.g., BitVec32)",
                    )
                    .into());
                }

                let width: u32 = width_str.parse().map_err(|_| {
                    WasmError::new(
                        WasmErrorKind::InvalidSort,
                        format!(
                            "Invalid BitVec width '{}': must be a positive integer",
                            width_str
                        ),
                    )
                })?;

                if width == 0 {
                    return Err(WasmError::new(
                        WasmErrorKind::InvalidSort,
                        "BitVec width must be greater than 0",
                    )
                    .into());
                }

                Ok(self.ctx.terms.sorts.bitvec(width))
            }
            _ => Err(WasmError::new(
                WasmErrorKind::InvalidSort,
                format!(
                    "Unknown sort '{}'. Valid sorts: Bool, Int, Real, BitVecN (where N > 0)",
                    sort_name
                ),
            )
            .into()),
        }
    }
}
