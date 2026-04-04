//! Core solver operations: lifecycle, check-sat, execute, push/pop, cancel.

use crate::WasmSolver;
use crate::async_utils;
use crate::string_utils;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
    /// Create a new solver instance
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        use oxiz_solver::Context;
        Self {
            ctx: Context::new(),
            last_result: None,
            cancelled: false,
        }
    }

    /// Execute an SMT-LIB2 script and return the results
    ///
    /// This method takes a complete SMT-LIB2 script as a string and executes it,
    /// returning the output as a string. This is useful for batch operations or
    /// when you have a complete SMT-LIB2 file to execute.
    ///
    /// # Parameters
    ///
    /// * `script` - An SMT-LIB2 script string
    ///
    /// # Returns
    ///
    /// The output of the script execution as a string
    ///
    /// # Errors
    ///
    /// Returns an error if the script contains syntax errors or invalid commands
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const script = `
    ///   (set-logic QF_LIA)
    ///   (declare-const x Int)
    ///   (assert (> x 0))
    ///   (check-sat)
    /// `;
    /// const result = solver.execute(script);
    /// console.log(result); // outputs: sat
    /// ```
    #[wasm_bindgen]
    pub fn execute(&mut self, script: &str) -> Result<JsValue, JsValue> {
        if string_utils::is_effectively_empty(script) {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Script cannot be empty").into(),
            );
        }

        match self.ctx.execute_script(script) {
            Ok(output) => {
                let result = string_utils::join_lines(&output);
                Ok(JsValue::from_str(&result))
            }
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to execute script: {}", e),
            )
            .into()),
        }
    }

    /// Set the logic for the solver
    ///
    /// This sets the SMT logic to use for the solver. Common logics include:
    /// - `QF_UF` - Quantifier-free uninterpreted functions
    /// - `QF_LIA` - Quantifier-free linear integer arithmetic
    /// - `QF_LRA` - Quantifier-free linear real arithmetic
    /// - `QF_BV` - Quantifier-free bitvectors
    /// - `ALL` - All supported theories
    ///
    /// # Parameters
    ///
    /// * `logic` - The SMT-LIB2 logic name
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// ```
    #[wasm_bindgen(js_name = setLogic)]
    pub fn set_logic(&mut self, logic: &str) {
        self.ctx.set_logic(logic);
    }

    /// Check satisfiability of the current assertions
    ///
    /// This method checks whether the current set of assertions is satisfiable.
    /// It returns one of three possible results:
    /// - `"sat"` - The assertions are satisfiable
    /// - `"unsat"` - The assertions are unsatisfiable
    /// - `"unknown"` - The solver could not determine satisfiability
    ///
    /// After calling this method with a "sat" result, you can call `getModel()`
    /// to get a satisfying assignment. With an "unsat" result, you can call
    /// `getUnsatCore()` to get the unsatisfiable core.
    ///
    /// # Returns
    ///
    /// A string indicating the satisfiability result: "sat", "unsat", or "unknown"
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    /// solver.declareConst("p", "Bool");
    /// solver.assertFormula("p");
    /// const result = solver.checkSat();
    /// if (result === "sat") {
    ///   const model = solver.getModel();
    ///   console.log(model);
    /// }
    /// ```
    #[wasm_bindgen(js_name = checkSat)]
    pub fn check_sat(&mut self) -> String {
        let result = match self.ctx.check_sat() {
            oxiz_solver::SolverResult::Sat => "sat",
            oxiz_solver::SolverResult::Unsat => "unsat",
            oxiz_solver::SolverResult::Unknown => "unknown",
        };
        self.last_result = Some(result.to_string());
        result.to_string()
    }

    /// Check satisfiability under a set of assumptions
    ///
    /// This method checks whether the current assertions are satisfiable under
    /// the given temporary assumptions. The assumptions are only used for this
    /// single check and do not modify the assertion stack.
    ///
    /// This is useful for:
    /// - Incremental solving with different scenarios
    /// - Computing minimal unsatisfiable cores
    /// - Exploring different branches without push/pop overhead
    ///
    /// # Parameters
    ///
    /// * `assumptions` - An array of SMT-LIB2 boolean expressions to assume
    ///
    /// # Returns
    ///
    /// A string indicating the satisfiability result: "sat", "unsat", or "unknown"
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any assumption is empty or malformed
    /// - Any assumption contains syntax errors
    /// - Any assumption references undeclared variables
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    /// solver.declareConst("p", "Bool");
    /// solver.declareConst("q", "Bool");
    /// solver.assertFormula("(or p q)");
    ///
    /// // Check if satisfiable assuming p is true
    /// const result1 = solver.checkSatAssuming(["p"]);
    /// console.log(result1); // "sat"
    ///
    /// // Check if satisfiable assuming both p and q are false
    /// const result2 = solver.checkSatAssuming(["(not p)", "(not q)"]);
    /// console.log(result2); // "unsat"
    /// ```
    #[wasm_bindgen(js_name = checkSatAssuming)]
    pub fn check_sat_assuming(&mut self, assumptions: Vec<String>) -> Result<String, JsValue> {
        if assumptions.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Assumptions array cannot be empty",
            )
            .into());
        }

        // Validate all assumptions are non-empty
        for (idx, assumption) in assumptions.iter().enumerate() {
            if assumption.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Assumption at index {} cannot be empty", idx),
                )
                .into());
            }
        }

        // Build check-sat-assuming command
        let assumptions_str = assumptions.join(" ");
        let script = format!("(check-sat-assuming ({}))", assumptions_str);

        match self.ctx.execute_script(&script) {
            Ok(output) => {
                let result = output.join("");
                // Normalize the result
                let normalized = match result.trim() {
                    "sat" => "sat",
                    "unsat" => "unsat",
                    _ => "unknown",
                };
                self.last_result = Some(normalized.to_string());
                Ok(normalized.to_string())
            }
            Err(e) => Err(WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to check-sat with assumptions: {}", e),
            )
            .into()),
        }
    }

    /// Push a new context level
    ///
    /// Creates a new backtracking point. Assertions and declarations made after
    /// pushing can be undone by calling `pop()`. This is useful for trying
    /// different sets of constraints without resetting the entire solver.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    /// solver.push(); // Create backtracking point
    /// solver.assertFormula("(< x 5)");
    /// console.log(solver.checkSat()); // sat
    /// solver.pop(); // Undo the (< x 5) assertion
    /// ```
    #[wasm_bindgen]
    pub fn push(&mut self) {
        self.ctx.push();
    }

    /// Pop a context level
    ///
    /// Backtracks to the previous context level, undoing all assertions and
    /// declarations made since the last `push()` call.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.push();
    /// solver.assertFormula("(> x 10)");
    /// solver.pop(); // Remove the (> x 10) assertion
    /// ```
    #[wasm_bindgen]
    pub fn pop(&mut self) {
        self.ctx.pop();
    }

    /// Reset the solver completely
    ///
    /// Clears all assertions, declarations, options, and state, returning the
    /// solver to its initial state as if newly constructed.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    /// solver.reset(); // Clear everything
    /// // Solver is now empty, must redeclare and reassert
    /// ```
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.ctx.reset();
        self.last_result = None;
        self.cancelled = false;
    }

    /// Reset only assertions, keeping declarations and options
    ///
    /// Removes all assertions but keeps variable and function declarations
    /// and solver options. This is useful for solving multiple related
    /// problems with the same variables.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    /// solver.checkSat();
    /// solver.resetAssertions(); // Keep x declared, remove assertions
    /// solver.assertFormula("(< x 0)"); // Can still use x
    /// ```
    #[wasm_bindgen(js_name = resetAssertions)]
    pub fn reset_assertions(&mut self) {
        self.ctx.reset_assertions();
        self.last_result = None;
    }

    /// Cancel the current solver operation
    ///
    /// Sets a cancellation flag that can be checked during long-running
    /// operations. Note: This is a hint to the solver and may not take
    /// effect immediately.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// // In a web worker or with async operations
    /// setTimeout(() => solver.cancel(), 5000); // Cancel after 5 seconds
    /// const result = await solver.checkSatAsync();
    /// ```
    #[wasm_bindgen]
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Check if the solver has been cancelled
    ///
    /// # Returns
    ///
    /// `true` if cancellation has been requested, `false` otherwise
    #[wasm_bindgen(js_name = isCancelled)]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Check satisfiability asynchronously
    ///
    /// This is an async version of `checkSat()` that allows the browser to remain
    /// responsive during long-running solver operations. It returns a Promise that
    /// resolves to the satisfiability result.
    ///
    /// # Returns
    ///
    /// A Promise that resolves to a string: "sat", "unsat", or "unknown"
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> x 0)");
    ///
    /// const result = await solver.checkSatAsync();
    /// console.log(result); // "sat"
    ///
    /// if (result === "sat") {
    ///   const model = solver.getModel();
    ///   console.log(model);
    /// }
    /// ```
    #[wasm_bindgen(js_name = checkSatAsync)]
    pub async fn check_sat_async(&mut self) -> String {
        // Yield to event loop before starting
        async_utils::yield_now().await;

        // Perform the actual check-sat operation
        let result = self.check_sat();

        // Yield again after completion to ensure UI responsiveness
        async_utils::yield_now().await;

        result
    }

    /// Execute an SMT-LIB2 script asynchronously
    ///
    /// This is an async version of `execute()` that allows the browser to remain
    /// responsive during execution of complex scripts.
    ///
    /// # Parameters
    ///
    /// * `script` - An SMT-LIB2 script string
    ///
    /// # Returns
    ///
    /// A Promise that resolves to the output string
    ///
    /// # Errors
    ///
    /// Returns a Promise that rejects if the script contains errors
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const script = `
    ///   (set-logic QF_LIA)
    ///   (declare-const x Int)
    ///   (assert (> x 0))
    ///   (check-sat)
    ///   (get-model)
    /// `;
    /// const result = await solver.executeAsync(script);
    /// console.log(result);
    /// ```
    #[wasm_bindgen(js_name = executeAsync)]
    pub async fn execute_async(&mut self, script: String) -> Result<JsValue, JsValue> {
        // Yield to event loop before starting
        async_utils::yield_now().await;

        // Split script into lines and process with periodic yields
        let lines: Vec<&str> = script.lines().collect();
        let total_lines = lines.len();

        // For small scripts (< 10 lines), just execute directly
        if total_lines < 10 {
            let result = self.execute(&script);
            async_utils::yield_now().await;
            return result;
        }

        // For larger scripts, yield periodically
        // Process script in chunks to maintain responsiveness
        let chunk_size = 20; // Process 20 lines before yielding
        let mut result_parts = Vec::new();

        for (i, chunk) in lines.chunks(chunk_size).enumerate() {
            // Yield every 5 chunks (every ~100 lines)
            if i > 0 && i % 5 == 0 {
                async_utils::yield_now().await;

                // Check for cancellation
                if self.cancelled {
                    return Err(JsValue::from_str("Operation cancelled"));
                }
            }

            // Execute this chunk
            let chunk_script = chunk.join("\n");
            match self.execute(&chunk_script) {
                Ok(output) => {
                    if let Some(s) = output.as_string()
                        && !s.trim().is_empty()
                    {
                        result_parts.push(s);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        // Yield before returning final result
        async_utils::yield_now().await;

        Ok(JsValue::from_str(&result_parts.join("\n")))
    }

    /// Execute an SMT-LIB2 script asynchronously with progress callbacks
    ///
    /// This method is similar to `executeAsync()` but also accepts a callback function
    /// that will be invoked periodically with progress updates. This is useful for
    /// long-running operations where you want to show progress to the user.
    ///
    /// # Parameters
    ///
    /// * `script` - An SMT-LIB2 script string
    /// * `progress_callback` - Optional callback function that receives progress updates
    ///   The callback receives two arguments: (current_line, total_lines)
    ///
    /// # Returns
    ///
    /// A Promise that resolves to the output string
    ///
    /// # Errors
    ///
    /// Returns a Promise that rejects if the script contains errors
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const script = `
    ///   (set-logic QF_LIA)
    ///   (declare-const x Int)
    ///   (assert (> x 0))
    ///   (check-sat)
    ///   (get-model)
    /// `;
    ///
    /// const result = await solver.executeWithProgress(script, (current, total) => {
    ///   console.log(`Progress: ${current}/${total} lines processed`);
    ///   document.getElementById('progress').innerText =
    ///     `${Math.round(current / total * 100)}%`;
    /// });
    /// console.log(result);
    /// ```
    #[wasm_bindgen(js_name = executeWithProgress)]
    pub async fn execute_with_progress(
        &mut self,
        script: String,
        progress_callback: Option<js_sys::Function>,
    ) -> Result<JsValue, JsValue> {
        // Yield to event loop before starting
        async_utils::yield_now().await;

        // Split script into lines
        let lines: Vec<&str> = script.lines().collect();
        let total_lines = lines.len();

        // For small scripts, just execute directly
        if total_lines < 10 {
            let result = self.execute(&script);
            if let Some(callback) = progress_callback {
                let this = JsValue::NULL;
                let _ = callback.call2(
                    &this,
                    &JsValue::from(total_lines),
                    &JsValue::from(total_lines),
                );
            }
            async_utils::yield_now().await;
            return result;
        }

        // For larger scripts, process in chunks with progress updates
        let chunk_size = 20;
        let mut result_parts = Vec::new();
        let mut lines_processed = 0;

        for (i, chunk) in lines.chunks(chunk_size).enumerate() {
            // Yield every 5 chunks
            if i > 0 && i % 5 == 0 {
                async_utils::yield_now().await;

                // Check for cancellation
                if self.cancelled {
                    return Err(JsValue::from_str("Operation cancelled"));
                }
            }

            // Execute this chunk
            let chunk_script = chunk.join("\n");
            match self.execute(&chunk_script) {
                Ok(output) => {
                    if let Some(s) = output.as_string()
                        && !s.trim().is_empty()
                    {
                        result_parts.push(s);
                    }
                }
                Err(e) => return Err(e),
            }

            // Update progress
            lines_processed += chunk.len();
            if let Some(ref callback) = progress_callback {
                let this = JsValue::NULL;
                let _ = callback.call2(
                    &this,
                    &JsValue::from(lines_processed),
                    &JsValue::from(total_lines),
                );
            }
        }

        // Final progress update
        if let Some(callback) = progress_callback {
            let this = JsValue::NULL;
            let _ = callback.call2(
                &this,
                &JsValue::from(total_lines),
                &JsValue::from(total_lines),
            );
        }

        // Yield before returning
        async_utils::yield_now().await;

        Ok(JsValue::from_str(&result_parts.join("\n")))
    }
}
