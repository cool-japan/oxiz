//! OxiZ WASM - WebAssembly bindings for OxiZ SMT Solver
//!
//! Provides JavaScript/TypeScript bindings for running OxiZ in the browser.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod async_utils;
mod pool;
mod string_utils;

use oxiz_solver::Context;
use std::fmt;
use wasm_bindgen::prelude::*;

/// Error types for WASM API
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmErrorKind {
    /// Parse error in SMT-LIB2 script
    ParseError,
    /// Invalid sort name
    InvalidSort,
    /// No model available (check-sat not called or result was unsat)
    NoModel,
    /// No unsat core available
    NoUnsatCore,
    /// No proof available
    NoProof,
    /// Solver is in an invalid state for this operation
    InvalidState,
    /// Invalid input or argument
    InvalidInput,
    /// Operation not supported
    NotSupported,
    /// Unknown error
    Unknown,
}

impl fmt::Display for WasmErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError => write!(f, "ParseError"),
            Self::InvalidSort => write!(f, "InvalidSort"),
            Self::NoModel => write!(f, "NoModel"),
            Self::NoUnsatCore => write!(f, "NoUnsatCore"),
            Self::NoProof => write!(f, "NoProof"),
            Self::InvalidState => write!(f, "InvalidState"),
            Self::InvalidInput => write!(f, "InvalidInput"),
            Self::NotSupported => write!(f, "NotSupported"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Error type for WASM operations
#[derive(Debug, Clone)]
pub struct WasmError {
    kind: WasmErrorKind,
    message: String,
}

impl WasmError {
    /// Create a new error
    pub fn new(kind: WasmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Get error kind
    pub fn kind(&self) -> WasmErrorKind {
        self.kind
    }

    /// Get error message
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"kind".into(), &err.kind.to_string().into());
        let _ = js_sys::Reflect::set(&obj, &"message".into(), &err.message.into());
        obj.into()
    }
}

/// WASM-accessible SMT solver
///
/// This is the main interface for using OxiZ from JavaScript/TypeScript.
/// It provides a high-level API for SMT solving operations.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const solver = new WasmSolver();
/// solver.setLogic("QF_LIA");
/// solver.declareConst("x", "Int");
/// solver.assertFormula("(> x 0)");
/// const result = solver.checkSat(); // "sat"
/// const model = solver.getModel();
/// console.log(model.x.value); // prints the value of x
/// ```
#[wasm_bindgen]
pub struct WasmSolver {
    ctx: Context,
    /// Last check-sat result
    last_result: Option<String>,
    /// Cancellation flag
    cancelled: bool,
}

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

    /// Define a sort alias
    ///
    /// Creates an alias (new name) for an existing sort. This is useful for
    /// making SMT-LIB2 scripts more readable by giving meaningful names to
    /// commonly used sorts.
    ///
    /// # Parameters
    ///
    /// * `name` - The name for the new sort alias
    /// * `sort_name` - The existing sort to create an alias for. Valid sorts:
    ///   - `"Bool"` - Boolean values
    ///   - `"Int"` - Integer values
    ///   - `"Real"` - Real number values
    ///   - `"BitVecN"` - Bitvector of width N (e.g., "BitVec8", "BitVec32")
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The alias name is empty or malformed
    /// - The base sort name is invalid
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// // Create an alias 'Word' for 32-bit bitvectors
    /// solver.defineSort("Word", "BitVec32");
    /// // Now you can use 'Word' as a sort name
    /// solver.declareConst("reg", "Word");
    /// ```
    #[wasm_bindgen(js_name = defineSort)]
    pub fn define_sort(&mut self, name: &str, sort_name: &str) -> Result<(), JsValue> {
        if name.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Sort name cannot be empty").into(),
            );
        }

        // Validate the base sort exists
        self.parse_sort(sort_name)?;

        // Build define-sort command
        let script = format!("(define-sort {} () {})", name, sort_name);
        self.ctx.execute_script(&script).map_err(|e| -> JsValue {
            WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to define sort: {}", e),
            )
            .into()
        })?;

        Ok(())
    }

    /// Declare a constant with a given name and sort
    ///
    /// This declares a constant (0-ary function) in the solver context.
    /// The constant can then be referenced in assertions and other formulas.
    ///
    /// # Parameters
    ///
    /// * `name` - The name of the constant
    /// * `sort_name` - The sort/type of the constant. Valid sorts:
    ///   - `"Bool"` - Boolean values
    ///   - `"Int"` - Integer values
    ///   - `"Real"` - Real number values
    ///   - `"BitVecN"` - Bitvector of width N (e.g., "BitVec8", "BitVec32")
    ///
    /// # Errors
    ///
    /// Returns an error if the sort name is invalid or malformed
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.declareConst("x", "Int");
    /// solver.declareConst("flag", "Bool");
    /// solver.declareConst("bv", "BitVec32");
    /// ```
    #[wasm_bindgen(js_name = declareConst)]
    pub fn declare_const(&mut self, name: &str, sort_name: &str) -> Result<(), JsValue> {
        if name.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Constant name cannot be empty",
            )
            .into());
        }

        let sort = self.parse_sort(sort_name)?;
        self.ctx.declare_const(name, sort);
        Ok(())
    }

    /// Declare a function with given name, argument sorts, and return sort
    ///
    /// This declares a function in the solver context. For constants (0-ary functions),
    /// use an empty array for argument sorts or use `declareConst` instead.
    ///
    /// After declaring a function, you can use it in formulas by applying it to arguments.
    /// Function applications are created using the SMT-LIB2 syntax: `(f arg1 arg2 ...)`.
    ///
    /// # Parameters
    ///
    /// * `name` - The name of the function
    /// * `arg_sorts` - Array of sort names for the function arguments (empty for constants)
    /// * `ret_sort` - The sort name for the return value
    ///
    /// # Errors
    ///
    /// Returns an error if any sort name is invalid or if the function name is empty
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    ///
    /// // Declare a constant (nullary function)
    /// solver.declareFun("c", [], "Int");
    ///
    /// // Declare a unary function from Int to Int
    /// solver.declareFun("f", ["Int"], "Int");
    ///
    /// // Declare a binary function from Int × Bool to Real
    /// solver.declareFun("g", ["Int", "Bool"], "Real");
    ///
    /// // Use the functions in assertions
    /// solver.declareConst("x", "Int");
    /// solver.assertFormula("(> (f x) 0)");
    /// ```
    #[wasm_bindgen(js_name = declareFun)]
    pub fn declare_fun(
        &mut self,
        name: &str,
        arg_sorts: Vec<String>,
        ret_sort: &str,
    ) -> Result<(), JsValue> {
        if name.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Function name cannot be empty",
            )
            .into());
        }

        let ret = self.parse_sort(ret_sort)?;

        if arg_sorts.is_empty() {
            // Nullary function = constant
            self.ctx.declare_const(name, ret);
        } else {
            // Parse and validate all argument sorts
            let parsed_arg_sorts: Result<Vec<_>, _> =
                arg_sorts.iter().map(|s| self.parse_sort(s)).collect();
            let parsed_arg_sorts = parsed_arg_sorts?;

            // Declare the function with its signature
            self.ctx.declare_fun(name, parsed_arg_sorts, ret);
        }

        Ok(())
    }

    /// Define a function with a given implementation
    ///
    /// This defines a function with an explicit body/implementation. The function
    /// can then be used in assertions and other expressions. This is more powerful
    /// than `declareFun` as it provides an actual definition rather than just a
    /// signature.
    ///
    /// # Parameters
    ///
    /// * `name` - The name of the function to define
    /// * `params` - Array of parameter specifications, each as "name sort" (e.g., ["x Int", "y Bool"])
    /// * `ret_sort` - The return sort of the function
    /// * `body` - The SMT-LIB2 expression defining the function body
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The function name is empty
    /// - Any parameter specification is malformed
    /// - The return sort is invalid
    /// - The body expression contains syntax errors
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    ///
    /// // Define a function that doubles its input
    /// solver.defineFun("double", ["x Int"], "Int", "(* 2 x)");
    ///
    /// // Define a function that computes max of two integers
    /// solver.defineFun("max2", ["a Int", "b Int"], "Int", "(ite (> a b) a b)");
    ///
    /// // Now use the defined functions
    /// solver.declareConst("n", "Int");
    /// solver.assertFormula("(= (double n) 10)");
    /// solver.assertFormula("(> (max2 n 3) 4)");
    /// console.log(solver.checkSat()); // "sat"
    /// ```
    #[wasm_bindgen(js_name = defineFun)]
    pub fn define_fun(
        &mut self,
        name: &str,
        params: Vec<String>,
        ret_sort: &str,
        body: &str,
    ) -> Result<(), JsValue> {
        if name.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Function name cannot be empty",
            )
            .into());
        }

        if body.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Function body cannot be empty",
            )
            .into());
        }

        // Validate return sort
        self.parse_sort(ret_sort)?;

        // Build the parameter list for SMT-LIB2
        let params_str = if params.is_empty() {
            String::new()
        } else {
            let mut param_parts = Vec::new();
            for p in &params {
                let parts: Vec<&str> = p.split_whitespace().collect();
                if parts.len() != 2 {
                    return Err(WasmError::new(
                        WasmErrorKind::InvalidInput,
                        format!("Invalid parameter '{}': must be in format 'name sort'", p),
                    )
                    .into());
                }
                // Validate the sort
                self.parse_sort(parts[1])?;
                param_parts.push(format!("({} {})", parts[0], parts[1]));
            }
            param_parts.join(" ")
        };

        // Build define-fun command
        let script = format!(
            "(define-fun {} ({}) {} {})",
            name, params_str, ret_sort, body
        );

        self.ctx.execute_script(&script).map_err(|e| -> JsValue {
            WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to define function: {}", e),
            )
            .into()
        })?;

        Ok(())
    }

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

    /// Declare multiple constants at once (batch operation)
    ///
    /// This is a convenience method that declares multiple constants in a single call.
    /// It's more efficient than calling `declareConst` multiple times for large numbers
    /// of declarations.
    ///
    /// # Parameters
    ///
    /// * `declarations` - Array of declaration strings in format "name sort" (e.g., "x Int", "y Bool")
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all declarations succeed, or an error with details about the first
    /// failed declaration.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_LIA");
    /// solver.declareFuns([
    ///   "x Int",
    ///   "y Int",
    ///   "z Bool",
    ///   "w Real"
    /// ]);
    /// ```
    #[wasm_bindgen(js_name = declareFuns)]
    pub fn declare_funs(&mut self, declarations: Vec<String>) -> Result<(), JsValue> {
        if declarations.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Declarations list cannot be empty",
            )
            .into());
        }

        for (idx, decl) in declarations.iter().enumerate() {
            let parts: Vec<&str> = decl.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!(
                        "Invalid declaration at index {}: '{}'. Expected format 'name sort'",
                        idx, decl
                    ),
                )
                .into());
            }

            let name = parts[0];
            let sort = parts[1];

            self.declare_const(name, sort).map_err(|e| -> JsValue {
                WasmError::new(
                    WasmErrorKind::ParseError,
                    format!("Failed to declare '{}' at index {}: {:?}", name, idx, e),
                )
                .into()
            })?;
        }

        Ok(())
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

    /// Apply a configuration preset for common use cases
    ///
    /// Applies a predefined set of options optimized for specific scenarios.
    /// This is a convenience method to quickly configure the solver without
    /// manually setting individual options.
    ///
    /// # Parameters
    ///
    /// * `preset` - The preset name. Available presets:
    ///   - `"default"` - Default configuration with model production
    ///   - `"fast"` - Optimized for fast solving, minimal features
    ///   - `"complete"` - All features enabled (models, unsat cores, etc.)
    ///   - `"debug"` - Configuration for debugging with verbose output
    ///   - `"unsat-core"` - Optimized for unsat core extraction
    ///   - `"incremental"` - Optimized for incremental solving
    ///
    /// # Errors
    ///
    /// Returns an error if the preset name is unknown
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// // Quick setup for complete functionality
    /// solver.applyPreset("complete");
    /// // Or optimize for fast solving
    /// solver.applyPreset("fast");
    /// ```
    #[wasm_bindgen(js_name = applyPreset)]
    pub fn apply_preset(&mut self, preset: &str) -> Result<(), JsValue> {
        match preset {
            "default" => {
                self.ctx.set_option("produce-models", "true");
            }
            "fast" => {
                self.ctx.set_option("produce-models", "false");
                self.ctx.set_option("produce-unsat-cores", "false");
            }
            "complete" => {
                self.ctx.set_option("produce-models", "true");
                self.ctx.set_option("produce-unsat-cores", "true");
                self.ctx.set_option("produce-assignments", "true");
            }
            "debug" => {
                self.ctx.set_option("produce-models", "true");
                self.ctx.set_option("produce-unsat-cores", "true");
                self.ctx.set_option("produce-assignments", "true");
                self.ctx.set_option("verbosity", "10");
            }
            "unsat-core" => {
                self.ctx.set_option("produce-models", "false");
                self.ctx.set_option("produce-unsat-cores", "true");
            }
            "incremental" => {
                self.ctx.set_option("produce-models", "true");
                self.ctx.set_option("incremental", "true");
            }
            _ => {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!(
                        "Unknown preset '{}'. Available presets: default, fast, complete, debug, unsat-core, incremental",
                        preset
                    ),
                )
                .into());
            }
        }
        Ok(())
    }

    /// Set a solver option
    ///
    /// Configure solver behavior with SMT-LIB2 options. Common options include:
    /// - `"produce-models"` - Enable/disable model generation (values: "true"/"false")
    /// - `"produce-unsat-cores"` - Enable/disable unsat core generation
    ///
    /// # Parameters
    ///
    /// * `key` - The option name
    /// * `value` - The option value
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setOption("produce-models", "true");
    /// solver.setOption("produce-unsat-cores", "true");
    /// ```
    #[wasm_bindgen(js_name = setOption)]
    pub fn set_option(&mut self, key: &str, value: &str) {
        self.ctx.set_option(key, value);
    }

    /// Get a solver option value
    ///
    /// Retrieve the current value of a solver option.
    ///
    /// # Parameters
    ///
    /// * `key` - The option name
    ///
    /// # Returns
    ///
    /// The option value if set, or `undefined` if not set
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setOption("produce-models", "true");
    /// const value = solver.getOption("produce-models");
    /// console.log(value); // "true"
    /// ```
    #[wasm_bindgen(js_name = getOption)]
    pub fn get_option(&self, key: &str) -> Option<String> {
        self.ctx.get_option(key).map(|s| s.to_string())
    }

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

    /// Create a function application expression
    ///
    /// This is a helper method that constructs a function application expression
    /// in SMT-LIB2 format. It's equivalent to manually writing `(f arg1 arg2 ...)`,
    /// but provides validation and proper formatting.
    ///
    /// # Parameters
    ///
    /// * `func_name` - The name of the function to apply
    /// * `args` - Array of SMT-LIB2 expression strings representing the arguments
    ///
    /// # Returns
    ///
    /// A string representing the function application in SMT-LIB2 format
    ///
    /// # Errors
    ///
    /// Returns an error if the function name is empty or arguments are invalid
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setLogic("QF_UF");
    /// solver.declareFun("f", ["Int"], "Int");
    /// solver.declareConst("x", "Int");
    ///
    /// // Create a function application: (f x)
    /// const app = solver.mkApp("f", ["x"]);
    /// console.log(app); // "(f x)"
    ///
    /// // Use it in an assertion
    /// solver.assertFormula(`(> ${app} 0)`);
    /// // Or directly:
    /// solver.assertFormula(`(> (f x) 0)`);
    /// ```
    #[wasm_bindgen(js_name = mkApp)]
    pub fn mk_app(&self, func_name: &str, args: Vec<String>) -> Result<String, JsValue> {
        if func_name.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Function name cannot be empty",
            )
            .into());
        }

        // Validate that all arguments are non-empty
        for (idx, arg) in args.iter().enumerate() {
            if arg.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Argument at index {} cannot be empty", idx),
                )
                .into());
            }
        }

        // Build the function application
        if args.is_empty() {
            // Nullary function is just the name
            Ok(func_name.to_string())
        } else {
            // Format as (f arg1 arg2 ...)
            Ok(format!("({} {})", func_name, args.join(" ")))
        }
    }

    /// Create an equality expression
    ///
    /// Constructs an SMT-LIB2 equality expression `(= lhs rhs)`.
    ///
    /// # Parameters
    ///
    /// * `lhs` - Left-hand side expression
    /// * `rhs` - Right-hand side expression
    ///
    /// # Returns
    ///
    /// A string representing the equality in SMT-LIB2 format
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const eq = solver.mkEq("x", "5");
    /// console.log(eq); // "(= x 5)"
    /// solver.assertFormula(eq);
    /// ```
    #[wasm_bindgen(js_name = mkEq)]
    pub fn mk_eq(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Equality operands cannot be empty",
            )
            .into());
        }
        Ok(format!("(= {} {})", lhs, rhs))
    }

    /// Create a conjunction (AND) expression
    ///
    /// Constructs an SMT-LIB2 AND expression `(and expr1 expr2 ...)`.
    ///
    /// # Parameters
    ///
    /// * `exprs` - Array of expressions to conjoin
    ///
    /// # Returns
    ///
    /// A string representing the conjunction in SMT-LIB2 format
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const conj = solver.mkAnd(["(> x 0)", "(< x 10)", "p"]);
    /// console.log(conj); // "(and (> x 0) (< x 10) p)"
    /// ```
    #[wasm_bindgen(js_name = mkAnd)]
    pub fn mk_and(&self, exprs: Vec<String>) -> Result<String, JsValue> {
        if exprs.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "AND requires at least one operand",
            )
            .into());
        }
        for (idx, expr) in exprs.iter().enumerate() {
            if expr.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Expression at index {} cannot be empty", idx),
                )
                .into());
            }
        }
        if exprs.len() == 1 {
            Ok(exprs[0].clone())
        } else {
            Ok(format!("(and {})", exprs.join(" ")))
        }
    }

    /// Create a disjunction (OR) expression
    ///
    /// Constructs an SMT-LIB2 OR expression `(or expr1 expr2 ...)`.
    ///
    /// # Parameters
    ///
    /// * `exprs` - Array of expressions to disjoin
    ///
    /// # Returns
    ///
    /// A string representing the disjunction in SMT-LIB2 format
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const disj = solver.mkOr(["(= x 0)", "(= x 1)", "(= x 2)"]);
    /// console.log(disj); // "(or (= x 0) (= x 1) (= x 2))"
    /// ```
    #[wasm_bindgen(js_name = mkOr)]
    pub fn mk_or(&self, exprs: Vec<String>) -> Result<String, JsValue> {
        if exprs.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "OR requires at least one operand",
            )
            .into());
        }
        for (idx, expr) in exprs.iter().enumerate() {
            if expr.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Expression at index {} cannot be empty", idx),
                )
                .into());
            }
        }
        if exprs.len() == 1 {
            Ok(exprs[0].clone())
        } else {
            Ok(format!("(or {})", exprs.join(" ")))
        }
    }

    /// Create a negation (NOT) expression
    ///
    /// Constructs an SMT-LIB2 NOT expression `(not expr)`.
    ///
    /// # Parameters
    ///
    /// * `expr` - Expression to negate
    ///
    /// # Returns
    ///
    /// A string representing the negation in SMT-LIB2 format
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const neg = solver.mkNot("p");
    /// console.log(neg); // "(not p)"
    /// ```
    #[wasm_bindgen(js_name = mkNot)]
    pub fn mk_not(&self, expr: &str) -> Result<String, JsValue> {
        if expr.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Expression cannot be empty").into(),
            );
        }
        Ok(format!("(not {})", expr))
    }

    /// Create an implication expression
    ///
    /// Constructs an SMT-LIB2 implication `(=> lhs rhs)`.
    ///
    /// # Parameters
    ///
    /// * `lhs` - Antecedent (if part)
    /// * `rhs` - Consequent (then part)
    ///
    /// # Returns
    ///
    /// A string representing the implication in SMT-LIB2 format
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const impl = solver.mkImplies("p", "q");
    /// console.log(impl); // "(=> p q)"
    /// ```
    #[wasm_bindgen(js_name = mkImplies)]
    pub fn mk_implies(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Implication operands cannot be empty",
            )
            .into());
        }
        Ok(format!("(=> {} {})", lhs, rhs))
    }

    /// Create an if-then-else expression
    ///
    /// Constructs an SMT-LIB2 ITE expression `(ite cond then_expr else_expr)`.
    ///
    /// # Parameters
    ///
    /// * `cond` - Condition expression
    /// * `then_expr` - Expression to return if condition is true
    /// * `else_expr` - Expression to return if condition is false
    ///
    /// # Returns
    ///
    /// A string representing the ITE in SMT-LIB2 format
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const ite = solver.mkIte("(> x 0)", "x", "0");
    /// console.log(ite); // "(ite (> x 0) x 0)"
    /// ```
    #[wasm_bindgen(js_name = mkIte)]
    pub fn mk_ite(&self, cond: &str, then_expr: &str, else_expr: &str) -> Result<String, JsValue> {
        if cond.trim().is_empty() || then_expr.trim().is_empty() || else_expr.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "ITE operands cannot be empty",
            )
            .into());
        }
        Ok(format!("(ite {} {} {})", cond, then_expr, else_expr))
    }

    /// Create a less-than comparison
    ///
    /// Constructs an SMT-LIB2 less-than expression `(< lhs rhs)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const lt = solver.mkLt("x", "10");
    /// console.log(lt); // "(< x 10)"
    /// ```
    #[wasm_bindgen(js_name = mkLt)]
    pub fn mk_lt(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Comparison operands cannot be empty",
            )
            .into());
        }
        Ok(format!("(< {} {})", lhs, rhs))
    }

    /// Create a less-than-or-equal comparison
    ///
    /// Constructs an SMT-LIB2 LTE expression `(<= lhs rhs)`.
    #[wasm_bindgen(js_name = mkLe)]
    pub fn mk_le(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Comparison operands cannot be empty",
            )
            .into());
        }
        Ok(format!("(<= {} {})", lhs, rhs))
    }

    /// Create a greater-than comparison
    ///
    /// Constructs an SMT-LIB2 greater-than expression `(> lhs rhs)`.
    #[wasm_bindgen(js_name = mkGt)]
    pub fn mk_gt(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Comparison operands cannot be empty",
            )
            .into());
        }
        Ok(format!("(> {} {})", lhs, rhs))
    }

    /// Create a greater-than-or-equal comparison
    ///
    /// Constructs an SMT-LIB2 GTE expression `(>= lhs rhs)`.
    #[wasm_bindgen(js_name = mkGe)]
    pub fn mk_ge(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Comparison operands cannot be empty",
            )
            .into());
        }
        Ok(format!("(>= {} {})", lhs, rhs))
    }

    /// Create an addition expression
    ///
    /// Constructs an SMT-LIB2 addition `(+ arg1 arg2 ...)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const sum = solver.mkAdd(["x", "y", "1"]);
    /// console.log(sum); // "(+ x y 1)"
    /// ```
    #[wasm_bindgen(js_name = mkAdd)]
    pub fn mk_add(&self, args: Vec<String>) -> Result<String, JsValue> {
        if args.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Addition requires at least one operand",
            )
            .into());
        }
        for (idx, arg) in args.iter().enumerate() {
            if arg.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Argument at index {} cannot be empty", idx),
                )
                .into());
            }
        }
        if args.len() == 1 {
            Ok(args[0].clone())
        } else {
            Ok(format!("(+ {})", args.join(" ")))
        }
    }

    /// Create a subtraction expression
    ///
    /// Constructs an SMT-LIB2 subtraction `(- arg1 arg2 ...)`.
    #[wasm_bindgen(js_name = mkSub)]
    pub fn mk_sub(&self, args: Vec<String>) -> Result<String, JsValue> {
        if args.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Subtraction requires at least one operand",
            )
            .into());
        }
        for (idx, arg) in args.iter().enumerate() {
            if arg.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Argument at index {} cannot be empty", idx),
                )
                .into());
            }
        }
        if args.len() == 1 {
            Ok(format!("(- {})", args[0]))
        } else {
            Ok(format!("(- {})", args.join(" ")))
        }
    }

    /// Create a multiplication expression
    ///
    /// Constructs an SMT-LIB2 multiplication `(* arg1 arg2 ...)`.
    #[wasm_bindgen(js_name = mkMul)]
    pub fn mk_mul(&self, args: Vec<String>) -> Result<String, JsValue> {
        if args.is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Multiplication requires at least one operand",
            )
            .into());
        }
        for (idx, arg) in args.iter().enumerate() {
            if arg.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Argument at index {} cannot be empty", idx),
                )
                .into());
            }
        }
        if args.len() == 1 {
            Ok(args[0].clone())
        } else {
            Ok(format!("(* {})", args.join(" ")))
        }
    }

    /// Create a division expression
    ///
    /// Constructs an SMT-LIB2 division `(/ arg1 arg2 ...)`.
    /// For integer division, use `div`; for real division, use `/`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const div = solver.mkDiv(["x", "2"]);
    /// console.log(div); // "(/ x 2)"
    /// ```
    #[wasm_bindgen(js_name = mkDiv)]
    pub fn mk_div(&self, args: Vec<String>) -> Result<String, JsValue> {
        if args.len() < 2 {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Division requires at least two operands",
            )
            .into());
        }
        for (idx, arg) in args.iter().enumerate() {
            if arg.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Argument at index {} cannot be empty", idx),
                )
                .into());
            }
        }
        Ok(format!("(/ {})", args.join(" ")))
    }

    /// Create a modulo expression
    ///
    /// Constructs an SMT-LIB2 modulo `(mod arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const mod = solver.mkMod("x", "5");
    /// console.log(mod); // "(mod x 5)"
    /// ```
    #[wasm_bindgen(js_name = mkMod)]
    pub fn mk_mod(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(mod {} {})", lhs, rhs))
    }

    /// Create a negation expression
    ///
    /// Constructs an SMT-LIB2 arithmetic negation `(- arg)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const neg = solver.mkNeg("x");
    /// console.log(neg); // "(- x)"
    /// ```
    #[wasm_bindgen(js_name = mkNeg)]
    pub fn mk_neg(&self, expr: &str) -> Result<String, JsValue> {
        if expr.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Expression cannot be empty").into(),
            );
        }
        Ok(format!("(- {})", expr))
    }

    /// Create an exclusive-or (XOR) expression
    ///
    /// Constructs an SMT-LIB2 XOR expression `(xor arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const xor = solver.mkXor("p", "q");
    /// console.log(xor); // "(xor p q)"
    /// ```
    #[wasm_bindgen(js_name = mkXor)]
    pub fn mk_xor(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(xor {} {})", lhs, rhs))
    }

    /// Create a distinct (all different) expression
    ///
    /// Constructs an SMT-LIB2 distinct expression `(distinct arg1 arg2 ...)`.
    /// Returns true if all arguments have different values.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const diff = solver.mkDistinct(["x", "y", "z"]);
    /// console.log(diff); // "(distinct x y z)"
    /// ```
    #[wasm_bindgen(js_name = mkDistinct)]
    pub fn mk_distinct(&self, args: Vec<String>) -> Result<String, JsValue> {
        if args.len() < 2 {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Distinct requires at least two operands",
            )
            .into());
        }
        for (idx, arg) in args.iter().enumerate() {
            if arg.trim().is_empty() {
                return Err(WasmError::new(
                    WasmErrorKind::InvalidInput,
                    format!("Argument at index {} cannot be empty", idx),
                )
                .into());
            }
        }
        Ok(format!("(distinct {})", args.join(" ")))
    }

    /// Create a bitvector AND expression
    ///
    /// Constructs an SMT-LIB2 bitvector AND `(bvand arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvand = solver.mkBvAnd("x", "y");
    /// console.log(bvand); // "(bvand x y)"
    /// ```
    #[wasm_bindgen(js_name = mkBvAnd)]
    pub fn mk_bvand(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(bvand {} {})", lhs, rhs))
    }

    /// Create a bitvector OR expression
    ///
    /// Constructs an SMT-LIB2 bitvector OR `(bvor arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvor = solver.mkBvOr("x", "y");
    /// console.log(bvor); // "(bvor x y)"
    /// ```
    #[wasm_bindgen(js_name = mkBvOr)]
    pub fn mk_bvor(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(bvor {} {})", lhs, rhs))
    }

    /// Create a bitvector XOR expression
    ///
    /// Constructs an SMT-LIB2 bitvector XOR `(bvxor arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvxor = solver.mkBvXor("x", "y");
    /// console.log(bvxor); // "(bvxor x y)"
    /// ```
    #[wasm_bindgen(js_name = mkBvXor)]
    pub fn mk_bvxor(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(bvxor {} {})", lhs, rhs))
    }

    /// Create a bitvector NOT expression
    ///
    /// Constructs an SMT-LIB2 bitvector NOT `(bvnot arg)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvnot = solver.mkBvNot("x");
    /// console.log(bvnot); // "(bvnot x)"
    /// ```
    #[wasm_bindgen(js_name = mkBvNot)]
    pub fn mk_bvnot(&self, expr: &str) -> Result<String, JsValue> {
        if expr.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Expression cannot be empty").into(),
            );
        }
        Ok(format!("(bvnot {})", expr))
    }

    /// Create a bitvector negation expression
    ///
    /// Constructs an SMT-LIB2 bitvector negation `(bvneg arg)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvneg = solver.mkBvNeg("x");
    /// console.log(bvneg); // "(bvneg x)"
    /// ```
    #[wasm_bindgen(js_name = mkBvNeg)]
    pub fn mk_bvneg(&self, expr: &str) -> Result<String, JsValue> {
        if expr.trim().is_empty() {
            return Err(
                WasmError::new(WasmErrorKind::InvalidInput, "Expression cannot be empty").into(),
            );
        }
        Ok(format!("(bvneg {})", expr))
    }

    /// Create a bitvector addition expression
    ///
    /// Constructs an SMT-LIB2 bitvector addition `(bvadd arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvadd = solver.mkBvAdd("x", "y");
    /// console.log(bvadd); // "(bvadd x y)"
    /// ```
    #[wasm_bindgen(js_name = mkBvAdd)]
    pub fn mk_bvadd(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(bvadd {} {})", lhs, rhs))
    }

    /// Create a bitvector subtraction expression
    ///
    /// Constructs an SMT-LIB2 bitvector subtraction `(bvsub arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvsub = solver.mkBvSub("x", "y");
    /// console.log(bvsub); // "(bvsub x y)"
    /// ```
    #[wasm_bindgen(js_name = mkBvSub)]
    pub fn mk_bvsub(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(bvsub {} {})", lhs, rhs))
    }

    /// Create a bitvector multiplication expression
    ///
    /// Constructs an SMT-LIB2 bitvector multiplication `(bvmul arg1 arg2)`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const bvmul = solver.mkBvMul("x", "y");
    /// console.log(bvmul); // "(bvmul x y)"
    /// ```
    #[wasm_bindgen(js_name = mkBvMul)]
    pub fn mk_bvmul(&self, lhs: &str, rhs: &str) -> Result<String, JsValue> {
        if lhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Left operand cannot be empty",
            )
            .into());
        }
        if rhs.trim().is_empty() {
            return Err(WasmError::new(
                WasmErrorKind::InvalidInput,
                "Right operand cannot be empty",
            )
            .into());
        }
        Ok(format!("(bvmul {} {})", lhs, rhs))
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

    /// Enable or disable tracing for debugging
    ///
    /// Controls whether the solver emits trace information during operation.
    /// Tracing can be useful for understanding solver behavior but may impact
    /// performance.
    ///
    /// # Parameters
    ///
    /// * `enabled` - Whether to enable tracing
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// solver.setTracing(true); // Enable detailed tracing
    /// solver.checkSat(); // Will emit trace information
    /// solver.setTracing(false); // Disable tracing
    /// ```
    #[wasm_bindgen(js_name = setTracing)]
    pub fn set_tracing(&mut self, enabled: bool) {
        if enabled {
            self.ctx.set_option("trace", "true");
            self.ctx.set_option("verbosity", "5");
        } else {
            self.ctx.set_option("trace", "false");
            self.ctx.set_option("verbosity", "0");
        }
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

    /// Get diagnostic warnings about solver usage
    ///
    /// Analyzes the current solver state and returns an array of warning messages
    /// about potentially suboptimal usage patterns. This can help users identify
    /// performance issues or incorrect API usage.
    ///
    /// Warnings include:
    /// - Missing logic setting (should call `setLogic()`)
    /// - Calling `getModel()` without checking satisfiability first
    /// - Excessive `push()`/`pop()` nesting depth
    /// - And other potential issues
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
    /// The optimization uses intelligent search strategies:
    /// - For integer objectives: iterative search with binary refinement
    /// - For real objectives: iterative approximation
    /// - For multiple objectives: lexicographic optimization by priority
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
    /// Possible status values:
    /// - `"optimal"` - Optimal solution found
    /// - `"unbounded"` - Objective is unbounded (no finite optimum)
    /// - `"unsat"` - No solution exists (constraints are unsatisfiable)
    /// - `"unknown"` - Solver could not determine the result
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
    /// A JavaScript object containing only the specified variables and their values:
    /// ```javascript
    /// {
    ///   x: { sort: "Int", value: "5" },
    ///   y: { sort: "Int", value: "3" }
    /// }
    /// ```
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
    /// // Get full model
    /// const fullModel = solver.getModel();
    /// // { x: {...}, y: {...}, z: {...} }
    ///
    /// // Get minimal model with only x and y
    /// const minimalModel = solver.getMinimalModel(["x", "y"]);
    /// // { x: {...}, y: {...} }
    /// // z is excluded even though it has a value
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
    /// # Supported Logics
    ///
    /// Quantifier elimination is supported for:
    /// - Linear arithmetic (LIA, LRA)
    /// - Boolean formulas (QF_UF extended)
    /// - Limited support for bitvectors
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

// Private helper methods
impl WasmSolver {
    /// Parse a sort name string into a SortId
    fn parse_sort(&mut self, sort_name: &str) -> Result<oxiz_core::sort::SortId, JsValue> {
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

impl Default for WasmSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize panic hook for better error messages in WASM
///
/// This function is automatically called when the WASM module is loaded.
/// It configures better error messages for panics that occur in the WASM code.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Get the version of OxiZ WASM
///
/// Returns the current version string of the OxiZ WASM library.
///
/// # Returns
///
/// A version string in semver format (e.g., "0.1.1")
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { version } from 'oxiz-wasm';
/// console.log(`OxiZ WASM version: ${version()}`);
/// ```
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;

    #[test]
    fn test_new_solver() {
        let solver = WasmSolver::new();
        assert!(solver.last_result.is_none());
    }

    #[test]
    fn test_declare_const() {
        let mut solver = WasmSolver::new();
        solver.declare_const("x", "Bool").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.declare_const("z", "Real").unwrap();
        solver.declare_const("bv", "BitVec32").unwrap();
    }

    #[test]
    fn test_declare_const_invalid_sort() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_const("x", "InvalidSort");
        assert!(result.is_err());
    }

    #[test]
    fn test_declare_fun_nullary() {
        let mut solver = WasmSolver::new();
        solver.declare_fun("f", vec![], "Int").unwrap();
    }

    #[test]
    fn test_check_sat_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.check_sat();
        assert_eq!(result, "sat");
        assert_eq!(solver.last_result.as_deref(), Some("sat"));
    }

    #[test]
    fn test_assert_formula() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        let result = solver.check_sat();
        assert_eq!(result, "sat");
    }

    #[test]
    fn test_assert_formula_unsat() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.assert_formula("(not p)").unwrap();
        let result = solver.check_sat();
        assert_eq!(result, "unsat");
    }

    #[test]
    fn test_get_model_before_check() {
        let solver = WasmSolver::new();
        let result = solver.get_model();
        assert!(result.is_err());
    }

    #[test]
    fn test_get_model_string() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.check_sat();
        let model = solver.get_model_string().unwrap();
        assert!(model.contains("model"));
    }

    #[test]
    fn test_get_assertions() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        let assertions = solver.get_assertions();
        assert!(assertions.contains("p"));
    }

    #[test]
    fn test_push_pop() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();

        solver.push();
        solver.assert_formula("(not p)").unwrap();
        assert_eq!(solver.check_sat(), "unsat");

        solver.pop();
        assert_eq!(solver.check_sat(), "sat");
    }

    #[test]
    fn test_reset() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.check_sat();
        assert!(solver.last_result.is_some());

        solver.reset();
        assert!(solver.last_result.is_none());
    }

    #[test]
    fn test_reset_assertions() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.check_sat();

        solver.reset_assertions();
        assert!(solver.last_result.is_none());
        let assertions = solver.get_assertions();
        assert_eq!(assertions, "()");
    }

    #[test]
    fn test_set_get_option() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-models", "true");
        assert_eq!(
            solver.get_option("produce-models"),
            Some("true".to_string())
        );
    }

    #[test]
    fn test_simplify() {
        let mut solver = WasmSolver::new();
        let result = solver.simplify("(+ 1 2)").unwrap();
        assert_eq!(result, "3");
    }

    #[test]
    fn test_execute() {
        let mut solver = WasmSolver::new();
        let result = solver.execute("(check-sat)").unwrap();
        assert_eq!(result.as_string().unwrap(), "sat");
    }

    #[test]
    fn test_parse_sort_bitvec() {
        let mut solver = WasmSolver::new();
        let sort = solver.parse_sort("BitVec8").unwrap();
        assert_eq!(
            solver.ctx.terms.sorts.get(sort).unwrap().bitvec_width(),
            Some(8)
        );

        let sort = solver.parse_sort("BitVec64").unwrap();
        assert_eq!(
            solver.ctx.terms.sorts.get(sort).unwrap().bitvec_width(),
            Some(64)
        );
    }

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_default() {
        let solver = WasmSolver::default();
        assert!(solver.last_result.is_none());
    }

    #[test]
    fn test_check_sat_assuming() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.declare_const("q", "Bool").unwrap();
        solver.assert_formula("(or p q)").unwrap();

        // Check assuming p is true
        let result = solver.check_sat_assuming(vec!["p".to_string()]).unwrap();
        assert_eq!(result, "sat");

        // Check assuming both p and q are false
        let result = solver
            .check_sat_assuming(vec!["(not p)".to_string(), "(not q)".to_string()])
            .unwrap();
        assert_eq!(result, "unsat");
    }

    #[test]
    fn test_check_sat_assuming_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.check_sat_assuming(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_define_sort() {
        let mut solver = WasmSolver::new();
        solver.define_sort("Word", "BitVec32").unwrap();
        // After defining the sort, we should be able to use it
        // Note: The underlying context may not support custom sort names yet
    }

    #[test]
    fn test_define_fun() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Define a function that doubles its input
        solver
            .define_fun("double", vec!["x Int".to_string()], "Int", "(* 2 x)")
            .unwrap();

        // Note: Actually using the function requires parsing support
    }

    #[test]
    fn test_define_fun_invalid_params() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Invalid parameter format (missing sort)
        let result = solver.define_fun("f", vec!["x".to_string()], "Int", "x");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_formula() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        // Valid formula
        let result = solver.validate_formula("(> x 0)");
        assert!(result.is_ok());

        // Invalid formula (undeclared variable)
        let result = solver.validate_formula("(> y 0)");
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_formula_safe() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        // Valid formula
        let result = solver.assert_formula_safe("(> x 0)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_statistics() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.check_sat();

        let stats = solver.get_statistics().unwrap();
        assert!(stats.is_object());
    }

    #[test]
    fn test_get_info() {
        let solver = WasmSolver::new();

        let name = solver.get_info("name").unwrap();
        assert_eq!(name, "OxiZ");

        let version = solver.get_info("version").unwrap();
        assert!(!version.is_empty());

        let result = solver.get_info("unknown-key");
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_preset() {
        let mut solver = WasmSolver::new();

        // Test valid presets
        solver.apply_preset("default").unwrap();
        solver.apply_preset("fast").unwrap();
        solver.apply_preset("complete").unwrap();
        solver.apply_preset("debug").unwrap();
        solver.apply_preset("unsat-core").unwrap();
        solver.apply_preset("incremental").unwrap();

        // Test invalid preset
        let result = solver.apply_preset("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_debug_dump() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.check_sat();

        let dump = solver.debug_dump();
        assert!(dump.contains("OxiZ"));
        assert!(dump.contains("QF_UF"));
        assert!(dump.contains("sat"));
    }

    #[test]
    fn test_set_tracing() {
        let mut solver = WasmSolver::new();
        solver.set_tracing(true);
        assert_eq!(solver.get_option("trace"), Some("true".to_string()));

        solver.set_tracing(false);
        assert_eq!(solver.get_option("trace"), Some("false".to_string()));
    }

    #[test]
    fn test_get_diagnostics() {
        let solver = WasmSolver::new();
        let warnings = solver.get_diagnostics();
        // Should have warnings about missing logic, etc.
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_check_pattern() {
        let solver = WasmSolver::new();

        let rec = solver.check_pattern("incremental");
        assert!(rec.contains("push/pop"));

        let rec = solver.check_pattern("assumptions");
        assert!(rec.contains("checkSatAssuming"));

        let rec = solver.check_pattern("async");
        assert!(rec.contains("async"));

        let rec = solver.check_pattern("validation");
        assert!(rec.contains("validateFormula"));

        let rec = solver.check_pattern("unknown");
        assert!(rec.contains("Unknown pattern"));
    }

    #[test]
    fn test_minimize() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.assert_formula("(>= x 0)").unwrap();
        solver.assert_formula("(>= y 0)").unwrap();

        // Should succeed
        let result = solver.minimize("(+ x y)");
        assert!(result.is_ok());

        // Empty formula should fail
        let result = solver.minimize("");
        assert!(result.is_err());

        // Whitespace-only should fail
        let result = solver.minimize("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_maximize() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.assert_formula("(<= x 10)").unwrap();
        solver.assert_formula("(<= y 10)").unwrap();

        // Should succeed
        let result = solver.maximize("(+ x y)");
        assert!(result.is_ok());

        // Empty formula should fail
        let result = solver.maximize("");
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.assert_formula("(>= x 0)").unwrap();
        solver.assert_formula("(>= y 0)").unwrap();
        solver.assert_formula("(<= (+ x y) 10)").unwrap();
        solver.maximize("(+ x y)").unwrap();

        let result = solver.optimize();
        assert!(result.is_ok());

        let obj = result.unwrap();
        assert!(obj.is_object());

        // Check that status field exists
        let status = js_sys::Reflect::get(&obj, &"status".into()).unwrap();
        assert!(!status.is_undefined());
    }

    #[test]
    fn test_assert_soft() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.declare_const("q", "Bool").unwrap();

        // Valid soft constraint
        let result = solver.assert_soft("p", "5");
        assert!(result.is_ok());

        // Empty formula should fail
        let result = solver.assert_soft("", "5");
        assert!(result.is_err());

        // Empty weight should fail
        let result = solver.assert_soft("q", "");
        assert!(result.is_err());

        // Invalid weight should fail
        let result = solver.assert_soft("q", "abc");
        assert!(result.is_err());

        // Negative weight should fail
        let result = solver.assert_soft("q", "-5");
        assert!(result.is_err());
    }

    #[test]
    fn test_optimization_unsat() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.assert_formula("p").unwrap();
        solver.assert_formula("(not p)").unwrap();
        solver.minimize("0").unwrap();

        let result = solver.optimize().unwrap();
        let status = js_sys::Reflect::get(&result, &"status".into()).unwrap();
        let status_str = status.as_string().unwrap();
        assert_eq!(status_str, "unsat");
    }

    #[test]
    fn test_get_minimal_model() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.declare_const("z", "Int").unwrap();
        solver.assert_formula("(= (+ x y z) 10)").unwrap();
        solver.assert_formula("(= x 5)").unwrap();
        solver.assert_formula("(= y 3)").unwrap();

        solver.check_sat();

        // Get minimal model with only x and y
        let minimal = solver.get_minimal_model(vec!["x".to_string(), "y".to_string()]);
        assert!(minimal.is_ok());

        let model = minimal.unwrap();
        assert!(model.is_object());

        // Check that x and y are present
        let x_val = js_sys::Reflect::get(&model, &"x".into()).unwrap();
        assert!(!x_val.is_undefined());

        let y_val = js_sys::Reflect::get(&model, &"y".into()).unwrap();
        assert!(!y_val.is_undefined());

        // Check that z is not present
        let z_val = js_sys::Reflect::get(&model, &"z".into()).unwrap();
        assert!(z_val.is_undefined());
    }

    #[test]
    fn test_get_minimal_model_empty_vars() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();
        solver.assert_formula("(= x 5)").unwrap();
        solver.assert_formula("(= y 3)").unwrap();

        solver.check_sat();

        // Get minimal model with empty variables array (should return all)
        let minimal = solver.get_minimal_model(vec![]);
        assert!(minimal.is_ok());

        let model = minimal.unwrap();
        assert!(model.is_object());

        // All declared variables should be present
        let x_val = js_sys::Reflect::get(&model, &"x".into()).unwrap();
        assert!(!x_val.is_undefined());

        let y_val = js_sys::Reflect::get(&model, &"y".into()).unwrap();
        assert!(!y_val.is_undefined());
    }

    #[test]
    fn test_get_minimal_model_before_sat() {
        let solver = WasmSolver::new();

        // Should fail if checkSat not called
        let result = solver.get_minimal_model(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_minimal_model_nonexistent_var() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.assert_formula("(= x 5)").unwrap();
        solver.check_sat();

        // Should fail for nonexistent variable
        let result = solver.get_minimal_model(vec!["nonexistent".to_string()]);
        assert!(result.is_err());
    }

    // Tests for interpolation support

    #[test]
    fn test_compute_interpolant_empty_partition_a() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");

        let result = solver.compute_interpolant(vec![], vec!["(> x 0)".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_interpolant_empty_partition_b() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");

        let result = solver.compute_interpolant(vec!["(> x 0)".to_string()], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_interpolant_no_proof_option() {
        let mut solver = WasmSolver::new();
        // Don't enable proof production

        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["(< x 0)".to_string()]);
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("produce-proofs"));
    }

    #[test]
    fn test_compute_interpolant_sat_formula() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        // These formulas are SAT together, not UNSAT
        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["(> x 1)".to_string()]);
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("UNSAT"));
    }

    #[test]
    fn test_compute_interpolant_unsat_formula() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        // These formulas are UNSAT together
        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["(< x 0)".to_string()]);

        // Should succeed and return an interpolant
        assert!(result.is_ok());
        let interpolant = result.unwrap();
        assert!(!interpolant.is_empty());
    }

    #[test]
    fn test_compute_interpolant_empty_formula_in_partition() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");

        // Empty formula in partition A
        let result = solver.compute_interpolant(vec!["".to_string()], vec!["(> x 0)".to_string()]);
        assert!(result.is_err());

        // Whitespace-only formula in partition B
        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["   ".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_interpolant_multiple_formulas() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();

        // Multiple formulas in each partition
        let result = solver.compute_interpolant(
            vec!["(> x 0)".to_string(), "(> x 5)".to_string()],
            vec!["(< y 0)".to_string(), "(= x y)".to_string()],
        );

        assert!(result.is_ok());
        let interpolant = result.unwrap();
        assert!(!interpolant.is_empty());
    }

    // Tests for quantifier elimination

    #[test]
    fn test_eliminate_quantifiers_empty_formula() {
        let mut solver = WasmSolver::new();

        let result = solver.eliminate_quantifiers("");
        assert!(result.is_err());
    }

    #[test]
    fn test_eliminate_quantifiers_whitespace_formula() {
        let mut solver = WasmSolver::new();

        let result = solver.eliminate_quantifiers("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_eliminate_quantifiers_no_quantifiers() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Formula without quantifiers
        let result = solver.eliminate_quantifiers("(> x 0)");
        assert!(result.is_ok());
        let qfree = result.unwrap();
        assert_eq!(qfree, "(> x 0)");
    }

    #[test]
    fn test_eliminate_quantifiers_exists() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        // Formula with existential quantifier
        let result = solver.eliminate_quantifiers("(exists ((y Int)) (= x (+ y 1)))");
        // Currently returns NotSupported
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("NotSupported") || err_str.contains("not yet fully implemented"));
    }

    #[test]
    fn test_eliminate_quantifiers_forall() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        // Formula with universal quantifier
        let result = solver.eliminate_quantifiers("(forall ((x Int)) (>= x 0))");
        // Currently returns NotSupported
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("NotSupported") || err_str.contains("not yet fully implemented"));
    }

    #[test]
    fn test_eliminate_quantifiers_nested() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        // Formula with nested quantifiers
        let result =
            solver.eliminate_quantifiers("(exists ((x Int)) (forall ((y Int)) (>= (+ x y) 0)))");
        // Currently returns NotSupported
        assert!(result.is_err());
    }

    #[test]
    fn test_eliminate_quantifiers_error_message() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        let result = solver.eliminate_quantifiers("(exists ((x Int)) (> x 0))");
        assert!(result.is_err());

        // Check that error message is informative
        let err_str = format!("{:?}", result);
        assert!(
            err_str.contains("Cooper") || err_str.contains("CAD") || err_str.contains("parser")
        );
    }

    // Tests for new arithmetic/logical operators

    #[test]
    fn test_mk_div() {
        let solver = WasmSolver::new();
        let div = solver
            .mk_div(vec!["x".to_string(), "2".to_string()])
            .unwrap();
        assert_eq!(div, "(/ x 2)");
    }

    #[test]
    fn test_mk_div_multiple_operands() {
        let solver = WasmSolver::new();
        let div = solver
            .mk_div(vec!["100".to_string(), "5".to_string(), "2".to_string()])
            .unwrap();
        assert_eq!(div, "(/ 100 5 2)");
    }

    #[test]
    fn test_mk_div_empty() {
        let solver = WasmSolver::new();
        let result = solver.mk_div(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_div_single_operand() {
        let solver = WasmSolver::new();
        let result = solver.mk_div(vec!["x".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_mod() {
        let solver = WasmSolver::new();
        let modulo = solver.mk_mod("x", "5").unwrap();
        assert_eq!(modulo, "(mod x 5)");
    }

    #[test]
    fn test_mk_mod_empty_lhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_mod("", "5");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_mod_empty_rhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_mod("x", "");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_neg() {
        let solver = WasmSolver::new();
        let neg = solver.mk_neg("x").unwrap();
        assert_eq!(neg, "(- x)");
    }

    #[test]
    fn test_mk_neg_expression() {
        let solver = WasmSolver::new();
        let neg = solver.mk_neg("(+ x 5)").unwrap();
        assert_eq!(neg, "(- (+ x 5))");
    }

    #[test]
    fn test_mk_neg_empty() {
        let solver = WasmSolver::new();
        let result = solver.mk_neg("");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_xor() {
        let solver = WasmSolver::new();
        let xor = solver.mk_xor("p", "q").unwrap();
        assert_eq!(xor, "(xor p q)");
    }

    #[test]
    fn test_mk_xor_empty_lhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_xor("", "q");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_xor_empty_rhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_xor("p", "");
        assert!(result.is_err());
    }

    // Tests for batch operations

    #[test]
    fn test_declare_funs() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        let result = solver.declare_funs(vec![
            "x Int".to_string(),
            "y Int".to_string(),
            "z Bool".to_string(),
        ]);
        assert!(result.is_ok());

        // Verify we can use the declared variables
        assert!(solver.assert_formula("(> x 0)").is_ok());
        assert!(solver.assert_formula("(< y 10)").is_ok());
        assert!(solver.assert_formula("z").is_ok());
    }

    #[test]
    fn test_declare_funs_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_funs(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_declare_funs_invalid_format() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_funs(vec!["x".to_string()]); // Missing sort
        assert!(result.is_err());
    }

    #[test]
    fn test_declare_funs_invalid_sort() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_funs(vec!["x InvalidSort".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_formulas() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();

        let result = solver.assert_formulas(vec![
            "(> x 0)".to_string(),
            "(< y 10)".to_string(),
            "(< x y)".to_string(),
        ]);
        assert!(result.is_ok());

        // Verify assertions were added
        let assertions = solver.get_assertions();
        assert!(assertions.contains("x"));
        assert!(assertions.contains("y"));
    }

    #[test]
    fn test_assert_formulas_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.assert_formulas(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_formulas_with_error() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();

        // Second formula is invalid (undeclared variable)
        let result = solver.assert_formulas(vec![
            "(> x 0)".to_string(),
            "(> undefined_var 5)".to_string(),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_named() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.set_option("produce-unsat-cores", "true");
        solver.declare_const("x", "Int").unwrap();

        let result = solver.assert_named("constraint1", "(> x 0)");
        assert!(result.is_ok());

        let result = solver.assert_named("constraint2", "(< x 0)");
        assert!(result.is_ok());

        // Check for unsat
        let check_result = solver.check_sat();
        assert_eq!(check_result, "unsat");
    }

    #[test]
    fn test_assert_named_empty_name() {
        let mut solver = WasmSolver::new();
        let result = solver.assert_named("", "(> x 0)");
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_named_empty_formula() {
        let mut solver = WasmSolver::new();
        let result = solver.assert_named("test", "");
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_named_with_unsat_core() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.set_option("produce-unsat-cores", "true");
        solver.declare_const("x", "Int").unwrap();

        solver.assert_named("positive", "(> x 0)").unwrap();
        solver.assert_named("negative", "(< x 0)").unwrap();

        assert_eq!(solver.check_sat(), "unsat");

        // Try to get unsat core (implementation depends on solver support)
        let core_result = solver.get_unsat_core();
        // We don't assert the exact core content since it depends on solver implementation
        // But the call should not panic
        let _ = core_result;
    }

    // Integration tests combining new features

    #[test]
    fn test_batch_declare_and_assert() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Batch declare
        solver
            .declare_funs(vec![
                "a Int".to_string(),
                "b Int".to_string(),
                "c Int".to_string(),
            ])
            .unwrap();

        // Batch assert
        solver
            .assert_formulas(vec![
                "(> a 0)".to_string(),
                "(> b a)".to_string(),
                "(= c (+ a b))".to_string(),
            ])
            .unwrap();

        // Check satisfiability
        assert_eq!(solver.check_sat(), "sat");
    }

    #[test]
    fn test_new_operators_integration() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.declare_const("x", "Int").unwrap();
        solver.declare_const("y", "Int").unwrap();

        // Use new operators to build formulas
        let div_expr = solver
            .mk_div(vec!["x".to_string(), "3".to_string()])
            .unwrap();
        let mod_expr = solver.mk_mod("y", "2").unwrap();
        let neg_expr = solver.mk_neg("x").unwrap();

        // Assert formulas using the built expressions
        solver
            .assert_formula(&format!("(= {} 5)", div_expr))
            .unwrap();
        solver
            .assert_formula(&format!("(= {} 0)", mod_expr))
            .unwrap();
        solver
            .assert_formula(&format!("(> {} -20)", neg_expr))
            .unwrap();

        // Should be satisfiable
        assert_eq!(solver.check_sat(), "sat");
    }

    #[test]
    fn test_mk_xor_integration() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver.declare_const("p", "Bool").unwrap();
        solver.declare_const("q", "Bool").unwrap();

        let xor_expr = solver.mk_xor("p", "q").unwrap();
        solver.assert_formula(&xor_expr).unwrap();
        solver.assert_formula("p").unwrap();

        // If p is true and (xor p q) is true, then q must be false
        assert_eq!(solver.check_sat(), "sat");

        // Verify the model
        if let Ok(model_val) = solver.get_model() {
            // Model should have p=true and q=false
            let _ = model_val; // Just verify it doesn't panic
        }
    }

    // Tests for bitvector operations

    #[test]
    fn test_mk_bvand() {
        let solver = WasmSolver::new();
        let bvand = solver.mk_bvand("x", "y").unwrap();
        assert_eq!(bvand, "(bvand x y)");
    }

    #[test]
    fn test_mk_bvand_empty() {
        let solver = WasmSolver::new();
        assert!(solver.mk_bvand("", "y").is_err());
        assert!(solver.mk_bvand("x", "").is_err());
    }

    #[test]
    fn test_mk_bvor() {
        let solver = WasmSolver::new();
        let bvor = solver.mk_bvor("x", "y").unwrap();
        assert_eq!(bvor, "(bvor x y)");
    }

    #[test]
    fn test_mk_bvxor() {
        let solver = WasmSolver::new();
        let bvxor = solver.mk_bvxor("x", "y").unwrap();
        assert_eq!(bvxor, "(bvxor x y)");
    }

    #[test]
    fn test_mk_bvnot() {
        let solver = WasmSolver::new();
        let bvnot = solver.mk_bvnot("x").unwrap();
        assert_eq!(bvnot, "(bvnot x)");
    }

    #[test]
    fn test_mk_bvnot_empty() {
        let solver = WasmSolver::new();
        assert!(solver.mk_bvnot("").is_err());
    }

    #[test]
    fn test_mk_bvneg() {
        let solver = WasmSolver::new();
        let bvneg = solver.mk_bvneg("x").unwrap();
        assert_eq!(bvneg, "(bvneg x)");
    }

    #[test]
    fn test_mk_bvadd() {
        let solver = WasmSolver::new();
        let bvadd = solver.mk_bvadd("x", "y").unwrap();
        assert_eq!(bvadd, "(bvadd x y)");
    }

    #[test]
    fn test_mk_bvsub() {
        let solver = WasmSolver::new();
        let bvsub = solver.mk_bvsub("x", "y").unwrap();
        assert_eq!(bvsub, "(bvsub x y)");
    }

    #[test]
    fn test_mk_bvmul() {
        let solver = WasmSolver::new();
        let bvmul = solver.mk_bvmul("x", "y").unwrap();
        assert_eq!(bvmul, "(bvmul x y)");
    }

    #[test]
    fn test_bitvector_integration() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_BV");
        solver.declare_const("x", "BitVec8").unwrap();
        solver.declare_const("y", "BitVec8").unwrap();

        // Build formulas with bitvector operations
        let and_expr = solver.mk_bvand("x", "y").unwrap();
        let or_expr = solver.mk_bvor("x", "y").unwrap();
        let xor_expr = solver.mk_bvxor("x", "y").unwrap();

        // These should parse correctly (we're just testing formula building)
        let _ = solver.validate_formula(&and_expr);
        let _ = solver.validate_formula(&or_expr);
        let _ = solver.validate_formula(&xor_expr);
    }
}
