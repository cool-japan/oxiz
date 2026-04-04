//! Sort and function declaration operations.

use crate::WasmSolver;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
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
}
