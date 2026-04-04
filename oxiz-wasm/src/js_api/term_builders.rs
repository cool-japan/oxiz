//! SMT-LIB2 term builder helpers: `mk_*` functions for constructing expressions.

use crate::WasmSolver;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
    /// Create a function application expression
    ///
    /// Constructs a function application expression in SMT-LIB2 format.
    /// It's equivalent to manually writing `(f arg1 arg2 ...)`,
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

    /// Create an equality expression `(= lhs rhs)`
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const solver = new WasmSolver();
    /// const eq = solver.mkEq("x", "5");
    /// console.log(eq); // "(= x 5)"
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

    /// Create a conjunction (AND) expression `(and expr1 expr2 ...)`
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

    /// Create a disjunction (OR) expression `(or expr1 expr2 ...)`
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

    /// Create a negation (NOT) expression `(not expr)`
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

    /// Create an implication expression `(=> lhs rhs)`
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

    /// Create an if-then-else expression `(ite cond then_expr else_expr)`
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

    /// Create a less-than comparison `(< lhs rhs)`
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

    /// Create a less-than-or-equal comparison `(<= lhs rhs)`
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

    /// Create a greater-than comparison `(> lhs rhs)`
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

    /// Create a greater-than-or-equal comparison `(>= lhs rhs)`
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

    /// Create an addition expression `(+ arg1 arg2 ...)`
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

    /// Create a subtraction expression `(- arg1 arg2 ...)`
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

    /// Create a multiplication expression `(* arg1 arg2 ...)`
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

    /// Create a division expression `(/ arg1 arg2 ...)`
    ///
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

    /// Create a modulo expression `(mod arg1 arg2)`
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

    /// Create a negation expression `(- arg)`
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

    /// Create an exclusive-or (XOR) expression `(xor arg1 arg2)`
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

    /// Create a distinct (all different) expression `(distinct arg1 arg2 ...)`
    ///
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

    /// Create a bitvector AND expression `(bvand arg1 arg2)`
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

    /// Create a bitvector OR expression `(bvor arg1 arg2)`
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

    /// Create a bitvector XOR expression `(bvxor arg1 arg2)`
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

    /// Create a bitvector NOT expression `(bvnot arg)`
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

    /// Create a bitvector negation expression `(bvneg arg)`
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

    /// Create a bitvector addition expression `(bvadd arg1 arg2)`
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

    /// Create a bitvector subtraction expression `(bvsub arg1 arg2)`
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

    /// Create a bitvector multiplication expression `(bvmul arg1 arg2)`
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
}
