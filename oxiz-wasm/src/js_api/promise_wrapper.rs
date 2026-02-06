//! Promise-based Async API for OxiZ
//!
//! This module provides a Promise-based async API for OxiZ solver operations,
//! allowing JavaScript/TypeScript code to use async/await syntax naturally.
//!
//! # Design
//!
//! - All long-running operations return JavaScript Promises
//! - Solver state is managed safely across async boundaries
//! - Cancellation support via AbortSignal
//! - Progress reporting via callbacks
//!
//! # Example
//!
//! ```javascript
//! const solver = new AsyncSolver();
//! solver.setLogic("QF_LIA");
//! solver.declareConst("x", "Int");
//! solver.assertFormula("(> x 0)");
//!
//! // Returns a Promise
//! const result = await solver.checkSatAsync();
//! if (result === "sat") {
//!     const model = await solver.getModelAsync();
//!     console.log(model);
//! }
//! ```

#![forbid(unsafe_code)]

use crate::{WasmError, WasmErrorKind};
use js_sys::Promise;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

/// Async solver wrapper with Promise-based API
#[wasm_bindgen]
pub struct AsyncSolver {
    /// Inner solver context
    ctx: Rc<RefCell<oxiz_solver::Context>>,
    /// Cancellation flag
    cancelled: Rc<RefCell<bool>>,
    /// Progress callback
    progress_callback: Option<js_sys::Function>,
}

#[wasm_bindgen]
impl AsyncSolver {
    /// Create a new async solver
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            ctx: Rc::new(RefCell::new(oxiz_solver::Context::new())),
            cancelled: Rc::new(RefCell::new(false)),
            progress_callback: None,
        }
    }

    /// Set the logic for the solver
    #[wasm_bindgen(js_name = setLogic)]
    pub fn set_logic(&self, logic: &str) {
        self.ctx.borrow_mut().set_logic(logic);
    }

    /// Declare a constant
    #[wasm_bindgen(js_name = declareConst)]
    pub fn declare_const(&self, name: &str, sort_name: &str) -> Result<(), JsValue> {
        let sort = self.parse_sort(sort_name)?;
        self.ctx.borrow_mut().declare_const(name, sort);
        Ok(())
    }

    /// Assert a formula
    #[wasm_bindgen(js_name = assertFormula)]
    pub fn assert_formula(&self, formula: &str) -> Result<(), JsValue> {
        let script = format!("(assert {})", formula);
        self.ctx.borrow_mut().execute_script(&script).map_err(|e| {
            WasmError::new(
                WasmErrorKind::ParseError,
                format!("Failed to assert formula: {}", e),
            )
        })?;
        Ok(())
    }

    /// Check satisfiability asynchronously
    #[wasm_bindgen(js_name = checkSatAsync)]
    pub fn check_sat_async(&self) -> Promise {
        let ctx = self.ctx.clone();
        let cancelled = self.cancelled.clone();
        let progress_cb = self.progress_callback.clone();

        future_to_promise(async move {
            // Check for cancellation
            if *cancelled.borrow() {
                return Err(
                    WasmError::new(WasmErrorKind::InvalidState, "Operation cancelled").into(),
                );
            }

            // Report progress if callback is set
            if let Some(ref cb) = progress_cb {
                let progress = js_sys::Object::new();
                let _ = js_sys::Reflect::set(&progress, &"status".into(), &"solving".into());
                let _ = js_sys::Reflect::set(&progress, &"progress".into(), &0.5.into());
                let _ = cb.call1(&JsValue::NULL, &progress);
            }

            // Perform the check-sat operation
            let result = ctx.borrow_mut().check_sat();

            let result_str = match result {
                oxiz_solver::SolverResult::Sat => "sat",
                oxiz_solver::SolverResult::Unsat => "unsat",
                oxiz_solver::SolverResult::Unknown => "unknown",
            };

            // Report completion
            if let Some(ref cb) = progress_cb {
                let progress = js_sys::Object::new();
                let _ = js_sys::Reflect::set(&progress, &"status".into(), &"done".into());
                let _ = js_sys::Reflect::set(&progress, &"progress".into(), &1.0.into());
                let _ = js_sys::Reflect::set(&progress, &"result".into(), &result_str.into());
                let _ = cb.call1(&JsValue::NULL, &progress);
            }

            Ok(JsValue::from_str(result_str))
        })
    }

    /// Get model asynchronously
    #[wasm_bindgen(js_name = getModelAsync)]
    pub fn get_model_async(&self) -> Promise {
        let ctx = self.ctx.clone();

        future_to_promise(async move {
            let model = ctx.borrow().get_model();
            let model_str = match model {
                Some(entries) => entries
                    .iter()
                    .map(|(name, sort, value)| format!("({} {} {})", name, sort, value))
                    .collect::<Vec<_>>()
                    .join("\n"),
                None => String::new(),
            };
            Ok(JsValue::from_str(&model_str))
        })
    }

    /// Execute script asynchronously
    #[wasm_bindgen(js_name = executeAsync)]
    pub fn execute_async(&self, script: String) -> Promise {
        let ctx = self.ctx.clone();
        let cancelled = self.cancelled.clone();

        future_to_promise(async move {
            if *cancelled.borrow() {
                return Err(
                    WasmError::new(WasmErrorKind::InvalidState, "Operation cancelled").into(),
                );
            }

            let output = ctx.borrow_mut().execute_script(&script).map_err(|e| {
                WasmError::new(
                    WasmErrorKind::ParseError,
                    format!("Failed to execute script: {}", e),
                )
            })?;

            let result = output.join("\n");
            Ok(JsValue::from_str(&result))
        })
    }

    /// Set progress callback
    #[wasm_bindgen(js_name = onProgress)]
    pub fn on_progress(&mut self, callback: js_sys::Function) {
        self.progress_callback = Some(callback);
    }

    /// Cancel ongoing operation
    #[wasm_bindgen(js_name = cancel)]
    pub fn cancel(&self) {
        *self.cancelled.borrow_mut() = true;
    }

    /// Reset cancellation flag
    #[wasm_bindgen(js_name = resetCancellation)]
    pub fn reset_cancellation(&self) {
        *self.cancelled.borrow_mut() = false;
    }

    /// Push a new solving context
    #[wasm_bindgen(js_name = pushAsync)]
    pub fn push_async(&self) -> Promise {
        let ctx = self.ctx.clone();

        future_to_promise(async move {
            ctx.borrow_mut().push();
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Pop a solving context
    #[wasm_bindgen(js_name = popAsync)]
    pub fn pop_async(&self) -> Promise {
        let ctx = self.ctx.clone();

        future_to_promise(async move {
            ctx.borrow_mut().pop();
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Reset the solver
    #[wasm_bindgen(js_name = resetAsync)]
    pub fn reset_async(&self) -> Promise {
        let ctx = self.ctx.clone();

        future_to_promise(async move {
            *ctx.borrow_mut() = oxiz_solver::Context::new();
            Ok(JsValue::UNDEFINED)
        })
    }

    // Helper to parse sorts - returns SortId from context's SortManager
    fn parse_sort(&self, sort_name: &str) -> Result<oxiz_core::SortId, JsValue> {
        match sort_name {
            "Bool" => Ok(self.ctx.borrow().terms.sorts.bool_sort),
            "Int" => Ok(self.ctx.borrow().terms.sorts.int_sort),
            "Real" => Ok(self.ctx.borrow().terms.sorts.real_sort),
            _ => {
                if sort_name.starts_with("BitVec") {
                    Err(WasmError::new(
                        WasmErrorKind::NotSupported,
                        "BitVec sorts not yet supported",
                    )
                    .into())
                } else {
                    Err(WasmError::new(
                        WasmErrorKind::InvalidSort,
                        format!("Unknown sort: {}", sort_name),
                    )
                    .into())
                }
            }
        }
    }
}

impl Default for AsyncSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Promise utilities for batch operations
#[wasm_bindgen]
pub struct PromiseBatch {
    /// List of promises to batch
    promises: Vec<Promise>,
}

#[wasm_bindgen]
impl PromiseBatch {
    /// Create a new promise batch
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            promises: Vec::new(),
        }
    }

    /// Add a promise to the batch
    #[wasm_bindgen(js_name = add)]
    pub fn add(&mut self, promise: Promise) {
        self.promises.push(promise);
    }

    /// Wait for all promises to complete
    #[wasm_bindgen(js_name = all)]
    pub fn all(&self) -> Promise {
        let array = js_sys::Array::new();
        for promise in &self.promises {
            array.push(promise);
        }
        js_sys::Promise::all(&array)
    }

    /// Race all promises (return first to complete)
    #[wasm_bindgen(js_name = race)]
    pub fn race(&self) -> Promise {
        let array = js_sys::Array::new();
        for promise in &self.promises {
            array.push(promise);
        }
        js_sys::Promise::race(&array)
    }

    /// Get the number of promises in the batch
    #[wasm_bindgen(js_name = length)]
    pub fn length(&self) -> usize {
        self.promises.len()
    }
}

impl Default for PromiseBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Async operation with timeout support
#[wasm_bindgen]
pub struct AsyncOperation {
    /// The promise for the operation
    promise: Promise,
    /// Timeout in milliseconds
    timeout_ms: Option<u32>,
}

#[wasm_bindgen]
impl AsyncOperation {
    /// Create a new async operation
    #[wasm_bindgen(constructor)]
    pub fn new(promise: Promise) -> Self {
        Self {
            promise,
            timeout_ms: None,
        }
    }

    /// Set a timeout
    #[wasm_bindgen(js_name = withTimeout)]
    pub fn with_timeout(mut self, timeout_ms: u32) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Execute the operation
    #[wasm_bindgen(js_name = execute)]
    pub fn execute(&self) -> Promise {
        let promise = self.promise.clone();

        if let Some(timeout_ms) = self.timeout_ms {
            // Create a timeout promise
            let timeout_promise = js_sys::Promise::new(&mut |resolve, _reject| {
                let window = web_sys::window().expect("no global window");
                let callback = Closure::once(move || {
                    let error = WasmError::new(
                        WasmErrorKind::InvalidState,
                        format!("Operation timed out after {}ms", timeout_ms),
                    );
                    resolve.call1(&JsValue::NULL, &error.into()).ok();
                });

                window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        callback.as_ref().unchecked_ref(),
                        timeout_ms as i32,
                    )
                    .ok();
                callback.forget();
            });

            // Race the operation against the timeout
            let array = js_sys::Array::new();
            array.push(&promise);
            array.push(&timeout_promise);
            js_sys::Promise::race(&array)
        } else {
            promise
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_solver_creation() {
        let solver = AsyncSolver::new();
        assert!(!*solver.cancelled.borrow());
    }

    #[test]
    fn test_promise_batch() {
        let batch = PromiseBatch::new();
        assert_eq!(batch.length(), 0);
    }

    #[test]
    fn test_cancellation() {
        let solver = AsyncSolver::new();
        assert!(!*solver.cancelled.borrow());

        solver.cancel();
        assert!(*solver.cancelled.borrow());

        solver.reset_cancellation();
        assert!(!*solver.cancelled.borrow());
    }
}
