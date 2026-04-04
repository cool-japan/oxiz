//! Solver configuration, options, and presets.

use crate::WasmSolver;
use crate::{WasmError, WasmErrorKind};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmSolver {
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
}
