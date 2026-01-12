/**
 * OxiZ WASM SMT Solver for Deno
 *
 * This module provides Deno support for the OxiZ WASM SMT Solver.
 *
 * @example
 * ```typescript
 * import { WasmSolver, init } from "https://deno.land/x/oxiz/mod.ts";
 *
 * // Initialize the WASM module
 * await init();
 *
 * // Create a solver instance
 * const solver = new WasmSolver();
 * solver.setLogic("QF_LIA");
 * solver.declareConst("x", "Int");
 * solver.assertFormula("(> x 0)");
 *
 * const result = solver.checkSat();
 * console.log("Result:", result); // "sat"
 *
 * if (result === "sat") {
 *   const model = solver.getModel();
 *   console.log("Model:", model);
 * }
 * ```
 */

// Re-export all types and functions from the WASM module
// Note: The actual WASM file path will need to be configured based on deployment
export type {
  WasmErrorKind,
} from "../pkg/oxiz_wasm.d.ts";

export {
  WasmSolver,
  init,
  version,
} from "../pkg/oxiz_wasm.js";

/**
 * Initialize the OxiZ WASM module for Deno
 *
 * @param wasmPath - Optional path to the WASM file
 * @returns Promise that resolves when initialization is complete
 */
export async function initOxiZ(wasmPath?: string): Promise<void> {
  const { default: init } = await import("../pkg/oxiz_wasm.js");

  if (wasmPath) {
    const wasmBytes = await Deno.readFile(wasmPath);
    await init(wasmBytes);
  } else {
    // Try to load from default location
    try {
      const wasmBytes = await Deno.readFile("./pkg/oxiz_wasm_bg.wasm");
      await init(wasmBytes);
    } catch {
      // If file reading fails, let the module try to fetch it
      await init();
    }
  }
}
