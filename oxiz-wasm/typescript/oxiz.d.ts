// OxiZ WASM — TypeScript type definitions
// Mirrors oxiz-wasm::js_api::typescript::generate_typescript_dts()
// See also: oxiz-wasm/src/js_api/typescript.rs

/** The result returned by {@link OxizSolver.check} and {@link solve}. */
export interface SolverResult {
  /** Satisfiability status. */
  status: "sat" | "unsat" | "unknown";

  /**
   * Satisfying model, present only when `status === "sat"` and model
   * production is enabled.  Maps variable/constant names to their values.
   */
  model?: Record<string, string | number | boolean>;

  /**
   * Unsatisfiable core — a minimal set of named assertions that are jointly
   * unsatisfiable.  Present only when `status === "unsat"` and unsat-core
   * production is enabled.
   */
  unsatCore?: string[];

  /** Wall-clock time in milliseconds spent inside the solver. */
  time_ms?: number;
}

/**
 * A single variable assignment within a satisfying model.
 *
 * @example
 * ```typescript
 * const { status, model } = solver.check();
 * if (status === "sat" && model) {
 *   for (const [name, entry] of Object.entries(model)) {
 *     console.log(`${name} = ${entry}`);
 *   }
 * }
 * ```
 */
export type ModelValue = string | number | boolean;

/** Configuration options accepted by the {@link OxizSolver} constructor. */
export interface SolverConfig {
  /**
   * SMT-LIB2 logic to use (e.g. `"QF_LIA"`, `"QF_BV"`, `"ALL"`).
   * Defaults to `"ALL"` (all theories enabled).
   */
  logic?: string;

  /** Enable model generation. Defaults to `true`. */
  produceModels?: boolean;

  /** Enable unsat-core production. Defaults to `false`. */
  produceUnsatCores?: boolean;

  /** Solver timeout in milliseconds. `0` means no timeout. Defaults to `0`. */
  timeoutMs?: number;

  /** Verbosity level (0 = silent, 10 = maximum). Defaults to `0`. */
  verbosity?: number;
}

/**
 * Pure-Rust SMT solver exposed to JavaScript/TypeScript via WebAssembly.
 *
 * @example
 * ```typescript
 * import init, { OxizSolver } from "@cool-japan/oxiz";
 *
 * await init();
 * const solver = new OxizSolver({ logic: "QF_LIA", produceModels: true });
 * solver.assert("(declare-const x Int)");
 * solver.assert("(assert (> x 0))");
 * const { status, model } = solver.check();
 * console.log(status);        // "sat"
 * console.log(model?.["x"]);  // e.g. 1
 * ```
 */
export class OxizSolver {
  /**
   * Construct a new solver instance.
   *
   * @param config - Optional configuration overrides.  Any omitted fields
   *                 use their documented defaults.
   */
  constructor(config?: Partial<SolverConfig>);

  /**
   * Assert a single SMT-LIB2 expression or command.
   *
   * Accepts both bare expressions such as `"(> x 0)"` and full SMT-LIB2
   * commands such as `"(assert (> x 0))"`.
   *
   * @param smtlib2 - An SMT-LIB2 expression or command string.
   * @throws If the expression cannot be parsed or type-checked.
   */
  assert(smtlib2: string): void;

  /**
   * Run `check-sat` on the current assertion stack.
   *
   * @returns A {@link SolverResult} with `status`, and optionally `model`
   *          or `unsatCore` depending on configuration.
   */
  check(): SolverResult;

  /**
   * Push a new backtracking scope onto the assertion stack.
   *
   * All assertions and declarations added after `push()` can be undone
   * atomically by calling {@link pop}.
   */
  push(): void;

  /**
   * Pop the innermost backtracking scope, undoing all assertions and
   * declarations made since the matching {@link push}.
   */
  pop(): void;

  /**
   * Reset the solver completely, discarding all assertions, declarations,
   * and solver state.  The solver returns to the same state as after
   * construction (with the original config).
   */
  reset(): void;

  /**
   * Set (or replace) the solver timeout.
   *
   * @param ms - Timeout in milliseconds.  Pass `0` to disable the timeout.
   */
  setTimeout(ms: number): void;

  /**
   * Execute a multi-command SMT-LIB2 script and return the combined output.
   *
   * @param script - A complete SMT-LIB2 script (may contain multiple commands).
   * @returns The solver output for all `get-*` and `check-sat` commands in
   *          the script, joined with newlines.
   * @throws If the script contains syntax or semantic errors.
   */
  execute(script: string): string;

  /**
   * Asynchronous variant of {@link execute} that yields to the event loop
   * periodically, keeping the browser responsive for long-running scripts.
   *
   * @param script - A complete SMT-LIB2 script.
   * @returns A `Promise` that resolves to the combined solver output.
   */
  executeAsync(script: string): Promise<string>;

  /**
   * Retrieve the current satisfying model as a JSON-serialisable object.
   *
   * @returns A mapping from variable names to their model values.
   * @throws If no model is available (check-sat not yet called, or result
   *         was not `"sat"`).
   */
  getModel(): Record<string, ModelValue>;

  /**
   * Retrieve the current satisfying model as an SMT-LIB2 `model` response
   * string.
   *
   * @returns The raw SMT-LIB2 `(model ...)` string.
   * @throws If no model is available.
   */
  getModelString(): string;

  /**
   * Retrieve the unsatisfiable core as an array of named assertion labels.
   *
   * @returns An array of assertion labels that form a minimal unsatisfiable
   *          subset.
   * @throws If no unsat core is available.
   */
  getUnsatCore(): string[];

  /**
   * Set the SMT-LIB2 logic.
   *
   * @param logic - Logic string such as `"QF_BV"` or `"ALL"`.
   */
  setLogic(logic: string): void;

  /**
   * Set a raw SMT-LIB2 solver option (`:set-option`).
   *
   * @param key   - Option keyword (without leading `:`), e.g. `"produce-models"`.
   * @param value - Option value, e.g. `"true"`.
   */
  setOption(key: string, value: string): void;

  /**
   * Get the current value of a solver option.
   *
   * @param key - Option keyword.
   * @returns The option value, or `undefined` if not set.
   */
  getOption(key: string): string | undefined;

  /**
   * Apply a named configuration preset.
   *
   * Available presets: `"default"`, `"fast"`, `"complete"`, `"debug"`,
   * `"unsat-core"`, `"incremental"`.
   *
   * @param preset - Preset name.
   * @throws If the preset name is unknown.
   */
  applyPreset(preset: string): void;

  /**
   * Cancel a running solver operation.
   *
   * Sets an internal cancellation flag.  The solver will check this flag
   * periodically; cancellation is advisory and may not take effect
   * immediately.
   */
  cancel(): void;

  /** Returns `true` if cancellation has been requested. */
  isCancelled(): boolean;

  /**
   * Check satisfiability under temporary assumptions.
   *
   * The assumptions are only active for this single call and do not modify
   * the persistent assertion stack.
   *
   * @param assumptions - Array of SMT-LIB2 boolean expressions.
   * @returns `"sat"`, `"unsat"`, or `"unknown"`.
   * @throws If `assumptions` is empty or contains invalid expressions.
   */
  checkSatAssuming(assumptions: string[]): string;

  /**
   * Declare a constant of the given sort.
   *
   * Equivalent to `(declare-const name sort)` in SMT-LIB2.
   *
   * @param name - Constant name.
   * @param sort - Sort name (e.g. `"Int"`, `"Bool"`, `"BitVec32"`).
   * @throws If the sort name is unrecognised.
   */
  declareConst(name: string, sort: string): void;

  /**
   * Declare a function symbol.
   *
   * Equivalent to `(declare-fun name (argSorts...) returnSort)`.
   *
   * @param name       - Function name.
   * @param argSorts   - Array of argument sort names.
   * @param returnSort - Return sort name.
   * @throws If any sort name is unrecognised.
   */
  declareFun(name: string, argSorts: string[], returnSort: string): void;

  /**
   * Return solver statistics as a plain JS object.
   *
   * The exact shape depends on the solver configuration but typically
   * includes fields such as `conflicts`, `propagations`, `decisions`, etc.
   *
   * @throws If called before any solving has occurred.
   */
  getStatistics(): Record<string, number | string>;

  /**
   * Return solver meta-information.
   *
   * @param key - Info key, e.g. `"name"`, `"version"`, `"authors"`.
   * @returns The requested info string.
   * @throws If the key is unknown.
   */
  getInfo(key: string): string;

  /**
   * Simplify an SMT-LIB2 expression and return the normal form.
   *
   * @param expr - An SMT-LIB2 expression string.
   * @returns The simplified expression string.
   * @throws If the expression cannot be parsed.
   */
  simplify(expr: string): string;

  /**
   * Enable or disable internal solver tracing.
   *
   * @param enabled - `true` to enable verbose tracing, `false` to disable.
   */
  setTracing(enabled: boolean): void;

  /**
   * Return a human-readable debug dump of the current solver state.
   *
   * Includes the current logic, last result, assertion count, and other
   * diagnostic information.
   *
   * @returns A multi-line debug string.
   */
  debugDump(): string;
}

/**
 * Convenience one-shot solver function.
 *
 * Parses and executes `smtlib2Script` from scratch (no persistent state).
 * Suitable for quick scripting scenarios where you have a complete SMT-LIB2
 * script and want a single result without managing a solver instance.
 *
 * @param smtlib2Script - A complete SMT-LIB2 script including `check-sat`.
 * @returns A {@link SolverResult} derived from the last `check-sat` in the
 *          script.
 *
 * @example
 * ```typescript
 * import init, { solve } from "@cool-japan/oxiz";
 * await init();
 *
 * const script = `
 *   (set-logic QF_LIA)
 *   (declare-const x Int)
 *   (assert (> x 5))
 *   (check-sat)
 *   (get-model)
 * `;
 * const result = solve(script);
 * console.log(result.status);  // "sat"
 * ```
 */
export function solve(smtlib2Script: string): SolverResult;

/**
 * Return the OxiZ WASM library version string.
 *
 * @returns A semver string, e.g. `"0.2.0"`.
 *
 * @example
 * ```typescript
 * import init, { version } from "@cool-japan/oxiz";
 * await init();
 * console.log(`OxiZ WASM version: ${version()}`);
 * ```
 */
export function version(): string;
