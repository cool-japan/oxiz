/**
 * OxiZ WASM - WebAssembly bindings for OxiZ SMT Solver
 *
 * This module provides TypeScript type definitions for the OxiZ WASM bindings.
 *
 * @packageDocumentation
 */

/**
 * Error kind enumeration for WASM errors
 */
export type WasmErrorKind =
    | "ParseError"
    | "InvalidSort"
    | "NoModel"
    | "NoUnsatCore"
    | "InvalidState"
    | "InvalidInput"
    | "NotSupported"
    | "Unknown";

/**
 * Structured error object returned by WASM operations
 */
export interface WasmError {
    /** The kind/type of error that occurred */
    kind: WasmErrorKind;
    /** Detailed error message */
    message: string;
}

/**
 * Satisfiability result from checkSat operations
 */
export type SatResult = "sat" | "unsat" | "unknown";

/**
 * Model entry containing sort and value information
 */
export interface ModelEntry {
    /** The sort/type of the variable */
    sort: string;
    /** The value assigned to the variable */
    value: string;
}

/**
 * Model object mapping variable names to their assignments
 */
export interface Model {
    [variableName: string]: ModelEntry;
}

/**
 * Valid SMT-LIB2 sort names
 */
export type SortName = "Bool" | "Int" | "Real" | `BitVec${number}`;

/**
 * Common SMT-LIB2 logic names
 */
export type LogicName =
    | "QF_UF"   // Quantifier-free uninterpreted functions
    | "QF_LIA"  // Quantifier-free linear integer arithmetic
    | "QF_LRA"  // Quantifier-free linear real arithmetic
    | "QF_NIA"  // Quantifier-free nonlinear integer arithmetic
    | "QF_NRA"  // Quantifier-free nonlinear real arithmetic
    | "QF_BV"   // Quantifier-free bitvectors
    | "QF_AUFLIA" // Quantifier-free arrays, uninterpreted functions, linear integer arithmetic
    | "ALL"     // All supported theories
    | string;   // Allow custom logic names

/**
 * Main WASM solver class providing SMT solving capabilities
 *
 * @example
 * ```typescript
 * import init, { WasmSolver } from 'oxiz-wasm';
 *
 * await init();
 * const solver = new WasmSolver();
 * solver.setLogic('QF_LIA');
 * solver.declareConst('x', 'Int');
 * solver.assertFormula('(> x 0)');
 * const result = solver.checkSat(); // "sat"
 * const model = solver.getModel();
 * console.log(model.x.value);
 * ```
 */
export class WasmSolver {
    /**
     * Create a new solver instance
     */
    constructor();

    /**
     * Execute a complete SMT-LIB2 script
     *
     * @param script - SMT-LIB2 script string
     * @returns The output of the script execution
     * @throws {WasmError} If the script contains errors
     *
     * @example
     * ```typescript
     * const solver = new WasmSolver();
     * const script = `
     *   (set-logic QF_LIA)
     *   (declare-const x Int)
     *   (assert (> x 0))
     *   (check-sat)
     * `;
     * const result = solver.execute(script);
     * ```
     */
    execute(script: string): string;

    /**
     * Execute a complete SMT-LIB2 script asynchronously
     *
     * @param script - SMT-LIB2 script string
     * @returns Promise resolving to the output
     * @throws {WasmError} If the script contains errors
     *
     * @example
     * ```typescript
     * const result = await solver.executeAsync(script);
     * ```
     */
    executeAsync(script: string): Promise<string>;

    /**
     * Execute an SMT-LIB2 script asynchronously with progress callbacks
     *
     * This method is similar to executeAsync() but also accepts a callback function
     * that will be invoked periodically with progress updates. This is useful for
     * long-running operations where you want to show progress to the user.
     *
     * @param script - SMT-LIB2 script to execute
     * @param progressCallback - Optional callback that receives (current, total) line counts
     * @returns Promise that resolves to the execution output
     *
     * @example
     * ```typescript
     * const result = await solver.executeWithProgress(
     *   script,
     *   (current, total) => {
     *     console.log(`Progress: ${current}/${total} lines`);
     *     const percent = Math.round((current / total) * 100);
     *     document.getElementById('progress').innerText = `${percent}%`;
     *   }
     * );
     * ```
     */
    executeWithProgress(
        script: string,
        progressCallback?: (current: number, total: number) => void
    ): Promise<string>;

    /**
     * Set the SMT logic for the solver
     *
     * @param logic - SMT-LIB2 logic name (e.g., "QF_LIA", "QF_UF")
     *
     * @example
     * ```typescript
     * solver.setLogic('QF_LIA');
     * ```
     */
    setLogic(logic: LogicName): void;

    /**
     * Set a solver option
     *
     * @param key - Option name
     * @param value - Option value
     *
     * @example
     * ```typescript
     * solver.setOption('produce-models', 'true');
     * solver.setOption('produce-unsat-cores', 'true');
     * ```
     */
    setOption(key: string, value: string): void;

    /**
     * Get a solver option value
     *
     * @param key - Option name
     * @returns The option value, or undefined if not set
     *
     * @example
     * ```typescript
     * const value = solver.getOption('produce-models');
     * ```
     */
    getOption(key: string): string | undefined;

    /**
     * Declare a constant with given name and sort
     *
     * @param name - Constant name
     * @param sort - Sort name (Bool, Int, Real, BitVecN)
     * @throws {WasmError} If the sort is invalid or name is empty
     *
     * @example
     * ```typescript
     * solver.declareConst('x', 'Int');
     * solver.declareConst('flag', 'Bool');
     * solver.declareConst('bv', 'BitVec32');
     * ```
     */
    declareConst(name: string, sort: SortName): void;

    /**
     * Declare a function with given signature
     *
     * Note: Currently only nullary functions (constants) are fully supported
     *
     * @param name - Function name
     * @param argSorts - Array of argument sort names (empty for constants)
     * @param retSort - Return sort name
     * @throws {WasmError} If sorts are invalid or non-nullary functions are used
     *
     * @example
     * ```typescript
     * solver.declareFun('c', [], 'Int'); // Declare constant
     * ```
     */
    declareFun(name: string, argSorts: SortName[], retSort: SortName): void;

    /**
     * Assert a formula to the solver
     *
     * @param formula - SMT-LIB2 boolean expression
     * @throws {WasmError} If the formula is invalid or malformed
     *
     * @example
     * ```typescript
     * solver.assertFormula('(> x 0)');
     * solver.assertFormula('(and (> x 5) (< y 10))');
     * ```
     */
    assertFormula(formula: string): void;

    /**
     * Get all current assertions as SMT-LIB2 string
     *
     * @returns SMT-LIB2 formatted string of all assertions
     *
     * @example
     * ```typescript
     * const assertions = solver.getAssertions();
     * console.log(assertions);
     * ```
     */
    getAssertions(): string;

    /**
     * Reset all assertions while keeping declarations
     *
     * @example
     * ```typescript
     * solver.resetAssertions();
     * ```
     */
    resetAssertions(): void;

    /**
     * Check satisfiability of current assertions
     *
     * @returns "sat", "unsat", or "unknown"
     *
     * @example
     * ```typescript
     * const result = solver.checkSat();
     * if (result === 'sat') {
     *   const model = solver.getModel();
     * }
     * ```
     */
    checkSat(): SatResult;

    /**
     * Check satisfiability asynchronously
     *
     * @returns Promise resolving to "sat", "unsat", or "unknown"
     *
     * @example
     * ```typescript
     * const result = await solver.checkSatAsync();
     * ```
     */
    checkSatAsync(): Promise<SatResult>;

    /**
     * Get the model as a JavaScript object
     *
     * Only valid after checkSat() returns "sat"
     *
     * @returns Model object mapping variables to their values
     * @throws {WasmError} If no model is available
     *
     * @example
     * ```typescript
     * if (solver.checkSat() === 'sat') {
     *   const model = solver.getModel();
     *   console.log(model.x.value); // Get value of x
     *   console.log(model.x.sort);  // Get sort of x
     * }
     * ```
     */
    getModel(): Model;

    /**
     * Get the model as SMT-LIB2 formatted string
     *
     * Only valid after checkSat() returns "sat"
     *
     * @returns SMT-LIB2 formatted model string
     * @throws {WasmError} If no model is available
     *
     * @example
     * ```typescript
     * const modelStr = solver.getModelString();
     * console.log(modelStr);
     * ```
     */
    getModelString(): string;

    /**
     * Get values of specific terms in the model
     *
     * Only valid after checkSat() returns "sat"
     *
     * @param terms - Array of SMT-LIB2 term strings to evaluate
     * @returns SMT-LIB2 formatted values
     * @throws {WasmError} If no model is available or terms are invalid
     *
     * @example
     * ```typescript
     * const values = solver.getValue(['x', '(+ x 1)']);
     * ```
     */
    getValue(terms: string[]): string;

    /**
     * Get the unsatisfiable core
     *
     * Only valid after checkSat() returns "unsat"
     *
     * @returns SMT-LIB2 formatted unsat core
     * @throws {WasmError} If no unsat core is available
     *
     * @example
     * ```typescript
     * if (solver.checkSat() === 'unsat') {
     *   const core = solver.getUnsatCore();
     *   console.log(core);
     * }
     * ```
     */
    getUnsatCore(): string;

    /**
     * Push a new context level (create backtracking point)
     *
     * @example
     * ```typescript
     * solver.push();
     * solver.assertFormula('(> x 10)');
     * solver.checkSat();
     * solver.pop(); // Undo the assertion
     * ```
     */
    push(): void;

    /**
     * Pop a context level (backtrack to previous state)
     *
     * @example
     * ```typescript
     * solver.pop();
     * ```
     */
    pop(): void;

    /**
     * Reset the solver completely
     *
     * Clears all assertions, declarations, and options
     *
     * @example
     * ```typescript
     * solver.reset();
     * ```
     */
    reset(): void;

    /**
     * Simplify an SMT-LIB2 expression
     *
     * @param expr - Expression to simplify
     * @returns Simplified expression
     * @throws {WasmError} If the expression is invalid
     *
     * @example
     * ```typescript
     * const simplified = solver.simplify('(+ 1 2)');
     * console.log(simplified); // "3"
     * ```
     */
    simplify(expr: string): string;

    // Formula Builder Methods

    /**
     * Create an equality expression
     *
     * @param lhs - Left-hand side expression
     * @param rhs - Right-hand side expression
     * @returns SMT-LIB2 equality expression
     */
    mkEq(lhs: string, rhs: string): string;

    /**
     * Create a conjunction (AND) expression
     *
     * @param exprs - Array of boolean expressions
     * @returns SMT-LIB2 AND expression
     */
    mkAnd(exprs: string[]): string;

    /**
     * Create a disjunction (OR) expression
     *
     * @param exprs - Array of boolean expressions
     * @returns SMT-LIB2 OR expression
     */
    mkOr(exprs: string[]): string;

    /**
     * Create a negation (NOT) expression
     *
     * @param expr - Boolean expression to negate
     * @returns SMT-LIB2 NOT expression
     */
    mkNot(expr: string): string;

    /**
     * Create an implication expression
     *
     * @param lhs - Antecedent (if this)
     * @param rhs - Consequent (then that)
     * @returns SMT-LIB2 implication expression
     */
    mkImplies(lhs: string, rhs: string): string;

    /**
     * Create an if-then-else expression
     *
     * @param cond - Condition expression
     * @param thenExpr - Expression if condition is true
     * @param elseExpr - Expression if condition is false
     * @returns SMT-LIB2 ITE expression
     */
    mkIte(cond: string, thenExpr: string, elseExpr: string): string;

    /**
     * Create an exclusive-or (XOR) expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 XOR expression
     */
    mkXor(lhs: string, rhs: string): string;

    /**
     * Create a less-than comparison
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 < expression
     */
    mkLt(lhs: string, rhs: string): string;

    /**
     * Create a less-than-or-equal comparison
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 <= expression
     */
    mkLe(lhs: string, rhs: string): string;

    /**
     * Create a greater-than comparison
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 > expression
     */
    mkGt(lhs: string, rhs: string): string;

    /**
     * Create a greater-than-or-equal comparison
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 >= expression
     */
    mkGe(lhs: string, rhs: string): string;

    /**
     * Create an addition expression
     *
     * @param args - Array of numeric expressions
     * @returns SMT-LIB2 + expression
     */
    mkAdd(args: string[]): string;

    /**
     * Create a subtraction expression
     *
     * @param args - Array of numeric expressions
     * @returns SMT-LIB2 - expression
     */
    mkSub(args: string[]): string;

    /**
     * Create a multiplication expression
     *
     * @param args - Array of numeric expressions
     * @returns SMT-LIB2 * expression
     */
    mkMul(args: string[]): string;

    /**
     * Create a division expression
     *
     * @param args - Array of numeric expressions
     * @returns SMT-LIB2 / expression
     */
    mkDiv(args: string[]): string;

    /**
     * Create a modulo expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 mod expression
     */
    mkMod(lhs: string, rhs: string): string;

    /**
     * Create an arithmetic negation expression
     *
     * @param expr - Expression to negate
     * @returns SMT-LIB2 unary - expression
     */
    mkNeg(expr: string): string;

    /**
     * Create a distinct (all-different) expression
     *
     * @param args - Array of expressions that must have different values
     * @returns SMT-LIB2 distinct expression
     */
    mkDistinct(args: string[]): string;

    // Bitvector Operations

    /**
     * Create a bitvector AND expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 bvand expression
     */
    mkBvAnd(lhs: string, rhs: string): string;

    /**
     * Create a bitvector OR expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 bvor expression
     */
    mkBvOr(lhs: string, rhs: string): string;

    /**
     * Create a bitvector XOR expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 bvxor expression
     */
    mkBvXor(lhs: string, rhs: string): string;

    /**
     * Create a bitvector NOT expression
     *
     * @param expr - Expression to negate
     * @returns SMT-LIB2 bvnot expression
     */
    mkBvNot(expr: string): string;

    /**
     * Create a bitvector negation expression
     *
     * @param expr - Expression to negate
     * @returns SMT-LIB2 bvneg expression
     */
    mkBvNeg(expr: string): string;

    /**
     * Create a bitvector addition expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 bvadd expression
     */
    mkBvAdd(lhs: string, rhs: string): string;

    /**
     * Create a bitvector subtraction expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 bvsub expression
     */
    mkBvSub(lhs: string, rhs: string): string;

    /**
     * Create a bitvector multiplication expression
     *
     * @param lhs - Left operand
     * @param rhs - Right operand
     * @returns SMT-LIB2 bvmul expression
     */
    mkBvMul(lhs: string, rhs: string): string;

    // Batch Operations

    /**
     * Declare multiple constants at once
     *
     * More efficient than calling declareConst multiple times
     *
     * @param declarations - Array of declaration strings in format "name sort"
     * @throws {WasmError} If any declaration is invalid
     *
     * @example
     * ```typescript
     * solver.declareFuns([
     *   'x Int',
     *   'y Int',
     *   'z Bool'
     * ]);
     * ```
     */
    declareFuns(declarations: string[]): void;

    /**
     * Assert multiple formulas at once
     *
     * More efficient than calling assertFormula multiple times
     *
     * @param formulas - Array of SMT-LIB2 boolean expressions
     * @throws {WasmError} If any formula is invalid
     *
     * @example
     * ```typescript
     * solver.assertFormulas([
     *   '(> x 0)',
     *   '(< y 10)',
     *   '(< x y)'
     * ]);
     * ```
     */
    assertFormulas(formulas: string[]): void;

    /**
     * Assert a formula with a label/name
     *
     * Useful for tracking which assertions appear in unsat cores
     *
     * @param name - Unique name for this assertion
     * @param formula - SMT-LIB2 boolean expression
     * @throws {WasmError} If the formula is invalid
     *
     * @example
     * ```typescript
     * solver.setOption('produce-unsat-cores', 'true');
     * solver.assertNamed('positive', '(> x 0)');
     * solver.assertNamed('negative', '(< x 0)');
     * if (solver.checkSat() === 'unsat') {
     *   const core = solver.getUnsatCore();
     *   console.log(core); // Shows which named assertions conflict
     * }
     * ```
     */
    assertNamed(name: string, formula: string): void;

    /**
     * Cancel the current operation
     *
     * Sets a cancellation flag that may be checked during long operations
     *
     * @example
     * ```typescript
     * setTimeout(() => solver.cancel(), 5000);
     * await solver.checkSatAsync();
     * ```
     */
    cancel(): void;

    /**
     * Check if cancellation was requested
     *
     * @returns true if cancel() was called, false otherwise
     *
     * @example
     * ```typescript
     * if (solver.isCancelled()) {
     *   console.log('Operation was cancelled');
     * }
     * ```
     */
    isCancelled(): boolean;

    /**
     * Free the solver resources
     *
     * Called automatically, but can be called manually for explicit cleanup
     */
    free(): void;
}

/**
 * Get the version of OxiZ WASM
 *
 * @returns Version string in semver format
 *
 * @example
 * ```typescript
 * import { version } from 'oxiz-wasm';
 * console.log(`OxiZ WASM version: ${version()}`);
 * ```
 */
export function version(): string;

/**
 * Initialize the WASM module
 *
 * This function is automatically called on import, but can be called
 * manually if needed. It loads and initializes the WebAssembly module.
 *
 * @param module_or_path - Optional WASM module or path to WASM file
 * @returns Promise that resolves when initialization is complete
 *
 * @example
 * ```typescript
 * import init from 'oxiz-wasm';
 * await init();
 * ```
 */
export default function init(module_or_path?: WebAssembly.Module | string | URL): Promise<void>;
