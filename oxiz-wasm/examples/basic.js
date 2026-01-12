/**
 * Basic JavaScript example for OxiZ WASM
 *
 * This example demonstrates the core functionality of the OxiZ SMT solver
 * in a browser or Node.js environment.
 */

import init, { WasmSolver, version } from '../pkg/oxiz_wasm.js';

async function main() {
    // Initialize the WASM module
    await init();

    console.log(`OxiZ WASM version: ${version()}\n`);

    // Example 1: Boolean Satisfiability
    console.log("=== Example 1: Boolean Satisfiability ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_UF");

        // Declare boolean variables
        solver.declareConst("p", "Bool");
        solver.declareConst("q", "Bool");

        // Assert: (p OR q) AND (NOT p)
        solver.assertFormula("(or p q)");
        solver.assertFormula("(not p)");

        const result = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "sat") {
            const model = solver.getModel();
            console.log("Model:", JSON.stringify(model, null, 2));
        }
    }
    console.log();

    // Example 2: Integer Arithmetic
    console.log("=== Example 2: Integer Arithmetic ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_LIA");

        solver.declareConst("x", "Int");
        solver.declareConst("y", "Int");

        // Find x and y such that:
        // x >= 0, y >= 0, x + y = 10, x * 2 = y
        solver.assertFormula("(>= x 0)");
        solver.assertFormula("(>= y 0)");
        solver.assertFormula("(= (+ x y) 10)");
        solver.assertFormula("(= (* x 2) y)");

        const result = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "sat") {
            const values = solver.getValue(["x", "y", "(+ x y)"]);
            console.log("Values:", values);
        }
    }
    console.log();

    // Example 3: Push/Pop (Incremental Solving)
    console.log("=== Example 3: Incremental Solving ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_UF");

        solver.declareConst("p", "Bool");
        solver.assertFormula("p");

        console.log("After asserting p:");
        console.log(`  Result: ${solver.checkSat()}`);

        // Create a backtracking point
        solver.push();
        solver.assertFormula("(not p)");

        console.log("After asserting (not p):");
        console.log(`  Result: ${solver.checkSat()}`);

        // Backtrack
        solver.pop();

        console.log("After pop:");
        console.log(`  Result: ${solver.checkSat()}`);
    }
    console.log();

    // Example 4: Simplification
    console.log("=== Example 4: Simplification ===");
    {
        const solver = new WasmSolver();

        console.log("(+ 1 2) =>", solver.simplify("(+ 1 2)"));
        console.log("(and true false) =>", solver.simplify("(and true false)"));
        console.log("(or true false) =>", solver.simplify("(or true false)"));
    }
    console.log();

    // Example 5: Async Solving
    console.log("=== Example 5: Async Solving ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_LIA");

        solver.declareConst("x", "Int");
        solver.assertFormula("(> x 0)");

        const result = await solver.checkSatAsync();
        console.log(`Async result: ${result}`);
    }
    console.log();

    // Example 6: Unsat Core
    console.log("=== Example 6: Unsat Core ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_UF");

        solver.declareConst("p", "Bool");
        solver.assertFormula("p");
        solver.assertFormula("(not p)");

        const result = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "unsat") {
            const core = solver.getUnsatCore();
            console.log("Unsat core:", core);
        }
    }
    console.log();

    // Example 7: Bit-Vectors
    console.log("=== Example 7: Bit-Vectors ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_BV");

        solver.declareConst("bv1", "BitVec8");
        solver.declareConst("bv2", "BitVec8");

        solver.assertFormula("(= bv1 #b11110000)");
        solver.assertFormula("(= bv2 #b00001111)");

        const result = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "sat") {
            const model = solver.getModelString();
            console.log("Model:", model);
        }
    }
    console.log();

    console.log("=== All examples completed successfully! ===");
}

main().catch(console.error);
