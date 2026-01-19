// Basic OxiZ WASM Usage Example
// Run with: node --experimental-wasm-modules wasm_basic.mjs
//
// This example demonstrates basic usage of oxiz-wasm for SMT solving
// in JavaScript/Node.js environments.

import init, { WasmSolver } from 'oxiz-wasm';

async function main() {
    // Initialize WASM module (required before creating solver instances)
    await init();

    // Create a new solver instance
    const solver = new WasmSolver();

    console.log("=== Basic OxiZ WASM Example ===\n");

    // Set logic to Quantifier-Free Linear Integer Arithmetic
    solver.setLogic("QF_LIA");
    console.log("Logic: QF_LIA (Quantifier-Free Linear Integer Arithmetic)");

    // Declare integer variables
    solver.declareConst("x", "Int");
    solver.declareConst("y", "Int");
    console.log("Declared variables: x (Int), y (Int)");

    // Add constraints
    solver.assertFormula("(> x 0)");          // x > 0
    solver.assertFormula("(< y 100)");        // y < 100
    solver.assertFormula("(= (+ x y) 50)");   // x + y = 50
    console.log("Constraints: x > 0, y < 100, x + y = 50");

    // Check satisfiability
    console.log("\nSolving...");
    const result = solver.checkSat();
    console.log("Result:", result);

    if (result === "sat") {
        // Get satisfying model
        const model = solver.getModel();
        console.log("\nSatisfying Model:");
        console.log("  x =", model.x.value);
        console.log("  y =", model.y.value);
        console.log("  Verification: x + y =",
            Number(model.x.value) + Number(model.y.value));
    }

    // Clean up resources
    solver.free();
    console.log("\nDone!");
}

main().catch(console.error);
