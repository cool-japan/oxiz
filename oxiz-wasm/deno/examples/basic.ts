/**
 * Basic example of using OxiZ with Deno
 *
 * Run with: deno run --allow-read basic.ts
 */

import { WasmSolver, initOxiZ } from "../mod.ts";

async function main() {
  console.log("Initializing OxiZ WASM...");
  await initOxiZ();

  console.log("Creating solver...");
  const solver = new WasmSolver();

  // Set logic and configuration
  solver.setLogic("QF_LIA");
  solver.applyPreset("complete");

  // Declare variables
  console.log("Declaring variables...");
  solver.declareConst("x", "Int");
  solver.declareConst("y", "Int");

  // Add constraints
  console.log("Adding constraints...");
  solver.assertFormula("(> x 0)");
  solver.assertFormula("(< y 10)");
  solver.assertFormula("(= (+ x y) 15)");

  // Check satisfiability
  console.log("Checking satisfiability...");
  const result = solver.checkSat();
  console.log("Result:", result);

  if (result === "sat") {
    console.log("\nGetting model...");
    const model = solver.getModel();
    console.log("Model:", JSON.stringify(model, null, 2));

    console.log("\nGetting specific values...");
    const values = solver.getValue(["x", "y", "(+ x y)"]);
    console.log("Values:", values);
  }

  console.log("\nSolver statistics:");
  const stats = solver.getStatistics();
  console.log(JSON.stringify(stats, null, 2));

  console.log("\nDone!");
}

if (import.meta.main) {
  main().catch(console.error);
}
