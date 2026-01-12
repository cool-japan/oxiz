/**
 * Example of incremental solving with push/pop
 *
 * Run with: deno run --allow-read incremental.ts
 */

import { WasmSolver, initOxiZ } from "../mod.ts";

async function main() {
  console.log("Initializing OxiZ WASM...");
  await initOxiZ();

  const solver = new WasmSolver();
  solver.setLogic("QF_LIA");
  solver.declareConst("x", "Int");
  solver.declareConst("y", "Int");

  // Base constraints
  solver.assertFormula("(> x 0)");
  solver.assertFormula("(> y 0)");

  console.log("Base constraints: x > 0 and y > 0");

  // Scenario 1
  console.log("\nScenario 1: x + y = 10");
  solver.push();
  solver.assertFormula("(= (+ x y) 10)");
  console.log("Result:", solver.checkSat());
  if (solver.checkSat() === "sat") {
    const model = solver.getModel();
    console.log("Model:", model);
  }
  solver.pop();

  // Scenario 2
  console.log("\nScenario 2: x > 100 and y < 5");
  solver.push();
  solver.assertFormula("(> x 100)");
  solver.assertFormula("(< y 5)");
  console.log("Result:", solver.checkSat());
  if (solver.checkSat() === "sat") {
    const model = solver.getModel();
    console.log("Model:", model);
  }
  solver.pop();

  // Scenario 3
  console.log("\nScenario 3: x = y");
  solver.push();
  solver.assertFormula("(= x y)");
  console.log("Result:", solver.checkSat());
  if (solver.checkSat() === "sat") {
    const model = solver.getModel();
    console.log("Model:", model);
  }
  solver.pop();

  console.log("\nDone!");
}

if (import.meta.main) {
  main().catch(console.error);
}
