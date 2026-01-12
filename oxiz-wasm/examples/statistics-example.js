/**
 * Example: Using Solver Statistics
 *
 * This example demonstrates how to use the getStatistics() method
 * to analyze solver performance and behavior.
 */

import init, { WasmSolver } from '../pkg/oxiz_wasm.js';

await init();

console.log('=== Solver Statistics Example ===\n');

// Create a solver instance
const solver = new WasmSolver();
solver.setLogic("QF_UF");

// Declare some boolean variables
solver.declareConst("p", "Bool");
solver.declareConst("q", "Bool");
solver.declareConst("r", "Bool");

// Get initial statistics (before any solving)
console.log('Initial statistics (before solving):');
let stats = solver.getStatistics();
console.log(`  Decisions: ${stats.decisions}`);
console.log(`  Propagations: ${stats.propagations}`);
console.log(`  Conflicts: ${stats.conflicts}`);
console.log();

// Add some simple constraints
solver.assertFormula("(or p q)");
solver.assertFormula("(or (not p) r)");
solver.assertFormula("(or (not q) (not r))");

// Solve and check statistics
console.log('Solving simple problem...');
const result1 = solver.checkSat();
console.log(`Result: ${result1}`);

stats = solver.getStatistics();
console.log('Statistics after first solve:');
console.log(`  Decisions: ${stats.decisions}`);
console.log(`  Propagations: ${stats.propagations}`);
console.log(`  Conflicts: ${stats.conflicts}`);
console.log(`  Restarts: ${stats.restarts}`);
console.log();

// Add more complex constraints
solver.push();
solver.declareConst("s", "Bool");
solver.declareConst("t", "Bool");
solver.declareConst("u", "Bool");

solver.assertFormula("(and (or s t) (or (not s) u) (or (not t) (not u)))");

console.log('Solving more complex problem...');
const result2 = solver.checkSat();
console.log(`Result: ${result2}`);

stats = solver.getStatistics();
console.log('Statistics after second solve:');
console.log(`  Decisions: ${stats.decisions}`);
console.log(`  Propagations: ${stats.propagations}`);
console.log(`  Conflicts: ${stats.conflicts}`);
console.log(`  Restarts: ${stats.restarts}`);
console.log(`  Learned clauses: ${stats.learned_clauses}`);
console.log(`  Binary clauses: ${stats.binary_clauses}`);
console.log(`  Unit clauses: ${stats.unit_clauses}`);
console.log();

// Pop back to simpler problem
solver.pop();

// Create an unsatisfiable problem
solver.push();
solver.assertFormula("p");
solver.assertFormula("(not p)");

console.log('Solving unsatisfiable problem...');
const result3 = solver.checkSat();
console.log(`Result: ${result3}`);

stats = solver.getStatistics();
console.log('Statistics after UNSAT solve:');
console.log(`  Decisions: ${stats.decisions}`);
console.log(`  Propagations: ${stats.propagations}`);
console.log(`  Conflicts: ${stats.conflicts}`);
console.log(`  Restarts: ${stats.restarts}`);
console.log();

// Demonstrate statistics across multiple incremental solves
solver.reset();
solver.setLogic("QF_LIA");

console.log('=== Incremental Solving with Statistics ===\n');

solver.declareConst("x", "Int");
solver.declareConst("y", "Int");

for (let i = 1; i <= 5; i++) {
  solver.push();
  solver.assertFormula(`(= (+ x y) ${i * 10})`);
  solver.assertFormula(`(> x ${i - 1})`);

  const result = solver.checkSat();
  const stats = solver.getStatistics();

  console.log(`Iteration ${i}:`);
  console.log(`  Result: ${result}`);
  console.log(`  Decisions: ${stats.decisions}, Propagations: ${stats.propagations}, Conflicts: ${stats.conflicts}`);

  solver.pop();
}

console.log('\n=== Example Complete ===');
