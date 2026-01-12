/**
 * Advanced Features Example (TypeScript)
 *
 * Demonstrates the complete OxiZ WASM API including:
 * - Solver statistics
 * - Assumptions-based checking
 * - Solver information
 * - Incremental solving
 * - Error handling
 */

import init, {
  WasmSolver,
  SatResult,
  SolverStatistics,
  SolverInfo,
  version
} from '../pkg/oxiz_wasm.js';

// Initialize the WASM module
await init();

console.log('=== OxiZ WASM Advanced Features ===\n');

// Display solver information
console.log('--- Solver Information ---');
const solver = new WasmSolver();
const info: SolverInfo = solver.getInfo();
console.log(`Name: ${info.name}`);
console.log(`Version: ${info.version}`);
console.log(`Authors: ${info.authors}`);
console.log(`Description: ${info.description}`);
console.log(`Library version: ${version()}`);
console.log();

// Example 1: Statistics-driven optimization
console.log('--- Example 1: Performance Analysis ---\n');

solver.setLogic("QF_LIA");

// Declare variables for a scheduling problem
const numTasks = 5;
for (let i = 0; i < numTasks; i++) {
  solver.declareConst(`task_${i}`, "Int");
  solver.assertFormula(`(>= task_${i} 0)`);
  solver.assertFormula(`(<= task_${i} 10)`);
}

// Add precedence constraints
for (let i = 0; i < numTasks - 1; i++) {
  solver.assertFormula(`(< task_${i} task_${i + 1})`);
}

console.log(`Solving scheduling problem with ${numTasks} tasks...`);
const scheduleResult: SatResult = solver.checkSat();
console.log(`Result: ${scheduleResult}`);

const stats: SolverStatistics = solver.getStatistics();
console.log('\nSolver Statistics:');
console.log(`  Decisions:        ${stats.decisions}`);
console.log(`  Propagations:     ${stats.propagations}`);
console.log(`  Conflicts:        ${stats.conflicts}`);
console.log(`  Restarts:         ${stats.restarts}`);
console.log(`  Learned clauses:  ${stats.learned_clauses}`);
console.log(`  Binary clauses:   ${stats.binary_clauses}`);
console.log(`  Unit clauses:     ${stats.unit_clauses}`);
console.log(`  Minimizations:    ${stats.minimizations}`);

if (stats.conflicts > 0) {
  const avgLbd = stats.total_lbd / stats.learned_clauses;
  console.log(`  Avg LBD:          ${avgLbd.toFixed(2)}`);
}
console.log();

// Example 2: Assumptions for scenario testing
console.log('--- Example 2: Scenario Analysis with Assumptions ---\n');

solver.reset();
solver.setLogic("QF_LIA");

// Model a simple resource allocation problem
solver.declareConst("cpu_cores", "Int");
solver.declareConst("memory_gb", "Int");
solver.declareConst("disk_gb", "Int");

// Resource constraints
solver.assertFormula("(>= cpu_cores 1)");
solver.assertFormula("(>= memory_gb 1)");
solver.assertFormula("(>= disk_gb 10)");

// Cost model: CPU=$10/core, RAM=$5/GB, Disk=$1/GB
solver.declareConst("total_cost", "Int");
solver.assertFormula("(= total_cost (+ (* cpu_cores 10) (+ (* memory_gb 5) (* disk_gb 1))))");

interface Scenario {
  name: string;
  budget: number;
  minCpu?: number;
  minMemory?: number;
}

const scenarios: Scenario[] = [
  { name: "Budget server", budget: 50, minCpu: 2, minMemory: 4 },
  { name: "Mid-range", budget: 100, minCpu: 4, minMemory: 8 },
  { name: "High-end", budget: 200, minCpu: 8, minMemory: 16 },
  { name: "Ultra", budget: 500, minCpu: 16, minMemory: 32 },
];

console.log('Testing server configurations:');
for (const scenario of scenarios) {
  const assumptions: string[] = [
    `(<= total_cost ${scenario.budget})`,
  ];

  if (scenario.minCpu) {
    assumptions.push(`(>= cpu_cores ${scenario.minCpu})`);
  }
  if (scenario.minMemory) {
    assumptions.push(`(>= memory_gb ${scenario.minMemory})`);
  }

  try {
    const result: SatResult = solver.checkSatAssuming(assumptions);
    const feasible = result === "sat" ? "✓ Feasible" : "✗ Not feasible";
    console.log(`  ${scenario.name.padEnd(15)} (budget: $${scenario.budget}): ${feasible}`);
  } catch (error: any) {
    console.log(`  ${scenario.name}: Error - ${error.message}`);
  }
}
console.log();

// Example 3: Incremental solving with statistics tracking
console.log('--- Example 3: Incremental Problem Solving ---\n');

solver.reset();
solver.setLogic("QF_UF");

// Sudoku-like constraint satisfaction
const gridSize = 3;

console.log(`Creating ${gridSize}x${gridSize} constraint grid...`);
for (let i = 0; i < gridSize; i++) {
  for (let j = 0; j < gridSize; j++) {
    solver.declareConst(`cell_${i}_${j}`, "Bool");
  }
}

// Track statistics over incremental additions
interface IterationStats {
  iteration: number;
  constraints: number;
  result: SatResult;
  decisions: number;
  conflicts: number;
}

const iterations: IterationStats[] = [];

for (let iter = 1; iter <= gridSize; iter++) {
  solver.push();

  // Add row constraint: at least one true per row
  const rowLits: string[] = [];
  for (let j = 0; j < gridSize; j++) {
    rowLits.push(`cell_${iter - 1}_${j}`);
  }
  solver.assertFormula(`(or ${rowLits.join(" ")})`);

  const result: SatResult = solver.checkSat();
  const stats: SolverStatistics = solver.getStatistics();

  iterations.push({
    iteration: iter,
    constraints: iter,
    result,
    decisions: stats.decisions,
    conflicts: stats.conflicts,
  });
}

console.log('\nIncremental solving results:');
console.log('Iter | Constraints | Result | Decisions | Conflicts');
console.log('-----|-------------|--------|-----------|----------');
for (const it of iterations) {
  console.log(
    `  ${it.iteration}  |      ${it.constraints}      | ${it.result.padEnd(6)} |    ${String(it.decisions).padEnd(6)} |    ${it.conflicts}`
  );
}
console.log();

// Example 4: Error handling
console.log('--- Example 4: Error Handling ---\n');

solver.reset();
solver.setLogic("QF_UF");

console.log('Testing error cases:');

// Test 1: Empty assumptions
try {
  solver.checkSatAssuming([]);
  console.log('  ✗ Empty assumptions should fail');
} catch (error: any) {
  console.log(`  ✓ Empty assumptions: ${error.kind} - "${error.message}"`);
}

// Test 2: Invalid sort
try {
  solver.declareConst("x", "InvalidSort" as any);
  console.log('  ✗ Invalid sort should fail');
} catch (error: any) {
  console.log(`  ✓ Invalid sort: ${error.kind} - "${error.message}"`);
}

// Test 3: Empty formula
try {
  solver.assertFormula("");
  console.log('  ✗ Empty formula should fail');
} catch (error: any) {
  console.log(`  ✓ Empty formula: ${error.kind} - "${error.message}"`);
}

// Test 4: Get model before check-sat
try {
  solver.getModel();
  console.log('  ✗ Get model without check-sat should fail');
} catch (error: any) {
  console.log(`  ✓ Model without SAT: ${error.kind} - "${error.message}"`);
}

console.log();

// Example 5: Complex boolean formula with assumptions
console.log('--- Example 5: Boolean Logic Puzzles ---\n');

solver.reset();
solver.setLogic("QF_UF");

// Knights and Knaves puzzle
// - Knights always tell the truth
// - Knaves always lie
// - Person A says: "B is a knave"
// - Person B says: "We are of different types"

solver.declareConst("a_knight", "Bool"); // true if A is a knight
solver.declareConst("b_knight", "Bool"); // true if B is a knight

// A's statement: "B is a knave" (B is not a knight)
// If A is knight (truthful), then B is not knight
// If A is knave (liar), then B is knight
solver.assertFormula("(= a_knight (not b_knight))");

// B's statement: "We are of different types"
// If B is knight (truthful), then they are different
// If B is knave (liar), then they are the same
solver.assertFormula("(= b_knight (not (= a_knight b_knight)))");

console.log('Knights and Knaves puzzle:');
console.log('  A says: "B is a knave"');
console.log('  B says: "We are of different types"');
console.log();

const result: SatResult = solver.checkSat();
console.log(`Solution exists: ${result}`);

if (result === "sat") {
  const model = solver.getModel();
  const aType = model.a_knight.value === "true" ? "Knight" : "Knave";
  const bType = model.b_knight.value === "true" ? "Knight" : "Knave";

  console.log(`  A is a: ${aType}`);
  console.log(`  B is a: ${bType}`);
}

// Test different scenarios with assumptions
console.log('\nWhat if we assume A is a knight?');
const ifAKnight: SatResult = solver.checkSatAssuming(["a_knight"]);
console.log(`  Consistent: ${ifAKnight === "sat" ? "Yes" : "No"}`);

console.log('What if we assume both are knights?');
const ifBothKnights: SatResult = solver.checkSatAssuming(["a_knight", "b_knight"]);
console.log(`  Consistent: ${ifBothKnights === "sat" ? "Yes" : "No"}`);

console.log();

// Example 6: Performance comparison
console.log('--- Example 6: Performance Comparison ---\n');

solver.reset();
solver.setLogic("QF_LIA");

const problemSizes = [5, 10, 15, 20];

console.log('Comparing performance across problem sizes:');
console.log('Size | Time (ms) | Decisions | Conflicts | Propagations');
console.log('-----|-----------|-----------|-----------|-------------');

for (const size of problemSizes) {
  solver.push();

  // Create a linear constraint problem
  for (let i = 0; i < size; i++) {
    solver.declareConst(`v${i}`, "Int");
    solver.assertFormula(`(>= v${i} 0)`);
    solver.assertFormula(`(<= v${i} 100)`);

    if (i > 0) {
      solver.assertFormula(`(< v${i - 1} v${i})`);
    }
  }

  const startTime = performance.now();
  solver.checkSat();
  const endTime = performance.now();

  const stats: SolverStatistics = solver.getStatistics();
  const timeMs = (endTime - startTime).toFixed(2);

  console.log(
    ` ${String(size).padEnd(3)} | ${String(timeMs).padEnd(9)} | ${String(stats.decisions).padEnd(9)} | ${String(stats.conflicts).padEnd(9)} | ${stats.propagations}`
  );

  solver.pop();
}

console.log('\n=== All Examples Complete ===');
