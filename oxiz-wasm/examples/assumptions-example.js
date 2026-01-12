/**
 * Example: Check-SAT with Assumptions
 *
 * This example demonstrates how to use checkSatAssuming() to test
 * different scenarios without permanently adding assertions.
 */

import init, { WasmSolver } from '../pkg/oxiz_wasm.js';

await init();

console.log('=== Check-SAT with Assumptions Example ===\n');

// Create a solver instance
const solver = new WasmSolver();
solver.setLogic("QF_UF");

// Example 1: Boolean constraints with assumptions
console.log('Example 1: Testing different scenarios\n');

solver.declareConst("rain", "Bool");
solver.declareConst("wet", "Bool");
solver.declareConst("umbrella", "Bool");

// Base rule: if it rains, the ground gets wet
solver.assertFormula("(=> rain wet)");

console.log('Base constraint: (=> rain wet)');
console.log('Testing different assumptions:\n');

// Test: What if it rains?
let result = solver.checkSatAssuming(["rain"]);
console.log(`  Assume rain: ${result}`);
if (result === "sat") {
  console.log('    ✓ Consistent - it can rain');
}

// Test: What if it doesn't rain?
result = solver.checkSatAssuming(["(not rain)"]);
console.log(`  Assume (not rain): ${result}`);
if (result === "sat") {
  console.log('    ✓ Consistent - it may not rain');
}

// Test: What if it rains but ground is not wet? (should be UNSAT)
result = solver.checkSatAssuming(["rain", "(not wet)"]);
console.log(`  Assume rain AND (not wet): ${result}`);
if (result === "unsat") {
  console.log('    ✗ Inconsistent - violates our rule!');
}

// Test: What if ground is wet but it didn't rain? (should be SAT - other causes)
result = solver.checkSatAssuming(["(not rain)", "wet"]);
console.log(`  Assume (not rain) AND wet: ${result}`);
if (result === "sat") {
  console.log('    ✓ Consistent - ground can be wet from other sources');
}

console.log();

// Example 2: Configuration testing
console.log('Example 2: Feature configuration testing\n');

solver.reset();
solver.setLogic("QF_UF");

solver.declareConst("feature_a", "Bool");
solver.declareConst("feature_b", "Bool");
solver.declareConst("feature_c", "Bool");

// Feature dependencies and conflicts
solver.assertFormula("(=> feature_a feature_b)"); // A requires B
solver.assertFormula("(not (and feature_b feature_c))"); // B and C conflict

console.log('Constraints:');
console.log('  - Feature A requires Feature B');
console.log('  - Features B and C conflict');
console.log();

const configurations = [
  { name: "A only", assumption: ["feature_a", "(not feature_b)", "(not feature_c)"] },
  { name: "A + B", assumption: ["feature_a", "feature_b", "(not feature_c)"] },
  { name: "A + C", assumption: ["feature_a", "(not feature_b)", "feature_c"] },
  { name: "B + C", assumption: ["(not feature_a)", "feature_b", "feature_c"] },
  { name: "All features", assumption: ["feature_a", "feature_b", "feature_c"] },
  { name: "No features", assumption: ["(not feature_a)", "(not feature_b)", "(not feature_c)"] },
];

console.log('Testing configurations:');
for (const config of configurations) {
  const result = solver.checkSatAssuming(config.assumption);
  const status = result === "sat" ? "✓ Valid" : "✗ Invalid";
  console.log(`  ${config.name.padEnd(15)} ${status}`);
}

console.log();

// Example 3: Integer constraints with assumptions
console.log('Example 3: Planning with resource constraints\n');

solver.reset();
solver.setLogic("QF_LIA");

solver.declareConst("budget", "Int");
solver.declareConst("cost_a", "Int");
solver.declareConst("cost_b", "Int");
solver.declareConst("cost_c", "Int");

// Base constraints
solver.assertFormula("(>= budget 0)");
solver.assertFormula("(= cost_a 100)");
solver.assertFormula("(= cost_b 200)");
solver.assertFormula("(= cost_c 150)");

console.log('Fixed costs: A=$100, B=$200, C=$150');
console.log('Testing different budget scenarios:\n');

const budgets = [250, 300, 450, 500];

for (const b of budgets) {
  // Can we afford A and B with this budget?
  const canAffordAB = solver.checkSatAssuming([
    `(= budget ${b})`,
    "(>= budget (+ cost_a cost_b))"
  ]);

  // Can we afford all three?
  const canAffordAll = solver.checkSatAssuming([
    `(= budget ${b})`,
    "(>= budget (+ cost_a (+ cost_b cost_c)))"
  ]);

  console.log(`Budget $${b}:`);
  console.log(`  Can afford A+B: ${canAffordAB === "sat" ? "Yes ✓" : "No ✗"}`);
  console.log(`  Can afford A+B+C: ${canAffordAll === "sat" ? "Yes ✓" : "No ✗"}`);
}

console.log();

// Example 4: Temporal reasoning
console.log('Example 4: Temporal reasoning\n');

solver.reset();
solver.setLogic("QF_UF");

solver.declareConst("morning", "Bool");
solver.declareConst("afternoon", "Bool");
solver.declareConst("evening", "Bool");

// Only one time period can be true
solver.assertFormula("(or morning (or afternoon evening))");
solver.assertFormula("(not (and morning afternoon))");
solver.assertFormula("(not (and morning evening))");
solver.assertFormula("(not (and afternoon evening))");

console.log('Constraints: Exactly one time period is active');
console.log();

const timeScenarios = [
  { period: "morning", assumption: ["morning", "(not afternoon)", "(not evening)"] },
  { period: "afternoon", assumption: ["(not morning)", "afternoon", "(not evening)"] },
  { period: "evening", assumption: ["(not morning)", "(not afternoon)", "evening"] },
  { period: "multiple", assumption: ["morning", "afternoon", "(not evening)"] },
];

console.log('Testing time scenarios:');
for (const scenario of timeScenarios) {
  const result = solver.checkSatAssuming(scenario.assumption);
  const status = result === "sat" ? "✓ Valid" : "✗ Invalid";
  console.log(`  ${scenario.period.padEnd(12)} ${status}`);
}

console.log();

// Example 5: Using assumptions for what-if analysis
console.log('Example 5: What-if analysis\n');

solver.reset();
solver.setLogic("QF_LIA");

solver.declareConst("x", "Int");
solver.declareConst("y", "Int");

solver.assertFormula("(= (+ x y) 100)");
solver.assertFormula("(>= x 0)");
solver.assertFormula("(>= y 0)");

console.log('Base constraint: x + y = 100, x >= 0, y >= 0');
console.log();

const whatIfs = [
  { scenario: "x = 50", assumption: "(= x 50)" },
  { scenario: "x > 60", assumption: "(> x 60)" },
  { scenario: "x < 30", assumption: "(< x 30)" },
  { scenario: "x = y", assumption: "(= x y)" },
  { scenario: "x > y + 20", assumption: "(> x (+ y 20))" },
];

console.log('What-if scenarios:');
for (const { scenario, assumption } of whatIfs) {
  const result = solver.checkSatAssuming([assumption]);
  console.log(`  If ${scenario.padEnd(15)} → ${result === "sat" ? "Possible ✓" : "Impossible ✗"}`);

  if (result === "sat") {
    // We could get the model here, but assumptions don't persist
    // so we'd need to assert them permanently or use getValue
  }
}

console.log('\n=== Example Complete ===');
console.log('\nKey takeaway: checkSatAssuming() lets you test scenarios');
console.log('without modifying your assertion set. Perfect for what-if analysis!');
