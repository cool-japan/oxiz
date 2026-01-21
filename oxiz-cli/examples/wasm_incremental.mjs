// Incremental Solving with OxiZ WASM
// Run with: node --experimental-wasm-modules wasm_incremental.mjs
//
// This example demonstrates incremental solving using push/pop operations
// to efficiently explore multiple scenarios without recreating the solver.

import init, { WasmSolver } from 'oxiz-wasm';

async function main() {
    await init();
    const solver = new WasmSolver();

    console.log("=== Incremental Solving Example ===\n");

    solver.setLogic("QF_LIA");

    // Declare variables for a scheduling problem
    solver.declareConst("task_a_start", "Int");
    solver.declareConst("task_b_start", "Int");
    solver.declareConst("task_a_duration", "Int");
    solver.declareConst("task_b_duration", "Int");

    // Base constraints (always active)
    // Tasks must start at time >= 0
    solver.assertFormula("(>= task_a_start 0)");
    solver.assertFormula("(>= task_b_start 0)");

    // Fixed durations
    solver.assertFormula("(= task_a_duration 5)");
    solver.assertFormula("(= task_b_duration 3)");

    // Both tasks must complete by time 20
    solver.assertFormula("(<= (+ task_a_start task_a_duration) 20)");
    solver.assertFormula("(<= (+ task_b_start task_b_duration) 20)");

    console.log("Base constraints:");
    console.log("  - Tasks start at time >= 0");
    console.log("  - Task A duration = 5, Task B duration = 3");
    console.log("  - Both tasks must complete by time 20");

    // Scenario 1: Task A must finish before Task B starts
    console.log("\n--- Scenario 1: A before B ---");
    solver.push();
    solver.assertFormula("(<= (+ task_a_start task_a_duration) task_b_start)");

    if (solver.checkSat() === "sat") {
        const model = solver.getModel();
        console.log("Satisfiable!");
        console.log("  Task A: starts at", model.task_a_start.value,
                    ", ends at", Number(model.task_a_start.value) + 5);
        console.log("  Task B: starts at", model.task_b_start.value,
                    ", ends at", Number(model.task_b_start.value) + 3);
    }
    solver.pop();

    // Scenario 2: Task B must finish before Task A starts
    console.log("\n--- Scenario 2: B before A ---");
    solver.push();
    solver.assertFormula("(<= (+ task_b_start task_b_duration) task_a_start)");

    if (solver.checkSat() === "sat") {
        const model = solver.getModel();
        console.log("Satisfiable!");
        console.log("  Task A: starts at", model.task_a_start.value,
                    ", ends at", Number(model.task_a_start.value) + 5);
        console.log("  Task B: starts at", model.task_b_start.value,
                    ", ends at", Number(model.task_b_start.value) + 3);
    }
    solver.pop();

    // Scenario 3: Tasks must run concurrently (overlap)
    console.log("\n--- Scenario 3: Tasks must overlap ---");
    solver.push();
    // A starts before B ends AND B starts before A ends
    solver.assertFormula("(< task_a_start (+ task_b_start task_b_duration))");
    solver.assertFormula("(< task_b_start (+ task_a_start task_a_duration))");

    if (solver.checkSat() === "sat") {
        const model = solver.getModel();
        console.log("Satisfiable!");
        console.log("  Task A: starts at", model.task_a_start.value,
                    ", ends at", Number(model.task_a_start.value) + 5);
        console.log("  Task B: starts at", model.task_b_start.value,
                    ", ends at", Number(model.task_b_start.value) + 3);
    }
    solver.pop();

    // Scenario 4: Impossible constraint (both must start and end at same time)
    console.log("\n--- Scenario 4: Impossible constraints ---");
    solver.push();
    solver.assertFormula("(= task_a_start task_b_start)");
    solver.assertFormula("(= (+ task_a_start task_a_duration) (+ task_b_start task_b_duration))");
    // This means duration must be equal, but they're 5 and 3

    const result = solver.checkSat();
    console.log("Result:", result);
    if (result === "unsat") {
        console.log("(As expected - tasks have different fixed durations)");
    }
    solver.pop();

    solver.free();
    console.log("\nDone!");
}

main().catch(console.error);
