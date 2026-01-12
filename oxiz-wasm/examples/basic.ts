/**
 * Basic TypeScript example for OxiZ WASM
 *
 * This example demonstrates type-safe usage of the OxiZ SMT solver
 * with full TypeScript support.
 */

import init, { WasmSolver, SatResult, version } from '../pkg/oxiz_wasm';

async function main(): Promise<void> {
    // Initialize the WASM module
    await init();

    console.log(`OxiZ WASM version: ${version()}\n`);

    // Example 1: Boolean Satisfiability with Type Safety
    console.log("=== Example 1: Boolean Satisfiability ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_UF");

        solver.declareConst("p", "Bool");
        solver.declareConst("q", "Bool");

        solver.assertFormula("(or p q)");
        solver.assertFormula("(not p)");

        const result: SatResult = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "sat") {
            const model = solver.getModel();
            console.log("Model:", JSON.stringify(model, null, 2));
        }
    }
    console.log();

    // Example 2: Integer Arithmetic with Type Annotations
    console.log("=== Example 2: Integer Arithmetic ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_LIA");

        solver.declareConst("x", "Int");
        solver.declareConst("y", "Int");

        // Find x and y such that: x >= 0, y >= 0, x + y = 10, x * 2 = y
        solver.assertFormula("(>= x 0)");
        solver.assertFormula("(>= y 0)");
        solver.assertFormula("(= (+ x y) 10)");
        solver.assertFormula("(= (* x 2) y)");

        const result: SatResult = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "sat") {
            const values: string = solver.getValue(["x", "y"]);
            console.log("Values:", values);
        }
    }
    console.log();

    // Example 3: Error Handling with Type Safety
    console.log("=== Example 3: Error Handling ===");
    {
        const solver = new WasmSolver();

        try {
            // This will throw an error
            solver.declareConst("x", "InvalidSort");
        } catch (error: any) {
            console.log("Caught error:");
            console.log(`  Kind: ${error.kind}`);
            console.log(`  Message: ${error.message}`);
        }

        try {
            // This will also throw an error
            solver.assertFormula("");
        } catch (error: any) {
            console.log("Caught error:");
            console.log(`  Kind: ${error.kind}`);
            console.log(`  Message: ${error.message}`);
        }
    }
    console.log();

    // Example 4: Advanced Type Usage
    console.log("=== Example 4: Advanced Solving ===");
    {
        interface SolverConfig {
            logic: string;
            options?: Record<string, string>;
        }

        function configureSolver(config: SolverConfig): WasmSolver {
            const solver = new WasmSolver();
            solver.setLogic(config.logic);

            if (config.options) {
                for (const [key, value] of Object.entries(config.options)) {
                    solver.setOption(key, value);
                }
            }

            return solver;
        }

        const config: SolverConfig = {
            logic: "QF_LIA",
            options: {
                "produce-models": "true",
            },
        };

        const solver = configureSolver(config);

        solver.declareConst("x", "Int");
        solver.assertFormula("(= x 42)");

        const result: SatResult = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "sat") {
            const model = solver.getModel();
            console.log("x =", model.x?.value);
        }
    }
    console.log();

    // Example 5: Async/Await with Type Safety
    console.log("=== Example 5: Async Solving ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_LIA");

        solver.declareConst("x", "Int");
        solver.assertFormula("(> x 0)");

        // Type-safe async operation
        const result: SatResult = await solver.checkSatAsync();
        console.log(`Async result: ${result}`);
    }
    console.log();

    // Example 6: Complex Assertions
    console.log("=== Example 6: Complex Constraints ===");
    {
        const solver = new WasmSolver();
        solver.setLogic("QF_LIA");

        // Declare variables
        const vars: string[] = ["x", "y", "z"];
        for (const v of vars) {
            solver.declareConst(v, "Int");
        }

        // Assert constraints
        const constraints: string[] = [
            "(>= x 0)",
            "(>= y 0)",
            "(>= z 0)",
            "(= (+ x y z) 100)",
            "(= (* x 2) y)",
            "(= (* y 3) z)",
        ];

        for (const constraint of constraints) {
            solver.assertFormula(constraint);
        }

        const result: SatResult = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === "sat") {
            const values = solver.getValue(vars);
            console.log("Solution:", values);
        }
    }
    console.log();

    // Example 7: Solver Options with Types
    console.log("=== Example 7: Solver Options ===");
    {
        const solver = new WasmSolver();

        // Set options
        solver.setOption("produce-models", "true");
        solver.setOption("random-seed", "42");

        // Get options
        const produceModels: string | undefined = solver.getOption("produce-models");
        const randomSeed: string | undefined = solver.getOption("random-seed");

        console.log(`produce-models: ${produceModels}`);
        console.log(`random-seed: ${randomSeed}`);
    }
    console.log();

    console.log("=== All examples completed successfully! ===");
}

main().catch((error: Error) => {
    console.error("Error:", error.message);
    process.exit(1);
});
