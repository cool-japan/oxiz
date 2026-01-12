/**
 * Deno example for OxiZ WASM
 *
 * Run with: deno run --allow-read examples/deno-example.ts
 */

import init, { WasmSolver, version } from '../pkg/oxiz_wasm.js';

async function main() {
    // Initialize the WASM module
    await init();

    console.log('='.repeat(60));
    console.log(`OxiZ WASM Deno Example v${version()}`);
    console.log('='.repeat(60));
    console.log();

    // Example 1: Basic satisfiability checking
    console.log('Example 1: Basic SAT checking');
    console.log('-'.repeat(40));
    {
        const solver = new WasmSolver();
        solver.declareConst('p', 'Bool');
        solver.declareConst('q', 'Bool');

        solver.assertFormula('(or p q)');
        solver.assertFormula('(not p)');

        const result = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === 'sat') {
            const model = solver.getModel();
            console.log('Model:', model);
        }
    }
    console.log();

    // Example 2: Integer arithmetic
    console.log('Example 2: Integer Arithmetic');
    console.log('-'.repeat(40));
    {
        const solver = new WasmSolver();
        solver.setLogic('QF_LIA');

        solver.declareConst('x', 'Int');
        solver.declareConst('y', 'Int');

        solver.assertFormula('(= x 10)');
        solver.assertFormula('(= y 20)');
        solver.assertFormula('(= (+ x y) 30)');

        const result = solver.checkSat();
        console.log(`Result: ${result}`);

        if (result === 'sat') {
            const model = solver.getModel();
            console.log(`x = ${model.x.value}`);
            console.log(`y = ${model.y.value}`);
        }
    }
    console.log();

    // Example 3: Using push/pop
    console.log('Example 3: Backtracking with Push/Pop');
    console.log('-'.repeat(40));
    {
        const solver = new WasmSolver();
        solver.declareConst('p', 'Bool');

        solver.assertFormula('p');
        console.log('After asserting p:', solver.checkSat());

        solver.push();
        solver.assertFormula('(not p)');
        console.log('After asserting (not p):', solver.checkSat());

        solver.pop();
        console.log('After pop:', solver.checkSat());
    }
    console.log();

    // Example 4: Simplification
    console.log('Example 4: Expression Simplification');
    console.log('-'.repeat(40));
    {
        const solver = new WasmSolver();

        const expr1 = '(+ 1 2)';
        const simplified1 = solver.simplify(expr1);
        console.log(`${expr1} => ${simplified1}`);

        const expr2 = '(and true false)';
        const simplified2 = solver.simplify(expr2);
        console.log(`${expr2} => ${simplified2}`);

        const expr3 = '(or true false)';
        const simplified3 = solver.simplify(expr3);
        console.log(`${expr3} => ${simplified3}`);
    }
    console.log();

    // Example 5: BitVectors
    console.log('Example 5: BitVector Operations');
    console.log('-'.repeat(40));
    {
        const solver = new WasmSolver();
        solver.setLogic('QF_BV');

        solver.declareConst('bv1', 'BitVec8');
        solver.declareConst('bv2', 'BitVec8');

        solver.assertFormula('(= bv1 #b00000001)');
        solver.assertFormula('(= bv2 #b00000010)');

        const result = solver.checkSat();
        console.log(`Result: ${result}`);
    }
    console.log();

    // Example 6: Real arithmetic
    console.log('Example 6: Real Arithmetic');
    console.log('-'.repeat(40));
    {
        const solver = new WasmSolver();
        solver.setLogic('QF_LRA');

        solver.declareConst('x', 'Real');
        solver.assertFormula('(> x 0.0)');
        solver.assertFormula('(< x 1.0)');

        const result = solver.checkSat();
        console.log(`Result: ${result}`);
    }
    console.log();

    // Example 7: Async operations
    console.log('Example 7: Async Operations');
    console.log('-'.repeat(40));
    {
        const solver = new WasmSolver();
        solver.declareConst('p', 'Bool');
        solver.assertFormula('p');

        const result = await solver.checkSatAsync();
        console.log(`Async result: ${result}`);
    }
    console.log();

    console.log('='.repeat(60));
    console.log('All examples completed successfully!');
    console.log('='.repeat(60));
}

// Run the main function
if (import.meta.main) {
    main().catch(console.error);
}

export { main };
