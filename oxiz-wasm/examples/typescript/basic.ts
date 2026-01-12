/**
 * Basic TypeScript example for OxiZ WASM
 *
 * This example demonstrates type-safe usage of the OxiZ SMT solver
 * from TypeScript.
 */

import init, { WasmSolver, version, type Model, type SatResult } from '../../oxiz-wasm';

/**
 * Example 1: Boolean satisfiability with type safety
 */
async function booleanSat(): Promise<void> {
    console.log('=== Example 1: Boolean Satisfiability ===');

    const solver = new WasmSolver();
    solver.setLogic('QF_UF');

    // TypeScript ensures we use correct sort names
    solver.declareConst('p', 'Bool');
    solver.declareConst('q', 'Bool');

    solver.assertFormula('(and p (not q))');

    const result: SatResult = solver.checkSat();
    console.log(`Result: ${result}`);

    if (result === 'sat') {
        const model: Model = solver.getModel();
        console.log(`p = ${model.p.value} (${model.p.sort})`);
        console.log(`q = ${model.q.value} (${model.q.sort})`);
    }

    solver.free();
}

/**
 * Example 2: Integer linear arithmetic with type annotations
 */
async function integerArithmetic(): Promise<void> {
    console.log('\n=== Example 2: Integer Linear Arithmetic ===');

    const solver = new WasmSolver();
    solver.setLogic('QF_LIA');

    // Type-safe constant declarations
    const variables: Array<{name: string, sort: 'Int'}> = [
        { name: 'x', sort: 'Int' },
        { name: 'y', sort: 'Int' },
        { name: 'z', sort: 'Int' },
    ];

    for (const {name, sort} of variables) {
        solver.declareConst(name, sort);
    }

    // Add constraints
    const constraints: string[] = [
        '(> x 0)',
        '(> y 0)',
        '(> z 0)',
        '(= (+ x y z) 10)',
        '(< x y)',
        '(< y z)',
    ];

    for (const constraint of constraints) {
        solver.assertFormula(constraint);
    }

    const result: SatResult = solver.checkSat();
    console.log(`Result: ${result}`);

    if (result === 'sat') {
        const model: Model = solver.getModel();
        console.log(`Solution: x=${model.x.value}, y=${model.y.value}, z=${model.z.value}`);

        // Verify the solution
        const x = parseInt(model.x.value);
        const y = parseInt(model.y.value);
        const z = parseInt(model.z.value);
        console.log(`Verification: ${x} + ${y} + ${z} = ${x + y + z}`);
    }

    solver.free();
}

/**
 * Example 3: Error handling with TypeScript
 */
async function errorHandling(): Promise<void> {
    console.log('\n=== Example 3: Error Handling ===');

    const solver = new WasmSolver();

    try {
        // This will throw because the sort is invalid
        solver.declareConst('x', 'InvalidSort' as any);
    } catch (error: any) {
        console.log(`Caught error: ${error.kind}`);
        console.log(`Message: ${error.message}`);
    }

    try {
        // This will throw because formula is empty
        solver.assertFormula('');
    } catch (error: any) {
        console.log(`Caught error: ${error.kind}`);
        console.log(`Message: ${error.message}`);
    }

    try {
        // This will throw because no sat check was performed
        solver.getModel();
    } catch (error: any) {
        console.log(`Caught error: ${error.kind}`);
        console.log(`Message: ${error.message}`);
    }

    solver.free();
}

/**
 * Example 4: Async operations with proper typing
 */
async function asyncOperations(): Promise<void> {
    console.log('\n=== Example 4: Async Operations ===');

    const solver = new WasmSolver();
    solver.setLogic('QF_LIA');

    solver.declareConst('a', 'Int');
    solver.declareConst('b', 'Int');
    solver.assertFormula('(and (> a 0) (< b 100) (= (* a b) 42))');

    console.log('Checking satisfiability asynchronously...');
    const result: SatResult = await solver.checkSatAsync();
    console.log(`Result: ${result}`);

    if (result === 'sat') {
        const model: Model = solver.getModel();
        console.log(`a = ${model.a.value}, b = ${model.b.value}`);
    }

    solver.free();
}

/**
 * Example 5: Push/Pop with incremental solving
 */
async function incrementalSolving(): Promise<void> {
    console.log('\n=== Example 5: Incremental Solving ===');

    const solver = new WasmSolver();
    solver.setLogic('QF_LIA');

    solver.declareConst('x', 'Int');
    solver.assertFormula('(> x 0)');

    // First check
    console.log('Check 1: x > 0');
    let result: SatResult = solver.checkSat();
    console.log(`Result: ${result}`);

    // Push and add more constraints
    solver.push();
    solver.assertFormula('(< x 10)');

    console.log('\nCheck 2: x > 0 AND x < 10');
    result = solver.checkSat();
    console.log(`Result: ${result}`);

    if (result === 'sat') {
        const model = solver.getModel();
        console.log(`x = ${model.x.value}`);
    }

    // Pop back
    solver.pop();

    // Add different constraint
    solver.push();
    solver.assertFormula('(> x 1000)');

    console.log('\nCheck 3: x > 0 AND x > 1000');
    result = solver.checkSat();
    console.log(`Result: ${result}`);

    if (result === 'sat') {
        const model = solver.getModel();
        console.log(`x = ${model.x.value}`);
    }

    solver.free();
}

/**
 * Example 6: Bitvector operations
 */
async function bitvectorOperations(): Promise<void> {
    console.log('\n=== Example 6: Bitvector Operations ===');

    const solver = new WasmSolver();
    solver.setLogic('QF_BV');

    // 8-bit bitvectors
    solver.declareConst('bv1', 'BitVec8');
    solver.declareConst('bv2', 'BitVec8');

    // Add constraints using bitvector operations
    solver.assertFormula('(= (bvadd bv1 bv2) #x42)');  // bv1 + bv2 = 0x42 (66 in decimal)
    solver.assertFormula('(bvult bv1 #x20)');           // bv1 < 0x20 (32 in decimal)

    const result: SatResult = solver.checkSat();
    console.log(`Result: ${result}`);

    if (result === 'sat') {
        const model: Model = solver.getModel();
        console.log(`bv1 = ${model.bv1.value}`);
        console.log(`bv2 = ${model.bv2.value}`);
    }

    solver.free();
}

/**
 * Main function to run all examples
 */
async function main(): Promise<void> {
    console.log(`OxiZ WASM TypeScript Examples`);
    console.log(`Version: ${version()}\n`);

    await booleanSat();
    await integerArithmetic();
    await errorHandling();
    await asyncOperations();
    await incrementalSolving();
    await bitvectorOperations();

    console.log('\n=== All examples completed ===');
}

// Initialize WASM and run examples
init().then(main).catch((error) => {
    console.error('Failed to initialize WASM:', error);
    process.exit(1);
});
