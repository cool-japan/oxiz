/**
 * Node.js integration tests for OxiZ WASM
 *
 * Tests the WASM bindings in a Node.js environment
 */

const { WasmSolver, version } = require('../pkg/oxiz_wasm');

// Simple test framework
class TestRunner {
    constructor() {
        this.passed = 0;
        this.failed = 0;
        this.tests = [];
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    async run() {
        console.log('='.repeat(60));
        console.log(`OxiZ WASM Node.js Integration Tests v${version()}`);
        console.log('='.repeat(60));
        console.log();

        for (const { name, fn } of this.tests) {
            try {
                await fn();
                console.log(`✓ ${name}`);
                this.passed++;
            } catch (error) {
                console.log(`✗ ${name}`);
                console.error(`  Error: ${error.message}`);
                this.failed++;
            }
        }

        console.log();
        console.log('='.repeat(60));
        console.log(`Tests: ${this.passed} passed, ${this.failed} failed, ${this.passed + this.failed} total`);
        console.log('='.repeat(60));

        return this.failed === 0;
    }
}

function assertEquals(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
}

function assertNotNull(value, message) {
    if (value === null || value === undefined) {
        throw new Error(message || 'Expected value to not be null');
    }
}

function assertTrue(condition, message) {
    if (!condition) {
        throw new Error(message || 'Expected condition to be true');
    }
}

function assertThrows(fn, message) {
    let threw = false;
    try {
        fn();
    } catch (e) {
        threw = true;
    }
    if (!threw) {
        throw new Error(message || 'Expected function to throw');
    }
}

// Create test runner
const runner = new TestRunner();

// Basic tests
runner.test('Version is not empty', () => {
    const v = version();
    assertTrue(v.length > 0, 'Version should not be empty');
});

runner.test('Create new solver', () => {
    const solver = new WasmSolver();
    assertNotNull(solver, 'Solver should not be null');
});

runner.test('Check-sat on empty solver returns sat', () => {
    const solver = new WasmSolver();
    const result = solver.checkSat();
    assertEquals(result, 'sat', 'Empty solver should be satisfiable');
});

runner.test('Declare boolean constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Declare integer constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Declare real constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('r', 'Real');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Declare bitvector constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('bv', 'BitVec32');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Assert simple boolean formula (SAT)', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Assert contradictory formulas (UNSAT)', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    solver.assertFormula('(not p)');
    const result = solver.checkSat();
    assertEquals(result, 'unsat');
});

runner.test('Integer arithmetic (SAT)', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_LIA');
    solver.declareConst('x', 'Int');
    solver.declareConst('y', 'Int');
    solver.assertFormula('(= x 10)');
    solver.assertFormula('(= y 20)');
    solver.assertFormula('(= (+ x y) 30)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Integer arithmetic (UNSAT)', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_LIA');
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 10)');
    solver.assertFormula('(= x 20)');
    const result = solver.checkSat();
    assertEquals(result, 'unsat');
});

runner.test('Real arithmetic', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_LRA');
    solver.declareConst('x', 'Real');
    solver.assertFormula('(> x 0.0)');
    solver.assertFormula('(< x 1.0)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('BitVector operations', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_BV');
    solver.declareConst('bv1', 'BitVec8');
    solver.declareConst('bv2', 'BitVec8');
    solver.assertFormula('(= bv1 #b00000001)');
    solver.assertFormula('(= bv2 #b00000010)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Push and pop', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');

    solver.push();
    solver.assertFormula('(not p)');
    let result = solver.checkSat();
    assertEquals(result, 'unsat');

    solver.pop();
    result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Multiple push/pop levels', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');

    solver.push();
    solver.assertFormula('p');
    assertEquals(solver.checkSat(), 'sat');

    solver.push();
    solver.assertFormula('(not p)');
    assertEquals(solver.checkSat(), 'unsat');

    solver.pop();
    assertEquals(solver.checkSat(), 'sat');

    solver.pop();
    assertEquals(solver.checkSat(), 'sat');
});

runner.test('Reset clears all state', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 42)');
    solver.checkSat();

    solver.reset();
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Reset assertions preserves declarations', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 42)');

    solver.resetAssertions();

    // x should still be declared
    solver.assertFormula('(= x 100)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Get model after SAT', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 42)');
    solver.checkSat();

    const model = solver.getModel();
    assertNotNull(model);
});

runner.test('Get model string after SAT', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    solver.checkSat();

    const modelString = solver.getModelString();
    assertTrue(modelString.includes('model'));
});

runner.test('Get assertions', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');

    const assertions = solver.getAssertions();
    assertTrue(assertions.includes('p'));
});

runner.test('Simplify arithmetic', () => {
    const solver = new WasmSolver();
    const result = solver.simplify('(+ 1 2)');
    assertEquals(result, '3');
});

runner.test('Simplify boolean', () => {
    const solver = new WasmSolver();
    const result = solver.simplify('(and true false)');
    assertEquals(result, 'false');
});

runner.test('Set and get option', () => {
    const solver = new WasmSolver();
    solver.setOption('produce-models', 'true');
    const value = solver.getOption('produce-models');
    assertEquals(value, 'true');
});

runner.test('Get non-existent option returns null', () => {
    const solver = new WasmSolver();
    const value = solver.getOption('nonexistent');
    assertTrue(value === null || value === undefined);
});

runner.test('Execute SMT-LIB2 script', () => {
    const solver = new WasmSolver();
    const result = solver.execute('(check-sat)');
    assertNotNull(result);
});

runner.test('Declare function (nullary)', () => {
    const solver = new WasmSolver();
    solver.declareFun('f', [], 'Int');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Invalid sort throws error', () => {
    const solver = new WasmSolver();
    assertThrows(() => {
        solver.declareConst('x', 'InvalidSort');
    });
});

runner.test('Empty script throws error', () => {
    const solver = new WasmSolver();
    assertThrows(() => {
        solver.execute('');
    });
});

runner.test('Empty formula throws error', () => {
    const solver = new WasmSolver();
    assertThrows(() => {
        solver.assertFormula('');
    });
});

runner.test('Empty expression in simplify throws error', () => {
    const solver = new WasmSolver();
    assertThrows(() => {
        solver.simplify('');
    });
});

runner.test('Complex boolean formula with multiple variables', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_UF');
    for (let i = 0; i < 10; i++) {
        solver.declareConst(`p${i}`, 'Bool');
    }
    solver.assertFormula('(or p0 p1 p2 p3 p4 p5 p6 p7 p8 p9)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

runner.test('Async check-sat', async () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    const result = await solver.checkSatAsync();
    assertEquals(result, 'sat');
});

runner.test('Async execute', async () => {
    const solver = new WasmSolver();
    const result = await solver.executeAsync('(check-sat)');
    assertNotNull(result);
});

// Run all tests
runner.run().then(success => {
    process.exit(success ? 0 : 1);
});
