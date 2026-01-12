/**
 * Deno tests for OxiZ WASM
 *
 * Run with: deno test --allow-read examples/deno-test.ts
 */

import { assertEquals, assertExists, assertThrows } from 'https://deno.land/std@0.208.0/assert/mod.ts';
import init, { WasmSolver, version } from '../pkg/oxiz_wasm.js';

// Initialize WASM before tests
await init();

Deno.test('Version is not empty', () => {
    const v = version();
    assertExists(v);
    assertEquals(typeof v, 'string');
});

Deno.test('Create new solver', () => {
    const solver = new WasmSolver();
    assertExists(solver);
});

Deno.test('Check-sat on empty solver returns sat', () => {
    const solver = new WasmSolver();
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Declare boolean constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Declare integer constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Declare real constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('r', 'Real');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Declare bitvector constant', () => {
    const solver = new WasmSolver();
    solver.declareConst('bv', 'BitVec32');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Assert simple boolean formula (SAT)', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Assert contradictory formulas (UNSAT)', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    solver.assertFormula('(not p)');
    const result = solver.checkSat();
    assertEquals(result, 'unsat');
});

Deno.test('Integer arithmetic (SAT)', () => {
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

Deno.test('Integer arithmetic (UNSAT)', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_LIA');
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 10)');
    solver.assertFormula('(= x 20)');
    const result = solver.checkSat();
    assertEquals(result, 'unsat');
});

Deno.test('Real arithmetic', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_LRA');
    solver.declareConst('x', 'Real');
    solver.assertFormula('(> x 0.0)');
    solver.assertFormula('(< x 1.0)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('BitVector operations', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_BV');
    solver.declareConst('bv1', 'BitVec8');
    solver.declareConst('bv2', 'BitVec8');
    solver.assertFormula('(= bv1 #b00000001)');
    solver.assertFormula('(= bv2 #b00000010)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Push and pop', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');

    solver.push();
    solver.assertFormula('(not p)');
    assertEquals(solver.checkSat(), 'unsat');

    solver.pop();
    assertEquals(solver.checkSat(), 'sat');
});

Deno.test('Multiple push/pop levels', () => {
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

Deno.test('Reset clears all state', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 42)');
    solver.checkSat();

    solver.reset();
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Reset assertions preserves declarations', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 42)');

    solver.resetAssertions();

    // x should still be declared
    solver.assertFormula('(= x 100)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});

Deno.test('Get model after SAT', () => {
    const solver = new WasmSolver();
    solver.declareConst('x', 'Int');
    solver.assertFormula('(= x 42)');
    solver.checkSat();

    const model = solver.getModel();
    assertExists(model);
});

Deno.test('Get model string after SAT', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    solver.checkSat();

    const modelString = solver.getModelString();
    assertEquals(typeof modelString, 'string');
});

Deno.test('Get assertions', () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');

    const assertions = solver.getAssertions();
    assertEquals(typeof assertions, 'string');
});

Deno.test('Simplify arithmetic', () => {
    const solver = new WasmSolver();
    const result = solver.simplify('(+ 1 2)');
    assertEquals(result, '3');
});

Deno.test('Simplify boolean', () => {
    const solver = new WasmSolver();
    const result = solver.simplify('(and true false)');
    assertEquals(result, 'false');
});

Deno.test('Set and get option', () => {
    const solver = new WasmSolver();
    solver.setOption('produce-models', 'true');
    const value = solver.getOption('produce-models');
    assertEquals(value, 'true');
});

Deno.test('Execute SMT-LIB2 script', () => {
    const solver = new WasmSolver();
    const result = solver.execute('(check-sat)');
    assertExists(result);
});

Deno.test('Async check-sat', async () => {
    const solver = new WasmSolver();
    solver.declareConst('p', 'Bool');
    solver.assertFormula('p');
    const result = await solver.checkSatAsync();
    assertEquals(result, 'sat');
});

Deno.test('Async execute', async () => {
    const solver = new WasmSolver();
    const result = await solver.executeAsync('(check-sat)');
    assertExists(result);
});

Deno.test('Complex boolean formula', () => {
    const solver = new WasmSolver();
    solver.setLogic('QF_UF');
    for (let i = 0; i < 10; i++) {
        solver.declareConst(`p${i}`, 'Bool');
    }
    solver.assertFormula('(or p0 p1 p2 p3 p4 p5 p6 p7 p8 p9)');
    const result = solver.checkSat();
    assertEquals(result, 'sat');
});
