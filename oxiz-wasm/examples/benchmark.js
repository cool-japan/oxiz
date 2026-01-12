/**
 * Performance benchmarks for OxiZ WASM
 *
 * Tests various operations and measures execution time
 */

const { WasmSolver, version } = require('../pkg/oxiz_wasm');

/**
 * Run a benchmark and measure execution time
 */
function benchmark(name, fn, iterations = 1000) {
    // Warmup
    for (let i = 0; i < 10; i++) {
        fn();
    }

    // Measure
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
        fn();
    }
    const end = performance.now();
    const totalTime = end - start;
    const avgTime = totalTime / iterations;

    console.log(`${name}:`);
    console.log(`  Total: ${totalTime.toFixed(2)}ms`);
    console.log(`  Average: ${avgTime.toFixed(4)}ms`);
    console.log(`  Ops/sec: ${(1000 / avgTime).toFixed(0)}`);
    console.log();
}

/**
 * Run all benchmarks
 */
function runBenchmarks() {
    console.log('='.repeat(60));
    console.log(`OxiZ WASM Performance Benchmarks v${version()}`);
    console.log('='.repeat(60));
    console.log();

    // Benchmark: Solver creation
    benchmark('Solver Creation', () => {
        const solver = new WasmSolver();
    }, 1000);

    // Benchmark: declare_const
    benchmark('Declare Const (Bool)', () => {
        const solver = new WasmSolver();
        solver.declareConst('x', 'Bool');
    }, 1000);

    // Benchmark: declare_const (Int)
    benchmark('Declare Const (Int)', () => {
        const solver = new WasmSolver();
        solver.declareConst('x', 'Int');
    }, 1000);

    // Benchmark: Simple check-sat
    benchmark('Check-Sat (Empty)', () => {
        const solver = new WasmSolver();
        solver.checkSat();
    }, 1000);

    // Benchmark: check-sat with simple assertion
    benchmark('Check-Sat (Bool SAT)', () => {
        const solver = new WasmSolver();
        solver.declareConst('p', 'Bool');
        solver.assertFormula('p');
        solver.checkSat();
    }, 1000);

    // Benchmark: check-sat with simple unsat
    benchmark('Check-Sat (Bool UNSAT)', () => {
        const solver = new WasmSolver();
        solver.declareConst('p', 'Bool');
        solver.assertFormula('p');
        solver.assertFormula('(not p)');
        solver.checkSat();
    }, 1000);

    // Benchmark: push/pop
    benchmark('Push/Pop', () => {
        const solver = new WasmSolver();
        solver.push();
        solver.pop();
    }, 1000);

    // Benchmark: reset
    benchmark('Reset', () => {
        const solver = new WasmSolver();
        solver.declareConst('x', 'Int');
        solver.assertFormula('(= x 42)');
        solver.reset();
    }, 1000);

    // Benchmark: Integer arithmetic (SAT)
    benchmark('Integer Arithmetic (SAT)', () => {
        const solver = new WasmSolver();
        solver.setLogic('QF_LIA');
        solver.declareConst('x', 'Int');
        solver.declareConst('y', 'Int');
        solver.assertFormula('(= x 10)');
        solver.assertFormula('(= y 20)');
        solver.assertFormula('(= (+ x y) 30)');
        solver.checkSat();
    }, 500);

    // Benchmark: Complex boolean formula
    benchmark('Complex Boolean Formula', () => {
        const solver = new WasmSolver();
        solver.setLogic('QF_UF');
        for (let i = 0; i < 10; i++) {
            solver.declareConst(`p${i}`, 'Bool');
        }
        solver.assertFormula('(or p0 p1 p2 p3 p4 p5 p6 p7 p8 p9)');
        solver.checkSat();
    }, 500);

    // Benchmark: Simplify
    benchmark('Simplify (+ 1 2)', () => {
        const solver = new WasmSolver();
        solver.simplify('(+ 1 2)');
    }, 1000);

    // Benchmark: Simplify boolean
    benchmark('Simplify (and true false)', () => {
        const solver = new WasmSolver();
        solver.simplify('(and true false)');
    }, 1000);

    // Benchmark: Get model
    benchmark('Get Model', () => {
        const solver = new WasmSolver();
        solver.declareConst('x', 'Int');
        solver.assertFormula('(= x 42)');
        solver.checkSat();
        solver.getModel();
    }, 500);

    // Benchmark: Get model string
    benchmark('Get Model String', () => {
        const solver = new WasmSolver();
        solver.declareConst('x', 'Int');
        solver.assertFormula('(= x 42)');
        solver.checkSat();
        solver.getModelString();
    }, 500);

    // Benchmark: Set/Get option
    benchmark('Set/Get Option', () => {
        const solver = new WasmSolver();
        solver.setOption('produce-models', 'true');
        solver.getOption('produce-models');
    }, 1000);

    // Benchmark: Multiple push/pop levels
    benchmark('Multiple Push/Pop (5 levels)', () => {
        const solver = new WasmSolver();
        solver.declareConst('p', 'Bool');
        for (let i = 0; i < 5; i++) {
            solver.push();
            solver.assertFormula('p');
        }
        for (let i = 0; i < 5; i++) {
            solver.pop();
        }
    }, 500);

    // Benchmark: BitVector operations
    benchmark('BitVector Operations', () => {
        const solver = new WasmSolver();
        solver.setLogic('QF_BV');
        solver.declareConst('bv1', 'BitVec8');
        solver.declareConst('bv2', 'BitVec8');
        solver.assertFormula('(= bv1 #b00000001)');
        solver.assertFormula('(= bv2 #b00000010)');
        solver.checkSat();
    }, 500);

    // Benchmark: Real arithmetic
    benchmark('Real Arithmetic', () => {
        const solver = new WasmSolver();
        solver.setLogic('QF_LRA');
        solver.declareConst('x', 'Real');
        solver.assertFormula('(> x 0.0)');
        solver.assertFormula('(< x 1.0)');
        solver.checkSat();
    }, 500);

    console.log('='.repeat(60));
    console.log('Benchmarks complete!');
    console.log('='.repeat(60));
}

// Run benchmarks if executed directly
if (require.main === module) {
    runBenchmarks();
}

module.exports = { runBenchmarks, benchmark };
