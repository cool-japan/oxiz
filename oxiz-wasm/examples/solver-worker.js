// Web Worker for OxiZ WASM Solver
// This worker runs the SMT solver in a background thread to keep the UI responsive

importScripts('../pkg/oxiz_wasm.js');

const { WasmSolver, default: init } = wasm_bindgen;

let solver = null;
let initialized = false;

async function initializeSolver() {
    if (!initialized) {
        postMessage({ type: 'log', message: 'Initializing WASM module in worker...' });
        await init('../pkg/oxiz_wasm_bg.wasm');
        solver = new WasmSolver();
        initialized = true;
        postMessage({ type: 'log', message: 'WASM module initialized successfully!' });
    }
}

async function runSolver() {
    try {
        await initializeSolver();

        postMessage({ type: 'status', message: 'Setting up problem...' });
        postMessage({ type: 'log', message: '1. Setting logic to QF_LIA' });
        solver.setLogic('QF_LIA');

        postMessage({ type: 'log', message: '2. Declaring variables: x, y, z' });
        solver.declareConst('x', 'Int');
        solver.declareConst('y', 'Int');
        solver.declareConst('z', 'Int');

        postMessage({ type: 'log', message: '3. Adding constraints:' });
        postMessage({ type: 'log', message: '   - x + y + z = 15' });
        solver.assertFormula('(= (+ (+ x y) z) 15)');

        postMessage({ type: 'log', message: '   - x > 0' });
        solver.assertFormula('(> x 0)');

        postMessage({ type: 'log', message: '   - y > x' });
        solver.assertFormula('(> y x)');

        postMessage({ type: 'log', message: '   - z < y' });
        solver.assertFormula('(< z y)');

        postMessage({ type: 'log', message: '   - x + y = 2 * z' });
        solver.assertFormula('(= (+ x y) (* 2 z))');

        postMessage({ type: 'status', message: 'Solving (this runs in background)...' });
        postMessage({ type: 'log', message: '4. Checking satisfiability...' });

        // Use async version to allow message processing
        const result = await solver.checkSatAsync();

        if (solver.isCancelled()) {
            postMessage({ type: 'cancelled' });
            return;
        }

        postMessage({ type: 'log', message: `Result: ${result}` });

        let model = null;
        if (result === 'sat') {
            postMessage({ type: 'log', message: '5. Extracting model...' });
            model = solver.getModel();
        }

        postMessage({
            type: 'result',
            result: {
                sat: result,
                model: model
            }
        });

    } catch (error) {
        postMessage({
            type: 'error',
            error: error.toString()
        });
    }
}

// Handle messages from main thread
self.onmessage = async function(e) {
    const { command } = e.data;

    switch(command) {
        case 'solve':
            await runSolver();
            break;

        case 'cancel':
            if (solver) {
                solver.cancel();
                postMessage({ type: 'log', message: 'Cancellation requested' });
            }
            break;

        default:
            postMessage({
                type: 'error',
                error: `Unknown command: ${command}`
            });
    }
};
