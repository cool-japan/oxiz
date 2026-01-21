// TypeScript Example with Full Type Safety
// Compile with: npx tsc wasm_typescript.ts
// Run with: node wasm_typescript.js
//
// This example demonstrates TypeScript usage with proper typing
// and error handling for production applications.

import init, { WasmSolver } from 'oxiz-wasm';

// Type definitions (these would normally come from oxiz-wasm types)
interface OxiZError {
    kind: string;
    message: string;
}

interface ModelValue {
    name: string;
    sort: string;
    value: string | number | boolean | bigint;
}

interface Model {
    [key: string]: ModelValue;
}

type SatResult = 'sat' | 'unsat' | 'unknown';

// Solver wrapper with better TypeScript integration
class SMTSolver {
    private solver: WasmSolver | null = null;
    private initialized = false;

    async init(): Promise<void> {
        if (!this.initialized) {
            await init();
            this.solver = new WasmSolver();
            this.initialized = true;
        }
    }

    private ensureInitialized(): WasmSolver {
        if (!this.solver) {
            throw new Error('Solver not initialized. Call init() first.');
        }
        return this.solver;
    }

    setLogic(logic: string): void {
        this.ensureInitialized().setLogic(logic);
    }

    declareInt(name: string): void {
        this.ensureInitialized().declareConst(name, 'Int');
    }

    declareBool(name: string): void {
        this.ensureInitialized().declareConst(name, 'Bool');
    }

    declareReal(name: string): void {
        this.ensureInitialized().declareConst(name, 'Real');
    }

    declareBitVec(name: string, width: number): void {
        this.ensureInitialized().declareConst(name, `(_ BitVec ${width})`);
    }

    assert(formula: string): void {
        this.ensureInitialized().assertFormula(formula);
    }

    checkSat(): SatResult {
        return this.ensureInitialized().checkSat() as SatResult;
    }

    getModel(): Model {
        return this.ensureInitialized().getModel() as Model;
    }

    push(): void {
        this.ensureInitialized().push();
    }

    pop(): void {
        this.ensureInitialized().pop();
    }

    reset(): void {
        this.ensureInitialized().reset();
    }

    dispose(): void {
        if (this.solver) {
            this.solver.free();
            this.solver = null;
            this.initialized = false;
        }
    }
}

// Example: Sudoku constraint checking
async function sudokuExample(): Promise<void> {
    console.log('=== Sudoku Constraint Example ===\n');

    const solver = new SMTSolver();
    await solver.init();

    try {
        solver.setLogic('QF_LIA');

        // Create a 4x4 Sudoku grid
        const grid: string[][] = [];
        for (let i = 0; i < 4; i++) {
            grid[i] = [];
            for (let j = 0; j < 4; j++) {
                const cellName = `cell_${i}_${j}`;
                grid[i][j] = cellName;
                solver.declareInt(cellName);

                // Each cell is 1-4
                solver.assert(`(>= ${cellName} 1)`);
                solver.assert(`(<= ${cellName} 4)`);
            }
        }

        // Row constraints: all different in each row
        for (let i = 0; i < 4; i++) {
            for (let j1 = 0; j1 < 4; j1++) {
                for (let j2 = j1 + 1; j2 < 4; j2++) {
                    solver.assert(`(distinct ${grid[i][j1]} ${grid[i][j2]})`);
                }
            }
        }

        // Column constraints: all different in each column
        for (let j = 0; j < 4; j++) {
            for (let i1 = 0; i1 < 4; i1++) {
                for (let i2 = i1 + 1; i2 < 4; i2++) {
                    solver.assert(`(distinct ${grid[i1][j]} ${grid[i2][j]})`);
                }
            }
        }

        // 2x2 box constraints
        for (let boxRow = 0; boxRow < 2; boxRow++) {
            for (let boxCol = 0; boxCol < 2; boxCol++) {
                const cells: string[] = [];
                for (let i = 0; i < 2; i++) {
                    for (let j = 0; j < 2; j++) {
                        cells.push(grid[boxRow * 2 + i][boxCol * 2 + j]);
                    }
                }
                // All cells in box must be distinct
                for (let c1 = 0; c1 < cells.length; c1++) {
                    for (let c2 = c1 + 1; c2 < cells.length; c2++) {
                        solver.assert(`(distinct ${cells[c1]} ${cells[c2]})`);
                    }
                }
            }
        }

        // Add some fixed values (partial Sudoku)
        solver.assert('(= cell_0_0 1)');
        solver.assert('(= cell_1_1 3)');
        solver.assert('(= cell_2_2 4)');
        solver.assert('(= cell_3_3 2)');

        console.log('Checking 4x4 Sudoku...');
        const result = solver.checkSat();
        console.log('Result:', result);

        if (result === 'sat') {
            const model = solver.getModel();
            console.log('\nSolution:');
            for (let i = 0; i < 4; i++) {
                const row = [];
                for (let j = 0; j < 4; j++) {
                    row.push(model[grid[i][j]].value);
                }
                console.log(' ', row.join(' '));
            }
        }
    } finally {
        solver.dispose();
    }
}

// Example: Safe arithmetic
async function safeArithmeticExample(): Promise<void> {
    console.log('\n=== Safe Arithmetic Example ===\n');

    const solver = new SMTSolver();
    await solver.init();

    try {
        solver.setLogic('QF_LIA');

        // Define a function that safely divides
        // Find x and y where x / y = 5 and y > 0
        solver.declareInt('x');
        solver.declareInt('y');
        solver.declareInt('quotient');

        // y must be positive (avoid division by zero)
        solver.assert('(> y 0)');

        // quotient = x / y (integer division)
        solver.assert('(= quotient 5)');
        solver.assert('(= (* quotient y) x)');

        // Additional constraint: x should be reasonable
        solver.assert('(> x 0)');
        solver.assert('(< x 1000)');

        console.log('Finding x, y where x / y = 5...');
        const result = solver.checkSat();

        if (result === 'sat') {
            const model = solver.getModel();
            const x = Number(model.x.value);
            const y = Number(model.y.value);
            console.log(`Found: x = ${x}, y = ${y}`);
            console.log(`Verification: ${x} / ${y} = ${Math.floor(x / y)}`);
        }
    } finally {
        solver.dispose();
    }
}

// Main entry point
async function main(): Promise<void> {
    try {
        await sudokuExample();
        await safeArithmeticExample();
        console.log('\nAll examples completed successfully!');
    } catch (error) {
        const oxizError = error as OxiZError;
        if (oxizError.kind && oxizError.message) {
            console.error(`OxiZ Error [${oxizError.kind}]: ${oxizError.message}`);
        } else {
            console.error('Error:', error);
        }
        process.exit(1);
    }
}

main();
