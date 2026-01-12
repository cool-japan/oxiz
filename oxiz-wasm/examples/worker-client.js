/**
 * Client wrapper for OxiZ WASM Web Worker
 *
 * Provides a Promise-based API for interacting with the solver worker
 */

export class SolverWorkerClient {
    constructor(workerPath = './solver-worker.js') {
        this.worker = new Worker(workerPath, { type: 'module' });
        this.messageId = 0;
        this.pending = new Map();
        this.ready = false;

        this.worker.onmessage = (e) => {
            const { id, success, data, error, ready } = e.data;

            if (ready) {
                this.ready = true;
                return;
            }

            const resolver = this.pending.get(id);
            if (!resolver) return;

            this.pending.delete(id);

            if (success) {
                resolver.resolve(data);
            } else {
                resolver.reject(new Error(error.message));
            }
        };

        this.worker.onerror = (error) => {
            console.error('Worker error:', error);
        };
    }

    /**
     * Send a command to the worker and wait for response
     */
    async sendCommand(command, args = {}) {
        if (!this.ready) {
            await this.waitForReady();
        }

        const id = this.messageId++;

        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });

            this.worker.postMessage({
                id,
                command,
                args,
            });

            // Timeout after 30 seconds
            setTimeout(() => {
                if (this.pending.has(id)) {
                    this.pending.delete(id);
                    reject(new Error('Operation timed out'));
                }
            }, 30000);
        });
    }

    /**
     * Wait for worker to be ready
     */
    async waitForReady() {
        return new Promise((resolve) => {
            const check = () => {
                if (this.ready) {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    }

    /**
     * Set the logic for the solver
     */
    async setLogic(logic) {
        return this.sendCommand('setLogic', { logic });
    }

    /**
     * Declare a constant
     */
    async declareConst(name, sort) {
        return this.sendCommand('declareConst', { name, sort });
    }

    /**
     * Assert a formula
     */
    async assertFormula(formula) {
        return this.sendCommand('assertFormula', { formula });
    }

    /**
     * Check satisfiability
     */
    async checkSat() {
        const { result } = await this.sendCommand('checkSat');
        return result;
    }

    /**
     * Check satisfiability (async version)
     */
    async checkSatAsync() {
        const { result } = await this.sendCommand('checkSatAsync');
        return result;
    }

    /**
     * Get the model
     */
    async getModel() {
        const { model } = await this.sendCommand('getModel');
        return model;
    }

    /**
     * Get the model as a string
     */
    async getModelString() {
        const { modelString } = await this.sendCommand('getModelString');
        return modelString;
    }

    /**
     * Get assertions
     */
    async getAssertions() {
        const { assertions } = await this.sendCommand('getAssertions');
        return assertions;
    }

    /**
     * Push a context level
     */
    async push() {
        return this.sendCommand('push');
    }

    /**
     * Pop a context level
     */
    async pop() {
        return this.sendCommand('pop');
    }

    /**
     * Reset the solver
     */
    async reset() {
        return this.sendCommand('reset');
    }

    /**
     * Reset assertions only
     */
    async resetAssertions() {
        return this.sendCommand('resetAssertions');
    }

    /**
     * Simplify an expression
     */
    async simplify(expr) {
        const { simplified } = await this.sendCommand('simplify', { expr });
        return simplified;
    }

    /**
     * Execute an SMT-LIB2 script
     */
    async execute(script) {
        const { output } = await this.sendCommand('execute', { script });
        return output;
    }

    /**
     * Set an option
     */
    async setOption(key, value) {
        return this.sendCommand('setOption', { key, value });
    }

    /**
     * Get an option
     */
    async getOption(key) {
        const { value } = await this.sendCommand('getOption', { key });
        return value;
    }

    /**
     * Terminate the worker
     */
    terminate() {
        this.worker.terminate();
    }
}
