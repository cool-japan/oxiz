/**
 * Svelte store for OxiZ SMT Solver
 */

import { writable } from 'svelte/store';
import type { SolverOptions, SolverStore, SolverState } from './types';

/**
 * Create a Svelte store for the OxiZ SMT Solver
 *
 * @example
 * ```svelte
 * <script>
 *   import { createSolverStore } from '@oxiz/svelte';
 *
 *   const solver = createSolverStore({
 *     logic: 'QF_LIA',
 *     preset: 'complete'
 *   });
 *
 *   async function solve() {
 *     await solver.declareConst('x', 'Int');
 *     await solver.assertFormula('(> x 0)');
 *     const result = await solver.checkSat();
 *
 *     if (result === 'sat') {
 *       const model = await solver.getModel();
 *       console.log('Model:', model);
 *     }
 *   }
 * </script>
 *
 * {#if $solver.loading}
 *   <div>Loading solver...</div>
 * {:else if $solver.error}
 *   <div>Error: {$solver.error.message}</div>
 * {:else}
 *   <button on:click={solve}>Solve</button>
 * {/if}
 * ```
 */
export function createSolverStore(options: SolverOptions = {}): SolverStore {
  const { subscribe, set, update } = writable<SolverState>({
    solver: null,
    loading: true,
    error: null,
  });

  let solverInstance: any | null = null;

  // Initialize solver
  async function init() {
    try {
      update(state => ({ ...state, loading: true, error: null }));

      // Dynamic import to avoid SSR issues
      const { default: initWasm, WasmSolver } = await import('oxiz-wasm');

      // Initialize WASM module
      await initWasm();

      // Create solver instance
      const instance = new WasmSolver();

      // Apply configuration
      if (options.logic) {
        instance.setLogic(options.logic);
      }

      if (options.preset) {
        instance.applyPreset(options.preset);
      }

      if (options.options) {
        for (const [key, value] of Object.entries(options.options)) {
          instance.setOption(key, value);
        }
      }

      solverInstance = instance;
      set({ solver: instance, loading: false, error: null });
    } catch (err) {
      set({
        solver: null,
        loading: false,
        error: err instanceof Error ? err : new Error(String(err)),
      });
    }
  }

  // Start initialization
  init();

  async function declareConst(name: string, sort: string): Promise<void> {
    if (!solverInstance) throw new Error('Solver not initialized');
    return solverInstance.declareConst(name, sort);
  }

  async function assertFormula(formula: string): Promise<void> {
    if (!solverInstance) throw new Error('Solver not initialized');
    return solverInstance.assertFormula(formula);
  }

  async function checkSat(): Promise<'sat' | 'unsat' | 'unknown'> {
    if (!solverInstance) throw new Error('Solver not initialized');
    return solverInstance.checkSat();
  }

  async function getModel(): Promise<any> {
    if (!solverInstance) throw new Error('Solver not initialized');
    return solverInstance.getModel();
  }

  async function getModelString(): Promise<string> {
    if (!solverInstance) throw new Error('Solver not initialized');
    return solverInstance.getModelString();
  }

  function reset(): void {
    if (solverInstance) {
      solverInstance.reset();
    }
  }

  function push(): void {
    if (solverInstance) {
      solverInstance.push();
    }
  }

  function pop(): void {
    if (solverInstance) {
      solverInstance.pop();
    }
  }

  async function simplify(expr: string): Promise<string> {
    if (!solverInstance) throw new Error('Solver not initialized');
    return solverInstance.simplify(expr);
  }

  async function execute(script: string): Promise<string> {
    if (!solverInstance) throw new Error('Solver not initialized');
    const result = await solverInstance.execute(script);
    return result;
  }

  return {
    subscribe,
    declareConst,
    assertFormula,
    checkSat,
    getModel,
    getModelString,
    reset,
    push,
    pop,
    simplify,
    execute,
  };
}
