/**
 * Svelte composable-style function for using OxiZ SMT Solver
 */

import { onMount, onDestroy } from 'svelte';
import { writable, type Writable } from 'svelte/store';
import type { SolverOptions } from './types';

export interface UseSolverResult {
  /** The solver instance (null until loaded) */
  solver: Writable<any | null>;
  /** Whether the solver is loading */
  loading: Writable<boolean>;
  /** Error during initialization */
  error: Writable<Error | null>;
  /** Declare a constant */
  declareConst: (name: string, sort: string) => Promise<void>;
  /** Assert a formula */
  assertFormula: (formula: string) => Promise<void>;
  /** Check satisfiability */
  checkSat: () => Promise<'sat' | 'unsat' | 'unknown'>;
  /** Get the model (after SAT) */
  getModel: () => Promise<any>;
  /** Get model as string */
  getModelString: () => Promise<string>;
  /** Reset the solver */
  reset: () => void;
  /** Push a context level */
  push: () => void;
  /** Pop a context level */
  pop: () => void;
  /** Simplify an expression */
  simplify: (expr: string) => Promise<string>;
  /** Execute an SMT-LIB2 script */
  execute: (script: string) => Promise<string>;
}

/**
 * Function for using the OxiZ SMT Solver in Svelte components
 *
 * @example
 * ```svelte
 * <script>
 *   import { useSolver } from '@oxiz/svelte';
 *   import { onMount } from 'svelte';
 *
 *   const { solver, loading, declareConst, assertFormula, checkSat, getModel } = useSolver({
 *     logic: 'QF_LIA',
 *     preset: 'complete'
 *   });
 *
 *   onMount(async () => {
 *     await declareConst('x', 'Int');
 *     await assertFormula('(> x 0)');
 *     const result = await checkSat();
 *
 *     if (result === 'sat') {
 *       const model = await getModel();
 *       console.log('Model:', model);
 *     }
 *   });
 * </script>
 *
 * {#if $loading}
 *   <div>Loading solver...</div>
 * {:else}
 *   <div>Solver ready!</div>
 * {/if}
 * ```
 */
export function useSolver(options: SolverOptions = {}): UseSolverResult {
  const solver = writable<any | null>(null);
  const loading = writable(true);
  const error = writable<Error | null>(null);

  let solverInstance: any | null = null;

  onMount(async () => {
    try {
      loading.set(true);
      error.set(null);

      // Dynamic import to avoid SSR issues
      const { default: init, WasmSolver } = await import('oxiz-wasm');

      // Initialize WASM module
      await init();

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
      solver.set(instance);
      loading.set(false);
    } catch (err) {
      error.set(err instanceof Error ? err : new Error(String(err)));
      loading.set(false);
    }
  });

  onDestroy(() => {
    solverInstance = null;
  });

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
    solver,
    loading,
    error,
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
