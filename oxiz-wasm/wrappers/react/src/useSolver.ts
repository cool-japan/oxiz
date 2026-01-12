/**
 * React hook for using OxiZ SMT Solver
 */

import { useEffect, useState, useCallback, useRef } from 'react';

export interface SolverOptions {
  /** SMT-LIB2 logic to use (e.g., "QF_LIA", "QF_UF") */
  logic?: string;
  /** Solver configuration preset */
  preset?: 'default' | 'fast' | 'complete' | 'debug' | 'unsat-core' | 'incremental';
  /** Custom solver options */
  options?: Record<string, string>;
}

export interface SolverHookResult {
  /** The solver instance (null until loaded) */
  solver: any | null;
  /** Whether the solver is loading */
  loading: boolean;
  /** Error during initialization */
  error: Error | null;
  /** Declare a constant */
  declareConst: (name: string, sort: string) => Promise<void>;
  /** Assert a formula */
  assertFormula: (formula: string) => Promise<void>;
  /** Check satisfiability */
  checkSat: () => Promise<'sat' | 'unsat' | 'unknown'>;
  /** Get the model (after SAT) */
  getModel: () => Promise<any>;
  /** Reset the solver */
  reset: () => void;
  /** Push a context level */
  push: () => void;
  /** Pop a context level */
  pop: () => void;
}

/**
 * Hook for using the OxiZ SMT Solver in React components
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { solver, loading, declareConst, assertFormula, checkSat, getModel } = useSolver({
 *     logic: 'QF_LIA',
 *     preset: 'complete'
 *   });
 *
 *   useEffect(() => {
 *     if (!solver) return;
 *
 *     async function solve() {
 *       await declareConst('x', 'Int');
 *       await assertFormula('(> x 0)');
 *       const result = await checkSat();
 *
 *       if (result === 'sat') {
 *         const model = await getModel();
 *         console.log('Model:', model);
 *       }
 *     }
 *
 *     solve();
 *   }, [solver]);
 *
 *   if (loading) return <div>Loading solver...</div>;
 *   return <div>Solver ready!</div>;
 * }
 * ```
 */
export function useSolver(options: SolverOptions = {}): SolverHookResult {
  const [solver, setSolver] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const solverRef = useRef<any | null>(null);

  useEffect(() => {
    let mounted = true;

    async function initSolver() {
      try {
        setLoading(true);
        setError(null);

        // Dynamic import to avoid SSR issues
        const { default: init, WasmSolver } = await import('oxiz-wasm');

        // Initialize WASM module
        await init();

        if (!mounted) return;

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

        solverRef.current = instance;
        setSolver(instance);
        setLoading(false);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err : new Error(String(err)));
        setLoading(false);
      }
    }

    initSolver();

    return () => {
      mounted = false;
    };
  }, [options.logic, options.preset]);

  const declareConst = useCallback(async (name: string, sort: string) => {
    if (!solverRef.current) throw new Error('Solver not initialized');
    return solverRef.current.declareConst(name, sort);
  }, []);

  const assertFormula = useCallback(async (formula: string) => {
    if (!solverRef.current) throw new Error('Solver not initialized');
    return solverRef.current.assertFormula(formula);
  }, []);

  const checkSat = useCallback(async (): Promise<'sat' | 'unsat' | 'unknown'> => {
    if (!solverRef.current) throw new Error('Solver not initialized');
    return solverRef.current.checkSat();
  }, []);

  const getModel = useCallback(async () => {
    if (!solverRef.current) throw new Error('Solver not initialized');
    return solverRef.current.getModel();
  }, []);

  const reset = useCallback(() => {
    if (solverRef.current) {
      solverRef.current.reset();
    }
  }, []);

  const push = useCallback(() => {
    if (solverRef.current) {
      solverRef.current.push();
    }
  }, []);

  const pop = useCallback(() => {
    if (solverRef.current) {
      solverRef.current.pop();
    }
  }, []);

  return {
    solver,
    loading,
    error,
    declareConst,
    assertFormula,
    checkSat,
    getModel,
    reset,
    push,
    pop,
  };
}
