/**
 * Vue composable for using OxiZ SMT Solver
 */

import { ref, onMounted, onUnmounted, readonly, Ref } from 'vue';

export interface SolverOptions {
  /** SMT-LIB2 logic to use (e.g., "QF_LIA", "QF_UF") */
  logic?: string;
  /** Solver configuration preset */
  preset?: 'default' | 'fast' | 'complete' | 'debug' | 'unsat-core' | 'incremental';
  /** Custom solver options */
  options?: Record<string, string>;
}

export interface SolverComposable {
  /** The solver instance (null until loaded) */
  solver: Readonly<Ref<any | null>>;
  /** Whether the solver is loading */
  loading: Readonly<Ref<boolean>>;
  /** Error during initialization */
  error: Readonly<Ref<Error | null>>;
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
 * Composable for using the OxiZ SMT Solver in Vue components
 *
 * @example
 * ```vue
 * <script setup>
 * import { useSolver } from '@oxiz/vue';
 * import { onMounted } from 'vue';
 *
 * const { solver, loading, declareConst, assertFormula, checkSat, getModel } = useSolver({
 *   logic: 'QF_LIA',
 *   preset: 'complete'
 * });
 *
 * onMounted(async () => {
 *   if (!solver.value) return;
 *
 *   await declareConst('x', 'Int');
 *   await assertFormula('(> x 0)');
 *   const result = await checkSat();
 *
 *   if (result === 'sat') {
 *     const model = await getModel();
 *     console.log('Model:', model);
 *   }
 * });
 * </script>
 *
 * <template>
 *   <div v-if="loading">Loading solver...</div>
 *   <div v-else>Solver ready!</div>
 * </template>
 * ```
 */
export function useSolver(options: SolverOptions = {}): SolverComposable {
  const solver = ref<any | null>(null);
  const loading = ref(true);
  const error = ref<Error | null>(null);

  async function initSolver() {
    try {
      loading.value = true;
      error.value = null;

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

      solver.value = instance;
      loading.value = false;
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err));
      loading.value = false;
    }
  }

  onMounted(() => {
    initSolver();
  });

  onUnmounted(() => {
    if (solver.value) {
      solver.value = null;
    }
  });

  async function declareConst(name: string, sort: string): Promise<void> {
    if (!solver.value) throw new Error('Solver not initialized');
    return solver.value.declareConst(name, sort);
  }

  async function assertFormula(formula: string): Promise<void> {
    if (!solver.value) throw new Error('Solver not initialized');
    return solver.value.assertFormula(formula);
  }

  async function checkSat(): Promise<'sat' | 'unsat' | 'unknown'> {
    if (!solver.value) throw new Error('Solver not initialized');
    return solver.value.checkSat();
  }

  async function getModel(): Promise<any> {
    if (!solver.value) throw new Error('Solver not initialized');
    return solver.value.getModel();
  }

  async function getModelString(): Promise<string> {
    if (!solver.value) throw new Error('Solver not initialized');
    return solver.value.getModelString();
  }

  function reset(): void {
    if (solver.value) {
      solver.value.reset();
    }
  }

  function push(): void {
    if (solver.value) {
      solver.value.push();
    }
  }

  function pop(): void {
    if (solver.value) {
      solver.value.pop();
    }
  }

  async function simplify(expr: string): Promise<string> {
    if (!solver.value) throw new Error('Solver not initialized');
    return solver.value.simplify(expr);
  }

  async function execute(script: string): Promise<string> {
    if (!solver.value) throw new Error('Solver not initialized');
    const result = await solver.value.execute(script);
    return result;
  }

  return {
    solver: readonly(solver),
    loading: readonly(loading),
    error: readonly(error),
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
