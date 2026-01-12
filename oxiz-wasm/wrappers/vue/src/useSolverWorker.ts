/**
 * Vue composable for using OxiZ SMT Solver in a Web Worker
 */

import { ref, onMounted, onUnmounted, readonly, Ref } from 'vue';

export interface SolverWorkerComposable {
  /** Whether the worker is ready */
  ready: Readonly<Ref<boolean>>;
  /** Loading state */
  loading: Readonly<Ref<boolean>>;
  /** Error state */
  error: Readonly<Ref<Error | null>>;
  /** Execute an SMT-LIB2 script in the worker */
  execute: (script: string) => Promise<string>;
  /** Declare a constant */
  declareConst: (name: string, sort: string) => Promise<void>;
  /** Assert a formula */
  assertFormula: (formula: string) => Promise<void>;
  /** Check satisfiability */
  checkSat: () => Promise<'sat' | 'unsat' | 'unknown'>;
  /** Get the model */
  getModel: () => Promise<any>;
  /** Reset the solver */
  reset: () => Promise<void>;
  /** Terminate the worker */
  terminate: () => void;
}

interface WorkerMessage {
  id: number;
  type: 'execute' | 'declareConst' | 'assertFormula' | 'checkSat' | 'getModel' | 'reset';
  payload?: any;
}

interface WorkerResponse {
  id: number;
  success: boolean;
  result?: any;
  error?: string;
}

/**
 * Composable for using OxiZ in a Web Worker
 *
 * @param workerUrl - URL to the worker script
 * @param logic - SMT-LIB2 logic to use
 *
 * @example
 * ```vue
 * <script setup>
 * import { useSolverWorker } from '@oxiz/vue';
 *
 * const { ready, execute, checkSat } = useSolverWorker('/solver-worker.js', 'QF_LIA');
 *
 * async function solve() {
 *   if (!ready.value) return;
 *
 *   await execute(`
 *     (declare-const x Int)
 *     (assert (> x 0))
 *   `);
 *
 *   const result = await checkSat();
 *   console.log('Result:', result);
 * }
 * </script>
 *
 * <template>
 *   <button @click="solve" :disabled="!ready">Solve</button>
 * </template>
 * ```
 */
export function useSolverWorker(
  workerUrl: string,
  logic?: string
): SolverWorkerComposable {
  const ready = ref(false);
  const loading = ref(true);
  const error = ref<Error | null>(null);
  let worker: Worker | null = null;
  const pendingCalls = new Map<number, {
    resolve: (value: any) => void;
    reject: (reason: any) => void;
  }>();
  let nextId = 0;

  onMounted(() => {
    try {
      worker = new Worker(workerUrl);

      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const { id, success, result, error: workerError } = e.data;
        const pending = pendingCalls.get(id);

        if (pending) {
          pendingCalls.delete(id);
          if (success) {
            pending.resolve(result);
          } else {
            pending.reject(new Error(workerError || 'Worker error'));
          }
        }
      };

      worker.onerror = (err) => {
        error.value = new Error(err.message);
        loading.value = false;
      };

      // Initialize worker
      const initMessage: WorkerMessage = {
        id: nextId++,
        type: 'execute',
        payload: logic ? `(set-logic ${logic})` : '(check-sat)',
      };

      worker.postMessage(initMessage);

      // Wait for initialization
      setTimeout(() => {
        ready.value = true;
        loading.value = false;
      }, 100);
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err));
      loading.value = false;
    }
  });

  onUnmounted(() => {
    if (worker) {
      worker.terminate();
      worker = null;
    }
  });

  function sendMessage<T>(type: WorkerMessage['type'], payload?: any): Promise<T> {
    return new Promise((resolve, reject) => {
      if (!worker) {
        reject(new Error('Worker not initialized'));
        return;
      }

      const id = nextId++;
      pendingCalls.set(id, { resolve, reject });

      const message: WorkerMessage = { id, type, payload };
      worker.postMessage(message);
    });
  }

  async function execute(script: string): Promise<string> {
    return sendMessage('execute', script);
  }

  async function declareConst(name: string, sort: string): Promise<void> {
    return sendMessage('declareConst', { name, sort });
  }

  async function assertFormula(formula: string): Promise<void> {
    return sendMessage('assertFormula', formula);
  }

  async function checkSat(): Promise<'sat' | 'unsat' | 'unknown'> {
    return sendMessage('checkSat');
  }

  async function getModel(): Promise<any> {
    return sendMessage('getModel');
  }

  async function reset(): Promise<void> {
    return sendMessage('reset');
  }

  function terminate(): void {
    if (worker) {
      worker.terminate();
      worker = null;
      ready.value = false;
    }
  }

  return {
    ready: readonly(ready),
    loading: readonly(loading),
    error: readonly(error),
    execute,
    declareConst,
    assertFormula,
    checkSat,
    getModel,
    reset,
    terminate,
  };
}
