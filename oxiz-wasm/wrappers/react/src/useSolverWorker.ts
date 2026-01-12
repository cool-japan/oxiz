/**
 * React hook for using OxiZ SMT Solver in a Web Worker
 *
 * This hook runs the solver in a separate thread to avoid blocking the UI.
 */

import { useEffect, useState, useCallback, useRef } from 'react';

export interface SolverWorkerResult {
  /** Whether the worker is ready */
  ready: boolean;
  /** Loading state */
  loading: boolean;
  /** Error state */
  error: Error | null;
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
 * Hook for using OxiZ in a Web Worker
 *
 * @param workerUrl - URL to the worker script
 * @param logic - SMT-LIB2 logic to use
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { ready, execute, checkSat } = useSolverWorker('/solver-worker.js', 'QF_LIA');
 *
 *   async function solve() {
 *     if (!ready) return;
 *
 *     await execute(`
 *       (declare-const x Int)
 *       (assert (> x 0))
 *     `);
 *
 *     const result = await checkSat();
 *     console.log('Result:', result);
 *   }
 *
 *   return (
 *     <button onClick={solve} disabled={!ready}>
 *       Solve
 *     </button>
 *   );
 * }
 * ```
 */
export function useSolverWorker(
  workerUrl: string,
  logic?: string
): SolverWorkerResult {
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const pendingCallsRef = useRef<Map<number, {
    resolve: (value: any) => void;
    reject: (reason: any) => void;
  }>>(new Map());
  const nextIdRef = useRef(0);

  useEffect(() => {
    let mounted = true;

    try {
      const worker = new Worker(workerUrl);
      workerRef.current = worker;

      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const { id, success, result, error } = e.data;
        const pending = pendingCallsRef.current.get(id);

        if (pending) {
          pendingCallsRef.current.delete(id);
          if (success) {
            pending.resolve(result);
          } else {
            pending.reject(new Error(error || 'Worker error'));
          }
        }
      };

      worker.onerror = (err) => {
        if (mounted) {
          setError(new Error(err.message));
          setLoading(false);
        }
      };

      // Initialize worker
      const initMessage: WorkerMessage = {
        id: nextIdRef.current++,
        type: 'execute',
        payload: logic ? `(set-logic ${logic})` : '(check-sat)',
      };

      worker.postMessage(initMessage);

      // Wait for initialization
      setTimeout(() => {
        if (mounted) {
          setReady(true);
          setLoading(false);
        }
      }, 100);
    } catch (err) {
      if (mounted) {
        setError(err instanceof Error ? err : new Error(String(err)));
        setLoading(false);
      }
    }

    return () => {
      mounted = false;
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [workerUrl, logic]);

  const sendMessage = useCallback(<T,>(type: WorkerMessage['type'], payload?: any): Promise<T> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error('Worker not initialized'));
        return;
      }

      const id = nextIdRef.current++;
      pendingCallsRef.current.set(id, { resolve, reject });

      const message: WorkerMessage = { id, type, payload };
      workerRef.current.postMessage(message);
    });
  }, []);

  const execute = useCallback((script: string): Promise<string> => {
    return sendMessage('execute', script);
  }, [sendMessage]);

  const declareConst = useCallback((name: string, sort: string): Promise<void> => {
    return sendMessage('declareConst', { name, sort });
  }, [sendMessage]);

  const assertFormula = useCallback((formula: string): Promise<void> => {
    return sendMessage('assertFormula', formula);
  }, [sendMessage]);

  const checkSat = useCallback((): Promise<'sat' | 'unsat' | 'unknown'> => {
    return sendMessage('checkSat');
  }, [sendMessage]);

  const getModel = useCallback((): Promise<any> => {
    return sendMessage('getModel');
  }, [sendMessage]);

  const reset = useCallback((): Promise<void> => {
    return sendMessage('reset');
  }, [sendMessage]);

  const terminate = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
      setReady(false);
    }
  }, []);

  return {
    ready,
    loading,
    error,
    execute,
    declareConst,
    assertFormula,
    checkSat,
    getModel,
    reset,
    terminate,
  };
}
