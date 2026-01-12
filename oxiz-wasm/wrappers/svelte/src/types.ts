/**
 * Type definitions for @oxiz/svelte
 */

import type { Readable } from 'svelte/store';

export interface SolverOptions {
  /** SMT-LIB2 logic to use (e.g., "QF_LIA", "QF_UF") */
  logic?: string;
  /** Solver configuration preset */
  preset?: 'default' | 'fast' | 'complete' | 'debug' | 'unsat-core' | 'incremental';
  /** Custom solver options */
  options?: Record<string, string>;
}

export interface SolverState {
  /** The solver instance (null until loaded) */
  solver: any | null;
  /** Whether the solver is loading */
  loading: boolean;
  /** Error during initialization */
  error: Error | null;
}

export interface SolverStore extends Readable<SolverState> {
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
