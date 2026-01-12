/**
 * React Context for sharing a solver instance across components
 */

import React, { createContext, useContext, ReactNode } from 'react';
import { useSolver, SolverOptions, SolverHookResult } from './useSolver';

export interface SolverContextValue extends SolverHookResult {
}

const SolverContext = createContext<SolverContextValue | null>(null);

export interface SolverProviderProps {
  children: ReactNode;
  options?: SolverOptions;
}

/**
 * Provider component for sharing a solver instance
 *
 * @example
 * ```tsx
 * function App() {
 *   return (
 *     <SolverProvider options={{ logic: 'QF_LIA', preset: 'complete' }}>
 *       <MyComponent />
 *     </SolverProvider>
 *   );
 * }
 *
 * function MyComponent() {
 *   const { solver, declareConst, checkSat } = useSolverContext();
 *
 *   // Use the solver...
 * }
 * ```
 */
export function SolverProvider({ children, options }: SolverProviderProps) {
  const solverHook = useSolver(options);

  return (
    <SolverContext.Provider value={solverHook}>
      {children}
    </SolverContext.Provider>
  );
}

/**
 * Hook to access the solver context
 *
 * @throws Error if used outside of SolverProvider
 */
export function useSolverContext(): SolverContextValue {
  const context = useContext(SolverContext);
  if (!context) {
    throw new Error('useSolverContext must be used within a SolverProvider');
  }
  return context;
}
