/**
 * React wrapper and hooks for OxiZ WASM
 *
 * Provides React-friendly hooks and components for SMT solving
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import init, { WasmSolver, version } from '../pkg/oxiz_wasm';

/**
 * Hook to initialize and manage the WASM module
 */
export function useWasmInit() {
    const [initialized, setInitialized] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    useEffect(() => {
        init()
            .then(() => setInitialized(true))
            .catch((err) => setError(err));
    }, []);

    return { initialized, error };
}

/**
 * Hook to create and manage a solver instance
 */
export function useSolver() {
    const solverRef = useRef<WasmSolver | null>(null);
    const [ready, setReady] = useState(false);

    useEffect(() => {
        init().then(() => {
            solverRef.current = new WasmSolver();
            setReady(true);
        });

        return () => {
            if (solverRef.current) {
                // Cleanup if needed
                solverRef.current = null;
            }
        };
    }, []);

    const declareConst = useCallback((name: string, sort: string) => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.declareConst(name, sort);
    }, []);

    const assertFormula = useCallback((formula: string) => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.assertFormula(formula);
    }, []);

    const checkSat = useCallback(() => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.checkSat();
    }, []);

    const checkSatAsync = useCallback(async () => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return await solverRef.current.checkSatAsync();
    }, []);

    const getModel = useCallback(() => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.getModel();
    }, []);

    const getModelString = useCallback(() => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.getModelString();
    }, []);

    const reset = useCallback(() => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.reset();
    }, []);

    const push = useCallback(() => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.push();
    }, []);

    const pop = useCallback(() => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.pop();
    }, []);

    const setLogic = useCallback((logic: string) => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.setLogic(logic);
    }, []);

    const simplify = useCallback((expr: string) => {
        if (!solverRef.current) throw new Error('Solver not initialized');
        return solverRef.current.simplify(expr);
    }, []);

    return {
        ready,
        declareConst,
        assertFormula,
        checkSat,
        checkSatAsync,
        getModel,
        getModelString,
        reset,
        push,
        pop,
        setLogic,
        simplify,
    };
}

/**
 * Hook for checking satisfiability with React state
 */
export function useSatCheck() {
    const solver = useSolver();
    const [result, setResult] = useState<string | null>(null);
    const [model, setModel] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const check = useCallback(async () => {
        if (!solver.ready) {
            setError(new Error('Solver not ready'));
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const res = await solver.checkSatAsync();
            setResult(res);

            if (res === 'sat') {
                const m = solver.getModel();
                setModel(m);
            } else {
                setModel(null);
            }
        } catch (err) {
            setError(err instanceof Error ? err : new Error(String(err)));
        } finally {
            setLoading(false);
        }
    }, [solver]);

    return {
        result,
        model,
        loading,
        error,
        check,
        solver,
    };
}

/**
 * Example: SMT Solver Component
 */
export function SmtSolverDemo() {
    const { result, model, loading, error, check, solver } = useSatCheck();
    const [variables, setVariables] = useState<Array<{ name: string; sort: string }>>([]);
    const [formulas, setFormulas] = useState<string[]>([]);
    const [newVarName, setNewVarName] = useState('');
    const [newVarSort, setNewVarSort] = useState('Bool');
    const [newFormula, setNewFormula] = useState('');

    const addVariable = () => {
        if (!newVarName || !solver.ready) return;

        try {
            solver.declareConst(newVarName, newVarSort);
            setVariables([...variables, { name: newVarName, sort: newVarSort }]);
            setNewVarName('');
        } catch (err) {
            alert(`Error: ${err}`);
        }
    };

    const addFormula = () => {
        if (!newFormula || !solver.ready) return;

        try {
            solver.assertFormula(newFormula);
            setFormulas([...formulas, newFormula]);
            setNewFormula('');
        } catch (err) {
            alert(`Error: ${err}`);
        }
    };

    const resetSolver = () => {
        if (solver.ready) {
            solver.reset();
            setVariables([]);
            setFormulas([]);
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
            <h1>OxiZ SMT Solver (React)</h1>
            <p>Version: {solver.ready ? version() : 'Loading...'}</p>

            <div style={{ marginBottom: '20px' }}>
                <h2>Declare Variables</h2>
                <input
                    type="text"
                    placeholder="Variable name"
                    value={newVarName}
                    onChange={(e) => setNewVarName(e.target.value)}
                    disabled={!solver.ready}
                />
                <select
                    value={newVarSort}
                    onChange={(e) => setNewVarSort(e.target.value)}
                    disabled={!solver.ready}
                >
                    <option value="Bool">Bool</option>
                    <option value="Int">Int</option>
                    <option value="Real">Real</option>
                    <option value="BitVec8">BitVec8</option>
                    <option value="BitVec32">BitVec32</option>
                </select>
                <button onClick={addVariable} disabled={!solver.ready || !newVarName}>
                    Add Variable
                </button>

                <ul>
                    {variables.map((v, i) => (
                        <li key={i}>
                            {v.name}: {v.sort}
                        </li>
                    ))}
                </ul>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <h2>Assert Formulas</h2>
                <input
                    type="text"
                    placeholder="SMT-LIB2 formula"
                    value={newFormula}
                    onChange={(e) => setNewFormula(e.target.value)}
                    style={{ width: '400px' }}
                    disabled={!solver.ready}
                />
                <button onClick={addFormula} disabled={!solver.ready || !newFormula}>
                    Add Formula
                </button>

                <ul>
                    {formulas.map((f, i) => (
                        <li key={i}>{f}</li>
                    ))}
                </ul>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <button onClick={check} disabled={loading || !solver.ready}>
                    {loading ? 'Checking...' : 'Check Satisfiability'}
                </button>
                <button onClick={resetSolver} disabled={!solver.ready}>
                    Reset
                </button>
            </div>

            {error && (
                <div style={{ color: 'red', marginBottom: '20px' }}>
                    Error: {error.message}
                </div>
            )}

            {result && (
                <div style={{ marginBottom: '20px' }}>
                    <h3>Result: {result}</h3>
                    {model && (
                        <div>
                            <h4>Model:</h4>
                            <pre>{JSON.stringify(model, null, 2)}</pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

/**
 * Example: Simple boolean formula checker
 */
export function BooleanFormulaChecker() {
    const { result, loading, check, solver } = useSatCheck();
    const [p, setP] = useState(true);
    const [q, setQ] = useState(true);

    useEffect(() => {
        if (solver.ready) {
            solver.reset();
            solver.declareConst('p', 'Bool');
            solver.declareConst('q', 'Bool');
        }
    }, [solver.ready]);

    const checkFormula = (formula: string) => {
        if (!solver.ready) return;

        solver.reset();
        solver.declareConst('p', 'Bool');
        solver.declareConst('q', 'Bool');

        if (p) solver.assertFormula('p');
        if (q) solver.assertFormula('q');

        solver.assertFormula(formula);
        check();
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
            <h2>Boolean Formula Checker</h2>

            <div>
                <label>
                    <input
                        type="checkbox"
                        checked={p}
                        onChange={(e) => setP(e.target.checked)}
                    />
                    p
                </label>
                <label style={{ marginLeft: '10px' }}>
                    <input
                        type="checkbox"
                        checked={q}
                        onChange={(e) => setQ(e.target.checked)}
                    />
                    q
                </label>
            </div>

            <div style={{ marginTop: '20px' }}>
                <button onClick={() => checkFormula('(and p q)')} disabled={loading}>
                    Check (and p q)
                </button>
                <button onClick={() => checkFormula('(or p q)')} disabled={loading}>
                    Check (or p q)
                </button>
                <button onClick={() => checkFormula('(not (and p q))')} disabled={loading}>
                    Check (not (and p q))
                </button>
            </div>

            {result && (
                <div style={{ marginTop: '20px' }}>
                    <strong>Result: {result}</strong>
                </div>
            )}
        </div>
    );
}

export default {
    useWasmInit,
    useSolver,
    useSatCheck,
    SmtSolverDemo,
    BooleanFormulaChecker,
};
