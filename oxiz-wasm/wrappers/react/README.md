# @oxiz/react

React wrapper for OxiZ WASM SMT Solver.

## Installation

```bash
npm install @oxiz/react oxiz-wasm
```

## Usage

### Basic Hook

```tsx
import { useSolver } from '@oxiz/react';

function MyComponent() {
  const { solver, loading, declareConst, assertFormula, checkSat, getModel } = useSolver({
    logic: 'QF_LIA',
    preset: 'complete'
  });

  useEffect(() => {
    if (!solver) return;

    async function solve() {
      await declareConst('x', 'Int');
      await assertFormula('(> x 0)');
      const result = await checkSat();

      if (result === 'sat') {
        const model = await getModel();
        console.log('Model:', model);
      }
    }

    solve();
  }, [solver]);

  if (loading) return <div>Loading solver...</div>;
  return <div>Solver ready!</div>;
}
```

### Context Provider

```tsx
import { SolverProvider, useSolverContext } from '@oxiz/react';

function App() {
  return (
    <SolverProvider options={{ logic: 'QF_LIA' }}>
      <MyComponent />
    </SolverProvider>
  );
}

function MyComponent() {
  const { checkSat, declareConst } = useSolverContext();
  // Use the solver...
}
```

### Web Worker

```tsx
import { useSolverWorker } from '@oxiz/react';

function MyComponent() {
  const { ready, execute, checkSat } = useSolverWorker('/solver-worker.js', 'QF_LIA');

  async function solve() {
    if (!ready) return;

    await execute(`
      (declare-const x Int)
      (assert (> x 0))
    `);

    const result = await checkSat();
    console.log('Result:', result);
  }

  return (
    <button onClick={solve} disabled={!ready}>
      Solve
    </button>
  );
}
```

## API

### `useSolver(options)`

Main hook for using the solver.

**Options:**
- `logic?: string` - SMT-LIB2 logic (e.g., "QF_LIA", "QF_UF")
- `preset?: string` - Configuration preset
- `options?: Record<string, string>` - Custom options

**Returns:**
- `solver` - Solver instance
- `loading` - Loading state
- `error` - Error state
- `declareConst(name, sort)` - Declare a constant
- `assertFormula(formula)` - Assert a formula
- `checkSat()` - Check satisfiability
- `getModel()` - Get the model
- `reset()` - Reset solver
- `push()` - Push context
- `pop()` - Pop context

### `useSolverWorker(workerUrl, logic?)`

Hook for using the solver in a Web Worker.

### `SolverProvider`

Context provider for sharing a solver instance.

### `useSolverContext()`

Hook to access the solver from context.

## License

MIT
