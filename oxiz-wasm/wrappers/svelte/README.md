# @oxiz/svelte

Svelte wrapper for OxiZ WASM SMT Solver.

## Installation

```bash
npm install @oxiz/svelte oxiz-wasm
```

## Usage

### Using the composable

```svelte
<script>
  import { useSolver } from '@oxiz/svelte';
  import { onMount } from 'svelte';

  const { solver, loading, declareConst, assertFormula, checkSat, getModel } = useSolver({
    logic: 'QF_LIA',
    preset: 'complete'
  });

  onMount(async () => {
    await declareConst('x', 'Int');
    await assertFormula('(> x 0)');
    const result = await checkSat();

    if (result === 'sat') {
      const model = await getModel();
      console.log('Model:', model);
    }
  });
</script>

{#if $loading}
  <div>Loading solver...</div>
{:else}
  <div>Solver ready!</div>
{/if}
```

### Using the store

```svelte
<script>
  import { createSolverStore } from '@oxiz/svelte';

  const solver = createSolverStore({
    logic: 'QF_LIA',
    preset: 'complete'
  });

  let result = '';
  let model = null;

  async function solve() {
    await solver.declareConst('x', 'Int');
    await solver.declareConst('y', 'Int');
    await solver.assertFormula('(> x 0)');
    await solver.assertFormula('(< y 10)');
    await solver.assertFormula('(= (+ x y) 15)');

    result = await solver.checkSat();

    if (result === 'sat') {
      model = await solver.getModel();
    }
  }
</script>

{#if $solver.loading}
  <div>Loading solver...</div>
{:else if $solver.error}
  <div>Error: {$solver.error.message}</div>
{:else}
  <button on:click={solve}>Solve</button>
  {#if result}
    <p>Result: {result}</p>
    {#if model}
      <pre>{JSON.stringify(model, null, 2)}</pre>
    {/if}
  {/if}
{/if}
```

### Complete Example

```svelte
<script lang="ts">
  import { useSolver } from '@oxiz/svelte';

  const {
    solver,
    loading,
    error,
    declareConst,
    assertFormula,
    checkSat,
    getModel
  } = useSolver({
    logic: 'QF_LIA',
    preset: 'complete'
  });

  let result: string = '';
  let model: any = null;
  let errorMsg: string = '';

  async function solve() {
    try {
      errorMsg = '';
      await declareConst('x', 'Int');
      await declareConst('y', 'Int');
      await assertFormula('(> x 0)');
      await assertFormula('(< y 10)');
      await assertFormula('(= (+ x y) 15)');

      result = await checkSat();

      if (result === 'sat') {
        model = await getModel();
      }
    } catch (err) {
      errorMsg = err instanceof Error ? err.message : String(err);
    }
  }
</script>

<main>
  <h1>OxiZ SMT Solver</h1>

  {#if $loading}
    <p>Loading solver...</p>
  {:else if $error}
    <p class="error">Error: {$error.message}</p>
  {:else}
    <button on:click={solve}>Solve</button>

    {#if errorMsg}
      <p class="error">{errorMsg}</p>
    {/if}

    {#if result}
      <h2>Result: {result}</h2>
      {#if model}
        <pre>{JSON.stringify(model, null, 2)}</pre>
      {/if}
    {/if}
  {/if}
</main>

<style>
  .error {
    color: red;
  }
  pre {
    background: #f5f5f5;
    padding: 1rem;
    border-radius: 4px;
  }
</style>
```

## API

### `useSolver(options)`

Composable for using the solver in Svelte components.

**Options:**
- `logic?: string` - SMT-LIB2 logic (e.g., "QF_LIA", "QF_UF")
- `preset?: string` - Configuration preset
- `options?: Record<string, string>` - Custom options

**Returns:**
- `solver: Writable<Solver | null>` - Solver instance store
- `loading: Writable<boolean>` - Loading state store
- `error: Writable<Error | null>` - Error state store
- `declareConst(name, sort)` - Declare a constant
- `assertFormula(formula)` - Assert a formula
- `checkSat()` - Check satisfiability
- `getModel()` - Get the model
- `getModelString()` - Get model as string
- `reset()` - Reset solver
- `push()` - Push context
- `pop()` - Pop context
- `simplify(expr)` - Simplify expression
- `execute(script)` - Execute SMT-LIB2 script

### `createSolverStore(options)`

Create a Svelte store for the solver.

## License

Apache-2.0
