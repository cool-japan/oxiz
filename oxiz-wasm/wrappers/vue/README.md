# @oxiz/vue

Vue 3 wrapper for OxiZ WASM SMT Solver.

## Installation

```bash
npm install @oxiz/vue oxiz-wasm
```

## Usage

### Basic Composable

```vue
<script setup>
import { useSolver } from '@oxiz/vue';
import { onMounted } from 'vue';

const { solver, loading, declareConst, assertFormula, checkSat, getModel } = useSolver({
  logic: 'QF_LIA',
  preset: 'complete'
});

onMounted(async () => {
  if (!solver.value) return;

  await declareConst('x', 'Int');
  await assertFormula('(> x 0)');
  const result = await checkSat();

  if (result === 'sat') {
    const model = await getModel();
    console.log('Model:', model);
  }
});
</script>

<template>
  <div v-if="loading">Loading solver...</div>
  <div v-else>Solver ready!</div>
</template>
```

### Web Worker

```vue
<script setup>
import { useSolverWorker } from '@oxiz/vue';

const { ready, execute, checkSat } = useSolverWorker('/solver-worker.js', 'QF_LIA');

async function solve() {
  if (!ready.value) return;

  await execute(`
    (declare-const x Int)
    (assert (> x 0))
  `);

  const result = await checkSat();
  console.log('Result:', result);
}
</script>

<template>
  <button @click="solve" :disabled="!ready">Solve</button>
</template>
```

### Complete Example

```vue
<script setup>
import { ref } from 'vue';
import { useSolver } from '@oxiz/vue';

const { solver, loading, declareConst, assertFormula, checkSat, getModel } = useSolver({
  logic: 'QF_LIA',
  preset: 'complete'
});

const result = ref('');
const model = ref(null);

async function solve() {
  if (!solver.value) return;

  try {
    await declareConst('x', 'Int');
    await declareConst('y', 'Int');
    await assertFormula('(> x 0)');
    await assertFormula('(< y 10)');
    await assertFormula('(= (+ x y) 15)');

    const satResult = await checkSat();
    result.value = satResult;

    if (satResult === 'sat') {
      model.value = await getModel();
    }
  } catch (err) {
    console.error('Solver error:', err);
  }
}
</script>

<template>
  <div>
    <h1>OxiZ SMT Solver</h1>
    <div v-if="loading">Loading solver...</div>
    <div v-else>
      <button @click="solve">Solve</button>
      <div v-if="result">
        <p>Result: {{ result }}</p>
        <pre v-if="model">{{ JSON.stringify(model, null, 2) }}</pre>
      </div>
    </div>
  </div>
</template>
```

## API

### `useSolver(options)`

Main composable for using the solver.

**Options:**
- `logic?: string` - SMT-LIB2 logic (e.g., "QF_LIA", "QF_UF")
- `preset?: string` - Configuration preset
- `options?: Record<string, string>` - Custom options

**Returns:**
- `solver: Ref<Solver | null>` - Solver instance
- `loading: Ref<boolean>` - Loading state
- `error: Ref<Error | null>` - Error state
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

### `useSolverWorker(workerUrl, logic?)`

Composable for using the solver in a Web Worker.

## License

MIT
