# OxiZ Python Bindings — z3-python Parity Matrix

Status as of version 0.2.1.

Ground truth: `oxiz-py/src/lib.rs` (PyO3 module registration) and `oxiz-py/src/` source files.

| z3 method / feature | oxiz wrapper | status |
|---|---|---|
| `Bool(name)` / `ctx.bool_const(name)` | `Context.bool_const(name)` | ✅ supported |
| `Int(name)` / `ctx.int_const(name)` | `Context.int_const(name)` | ✅ supported |
| `Real(name)` / `ctx.real_const(name)` | `Context.real_const(name)` | ✅ supported |
| `BitVec(name, width)` / `ctx.bv_const(name, width)` | `Context.bv_const(name, width)` | ✅ supported |
| `Array(name, dom, rng)` | `TermManager.mk_select` / `mk_store` (no sort constructor) | ⚠️ partial |
| `FP(name, sort)` / `FPSort(eb, sb)` | not exported | ❌ missing |
| `StringVal(s)` / string sort | not exported | ❌ missing |
| `ForAll(vars, body)` | not exported | ❌ missing |
| `Exists(vars, body)` | not exported | ❌ missing |
| `Solver.check()` | `Solver.check(tm)` / `Solver.check_sat(tm)` | ✅ supported |
| `Solver.model()` | `Solver.model()` (typed) and `Solver.get_model(tm)` (string) | ✅ supported |
| `Solver.unsat_core()` | `Solver.unsat_core()` / `Solver.get_unsat_core()` | ✅ supported |
| `Optimizer.minimize(obj)` | `Optimizer.minimize(obj)` | ✅ supported |
| `Solver.push()` / `Solver.pop()` | `Solver.push()` / `Solver.pop(n=1)` | ✅ supported |
| `set_timeout(ms)` | `Solver.set_timeout(milliseconds)` | ✅ supported |
| `And(*args)` / `Or(*args)` / `Not(x)` | `oxiz.And` / `oxiz.Or` / `oxiz.Not` | ✅ supported |
| `Implies(a, b)` | `oxiz.Implies(a, b)` | ✅ supported |
| `If(cond, t, e)` | `oxiz.If(cond, t, e)` | ✅ supported |
| `Solver.assert_and_track(expr, label)` | `Solver.assert_and_track(term, label, tm)` | ✅ supported |
| `set_option(key, value)` | `Solver.set_option(key, value)` | ✅ supported |

## Notes

- **Array (⚠️ partial)**: `TermManager.mk_select` and `mk_store` build AST nodes, but `mk_var` does not accept an `"Array[...]"` sort string. No typed array-sort constructor is exposed. Full array theory requires adding `ArraySort` to `parse_sort_name()`.
- **FP (❌ missing)**: Floating-point sorts and operations (FPSort, FPVal, fp_add, etc.) are not wrapped. The underlying oxiz-core may have partial FP AST nodes, but no PyO3 layer exists.
- **String (❌ missing)**: String sort and operations (StringVal, Length, Concat, Contains) are not wrapped. No string sort is registered in `parse_sort_name()`.
- **ForAll / Exists (❌ missing)**: Quantified formulas are not constructible from Python. No `mk_forall` or `mk_exists` on TermManager, no module-level `ForAll` / `Exists` function.
- **Optimizer**: `Optimizer.push()` / `Optimizer.pop()` exist but take no arguments (z3-python's `pop()` also takes no arguments for Optimize).
- **Timeout for Optimizer**: not separately exposed; only `Solver.set_timeout()` is available.
