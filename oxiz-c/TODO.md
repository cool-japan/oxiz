# oxiz-c Phase-2 TODO

## Phase-1 Status (completed)

Phase-1 landed 13 `extern "C"` functions across 5 source files (≤381 LoC, under the 1000 LoC cap):
- Version/error utilities: `oxiz_version_string`, `oxiz_error_string`
- Context lifecycle: `oxiz_context_new`, `oxiz_context_free`
- Solver lifecycle: `oxiz_solver_new`, `oxiz_solver_free`, `oxiz_solver_push`, `oxiz_solver_pop`
- Assertion + solve: `oxiz_solver_assert_smtlib2`, `oxiz_solver_check`
- Model access (string): `oxiz_solver_get_model_string`, `oxiz_model_string_data`, `oxiz_model_string_free`

Phase-1 delivers an opaque-handle, SMT-LIB2-string boundary. All term construction and
model inspection go through string I/O. `OxizContext._private: ()` is explicitly reserved
for Phase-2 options. `OxizSolver` wraps `oxiz_solver::Context` directly; the context
parameter to `oxiz_solver_new` is accepted but not yet read. Phase-2 expands to typed
handles and richer APIs.

## Phase-2 Scope Discussion

Phase-1 honored ≤1000 LoC. Phase-2 will exceed this substantially. Proposed approach:
split into 5 sub-phases, each with its own ≤1000 LoC cap, implementable independently.
Phase-2 sub-phases can proceed in any order; term handles (Phase 2a) are a prerequisite
for model-eval (Phase 2c) and tactic (Phase 2b) ABIs.

---

## Phase 2a — Term-handle ABI

- [ ] Add opaque `OxizTerm` and `OxizSort` handle types in new `oxiz-c/src/term.rs` (~100 LoC);
  intern them into an arena owned by `OxizContext` (upgrade `_private: ()` in `src/context.rs`
  to hold the arena, ~30 LoC change); reference Z3's `Z3_ast` / `Z3_sort` design in
  `../z3/src/api/z3_api.h`.

- [ ] Add sort constructors in `src/term.rs` (~80 LoC): `oxiz_mk_bool_sort(ctx) -> *mut OxizSort`,
  `oxiz_mk_int_sort(ctx) -> *mut OxizSort`, `oxiz_mk_real_sort(ctx) -> *mut OxizSort`,
  `oxiz_mk_bv_sort(ctx, width: u32) -> *mut OxizSort`,
  `oxiz_mk_array_sort(ctx, domain: *mut OxizSort, range: *mut OxizSort) -> *mut OxizSort`;
  mirror `Z3_mk_*_sort` in `../z3/src/api/api_ast.cpp`.

- [ ] Add variable/constant constructors in `src/term.rs` (~60 LoC):
  `oxiz_mk_const(ctx, name: *const c_char, sort: *mut OxizSort) -> *mut OxizTerm`,
  `oxiz_mk_bool_val(ctx, val: bool) -> *mut OxizTerm`,
  `oxiz_mk_int_val(ctx, val: i64) -> *mut OxizTerm`; all return `NULL` + set last-error on
  failure; add `OxizError::InvalidSort = 7` to `src/error.rs` (~5 LoC).

- [ ] Add term accessor/traversal functions in `src/term.rs` (~60 LoC):
  `oxiz_term_kind(ctx, term: *const OxizTerm) -> u32` (maps to an `OxizTermKind` repr-C enum),
  `oxiz_term_num_args(ctx, term: *const OxizTerm) -> u32`,
  `oxiz_term_get_arg(ctx, term: *const OxizTerm, idx: u32) -> *mut OxizTerm`;
  mirror `Z3_get_ast_kind` / `Z3_get_app_num_args` in `../z3/src/api/api_ast.cpp`.

- [ ] Update `build.rs` and `cbindgen.toml` to expose `OxizTerm`, `OxizSort`, and
  `OxizTermKind` in `include/oxiz.h` (~10 LoC); verify cbindgen emits `#[repr(C)]` enum
  correctly under `OXIZ_C_GEN_HEADER=1`.

---

## Phase 2b — Tactic ABI

- [ ] Add opaque `OxizTactic` handle in new `oxiz-c/src/tactic.rs` (~50 LoC); back it with a
  `Box<dyn oxiz_core::tactic::Tactic>` (or equivalent trait object); add `oxiz_tactic_free`
  (NULL-safe, ~10 LoC); reference `Z3_tactic` design in `../z3/src/api/api_tactic.cpp`.

- [ ] Add `oxiz_mk_tactic(ctx, name: *const c_char) -> *mut OxizTactic` in `src/tactic.rs`
  (~30 LoC): perform UTF-8 validation, look up the name in `oxiz-core::tactic` registry,
  return `NULL` + `OxizError::Unimplemented` for unknown names; mirror `Z3_mk_tactic`.

- [ ] Add composition constructors in `src/tactic.rs` (~40 LoC):
  `oxiz_tactic_and_then(ctx, t1: *mut OxizTactic, t2: *mut OxizTactic) -> *mut OxizTactic`,
  `oxiz_tactic_or_else(ctx, t1: *mut OxizTactic, t2: *mut OxizTactic) -> *mut OxizTactic`;
  the returned handle owns both sub-tactics (do not double-free `t1`/`t2` after composition).

- [ ] Add opaque `OxizGoal` and `OxizGoalList` handle types (append to `src/tactic.rs`, ~50 LoC)
  and implement `oxiz_tactic_apply(ctx, tactic: *mut OxizTactic, goal: *mut OxizGoal) -> *mut OxizGoalList`
  (~80 LoC) returning a list of subgoals; add `oxiz_goal_list_size`, `oxiz_goal_list_get`,
  and `oxiz_goal_list_free`; mirror `Z3_tactic_apply` in `../z3/src/api/api_tactic.cpp`.

---

## Phase 2c — Model-eval ABI

Phase-1 `OxizModelString` returns a raw SMT-LIB2 `(model ...)` string (see `src/model.rs`).
Phase-2c adds a typed `OxizModel` handle that requires Phase-2a term handles.

- [ ] Add opaque `OxizModel` handle in new `oxiz-c/src/model_typed.rs` (~80 LoC); bridge to
  the native model returned by `oxiz_solver::Context::format_model` (or a richer API once
  exposed); add `oxiz_model_free` (NULL-safe); keep `OxizModelString` and
  `oxiz_solver_get_model_string` intact for backward compatibility.

- [ ] Implement `oxiz_model_eval(ctx, model: *mut OxizModel, term: *mut OxizTerm, completion: bool, out: *mut *mut OxizTerm) -> c_int`
  in `src/model_typed.rs` (~40 LoC): evaluates a term in the model; writes result to `*out`;
  returns `OxizError::SolverError` if model evaluation fails; mirror `Z3_model_eval`.

- [ ] Implement `oxiz_model_get_const_interp(ctx, model: *mut OxizModel, const_term: *mut OxizTerm) -> *mut OxizTerm`
  (~30 LoC) and `oxiz_model_get_func_interp(ctx, model: *mut OxizModel, func_term: *mut OxizTerm) -> *mut OxizFuncInterp`
  (~60 LoC, requires new opaque `OxizFuncInterp` type with `oxiz_func_interp_get_num_entries`,
  `oxiz_func_interp_get_entry`, `oxiz_func_interp_free`); mirror `Z3_model_get_const_interp`
  and `Z3_model_get_func_interp` in `../z3/src/api/api_model.cpp`.

- [ ] Add model iteration helpers in `src/model_typed.rs` (~30 LoC):
  `oxiz_model_get_num_consts(ctx, model: *mut OxizModel) -> u32`,
  `oxiz_model_get_const_decl(ctx, model: *mut OxizModel, i: u32) -> *mut OxizTerm`; these
  allow a C caller to iterate all constant assignments without parsing the SMT-LIB2 string.

---

## Phase 2d — Callback ABI

- [ ] Define `OxizDecideCallback` type alias and `OxizDecideClosure` struct in new
  `oxiz-c/src/callback.rs` (~40 LoC):
  `type OxizDecideCallback = unsafe extern "C" fn(ctx: *mut OxizContext, data: *mut c_void) -> *mut OxizTerm`;
  store `(cb, data)` pair in `OxizContext` (update `src/context.rs`, ~20 LoC), replacing the
  `_private: ()` placeholder introduced in Phase-2a with a proper config struct.

- [ ] Implement `oxiz_set_decide_callback(ctx: *mut OxizContext, cb: OxizDecideCallback, data: *mut c_void) -> c_int`
  in `src/callback.rs` (~30 LoC): stores `cb` and `data` in the context; passing `cb = NULL`
  clears the callback; propagate to the solver via `oxiz_solver_new` (which already receives
  `ctx` but currently does not read it).

- [ ] Document safety contract in `src/callback.rs` module-level doc comment (~20 LoC):
  the callback MUST NOT call back into any `oxiz_solver_*` function (re-entrancy undefined),
  MUST NOT mutate the context, and MUST return either a valid interned term or NULL to
  defer to the default decision heuristic.

- [ ] Add threading model documentation in `src/callback.rs` (~15 LoC): clarify that
  `OxizContext` and `OxizSolver` are NOT `Send`/`Sync` across the C ABI boundary, callbacks
  are invoked on the same thread as `oxiz_solver_check`, and no locking is performed.

- [ ] Add integration test `oxiz-c/tests/callback_smoke.rs` (~50 LoC): register a trivial
  callback via `oxiz_set_decide_callback`, call `oxiz_solver_assert_smtlib2` + `oxiz_solver_check`,
  verify the callback was invoked at least once; use `std::sync::atomic::AtomicUsize` as the
  `data` pointer counter.

---

## Phase 2e — Header Stability + CI

- [ ] Add snapshot test `oxiz-c/tests/header_snapshot.rs` (~40 LoC): run cbindgen in-process
  (or as a `Command`) under `OXIZ_C_GEN_HEADER=1`, diff the output against a committed golden
  file at `oxiz-c/include/oxiz.h.golden`; fail with a clear message if they diverge so that
  silent ABI breaks are caught in CI.

- [ ] Add C smoke-test `oxiz-c/tests/c_smoke.c` (~80 LoC) compiled via `cc` crate in a
  dedicated `oxiz-c/tests/build_c_smoke.rs` test (~30 LoC): the C file calls
  `oxiz_context_new`, `oxiz_solver_new`, `oxiz_solver_assert_smtlib2`, `oxiz_solver_check`,
  `oxiz_solver_get_model_string`, `oxiz_model_string_data`, `oxiz_model_string_free`,
  `oxiz_solver_free`, `oxiz_context_free` and asserts expected return codes; add `cc` to
  `[dev-dependencies]` in `Cargo.toml`.

- [ ] Extend `.github/workflows/` (or create `oxiz-c-ci.yml`) with a matrix step that builds
  `oxiz-c` as both `cdylib` and `staticlib` on `ubuntu-latest` and `macos-latest`; verify
  the `include/oxiz.h` header is regenerated cleanly under `OXIZ_C_GEN_HEADER=1` and matches
  the golden snapshot.

- [ ] Add `OXIZ_ABI_VERSION` constant to `include/oxiz.h` (via a `#define` emitted by cbindgen
  or a hand-maintained `include/oxiz_abi.h` fragment, ~5 LoC) and document in `src/lib.rs`
  that Phase-1/2 ABI is explicitly **unstable** (0.x series); propose a stability guarantee
  policy for 1.0 (no removals, no signature changes without a major version bump).
