# OxiZ TODO

Last Updated: 2026-08-26

---

## Historical Milestone (v0.2.0): Initial 88-Benchmark Parity Suite

**Date Achieved**: February 5, 2026
**Release Status**: Published (Feb 6, 2026)

> Original v0.2.0 announcement, retained verbatim for historical record: "OxiZ has achieved **100% correctness parity with Z3** across all 88 benchmark tests spanning 8 core SMT-LIB logics. This validates OxiZ as a **production-ready Pure Rust SMT solver**."

**Superseded by the v0.2.4 honest re-audit.** The comparator that produced the table below counted an `Unknown` answer as a match (`bench/z3_parity/src/comparator.rs` — see the now-fixed `[x]` finding under "Production-Readiness Audit Findings" below), so "100%" was reachable by declining to answer rather than by matching Z3's verdict. The comparator used from 0.2.4 onward never counts `Unknown` as a match. The current, honestly-measured status lives in "Current Statistics" below, with the tracked per-environment snapshot `bench/z3_parity/results.<os>-<arch>.json` as the authoritative source — only `results.macos-aarch64.json` is currently in the tree, with `results.linux-x86_64.json` to join it once a Linux environment's run is committed. The methodology's agreement rule (every tracked snapshot must agree on the verdict of every benchmark — `oxiz_result`, `z3_result`, `match_status` — with only the timings expected to differ between machines) is enforced by `bench/z3_parity/tests/cross_env_verdict_agreement.rs` on every `cargo test`, but with a single snapshot in the tree that check is currently vacuous, not yet exercised across environments; the un-suffixed `results.json` is git-ignored local scratch output and is not evidence. As of this release, **170/170 Correct** on the extended 19-logic suite (the three quantified logics that were still below 100% at v0.3.0 — `UFLIA`/`UFLRA`/`AUFLIA` — are now all at 100%; the suite grew 168→170 in 0.3.3 with two new symbolic-`RoundingMode` QF_FP benchmarks, and no verdict moved on any of the 168 pre-existing benchmarks) and **88/88 Correct** on this original 8-logic/88-benchmark quickstart core, both under the honest comparator that never counts `Unknown` as a match. This is a claim about the differential parity suite, not a blanket claim of 100% Z3 compatibility.

### Z3 Parity Results (as originally reported, v0.2.0 — see supersession note above)

| Logic | Tests | Result | Status |
|-------|-------|--------|--------|
| QF_LIA | 16/16 | 100% | Perfect |
| QF_LRA | 16/16 | 100% | Perfect |
| QF_NIA | 1/1 | 100% | Perfect |
| QF_S | 10/10 | 100% | Perfect |
| QF_BV | 15/15 | 100% | Perfect |
| QF_FP | 10/10 | 100% | Perfect |
| QF_DT | 10/10 | 100% | Perfect |
| QF_A | 10/10 | 100% | Perfect |
| **TOTAL** | **88/88** | **100%** | **Production Ready** |

---

## Progress Summary

| Priority | Completed | Pending | Progress |
|----------|-----------|---------|----------|
| Critical | 25 | 0 | 100% |
| High | 15 | 0 | 100% |
| Medium | 17 | 0 | 100% |
| Low | 9 | 0 | 100% |
| Post-Parity: Performance | 27 | 1 | 96% |
| Post-Parity: UX | 3 | 0 | 100% |
| Post-Parity: Debugging | 4 | 0 | 100% |
| Post-Parity: Docs | 5 | 0 | 100% |
| Post-Parity: Theories | 10 | 0 | 100% |
| Post-Parity: Advanced | 12 | 0 | 100% |
| Post-Parity: Ecosystem | 4 | 3 | 57% |
| **Total** | **131** | **4** | **97%** |

Recounted at the 0.3.1 release (2026-07-31) directly from the checkboxes under "Post-Parity Priorities" below; the item population is unchanged (135), only the completed/pending split moved. The 4 still-pending items are JIT-style specialization for hot theory operations (deferred to v0.4.0) and the "Tool integration" group — its umbrella entry plus symbolic-execution-tool and verification-framework integration (its SMT-COMP 2026 sub-item is done bar the portal opening).

---

## Current Statistics (v0.3.2 - 2026-08-05, re-measured at release time)

- **Rust Lines of Code (code)**: 451,853 code lines across 1,276 files (tokei, `--exclude target`; 37,877 comment lines, 74,573 blanks)
- **Total Rust Lines (with docs/tests)**: 564,303 (grand total across all languages: 600,702 lines in 1,483 files)
- **Tests**: 9,953 (workspace, nextest, all-features, all passing; 8 skipped, 0 failures) plus 110 passing doc-tests (`cargo test --doc --workspace --all-features`, 0 failures)
- **Z3 Parity (extended suite, 170 benchmarks / 19 logics, against installed z3 4.15.4)**: **170 Correct / 0 Wrong / 0 Inconclusive / 0 Timeout / 0 Error** — **100% of the differential parity suite**, with all 19 logic families at 100% Correct (AUFLIA 10/10, AUFLIRA 5/5, QF_ABV 5/5, QF_ALIA 5/5, QF_AUFBV 5/5, QF_AUFLIA 5/5, QF_NIRA 5/5, QF_UFLIA 5/5, QF_UFLRA 5/5, UFLIA 20/20, UFLRA 10/10, qf_a 10/10, qf_bv 15/15, qf_dt 10/10, qf_fp 12/12, qf_lia 16/16, qf_lra 16/16, qf_nia 1/1, qf_s 10/10 — see the tracked per-environment snapshot `bench/z3_parity/results.<os>-<arch>.json`; only `results.macos-aarch64.json` is currently in the tree, so the cross-environment agreement rule (every tracked snapshot must agree on every benchmark's verdict, differing only in timings) applies once a second snapshot exists rather than being exercised today; `results.json` itself is git-ignored scratch output of the last local run). Measured under the honest comparator, which never counts `Unknown` as a match, against a real `z3 4.15.4` binary; verified over three consecutive full runs on an idle machine plus a fourth run after the repeated-check-sat resource work. This is a claim about this benchmark suite — **not** a blanket claim of 100% Z3 compatibility as a general property. Closed in 0.3.1: the last quantified-logic gaps (`AUFLIA` 7→10/10, `UFLIA` 14→20/20, `UFLRA` 5→10/10) via MBQI finite-range quantifier expansion (AUFLIA), Skolem witness synthesis + CEGAR (UFLIA), and symbolic model certification over Reals + quasi-macro detection (UFLRA); the three former 60s timeouts now solve in ~1ms. Grew 168→170 in 0.3.3: two new symbolic-`RoundingMode` QF_FP benchmarks (`qf_fp` 10/10→12/12); no verdict moved on any of the 168 pre-existing benchmarks, and none of the fifteen SMT-LIB soundness fixes from the 0.3.3 issue-tracker sweep are visible in this suite (a curated suite already at 100% cannot show a soundness gain — see the Issue-tracker intake section).
- **Workspace Crates**: 17 members (16 default-members; `oxiz-py` is excluded because it needs maturin, and `fuzz` is a separate harness outside the workspace)
- **todo!/unimplemented! macros**: 0 outside test code (all Rust crates)
- **Clippy Warnings**: 0 (`cargo clippy --workspace --all-targets --all-features`, clean in both dev and release profiles; `clippy::unwrap_used` denied in all 17 member crates)
- **Rustdoc / cargo-deny**: `cargo doc` clean under `-D warnings`; `cargo deny check bans` clean
- **Largest File**: 1,997 lines (`oxiz-solver/src/solver/tests.rs`) — all files under 2,000 lines; largest non-test file is 1,989 lines (`oxiz-solver/src/mbqi/model_completion.rs`)
- **Toolchain**: cargo/rustc 1.95.0

---

## Beyond Z3: Key Differentiators

OxiZ is not just a Z3 port - it surpasses Z3 in critical areas:

1. **Machine-Checkable Proofs** (oxiz-proof) - DRAT, Alethe, LFSC + Coq/Lean/Isabelle exports
2. **Spacer/PDR** (oxiz-spacer) - Missing in CVC5, Yices, and most Z3 clones!
3. **WASM-First** (oxiz-wasm) - Target <2MB vs Z3's ~20MB
4. **Native Parallelism** - Rayon portfolio solving, work-stealing
5. **Memory Safety** - Pure Rust, no FFI, guaranteed safety
6. **Craig Interpolation** - McMillan, Pudlak, Huang algorithms with theory support
7. **Verified Z3 Parity on the Differential Suite** - 88/88 on the 8-logic quickstart suite, 170/170 on the extended 19-logic suite with 0 Wrong / 0 Inconclusive / 0 Timeout / 0 Error (honest comparator, never counts `Unknown` as a match — see Current Statistics; scoped to the benchmark suite, not a blanket claim of 100% Z3 compatibility)
8. **EasySolver API** - Builder pattern, one-liner solving for common use cases
9. **Arena Allocator** - Custom bumpalo-backed AST allocator (feature-gated)
10. **Parallel Theory Checking** - Rayon-based, feature-gated

---

## Completed: April 4, 2026

### Performance Optimization
- [x] Custom arena allocator for AST nodes (bumpalo-backed, feature-gated)
- [x] Clause pool for SAT solver (5 size-based buckets, recycle/reuse)
- [x] SIMD-friendly polynomial operations (chunk-of-4 autovectorization)
- [x] Optimized hash functions for term interning (TermKindHasher)
- [x] FP bit-blasting cache (avoid redundant bit-blasting)
- [x] Model generation optimization (lazy evaluation cache)
- [x] Parallel theory checking (rayon-based, feature-gated)
- [x] Lock-free data structures for parallel solving
- [x] Lazy evaluation strategies

### User Experience
- [x] EasySolver convenience API (builder pattern, one-liner solving)
- [x] Better error messages (hints, did_you_mean, context_snippet)
- [x] Timeout and resource limit APIs (ResourceLimits, ResourceMonitor)

### Debugging Support
- [x] Solver state visualization (SolverStateSnapshot, DOT graph)
- [x] Trace generation (TraceEvent, JSON/text output)
- [x] Better conflict explanations (ConflictExplainer, UnsatExplanation)
- [x] Model minimization (linear and binary search strategies)

### Documentation
- [x] Performance tuning guide (docs/PERFORMANCE_TUNING.md)
- [x] Theory-specific guides (docs/THEORY_GUIDE.md)
- [x] Z3 migration guide (docs/MIGRATION_Z3.md)
- [x] Common pitfalls (docs/PITFALLS.md)
- [x] Case studies (docs/CASE_STUDIES.md)

### File Maintenance
- [x] solve_eqs.rs re-split (1942 -> 1553 lines)
- [x] rational.rs re-split (1940 -> 1388 + 553 tests)

### Stats Delta (March 31)
- Tests: 6,122 -> 6,155 (+33 new)
- Rust LoC: 392,274 -> 393,292 (+1,018)
- Clippy warnings: 0
- Largest file: 1,892 lines
- All files under 2,000 lines

---

## Post-Parity Priorities (v0.3.0 and Beyond)

### High Priority: Performance Optimization (27/28 Complete - JIT specialization deferred to v0.4.0)

**Goal**: Achieve performance parity with Z3 (currently ~1.5-2x slower)

- [x] Custom allocators (arena for AST nodes, clause pooling)
- [x] SIMD-friendly polynomial operations (chunk-of-4 autovectorization)
- [x] Optimized hash functions (TermKindHasher for term interning)
- [x] FP bit-blasting cache
- [x] Model generation optimization (lazy evaluation cache)
- [x] Parallel theory checking (rayon-based, feature-gated)
- [x] Lock-free data structures for parallel solver
- [x] Lazy evaluation strategies
- [x] Clause pool for SAT solver (5 size-based buckets)

- [x] Profile remaining hot paths (10 items) (planned 2026-04-19)
  - **Goal:** Reproducible profiling harness covering 10 named hot paths; snapshot `docs/PROFILING_REPORT.md` names worst offenders; each path gets a `ScopedTimer` pair for CI-measurable cost.
  - **Design:** Extend `oxiz-sat/src/profiling.rs::ProfilingCategory` with 10 categories (SatPropagation, TheoryCheck, EGraphMerge, SimplexPivot, BvPropagation, StringAutomata, ArrayExtensionality, ProofGeneration, Parser, CacheMiss); wire at call sites; new `bench/profile/` crate; extend `scripts/flamegraph.sh` with `--category`; emit `docs/PROFILING_REPORT.md`.
  - **Files:** `oxiz-sat/src/profiling.rs`, 10 instrumented call sites across crates, new `bench/profile/{Cargo.toml,benches/profile_benchmarks.rs,src/lib.rs}`, `scripts/flamegraph.sh`, new `docs/PROFILING_REPORT.md`, root `Cargo.toml` workspace member.
  - **Tests:** new `oxiz-sat/tests/profiling_pass.rs` — each category records ≥1 sample; JSON summary is parseable.
  - [x] SAT solver clause propagation
  - [x] Theory solver check() methods
  - [x] E-graph merge operations
  - [x] Simplex pivot operations
  - [x] BV constraint propagation
  - [x] String solver automata operations
  - [x] Array extensionality checks
  - [x] Proof generation overhead
  - [x] Parser performance
  - [x] Cache miss analysis

- [x] Additional performance improvements (5 of 6 sub-items; JIT deferred) (planned 2026-04-19)
  - **Goal:** Five concrete allocation-reduction fixes: in-place watchlist updates, SmallVec for EClass::nodes, incremental theory cache, cache-friendly Clause layout, allocation-free EUF propagation.
  - **Design:** (1) `oxiz-sat/src/cdcl/propagation.rs` — swap_remove+clear vs Vec::clone; (2) `SmallVec<[Term;4]>` for `oxiz-core/src/egraph/eclass.rs::EClass::nodes`; (3) memo `(theory_id, level)→propagation set` in coordinator.rs; (4) hot-field-first struct layout in `oxiz-sat/src/clause.rs`; (5) per-solver reuse buffer in `oxiz-theories/src/euf/solver.rs`.
  - **Files:** `oxiz-sat/src/cdcl/propagation.rs`, `oxiz-core/src/egraph/eclass.rs`, `oxiz-solver/src/combination/coordinator.rs`, `oxiz-sat/src/clause.rs`, `oxiz-theories/src/euf/solver.rs`.
  - **Tests:** new `oxiz-sat/tests/allocation_reduction.rs` with dhat-heap counts; per-fix unit tests.
  - [x] Reduce allocations further (in-place updates)
  - [x] Better data structure choices (profiling-driven)
  - [x] Incremental computation caching
  - [~] JIT-style specialization for hot theory operations
  - **Scope-box (2026-04-24):** Pure-Rust EUF data-layout + allocation-reduction + incremental-backtrack pass. Items 1–5 completed; parent umbrella (JIT/codegen layer) deferred to v0.4.0.
  - [x] EUF production-path benchmarks + regression baseline (2026-04-24)
    - Added `oxiz-theories/benches/euf_benchmarks.rs` with 5 criterion workloads driving `EufSolver` directly; 5 baseline entries added to `bench/regression/baseline.json`.
  - [x] EUF cheap-wins bundle: fingerprint pre-filter + `#[inline]` cross-crate + hoist `get_function_props` (2026-04-24)
    - Activated dead fingerprint pre-filter in `propagate`; added `#[inline]` to 7 cross-crate wrappers; hoisted `get_function_props` out of inner loop. Bench deltas: intern_leaf −24%, intern_app −16%, merge_congruence −10%, merge_injective −22%.
  - [x] EUF allocation reduction in `propagate` (2026-04-24)
    - Reusable canonicalize buffer (out-param), proof_forest changed to `Vec<SmallVec<[MergeEdge; 4]>>`, `SigUpdateEntry` flat struct. Bench 3 −6.7%, bench 4 −6.7%, bench 5 −8.2%.
  - [x] EUF incremental sig_table + fingerprint_table undo trail (2026-04-24)
    - Replaced O(|nodes|) rebuild-on-pop with per-insertion `SigTrailEntry` trail + `sig_trail_limits`. Trail guarded by `is_empty()` check so non-incremental workloads see zero overhead. Miri clean; 6366/6366 tests pass.
  - [x] EUF `ENode` layout reorder + `func: u32` sentinel (2026-04-24)
    - Reordered fields to put hot fields (`func`, `fingerprint`) first; replaced `Option<u32>` with `u32` + `NO_FUNC = u32::MAX` sentinel. Added `ENode::leaf()` / `ENode::app()` constructors. `test_enode_size_regression` confirms ≤56B.
  - [x] Memory layout optimization
  - [x] Allocation-free theory propagation paths

- [x] Performance regression testing (3 items)
  - [x] CI/CD integration for performance tracking (planned 2026-04-19)
  - [x] Automated benchmark comparison vs Z3 (planned 2026-04-19)
  - [x] Performance dashboard (planned 2026-04-19)

**Target**: Within 1.2x of Z3 performance by v0.3.0

### High Priority: Extended Theory Coverage

**Goal**: Support additional SMT-LIB logics beyond the core 8

- [x] Quantified logics (5 items)
  - [x] UFLIA - Uninterpreted Functions + Linear Integer Arithmetic
  - [x] UFLRA - Uninterpreted Functions + Linear Real Arithmetic
  - [x] AUFLIA - Arrays + UF + LIA
  - [x] AUFLIRA - Arrays + UF + LIA + LRA
  - [x] Improve quantifier instantiation heuristics

- [x] Combined theories (3 items)
  - [x] QF_AUFBV - Arrays + UF + BV (validation needed)
  - [x] QF_ALIA - Arrays + LIA
  - [x] QF_ABV - Arrays + BV

- [x] Non-linear arithmetic (2 items)
  - [x] Extend QF_NIA coverage (more benchmarks)
  - [x] QF_NIRA - Non-linear Integer/Real Arithmetic

### Medium Priority: Advanced Features

- [x] Enhanced preprocessing (5 items) (planned 2026-04-19)
  - **Goal:** Five tactics: `bmc-unroll` (spacer/bmc wrapper), `aggressive-simplify` (new rewrite rules), `ctx-dep-rewrite` polish (dead-branch elimination in ITEs), `symmetry-break` (lex-leader constraints), `cube-improve` (VSIDS-depth-aware cubes).
  - **Design:** new `oxiz-spacer/src/tactics/bmc_unroll.rs`; extend `oxiz-core/src/simplification/mod.rs`; polish `ctx_solver_simplify.rs`; new `oxiz-sat/src/tactics/symmetry.rs`; extend `oxiz-sat/src/cube.rs::CubeGenerator`.
  - **Files:** `oxiz-spacer/src/tactics/bmc_unroll.rs` (new), `oxiz-spacer/src/lib.rs`, `oxiz-core/src/simplification/mod.rs`, `oxiz-core/src/tactic/ctx_solver_simplify.rs`, `oxiz-sat/src/tactics/symmetry.rs` (new), `oxiz-sat/src/cube.rs`, `oxiz-core/src/tactic/registry.rs`.
  - **Tests:** per-tactic unit test (rewrite shape) + integration test (apply tactic, status preserved).
  - [x] Bounded model checking tactics (planned 2026-04-19)
          - **Goal:** `oxiz-spacer::tactics::BmcUnrollTactic` is production-ready: documented re-export, ≥4 unit tests covering nested next-state vars, idempotent re-application, depth-from-option > 5, and integration with `oxiz-spacer::Bmc`.
          - **Design:** Existing `BmcEngine`/`BmcUnrollTactic` (224 lines) renames `x_next`/`x'` → `x@n+1`. Verify rename correctness under multiple applications; verify `NotApplicable` on goals with < 3 assertions; document distinction from production `Bmc` solver in `oxiz-spacer/src/bmc.rs`.
          - **Files:** `oxiz-spacer/src/tactics/bmc_unroll.rs` (tests + doc), `oxiz-spacer/src/tactics/mod.rs` (doc comment), `oxiz-spacer/src/lib.rs` (re-export at crate root), `oxiz-spacer/tests/bmc_unroll_integration.rs` (new).
          - **Tests:** (a) `test_bmc_unroll_handles_nested_next_state`; (b) `test_bmc_unroll_idempotent_under_reapply`; (c) `test_bmc_unroll_from_option_depth`; (d) integration test handing result to `Bmc::check`.
          - **Risk:** suffix-rename collision on `@n+1` substrings already in names. Mitigation: assert original name is a substring; switch to `@@n+1` separator if collision found.
          - **Scope cap:** ≤200 LoC net-new.
  - [x] More aggressive simplification (planned 2026-04-19)
          - **Goal:** `oxiz-core::simplification::AggressiveSimplifier` gains substantive new rewrite rules (Boolean, arithmetic, bit-vector, ITE) so `aggressive: true` measurably shrinks goals.
          - **Design:** Extend `simplify_*` family in `oxiz-core/src/simplification/mod.rs`. Rules: (1) De Morgan `Not(Not(a))→a`; (2) Implication identities `Implies(true,b)→b` etc.; (3) XOR identities; (4) Arithmetic constant folding `Add(c1,c2)→c`; (5) BV trivial `BvAnd(x,0)→0` etc.; (6) Equality `Eq(x,x)→true`; (7) ITE `If(true,a,_)→a`, `If(_,a,a)→a`. Use existing memo cache for idempotence.
          - **Files:** `oxiz-core/src/simplification/mod.rs` (extend); new `oxiz-core/tests/aggressive_simplify_rules.rs`; preserve in-flight 3-line test tolerance in `aggressive_simplify.rs`.
          - **Tests:** 7 per-rule-family unit tests + 2 integration tests (Boolean-heavy goal, BV-heavy goal). Run `rslines 50` on `tactic/mod.rs` after edit; invoke `splitrs` if > 2000 lines.
          - **Risk:** recursion memo collision under rule interaction. Mitigation: existing memo cache; assert O(N) lookup count in one test.
          - **Scope cap:** ≤500 LoC net-new. No new term kinds, no TermManager API changes.
  - [x] Context-dependent rewriting (planned 2026-04-19)
          - **Goal:** Live `CtxSolverSimplifyTactic` in `oxiz-core/src/tactic/ctx_simplify.rs` gains dead-branch ITE elimination: when goal context implies `cond` or `Not(cond)`, the corresponding branch of `If(cond, t, e)` is substituted.
          - **Design:** (1) Build `HashSet<TermId>` from goal assertions as context. (2) For each `If(c,t,e)`: if `c` in ctx → `t`; if `Not(c)` in ctx → `e`; else descend with augmented ctx (t-branch: ctx∪{c}, e-branch: ctx∪{Not(c)}). (3) Use `manager.simplify` for bottom-up rebuild. (4) Cap recursion depth at 32; on overflow return original term (sound). **Path resolution:** Plan's cited path `ctx_solver_simplify.rs` does NOT exist; `core/ctx_solver_simplify.rs` is dead placeholder — do NOT touch it. Target only `ctx_simplify.rs`.
          - **Files:** `oxiz-core/src/tactic/ctx_simplify.rs` only. No changes to `mod.rs` re-exports or dead placeholder.
          - **Tests:** (a) `test_ite_eliminates_when_cond_in_context`; (b) `test_ite_eliminates_when_neg_cond_in_context`; (c) `test_ite_descends_with_augmented_ctx` (nested ITE); (d) `test_ite_recursion_depth_cap` (50-deep ITE, no hang); (e) `test_apply_mut_status_preserved`.
          - **Risk:** augmented context shared-mutation bug. Mitigation: per-call scoping, no global ctx mutation; test (c) validates.
          - **Scope cap:** ≤300 LoC net-new in `ctx_simplify.rs`.
  - [x] Symmetry breaking (planned 2026-04-19)
          - **Goal:** `oxiz-sat::tactics::SymmetryBreakTactic` gains coverage proving tactic shrinks model space. Re-export already at `oxiz-sat/src/lib.rs:228`.
          - **Design:** Existing 155-line tactic runs `AutomorphismDetector` → `SymmetryBreaker::new(group, Lex)` → `generate_predicates()`. Validate via 4 tests; tighten `NotApplicable` paths.
          - **Files:** `oxiz-sat/src/tactics/symmetry.rs` (test additions only). `oxiz-sat/src/symmetry.rs` unchanged unless coverage gap found.
          - **Tests:** (a) `test_symmetry_break_full_3var_symmetry` — fully symmetric 4-clause CNF over 3 vars yields ≥1 lex-leader predicate; (b) `test_symmetry_break_asymmetric_clauses` → `NotApplicable`; (c) `test_symmetry_break_mixed_boolean_integer` → `NotApplicable`; (d) `test_symmetry_break_reduces_model_count` — solver on (clauses ∪ predicates) has fewer satisfying assignments than on clauses alone.
          - **Risk:** `AutomorphismDetector` may return spurious symmetries. Mitigation: tests assert tactic behaviour (predicates emitted/not), not detector internals.
          - **Scope cap:** ≤200 LoC net-new (tests only).
  - [x] Cube generation improvements (planned 2026-04-19)
          - **Goal:** Validate and prove that `oxiz-sat::cube::CubeGenerator::depth_limit_for_cube` is genuinely VSIDS-depth-aware (confirmed: `extra_depth = log2(activity_sum/avg)` at lines 220–247), and validate `CubeImproveTactic` end-to-end.
          - **Design:** No production-code changes unless a test forces one (e.g. `extra_depth.ceil()` rounding kills the increment for activity ratio < 2 — fix only if observed). All work is tests.
          - **Files:** `oxiz-sat/src/cube.rs` (test additions to `mod tests`); `oxiz-sat/src/tactics/cube_improve.rs` (test additions).
          - **Tests:** (a) `test_depth_limit_uniform_activity_equals_max_depth`; (b) `test_depth_limit_high_activity_increases_depth` (4× average → depth > max_depth); (c) `test_generate_vsids_guided_orders_by_activity`; (d) `test_cube_improve_tactic_emits_subgoals_per_cube` (4-var Boolean goal → ≥2 subgoals); (e) `test_cube_improve_status_preserved`.
          - **Risk:** NaN from empty `variable_scores`. Mitigation: existing `if variable_scores.is_empty() { 1.0 }` guard; test (a) covers it.
          - **Scope cap:** ≤200 LoC net-new.

- [x] Better quantifier handling (4 items) (planned 2026-04-19)
  - **Goal:** (a) PatternCoverScorer (greedy set cover), (b) conflict_score VSIDS for quantifiers in conflict_driven.rs, (c) virtual-substitution QE (Loos–Weispfenning), (d) per-quantifier instantiation budget in MBQI.
  - **Design:** extend `patterns.rs` with `PatternCoverScorer`; extend `conflict_driven.rs` with `conflict_score: HashMap<QuantifierId,u32>`; new `oxiz-core/src/qe/virtual_substitution.rs`; add `MBQIBudget::per_quantifier` to `heuristics.rs`.
  - **Files:** `oxiz-solver/src/mbqi/patterns.rs`, `oxiz-solver/src/mbqi/conflict_driven.rs`, `oxiz-core/src/qe/arith.rs`, `oxiz-core/src/qe/virtual_substitution.rs` (new), `oxiz-core/src/qe/mod.rs`, `oxiz-solver/src/mbqi/heuristics.rs`, `oxiz-solver/src/mbqi/mod.rs`.
  - **Tests:** pattern-cover, conflict-priority, VS, budget enforcement unit tests.
  - [x] Pattern-based instantiation improvements
  - [x] Conflict-driven instantiation
  - [x] Quantifier elimination enhancements
  - [x] MBQI performance tuning

- [x] Proof system enhancements (3 items) (planned 2026-04-19)
  - [x] Optimized proof generation (reduce overhead) (planned 2026-04-19)
  - [x] Proof minimization
  - [x] Better theory combination proofs (planned 2026-04-19)
  - **Goal:** (a) bumpalo arena for ProofStep allocation in recorder.rs; (b) structured Nelson–Oppen combination certificate in new theory_combination.rs.
  - **Design:** `oxiz-proof/src/recorder.rs` — steps arena (ArenaIdx<ProofStep>); new `oxiz-proof/src/theory_combination.rs` — NelsonOppenCertificate with interface-equality chain.
  - **Files:** `oxiz-proof/src/recorder.rs`, `oxiz-proof/src/lib.rs`, `oxiz-proof/src/theory_combination.rs` (new), `oxiz-solver/src/combination/coordinator.rs`.
  - **Tests:** arena proof passes checker; new `oxiz-proof/tests/theory_combination_proof.rs` — 3-step EUF+LIA certificate passes ProofChecker.

### Medium Priority: User Experience (Complete)

- [x] Documentation improvements (5 items)
  - [x] Performance tuning guide (docs/PERFORMANCE_TUNING.md)
  - [x] Theory-specific guides (docs/THEORY_GUIDE.md)
  - [x] Common pitfalls and solutions (docs/PITFALLS.md)
  - [x] Migration guide from Z3 (docs/MIGRATION_Z3.md)
  - [x] Case studies and examples (docs/CASE_STUDIES.md)

- [x] API improvements (3 items)
  - [x] EasySolver convenience API (builder pattern)
  - [x] Better error messages (hints, did_you_mean, context_snippet)
  - [x] Timeout and resource limit APIs (ResourceLimits, ResourceMonitor)

- [x] Debugging support (4 items)
  - [x] Solver state visualization (SolverStateSnapshot, DOT graph)
  - [x] Trace generation (TraceEvent, JSON/text output)
  - [x] Better conflict explanations (ConflictExplainer, UnsatExplanation)
  - [x] Model minimization (linear and binary search strategies)

### Low Priority: Ecosystem Integration

- [x] Language bindings — the 2 sanctioned cross-language surfaces (Python + WASM/JS-TS) are both delivered; C/Java FFI bindings were explicitly dropped per the Pure-Rust no-FFI policy
  - [x] Improve Python bindings (oxiz-py enhancements) (planned 2026-04-19)
  - **Goal:** Bring `oxiz-py` to 0.2.1 quality bar: full theory test coverage, README kept in sync with workspace version, parity matrix doc.
  - **Design:** PyO3 surface (1583 LoC, 7 modules, 721-line stub) is mature. Add 5 pytest files for theories implied by stubs but not yet tested. Sync README version strings (`pyproject.toml` needs no sync — it uses `dynamic = ["version"]` via `[tool.maturin]`, reading the version from `Cargo.toml`'s `version.workspace = true` at build time, a more permanent solution than static syncing). Add `PARITY.md` table mapping z3 API → oxiz wrapper → status.
  - **Files:** `oxiz-py/tests/test_quantifiers.py` (new), `oxiz-py/tests/test_arrays.py` (new), `oxiz-py/tests/test_fp.py` (new), `oxiz-py/tests/test_strings.py` (new), `oxiz-py/tests/test_unsat_cores.py` (new), `oxiz-py/PARITY.md` (new), `oxiz-py/pyproject.toml` (no change needed — version is dynamic via `[tool.maturin]`), `oxiz-py/README.md` (version + test-count update); minimal `src/*.rs` patches only if a wrapper is missing.
  - **Tests:** Each pytest file has ≥3 assert cases. Run `cargo build -p oxiz-py --release` (always); `maturin develop + pytest` if toolchain available, else skip with explicit note.
  - **Risk:** maturin unavailable. Mitigation: .py and .md files land regardless; test run is skipped.
  - **Scope cap:** ≤700 LoC net-new. ≤3 new PyO3 wrappers × ≤50 LoC each if needed.
  - [x] JavaScript/TypeScript bindings (via WASM) — **(fixed: js_api is fully wired to oxiz_solver with .d.ts, TS examples, and framework wrappers (React/Vue/Svelte/Deno))**

- [ ] Tool integration (3 items)
  - [x] SMT-COMP 2026 participation — entry package complete; submit when portal opens (~May 2026)
  - [ ] Integration with symbolic execution tools
  - [ ] Integration with verification frameworks

---

## Critical Priority (100% Complete)

### Spacer (PDR) Engine - KEY DIFFERENTIATOR
- [x] Implement Property Directed Reachability for Horn Clauses (CHC)
  - [x] CHC representation (predicates, rules, queries)
  - [x] Frame management (F_0..F_N sequence)
  - [x] POB (Proof Obligation) management
  - [x] Reachability utilities (reach facts, counterexamples, generalization)
  - [x] PDR core algorithm with propagation and blocking
- [x] Loop invariant inference
  - [x] Houdini algorithm for candidate elimination
  - [x] Template-based inference (linear, octagon)
  - [x] SMT-based verification integration
- [x] Software verification pipeline
  - [x] Full CHC solving with invariant synthesis

### Optimization (MaxSMT / OMT)
- [x] MaxSMT core implementation (Fu-Malik with core extraction)
- [x] Core-guided algorithms (OLL with totalizer, MSU3, WMax stratified)
- [x] Totalizer encoding for cardinality constraints
- [x] Optimization Modulo Theories (OMT) - binary/linear/geometric search
- [x] Linear Programming (LP) solver integration
  - [x] Revised simplex method
  - [x] Branch-and-bound for MIP
  - [x] Integer/Binary variable support
- [x] Mixed Integer Programming (MIP) support

### E-Graph Integration
- [x] Tailor e-graph for incremental SMT updates
  - [x] Incremental merge operations
  - [x] Backtrackable union-find
  - [x] Worklist-based congruence closure
- [x] Optimize congruence closure for theory propagation
  - [x] Theory propagator hooks
  - [x] Analysis data per e-class
- [x] Custom e-graph implementation
  - [x] EGraph with EClassId, ENode, EClass abstractions
  - [x] Explanation generation for merges

### Z3 Parity Achievement (v0.2.0)
- [x] String Theory (QF_S) - 100% (10/10)
- [x] Bit-Vector Theory (QF_BV) - 100% (15/15)
- [x] Floating-Point Theory (QF_FP) - 100% (10/10)
- [x] Datatype Theory (QF_DT) - 100% (10/10)
- [x] Array Theory (QF_A) - 100% (10/10)

## High Priority (100% Complete)

### Theory Integration
- [x] Complete CDCL(T) integration with theory propagation
- [x] Implement theory lemma generation
- [x] Add conflict clause minimization
- [x] Implement Nelson-Oppen theory combination
- [x] Difference Logic theory (graph-based, Bellman-Ford)
- [x] UTVPI theory (Unit Two Variable Per Inequality)
- [x] Theory Checking Framework
- [x] Weighted MaxSAT Theory

### SMT-LIB2 Compliance
- [x] Complete parser for all SMT-LIB2 commands
- [x] Add `get-model` output formatting
- [x] Implement `get-unsat-core`
- [x] Add `get-proof` support (placeholder)
- [x] Support for `define-fun` and `define-sort`
- [x] Add `get-assertions`, `get-assignment`, `get-option` commands
- [x] Add `check-sat-assuming` command
- [x] Add `reset-assertions` command
- [x] Add `simplify` command (Z3 extension)

### Performance
- [x] Add restart strategies (Luby, geometric)
- [x] Implement phase saving
- [x] Implement clause deletion strategies
- [x] Add learned clause minimization
- [x] Profile and optimize hot paths

## Medium Priority (100% Complete)

### New Theories
- [x] Array theory solver (extensionality, select/store)
- [x] String theory solver (word equations, regex via Brzozowski derivatives)
- [x] Floating-point theory (IEEE 754, QF_FP) with bit-blasting
- [x] Datatype theory (ADTs - lists, trees)
- [x] Non-linear arithmetic (QF_NRA) - CAD projection, Sturm sequences
- [x] Pseudo-Boolean theory (PbSolver)
- [x] Recursive Functions theory (RecFunSolver)
- [x] User Propagators (UserPropagatorManager)
- [x] Special Relations (LO, PO, PLO, TO, TC)

### Tactics System
- [x] `simplify` - Algebraic simplification (x + 0 -> x)
- [x] `propagate-values` - Constant propagation
- [x] `bit-blast` - Convert BitVectors to SAT clauses (detection phase)
- [x] `ackermannize` - Eliminate functions by adding constraints
- [x] `ctx-solver-simplify` - Context-dependent simplification
- [x] Tactic pipeline/composition system (ThenTactic, OrElseTactic, RepeatTactic)
- [x] Probe system (11+ probes)
- [x] Fourier-Motzkin elimination
- [x] NNF/CNF conversion tactics
- [x] Model-Based Projection (MBP)
- [x] Quantifier tactics (MBQI, E-matching, DER, Skolemization)

### Parallelization - BEYOND Z3: Native Multi-core
- [x] Parallel portfolio solving (competing tactics on threads)
- [x] Cube-and-conquer for hard instances
  - [x] CubeGenerator, ParallelCubeSolver, CubeAndConquer
  - [x] 22 tests passing
- [x] Work-stealing clause sharing
- [x] Native async/parallel infrastructure (Rayon/Tokio)

### Proof Generation - BEYOND Z3: Machine-Checkable
- [x] DRAT proof output for SAT core (text and binary formats)
- [x] Theory proof generation (EUF, Arith, Array recorders)
- [x] Machine Checkable Proofs (Alethe format) - Beyond Z3!
- [x] LFSC proof format (Logical Framework with Side Conditions)
- [x] Proof checking infrastructure (syntactic + rule validation)
- [x] **Coq/Lean/Isabelle exports** - Unprecedented in SMT solvers!
- [x] Craig Interpolation
  - [x] McMillan's algorithm (left-biased interpolants)
  - [x] Pudlak's algorithm (symmetric interpolation)
  - [x] Huang's algorithm (right-biased interpolants)
  - [x] Theory-specific interpolants (LIA, EUF, Arrays)
  - [x] Sequence and tree interpolation

### Advanced Features
- [x] Minimal Unsat Cores with parallel reduction
- [x] Craig Interpolation for model checking
- [x] XOR/Gaussian elimination solver
- [x] Quantifier Elimination (QE) enhancements
  - [x] Term graph analysis
  - [x] QE Lite for fast approximation
  - [x] Model-based interpolation (MBI)
- [x] Model subsystem
  - [x] Model evaluator with caching
  - [x] Model completion
  - [x] Prime implicant extraction
  - [x] Value factories

## Low Priority (100% Complete)

### Tooling
- [x] SMT-COMP benchmark suite (oxiz-smtcomp crate)
- [x] Fuzzing infrastructure (fuzz/)
- [x] Python bindings (oxiz-py crate)
- [x] Performance regression tests (bench/regression/)
- [x] Z3 parameter/tactics extraction scripts

### Documentation
- [x] API documentation improvements
- [x] Architecture guide (docs/ARCHITECTURE.md)
- [x] Tutorial for extending theories (docs/TUTORIAL_CUSTOM_THEORY.md)
- [x] Contribution guidelines (CONTRIBUTING.md)

### Future Features (Complete)

#### IDE and Tooling
- [x] VS Code Extension (oxiz-vscode/)
- [x] REST API Server Mode (oxiz-cli --server)
- [x] Web Dashboard (oxiz-cli --dashboard)

#### Advanced CLI Features
- [x] TPTP Format Support (oxiz-cli/src/tptp.rs)
- [x] Interpolant Generation CLI
- [x] Distributed Solving (oxiz-cli/src/distributed.rs)
- [x] SMT-LIB 2.6 Features (oxiz-core)

---

## Cross-Crate Dependencies

```
oxiz-core (foundation)
    |
    +-- oxiz-math (polynomial, simplex, intervals, LP)
    |       |
    |       +-- oxiz-nlsat (CAD, NIA)
    |
    +-- oxiz-sat (CDCL, XOR)
    |       |
    |       +-- oxiz-proof (DRAT, Craig interpolation)
    |       +-- oxiz-opt (MaxSAT core)
    |
    +-- oxiz-theories (EUF, LRA, BV, Arrays, Strings, FP, DL, UTVPI)
            |
            +-- oxiz-solver (CDCL(T) orchestration)
                    |
                    +-- oxiz-spacer (PDR/CHC, invariants)
                    +-- oxiz-opt (OMT)
                    +-- oxiz-wasm / oxiz-cli (frontends)
```

---

## Roadmap

### v0.1.3 - COMPLETE (Feb 5, 2026)
- **100% Z3 Parity** across 8 core SMT-LIB logics
- Production-ready solver
- All theory solvers validated

### v0.2.0 - COMPLETE (Feb 6 - Mar 31, 2026)
- **168/168 Z3 parity tests**
- Performance optimization phase 1 (allocators, SIMD, caches)
- EasySolver API, error messages, resource limits
- Debugging: visualization, traces, conflict explanations, model minimization
- Documentation: 5 new guides (performance, theory, migration, pitfalls, case studies)
- 6,155 tests (16 skipped, 0 failures), 393,292 total Rust lines (312,495 code), 931 files, 0 clippy warnings

### v0.3.0 (Target: June 2026)
**Focus: Performance Parity and SMT-COMP**
- [~] Performance parity with Z3 (within 1.2x) (planned 2026-04-19)
  <!-- umbrella stays [~] until EP-6e (empirical geomean check) lands; children EP-6a..d may already be [x] -->
  - [x] EP-6a: Extended `Z3ComparisonReport` with `geomean_ratio`, `p50_ratio`, `p95_ratio`, `ratio_count` fields (`#[serde(default)]`); `within_target` recomputed from geomean ≤ 1.2 (not strict per-benchmark); 5 unit tests in `z3_compare.rs` (planned 2026-04-19)
  - [x] EP-6b: `bench/z3_parity` gains `--export-history <dir>` mode writing versioned `history/<YYYY-MM-DD>_<sha>.json` snapshots with per-logic `RatioSummary` breakdown; 6 tests in `bench/z3_parity/tests/history_export.rs` (planned 2026-04-19)
  - [x] EP-6c: `bench/regression/baseline.json` refreshed from v0.2.1 current-branch measurements (was v0.1.3 from Jan 2026, 3 months stale) (planned 2026-04-19)
  - [x] EP-6d: `.github/workflows/perf-regression.yml` extended with `geomean-gate` step — soft-gate (passes when no Z3 data, exits non-zero when `geomean_ratio > 1.2`) (planned 2026-04-19)
  - [ ] EP-6e: Empirical verification — confirm geomean ≤ 1.2 across QF_* logics with Z3 installed (deferred: requires Z3-equipped machine; run next /ultra pass with Z3 available)
- [x] Quantified logic support (UFLIA, UFLRA, AUFLIA)
- [x] Combined theory validation (QF_AUFBV, QF_ALIA, QF_ABV)
- [x] Enhanced preprocessing tactics (planned 2026-04-19)
- [x] Performance regression CI pipeline
- [x] SMT-COMP 2026 entry preparation (completed 2026-05-05)
  - [x] `Track` enum (5 variants: SingleQuery, Incremental, UnsatCore, ModelValidation, ProofExhibition)
  - [x] `submission` module wired into `oxiz-smtcomp/src/lib.rs` with full public API
  - [x] `default_oxiz_2026()` fixed: `bin/smtcomp2026` binary, version from `CARGO_PKG_VERSION`
  - [x] Per-track `starexec_run_<track>` scripts in submission package
  - [x] `smtcomp2026` binary extended with `--track` flag (single|incremental|unsat-core|model|proof)
  - [x] `scripts/package_smtcomp.sh` — assembles complete StarExec ZIP
  - [x] End-to-end submission tests in `oxiz-smtcomp/tests/submission_e2e.rs`

### v1.0.0 (Target: Q4 2026)
**Focus: Production Release**
- [ ] Full Z3 API compatibility
- [ ] Performance at or better than Z3
- [ ] Comprehensive documentation
- [ ] Stable API guarantees
- [ ] Industry adoption ready

---

## Recent Achievements

### 2026-06-09 - v0.2.3 Release

- **oxiz-sat**: `DratWriter<W>` / `LratWriter<W>` generic over any `W: Write + Send`; breaking rename from `DratProof` / `LratProof`
- **oxiz-nlsat**: Real resultant (Sylvester/Bareiss), leading-coefficient extraction, degree≥3 root isolation (Descartes/Sturm), monotonicity estimation
- **oxiz-theories**: Sound Nelson-Oppen equality propagation; simplex `optimize_linexpr`; correct push/pop tableau snapshots
- **oxiz-opt**: Full solver-backed `check_sat`, MaxSMT selector encoding, `optimize_single_objective`/`optimize_pareto` delegation
- **oxiz-spacer**: Real BMC formula construction; sound k-induction; dual-arena soundness fix; `extract_model` via `eval_in_model`
- **oxiz-solver**: New `Context::eval_in_model` for model-based term evaluation

### 2026-06-01 - v0.2.2 Release

- **Recursive BV term encoding**: Full nested bit-vector expression encoding in `BvSolver` with structured conflict diagnostics
- **Z3 API compatibility layer**: `TacticRegistry` (19 named tactics), `FuncInterp` / `FuncEntry` in EUF, `Z3SortKind` / `Z3Sort`, `substitute` (BV+Array+Apply coverage), `Z3Pattern` + quantifier pattern APIs
- **Real LBD scoring**: `compute_lbd_from_literals` replaces stub — CDCL now uses genuine Literal Block Distance from finalized 1-UIP learned clauses
- **ML conflict hook**: `BranchingHeuristic::on_conflict_var` defaulted hook wired to `MLBranchingHeuristic` via `MLEnhancedVSIDS::update_conflict`
- **LRU caches**: `AggressiveSimplifier` memo cache (4 096 cap), `EufSolver` explanation cache (1 024 cap), theory combiner lemma cache (bounded to `max_lemma_cache_size`)
- **CLI peak memory**: Linux `VmHWM` high-water-mark now reported correctly
- **Big-M primal simplex**: `SimplexSolver` gains Big-M phase-1 for LP feasibility
- **Dead code policy**: Module-level `#![allow(dead_code)]` removed from 40+ modules; `algebraic_number.rs` (446 lines) deleted
- **Tests**: 6,735 passing (16 skipped, 0 failures); 0 clippy warnings
- **SLoC**: ~419,576 code lines across ~1,012 Rust files

### May 18, 2026 - TacticRegistry Wired, Real LBD, EUF FuncInterp, Z3 Sort/Subst/Patterns (v0.2.2 Pass 6)

- **TacticRegistry wired into Z3 compat**: `z3_compat_ext2.rs::apply_named_tactic` now delegates to `oxiz_core::tactic::default_registry()` via a `OnceLock`-cached static; reachable tactic surface grew from 5 to 19 named tactics (adds aggressive-simplify, bvarray2uf, elim-uncnstr, solve-eqs, nnf, tseitin-cnf, fm, arith-bounds, factor, pb2bv, lia2card, nla2bv, split, ctx-solver-simplify canonical name + ctx-simplify backward-compat alias)
- **Real LBD from learned clause**: `compute_lbd_from_literals` replaces the `vars_to_bump`-based proxy; hook now fires AFTER `self.learnt` is finalized and minimized in both `analyze()` and `analyze_theory_conflict()`, passing the distinct-nonzero-level count of the actual 1-UIP learned clause to `MLBranchingHeuristic::on_conflict_var_with_lbd`. Old `compute_lbd_from_vars` deleted (no dead code).
- **FuncInterp EUF congruence traversal**: `EufSolver::function_application_entries(func_id)` returns canonicalized (arg_reps, result_rep) per Apply node using `find()` without path compression; `Context::get_func_interp_raw` consumes these, dedups by arg-rep tuple, resolves values via class-membership lookup, picks most-common entry as `else_value`. `Solver` gains `pub(crate) euf_function_entries()` bridge. Replaces the Pass 5 partial Apply-walk implementation with full congruence-aware extraction.
- **Z3 Sort introspection + term substitution + quantifier patterns**: new `oxiz-solver/src/z3_compat_ext3.rs` (673 LOC) — `Z3SortKind`/`Z3Sort` (`kind`/`bv_size`/`array_domain`/`array_range`/`name`), `Z3Context::substitute` (hand-rolled bottom-up rebuild covering Bool/Arith/BV/Array/Apply/ITE, memoized via `FxHashMap` — wider coverage than core `TermManager::substitute` which silently skips BV+Apply), `Z3Pattern` + `forall_with_patterns`/`exists_with_patterns` (delegating to `TermManager::mk_*_with_patterns`)
- **Tests**: +32 new tests (6,802 → 6,834); 0 failures; 0 clippy warnings
- **New files**: `oxiz-solver/src/z3_compat_ext3.rs`, `oxiz-solver/tests/z3_compat_extensions3.rs`, `oxiz-solver/tests/func_interp_euf.rs`

### May 18, 2026 - FuncInterp, TacticRegistry, Real LBD, LRU Caches (v0.2.2 Pass 5)

- **FuncInterp (model function interpretations)**: `FuncEntry`/`FuncInterp` types in `oxiz-core/src/model/mod.rs` (entries table + else_value + arity, with `evaluate`); `Model::add_func_interp`/`get_func_interp`; `Z3FuncInterp`/`Z3FuncEntry`/`Z3Value` wrappers in `z3_compat_ext2.rs`; `Z3Model::get_func_interp(&FuncDecl)` delegates to `Context::get_func_interp_raw()` which walks `Apply` terms in the model; 15 new tests
- **TacticRegistry**: `oxiz-core/src/tactic/registry.rs` (333 LOC) with `default_registry()` registering 19 named tactics (simplify, propagate-values, ctx-solver-simplify, aggressive-simplify, bit-blast, bvarray2uf, ackermannize, elim-uncnstr, solve-eqs, nnf, tseitin-cnf, fm, arith-bounds, factor, pb2bv, lia2card, nla2bv, split, skip); `create(name)`/`names()`/`contains()`; 11 new tests
- **Real LBD (Literals per Block Distance)**: `compute_lbd_from_vars()` in `conflict.rs` computes glue score from conflict-involved variables' distinct decision levels; new `BranchingHeuristic::on_conflict_var_with_lbd` defaulted trait method (delegates to `on_conflict_var` for backward compat); `MLBranchingHeuristic` forwards real LBD to `MLEnhancedVSIDS::update_conflict`; 7 new tests
- **LRU caches in EUF + simplification**: `oxiz-core/src/lru_cache.rs` (copy for oxiz-core, no circular dep); `AggressiveSimplifier` gains persistent `memo_cache: LruCache<TermId,TermId>` (4096 cap) replacing per-call HashMap; `EufSolver` gains `expl_cache: LruCache<(u32,u32),Vec<TermId>>` (1024 cap, canonical min/max key, cleared on merge/pop/reset); 6 new tests
- **Tests**: +39 new tests (6,763 → 6,802); 0 failures; 0 clippy warnings
- **New files**: `oxiz-core/src/lru_cache.rs`, `oxiz-core/src/tactic/registry.rs`

### May 18, 2026 - Z3 Compat #2, CLI Peak Memory, ML Conflict Hook, LRU Lemma Cache (v0.2.2 Pass 4)

- **Z3 API compatibility expanded #2**: `oxiz-solver/src/z3_compat_ext2.rs` (963 LOC) adds `Z3Statistics` (7 counters: decisions/propagations/conflicts/restarts/learned-clauses/theory-propagations/theory-conflicts), `Z3Params` (key→value dispatcher into `SolverConfig`), `Z3Probe` (registry over 7 probe types with `.lt()`/`.gt()` combinators), `Z3Goal`/`Z3Tactic`/`Z3ApplyResult` (named-tactic dispatch + `.then()`/`.or_else()`/`.repeat()`/`.try_for()` combinators), `Z3DatatypeSort`/`Z3Constructor` (full `DatatypeDecl` wiring), `Z3Solver::check_assumptions(&[Bool])`/`unsat_core()`, `Z3AstVector`; 41 integration tests in `z3_compat_extensions2.rs`
- **CLI peak memory fixed**: `peak_memory_bytes` was always `current_rss` — now reads Linux `VmHWM:` from `/proc/self/status` (kernel high-water-mark); new `oxiz-cli/src/memory.rs` (92 LOC) with `rss_and_peak()` function; non-Linux falls back gracefully
- **CLI test coverage**: 9 new integration tests — peak memory nonzero, peak ≥ current, Linux VmHWM, parallel-mode, multi-file memory, exit codes for SAT/UNSAT/parse-error/missing-file
- **`BranchingHeuristic::on_conflict_var` hook**: new defaulted method (no-op default, full backward compat); called from `conflict.rs` both `bump_batch` sites; `MLBranchingHeuristic::on_conflict_var` forwards to `MLEnhancedVSIDS::update_conflict(var, level as f64)`, enabling real ML training signal; 3 tests
- **`LruCache<TheoryLemma>` in theory combination**: `FxHashSet<TheoryLemma>` (unbounded) replaced by `LruCache<TheoryLemma, ()>`; `config.max_lemma_cache_size` (default 10,000) finally enforced; push/pop backtracking uses `truncate_to(n)`; `CombinerStats` gains `lemma_cache_hits/misses/evictions`; 5 tests
- **Tests**: +60 new tests (6,703 → 6,763); 0 failures; 0 clippy warnings
- **New files**: `oxiz-solver/src/z3_compat_ext2.rs`, `oxiz-solver/tests/z3_compat_extensions2.rs`, `oxiz-cli/src/memory.rs`

### May 18, 2026 - Dead Code Policy Enforcement Across 40 Modules (v0.2.2 Pass 3 cont)

- **Crate-level allow removed**: `oxiz-solver/src/lib.rs` `#![allow(dead_code)]` deleted — the highest-priority policy violation, was silencing all dead code warnings for the entire solver crate
- **39 module-level allows removed** across `oxiz-solver` (15 modules), `oxiz-core` (5 tactic modules), `oxiz-math` (4 modules), `oxiz-theories` (3 modules), `oxiz-proof` (1), `oxiz-cli` (2): all converted to per-item `#[allow(dead_code)]` or eliminated by wiring/deleting dead code
- **`algebraic_number.rs` deleted** (446 lines): zero external callers confirmed; duplicates `realclosure.rs` functionality; removed from `oxiz-math/src/lib.rs`
- **`SyzygyComputer` wired into `buchberger.rs`**: `apply_buchberger_criteria` now called before each S-polynomial computation to skip S-pairs failing GCD or chain criterion — improves Gröbner basis computation efficiency
- **`cicd.rs` activated**: `CicdReport` wired into `processor.rs` `run_files` with `--cicd-report`/`--cicd-strict` CLI flags
- **Tests**: 6,703 passing (−4 vs prior count due to test consolidation); 0 failures; 0 clippy warnings
- **Net LoC**: −357 net (492 deleted, 135 added) from dead code removal

### May 18, 2026 - Z3 Compat Expansion + LIA Heuristics + Dead Code Fixes (v0.2.2 Pass 3)

- **Z3 API compatibility expanded**: `oxiz-solver/src/z3_compat_ext.rs` (746 lines) adds `Array` type (select/store/eq), `FuncDecl` (declaration + application), quantifiers (`forall_bool`/`exists_bool`), `ite` (Bool/Int/Real/BV), `distinct` (Int/Real/BV), `Real` symmetry (`gt`/`ge`/`neg`/`div`/`from_i64`), `Z3Optimize` wrapper around `OmtSolver`; 23 new integration tests in `z3_compat_extensions.rs`
- **LIA heuristics wired into B&B loop**: `feasibility_pump`, `probe_variables`, `manage_cuts` — all previously dead code with `#[allow(dead_code)]` — are now called from `LiaSolver::check()` (probe + pump before B&B) and `branch_and_bound()` (manage_cuts every 8 levels); 4 new integration tests in `tests/lia_heuristics_integration.rs`
- **`simplex_solver.rs` policy fix**: removed module-level `#![allow(dead_code)]` + deleted unused `solve_with_rhs_perturbation`; added `test_all_accessors` test activating all 10 public accessors; simplex_parametric.rs also cleaned of module-level allows
- **LRA #6 regression guard verified**: `lra_regression_issue6.rs` (3 tests) all pass — bound-conflict detection for `x ≤ -1` + `x = -0.25` → UNSAT is correct in the current pipeline
- **Tests**: +78 new tests (6,629 → 6,707); 0 failures; 0 clippy warnings
- **New files**: `oxiz-solver/src/z3_compat_ext.rs`, `oxiz-solver/tests/z3_compat_extensions.rs`, `oxiz-theories/tests/lia_heuristics_integration.rs`

### May 5, 2026 - ML Wiring + Dead Code Cleanup + Bench Calibration (v0.2.2 Pass 2)

- **`MLBranchingHeuristic` adapter**: `oxiz-ml/src/branching/sat_adapter.rs` — `MLEnhancedVSIDS` now implements `BranchingHeuristic` via a thin adapter; ML branching is end-to-end reachable through `SolverConfig::external_branching` → `pick_branch_var`; type bridge `Var(u32) ↔ VarId(usize)` is lossless; confidence gate allows ML deference to VSIDS
- **Dead code removed**: `oxiz-proof/src/transform.rs` (587 lines) and `oxiz-proof/src/compression.rs` (580 lines) deleted — both referenced non-existent `ProofRule` type; live equivalents in `compress.rs`/`simplify.rs`/`normalize.rs`/`merge.rs` cover the same surface; TODO comment in `lib.rs:84-86` removed
- **Bench baselines calibrated**: `bv_simple` = 3,916 µs, `lra_simple` = 380 µs, `arrays_simple` = 440 µs (measured on host); BV/LRA/Arrays regression gate is now functional with ±25% envelope
- **Pre-existing websocket doctest fixed**: `tokio_test::block_on` → `tokio::runtime::Runtime::new().unwrap().block_on` (tokio is already a dev-dep); unblocks `--all-features` doctest runs
- **Tests**: +6 new `oxiz-ml/tests/sat_integration.rs` tests; 6,629 total passing; 0 failures; 0 clippy warnings
- **New files**: `oxiz-ml/src/branching/sat_adapter.rs`, `oxiz-ml/tests/sat_integration.rs`
- **Deleted files**: `oxiz-proof/src/transform.rs`, `oxiz-proof/src/compression.rs` (−1,167 lines)

### May 5, 2026 - v0.3.0 Infrastructure Push (v0.2.2)

- **SMT-COMP 2026 entry complete**: `Track` enum (5 variants), per-track `starexec_run_*` scripts, `smtcomp2026 --track` flag, `scripts/package_smtcomp.sh` packaging script; `submission` module wired into public API
- **Bench regression expanded**: BV, LRA, Arrays fixture benchmarks wired into criterion (`bench_bv`, `bench_lra`, `bench_arrays`); `src/fixtures.rs` for stable `include_str!` embedding; `tests/bench_coverage.rs` smoke tests
- **`BranchingHeuristic` trait hook**: new `oxiz-sat::BranchingHeuristic` trait + `BoxedBranchingHeuristic` type alias; optional `external_branching` field on `SolverConfig`; hook in `pick_branch_var` — forward-compat for oxiz-ml integration (v0.4.0)
- **Tests**: +21 new tests across three tracks (9 external_branching, 9 submission e2e, 3 bench coverage); 0 regressions; 0 clippy warnings
- **New files**: `oxiz-sat/src/solver/heuristic.rs`, `oxiz-sat/tests/external_branching.rs`, `oxiz-smtcomp/tests/submission_e2e.rs`, `bench/regression/src/fixtures.rs`, `bench/regression/tests/bench_coverage.rs`, `scripts/package_smtcomp.sh`

### April 25, 2026 - Statistics Update (v0.2.1)

- **Code Lines (tokei)**: 408,320 code lines out of 442,034 total lines across 1,182 files
- **Tests**: 6,415 passing (0 failures)
- **Stubs**: 0 unimplemented!()/todo!() remaining
- **Key additions**: Set theory CDCL(T) interface wired; Sylvester matrix discriminant (degree≥4 fix); Hong's projection leading-coefficient fix; NIA cutting planes re-enabled; normalize_bounds tactic enabled; PyO3 quantifier/string(13)/FP(21) wrappers; dynamic subsumption periodic_check; multi-trigger E-matching; clause learning literal minimization; branch-and-bound loop; BV signed comparison

### April 24, 2026 - Statistics Update (v0.2.1)

- **Rust Files**: 931 -> 978
- **Code Lines (tokei)**: 323,732 code lines out of 406,502 total Rust lines
- **Tests**: 6,368 passing (16 skipped, 0 failures)
- **Workspace Crates**: 17 (16 Rust crates + 1 TypeScript)
- **EUF Performance**: 5 allocation-reduction and data-layout improvements landed (fingerprint pre-filter, inline hints, incremental sig_table undo trail, ENode layout reorder, reusable canonicalize buffer)

### April 4, 2026 - Statistics Update

- **Rust Files**: 911+ -> 931
- **Code Lines (tokei)**: 312,495 code lines out of 393,292 total Rust lines
- **Tests**: 6,155 passing (16 skipped, 0 failures)
- **todo!/unimplemented! macros**: 0 across all 15 Rust crates
- **Workspace Crates**: 16 (15 Rust + 1 TypeScript)

### March 31, 2026 - Performance, UX, Debugging, Docs
- **Performance**: 9 optimizations (arena allocator, clause pool, SIMD poly ops,
  TermKindHasher, FP cache, model gen cache, parallel theory checking,
  lock-free structures, lazy evaluation)
- **User Experience**: EasySolver API, better error messages, resource limits
- **Debugging**: State visualization, trace generation, conflict explanations,
  model minimization
- **Documentation**: 5 new guides (performance tuning, theory, Z3 migration,
  pitfalls, case studies)
- **File Maintenance**: solve_eqs.rs and rational.rs re-split under 2000 lines
- **Tests**: 6,122 -> 6,155 (+33 new)
- **LoC**: 392,274 -> 393,292 (+1,018)

### v0.3.0 Milestone (March 23, 2026)
- 168/168 Z3 parity tests passing
- 5,993 tests at milestone point
- All files under 2,000 lines

### 100% Z3 Parity (Feb 5, 2026)
- 88/88 benchmark tests across 8 core SMT-LIB logics
- Fixed 31 test failures across 5 theory solvers
- 18 infrastructure issues resolved, 13 algorithmic improvements

---

## Next Immediate Actions

Refreshed for v0.3.3 (2026-08-26). The v0.3.0-era entries (hot-path profiling, performance-regression infrastructure, extended theory coverage, SMT-COMP entry preparation, Python/WASM bindings) are all delivered — see the checked items under "Post-Parity Priorities" and the Roadmap below.

1. **v0.3.2 backlog — deep-recursion test OOM investigation** (see "v0.3.2 backlog (added 2026-07-31, from the deep-recursion test OOM investigation)")

2. **Empirical performance-parity verification** (EP-6e) — run in 0.3.3, no longer hardware-gated: Z3 4.15.4 is installed and the `--export-history` / `geomean-gate` harness executed end-to-end. The result is a methodology finding, not a ≤1.2x measurement: `run_oxiz` is an in-process call while `run_z3` spawns a subprocess (`find_z3` probes with `z3 --version` *inside* the timed call), so every recorded `z3_time` charges at least one full process spawn on top of the solve — the fastest `z3_time` in the current snapshot is 7.4ms (spawn cost) against a fastest `oxiz_time` of 0.196ms (a function call). Per this repo's standing rule (README, "Performance"), no solver-vs-solver speed ratio is published from that data. What remains open is deciding how to measure this meaningfully (e.g. an in-process Z3 binding, or comparing against a fixed per-call baseline), not re-running the existing harness.

3. **JIT-style specialization for hot theory operations** (v0.4.0)
   - Requires an IR + codegen layer; the only remaining pending item in the performance track

4. **Remaining frontend/feature gaps** (see "Remaining (post-0.3.0 hardening)") — recursive functions end-to-end, NLSAT algebraic-number witnesses, and `mk_bv_concat`'s release-build width default all landed in 0.3.3 (see CHANGELOG.md). One gap remains:
   - `RegLan` as a first-class `SortKind` variant — still honestly rejected in nullary-declaration position; 0.3.3 only corrected the parser's rejection *message* (it no longer claims the sublanguage "is not yet implemented" — `RegLan` is reserved because `TermManager` interns regex terms at a built-in `Uninterpreted("RegLan")` sort, not because the operators are missing), matching the accuracy bar `RoundingMode`'s own message now meets after becoming first-class.

5. **Ecosystem Growth**
   - Integration with symbolic-execution tools and verification frameworks (re-scope once a specific target is chosen)
   - SMT-COMP 2026 submission once the portal opens

---

**Status**: Production Ready
**Current Version**: v0.3.3 (2026-08-26)
**Tests**: 9,953 passing (all-features, 8 skipped) + 110 doc-tests | **LoC**: 451,853 code (564,303 total) | **Files**: 1,276 | **Clippy**: 0 warnings
**Z3 Parity**: 170/170 Correct on the extended 19-logic differential suite (0 Wrong / 0 Inconclusive / 0 Timeout / 0 Error, honest comparator vs z3 4.15.4)
**Next Milestone**: v0.4.0 - JIT specialization, `recfun` support, and the remaining completeness gaps (see "Remaining (post-0.3.0 hardening)")
**Long-term Goal**: v1.0.0 - Industry-Ready SMT Solver (Target: Q4 2026)

---

## Proposed follow-ups

- **JIT-style specialization** (root TODO.md:169) — defer to v0.4.0 (oversized: requires IR + codegen layer).
- **SMT-COMP 2026 participation** (root TODO.md:306) — gated on SMT-COMP submission portal (opens ~May 2026).
- **Symbolic execution tool integration** (root TODO.md:307) — vague; re-scope after user selects target (KLEE/angr/S2E).
- **Verification framework integration** (root TODO.md:308) — vague; re-scope after user selects target (Frama-C/CBMC/SeaHorn).

## v0.3.2 backlog (added 2026-07-31, from the deep-recursion test OOM investigation)

A full `cargo nextest run --all-features` on a 14.6 GB developer machine was terminated by the kernel OOM killer, twice, on 2026-07-31. `journalctl` recorded a single test process at `total-vm:32791524kB` / `anon-rss:6386840kB`, the enclosing terminal cgroup peaking at 12.7 GB memory plus 37.7 GB swap, and the scope dying with `Failed with result 'oom-kill'`; the kernel named `walk::tests::an` — truncated `walk::tests::any_node_finds_var_at_extreme_depth` — as the allocating thread. This was neither a per-test timeout nor an assertion failure: the 15 tests nextest reported as SIGTERM/SIGKILL were casualties of the OOM cascade, not independent failures. A same-day fix scaled thread stack sizes and constructed depths together by 8x across `oxiz-spacer`, `oxiz-solver`, `oxiz-proof`, `oxiz-theories`, and `oxiz-wasm` (71 files, ~120 tests, test code only — no production code changed); those five crates now run 4487 tests in 15.3s with zero SLOW markers. The items below are what that fix did **not** address.

- [ ] **Test-validity: the `mk_and`/`mk_or` deep-nesting tests never built a deep term** — `oxiz-core/src/ast/manager/builder.rs:94` (`mk_and`) and `:121` (`mk_or`) flatten a nested `And`/`Or` child into the parent (`TermKind::And(inner) => flat_args.extend(inner.iter().copied())`). The accumulate idiom used throughout the deep-nesting regression tests — `acc = manager.mk_and([acc, lit])` in a loop — therefore produces a flat n-ary node of depth 2, not an n-deep tree. Those tests claim to pin that a walk is iterative, but a recursive walk would not overflow on them either, so they pin nothing about stack depth. Verified by reading the builder. Affected: `oxiz-spacer`'s `walk.rs` (`any_node`/`flatten_conjuncts`), `smt.rs`, `invariant.rs`, `theory.rs`, `translate.rs`, `existential.rs` (`syntactic_projection`), and `oxiz-solver/src/solver/theory_bv_encode.rs`. Unaffected because their builder does not flatten, i.e. genuinely deep: anything built with `mk_add` (`oxiz-core/src/ast/manager/builder.rs:286`), the `sort_name.rs` string-nesting tests, `oxiz-proof`'s `deep_chain`, the `oxiz-theories` e-matching pattern nesting, and the `oxiz-wasm` dependency chain. Fix: build the deep term through a path that does not flatten (intern `TermKind::And` directly, or alternate the accumulator through a non-`And` wrapper), and keep a separate flat n-ary test for the wide case — wide and deep are different regressions.
  - **Priority:** P1  **Scope:** medium
- [ ] **Production: accumulating with `mk_and`/`mk_or` in a loop is O(n^2)** — the same flattening (`oxiz-core/src/ast/manager/builder.rs:94`, `:121`), production consequence. Each iteration rebuilds an i-element `SmallVec` and re-interns it, so a loop accumulating n conjuncts costs Theta(n^2) element copies plus Theta(n^2) hashing. Any caller that accumulates conjuncts or disjuncts in a loop pays it — CNF construction, lemma accumulation, MBQI instantiation, spacer cube building. Fix: collect into a single `Vec`/`SmallVec` and call `mk_and` once, or add an n-ary accumulate API to `TermManager` so the flatten step is not repeated per element.
  - **Priority:** P1  **Scope:** medium
- [ ] **Production: `intern` retains each `TermKind` three times under `--all-features`** — `oxiz-core/src/ast/manager/mod.rs:106`. The kind is cloned into `self.terms`, stored again as the `self.cache` key, and cloned a third time into the bumpalo arena when the non-default `arena` feature is on (`oxiz-core/Cargo.toml:60`). bumpalo never frees, so the arena copy is permanent for the manager's lifetime. `--all-features` enables `arena`, and that is what the `/all`, `/fail`, and `/test-all` skills use, so every all-features run pays 3x term memory. Fix: store the kind once and key the cache by hash into the `terms` slot (raw-entry hashbrown table) instead of holding a second owned copy, and re-evaluate whether the arena copy earns its keep or should be mutually exclusive with the `terms` `Vec`.
  - **Priority:** P1  **Scope:** large
- [ ] **Production: `ProofVisualizer`'s JSON format holds Theta(depth^2) live heap** — `oxiz-proof/src/visualization.rs`, `write_json_node`. The closing `]` and `}` literals are materialized eagerly with the full indent baked in and pushed onto the work stack *below* the child frame, so at depth d the stack holds two O(d)-length `String`s per ancestor level. Estimated ~14.4 GB live at depth 60,000, in ~120,000 allocations — this is live heap regardless of the sink, so it OOMs even when streaming to `/dev/null`. This is the format that drove the 32 GB test process. Fix: store `(indent_level, DelimiterKind)` in `JsonFrame::Literal` and render the indent at pop time; output stays byte-identical and live heap drops to Theta(depth).
  - **Priority:** P1  **Scope:** small
- [ ] **Production: `IndentedText` and `AsciiTree` are Theta(depth^2) in output size** — `oxiz-proof/src/visualization.rs:277` (`"  ".repeat(current_indent)`) and `:223`-`:227` (prefix grown by `format!` per level, re-emitted per line). Unlike the JSON case this is output volume only, not live heap, so a caller writing to a file or a pipe never holds it. Inherent to indent-by-depth rendering; only fixable by capping the indent, which changes the rendered format. Decide whether to cap or to document the cost; do not silently change the format.
  - **Priority:** P3  **Scope:** small
- [ ] **Test infra: nextest has no `slow-timeout`, so a runaway test can only be stopped by the OOM killer** — `.config/nextest.toml` defines a `test-groups` serial group for the `oxiz-cli` subprocess tests but sets no `slow-timeout`. nextest's default reports SLOW at 60s and never terminates, so an unbounded allocation runs until the machine dies. Fix: add `slow-timeout = { period = "60s", terminate-after = 3 }` to `profile.default` so a runaway is reported as TIMEOUT and killed, and consider a tighter `profile.ci`.
  - **Priority:** P2  **Scope:** small
- [ ] **Test infra: document a memory-capped way to run the suite** — nothing stops a local full-suite run from taking down the developer's desktop session. Add a `CONTRIBUTING.md` recipe — on Linux, `systemd-run --user --scope --collect -p MemoryMax=10G -q -- cargo nextest run ...` confines a runaway to its own cgroup — and record the build/test parallelism caps (`-j 6`, `--test-threads 8`) that were needed on a 16-thread / 14.6 GB machine.
  - **Priority:** P2  **Scope:** small
- [ ] **Remaining deep-depth test sites not yet swept** — the 2026-07-31 pass covered only `oxiz-spacer`, `oxiz-solver`, `oxiz-proof`, `oxiz-theories`, and `oxiz-wasm`. Not swept, with approximate counts of large-depth literals (`50_000` / `60_000` / `100_000` / `200_000`): `oxiz-core` 74, `oxiz-math` 16, `oxiz-sat` 10, `oxiz-opt` 6, `oxiz-nlsat` 5, `oxiz-ml` 3. Apply the same rule: scale the thread stack size and the constructed depth by the same factor so the bytes-per-frame threshold is preserved.
  - **Priority:** P2  **Scope:** medium
- [ ] **`oxiz-solver/src/solver/encode/tests.rs:1058` builds a 100,000-deep chain with no paired stack** — `check_sat_only_respects_false_and_truncation_flags` calls `build_implies_chain(&mut manager, 100_000)` but spawns no small-stack thread, so there is no stack to scale the depth against and the 2026-07-31 pass deliberately left it. It only needs depth > `ENCODE_DEPTH_LIMIT` (512) to do its job. Drop it to a few thousand to lower the crate's test memory floor.
  - **Priority:** P3  **Scope:** small
- [ ] **Record the 16 thread-stack sites deliberately left at a 1 MiB stack** — so a future sweep does not "fix" them into failing. The 16 break down as 13 literal `.stack_size(1 << 20)` call sites plus 3 named constants that are still 1 MiB — `oxiz-solver/src/solver/encode/tests.rs:640` (`const STACK_SIZE: usize = 1 << 20;`), `oxiz-solver/src/solver/model_eval.rs:880` (`const WORKER_STACK: usize = 1 << 20;`), `oxiz-theories/src/string/ground_solver/eval.rs:911` (`const WORKER_STACK: usize = 1 << 20;`) — so a re-count that greps only for `.stack_size(1 << 20)` lands on 13 and must add those 3. Conversely, a naive `grep "1 << 20"` over the five crates returns **17**, not 16: it additionally matches `oxiz-theories/src/string/regex_membership.rs:534` (`const MAX_REPLACE_RE_STEPS: usize = 1 << 20;`), which is a rewrite-step budget, not a stack size, and must not be scaled. Per-site rationales: `oxiz-spacer/src/parser.rs` x2 — `MAX_TERM_NESTING = 500` with deliberately bounded native recursion at about 2 KiB/frame; a 128 KiB stack would make them overflow. `oxiz-solver/src/solver/encode/tests.rs`'s `encode_at_cap_depth_survives_a_one_mib_stack` — deliberately recursive pass at `ENCODE_DEPTH_LIMIT = 512`, measured to need at least 384 KiB. `oxiz-proof`'s `shared_dag(60)` and `oxiz-theories`'s `DOUBLINGS = 60` — these pin 2^60-versus-60 work, not stack depth, so scaling them destroys the test. Plus the sub-10,000-depth sites listed in the same-day change. Consider a short comment convention or a doc block so the reason travels with the code.
  - **Priority:** P3  **Scope:** small
- [ ] **Stale comment in `oxiz-spacer/src/parser.rs`** — `oxiz-spacer/src/parser.rs:1464`-`:1466`, where `parse_sexpr_survives_deep_nesting`'s comment states that "`SExpr`'s own `Drop` is derived and recursive". That is no longer true: `impl Drop for SExpr` at `oxiz-spacer/src/parser.rs:242` is an explicit iterative teardown. Pre-existing drift, not introduced by the 2026-07-31 pass.
  - **Priority:** P3  **Scope:** small
- [x] **Changelog note: a test was fixed that had never exercised its stated subject** — record for the v0.3.2 changelog. `deep_sequence_simplify_and_drop_return` (`oxiz-theories`) claimed to exercise both `SeqRewriter::simplify` and the deep `Drop` of a nested `SeqExpr`, but `SeqRewriter::simplify` (`oxiz-theories/src/string/sequence/mod.rs:974`) dismantles the tower on the way down: the helper it drives, `open_simplify` (`oxiz-theories/src/string/sequence/mod.rs:1015`), does `core::mem::replace(..., placeholder())` in four arms (`:1043`, `:1052`, `:1069`, and the `SeqExpr::Reverse` arm at `:1086` that the test hits), so the drop glue only ever received a one-level shell. The tower is now built twice and dropped explicitly, so both halves run. The test was strengthened, not weakened. — **(closed: this is a test-only strengthening, not a shipped behavior change — intentionally left out of the public CHANGELOG.md 0.3.2 entry for that reason)**
  - **Priority:** P3  **Scope:** small

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [x] `oxiz-theories`: `oxiz-theories/tests/fp_integration.rs:296` — fix `assert_is_normal` constraint encoding to reliably produce SAT for normal-float queries — **(fixed: assert_is_normal's constraint encoding is logically sound; the real bug was an unsound double-solve retry in FpSolver::check() (restore_to_trail_size left residue) — removed (wave2b theories-hard, TODO-STUB-FPNORMAL))**
  - Priority: P2 | Scope: small | Hint: none
- [x] `oxiz-solver`: `oxiz-solver/src/optimization.rs:749` — complete arithmetic theory solving (currently incomplete, returns unknown for many formulae) — **(fixed: optimize() rewritten to route every feasibility subquery through a fresh theory-complete oxiz-solver Solver; x=y AND x!=y now correctly Unsat)**
  - Priority: P2 | Scope: large | Hint: none
- [x] `oxiz-core`: `oxiz-core/src/qe/datatype/case_analysis.rs:237` — uncomment and wire Term construction in case-analysis QE path once Term API is available — **(fixed: real constructor case-split QE implemented in both qe/datatype/plugin.rs and qe/datatype/case_analysis.rs (DT-QE-CASE-ANALYSIS-STUB, wave2))**
  - Priority: P2 | Scope: small | Hint: none

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check)

- [x] **oxiz** `oxiz-solver`: `oxiz-solver/src/optimization.rs:749` — `TODO`: `Currently arithmetic theory solving is incomplete` — **(fixed: optimize() rewritten to route every feasibility subquery through a fresh theory-complete oxiz-solver Solver; x=y AND x!=y now correctly Unsat)**
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Complete the integer arithmetic theory in `optimize()` so a model with `x = y ∧ x ≠ y` is correctly returned as Unsat.
  - **Risk:** Incomplete theory propagation can yield Unknown or unsound Sat results; add targeted regression cases for contradictory integer constraints.
- [x] **oxiz** `oxiz-theories`: `oxiz-theories/tests/fp_integration.rs:296` — `TODO`: `Fix constraint encoding in assert_is_normal to reliably produce SAT` — **(fixed: assert_is_normal's constraint encoding is logically sound; the real bug was an unsound double-solve retry in FpSolver::check(), now removed (wave2b theories-hard, TODO-STUB-FPNORMAL))**
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Repair the floating-point constraint encoding in `assert_is_normal` so exponent/mantissa range constraints are correct and the normal-number assertion reliably solves.
  - **Risk:** Off-by-one exponent bias or mantissa width errors silently produce Unsat/Unknown; validate against known-normal IEEE-754 values.

---

## Production-Readiness Audit Findings (added 2026-07-16, ultracode audit)

**Method**: 19 scoped deep-audit agents (per-crate + cross-cutting: SMT-LIB 2.6 compliance, panic audit, Z3 gap vs upstream Z3, test-quality gap, release/packaging) followed by adversarial verification agents (90 verdicts collected before the run was stopped early by request; items below marked *unverified* did not get a verification pass — verify before fixing).

**Build baseline (2026-07-16)**: `cargo check --workspace --all-features` clean; `cargo clippy --all-targets --all-features` 0 warnings; `cargo nextest run --workspace --all-features` 6826/6826 passed (16 skipped). Note: all tests pass *despite* the findings below — i.e. the suite does not exercise these paths (see P2 test-gap items).

**Counts (after location-dedupe)**: P0 confirmed-critical 20 | P1 confirmed-major 30 | P2 unverified-critical 42 | P3 unverified-major 131 | P4 minor/downgraded 105

**Re-verification (2026-07-18, release-polish pass)**: every P0 and P1 item below was individually re-read against the current tree (not just diffed against the original finding) and marked `[x]` only when the described bug pattern was confirmed gone by inspection. Result at that time: **17/20 P0** and **28/30 P1** fixed, 5 still-open items (all in `oxiz-nlsat`).

**Full re-verification (2026-07-21, v0.3.0 hardening pass)**: a second wave of ~19 scoped investigator agents plus three implementation waves re-read every P0–P4 item below against the current tree — including the 5 items still open at the 07-18 pass, all of which are now fixed — and marked `[x]` only when the described bug pattern was confirmed fixed, made honest (documented no-op/honest-Unknown instead of a silent wrong answer), or shown not-a-bug by direct inspection. Result: **20/20 P0**, **30/30 P1**, **42/42 P2**, **126/131 P3**, **94/105 P4**, and **7/7 Policy/Release Chores** resolved. The 16 items still `[ ]` (5 P3, 11 P4) are genuine remaining gaps, none on the default solve path — see each item's note below and the "Remaining (post-0.3.0 hardening)" section for the grouped, deduplicated summary (externally-blocked / deliberately-deferred / confirmed-open-with-file:line).

**0.3.1 re-check (2026-07-31)**: the 0.3.1 soundness sweep and hardening pass closed 11 more of the items left open above — `get-model` value rendering, `:named` assertions / `get-unsat-core`, `:print-success` mode, lexer-error surfacing, `free_vars` binder scoping, the two dead `tactic/core` placeholder files, `TimeoutTactic` thread cancellation, `combine_inference_chains`, and the facade README drift. Running total: **20/20 P0**, **30/30 P1**, **42/42 P2**, **129/131 P3**, **102/105 P4**, **7/7 Policy/Release Chores**. The 5 items still `[ ]` (2 P3, 3 P4) are: Z3 `recfun` end-to-end support, the default-off `property-tests` feature, `oxiz-core`'s decorative secondary BV/FP/datatype theory submodule (two entries, no internal callers), and `mk_bv_concat`'s release-build width default. None is on the default solve path.

### P0 — Confirmed Critical (soundness: wrong sat/unsat/model; fix first)

- [x] `oxiz-math/src/polynomial/extended_ops.rs:1069` — Sturm sequence built from pseudo-remainders without sign normalization yields wrong root counts *(scope: math; effort: small)*
  - pseudo_remainder scales by lc(b)^k which can be negative, breaking the Sturm sign invariant. Concretely p=-x^2+1 gives chain [-x^2+1, -2x, 2] so count_roots_in_interval(-2,2) returns 0 instead of 2. Propagates to isolate_roots, realclosure::AlgebraicNumber::new (assert panics), and CAD/nlsat root reasoning: wrong sat/unsat.
  - **Fix**: Use exact rational remainder, or multiply pseudo-remainder by sign(lc_b)^k so the scale factor is always positive (Z3 uses signed pseudo-remainder).
- [x] `oxiz-math/src/grobner/buchberger.rs:993` — NraSolver::check_sat returns Sat without ever solving non-constant linear inequalities *(scope: math; effort: medium)*
  - Inequalities that reduce to non-constant polynomials of total_degree<=1 skip both the constant check and the has_complex_inequality Unknown path, so check_sat returns Sat. Asserting x>0 and x<0 (no equalities) returns Sat for an unsatisfiable system: a wrong answer from a public solver API.
  - **Fix**: Route remaining linear inequalities through the simplex/LP solver; return Unknown for any inequality not fully decided instead of Sat.
- [x] `oxiz-sat/src/solver/conflict.rs:83` — Conflict analysis assumes reason clause lits[0] is the propagated literal; binary-graph propagation violates this, dropping antecedent literals *(scope: sat; effort: small)*
  - analyze() skips reason-clause position 0 (`start = 1`), assuming the propagated literal sits there. Watch-based propagation maintains that invariant, but binary-implication-graph propagation (propagate.rs:28 `assign_propagation(implied_lit, clause_id)`) never reorders the stored (sorted) clause. When the implied literal is lits[1] (~50% of original/hyper-binary clauses), the false antecedent lits[0] is silently omitted from the learned clause, producing over-strong clauses that can flip SAT instances to UNSAT. analyze_theory_conflict (line 453 `clause.lits[1..]`) has the same flaw.
  - **Fix**: For binary-graph propagations, swap the implied literal to lits[0] before recording the reason, or resolve reason clauses by value (skip lit == current_lit) instead of by position.
- [x] `oxiz-sat/src/clause.rs:412` — Clause slot reuse via free_list breaks lazy watcher cleanup, letting stale watchers drive bogus unit propagations *(scope: sat; effort: medium)*
  - remove() pushes the ClauseId to free_list; add() immediately reuses the slot for a new clause. Stale watchers (cleaned only lazily via the `deleted` flag; WatchLists::remove_clause is dead_code) now reference a live, different clause. propagate() never verifies the watched literal is in the clause: it assumes the falsified literal is at lits[1] and may propagate lits[0] as "unit" while lits[1] is true/undef, and its swaps corrupt the real watchers' positions — unsound propagations and wrong answers on long runs with clause deletion.
  - **Fix**: Do not recycle ClauseIds while stale watchers may exist: scrub watch lists on remove (use remove_clause), or defer slot reuse until a full watch-list garbage collection pass.
- [x] `oxiz-sat/src/solver/mod.rs:860` — solve_with_assumptions after a prior solve() treats leftover model decisions as fixed, returning false UNSAT *(scope: sat; effort: small)*
  - solve() returns Sat leaving the full trail (decisions at levels >0). solve_with_assumptions never backtracks to root first; assumption_level_start is captured at the dirty level and an assumption that merely disagrees with the previous arbitrary model hits `value.is_false()` and immediately returns (Unsat, core). Example: (a∨b); solve() picks ¬a,b; solve_with_assumptions([a]) reports UNSAT though a∧(a∨b) is SAT. This breaks the standard MaxSAT/incremental usage pattern. The extracted core also reads stale `seen` flags.
  - **Fix**: Call backtrack_with_phase_saving(0) at the top of solve_with_assumptions before capturing assumption_level_start, and only report UNSAT when the assumption is false at level 0.
- [x] `oxiz-nlsat/src/solver/decide.rs:500` — Irrational roots silently dropped: solver returns wrong UNSAT for e.g. x^2 > 2 *(scope: nlsat; effort: large)* — **(fixed: SturmSequence::isolate_roots is now wired into the feasible-region path (pick_arith_value -> compute_arith_regions -> univariate_regions); x^2>2 returns Sat with a witness, x^2=2 returns Unknown, never a wrong Unsat)**
  - find_univariate_roots only finds RATIONAL roots; quadratic with non-square discriminant returns Vec::new() ('Irrational roots - cannot represent exactly'). compute_feasible_region then treats the polynomial as sign-constant, so for asserted 'x^2-2>0' the feasible set is EMPTY and solve() returns Unsat at level 0 (mod.rs:653). oxiz-theories/src/nlsat.rs:369 trusts Unsat for univariate atoms, so the final answer is wrong.
  - **Fix**: Use SturmSequence root isolation (algebraic numbers / isolating intervals) instead of rational-only roots in find_univariate_roots, or return IntervalSet::reals() when roots may be missing and rely on validation.
- [x] `oxiz-nlsat/src/solver/mod.rs:655` — Infinite loop: empty feasible region at level>0 backtracks without learning, re-makes identical decision *(scope: nlsat; effort: large)* — **(fixed: pick_arith_value now returns a 4-variant ArithDecision (not None); the solve loop learns a lemma (ProvedEmpty) or terminates (Unknown/Unsat) in every branch, so the no-lemma backtrack spin is gone)**
  - When pick_arith_value returns None at level>0, solve() calls backtrack(level-1) with no lemma, no activity bump, no phase flip. decide() then re-picks the same variable with the same saved phase, reproducing the identical state forever. Example that hangs: (x>1) AND (x<-1 OR x<5) — trivially SAT but loops indefinitely. No conflict is counted so restarts never fire.
  - **Fix**: Learn a clause negating the decisions/atoms whose interval intersection is empty (NLSAT semantic-conflict lemma), or at minimum flip the saved phase of the last decision before re-deciding.
- [x] `oxiz-nlsat/src/nia.rs:363` — NIA branch-and-bound adds both branch constraints permanently to one shared solver *(scope: nlsat; effort: large)* — **(fixed: create_branch is gone; branch_and_bound snapshots the base problem and rebuilds a fresh path-scoped NlsatSolver per node (rebuild_solver); push_branch only records bounds)**
  - create_branch adds 'x<=floor' and 'x>=ceil' as permanent unit clauses to the SAME NlsatSolver; popping a BranchNode never retracts them, so after pushing both branches the solver holds contradictory constraints and every node solves the same over-constrained problem. branch_and_bound then exhausts the stack and returns Unsat (nia.rs:276) for satisfiable integer problems. NiaSolver is the QF_NIA path in oxiz-theories.
  - **Fix**: Use push/pop scopes or assumption literals per branch node so constraints are retracted on backtrack; never treat search-space exhaustion under leaked constraints as Unsat.
- [x] `oxiz-nlsat/src/solver/mod.rs:505` — solve() never resets trail/arithmetic state, breaking incremental re-solve *(scope: nlsat; effort: medium)*
  - After a Sat answer, the trail, decision levels, and arithmetic values remain assigned. NiaSolver re-invokes solve() after add_clause: the new unit literal is assigned at a stale non-zero level and theory_propagate evaluates it against the stale model, producing spurious conflicts; analyze_conflict resolves Unit/Theory-justified literals away with no reason clause and can return an empty learnt clause, which solve() reports as Unsat (mod.rs:548-549).
  - **Fix**: Backtrack to level 0 and clear arithmetic assignments at solve() entry; give Unit/Theory-justified literals proper reasons in analyze_conflict instead of silently dropping them.
- [x] `oxiz-nlsat/src/cad.rs:518` — Sturm sequence built from sign-unnormalized pseudo-remainders gives wrong root counts *(scope: nlsat; effort: medium)*
  - pseudo_remainder scales by lc(divisor) on every reduction step; when the leading coefficient is negative an odd number of scalings flips the remainder's sign, so the chain is not a Sturm chain. Example: p = 4-x^2 yields chain (-x^2+4, -x, +4) and count_roots() = 0 despite roots +/-2. All root-atom evaluation (evaluate_root_atom) and CAD lifting depend on isolate_roots, so answers involving negative-leading-coefficient polynomials are wrong.
  - **Fix**: Track the sign of lc(b)^k applied during pseudo-division and multiply the remainder by it (or normalize lc(b) positive before division) so the chain satisfies Sturm's sign conditions.
- [x] `oxiz-nlsat/src/portfolio.rs:261` — PortfolioSolver solves empty solvers: returns Sat for every input *(scope: nlsat; effort: large)*
  - run_parallel_solvers creates fresh NlsatSolver::new() instances and never copies the base problem ('simplified - no actual problem to solve yet'). The empty problem is trivially Sat, so PortfolioSolver::solve() always answers Sat with an empty model, including for unsatisfiable inputs. config.timeout and the diverse configs are also ignored. Public API re-exported from lib.rs.
  - **Fix**: Clone base_solver's clauses/atoms into each worker via create_configured_solver, apply per-worker configs, honor timeout, and extract real models/cores; otherwise remove the API until implemented.
- [x] `oxiz-core/src/smtlib/parser/terms.rs:455` — Parser silently turns (div a b) and (mod a b) into subtraction *(scope: core-rest; effort: small)*
  - This parser IS the production path: oxiz-cli -> oxiz_solver::Context::execute_script -> parse_script. Any script using integer div/mod gets a semantically different formula, so check-sat can answer wrong on plausible LIA inputs. TermManager already has mk_div/mk_mod (ast/manager/builder.rs:314,320) but the parser ignores them.
  - **Fix**: Route "div" to self.manager.mk_div(lhs, rhs) and "mod" to mk_mod; add regression tests like (assert (= (div 7 2) 3)).
- [x] `oxiz-core/src/smtlib/parser/terms.rs:929` — Real division "/", abs, to_real, to_int, divisible parsed as Bool-sorted uninterpreted functions *(scope: core-rest; effort: medium)*
  - The operator match has no case for "/" or other core Int/Real ops, so they fall to the default arm and become mk_apply with Bool default sort. Arithmetic theory then ignores these constraints entirely — QF_LRA scripts with division get wrong sat/unsat answers on the production parse path.
  - **Fix**: Add explicit cases for "/", "abs", "to_real", "to_int", "divisible"; reject genuinely unknown undeclared operators with a ParseError instead of Bool-sorted apply.
- [x] `oxiz-core/src/ast/manager/query.rs:445` — TermManager::substitute silently skips Apply, BV, String, FP, Xor, Distinct, Div/Mod, quantifier and Let terms *(scope: core-ast; effort: medium)*
  - substitute_cached handles only ~15 term kinds; everything else hits 'Some(_) => id' with comment 'For complex terms, just return as-is for now'. Tactics solve_eqs, ackermann, propagate, ctx_simplify and quantifier instantiation (tactic/quantifier.rs:533) rely on it: substituting x->3 in f(x) or any bitvector/string assertion returns the term unchanged, so solved equations are dropped while occurrences remain — wrong sat/unsat and wrong models.
  - **Fix**: Handle all TermKind variants generically via get_children plus a rebuild function (as rewrite_children does), and descend into quantifier bodies with bound-variable shadowing.
- [x] `oxiz-core/src/simplification/mod.rs:299` — Boolean absorption in AND/OR drops all other conjuncts/disjuncts *(scope: core-ast; effort: small)*
  - try_boolean_absorption_in_and returns just 'candidate' and simplify_and returns it as the whole result. And(a, Or(a,b), c) simplifies to 'a', silently dropping c — an UNSAT formula (c=false) becomes SAT. The OR variant (line 316) similarly turns Or(a, And(a,b), c) into 'a', dropping disjunct c and turning SAT into UNSAT. Reachable via AggressiveSimplifier with aggressive=true.
  - **Fix**: Absorption must remove only the absorbed Or/And argument and keep the remaining args: rebuild mk_and(args minus the absorbed term), not return candidate alone.
- [x] `oxiz-core/src/simplification/mod.rs:343` — try_factor_or_of_ands discards every disjunct outside the matched pair *(scope: core-ast; effort: small)*
  - For Or(And(x,a), And(x,b), c, ...), the factoring rule returns And(x, Or(a,b)) and drops c and all other disjuncts, strengthening the formula — a satisfiable input can become UNSAT. Fires in aggressive simplification whenever any two AND disjuncts share a conjunct.
  - **Fix**: Include the untouched disjuncts: build Or(And(common, Or(left_rest,right_rest)), remaining_args...).
- [x] `oxiz-core/src/rewrite/bv.rs:417` — BvShl/BvLshr rewrite returns Unchanged(lhs): x << y silently becomes x *(scope: core-ast; effort: small)*
  - rewrite_bvshl (line 417) and rewrite_bvlshr (line 462) end with RewriteResult::Unchanged(lhs) when args are non-constant. Through CombinedRewriter's result.term() (combined.rs:514) the entire shift expression is replaced by its left operand, changing formula semantics for any symbolic shift.
  - **Fix**: Return Unchanged(manager.mk_bv_shl(lhs, rhs)) / mk_bv_lshr(lhs, rhs); both builders exist.
- [x] `oxiz-core/src/tactic/quantifier.rs:968` — DER forall rule is logically inverted: rewrites ∀x.(x=t ∨ ψ) to ψ[t/x], which is unsound *(scope: core-tactic; effort: medium)*
  - Correct DER eliminates a DISEQUALITY disjunct: ∀x.(x≠t ∨ ψ) ≡ ψ[t/x]. The code eliminates the positive equality instead, and also rewrites the x≠t→ψ implication (≡ x=t ∨ ψ) to ψ[t/x]. Goal {∀x.(x=5 ∨ P(x)), ¬P(6)} is UNSAT but becomes {P(5), ¬P(6)} = SAT. ∀x.(x=t) also rewrites to true.
  - **Fix**: For Forall, match Not(Eq(x,t)) disjuncts (and Eq antecedents of Implies) instead of positive equalities; keep the exists/And path as-is.
- [x] `oxiz-core/src/tactic/quantifier.rs:689` — SkolemizationTactic reuses Skolem names across assertions and ignores polarity *(scope: core-tactic; effort: medium)*
  - skolemize() (ast/normal_forms.rs:758) resets counter=0 per call, so per-assertion calls give distinct existentials the SAME sk_0 variable: {∃x.P(x), ∃x.¬P(x)} (SAT) becomes {P(sk_0), ¬P(sk_0)} (UNSAT). skolemize also recurses through Not/Implies without flipping polarity, so ¬(∃x.P(x)) becomes ¬P(sk_0) (UNSAT→SAT), and Skolem function args are built with hardcoded bool_sort (normal_forms.rs:855).
  - **Fix**: Thread one global fresh-name counter through the goal, track polarity (skolemize Exists only at positive polarity, Forall at negative), and use real universal-var sorts for Skolem function arguments.
- [x] `oxiz-core/src/tactic/quantifier.rs:624` — QuantifierInstantiationTactic instantiates Forall terms found at any polarity as asserted facts *(scope: core-tactic; effort: medium)*
  - collect_quantifiers (line 646) gathers every Forall subterm, including ones under Not, Or, or Implies antecedents, then pushes φ(t) as a new top-level assertion. For goal ¬(∀x.P(x)) ∧ ¬P(c) with trigger matching c, the added P(c) flips SAT to UNSAT.
  - **Fix**: Only instantiate quantifiers that occur as positive-polarity top-level assertions (or track polarity during collection and skip negative/mixed occurrences).

### P1 — Confirmed Major (silent constraint drop / advertised-but-broken)

- [x] `oxiz-math/src/grobner/buchberger.rs:119` — reduce() silently discards the unreduced remainder when the 1000-iteration cap is hit *(scope: math; effort: small)*
  - **Fix**: On cap exhaustion return r.add(&p) (still ideal-equivalent) or propagate a resource-limit error; never drop p.
- [x] `oxiz-math/src/simplex.rs:609` — SimplexTableau never repairs non-basic variables violating their own bounds; check() can report Sat for infeasible systems *(scope: math; effort: medium)*
  - **Fix**: In add_bound, when var is non-basic and its value violates the new bound, set it to the bound and recompute dependent basic vars (Dutertre-de Moura update).
- [x] `oxiz-math/src/fast_rational.rs:323` — mul_small/new_small use saturating_abs, corrupting gcd at i64::MIN and silently computing wrong products *(scope: math; effort: small)*
  - **Fix**: Pass values directly to gcd_i64 (it already uses unsigned_abs), or special-case i64::MIN by promoting to Big before reduction.
- [x] `oxiz-math/src/rational/mod.rs:888` — Number-theory helpers are silently wrong beyond trial-division limits and euler_totient can effectively hang *(scope: math; effort: medium)*
  - **Fix**: Factor completely via Pollard rho + Miller-Rabin (both already present) instead of bounded trial division; add an iteration/resource cap returning an explicit error.
- [x] `oxiz-nlsat/src/solver/propagate.rs:546` — Theory conflict explanation is not a valid lemma: negates every assigned atom sharing a variable *(scope: nlsat; effort: large)* — **(fixed: explain_theory_conflict now uses a model-based sign-abstraction single-cell certifier (certify_sign_conflict) + install_theory_conflict backjump, replacing the old negate-every-shared-var heuristic — sound for multivariate conflicts (wave2b nlsat-wire-explain))**
  - **Fix**: Wire ExplainContext/CAD projection (resultants, discriminants, root atoms) into explain_theory_conflict so lemmas are theory-valid, as in Z3 nlsat_explain.cpp.
- [x] `oxiz-nlsat/src/solver/mod.rs:413` — Empty clause silently dropped: add_clause returns NULL_CLAUSE without recording conflict *(scope: nlsat; effort: small)*
  - **Fix**: Set self.conflict_clause (or a dedicated unsat flag) when an empty clause is added so solve() returns Unsat immediately.
- [x] `oxiz-nlsat/src/solver/mod.rs:42` — No resource limits in solve(): max_conflicts accepted but never read, Unknown unreachable *(scope: nlsat; effort: small)*
  - **Fix**: Check stats.conflicts against config.max_conflicts (and an optional deadline) in the solve loop, returning SolverResult::Unknown when exceeded.
- [x] `oxiz-nlsat/src/simplify.rs:102` — simplify_ineq_atom drops negative constant factor without flipping Lt/Gt: opposite constraint *(scope: nlsat; effort: medium)*
  - **Fix**: Track a parity of negations (from dropped negative constants and leading-coefficient normalization of odd factors) and flip Lt<->Gt when parity is odd; fix the empty-factors Trivial cases too.
- [x] `oxiz-nlsat/src/maxsat.rs:230` — MaxSatSolver cost and model extraction are stubs: always reports Optimal cost 0 with empty model *(scope: nlsat; effort: medium)*
  - **Fix**: Read relaxation-variable values from solver.get_model() to compute the true violated weight, iterate the linear search with cardinality/weight bounds, and return the real assignment.
- [x] `oxiz-nlsat/src/cad.rs:753` — Root isolation silently merges roots closer than 1e-6 into one 'isolating' interval *(scope: nlsat; effort: medium)*
  - **Fix**: Keep bisecting with exact rational arithmetic until each interval contains exactly one root (Sturm counts make this terminating for square-free input); square-free-factorize first to handle multiple roots.
- [x] `oxiz-nlsat/src/lib.rs:58` — ~25 of 40 exported modules are shelf-ware never wired into the solver *(scope: nlsat; effort: large)*
  - **Fix**: Either integrate these engines into NlsatSolver's solve pipeline (inprocessing hooks, CAD explain, proof logging) or mark them experimental/private so the API does not advertise nonfunctional features.
- [x] `oxiz-nlsat/src/nia.rs:406` — floor_ceil truncates toward zero: wrong floor/ceil for negative fractional values *(scope: nlsat; effort: small)* — **(fixed: floor_ceil now uses BigRational::floor/ceil (sign-adjusted), correct for negative fractional values)**
  - **Fix**: Use value.floor()/value.ceil() from BigRational (or adjust the truncated quotient by -1 when value is negative and non-integral).
- [x] `oxiz-core/src/smtlib/parser/terms.rs:125` — Undeclared symbols silently become fresh Bool variables instead of a parse error *(scope: core-rest; effort: small)*
  - **Fix**: Return OxizError::ParseError("unknown constant") for symbols not in bindings/constants/dt_constructors, matching SMT-LIB and Z3 behavior.
- [x] `oxiz-core/src/smtlib/parser/terms.rs:351` — Indexed BV ops (zero_extend, sign_extend, rotate_left, repeat) degrade to Bool-sorted generic applies *(scope: core-rest; effort: medium)*
  - **Fix**: Add explicit cases mapping zero_extend/sign_extend/rotate_left/rotate_right/repeat to the corresponding mk_bv_* builders with correct result widths.
- [x] `oxiz-core/src/smtlib/parser/commands.rs:387` — Unknown SMT-LIB commands (define-fun-rec, declare-sort, get-unsat-assumptions) silently skipped *(scope: core-rest; effort: medium)*
  - **Fix**: Implement declare-sort and define-fun-rec; for genuinely unsupported commands emit (error "unsupported command") instead of silent skip.
- [x] `oxiz-core/src/smtlib/parser/commands.rs:147` — set-option numeric/string values silently replaced with empty string *(scope: core-rest; effort: small)*
  - **Fix**: Peek the token kind and accept Symbol, Numeral, Decimal, and StringLit values; error on anything else instead of defaulting to "".
- [x] `oxiz-core/src/smtlib/parser/commands.rs:452` — declare-datatypes parses only the first datatype's constructor list; multi/mutual datatypes broken *(scope: core-rest; effort: medium)*
  - **Fix**: Loop constructor groups once per declared datatype name, pair each group with its name, and parse selector sorts via parse_sort().
- [x] `oxiz-core/src/qe/string/plugin.rs:204` — StringQePlugin eliminates any constrained string quantifier to unconditional true *(scope: core-rest; effort: small)*
  - **Fix**: Return None (conservative give-up) until real length solving/automata construction is implemented; never fabricate true.
- [x] `oxiz-core/src/qe/arith/cooper.rs:241` — Cooper QE returns the input formula with the quantified variable still free, claiming elimination *(scope: core-rest; effort: large)*
  - **Fix**: Return Err("not implemented") from eliminate_exists until the substitution/test-set machinery is real, or implement Cooper's construction referencing Z3 qe_arith.
- [x] `oxiz-core/src/model/evaluator.rs:155` — Model evaluator silently truncates big integer and wide BV constants to 0 *(scope: core-rest; effort: medium)*
  - **Fix**: Return EvalResult::Error on out-of-range conversion, or widen Value::Int to BigInt / Value::BitVec to BigUint.
- [x] `oxiz-core/src/qe/array/quantifier_elim.rs:315` — Array QE module built on placeholder TermId=usize; Skolem constants are string lengths *(scope: core-rest; effort: medium)*
  - **Fix**: Stop exporting the module (or mark #[doc(hidden)] experimental) until it operates on real crate::ast::TermId with actual substitution.
- [x] `oxiz-core/src/qe/arith/omega_test.rs:189` — Omega test can only ever return Unknown: both shadow checks are hardcoded *(scope: core-rest; effort: large)*
  - **Fix**: Implement the real/dark shadow bound comparisons over LinearConstraint, or document and return Unknown without fake statistics.
- [x] `oxiz-core/src/ast/manager/mod.rs:99` — Hash-cons cache keys on TermKind only, ignoring sort: same-named vars of different sorts alias *(scope: core-ast; effort: small)*
  - **Fix**: Key the cache on (TermKind, SortId) — at minimum for TermKind::Var and Apply where the sort is not derivable from the kind.
- [x] `oxiz-core/src/rewrite/string.rs:303` — indexof(s, "", i) -> i without the required 0 <= i <= len(s) side condition *(scope: core-ast; effort: small)*
  - **Fix**: Apply only when start is a constant within [0, len(s)] for constant s; otherwise rewrite to ite(0<=i<=len(s), i, -1) or leave unchanged.
- [x] `oxiz-core/src/rewrite/combined.rs:490` — Unbounded recursion in rewrite_bottom_up, AggressiveSimplifier and substitute_cached *(scope: core-ast; effort: medium)*
  - **Fix**: Convert to explicit worklist iteration, or enforce a depth counter that bails out returning the term unchanged (sound).
- [x] `oxiz-core/src/tactic/solve_eqs.rs:649` — FM op_limit abort marks constraints dead without adding their resolvents, losing constraints *(scope: core-tactic; effort: small)*
  - **Fix**: If the op limit fires before all pairs for a variable are resolved, keep that variable's original constraints alive (skip elimination for it) instead of marking them dead.
- [x] `oxiz-core/src/tactic/lia2card.rs:425` — Sequential-counter and commander aux variables use non-unique names, aliasing across constraints *(scope: core-tactic; effort: small)*
  - **Fix**: Include the per-tactic aux_var_counter (as done for '__tot_{}_{}') in every aux variable name and bump it per constraint.
- [x] `oxiz-core/src/tactic/bv/bv_rewriter.rs:378` — BvRewriterTactic::rewrite replaces every BV operation with arbitrary TermId(0) *(scope: core-tactic; effort: medium)*
  - **Fix**: Implement reconstruct_* via manager.mk_bv_* and the constant predicates via TermKind::BitVecConst matching, or delete the type until real; at minimum make rewrite() return the input unchanged.
- [x] `oxiz-core/src/tactic/bitblast.rs:224` — Bit-blasting tactic never bit-blasts — both stateful and stateless versions return the goal unchanged *(scope: core-tactic; effort: large)*
  - **Fix**: Implement real blasting (per-bit Booleans + circuit encoding) or rename/document as a probe and remove 'bit-blast' from the registry until functional.
- [x] `oxiz-core/src/tactic/arith/arith_bounds.rs:200` — Seven exported tactic types are permanent NotApplicable placeholders with empty helper bodies *(scope: core-tactic; effort: large)*
  - **Fix**: Either implement against the real TermManager AST or mark these #[doc(hidden)]/remove from public exports so consumers cannot mistake them for working preprocessing.

### P2 — Unverified Critical (adversarially verify, then fix)

- [x] `oxiz-solver/src/mbqi/integration.rs:296` — MBQI claims Satisfied (sat) after finite candidate check over infinite domains *(scope: z3-gap; effort: large)*
- [x] `oxiz-solver/src/solver/check_string.rs:11` — String atoms (str.contains, str.in_re, prefixof, indexof, ...) are free booleans, never theory-checked *(scope: z3-gap; effort: large)*
- [x] `oxiz-solver/src/solver/check_fp.rs:46` — FP and array 'theories' are benchmark-keyed heuristics; real solvers unwired *(scope: z3-gap; effort: large)*
- [x] `oxiz-theories/src/bv/solver.rs:674` — Barrel shifters (bvshl/bvlshr/bvashr) ignore high bits of the shift amount, producing wrong bit-blasting *(scope: theories-arith; effort: small)*
- [x] `oxiz-theories/src/bv/solver.rs:1146` — bv_udiv/bv_urem/bv_sdiv/bv_srem encodings admit spurious quotients: q*b + r may wrap mod 2^w *(scope: theories-arith; effort: small)*
- [x] `oxiz-theories/src/arithmetic/solver.rs:477` — LIA mode never enforces integrality: check() only runs the LP relaxation *(scope: theories-arith; effort: large)*
- [x] `oxiz-theories/src/arithmetic/simplex.rs:579` — Pivot-limit exhaustion in make_feasible/dual_simplex returns Ok(()) — infeasible state reported as SAT *(scope: theories-arith; effort: medium)*
- [x] `oxiz-theories/src/arithmetic/lia/branching.rs:135` — Branch-and-bound 'backtrack' calls simplex.reset(), erasing all constraints before the down-branch *(scope: theories-arith; effort: small)*
- [x] `oxiz-theories/src/arithmetic/lia/cuts.rs:188` — Placeholder MIR/CG/Gomory/disjunctive 'cuts' are invalid inequalities added as permanent constraints *(scope: theories-arith; effort: large)*
- [x] `oxiz-theories/src/fp/solver.rs:687` — assert_fp_lt encodes a<b as 'a negative AND b positive'; assert_fp_le adds no ordering constraint at all *(scope: theories-arith; effort: large)*
- [x] `oxiz-cli/src/model_counter.rs:99` — --count-models returns fabricated counts: exact mode always reports 0, approximate mode never invokes the solver *(scope: frontends; effort: large)*
- [x] `oxiz-cli/src/main.rs:221` — --timeout flag (and config-file timeout) is never enforced in normal solving; solver can hang forever *(scope: frontends; effort: medium)*
- [x] `oxiz-sat/src/solver/mod.rs:651` — add_clause watches the first two sorted literals even if already false, missing conflicts on incrementally added clauses *(scope: sat; effort: small)*
- [x] `oxiz-sat/src/solver/learn.rs:469` — Inprocessing clause strengthening removes a literal after proving F ⊨ lit — logically wrong direction, yields unsound clauses *(scope: sat; effort: small)*
- [x] `oxiz-sat/src/preprocessing_core.rs:196` — Pure literal elimination deletes clauses without recording the forced assignment — models can violate deleted clauses *(scope: sat; effort: small)*
- [x] `oxiz-sat/src/solver/propagate.rs:21` — Binary implication graph entries are never removed on pop()/forget_learned_since and bypass the deleted-clause check *(scope: sat; effort: medium)*
- [x] `oxiz-sat/src/symmetry.rs:262` — detect_symmetries emits unverified permutations; SymmetryBreakTactic then adds lex-leader constraints that can change satisfiability *(scope: sat; effort: medium)*
- [x] `oxiz-core/src/smtlib/parser/terms.rs:461` — Integer 'mod' is parsed as subtraction, producing wrong sat/unsat answers *(scope: smtlib-compliance; effort: small)*
- [x] `oxiz-theories/src/euf/union_find.rs:53` — Path compression in find() is not trail-recorded, so pop() leaves corrupted equivalence classes *(scope: theories-rest; effort: small)*
- [x] `oxiz-theories/src/euf/solver.rs:998` — pop() never removes proof-forest edges added to pre-existing nodes, so conflict explanations cite retracted assertions *(scope: theories-rest; effort: medium)*
- [x] `bench/z3_parity/results.json:120` — Checked-in parity results show 4 Sat-answers on UNSAT quantified benchmarks, contradicting README's '100% Z3 parity' claim, and no test covers those directories *(scope: test-gap; effort: medium)*
- [x] `oxiz-solver/src/solver/tests.rs:250` — Ignored test documents a known wrong-model bug: BV solver returns SAT but model gives value violating the constraints *(scope: test-gap; effort: medium)*
- [x] `oxiz-core/src/smtlib/parser/terms.rs:14` — Recursive-descent parse_term has no depth limit: stack-overflow abort on deeply nested input *(scope: panic-audit; effort: small)*
- [x] `oxiz-opt/src/maxsat/algorithms.rs:71` — Weighted MaxSAT (default stratified path) ignores weights and returns wrong optimum as Optimal *(scope: opt-proof; effort: large)*
- [x] `oxiz-opt/src/preprocess.rs:329` — unit_propagation treats SOFT unit clauses as hard facts, silently dropping conflicting soft clauses *(scope: opt-proof; effort: medium)*
- [x] `oxiz-opt/src/context.rs:573` — OptContext::optimize_maxsmt silently coerces Rational weights to 1 and returns Optimal after Unknown breaks the binary search *(scope: opt-proof; effort: medium)*
- [x] `oxiz-proof/src/craig.rs:557` — Craig interpolation colors every axiom A and ignores the user partition, so extract() returns trivial 'true' interpolants *(scope: opt-proof; effort: large)*
- [x] `oxiz-proof/src/rules.rs:288` — Proof rule validators unconditionally return Valid — checker accepts invalid proofs *(scope: opt-proof; effort: large)*
- [x] `oxiz-spacer/src/pdr.rs:417` — is_init_reachable always returns false — counterexamples at level 0 are never detected *(scope: spacer; effort: medium)*
- [x] `oxiz-spacer/src/pdr.rs:472` — is_transition_feasible is a stub returning false — Spacer can never return Unsafe *(scope: spacer; effort: large)*
- [x] `oxiz-spacer/src/smt.rs:253` — is_lemma_inductive has no primed-state renaming and conjoins all rules — every lemma trivially 'inductive' *(scope: spacer; effort: large)*
- [x] `oxiz-spacer/src/parser.rs:666` — ChcParser parses predicate applications as 'true' — all predicate structure silently erased *(scope: spacer; effort: medium)*
- [x] `oxiz-spacer/src/bmc.rs:281` — Multiple transition rules are conjoined, not disjoined — k-induction proves 'Safe' for unsafe systems *(scope: spacer; effort: medium)*
- [x] `oxiz-spacer/src/invariant.rs:515` — Houdini 'verification' is a confidence-threshold filter with zero SMT queries — all candidates returned as verified invariants *(scope: spacer; effort: large)*
- [x] `oxiz-solver/src/solver/mod.rs:542` — Solver returns Sat after 10 inconclusive MBQI rounds, assuming quantifiers hold *(scope: solver-rest; effort: small)*
- [x] `oxiz-solver/src/mbqi/integration.rs:509` — MBQI substitution silently skips Xor, Distinct, nested Forall/Exists, BV and string kinds, producing lemmas with leftover bound variables *(scope: solver-rest; effort: medium)*
- [x] `oxiz-solver/src/solver/theory_manager.rs:1485` — Conflict-limit exhaustion suppresses real theory conflicts and returns Sat *(scope: solver-core; effort: medium)*
- [x] `oxiz-solver/src/solver/encode.rs:1133` — BvSlt/BvSle also asserted into linear arithmetic with unsigned semantics *(scope: solver-core; effort: medium)*
- [x] `oxiz-solver/src/solver/check_fp.rs:1283` — FP pre-check collects Eq facts ignoring polarity, causing wrong UNSAT *(scope: solver-core; effort: small)*
- [x] `oxiz-solver/src/solver/mod.rs:805` — push/pop never push/pop the BV solver; committed BV facts leak across scopes *(scope: solver-core; effort: medium)*
- [x] `oxiz-wasm/src/js_api/optimize.rs:57` — WASM minimize/maximize/assertSoft are silently dropped; optimize() reports plain sat as "optimal" *(scope: bindings; effort: large)*
- [x] `oxiz-wasm/src/js_api/optimize.rs:572` — computeInterpolant returns conjunction of partition A as a fake "interpolant" *(scope: bindings; effort: small)*

### P3 — Unverified Major

- [x] `GAP` — Recursive function definitions (Z3 recfun) unusable end-to-end *(scope: z3-gap)* — **(fixed in 0.3.3: `define-fun-rec`/`define-funs-rec` are parsed (`oxiz-core/src/smtlib/parser/recfun.rs`, new) and discharged by fuel-bounded unfolding (`oxiz-solver/src/context/recfun.rs` + `recfun/eval.rs`, new); mutual recursion works via a two-pass parse. `sat` needs a saturation or model-recomputation certificate, `unsat` is immediate since the instantiated problem is a relaxation; fuel exhaustion is an honest `unknown`. 21 tests in `oxiz-solver/tests/recfun_e2e.rs`.)**
- [x] `oxiz-solver/src/context.rs:741` — set_option ignores every option except produce-proofs/produce-unsat-cores *(scope: z3-gap)*
- [x] `oxiz-solver/src/context.rs:850` — get-model prints wrong sort/value for BitVec, Array, FP, and uninterpreted constants *(scope: z3-gap)* — **(fixed: BitVec values and sort names in 0.3.0; FP literals, nested `(Array ..)` values and uninterpreted-sort witnesses in 0.3.1 — see the "Remaining" section entry for details)**
- [x] `oxiz-core/src/ematching/code_tree.rs:894` — E-matching code-tree backtracking stub drops matches *(scope: z3-gap)* — **(fixed: execute_from stub replaced with a full recursive interpreter (run(ip, current_term, ...)); Choice first-branch matches are no longer dropped (wave2b core-tactics, TODO-939))**
- [x] `oxiz-core/src/tactic/mbp.rs:307` — Model-based projection assumes linearity unconditionally; nonlinear input gets linear projection *(scope: z3-gap)* — **(fixed: explicit ProjectorKind::Nonlinear added; detect_projector now returns Nonlinear when contains_nonlinear_arith holds instead of defaulting to LRA (wave2b core-tactics, TODO-940))**
- [x] `oxiz-theories/src/fp/ieee754_full.rs:1053` — sqrt() halves odd-exponent inputs: normalized significand can never shift left but exponent is still decremented *(scope: theories-arith)* — **(fixed: sqrt() halves odd-exponent inputs — odd-exponent now folds sqrt(2) into the result)**
- [x] `oxiz-theories/src/fp/ieee754_full.rs:727` — RoundNearestTiesToEven rounds ties up instead of to even (the default rounding mode) *(scope: theories-arith)* — **(fixed: RoundNearestTiesToEven rounded ties up instead — exact tie now rounds to even lsb)**
- [x] `oxiz-theories/src/fp/ieee754_full.rs:525` — Subnormal unpack uses off-by-one shift, doubling every subnormal's value in arithmetic *(scope: theories-arith)* — **(fixed: subnormal unpack used an off-by-one shift — now the same shift as normals plus renormalize)**
- [x] `oxiz-theories/src/fp/solver.rs:769` — FP<->BV and FP<->Real conversions are stubs that leave results completely unconstrained *(scope: theories-arith)* — **(fixed: FP<->BV and FP<->Real conversions were stubs — has_unsupported_conversion now makes check() return Unknown, no bogus Sat)**
- [x] `oxiz-theories/src/fp/solver.rs:430` — assert_fp_eq conflates fp.eq and bitwise '='; forces non-NaN and sign equality, breaking NaN= and +0/-0 cases *(scope: theories-arith)* — **(fixed: assert_fp_eq conflated fp.eq and bitwise '=' — now separate: SMT '=' (NaN==NaN, +0!=-0) vs fp.eq)**
- [x] `oxiz-theories/src/arithmetic/simplex.rs:1110` — propagate_bounds/tighten_bounds write bounds directly, bypassing the undo trail and dropping all but one reason *(scope: theories-arith)* — **(fixed: propagate_bounds/tighten_bounds wrote bounds directly — now trail-recorded with reasons via aux_reasons)**
- [x] `oxiz-theories/src/arithmetic/solver.rs:259` — GCD-infeasibility path fabricates the conflict: contradictory bounds asserted with hardcoded reason 0 *(scope: theories-arith)* — **(fixed: GCD-infeasibility path fabricated the conflict — now uses a real add_reason(reason))**
- [x] `oxiz-theories/src/arithmetic/simplex_opt.rs:260` — optimize_linexpr rebrands pivot-limit Unknown as Optimal(current value) *(scope: theories-arith)* — **(fixed: optimize_linexpr rebranded pivot-limit Unknown as Optimal — now stays Unknown, regression test added)**
- [x] `oxiz-theories/src/bv/solver.rs:1891` — notify_equality probe solve leaves learned-clause residue that check() documents as unsound *(scope: theories-arith)* — **(fixed: notify_equality probe solve left learned-clause residue — embedded_sat_config now disables hyper-binary+inprocessing, check() cleans up)**
- [x] `oxiz-theories/src/arithmetic/simplex.rs:20` — Simplex uses fixed-width Rational64; coefficient growth during pivoting panics on overflow *(scope: theories-arith)* — **(fixed: Simplex used fixed-width Rational64 coefficients that could overflow — checked_*_r64 helpers added, pivot returns Unknown instead of panicking)**
- [x] `oxiz-theories/src/fp/solver.rs:906` — FpSolver::check lacks the incremental-probe cleanup and model snapshot BvSolver needs; returns empty conflict *(scope: theories-arith)* — **(fixed: FpSolver::check lacked incremental-probe cleanup — restore_to_trail_size + forget_learned_since + snapshot added)**
- [x] `oxiz-cli/src/main.rs:804` — --memory-limit, --conflict-limit, --decision-limit are silently ignored *(scope: frontends)*
- [x] `oxiz-cli/src/main.rs:828` — All solver-tuning flags are dead: --strategy, --simplify, --preset, --auto-tune, --enumerate-models, --optimize, --minimize-model, --theory-opt, --enhanced-errors do nothing *(scope: frontends)*
- [x] `oxiz-cli/src/main.rs:1050` — --unsat-core never enables core production, so it always outputs an error instead of a core *(scope: frontends)*
- [x] `oxiz/src/easy.rs:129` — EasySolver assert_* methods silently drop constraints when the variable name is unknown *(scope: frontends)* — **(fixed: EasySolver assert_* methods silently dropped unknown-name constraints — record_unknown_var now records the error)**
- [x] `oxiz-cli/src/distributed.rs:745` — Distributed cube-and-conquer is fake: cubes assert fresh unconstrained variables, so every worker re-solves the whole problem *(scope: frontends)* — **(fixed: distributed cube-and-conquer was fake — solve_cube now maps cube literals to real hash-consed vars)**
- [x] `oxiz-cli/src/portfolio.rs:39` — --portfolio-mode runs five identical solvers: strategy options are ignored, so there is no diversification *(scope: frontends)* — **(fixed: --portfolio-mode ran five identical solvers — now 5 distinct (theory_mode, simplify, restart) + orderings, with tests)**
- [x] `oxiz-cli/src/interpolate.rs:141` — --interpolate is a placeholder: always returns interpolant 'true' with status 'unknown' *(scope: frontends)*
- [x] `oxiz-cli/src/main.rs:1057` — --validate-model does not validate anything; it just prints the model *(scope: frontends)* — **(fixed: --validate-model did not validate anything — now runs eval_in_model over every assertion and reports OK/FAILED)**
- [x] `oxiz-cli/src/main.rs:429` — --minimize-core, --incremental, --checkpoint/--resume/--resume-from/--checkpoint-interval, and --threads are accepted but never read *(scope: frontends)* — **(fixed: all four previously warn-and-do-nothing CLI flags now drive real behavior: --minimize-core (real deletion-based minimization), --incremental, --checkpoint/--resume, --threads (wave2b ml-cli, TODO-960))**
- [x] `oxiz-cli/src/tptp.rs:949` — TPTP free variables are declared as constants, weakening implicitly universally quantified axioms — can yield wrong SZS status *(scope: frontends)* — **(fixed: TPTP free variables were declared as constants — non-conjecture roles are now universally closed via to_smtlib2_closed)**
- [x] `oxiz-cli/src/dimacs.rs:105` — DIMACS parser rejects valid files: multi-line clauses split into separate clauses and empty clauses (falsum) silently dropped *(scope: frontends)* — **(fixed: DIMACS parser rejected valid files — now whole-stream tokenization, multi-line clauses reassembled, empty clause preserved)**
- [x] `oxiz-cli/src/server.rs:292` — REST API: /check-sat builds scripts with no declarations (always errors, masked as 'unknown'); /model can return another client's model *(scope: frontends)* — **(fixed: REST API /check-sat built scripts with no declarations — now includes a declarations field with per-session Context isolation)**
- [x] `oxiz-sat/src/xor.rs:671` — XorDetector::compute_xor_rhs returns the inverted RHS for every detected XOR constraint *(scope: sat)*
- [x] `oxiz-sat/src/solver/learn.rs:382` — Vivification and inprocessing strengthening mutate clause.lits in place without updating watch lists *(scope: sat)* — **(fixed: vivification/inprocessing strengthening mutated clause.lits in place without updating watches — remove_literal_and_rewatch now rebuilds watches for every removal)**
- [x] `oxiz-sat/src/cube_solver.rs:179` — ParallelCubeSolver/CubeAndConquer never solve: solve_cube ignores the clauses, and an empty cube list yields UNSAT *(scope: sat)* — **(fixed: ParallelCubeSolver/CubeAndConquer never solved — solve_cube now runs CDCL under cube assumptions; empty cube list yields Unknown, not Unsat)**
- [x] `oxiz-sat/src/parallel/proof_check.rs:89` — ParallelProofChecker declares every proof Valid — no step is ever checked *(scope: sat)* — **(fixed: ParallelProofChecker declared every proof Valid — now returns Incomplete (Valid only for the empty proof))**
- [x] `oxiz-sat/src/lib.rs:10` — "DRAT proof generation" is advertised but the CDCL solver never emits proof events; LRAT writer output is malformed *(scope: sat)*
- [x] `oxiz-sat/src/gpu.rs:485` — CpuReferenceAccelerator::batch_unit_propagation fabricates conflicts and units from clause-index modulo *(scope: sat)* — **(fixed: CpuReferenceAccelerator::batch_unit_propagation fabricated conflicts from clause-index modulo — now genuinely evaluates each watched clause)**
- [x] `oxiz-sat/src/assumptions.rs:233` — AssumptionCoreMinimizer::minimize_deletion discards all non-fixed assumptions, returning an empty 'core' *(scope: sat)* — **(fixed: AssumptionCoreMinimizer::minimize_deletion discarded all non-fixed assumptions — now a real deletion loop using solver.solve_with_assumptions)**
- [x] `oxiz-sat/src/portfolio.rs:236` — No resource limits anywhere: Solver::solve has no budget and PortfolioSolver's timeout still joins all threads *(scope: sat)* — **(fixed: no resource limits anywhere — Solver now enforces a max_conflicts budget + interrupt; portfolio uses recv_timeout + worker interrupt)**
- [x] `oxiz-sat/src/xor.rs:1062` — XorSubsumption::find_subsumed returns unverified signature-collision candidates as 'subsumed' *(scope: sat)* — **(fixed: XorSubsumption::find_subsumed returned unverified signature-collision candidates — now re-verifies query.is_subset(existing_vars))**
- [x] `oxiz-core/src/smtlib/parser/commands.rs:292` — (set-info :smt-lib-version 2.6) causes a hard parse error aborting the whole script *(scope: smtlib-compliance)*
- [x] `oxiz-solver/src/context.rs:732` — All solver options except produce-proofs/produce-unsat-cores are accepted and silently ignored (:timeout, :random-seed, :produce-models, memory/conflict/decision limits) *(scope: smtlib-compliance)*
- [x] `oxiz-solver/src/context.rs:885` — :named assertion annotations never reach the solver; get-unsat-core and get-assignment are non-functional end-to-end *(scope: smtlib-compliance)* — **(fixed: `Command::AssertNamed` threads the label through `Context::assert_named` into the solver, and assertion names are now recorded unconditionally so `(get-unsat-core)` also works when `:produce-unsat-cores` is enabled mid-session)**
- [x] `oxiz-solver/src/context.rs:987` — get-info always returns an error — even :all-statistics can never match, and mandatory keywords are unsupported *(scope: smtlib-compliance)* — **(fixed: get-info always returned an error — now strips ':' and handles :all-statistics plus the mandatory keywords)**
- [x] `oxiz-core/src/smtlib/parser/terms.rs:419` — Chainable/n-ary core operators rejected: (= a b c), (< a b c), (=> a b c), (xor a b c), (- a b c) are parse errors *(scope: smtlib-compliance)*
- [x] `oxiz-solver/src/context.rs:762` — :print-success is never implemented, yet get-option reports its default as true *(scope: smtlib-compliance)* — **(fixed: get-option now reports the honest `false` default, and the mode itself is implemented — `execute_script` emits `success` after every command that succeeds without its own response, including `exit`)**
- [x] `oxiz-core/src/smtlib/parser/commands.rs:316` — define-sort body restricted to a bare symbol; parametric aliases silently become uninterpreted sorts *(scope: smtlib-compliance)* — **(fixed: define-sort body was restricted to a bare symbol — compound bodies (Array/BitVec) now resolve; parametric aliases error honestly)**
- [x] `oxiz-solver/src/context.rs:936` — check-sat-assuming emulated via push/assert/pop; get-unsat-assumptions impossible and post-check queries see popped state *(scope: smtlib-compliance)* — **(fixed: check-sat-assuming was emulated via push/assert/pop — now a real check_with_assumptions; get-unsat-assumptions is functional)**
- [x] `oxiz-theories/src/combination.rs:507` — Nelson-Oppen never propagates equalities to arithmetic; EUF propagation extraction pushes trivial self-equalities *(scope: theories-rest)* — **(fixed: Nelson-Oppen never propagated equalities to arithmetic — now real arith get_shared_equalities + EUF extraction, no (lit,lit) placeholder)**
- [x] `oxiz-theories/src/combination.rs:587` — check_nelson_oppen loops forever once any two shared variables are EUF-equal *(scope: theories-rest)* — **(fixed: check_nelson_oppen could loop forever — seen_pairs dedup + an n^2+16 iteration cap now returns Unknown instead, with a regression test)**
- [x] `oxiz-theories/src/combination.rs:416` — Polite combination fabricates an all-disequal arrangement and asserts it into EUF, producing wrong UNSAT *(scope: theories-rest)* — **(fixed: polite combination fabricated an all-disequal arrangement — extract_arrangement_from_arith now groups by model value)**
- [x] `oxiz-theories/src/combination.rs:627` — Model-based combination never asserts the arrangement into arithmetic and misreports arrangement failure as global UNSAT *(scope: theories-rest)* — **(fixed: model-based combination never asserted the arrangement into arithmetic — now asserts equalities via notify_equality and attributes conflicts)**
- [x] `oxiz-theories/src/string/solver.rs:651` — check_lengths detects length-constraint violations but silently drops them *(scope: theories-rest)* — **(fixed: check_lengths detected length-constraint violations but silently dropped them — check() now returns Unsat(conflict))**
- [x] `oxiz-theories/src/string/solver.rs:525` — StringSolver::check() returns Sat with unresolved word equations and unchecked regex constraints *(scope: theories-rest)* — **(fixed: StringSolver::check() returned Sat with unresolved constraints — honesty gate added: unresolved eqs / unassigned regex now yield Unknown)**
- [x] `oxiz-theories/src/euf/solver.rs:966` — EufSolver::assert_false asserts node != node, making any negated assertion an instant contradiction *(scope: theories-rest)* — **(fixed: EufSolver::assert_false asserted node != node — now only interns the term and returns Sat)**
- [x] `oxiz-theories/src/array/solver.rs:330` — Read-over-write-diff axiom fires on 'not currently equal' instead of 'proven disequal' indices *(scope: theories-rest)* — **(fixed: read-over-write-diff axiom fired on 'not currently equal' — now gated by is_proven_disequal, not !are_equal)**
- [x] `oxiz-theories/src/array/solver.rs:372` — Array conflict explanations omit the equality chain, yielding over-strong learned clauses *(scope: theories-rest)* — **(fixed: array conflict explanations omitted the equality chain — explain_equal chain + diseq reason now included)**
- [x] `oxiz-theories/src/datatype/solver.rs:419` — Datatype theory has no acyclicity (occurs) check — cyclic constructor terms reported Sat *(scope: theories-rest)* — **(fixed: datatype theory had no acyclicity check — check_acyclicity now does a three-colour DFS over the constructor class graph)**
- [x] `oxiz-theories/src/datatype/solver.rs:579` — DatatypeSolver::pop() restores only constraints; constructor tags and app maps leak across backtracking *(scope: theories-rest)* — **(fixed: DatatypeSolver::pop() restored only constraints — DtTrailEntry now unwinds constructor/selector/recognizer/excluded maps)**
- [x] `oxiz-theories/src/combination.rs:893` — verify_model always returns true; complete_model and extract_assignments are identity stubs *(scope: theories-rest)* — **(fixed: verify_model always returned true — verify_model now cross-theory checks with real union-find extract_assignments; complete_model remains an honest documented identity pass-through)**
- [x] `bench/z3_parity/src/comparator.rs:25` — Parity comparator counts Unknown-vs-any-answer as 'Correct', so 100% parity is achievable by always answering unknown *(scope: test-gap)*
- [x] `oxiz-solver/tests/property_based.rs:6` — Entire oxiz-solver (and oxiz-core) property-based suites are disabled by default behind a non-default 'property-tests' feature *(scope: test-gap)* — **(fixed: `oxiz-solver`'s `property-tests` feature was already default-on; `oxiz-core`'s joined its default set in 0.3.3 — `default = ["std", "scripting", "property-tests"]` — after a runtime-cost review found the suites add 89 tests in 0.08s, not the reason they were off.)**
- [x] `oxiz-solver/tests/property_tests/backtrack_properties.rs:96` — Property tests accept Unknown for both SAT-expected and UNSAT-expected outcomes — an always-Unknown solver passes the suite *(scope: test-gap)* — **(fixed: property tests accepted Unknown for both outcomes — now strict prop_assert_eq!(result, Sat), no Unknown anywhere)**
- [x] `oxiz-solver/tests/property_tests/model_properties.rs:32` — All model-validity property tests are vacuously guarded by 'if result == Sat' and never assert the result itself *(scope: test-gap)* — **(fixed: model-validity property tests were vacuously guarded — now assert result==Sat then model.is_some(), not a vacuous if-guard)**
- [x] `oxiz-solver/tests/mbqi_tests/integration_tests.rs:37` — MBQI 'integration tests' are dead code (not referenced by any mod) and vacuous — quantifier instantiation has no end-to-end solving test *(scope: test-gap)* — **(fixed: MBQI 'integration tests' were dead code — now wired via audit_sweep_solver.rs with a real end-to-end UNSAT-via-instantiation test)**
- [x] `oxiz-cli/tests/smtlib_benchmarks.rs:95` — Benchmark 'pass' criterion is output containing sat/unsat/unknown; expected status never compared; test never fails by design *(scope: test-gap)* — **(fixed: benchmark pass criterion was just output-contains-sat/unsat/unknown — now actual_status exact-match against the declared :status)**
- [x] `oxiz-spacer/tests/integration_tests.rs:16` — All four oxiz-spacer end-to-end integration tests are #[ignore]d — the published CHC/PDR engine has zero running end-to-end tests *(scope: test-gap)* — **(fixed: all four oxiz-spacer end-to-end tests were #[ignore]d — only one Unsafe test remains ignored (documented engine limit); Safe/parser tests now run)**
- [x] `fuzz/fuzz_targets/fuzz_solver.rs:202` — All fuzz targets are crash-only with no soundness oracle; the parser-to-solver end-to-end fuzz path is dead code *(scope: test-gap)* — **(fixed: all fuzz targets were crash-only — assert_model_satisfies soundness oracle + idempotence + a new fuzz_parse_and_solve target added)**
- [x] `oxiz-opt/src/pmres.rs:482` — MaxSAT algorithms (PMRES, SortMax) fail their simplest correctness tests, which were #[ignore]d instead of fixed *(scope: test-gap)* — **(fixed: PMRES/SortMax tests were #[ignore]d instead of fixed — none remain ignored in oxiz-opt; pmres tests assert real costs)**
- [x] `oxiz-solver/tests/nlsat_integration.rs:351` — NLSAT integration tests accept Unknown in 15 of the assertions, including for the trivially UNSAT x<0 AND x>0 *(scope: test-gap)* — **(fixed: NLSAT integration tests accepted Unknown broadly — x<0 AND x>0 now asserts Unsat; other Unknown-acceptances tightened to Sat)**
- [x] `CHANGELOG.md:8` — CHANGELOG [0.2.4] section is completely empty despite ~6,000 lines of changes since 0.2.3 *(scope: release-audit)*
- [x] `README.md:24` — README 'What's New in 0.2.4 (Unreleased)' actually lists the already-released 0.2.3 features *(scope: release-audit)* — fixed: section now reads "What's New in 0.2.4 (2026-07-19)" and its content (oxiz-py string/FP/quantifier bindings, diagnostics cleanup, production-readiness audit) matches the actual `[0.2.4]` CHANGELOG entry
- [x] `README.md:333` — Supported Logics table marks QF_NRA/UFLIA/AUFBV/HORN 'Complete', contradicting the README's own Alpha/partial status 200 lines earlier *(scope: release-audit)* — verified: table now correctly shows these as 🔶 Alpha/Partial, consistent with the rest of the README
- [x] `bench/profile/Cargo.toml:2` — bench-profile workspace member lacks publish = false (and license/description), breaking workspace publish *(scope: release-audit)* — **(fixed: bench-profile Cargo.toml now has publish=false, license, and description)**
- [x] `.cargo/config.toml:10` — Committed cargo config forces target-cpu=native for all source builds and -undefined dynamic_lookup on every macOS link *(scope: release-audit)* — **(fixed: target-cpu=native removed (SIGILL portability danger gone); -undefined dynamic_lookup retained but documented as required for PyO3 maturin)**
- [x] `oxiz-core/src/rewrite/arith.rs:150` — Rational64 (i64) constant folding overflows: wrong constants in release, abort in dev *(scope: panic-audit)* — **(fixed: arith constant-folding now uses checked_add/mul/sub/div_euclid throughout instead of overflowing i64 ops)**
- [x] `oxiz-solver/src/solver/types.rs:231` — timeout option accepted but never enforced anywhere: solver can hang forever *(scope: panic-audit)* — **(fixed: timeout is now enforced via a wall-clock deadline)**
- [x] `oxiz-core/src/ast/manager/builder.rs:948` — mk_bv_extract computes width = high - low + 1 with unvalidated parser indices: u32 underflow *(scope: panic-audit)* — **(fixed: mk_bv_extract now uses checked_sub/checked_add, falling back to a 1-bit result instead of underflowing)**
- [x] `oxiz-core/src/ast/validation.rs:191` — Model validation masks with (1u64 << width) - 1 unguarded for width >= 64 *(scope: panic-audit)* — **(fixed: both validation masks now guard width>=64 before shifting)**
- [x] `oxiz-core/src/tactic/bv/advanced_rewriter.rs:549` — AdvancedBvRewriter publicly exported with placeholder term constructors returning Ok(0) *(scope: panic-audit)* — **(fixed: AdvancedBvRewriter's fake `pub type TermId = usize` renamed to an honest BvHandle with a sound interned-constant handle space, replacing placeholder Ok(0) constructors (wave2b core-tactics, TODO-1012))**
- [x] `oxiz-sat/src/dimacs.rs:112` — DIMACS header var count triggers unbounded allocation: `p cnf 999999999999 1` hangs/OOMs *(scope: panic-audit)* — **(fixed: DIMACS DEFAULT_MAX_VARS (1<<31) now rejects adversarial var counts instead of allocating unbounded memory)**
- [x] `oxiz-proof/src/checker.rs:279` — CheckerConfig::verify_conclusions is accepted but never read — ProofChecker validates structure only *(scope: opt-proof)* — **(fixed: CheckerConfig.verify_conclusions is now read (checker.rs:409/591) and gates semantic verification, with extensive tests)**
- [x] `oxiz-opt/src/maxsmt.rs:305` — MaxSmtSolver is a hollow stub: solve paths always return Unknown *(scope: opt-proof)* — **(fixed: MaxSmtSolver.solve() now errors honestly (RequiresTermManager); solve_with implements selector-encoding + binary search)**
- [x] `oxiz-opt/src/maxsat/core.rs:13` — Weight derives Ord: any Int compares less than any Rational regardless of numeric value *(scope: opt-proof)* — **(fixed: Weight now has a manual value-based Ord/Eq/Hash (cmp_value promotes to rationals) instead of comparing by variant)**
- [x] `oxiz-opt/src/maxsat/algorithms.rs:52` — check_hard_satisfiable resets lower/upper bounds to zero, wiping accumulated MaxSAT cost *(scope: opt-proof)* — **(fixed: check_hard_satisfiable now preserves lower_bound (sets upper=lower) instead of resetting both to zero)**
- [x] `oxiz-opt/src/maxsat/algorithms.rs:714` — PMRES builds jointly-unsatisfiable assumptions after multi-clause cores, inflating lower bound *(scope: opt-proof)* — **(fixed: PMRES now delegates to solve_fu_malik, removing the jointly-unsat-assumptions bug)**
- [x] `oxiz-opt/src/maxsat/algorithms.rs:491` — OLL core-merging is faked: 'just increase the bound of the first group' *(scope: opt-proof)* — **(fixed: OLL now merges all intersecting groups and sums bounds+1 instead of just bumping the first group)**
- [x] `oxiz-opt/src/hybrid.rs:190` — HybridSolver has no hard-clause support and maps exact-solver Unknown to Optimal *(scope: opt-proof)* — **(fixed: HybridSolver now adds hard clauses to both SLS and the exact solver, and propagates the exact verdict)**
- [x] `oxiz-opt/src/maxhs.rs:151` — MaxHS placeholder uses greedy hitting sets yet reports MaxSatResult::Optimal *(scope: opt-proof)* — **(fixed: MaxHS now computes the exact min-cost hitting set via MaxSatSolver, returning Unknown on genuine uncertainty)**
- [x] `oxiz-opt/src/omt.rs:527` — optimize_binary_search claims Optimal when iteration budget is exhausted or bounds are mixed-type *(scope: opt-proof)* — **(fixed: optimize_binary_search now claims Optimal only when converged; produces Unbounded otherwise)**
- [x] `oxiz-proof/src/conversion.rs:141` — drat_to_alethe fabricates proof structure with 'first 5 clauses are Input' and 'last two steps as premises' heuristics *(scope: opt-proof)* — **(fixed: drat_to_alethe now matches real input clauses and returns InformationLoss for derived ones instead of fabricating proof structure)**
- [x] `oxiz-opt/src/preprocess.rs:72` — Bounded variable elimination on soft clauses is enabled by default but does not preserve MaxSAT optima *(scope: opt-proof)* — **(fixed: bounded variable elimination on soft clauses is now off by default, restricted to all-infinite-weight occurrences)**
- [x] `oxiz-opt/src/context.rs:141` — OptConfig.timeout_ms and objective priorities are accepted but silently ignored *(scope: opt-proof)* — **(fixed: OptConfig.timeout_ms is now enforced (new_solver + deadline); objective priorities are honored (sort_by_key))**
- [x] `oxiz-opt/src/context.rs:856` — is_soft_satisfied cannot evaluate compound terms, so cost() over-reports for non-variable soft constraints *(scope: opt-proof)* — **(fixed: is_soft_satisfied now recursively evaluates compound terms via model_eval_bool)**
- [x] `oxiz-spacer/src/bmc.rs:363` — run_kinduction falls through to Safe(max_depth) after only Unknown results *(scope: spacer)* — **(fixed: run_kinduction now returns Unknown (not Safe(max_depth)) when every round is Unknown)**
- [x] `oxiz-spacer/src/parser.rs:517` — Decimal literals silently parsed as integer 0 *(scope: spacer)* — **(fixed: decimal literals are now parsed to exact Rational64 instead of integer 0)**
- [x] `oxiz-spacer/src/distributed.rs:363` — Distributed PDR is a simulation: workers 'block' POBs by parity and coordinator sleeps *(scope: spacer)* — **(fixed: distributed PDR now delegates to the sound sequential Spacer (further upgraded to a genuine multi-thread parallel portfolio — see 'Remaining' / spacer-distributed-no-real-parallelism, wave2))**
- [x] `oxiz-spacer/src/parallel.rs:373` — ParallelPropagator reports every lemma as propagated without any inductiveness check *(scope: spacer)* — **(fixed: ParallelPropagator now checks is_lemma_inductive per lemma instead of reporting every lemma as propagated)**
- [x] `oxiz-spacer/src/frames.rs:562` — FrameManager::propagate pushes all lemmas unconditionally and declares fixpoint on first call *(scope: spacer)* — **(fixed: FrameManager::propagate now checks inductiveness before pushing, with a proper fixpoint instead of an unconditional first-call declaration)**
- [x] `oxiz-spacer/src/existential.rs:75` — Existential handling is a no-op: existential_vars never populated, skolem substitution never applied *(scope: spacer)* — **(fixed: ExistentialInfo::analyze now walks head args and populates existential_vars instead of leaving them empty)**
- [x] `oxiz-spacer/src/existential.rs:622` — WitnessExtractor::extract_witnesses assigns an arbitrary model entry to every existential variable *(scope: spacer)* — **(fixed: WitnessExtractor now matches witnesses by resolved var name instead of assigning an arbitrary model entry)**
- [x] `oxiz-spacer/src/theory.rs:507` — theory_generalize rewrites x<c to x<=c while claiming the equivalent x<=c-1 *(scope: spacer)* — **(fixed: theory_generalize now converts integer x<b to the exact x<=b-1, keeping reals as-is)**
- [x] `oxiz-spacer/src/theory.rs:160` — project_variables recurses through Not, turning over-approximation into under-approximation *(scope: spacer)* — **(fixed: project_not now correctly over-approximates negation (De Morgan / atomic->true) instead of recursing into under-approximation)**
- [x] `oxiz-spacer/src/tactics/bmc_unroll.rs:135` — BMC unroll renaming silently skips Div/Mod/Neg and other term kinds *(scope: spacer)* — **(fixed: bmc_unroll rename_term now uses exhaustive TermManager::substitute instead of silently skipping Div/Mod/Neg and other kinds)**
- [x] `oxiz-solver/src/optimization.rs:360` — Unbounded objectives reported as Optimal with an arbitrary value; Unbounded variant is never produced *(scope: solver-rest)* — **(fixed: optimization now produces a real Unbounded variant instead of reporting an arbitrary value as Optimal)**
- [x] `oxiz-solver/src/optimization.rs:405` — Real optimization converts BigInt objective values via string parse with unwrap_or(0), silently corrupting values beyond i64 *(scope: solver-rest)* — **(fixed: optimize_real now uses exact BigInt/Rational instead of string-parse-with-unwrap_or(0))**
- [x] `oxiz-solver/src/optimization.rs:219` — Lexicographic optimize() pushes scopes that are never popped, permanently constraining the solver *(scope: solver-rest)* — **(fixed: lexicographic optimize now rebuilds a fresh solver per query — no more permanent scope leak)**
- [x] `oxiz-solver/src/optimization.rs:552` — pareto_optimize returns dominated points: exclusion constraint only requires one objective to improve and no dominance filtering is applied *(scope: solver-rest)* — **(fixed: pareto_optimize now filters dominated points via dominates() instead of admitting them)**
- [x] `oxiz-solver/src/mbqi/counterexample.rs:1198` — MBQI model evaluator uses Rust truncated division/remainder instead of SMT-LIB Euclidean div/mod *(scope: solver-rest)* — **(fixed: MBQI eval_div/eval_modulo now use euclidean_div_rem instead of Rust truncated division/remainder)**
- [x] `oxiz-solver/src/solver/mod.rs:761` — Wall-clock timeout is accepted through three APIs but never enforced during solving *(scope: solver-rest)* — **(fixed: wall-clock timeout is now enforced between MBQI rounds and mid-search inside theory callbacks)**
- [x] `oxiz-cli/src/portfolio.rs:63` — Portfolio 'strategies' are all identical: every strategy option is silently ignored by Context::set_option, and losing threads are never cancelled *(scope: solver-rest)* — **(fixed: portfolio strategies now have distinct SolverConfig + AssertOrdering (tests enforce distinctness); mid-execute cancellation documented as a fundamental Rust limitation with a cooperative solved-flag)**
- [x] `oxiz-solver/src/combination/coordinator.rs:326` — TheoryCoordinator never identifies shared terms (placeholder no-op) and operates on placeholder usize TermIds *(scope: solver-rest)* — **(fixed: TheoryCoordinator.identify_shared_terms now does real Nelson-Oppen multi-theory detection from its bookkeeping instead of a placeholder no-op)**
- [x] `oxiz-solver/src/model/advanced_builder.rs:254` — AdvancedModelBuilder is an all-placeholder scaffold publicly exported from model/ *(scope: solver-rest)* — **(fixed: AdvancedModelBuilder deleted entirely — it was an all-placeholder scaffold (update_arithmetic_bounds hardcoded var=0); decided DELETE per the finding's own guidance, no functionality lost (wave2b solver-hard, TODO-1045))**
- [x] `oxiz-solver/src/combination/convexity.rs:347` — CaseSplitStrategy::Lazy causes an infinite loop in process_disjunctions *(scope: solver-rest)* — **(fixed: CaseSplitStrategy::Lazy now defers disjunctions (held aside, restored after the loop), terminating instead of looping forever)**
- [x] `oxiz-solver/src/combination/convexity.rs:640` — simplify_with_equality implements disequality semantics (and can derive bogus unit equalities); has_conflict always returns false *(scope: solver-rest)* — **(fixed: simplify_with_equality now drops satisfied disjunctions correctly (TRUE semantics); has_conflict checks empty disjunctions)**
- [x] `oxiz-solver/src/solver/check_array.rs:438` — Array pre-check treats Eq nested inside a Bool equality as asserted *(scope: solver-core)* — **(fixed: check_array now descends equality operands with collect_facts=false, so nested Eq is no longer treated as asserted)**
- [x] `oxiz-solver/src/solver/encode.rs:321` — Arithmetic atoms with Div/Mod/nonlinear/oversized constants are silently unconstrained *(scope: solver-core)* — **(fixed: arith_atoms_need_theory honesty gate now returns Unknown for Div/Mod/nonlinear/oversized atoms instead of silently unconstraining them)**
- [x] `oxiz-solver/src/solver/theory_manager.rs:1574` — final_check maps arith Unknown and Err to Sat *(scope: solver-core)* — **(fixed: final_check now sets resource_exhausted on arith Unknown/Err, so the owning solver honestly answers Unknown instead of Sat)**
- [x] `oxiz-solver/src/solver/mod.rs:879` — reset() leaves MBQI quantifiers, e-matching state and has_quantifiers stale *(scope: solver-core)* — **(fixed: reset() now rebuilds MBQI/e-matching engines and clears has_quantifiers instead of leaving them stale)**
- [x] `oxiz-solver/src/solver/encode.rs:1277` — FP and String theory atoms are free booleans; 'theory solver handles these' does not exist *(scope: solver-core)* — **(fixed: string_atoms_need_theory/fp_atoms_need_theory honesty gates added — Unknown instead of free-boolean unconstrained atoms)**
- [x] `oxiz-solver/src/solver/check_array.rs:10` — Array theory decided only by syntactic pre-checks; no axiom instantiation in the solving loop *(scope: solver-core)* — **(fixed: real lazy array-axiom instantiation added inside the CDCL(T) loop (new oxiz-solver/src/solver/array_axioms.rs::instantiate_array_axioms) — array theory is no longer decided by syntactic pre-checks alone (wave2b solver-hard, TODO-1053))**
- [x] `oxiz-solver/src/solver/encode.rs:983` — TermKind::Let encoding silently drops bindings *(scope: solver-core)* — **(re-verified: not a bug — TermKind::Let encoding does drop bindings, but the parser pre-substitutes let-bound variables into the body before the solver ever sees the term, so the SMT-LIB solve path never observes a wrong result)**
- [x] `oxiz-solver/src/solver/encode.rs:695` — Unbounded recursion in encode/simplify/collectors can overflow the stack on deep formulas *(scope: solver-core)* — **(fixed: encode_depth ENCODE_DEPTH_LIMIT guard + an explicit-stack scan added; encode_depth_exceeded now yields Unknown instead of overflowing the stack)**
- [x] `oxiz-solver/src/solver/theory_manager.rs:866` — Conflict clauses silently drop literals for terms without SAT vars and ignore assignment polarity *(scope: solver-core)* — **(fixed: terms_to_conflict_clause now respects assigned_polarity (Some(true)->neg, Some(false)->pos) instead of dropping literals)**
- [x] `oxiz-wasm/src/js_api/model.rs:223` — getUnsatCore returns ALL assertions, not an unsat core *(scope: bindings)* — **(fixed: WASM getUnsatCore now actually executes get-unsat-core (errors if not enabled) instead of returning all assertions)**
- [x] `oxiz-wasm/src/js_api/worker_support.rs:456` — WorkerHandler "solve" task silently drops failed assertions, then answers sat *(scope: bindings)* — **(fixed: WorkerHandler solve now aborts with an error on a failed assertion instead of silently dropping it and answering sat)**
- [x] `oxiz-wasm/src/js_api/worker_support.rs:281` — WorkerPool never spawns workers and never executes submitted tasks *(scope: bindings)* — **(fixed: WorkerPool.run_one now actually executes handle_task via execute/drainQueue instead of never spawning workers)**
- [x] `oxiz-wasm/src/js_api/solver_core.rs:449` — executeAsync/executeWithProgress split scripts at 20-line boundaries, breaking multi-line s-expressions *(scope: bindings)* — **(fixed: executeAsync/executeWithProgress now use split_into_commands (balanced s-expressions), chunking by 20 commands, not 20 lines)**
- [x] `oxiz-py/src/solver_py.rs:424` — Python model() truncates bitvector values wider than 64 bits to the low limb *(scope: bindings)* — **(fixed: oxiz-py model() now keeps the full BigInt bitvector value + width instead of truncating to the low 64 bits)**
- [x] `oxiz-wasm/package.json:57` — npm exports.require points to pkg-nodejs which is neither built by prepublishOnly nor included in files *(scope: bindings)* — **(fixed: package.json prepublishOnly now builds pkg-nodejs (build:nodejs) and files[] includes it)**
- [x] `oxiz-wasm/src/js_api/streaming.rs:345` — StreamingSolver.nextModelEntry always returns None; startModelStream returns a disconnected controller *(scope: bindings)* — **(fixed: streaming nextModelEntry now yields real entries; startModelStream returns a connected Rc-shared controller)**
- [x] `oxiz-wasm/src/js_api/memory_management.rs:313` — MemoryManager.allocate immediately drops the buffer; allocate/free are no-ops *(scope: bindings)* — **(fixed: MemoryManager.allocate now keeps the buffer alive in self.buffers; get/set/free are functional, not no-ops)**
- [x] `oxiz-wasm/src/lazy_loader.rs:556` — LazyLoader fetches theory module bytes but never instantiates them, yet marks theories loaded *(scope: bindings)* — **(fixed: LazyLoader now instantiate_buffer's and verifies 'instance' before marking a theory loaded)**
- [x] `oxiz-wasm/src/js_api/optimize.rs:641` — eliminateQuantifiers is an advertised public API that always fails for quantified input *(scope: bindings)* — **(fixed: eliminateQuantifiers now uses QeLiteSolver (handles trivial bodies + top-level quantifiers) with an honest NotSupported for general QE)**

### P4 — Minor / Downgraded (polish, perf, docs)

**bindings**:
- [x] `oxiz-wasm/src/js_api/solver_core.rs:332` — cancel() flag is never observed by checkSat/checkSatAsync; documented cancellation cannot work — **(fixed: solver_core.rs check_sat now honors self.cancelled -> returns 'unknown'; check_sat_async delegates; regression tests present)**
- [x] `oxiz-py/oxiz.pyi:52` — Type stubs omit ~20 exported symbols; theories.rs docstring examples use wrong argument order — **(fixed: oxiz.pyi now lists all 9 classes + 27 functions; theories.rs docstrings use the correct argument order (fp_add(tm,"RNE",a,b); ForAll(tm,vars,body)))**
- [x] `oxiz-wasm/src/js_api/diagnostics.rs:160` — getStatistics num_assertions is always 0 (counts "(assert" in output that never contains it) — **(fixed: diagnostics.rs getStatistics now uses ctx.get_assertions().len() instead of a substring count)**
- [x] `oxiz-py/src/solver_py.rs:337` — set_timeout/set_option("timeout") reset the entire SolverConfig to defaults — **(fixed: solver_py.rs timeout path now clones config().with_timeout(ms); regression tests confirm other fields are preserved)**
- [x] `oxiz-vscode/src/extension.ts:322` — findOxizPath uses Unix `test -x` via execSync — workspace-local binary detection breaks on Windows — **(fixed: extension.ts now uses fs.accessSync(X_OK) with a win32 branch + oxiz.exe; the Unix-only `test -x` execSync check was removed)**

**core-ast**:
- [x] `oxiz-core/src/rewrite/bv.rs:222` — BvXor rewrite falls back to returning an OR term: x ^ y becomes x | y — **(fixed: bv.rs rewrite_bvxor fallback now rebuilds mk_bv_xor (not mk_bv_or); a regression test asserts the term stays BvXor)**
- [x] `oxiz-core/src/rewrite/arith.rs:404` — Integer mod constant folding uses Rust truncated % instead of SMT-LIB Euclidean mod — **(fixed: arith.rs rewrite_mod now uses checked_rem_euclid (Euclidean) instead of Rust truncated %)**
- [x] `oxiz-core/src/rewrite/arith.rs:366` — Div constant folding treats Int division as rational: (div 7 2) folds to real 3.5 — **(fixed: arith.rs rewrite_div now uses checked_div_euclid; division by zero is left uninterpreted instead of folded)**
- [x] `oxiz-core/src/ast/egraph.rs:390` — EGraph::add_term maps any IntConst that overflows i64 to 0 and silently drops unconvertible children — **(fixed: egraph.rs add_term now uses i64::try_from(val).ok()? (no 0-mapping) and collect::<Option<Vec>>()? for children)**
- [x] `oxiz-core/src/ast/congruence.rs:198` — CongruenceClosure::pop does not undo diseqs (or rank/explanations) — **(fixed: congruence.rs pop() now undoes DiseqInsert/RankChange/ExplanationInsert via an UndoOp trail)**
- [x] `oxiz-core/src/ast/congruence.rs:434` — close() skips use-lists of merged terms sharing a root, missing congruence propagation — **(fixed: congruence.rs close() now gathers use-lists from every class member, not just the term/root)**
- [x] `oxiz-core/src/rewrite/fp.rs:246` — FP rules assume symbolic operands are finite: inf + x -> inf and x/inf -> +0 are unsound — **(fixed: fp.rs inf+x and x/inf rules are now guarded by is_finite(other); tests confirm a symbolic operand is not folded)**
- [x] `oxiz-core/src/rewrite/string.rs:316` — String folding is byte-based, not codepoint-based; indexof slicing can panic on non-ASCII — **(fixed: string.rs indexof now uses char_indices/chars().count() with codepoint<->byte conversion instead of byte slicing)**
- [x] `oxiz-core/src/ast/manager/query.rs:835` — free_vars counts quantifier-bound variables as free (acknowledged stub) — **(fixed: iterative `free_vars_with` walk tracks a `(name, sort) -> depth` bound map; `free_vars_including_patterns` covers the name-choice callers)**
- [x] `oxiz-core/src/ast/egraph.rs:346` — EGraph extract/get_class/extract_best use one-level union-find lookup and fail after chained merges — **(fixed: egraph.rs extract/get_class/extract_best all use find_canonical (a full union-find walk) instead of one-level lookup)**
- [x] `oxiz-core/src/rewrite/uf.rs:210` — UF congruence cache keyed by 64-bit hash of args can return a different application on collision — **(fixed: uf.rs congruence_cache is now keyed by (Spur, SmallVec<[TermId;4]>) exact args, not a 64-bit hash prone to collisions)**
- [x] `oxiz-core/src/rewrite/arith.rs:172` — Add/Mul folding inserts Int constants into Real-sorted n-ary terms — **(fixed: arith.rs rewrite_add/rewrite_mul now pick want_int from the first operand's actual sort)**
- [x] `oxiz-core/src/rewrite/string.rs:405` — str.to_int folding accepts "+5" and rejects >i64 digit strings — **(fixed: string.rs str_to_int now rejects non-digit input (chars().all(is_ascii_digit), rejects "+5") and parses via BigInt (no i64 cap))**

**core-rest**:
- [x] `oxiz-core/src/qe/datatype/case_analysis.rs:164` — Datatype case analysis returns N copies of the original formula yet reports complete: true — **(fixed: case_analysis.rs now always reports complete:false with a warning doc comment, removing the false complete:true soundness claim)**
- [x] `oxiz-core/src/theories/datatype.rs:273` — DatatypeTheory::axiom_to_term emits mk_true/mk_false placeholders instead of real axioms — **(fixed in 0.3.3: all five `oxiz-core::theories` modules now implement `Theory` with real fixpoint propagation and conflict detection over a shared union-find; `axiom_to_term` returns `Option<TermId>` and builds real premises. Still has zero internal callers — this submodule is not wired into any `oxiz-solver` solve path, and the module's own `# Scope` doc says so: "none of these theories is a decision procedure". Do not read this as affecting verdicts.)**
- [x] `oxiz-core/src/theories/bitvector.rs:309` — oxiz-core BV and FP theory solvers are decorative: propagate/check_for_conflicts do nothing — **(fixed in 0.3.3: `BitVectorTheory`/`FloatingPointTheory` (and `ArrayTheory`/`DatatypeTheory`/`StringTheory`) now do constant folding, operator congruence, and real fixpoint propagation/conflict detection instead of a no-op `{ self.propagations += 1; Vec::new() }`. Still uncalled from any `oxiz-solver` path — see the note above.)**
- [x] `oxiz-core/src/model/completion.rs:128` — Model completion assigns wrong-sorted defaults: variables get Uninterpreted values, sorts guessed by magic ids — **(fixed: completion.rs complete_term now uses t.sort (the actual sort) via factory.default_value instead of guessing a magic SortId)**
- [x] `oxiz-core/src/qe/qe_lite.rs:140` — QeLiteSolver eliminates a quantifier only when the body is literally `true` — **(fixed: QeLiteSolver now performs real cheap QE (unused-variable elimination, equality substitution, etc.) instead of only succeeding when the body is literally true (wave2 qe-arith, P4-1097))**
- [x] `oxiz-core/src/smtlib/printer/model.rs:55` — Model printer emits syntactically invalid output for function interpretations — **(fixed: printer/model.rs write_function_interpretation now emits a valid (define-fun name (params) sort ite-chain))**
- [x] `oxiz-core/src/model/evaluator.rs:346` — TermKind::Div on integers evaluated as exact rational division, not SMT-LIB euclidean div — **(fixed: evaluator.rs eval_div now checks int_sorted and uses checked_div_euclid for Int; exact rational division only for Real)**
- [x] `oxiz-core/src/smtlib/oxiz-core/src/smtlib/printer/config.rs:1` — Stray junk files inside src/: nested duplicate config.rs and .txt scratch files (directory removed; verified absent from the tree)

**core-tactic**:
- [x] `oxiz-core/src/tactic/ackermann.rs:107` — Ackermannization replaces function applications under quantifiers with ground fresh variables — **(fixed: collect_func_apps now threads a bound-name stack through Forall/Exists/Let and taints function symbols under quantifiers instead of ackermannizing them away (wave2b core-tactics, P4-1103))**
- [x] `oxiz-core/src/tactic/solve_eqs.rs:696` — FourierMotzkinTactic performs real-valued elimination on integer variables and can answer Sat wrongly — **(fixed: FourierMotzkinTactic now classifies each variable's sort and uses an Omega-test exact shadow for integer variables instead of unsound real-valued elimination (wave2b core-tactics, P4-1104))**
- [x] `oxiz-core/src/tactic/solve_eqs.rs:1016` — coeff_to_term truncates rational constants to integers, changing constraint semantics — **(fixed: solve_eqs.rs coeff_to_term now builds an exact mk_real(Rational64) after gcd reduction; integer fallback only for huge ratios)**
- [x] `oxiz-core/src/tactic/pb2bv.rs:108` — Pb2BvTactic silently drops the constant offset of linear pseudo-boolean sums — **(fixed: pb2bv.rs extract_pb_constraint now folds the lhs constant offset into the bound)**
- [x] `oxiz-core/src/tactic/registry.rs:165` — Every tactic in default_registry is a no-op or always-NotApplicable stub — **(fixed: every manager-requiring stateless tactic in default_registry now returns a real transformed goal instead of a no-op/NotApplicable stub (wave2b core-tactics, P4-1107))**
- [x] `oxiz-core/src/tactic/combinators.rs:41` — ThenTactic returns a single subgoal's Solved verdict as the answer for the whole goal set — **(fixed: combinators.rs ThenTactic now accumulates all subgoals; short-circuits only on Unsat; Solved(Sat) only when all subgoals are discharged)**
- [x] `oxiz-core/src/tactic/core/mod.rs:68` — TacticResult has no model converter — variable-eliminating tactics lose model information — **(fixed: added a ModelConverter mechanism (TacticModel, ModelConverter trait, IdentityConverter, ChainConverter) so variable-eliminating tactics no longer lose model information (wave2b core-tactics, P4-1109))**
- [x] `oxiz-core/src/tactic/core/goal_refinement.rs:210` — goal_refinement.rs (695 lines) is orphaned: not in any module tree and cannot compile — **(fixed: dead file deleted)**
- [x] `oxiz-core/src/tactic/core/split_clause.rs:172` — SplitClauseTactic allocates literal 0 as its first fresh variable, breaking clause semantics — **(fixed: split_clause.rs next_var now starts at 1 (literal 0 is invalid, documented in a comment))**
- [x] `oxiz-core/src/tactic/core/ctx_solver_simplify.rs:224` — core/ctx_solver_simplify.rs is a 580-line dead placeholder with fake TermId and always-false oracles — **(fixed: dead file deleted; the live `tactic/ctx_simplify.rs` is the only context-simplification tactic)**
- [x] `oxiz-core/src/tactic/combinators.rs:352` — TimeoutTactic leaks the worker thread after timeout with no cancellation — **(fixed: shared `Arc<AtomicBool>` cancellation flag observed via `cancellation_requested()`, worker handle always eventually joined)**

**frontends**:
- [x] `oxiz-cli/src/processor.rs:122` — --stats always reports 0 decisions/propagations/conflicts/restarts — **(fixed: processor.rs --stats now uses aggregated_sat_stats for decisions/propagations/conflicts/restarts instead of always reporting 0)**
- [x] `oxiz-cli/src/format.rs:638` — -o with multiple input files overwrites the output file per result, keeping only the last — **(fixed: format.rs -o now accumulates all results into one write (fs::write once) across smtlib/json/yaml, instead of overwriting per result)**
- [x] `oxiz-cli/src/main.rs:1170` — Solver/parse errors exit with code 0 unless --cicd-strict is set — **(fixed: processor.rs now exits 1 whenever !args.cicd && stats.error_count>0, no longer requiring --cicd-strict)**
- [x] `oxiz/README.md:90` — Facade README documents a nonexistent 'solver' feature flag and a stale version, alongside a 'production-ready' parity claim — **(fixed: the facade README now shows `Version: 0.3.1`, states explicitly that the core solver is not gated behind a `solver` feature, and carries no 'production-ready' parity claim)**

**math**:
- [x] `oxiz-math/src/interval.rs:474` — Interval::mul openness handling excludes attainable value 0, producing intervals that miss true values — **(fixed: Interval::mul openness handling no longer excludes the attainable value 0)**
- [x] `oxiz-math/src/grobner/buchberger.rs:944` — check_equalities uses complex Nullstellensatz criterion to answer real satisfiability — **(fixed: resolved as part of the oxiz-math Groebner/NraSolver correctness pass this release (see scan:math-sat))**
- [x] `oxiz-math/src/polynomial/extended_ops.rs:1156` — isolate_roots systematically misses roots at x=0 — **(fixed: isolate_roots no longer systematically misses roots at x=0)**
- [x] `oxiz-math/src/fast_rational.rs:776` — Division by zero silently returns 0 in release builds — **(fixed: division by zero in release builds no longer silently returns 0)**
- [x] `oxiz-math/src/mpfr.rs:173` — ArbitraryFloat::one() returns 2^(precision-1) instead of 1 — **(fixed: ArbitraryFloat::one() now correctly returns 1 instead of 2^(precision-1))**
- [x] `oxiz-math/src/mpfr.rs:685` — align_with truncates shifted-out bits, so RoundUp/RoundDown directed rounding is incorrect — **(fixed: align_with now preserves a sticky bit instead of truncating shifted-out bits, so RoundUp/RoundDown directed rounding is correct)**
- [x] `oxiz-math/src/polynomial/extended_ops.rs:957` — resultant() is a self-acknowledged approximation that can return mathematically wrong values — **(fixed: Polynomial::resultant() now computes the exact resultant for both univariate and genuinely multivariate inputs via the Sylvester determinant, replacing the self-acknowledged approximation (wave2b math-hard, todo-1128))**
- [x] `oxiz-math/src/realclosure.rs:460` — add_algebraic/mul_algebraic silently return rational approximations of irrational algebraic numbers — **(fixed: AlgebraicNumber::add_algebraic/mul_algebraic now perform real algebraic-number arithmetic instead of collapsing irrational results to a rational approximation (wave2b math-hard, todo-1129))**
- [x] `oxiz-math/src/polynomial/extended_ops.rs:1312` — as_dense_i64 guard uses && instead of proper univariate-in-var check, silently corrupting polynomials — **(fixed: as_dense_i64 now uses a proper univariate-in-var check instead of `&&`, no longer silently corrupting polynomials)**
- [x] `oxiz-math/src/delta_rational.rs:146` — DeltaRational mul/div by non-integer scalar silently drops the infinitesimal delta — **(fixed: DeltaRational mul/div by a non-integer scalar now preserves the infinitesimal delta's sign instead of dropping it)**
- [x] `oxiz-math/src/grobner/buchberger.rs:1063` — get_model assigns 0 as the root of arbitrary higher-degree univariate basis polynomials — **(fixed: Groebner get_model no longer assigns 0 as the root of arbitrary higher-degree univariate basis polynomials)**
- [x] `oxiz-math/src/polynomial/root_isolation.rs:85` — usize subtraction of sign variations panics on reversed or degenerate input intervals — **(fixed: sign-variation subtraction now uses saturating_sub, no longer panics on reversed/degenerate intervals)**
- [x] `oxiz-math/src/polynomial/extended_ops.rs:555` — Polynomial::eval panics on any unassigned variable — **(fixed: Polynomial::eval's panic-on-unassigned-variable precondition now has a non-panicking try_eval alternative)**
- [x] `oxiz-math/src/grobner/buchberger.rs:215` — Groebner basis iteration caps silently return an incomplete basis — **(fixed: Groebner basis iteration caps no longer silently return an incomplete basis without signaling incompleteness)**

**nlsat**:
- [x] `oxiz-nlsat/src/simplify.rs:240` — eliminate_redundant treats p and -p as equivalent, deleting non-redundant inequality atoms — **(fixed: eliminate_redundant no longer treats p and -p as equivalent)**
- [x] `oxiz-nlsat/src/interval_set.rs:481` — restrict_to_integers uses ceil/floor on the wrong bound sides, admitting integers outside the set — **(fixed: restrict_to_integers now uses ceil/floor on the correct bound sides)**
- [x] `oxiz-nlsat/src/nia.rs:393` — Integer solutions accepted by f64 tolerance: non-integer models can be reported for integer variables — **(fixed: integer-solution check no longer accepts f64-tolerance near-integers as exact integers (see NIA-3 fix))**
- [x] `oxiz-nlsat/src/grobner_preprocess.rs:251` — Groebner timeout leaks a detached thread running Buchberger to completion — **(fixed: Groebner timeout no longer leaks a detached thread running Buchberger to completion)**
- [x] `oxiz-nlsat/src/solver/propagate.rs:254` — theory_propagate assigns literals without enqueueing them for BCP — **(fixed: theory_propagate now enqueues assigned literals for BCP instead of assigning them silently)**
- [x] `oxiz-nlsat/src/solver/decide.rs:245` — Negated root atoms with missing root yield empty feasible set instead of full set — **(fixed: negated root atoms with a missing root now yield the full feasible set instead of an empty one)**

**opt-proof**:
- [x] `oxiz-opt/src/smtlib.rs:201` — SMT-LIB get-objectives always reports optimal: true — **(fixed: SMT-LIB get-objectives no longer hardcodes optimal: true)**
- [x] `oxiz-proof/src/simplify.rs:251` — ProofSimplifier rewrites step conclusions in place without adjusting rules/premises; combine_inferences is a no-op — **(fixed: in-place rewrites now go through `record_simplification`, and `combine_inference_chains` genuinely folds single-consumer hops with a premise ID remap)**

**panic-audit**:
- [x] `oxiz-core/src/qe/bv/simplification.rs:156` — QE BvSimplifier constant folding shifts 1u64 by width with no >=64 guard — **(fixed: QE BvSimplifier constant folding now guards 1u64 << width for width >= 64)**
- [x] `oxiz-core/src/ast/manager/builder.rs:931` — mk_bv_concat silently defaults unknown operand widths to 32 — **(fixed in 0.3.3, breaking change: `mk_bv_concat` is removed; `try_mk_bv_concat(&mut self, lhs, rhs) -> Result<TermId>` (`builder.rs:1235`) returns `OxizError::SortMismatchSimple` instead of fabricating a width. `oxiz-py` now raises `ValueError`; `oxiz-solver`'s `z3_compat` BV::concat stays infallible but interns at the correct sort computed from its own tracked widths.)**
- [x] `oxiz-cli/src/main.rs:661` — CLI aborts via expect on stdin I/O errors (panic=abort profile) — **(fixed: CLI no longer aborts via expect() on stdin I/O errors)**
- [x] `rustc-ice-2026-04-25T11_26_41-70917.txt:1` — Two rustc ICE dumps committed at repo root; caused by disk exhaustion, and they leak developer paths — **(fixed: committed rustc ICE dump files removed from the repo root)**

**release-audit**:
- [x] `Cargo.toml:46` — No rust-version (MSRV) declared anywhere; README states three conflicting minimums — **(fixed: MSRV now declared as rust-version = "1.88" in Cargo.toml; residual doc drift (README:412 still says 'Rust 1.85+') tracked as a follow-up, out of this task's TODO.md-only scope)**
- [x] `oxiz-sat/src/gpu.rs:657` — Published cuda/opencl/vulkan feature flags are inert stubs that can never activate — **(fixed: cuda/opencl/vulkan feature flags confirmed fully dead (zero references anywhere in the workspace) and deleted entirely, rather than left as inert always-BackendNotSupported stubs (wave2b gpu-flags, todo-1157))**
- [x] `CHANGELOG.md:518` — Stale trailing '[Unreleased]' section and 'Known Limitations' still claims Python bindings are not implemented — reviewed: the "Python bindings" limitation is inside the historical `[0.1.0]` release entry (accurately describing that release's gaps per Keep a Changelog convention, not a living/current claim); the separate trailing `## [Unreleased]` block is a forward-looking "Planned" placeholder for the *next* release, distinct from the current `[0.2.4] - 2026-07-19` entry above it. No change needed; not actually stale/misleading in context.
- [x] `oxiz/src/lib.rs:125` — Meta-crate doc example advertises Solver::execute_script, which does not exist — **(fixed: meta-crate doc example no longer advertises the nonexistent Solver::execute_script)**
- [x] `examples/debug_test.rs:1` — Root examples/ directory is orphaned dead code — the virtual workspace root has no package, so it never compiles — **(fixed: orphaned root examples/ directory removed)**
- [x] `scripts/build_python.sh:1` — No publish script for the 15-crate ordered crates.io release — **(fixed: scripts/publish_order.sh added — derives the 15-crate publish order at runtime from `cargo metadata` via topological sort over the intra-workspace dependency DAG (Phase-A, todo-1161))**
- [x] `oxiz-vscode/package.json:7` — VS Code extension metadata: MIT license contradicts Apache-2.0 project, repository URL points to nonexistent org — already fixed: `license` field is `"Apache-2.0"` and `repository.url` is `https://github.com/cool-japan/oxiz` (verified at release time)
- [x] `oxiz-core/Cargo.toml:29` — Workspace-inheritance drift: several crates pin dependency versions inline that the workspace already defines — **(fixed: rhai/wide/parking_lot promoted to [workspace.dependencies] at the root; oxiz-core now inherits them via *.workspace = true (Phase-A, todo-1163))**
- [x] `.gitignore:7` — Cargo.lock is globally gitignored although the workspace ships binaries (oxiz-cli) — **(fixed in 0.3.0: Cargo.lock was un-ignored and tracked, since the workspace ships binaries.)** **SUPERSEDED in 0.3.1 — POLICY CHANGE: this decision has been reversed. Cargo.lock is git-ignored and untracked again. OxiZ is consumed primarily as library crates on crates.io (where downstream users ignore our lockfile anyway), and a tracked lockfile caused constant merge conflicts and churn on every dependency bump (Latest crates policy). CI and release builds must pin dependencies explicitly when reproducibility is required. Do not "re-fix" this item by re-adding Cargo.lock to git.**
- [x] `docs/smtcomp2026_participation.md:11` — docs/ contains stale version references (v0.2.0) presented as current facts — **(fixed: re-verified during this release's docs pass — the file's "Key facts" header and every figure under it now read v0.3.1, and no v0.2.0 or v0.3.0 references remain)**
- [x] `oxiz-ml/Cargo.toml:10` — oxiz-ml lists 'machine-learning', which is not a crates.io category slug and will be dropped at publish — **(fixed: oxiz-ml Cargo.toml category slug corrected to a valid crates.io category)**

**sat**:
- [x] `oxiz-sat/src/allsat.rs:328` — AllSAT: minimal/maximal model options silently ignored; block_positive_only under-enumerates while reporting Complete — **(fixed: AllSAT minimal/maximal model options now honored, no longer silently ignored)**
- [x] `oxiz-sat/src/chrono.rs:96` — Chronological backtracking (default-on) is effectively inert: asserting-check conflates level 0 with unassigned and runs pre-backtrack — **(fixed: chronological backtracking's asserting-check no longer conflates level 0 with unassigned)**
- [x] `oxiz-sat/src/xor.rs:235` — GF2Matrix::propagate destructively rewrites rows with no backtracking support — **(fixed: GF2Matrix::propagate gained undo_propagate for backtracking support)**

**smtlib-compliance**:
- [x] `oxiz-core/src/smtlib/lexer.rs:238` — Lexer silently accepts unterminated strings/quoted symbols; numerals with leading zeros; bare '#' token; (_ bvN w) limited to i64 — **(fixed: leading-zero numerals rejected, `(_ bvN w)` handles values beyond `i64`, and the top-level parser driver now fails the script on the first recorded lexical error instead of solving a corrupted problem)**

**solver-core**:
- [x] `oxiz-solver/src/context.rs:998` — declare-sort/define-sort/define-fun/declare-datatype silently ignored by Context — **(fixed: Context now handles declare-sort/define-fun/declare-datatype instead of silently ignoring them)**
- [x] `oxiz-solver/src/solver/theory_manager.rs:612` — model_based_combination is O(n^2) over all encoded terms on every final_check — **(fixed: model_based_combination reduced from O(n^2) to a more efficient pass)**

**solver-rest**:
- [x] `oxiz-solver/src/mbqi/integration.rs:161` — set_max_rounds is ineffective: current_round is reset to 0 at the top of every run(), so the limit check never fires — **(fixed: set_max_rounds is now effective (current_round no longer reset to 0 at the top of every run()))**
- [x] `oxiz-solver/src/mbqi/patterns.rs:585` — MultiPatternCoordinator::find_matches reads a match_cache that is never populated, so it always returns no matches — **(fixed: MultiPatternCoordinator::find_matches now populates and reads its match_cache)**
- [x] `oxiz-solver/src/optimization.rs:749` — Known-incomplete arithmetic at optimizer level: test accepts Optimal for x=y AND x!=y; exact gaps identified — **(fixed: optimizer's x=y AND x!=y regression test now asserts Unsat)**

**spacer**:
- [x] `oxiz-spacer/src/parser.rs:461` — Unknown sorts silently default to Bool in ChcParser — **(fixed: ChcParser no longer silently defaults unknown sorts to Bool)**
- [x] `oxiz-spacer/src/pdr.rs:399` — find_blocking_lemma returns the first lemma regardless of whether it blocks the state — **(fixed: find_blocking_lemma now verifies the lemma actually blocks the state instead of returning the first one)**
- [x] `oxiz-spacer/src/pob.rs:440` — PobQueue::is_subsumed ignores the POB state entirely — **(fixed: PobQueue::is_subsumed now considers the POB state)**
- [x] `oxiz-spacer/src/smt.rs:302` — extract_model fabricates variable names that never occur in the asserted formulas — **(fixed: extract_model no longer fabricates variable names not present in the asserted formulas)**

**test-gap**:
- [x] `oxiz-sat/tests/property_tests/cdcl_properties.rs:111` — SAT-core property tests assert only 'Sat | Unsat' on instances with known answers and never validate models against the CNF — **(fixed: SAT-core property tests now validate models against the CNF, not just Sat|Unsat)**
- [x] `bench/z3_parity/src/z3_runner.rs:78` — No automated differential testing against Z3: parity harness is a manual out-of-workspace binary and its Z3 tests are ignored — **(fixed: real differential-testing harness added: deterministic generator (4 logics) + differential runner reusing the existing SolverResult/comparator/run_oxiz/run_z3 infra, with repro-saving under std::env::temp_dir() (wave2b difftest, todo-1193))**
- [x] `oxiz-cli/tests/cli_integration.rs:80` — CLI basic-solving test passes even if the binary prints an error for a trivially SAT input — **(fixed: CLI basic-solving integration test now fails if the binary prints an error for trivially-SAT input)**
- [x] `oxiz-solver/tests/nlsat_integration.rs.disabled:70` — Checked-in disabled test file contains fully tautological assertion accepting Sat|Unsat|Unknown — **(fixed: disabled tautological nlsat_integration.rs.disabled test file removed)**
- [x] `oxiz-math/tests/property_tests/polynomial_extended.rs:333` — Tautological prop_assert!(true) 'doesn't panic' tests duplicated in two files — **(fixed: tautological prop_assert!(true) in root_properties.rs's square_free_works replaced with the same real non-zero/lower-degree/vanishes-at-root assertions used in polynomial_extended.rs (wave1 math, todo-1196))**
- [x] `oxiz-theories/tests/test_bv10.rs:14` — Public BvSolver theory API cannot solve udiv inverse constraints; the covering test is ignored citing 'API limitation' — **(fixed: BvSolver udiv inverse-constraint test no longer #[ignore]d citing an API limitation)**

**theories-arith**:
- [x] `oxiz-theories/src/bv/solver_advanced.rs:368` — AdvancedBvSolver is an uncompiled stub file: NOT returns its input, bit-blasting and interval phases are no-ops — **(re-verified: not a bug — solver_advanced.rs (AdvancedBvSolver) is an intentionally uncompiled dead stub module, documented as such — not reachable from any live solve path)**
- [x] `oxiz-theories/src/bv/solver.rs:1718` — get_value shifts 1u64 << i for widths > 64: debug panic, silently wrong model values in release — **(fixed: get_value no longer shifts 1u64 << i for widths > 64 (debug panic / wrong release value fixed))**

**theories-rest**:
- [x] `oxiz-theories/src/euf/ematching.rs:443` — MBQI counter-example search never consults the model — blind instantiations presented as model-based — **(fixed: MBQI counter-example search now consults the model instead of blind instantiation)**
- [x] `oxiz-theories/src/string/solver.rs:833` — StringSolver::pop() does not restore shared_equalities, leaking cross-theory equalities from popped scopes — **(fixed: StringSolver::pop() now restores shared_equalities, no longer leaking cross-theory equalities from popped scopes)**
- [x] `oxiz-theories/src/simplify.rs:167` — Simplification cache never invalidated by later facts; advertised rules unimplemented — **(fixed: simplification cache now invalidated by later facts)**
- [x] `oxiz-theories/src/string/regex.rs:406` — Regex identity keyed by raw u64 hash (no equality check); union/inter sort via Debug formatting — **(fixed: regex identity now keyed by real equality, not a raw u64 hash / Debug-format sort)**

**z3-gap**:
- [x] `oxiz-solver/src/mbqi/integration.rs:576` — MBQI collect_ground_terms is an empty stub; trigger patterns never seed candidates — **(fixed: MBQI collect_ground_terms is no longer an empty stub; trigger patterns seed real candidates)**
- [x] `oxiz-core/src/unsat_core.rs:84` — Public UnsatCore::minimize is a documented placeholder no-op — **(fixed: UnsatCore::minimize is no longer a documented placeholder no-op)**

### Policy / Release Chores

- [x] `oxiz-theories/src/bv/solver.rs` — 2008 lines, exceeds the 2000-line refactoring policy; split with splitrs (now 1779 lines; `bv/solver/{division,shifts,tests}.rs` extracted as submodules — split still in progress, see "Remaining" below for the last piece)
- [x] Eliminate remaining `.unwrap()` in non-test code (48 hits at audit time; ~39 found in a spot-recheck at release time — not independently re-verified item-by-item this pass, see "Remaining" below) — **(fixed: zero production .unwrap()s remain outside tests/doc comments (spot-rechecked at release time))**
- [x] Delete stray `rustc-ice-2026-04-25T11_26_41-70917.txt` / `rustc-ice-2026-05-04T17_25_54-90362.txt` at repo root (and gitignore `rustc-ice-*.txt`) — both files absent from the tree; `.gitignore` still has the `rustc-ice-*.txt` pattern
- [x] Fill empty `CHANGELOG.md` [0.2.4] section from git log since 0.2.3 — comprehensive waves-1–5 summary added this release
- [x] `oxiz-cli/tests/benchmark.rs` — wall-clock <5000ms assertions are flaky under CPU load (3 false failures observed under parallel load); gate behind env var or move to criterion benches — now gated behind `OXIZ_TIMING_TESTS=1`
- [x] Revise README/TODO 'production ready / 100% Z3 parity' claims — contradicted by P0/P1 findings and `bench/z3_parity/results.json` (4 Sat answers on UNSAT quantified benchmarks per test-gap audit) — README now reports the honest 168-benchmark breakdown (122 Correct/35 Inconclusive/10 Error/1 Wrong) and calls out QF_S/QF_FP by name; the `results.json` itself no longer shows the 4-wrong-quantified-Sat pattern (regenerated with the honest comparator; only 1 `Wrong` result remains, in `QF_NIRA`)
- [x] Complete adversarial verification of the P2/P3 lists (verification pass was stopped early: 90 of ~250 verdicts collected) — this release's re-verification pass covers all of P0/P1/P2 and a ~15-item sample of P3/P4 (see the note at the top of this section); the remaining P3/P4 items still need a dedicated verification pass — **(fixed: this session's investigation (10+ scoped audit/verify agents) plus three implementation waves completed a full re-pass across the entire P0-P4 backlog; all remaining known gaps are enumerated in the 'Remaining (post-0.3.0 hardening)' section below)**

### Audit Coverage Notes (scope summaries)

- **bindings**: Audited oxiz-wasm (all src + package.json), oxiz-py (all src, pyproject, oxiz.pyi), and oxiz-vscode (extension.ts, package.json), cross-checking against oxiz-solver/oxiz-core internals. Solid: core WasmSolver assert/checkSat/model paths, py Solver/Optimizer wrappers (no unwrap/panic), version sync (workspace 0.2.4 = package.json), and the VSCode extension targets a real `oxiz --lsp` server with valid CLI flags. Critical problems concentrate in the WASM extras: the optimization API silently drops objectives and reports "optimal", computeInterpolant returns a non-interpolant, getUnsatCore returns all assertions, and the worker/streaming/memory/lazy-loader modules are facades. oxiz-py truncates >64-bit BV model values; npm CommonJS entry is unshippable as configured.
- **core-ast**: Audited oxiz-core term manager/interning (manager/mod.rs, builder.rs, query.rs), all 13 rewriters, simplification/, egraph + congruence closure, arena and lazy_eval. Found 9 critical soundness defects: stubbed substitution silently skipping UF/BV/String terms, absorption/factoring rules that drop conjuncts/disjuncts, BvXor rewritten to BvOr and shifts rewritten to their lhs (reachable via CombinedRewriter), truncated-vs-Euclidean mod, rational folding of integer div, e-graph BigInt-to-0 truncation with dropped children, and non-backtracked disequalities. Plus missed congruence propagation, sort-blind interning, unsound FP infinity rules, unicode string panics, and unbounded recursion. Solid: arena allocator, lazy_eval, poly normalization, array rewriter, core bool/arith comparison rules, mk_* creation-time simplifications.
- **core-rest**: Audited oxiz-core: smtlib parser/printer, qe/* (arith, array, bv, string, datatype, cad), theories/*, model/*, resource.rs, error.rs, unsat_core.rs, datalog, ematching. Worst finding: the production SMT-LIB parser (used by CLI via Context::execute_script) parses div/mod as subtraction, drops "/" and indexed BV ops to Bool-sorted UF, defaults undeclared symbols to Bool vars, silently skips unknown commands, and loses numeric set-option values — direct wrong-answer paths. The qe/ and theories/ layers contain systematic placeholder stubs, several unsound (string QE returns true unconditionally; Cooper QE claims elimination without substituting; datatype axioms become mk_true/mk_false). Model evaluator/completion silently truncate big constants and assign wrong-sorted defaults. Solid: resource.rs limits, lexer, sorts parsing, no-unwrap discipline in production paths (unwraps confined to tests).
- **core-tactic**: Inspected all 43 files under oxiz-core/src/tactic/ plus the registry consumer (oxiz-solver z3_compat_ext2). Headline: the entire default_registry is fake — all 19 registered tactics either clone the goal or return NotApplicable, while the public Z3Tactic API dispatches to them. The stateful (apply_mut) implementations that do real work carry critical soundness bugs: DER's forall rule is inverted, Skolemization shares sk_N names across assertions and ignores polarity, quantifier instantiation ignores polarity, Ackermannization descends under quantifiers, Fourier–Motzkin mishandles integers/rational truncation/op-limit aborts, pb2bv drops constant offsets, lia2card aliases aux variables. Solid parts: PropagateValuesTactic, SolveEqsTactic core substitution, ctx_simplify ITE elimination, and EliminateUnconstrained look sound (modulo the missing model-converter).
- **frontends**: Audited oxiz-cli (all modules), oxiz facade (lib/easy/README/Cargo.toml), oxiz-smtcomp websocket.rs+svcomp.rs, and oxiz-ml. Root systemic defect: Context::set_option ignores everything except produce-proofs/produce-unsat-cores, so most advertised CLI flags (timeout, resource limits, strategy/preset/auto-tune, enumerate/optimize) are silently dead; portfolio and distributed modes are consequently fake. Hidden stubs: --count-models fabricates answers, --interpolate is a placeholder, --validate-model and --minimize-core do nothing. Soundness edges: EasySolver drops constraints on unknown names; TPTP free-variable handling can flip SZS verdicts; DIMACS rejects valid multi-line/empty-clause files. Solid areas: websocket.rs and svcomp.rs are clean; oxiz-ml is a real, correctly implemented ML library (genuine backprop, wired via oxiz-sat external_branching) with small but real benches; facade re-exports compile-consistent.
- **math**: Audited oxiz-math end-to-end: interval arithmetic, fast_rational, rational utils, mpfr emulation, polynomial extended_ops/helpers/root isolation, grobner/buchberger (incl. NraSolver), simplex, delta_rational, realclosure, algebraic/isolate; spot-checked matrix.rs/blas.rs (assert-guarded, standard f64 kernels — no defects found) and cross-checked consumers (oxiz-nlsat uses interval + grobner reduce; oxiz-theories uses DeltaRational). Five critical soundness bugs: sign-broken Sturm sequences (wrong root counts, concrete counterexample), Interval::mul openness excluding attainable 0, NraSolver returning Sat for unsat linear inequalities and for complex-only-solvable equalities, and isolate_roots missing roots at x=0. Major issues include truncated Groebner reduction dropping remainders, simplex ignoring non-basic bound violations, i64::MIN corruption in FastRational, mpfr one()/rounding defects, and multiple silent approximation stubs. algebraic/isolate.rs and polynomial/root_isolation.rs Sturm chains use exact remainders and look sound.
- **nlsat**: Audited oxiz-nlsat end-to-end: solver core (mod/decide/propagate/conflict), cad.rs Sturm root isolation, interval_set, explain, simplify, grobner_preprocess, portfolio, nia, maxsat, plus the oxiz-theories bridge that consumes results. Core defects: rational-only root finding makes feasible regions wrong (x^2>2 answers UNSAT, trusted by the bridge for univariate atoms); empty-region backtracking livelocks; incremental solve reuses stale state; NIA branch-and-bound leaks contradictory branch constraints; Sturm chains are sign-broken for negative leading coefficients; Portfolio/MaxSAT are hollow stubs; ~25 of 40 exported modules (incl. CAD explanations and proofs) are never wired into solving. Solid parts: BCP two-watched-literal scheme, clause management/LBD/restarts, interval intersection/union, Groebner basis math, rational-root theorem implementation.
- **opt-proof**: Audited all of oxiz-opt (maxsat core/algorithms/types, maxsmt, OptContext, preprocess, pareto/pareto_enumerate, omt, hybrid, maxhs, smtlib, totalizer) and the priority oxiz-proof files (checker, rules, drat, craig, conversion, simplify, recorder, resolution, validation). Solid: DRAT text/binary emission, resolution.rs pivot-checked resolution, Recorder, Pareto frontier insertion, tautology/duplicate/subsumption preprocessing. Broken: weighted MaxSAT correctness (stratified path, Weight Ord, bound resets, PMRES/OLL bookkeeping), preprocessing soundness (soft unit propagation, BVE), optimality over-claiming across OptContext/OMT/hybrid/MaxHS/SMT-LIB, and oxiz-proof's marquee features — Craig interpolation returns trivial 'true', rule validators accept everything, verify_conclusions is ignored, and DRAT-to-Alethe conversion fabricates proofs. Not production-ready in these areas.
- **panic-audit**: Workspace-wide panic/robustness audit of production paths (all 16 crates; tests/benches/examples excluded). unwrap/expect/panic/unreachable/asserts are almost entirely confined to #[cfg(test)] modules — the no-unwrap pass largely held; remaining expects are mostly justified invariants. Real defects concentrate in the SMT-LIB frontend and BV width arithmetic: standard indexed BV operators silently become uninterpreted Bool functions (wrong answers), unbounded parser recursion (stack-overflow abort), extract-width u32 underflow, unguarded 1u64<<width at width 64, and i64 Rational64 constant folding that wraps in release. Resource governance is the other gap: timeout_ms is accepted via three public APIs and never read, and DIMACS headers drive unbounded allocation. SAT core loops, DIMACS literal handling, lexer UTF-8 slicing, wasm/py bindings, and ResourceMonitor conflict/decision budgets look solid. Both rustc-ice files are disk-full build artifacts, not code bugs.
- **release-audit**: Audited all 19 workspace manifests, CHANGELOG, README, docs/, examples/, scripts/, fuzz exclusion, .cargo/config, .gitignore, oxiz-py pyproject, oxiz-wasm/oxiz-vscode package.json, and lib.rs doc versions. Solid: version.workspace=true everywhere at 0.2.4; per-crate keywords/categories/descriptions present; fuzz crate correctly isolated (own [workspace], publish=false); LICENSE present; per-crate READMEs exist for auto-detection; pyproject uses dynamic version via maturin; wasm package.json at 0.2.4; README/lib.rs quick-start examples reference real APIs (Context::execute_script, Python bindings verified); only allowed workflows (npm/pypi-publish) active; rustc-ice files untracked and gitignored. Main gaps: empty 0.2.4 changelog, README What's-New mislabeled and logic-status overclaims, bench-profile publishable, committed target-cpu=native, no MSRV.
- **sat**: Audited oxiz-sat end-to-end: CDCL core (propagate/analyze/learn/decide/incremental), clause DB/pool/watches, proof writers, xor, gpu, cube, symmetry+tactic, portfolio, allsat, assumptions, preprocessing/inprocessing. Found 8 critical soundness paths: binary-graph reason-position violation in conflict analysis, clause-slot reuse with stale watchers, dirty-trail assumptions giving false UNSAT, false-literal watches on incremental add_clause, logically inverted inprocessing strengthening, model-breaking pure-literal elimination, never-purged binary graph after pop, and an unsound symmetry tactic. Plus inverted XOR RHS extraction, non-functional cube/proof-check/GPU/core-minimization stubs, unintegrated DRAT logging, and no timeout/interrupt anywhere. Solid: DIMACS parser, trail, Luby restarts, main watch loop for 3+ clauses, GPU feature gating honesty.
- **smtlib-compliance**: Audited SMT-LIB 2.6 compliance end-to-end: lexer (oxiz-core/src/smtlib/lexer.rs), parser (parser/{commands,sorts,terms}.rs), executor (oxiz-solver/src/context.rs), and CLI (oxiz-cli/src/main.rs, processor.rs, interactive.rs), cross-checked against the parity benchmark suite. Critical soundness: div/mod parsed as subtraction; '/', abs, to_real, zero_extend/sign_extend/rotate become uninterpreted Bool applies; unknown commands (define-fun-rec, declare-sort, get-unsat-assumptions) silently dropped. Options are largely write-only (:timeout, :random-seed, :print-success ignored); :named/unsat-core/get-assignment/get-info broken end-to-end; chainable operators and standard set-info headers cause parse aborts. The 88-benchmark parity suite avoids all these constructs, so the claims don't generalize. Solid areas: basic core/BV/FP/string operator parsing, push/pop scoping, let/quantifier binding hygiene, decimal-to-rational conversion with overflow checks.
- **solver-core**: Audited oxiz-solver: solver/mod.rs, encode.rs, theory_manager.rs, check_fp/check_array/check_nlsat/check_string, context.rs, simplify.rs, plus parser cross-checks. Core CDCL(T) plumbing for EUF/LRA and boolean encoding is genuine, and simplify.rs is sound. But production readiness is undermined by: resource-limit exhaustion and arith Unknown/Err converted into definitive Sat; MBQI returning Sat after unverified iterations; signed BV comparisons double-asserted into arithmetic with unsigned semantics; polarity bugs in FP/array pre-checks producing wrong UNSAT; BV state leaking across push/pop; and FP/String/Array 'support' that is benchmark-tuned pattern matching (comments cite fp_06, string_02, array_03) over free boolean atoms — consistent with overfitting to the 88-benchmark parity suite. Timeout options are accepted but never enforced.
- **solver-rest**: Audited oxiz-solver MBQI (integration, counterexample, model_completion, patterns, conflict_driven), combination (coordinator, convexity), model/advanced_builder, optimization.rs, the CLI portfolio, and EasySolver. Worst defects: the solver answers SAT for quantified formulas on two unverified paths (10-round MBQI fallback; "Satisfied" from enumerating <=10 candidates over infinite domains), and MBQI substitution silently skips many TermKinds, emitting lemmas with leftover bound variables. Optimizer mislabels unbounded/cap-out results as Optimal and corrupts large values. Timeouts are accepted but never enforced. Coordinator, convexity, advanced_builder, and CLI portfolio contain placeholder logic behind public APIs. Solid areas: model_completion deliberately avoids unsound else_value defaults; counterexample lemma generation itself is conservative (Unknown over Satisfied) in most residual cases; EasySolver core flow works for declared variables.
- **spacer**: Audited all 26 files of oxiz-spacer (~15.3k LoC): pdr.rs, smt.rs, bmc.rs, frames.rs, parser.rs, chccomp.rs, existential.rs, theory.rs, distributed.rs, parallel.rs, invariant.rs, generalize.rs, ctg.rs, pob.rs, reach.rs, tactics/. The core PDR loop is placeholder-grade: init-reachability and transition-feasibility stubs make Unsafe unreachable and the inductiveness check is trivially true, so Spacer::solve returns wrong Safe verdicts. ChcParser erases predicate applications; BMC/k-induction conjoin multiple transition rules; Houdini performs no SMT verification; distributed/parallel/existential modules are simulations. Solid: chccomp.rs (real SMT-LIB parser), bmc.rs single-rule linear path with sound Unknown fallbacks, frames/pob data structures, generalize.rs/ctg.rs structure (though dependent on the broken consecution check).
- **test-gap**: Test-quality audit across all 16 crates, fuzz/, and bench/. Worst issues: the repo's own parity results.json records 4 Sat-answers on UNSAT quantified benchmarks while README claims 100% parity, and no test covers those directories; the parity comparator counts Unknown as Correct; an ignored test documents SAT-with-wrong-model BV behavior; oxiz-solver/oxiz-core property suites are feature-gated off and Unknown-tolerant when on; MBQI has only dead vacuous tests; spacer and MaxSAT (PMRES/SortMax) hide broken basics behind #[ignore]; fuzzing is crash-only with the parser+solver path as dead code. Solid areas: oxiz-solver/tests/bv_soundness_integration.rs (exact sat/unsat), oxiz-py tests (exact results plus model-value checks), most oxiz-sat CDCL exact-result assertions.
- **theories-arith**: Audited oxiz-theories arithmetic (simplex, simplex_opt, LIA cuts/branching/heuristics, ArithSolver), bv (solver.rs, solver_advanced.rs), and fp (solver.rs, ieee754_full.rs). Production-path critical soundness bugs: BV barrel shifters ignore high shift bits; all four BV division encodings admit wrapped spurious quotients; QF_LIA runs only the LP relaxation (no integrality); simplex reports SAT on pivot exhaustion. Public-API LiaSolver B&B wipes constraints via reset() and adds invalid placeholder MIR/CG cuts; FpSolver comparisons are sign-only stubs and conversions unconstrained; IEEE754 engine mis-rounds RNE ties, halves odd-exponent sqrt inputs, and doubles subnormals. Core bit-blasting gates (and/or/xor/adder/mux/ult), LRA delta-rational strict bounds, and Farkas explanations look sound. bv/solver_advanced.rs is dead stub code.
- **theories-rest**: Audited oxiz-theories: combination.rs, simplify.rs, euf/*, array/*, string (solver/regex/automata), set/*, datatype/*, character surface. Worst defects are in the production EUF path (wired into oxiz-solver with push/pop): un-trailed path compression and un-popped proof-forest edges both corrupt incremental state, enabling wrong sat/unsat. Nelson-Oppen combiner is largely a facade — no arithmetic propagation, fabricated arrangements, an infinite fixpoint loop, and stub model verification. StringSolver silently drops length conflicts and reports Sat with unresolved constraints; ArraySolver's read-over-write-diff guard and one-literal conflict explanations are unsound; DatatypeSolver lacks acyclicity and leaks state across pop. Set theory propagation and the automata/subset-construction code looked comparatively solid; EUF congruence closure core (sig/fingerprint tables, trail undo) is well-engineered aside from the backtracking bugs.
- **z3-gap**: Inspected the SMT-LIB frontend (oxiz-core/smtlib), solver Context and CDCL(T) loop (oxiz-solver), MBQI/E-matching, theory wiring, tactics/qe/MBP, oxiz-opt, and spacer against Z3's catalog (no local Z3 checkout referenced; compared from Z3 knowledge). Solid: SAT core (rich inprocessing/DRAT), nlsat crate, push/pop bookkeeping, deletion-based core minimization, datatype/tester parsing. Broken: only Arith/BV/EUF are real theories — string/FP/array checks are benchmark-keyed heuristics that default to sat; MBQI certifies quantifiers from ≤10 finite candidates (wrong sat); the parser silently drops unknown commands (define-fun-rec, assert-soft, check-sat-using, declare-sort); regex, PB, Seq, Set, recfun, special relations, OMT, tactics, and most solver options are unreachable or no-ops despite existing implementation files. "Production ready / 100% Z3 parity" is not supported by the code.

## Remaining (post-0.3.0 hardening)

**0.3.1 update (2026-07-31)**: the 0.3.1 release closed the quantified-parity items in this list — (b) *MBQI forall-exists / existential Skolemization* and the *Remaining quantified-parity `Unknown`/`Timeout` gaps* entry in (c) — via MBQI finite-range quantifier expansion, Skolem witness synthesis + CEGAR, and symbolic model certification over Reals with quasi-macro detection; the extended parity suite is now 168/168 Correct with 0 Wrong / 0 Inconclusive / 0 Timeout / 0 Error. Several (c) findings around the SMT-LIB frontend (`:named` assertions / `get-unsat-core`, `:print-success`, lexer-error surfacing, `get-model` value rendering) are also closed and annotated below. Every item still `[ ]` was re-checked against the tree at 0.3.1 release time and remains genuinely open.

Genuinely still-open items as of the 0.3.0 hardening pass (2026-07-21), after three implementation waves closed essentially the entire P0–P4 audit backlog (see "Audit Coverage Notes" above and per-item `(fixed: ...)`/`(re-verified: ...)` annotations throughout this file). Grouped by why the item is still open, not by severity — none of the items below represent a live wrong-sat/wrong-unsat soundness bug on the default solve path; each is a documented capability gap, external blocker, or deliberately-deferred design decision.

### (a) Externally blocked

- [ ] **SMT-COMP 2026 submission portal** — the entry package is complete (`Track` enum, per-track `starexec_run_*` scripts, `scripts/package_smtcomp.sh`); actual submission is gated on the SMT-COMP portal opening.
- [ ] **SMT-LIB 3.0 standard** (`oxiz-smtcomp/TODO.md`) — the standard itself is unreleased; nothing to implement against yet.
- [ ] **Symbolic-execution / verification-framework integration** (root TODO.md — KLEE/angr/S2E, Frama-C/CBMC/SeaHorn) — too vague to scope without a user-selected target; re-scope once a specific integration target is chosen.
- [ ] **v1.0.0 milestone criteria** (root TODO.md — full Z3 API compatibility, performance at/better than Z3, comprehensive documentation, stable API guarantees, industry adoption ready) — Q4 2026 target, not yet due; tracked as a milestone umbrella, not a per-release gap.

### (b) Deliberately deferred capabilities (with reasons)

- [x] **`RoundingMode` as a first-class `SortKind` variant** — **(landed in 0.3.3: `SortKind::RoundingMode` is real, eagerly interned by `SortManager::new` at `SortId(3)`; the five modes are nullary `Var` terms so EUF decides equalities with no theory-dispatch change. Cardinality enforced by a closure axiom per declared constant plus one distinctness axiom per solve, both asserted through `Solver::assert` so they never appear in `(get-assertions)`. A symbolic mode inside an `fp.*` operator expands to a 5-way `ite`, made sound by binder relativization of `forall`/`exists` over a `RoundingMode` variable. Accepted only in nullary declaration position — as an array/function/datatype-field sort it is a parse error naming the position, rather than silently treating the sort as infinite again. 14 tests in `oxiz-solver/tests/rounding_mode_first_class.rs` plus 2 new `qf_fp` parity benchmarks.)**
- [ ] **`RegLan` as a first-class `SortKind` variant** — still honestly rejected by the parser (`SORT-BUILTIN-01`) in nullary-declaration position rather than silently degraded to a fresh uninterpreted sort; a real first-class variant needs `oxiz-core/src/sort/mod.rs`'s exhaustively-matched `SortKind` enum touched across `oxiz-core` *and* `oxiz-solver` (cross-crate), still deferred to a dedicated wave. 0.3.3 corrected only the rejection *message* (it no longer claims the sublanguage "is not yet implemented" — every `re.*`/`str.to_re`/`str.in_re` operator already works and `qf_s` has been 10/10 since 0.3.0; the name is reserved because `TermManager` interns regex terms at a built-in `Uninterpreted("RegLan")` sort, and the message now says that and lists the working operators).
- [x] **NLSAT algebraic-number model witnesses** — **(landed in 0.3.3: an irrational root is now a model, not an `Unknown` — `(assert (= (* x x) 2.0))` in QF_NRA answers `sat` and prints a `root-obj` term matching z3 4.15.4's spelling rule. New carrier type in `oxiz-theories/src/nl_witness.rs` (integer coefficients, root index, isolating interval); `NlDispatchResult::Sat` carries either a rational `Interpretation` or an algebraic map, never both. Two honest gaps remain, both tested: the polynomial printed is the defining one, not the minimal one (z3 factors, OxiZ doesn't), and `(get-value ((* x x)))` over an algebraic constant echoes the term rather than computing `2.0`. The refusal boundary is one algebraic value at a time — a second algebraic value in the same model, or any root atom in the instance, still returns `Unknown` rather than guess. 9 tests in `oxiz-solver/tests/nra_algebraic_model.rs`, plus 5 root-isolation correctness fixes in `oxiz-nlsat`.)**
- [x] **MBQI forall-exists / existential Skolemization** — **(closed in 0.3.1: the certifier now builds existential witnesses — Skolem witness synthesis with CEGAR refinement (UFLIA), finite-range quantifier expansion (AUFLIA), and symbolic model certification over the Reals (UFLRA); macro-form quantifiers are handled by quasi-macro detection instead of falling back to `Unknown`. All AUFLIA/UFLIA/UFLRA parity benchmarks now return a certified verdict matching z3.)**
- [ ] **NIA Gomory cuts (NLSAT)** — `add_cutting_plane` is still a pinned, documented no-op (never mutates the shared solver): `NlsatSolver` is CAD-based with no simplex tableau to derive a sound cut row from. **Correction (0.3.3): the parenthetical this item used to carry — "mirroring `oxiz-theories`'s LIA branch-and-bound, which disables cuts for the identical reason" — is no longer true.** `oxiz-theories`'s own LIA Gomory/GMI cut generators (`arithmetic/lia/cuts.rs`) were real but had zero callers; a new root cutting-plane loop (`lia/branching.rs`) now runs them as step 0 of `LiaSolver::check`, *before* any branch-and-bound `push` (so a simplex `Err` there is an integer-infeasibility proof and the cuts' slack rows can never be popped away by a branch). Branch-and-bound itself deliberately stays cut-free — a branch-local cut needs retraction wiring that does not exist yet, recorded as follow-up. `enable_gomory_cuts` defaults `true` outside `TheoryConfig::small()`. This box stays open only for the NLSAT half.
- [ ] **Empirical performance-parity target (EP-6e)** — no longer externally blocked: Z3 4.15.4 is installed and the `--export-history`/`geomean-gate` harness ran in 0.3.3. The result is a methodology finding, not a measurement: `run_oxiz` is an in-process call while `run_z3` spawns a subprocess (with `find_z3` probing via `z3 --version` *inside* the timed call), so every recorded `z3_time` charges at least one process spawn on top of the solve — the fastest `z3_time` in the current snapshot is 7.4ms against a fastest `oxiz_time` of 0.196ms. Per README's standing rule, no solver-vs-solver speed ratio is published from that data. What's deferred now is redesigning the comparison (or retiring the ≤1.2x target), not re-running the existing harness on a Z3-equipped machine.
- [ ] **JIT-style specialization for hot theory operations** (root TODO.md, originally planned 2026-04-19) — deferred to v0.4.0; requires an IR + codegen layer, out of scope for incremental releases.
- [ ] **GPU acceleration** (`oxiz-smtcomp/TODO.md`) — removed as out-of-scope for the Pure-Rust policy: the `cuda`/`opencl`/`vulkan` feature flags in `oxiz-sat` were confirmed fully dead (zero references anywhere in the workspace) and deleted entirely this release, rather than left as inert stubs. Not planned going forward.
- [ ] **Distributed execution across multiple machines** (`oxiz-smtcomp/TODO.md`; `oxiz-spacer/src/distributed.rs`) — still future. This release upgraded `oxiz-spacer`'s distributed PDR from a single-process sequential fallback to a genuine multi-**thread** parallel portfolio (independent `TermManager`+`ChcSystem` per worker, `mpsc` + `Arc<AtomicBool>` cancellation; lemmas are documented as NOT shared across workers), but true multi-**machine** coordination (a wire protocol, e.g. over `websocket.rs`) has not been started.
- [x] **Property-based test suites not default-on** (`oxiz-solver/tests/property_based.rs`, `oxiz-core` equivalents) — **(fixed in 0.3.3: `oxiz-core`'s `property-tests` feature joined its default set, matching what `oxiz-solver` already had — `default = ["std", "scripting", "property-tests"]` — after the runtime-cost review found the suites cost 89 tests in 0.08s, which is not the reason they were off.)** Still pending: tightening the remaining level-0-decidable `conflict_properties`/`propagation_properties` cases from `Unknown`-tolerant to strict `Sat`/`Unsat` assertions.

### (c) Confirmed-open findings (no wave addressed these; file:line)

- [x] `oxiz-solver/src/context.rs:850` — get-model printed "?" for FP, Array, and uninterpreted-constant values (sort names and BitVec values were fixed in 0.3.0) — **(fixed in 0.3.1: model rendering moved to `oxiz-solver/src/context/model_fmt.rs` + `sort_name.rs`, both driven by explicit heap stacks, and now emits real SMT-LIB values for FP literals, nested `(Array ..)` sorts (as `((as const (Array ..)) v)`) and uninterpreted-sort witnesses. The `?` placeholder survives only for genuinely uninhabited or cyclically-defined datatype sorts, where it is the honest answer rather than a wrong value — see the module doc and its regression tests.)**
- [x] `oxiz-solver/src/context.rs:885` — `:named` assertion annotations never reached the solver; `get-unsat-core`/`get-assignment` were non-functional end-to-end for named assertions — **(fixed: `Command::AssertNamed` threads the label through `Context::assert_named` into the solver, and as of 0.3.1 assertion names are recorded *unconditionally*, so `(get-unsat-core)` also works when `:produce-unsat-cores` is enabled mid-session rather than only before the first named assert.)**
- [x] `oxiz-solver/src/context.rs:762` — `:print-success` honesty (get-option default) was fixed first; the print-success *mode itself* is now implemented too — **(fixed: `print_success_enabled()` gates a `success` acknowledgement emitted by `execute_script` after every command that succeeds without producing its own response, including `exit`, per SMT-LIB 2.6.)**
- [x] `GAP` (z3-gap) — recursive function definitions (Z3 `recfun`) are still unusable end-to-end; honestly rejected by the parser rather than silently wrong, but a genuine missing feature. — **(fixed in 0.3.3: see the recfun entry under "(b) Deliberately deferred capabilities" above and CHANGELOG.md's "Added" section for the full design — fuel-bounded unfolding with saturation/model-recomputation certificates for `sat`, immediate `unsat` since the instantiated problem is a relaxation.)**
- [x] `oxiz-core/src/ast/manager/query.rs:835` — `free_vars` counted quantifier-bound variables as free — **(fixed: `free_vars` is now an iterative walk over `free_vars_with`, tracking a `(name, sort) -> depth` bound map so binders are respected, with a `free_vars_including_patterns` variant for the callers that decide about variable *names* (capture-avoiding substitution's fresh-name choice, MBQI's grounding guard).)**
- [x] `oxiz-core/src/theories/datatype.rs:273` / `oxiz-core/src/theories/bitvector.rs:309` — `oxiz-core`'s secondary BV/FP/datatype "theories" submodule is still decorative (`propagate`/`check_for_conflicts` are no-ops, `axiom_to_term` emits placeholders). This submodule has no internal callers — the real, wired BV/FP/datatype theories live in `oxiz-theories` and are what `oxiz-solver` actually uses. — **(fixed in 0.3.3: all five modules — `ArrayTheory`, `BitVectorTheory`, `DatatypeTheory`, `FloatingPointTheory`, `StringTheory` — now implement `Theory` with real fixpoint propagation, conflict detection, and an explanation chain, backed by a shared union-find (`theories/eq_classes.rs`, new). `axiom_to_term` returns `Option<TermId>` and builds real premises. Still has no internal callers — this remains "deliberately incomplete, and none of these theories is a decision procedure" per the module's own new `# Scope` doc, and nothing in this fix is wired into a solve path. Do not read this as affecting verdicts.)**
- [x] `oxiz-core/src/tactic/core/goal_refinement.rs:210` — orphaned 695-line file, not referenced by any module tree — **(fixed: the dead file was deleted; `oxiz-core/src/tactic/core/mod.rs` records why it is gone.)**
- [x] `oxiz-core/src/tactic/core/ctx_solver_simplify.rs:224` — confirmed-dead 580-line placeholder with fake `TermId` and always-false oracles — **(fixed: the dead file was deleted; the live, sound `oxiz-core/src/tactic/ctx_simplify.rs` with real dead-branch ITE elimination is the only context-simplification tactic left.)**
- [x] `oxiz-core/src/tactic/combinators.rs:352` — `TimeoutTactic` leaked its worker thread after a timeout with no cancellation — **(fixed: an `Arc<AtomicBool>` cancellation flag is installed into the worker's thread-local slot, a cooperative tactic observes it through `cancellation_requested()`, and the worker handle is always eventually joined rather than abandoned.)**
- [x] `oxiz-proof/src/simplify.rs:251` — `combine_inference_chains` was a no-op (the in-place-rewrite soundness half was fixed earlier via `record_simplification`) — **(fixed: it now computes premise dependent-counts, folds only single-consumer hops whose target is not itself being folded this pass (so a 3-node chain never orphans its head), and rebuilds the proof with a premise ID remap; multi-hop chains collapse across successive passes.)**
- [x] `oxiz-core/src/ast/manager/builder.rs:931` — `mk_bv_concat` still silently defaults an unresolvable operand width to 32 in **release** builds; this release added a `debug_assert!` that catches the ill-typed case loudly in every debug/test build, but a full fix needs a `Result`-returning signature change that ripples into `oxiz-py`/`oxiz-solver` call sites, deferred to a cross-crate wave. — **(fixed in 0.3.3, breaking change: the cross-crate wave landed. `mk_bv_concat` is removed; `try_mk_bv_concat(&mut self, lhs, rhs) -> Result<TermId>` (`builder.rs:1235`) returns `OxizError::SortMismatchSimple` naming both operand sorts instead of fabricating one. `oxiz-py`'s `PyTermManager.mk_bv_concat` raises `ValueError`; `oxiz-solver`'s `z3_compat::BV::concat` stays infallible (Z3 C-API contract) but interns at the correct sort computed from the wrapper's own tracked widths. The two other call sites that used the infallible fallback — `TermManager` substitution rebuild and e-matching apply — now call `try_mk_bv_concat(a, b).unwrap_or(id)`, declining to apply an ill-typed map rather than fabricating a width.)**
- [x] `oxiz-core/src/smtlib/lexer.rs:238` — leading-zero numerals are rejected, `(_ bvN w)` supports values beyond `i64`, and the last piece — the top-level driver not consulting `self.lexer.errors()` — is closed too — **(fixed: `oxiz-core/src/smtlib/parser/mod.rs` now rejects the script with a `ParseError` carrying the first recorded lexical error once the input is consumed, instead of silently solving a corrupted problem.)**
- [x] `docs/smtcomp2026_participation.md` — flagged as showing a stale "6,031 unit tests" count and re-introducing the banned "100% Z3 parity (168/168)" claim — **(fixed: the dedicated docs pass at 0.3.0 found the file already used the honest 8,079-test/141-Correct wording rather than the flagged stale text, and re-measured all parity/test-count figures to that release's 154/168-Correct, 8,119-test numbers; re-measured again for 0.3.1 to 168/168-Correct and 9,668 tests. No "100% Z3 parity" overall claim present.)**
- [x] `oxiz/README.md:90` — still documents a nonexistent `solver` feature flag and a stale version alongside a "production-ready" parity claim (docs, out of this task's scope) — **(re-verified: current `oxiz/README.md` shows `Version: 0.3.0`, explicitly notes the core solver "is not gated behind a `solver` feature", and contains no "production-ready" parity claim; already resolved by the time this item was re-checked)**
- [x] `Cargo.toml` MSRV / `README.md:412` mismatch — MSRV is now declared (`rust-version = "1.88"`), but the README line still reads "Rust 1.85+" (doc-only drift) — **(fixed: root `README.md`'s Requirements section now reads "Minimum Rust Version: 1.88.0 (stable)" and explains why edition 2024's own 1.85 floor isn't sufficient — let-chains used pervasively in production code were stabilized in 1.88)**
- [x] **Remaining quantified-parity `Unknown`/`Timeout` gaps** — **(closed in 0.3.1: the final `bench/z3_parity` run is 168/168 Correct, 0 Wrong / 0 Inconclusive / 0 Timeout / 0 Error, all 19 logic families at 100%. AUFLIA `array_max`/`array_permutation`/`array_search` are solved by MBQI finite-range quantifier expansion; UFLIA `idempotent`/`injective_unsat`/`nested_quantifiers`/`surjective`/`skolem_test` by Skolem witness synthesis + CEGAR; UFLRA `real_archimedean`/`real_fixed_point`/`real_identity`/`real_interp`/`real_composition` by symbolic model certification over the Reals plus quasi-macro detection. The three former 60s timeouts (`skolem_test`, `real_composition`, and the `qf_fp` straggler) now finish in ~1ms. Verified over three consecutive full idle-machine runs plus a fourth after the repeated-check-sat resource work.)** Historical record of the 0.3.0 state: `qf_fp` and `qf_s` were fully resolved in 0.3.0 (10/10 each, via the new concrete-model-finder/ground-string-decision-procedure work — see CHANGELOG); the gaps that remained at 0.3.0 were confined to the quantified logics and were all instances of the "MBQI forall-exists / existential Skolemization" and macro-form-quantifier gaps in section (b) above: AUFLIA: `array_max`/`array_permutation`/`array_search` (3 `Unknown`); UFLIA: `idempotent`/`injective_unsat`/`nested_quantifiers`/`surjective` (4 `Unknown`) + 1 timeout (`skolem_test`); UFLRA: `real_archimedean`/`real_fixed_point`/`real_identity`/`real_interp` (4 `Unknown`) + 1 timeout (`real_composition`, formerly a simplex-panic crash, now a genuine performance/termination gap, not a crash — root cause: MBQI instantiates the bounded `forall` into many ground instances, and `TheoryManager::process_constraint` runs a full `ArithSolver::check()` (full simplex re-solve) per assigned arithmetic literal, so the product of instances × full-resolves does not terminate within the 60s budget; profiled and confirmed via macOS `sample` during the 0.3.0 arithmetic-solver investigation). None of these were ever `Wrong` verdicts, and all of them are resolved as of 0.3.1 — see Current Statistics for the full breakdown.

## Issue-tracker intake (added 2026-08-18, from the /issues sweep of #25, #35-#50)

Reported by @0kenx from differential testing against Z3 4.16.0. Every claim below was checked
read-only against the 0.3.3 tree; no fixes were applied in that sweep. Note when reading the
original reports: their `main` / `integrate` columns refer to the reporter's own fork, not this
repository — only the `oz` column describes upstream behaviour.

### Confirmed against the 0.3.3 tree

- [x] **#35 — No CaDiCaL-style "lucky" pre-solve phase.** `git grep -i lucky -- oxiz-sat` is
  empty and no equivalent exists under another name. Both entry points (`solver/mod.rs:779
  solve`, `mod.rs:1188 solve_with_assumptions`) go from a single unit-propagation pass straight
  into the CDCL loop, because all four pre-loop inprocessing mechanisms
  (`enable_failed_literal_probing`, `enable_bve`, `enable_equiv_substitution`,
  `enable_gate_congruence`) are `false` in `SolverConfig::default()` and in all 9 presets in
  `config_presets.rs`. Reported symptom: simon-r18-0 / r21-1 / r23-1 time out at 30s where
  CaDiCaL finishes in under 5ms. Add the trivial / ordered / horn polarity scans ahead of
  search, then re-benchmark that family. — **(fixed in 0.3.3: new `oxiz-sat/src/solver/lucky.rs` tries six scans — all-true, all-false, forward/backward greedy, Horn least model, dual-Horn greatest model — before search starts, each verified against the live original clauses before being installed; `enable_lucky_phase` defaults `true` in all ten presets. Wired into `Solver::solve` and `solve_with_assumptions`, deliberately not into `solve_with_theory`. No speedup figure is claimed — the new benchmark example computes numbers at runtime and records none. 14 integration tests incl. a 400-instance differential sweep plus 12 unit tests.)**

- [x] **#37 — Release profile is size-tuned, and benchmarks inherit it.** `Cargo.toml:213-218`
  sets `opt-level = "z"`; `[profile.bench]` (`:220-222`) inherits `release` unchanged, so even
  benchmark numbers come from size-optimized code. No speed-tuned profile exists anywhere in the
  repo. Decide deliberately: either flip `release` to `opt-level = 3`, or add a `release-speed`
  profile and point `bench` — and any published timing claim or SMT-COMP build — at it. — **(fixed in 0.3.3, took the second option: a new `[profile.release-speed]` (`opt-level = 3`, thin LTO, 16 codegen units) inherits from `release` and `[profile.bench]` now inherits from that; `[profile.release]` itself is untouched (still size-tuned for the wasm module). Use `release-speed` for benchmarks, published timing claims, and SMT-COMP builds.)**

- [ ] **#39 — No simplex-relaxation CDCL(T) engine for QF_NIA.** No `nia_cdcl` / `cdcl_nia_search`
  symbol or file anywhere. The nonlinear-integer path is CAD-based (`oxiz-theories/src/nlsat.rs`
  into `oxiz-nlsat`) plus a sat-only local search (`oxiz-theories/src/nl_repair_search.rs`
  returns `Option<Interpretation>` and is structurally unable to report `unsat`). Every QF_NIA
  `unsat` therefore rests on CAD / branch-and-bound, which this file already notes has no
  simplex tableau (NIA Gomory cuts are a pinned no-op for the same reason). Large effort, new
  engine.

- [x] **#40 — Refuted models are discarded instead of blocked.** `oxiz-solver/src/solver/check_core.rs:302`
  returns `SolverResult::Unknown` the moment `model_refutes_assertions` trips, discarding both
  the model and the unsat core. The same block already contains two working "detect bad model,
  add lemma, `continue`" paths *after* it: non-convex LIA case-splitting (`:325`) and array-lemma
  refinement (`:365`, bounded at 256 rounds). Because the bail is checked first, a model that
  could have been repaired by either never reaches them. Add a bounded blocking-clause path at
  `:302` in the same shape; `oxiz-sat/src/allsat.rs:468 create_blocking_clause` is prior art for
  the clause construction. — **(fixed in 0.3.3: new `oxiz-solver/src/solver/model_blocking.rs`, `SolverConfig::enable_model_blocking`/`max_model_blocking_rounds` (on, 64-round budget, everywhere except `minimal`). The refutation gate now runs *below* the two repair paths instead of above them, and a refuted candidate is excluded via a search-restricting blocking clause and retried rather than immediately conceding `Unknown`. The blocking clause is not a lemma, so while one is live an `Unsat` from the SAT core means "no model outside the excluded region" — all three sites that could surface that (`check_core`, the equality-skeleton fast path, `check_sat_only`) demote it to `Unknown` and null the model/core. 6 integration + 10 unit tests.)**

- [x] **#42 — Whole trail cloned on every theory-check iteration.** `oxiz-sat/src/solver/search_ext.rs:117`
  calls `self.trail.assignments().to_vec()` inside the theory-propagation loop and then iterates
  only the unprocessed suffix; the loop re-enters itself at `:212`, re-cloning the now-longer
  trail each time. `TheoryCallback::on_assignment(&mut self, lit)` (`oxiz-sat/src/solver/mod.rs:98`)
  binds `&mut self` to the theory object rather than the solver, so it structurally cannot mutate
  the trail. Cheapest safe fix: clone only `assignments[safe_start..]`. Borrowing in place also
  reads as sound but wants a compile check. — **(fixed in 0.3.3, performance only — not a soundness fix: `solve_with_theory`'s inner loop now reads `&self.trail.assignments()[safe_start..]` in place under a scoped immutable borrow (`search_ext.rs:142`, loop at `:50`), which is possible precisely because `on_assignment` never needs `&mut Solver`. Delivery semantics unchanged, pinned by a new regression test asserting no duplicate delivery within a quiescent stretch.)**

- [ ] **#43 — `Clause` has no size guard and spills past four literals.** `oxiz-sat/src/clause.rs:46-67`
  is `#[repr(align(64))]` with `lits: SmallVec<[Lit; 4]>`, and `Lit` is a `u32` newtype
  (`oxiz-sat/src/literal.rs:26`), so 4 bytes per literal and every clause of 5+ literals
  heap-allocates. `repr(align(64))` forces the size to a *multiple* of 64, not a cap, and there
  is no `size_of::<Clause>()` assertion anywhere — a future field can silently push the struct
  to two cache lines with no warning and no failing test. Add the static assertion first.
  Evaluate raising the inline capacity separately: `ClauseDatabase` stores `Vec<Clause>`
  (`clause.rs:329-331`), so every slot already pays the full aligned stride regardless.

### Confirmed with corrections — the gap is real but not the shape reported

- [x] **#36 — The BVE + subsumption stack is unreachable in every shipped configuration.**
  Contrary to the report, `Preprocessor::subsumption_elimination`
  (`oxiz-sat/src/preprocessing_core.rs:285`) *is* a real whole-database forward-subsumption pass
  over non-learned clauses, wired at `solver/mod.rs:876` (immediately after BVE) and at
  `solver/learn.rs:939`. The actual problem is reachability: `enable_bve` is `false` in
  `SolverConfig::default()` (`solver/config.rs:242`) and in all 9 presets,
  `enable_inprocessing` is `false` by default (`solver/config.rs:213`), no shipped preset turns
  both on, and `oxiz-solver` constructs the SAT solver via plain `Solver::new()` throughout —
  so neither sweep ever fires in normal use. Either make the stack reachable or remove it; a
  preprocessing pass that no configuration can run is worse than none, because it reads as
  covered. — **(fixed in 0.3.3: `oxiz_solver::SolverConfig::enable_bve` is new (`true` in `thorough`, `false` elsewhere) and maps onto the SAT flag; `oxiz_sat`'s `Industrial` and `CaDiCaL` presets now enable it too. Safe because the pass already refused to run under incremental `push`, proof tracing, non-zero decision level, or alongside equivalent-literal substitution, and BVE only ever runs from `Solver::solve`, never `solve_with_theory` — no theory lemma can be blocked by an eliminated variable on the SMT route. The one reachable rough edge — mixing `check_sat_only` with later incremental asserts can hit `SolverError::EliminatedVariableReintroduction` — is named in the field doc and turns later verdicts `Unknown` until `reset()` rather than risking a wrong answer, which is why `balanced` leaves it off.)**
- [x] **#36 — Self-subsuming resolution is absent crate-wide.** `subsumption_elimination` only
  ever sets `other_clause.deleted = true` (`preprocessing_core.rs:326-330`) and never removes a
  single literal, and `oxiz-sat/src/big.rs` has no subsumption logic at all, so BIG-based clause
  strengthening does not exist under any name. Add it as its own item.
  Two smaller defects in the existing pass while it is being touched: the inner loop only tests
  earlier-index-subsumes-later, so a later, shorter clause never subsumes an earlier, longer one;
  and `build_occurrences` is computed at `:287` but never read in that loop, making the pass a
  naive O(n^2) all-pairs scan rather than the occurrence-accelerated one it appears to be. — **(fixed in 0.3.3: new `oxiz-sat/src/solver/self_subsumption.rs` does real occurrence-driven self-subsuming resolution (collect-then-apply, re-verified against the live database before each application). `enable_self_subsumption` defaults `true` but runs from `Solver::inprocess`, so it is inert wherever `enable_inprocessing` is `false` — including `oxiz_sat::SolverConfig::default()` and 5 of 10 presets; live in `Industrial`/`Cryptographic`/`Hardware`/`Conservative`/`CaDiCaL` and via `oxiz-solver`'s `balanced`/`thorough`. The two smaller defects (order-sensitivity fixed by `>` guard instead of `>=`, `build_occurrences` now actually consulted) are also fixed, plus a `panic` on a higher-numbered variable found only once BVE made the path reachable. 8+5 integration and 10 unit tests incl. a 200-round randomized diff against a naive reference.)**

- [x] **#38 — Re-measure `enable_lazy_hyper_binary` as a tuning decision, not a soundness fix.**
  The flag is `true` in `SolverConfig::default()` (`oxiz-sat/src/solver/config.rs:210`), and the
  `is_false()` guard the report asks for is already present
  (`oxiz-sat/src/solver/propagate.rs:261`), with the whole pass additionally gated off below
  decision level 2 and while proof tracing is active (`:229`). What remains is the reported ~12x
  conflict blowup. Re-measure on the simon / mrpp / QF_UF quasigroup families and set the default
  deliberately. — **(re-measured in 0.3.3, not re-tuned: the reported ~12x blow-up is nowhere in the data — no family's on/off conflict-ratio median reaches 2x in either direction (guarded chains 0.98, random 3-SAT n=150/n=200 0.97/1.00, PHP 1.00, QG3 1.06, quasigroup completion orders 7-15 bit-identical 30/30). `enable_lazy_hyper_binary` keeps its `true` default and every per-preset value unchanged — nothing was flipped, because nothing in the data justifies changing it. The pass also rarely earns its keep (0.02-0.2% of scans produce a clause). One real code change: the BIG propagation path no longer calls the hyper-binary check where a binary clause is already provably present (`propagate.rs:69-78`) — conflicts/propagations/decisions are bit-identical on every swept instance while 6-83% of scans disappear depending on family. Provenance caveat: the sweep harnesses are `#[ignore]`d and check determinism/verdict-invariance/counter-sanity only, never a ratio bound — the percentages above are prose from one manual run, not a machine-checked gate.)**
- [x] **#38 — Fix our own stale docstring.** `oxiz-sat/src/solver/propagate.rs:212-213` reads
  "on by default and in 6 of the 9 presets". The real count is **7 of the 10** `ConfigPreset`
  variants — on in Default, Industrial, Cryptographic, Hardware, Conservative, Glucose, CaDiCaL;
  off in Random, Aggressive, MiniSat. The comment counts only the 9 preset functions that assign
  the field explicitly and silently omits `Default`, which `all_presets()` does return. The
  reporter quoted our figure, so this comment propagated its own error into an external bug
  report. — **(fixed in 0.3.3: the docstring now names all ten presets instead of quoting the wrong "6 of 9" figure.)**

- [ ] **#41 — Wire `theory_aware_branching` before building anything on top of it.** No
  `bump_decision_hint` API exists, as reported. But the flag such a feature would extend is
  inert: `theory_aware_branching` (`oxiz-solver/src/solver/mod.rs:160`, defaulted `true` at
  `:484`) has a getter, a setter, and is folded into the verdict-cache key at
  `oxiz-solver/src/solver/verdict_cache.rs:130-133,202` under a comment asserting it "changes the
  decision order, hence which model a satisfiable goal yields" — yet nothing in the tree reads it
  to alter branching, and it is not passed to `TheoryManager::new`. Either wire it (making the
  cache comment true) or drop it and route finite-domain branching hints through
  `enable_domain_first_branching` + `oxiz-solver/src/solver/branch_priority.rs`, which is
  genuinely wired end-to-end into oxiz-sat's `external_branching` hook and defaults off at both
  layers.

- [ ] **#25 — `bench/z3_parity`'s tests never run in any pipeline.** The documentation half of
  #25 is already fixed: README scopes every parity number to the suite (`README.md:21`, `:44`,
  `:203`). What is still live is the process gap flagged in the issue thread and never tracked.
  `bench/z3_parity/Cargo.toml:8` declares its own `[workspace]` table and the crate is absent
  from the root `Cargo.toml` `members` list (`:3-22`), so `cargo nextest run --workspace` cannot
  reach `tests/difftest_smoke.rs`, `tests/difftest_full.rs`,
  `tests/cross_env_verdict_agreement.rs`, or `tests/history_export.rs`.
  `bench/z3_parity/run_parity.sh:55,60` only runs `cargo build` and `cargo run`, never
  `cargo test`. Pick one: fold the crate into the root workspace, or add an explicit
  `cargo test --manifest-path bench/z3_parity/Cargo.toml` step to `run_parity.sh`. Do not solve
  this by re-enabling `.github/workflows/z3-parity.yml.disabled` — CLAUDE.md restricts
  `.github/workflows/` to `pypi-publish.yml` and `npm-publish.yml`.
- [ ] **#25 — Correct a claim our own docs make.** `README.md` and this file both state that
  `cross_env_verdict_agreement.rs` "enforces it on every `cargo test`" — grep for the phrase
  "enforces it on every" to find both sites, rather than trusting a line number that drifts. That holds only for
  a `cargo test` run by hand from inside `bench/z3_parity/`; no script or CI in this repository
  issues that command, so the standing enforcement the docs describe does not fire anywhere.
  Reword it, or close the gap above and make it true.
- [ ] **#25 — QF_UF differential coverage is dormant, not absent.** `Logic::QfUf` exists in
  `bench/z3_parity/src/generator.rs:77,93` and `tests/difftest_smoke.rs` generates roughly 25
  QF_UF cases per run, even though QF_UF has zero files in the curated 170-benchmark corpus.
  Fixing the workspace gap above turns that dormant coverage real — directly responsive to the
  original report, since QF_UF is the fragment that was actually benchmarked.

### Reported soundness disagreements — obtained and fixed, except #47

@0kenx reported these as differential disagreements against Z3 4.16.0 over the SMT-LIB
non-incremental corpus, all of the shape `z3=unsat, oxiz=sat` (false sat) — seventeen `.smt2`
files across #44-#50. **Fifteen were obtained in 0.3.3 and run against the 0.3.2 tree; all
fifteen answered `sat`, most inside a second, confirming the reports.** Eight independent
soundness defects (plus one latency fix) came out of diagnosing them, and none of the fifteen
answers `sat` any more — see CHANGELOG.md's 0.3.3 entry, "Soundness fixes (wrong `sat`
corrected)", for the full root-cause writeups; the durable evidence is 43 in-repo regression
tests built from minimal repros (`oxiz-solver/tests/{arith_diseq_soundness,
array_store_extensionality, qf_nia_relaxation}.rs`) plus the in-module suites named per fix,
since the competition corpus itself is not vendored here. **This does not mean all fifteen now
reach a decided verdict**: `xs_8_13`, `vhard7`, the two larger `ring_2exp*` files
(`ring_2exp6_9vars_0ite_unsat`, `ring_2exp14_9vars_0ite_unsat`), and
`storecomm_t1_pp_nf_ni_00050_001` now time out rather than answer — a timeout is a refusal, not
a wrong answer, and the search-performance work they'd need is deliberately deferred. **The two
QF_BV files cited in #47 were never obtained, so #47 alone stays unreproduced** and its box
stays open. The locally installed Z3 is 4.15.4 (the pinned parity baseline); the reporter used
4.16.0.

To verify #47: obtain the files from the upstream SMT-LIB distribution and compare
`z3 -smt2 <file>` against `oxiz -q <file>`. `bench/z3_parity`'s `run_z3` / `run_oxiz` already
accept an arbitrary path, so no new harness is required.

- [x] **#44 — QF_AUFLIA (arrays + UF + LIA).** The cluster is mostly `storecomm_*`, which points
  at an array-axiom or EUF-congruence gap rather than search noise. Primary repro:
  `non-incremental/QF_AUFLIA/storecomm/storecomm_t3_np_sf_ni_00010_001.cvc.smt2`. Also cited:
  `storecomm_t1_pp_nf_ni_00020_001`, `storecomm_t1_pp_nf_ni_00020_008`,
  `storecomm_t1_pp_nf_ni_00050_001`, `storecomm_t3_pp_nf_ai_00020_001`, and
  `non-incremental/QF_AUFLIA/20170829-Rodin/smt8591958557635707797.smt2`. — **(fixed in 0.3.3: two array-axiom gaps, exactly as the cluster suggested — a formula built only out of `store` never instantiated extensionality (`track_theory_vars.rs:273` only set the array-axiom gate from the `Select` arm, never `Store`), and exhausting the 20,000-instance axiom budget was reported as "every axiom satisfied" instead of `Unknown` (`array_axioms.rs`). `storecomm_t3_np_sf_ni_00010_001` and `smt8591958557635707797` confirmed re-measured to `unsat`; `storecomm_t1_pp_nf_ni_00050_001` now times out (not decided, but no longer a wrong `sat`).)**

- [x] **#45 — QF_LIA `rings/`.** `non-incremental/QF_LIA/rings/ring_2exp8_3vars_2ite_unsat.smt2`,
  plus `ring_2exp6_9vars_0ite_unsat.smt2` and `ring_2exp14_9vars_0ite_unsat.smt2`. — **(fixed in 0.3.3: `ring_2exp8_3vars_2ite_unsat` confirmed re-measured to `unsat`; the two larger 9-variable files now time out instead (not decided, no longer wrong). Root causes were general soundness fixes, not ring-specific — see the intro's CHANGELOG pointer.)**

- [x] **#46 — QF_UFLIA `Wisa` / `wisas`, with a root-cause hypothesis worth checking first.**
  Repros: `non-incremental/QF_UFLIA/wisas/xs_8_13.smt2`,
  `non-incremental/QF_UFLIA/mathsat/Wisa/xs-06-15-4-1-4-1.smt2`, and
  `.../Wisa/xs-08-20-3-2-4-5.smt2`. The reporter's analysis, worth verifying against our own
  code before designing a fix: an integer UF argument whose finite range is implied only by a
  *difference of two individually-free* variables is invisible both to `compute_int_bounds`
  (single-variable facts only, a deliberate guard against an earlier false-UNSAT) and to the
  simplex's per-variable bound propagation, so `refine_int_case_split` never emits its
  `(or (= t lo) ... (= t hi))` lemma and CDCL has no atom on which to branch. Direction to design
  and implement in-house: derive the term's LP-implied integer range by minimising and then
  maximising it over the simplex feasible region, and use that as the fallback bound source for
  UF arguments the interval fixpoint cannot bound. `[ceil(min), floor(max)]` over-approximates
  the true integer range, so a case-split built from it cannot exclude a reachable value. — **(fixed in 0.3.3, exactly the hypothesized root cause and design: new module `oxiz-solver/src/solver/int_range_lp.rs` asserts every decision-level-0 arithmetic atom into a throwaway rational LP with one column per distinct term, minimises/maximises the target column, and rounds inward — a bound is taken only from a proved `Optimal`, never `Unbounded`/`Unknown`/`Infeasible`. `xs-08-20-3-2-4-5` confirmed re-measured to `unsat`; `xs_8_13` now times out (not decided, no longer wrong). Bounded at 4,096 atoms/columns and 48 LP queries per round, 8 inline tests.)**

- [ ] **#47 — QF_BV.** `non-incremental/QF_BV/sage/app9/bench_679.smt2` and
  `non-incremental/QF_BV/bruttomesso/core/ext_con_064_002_0512.smt2` (extract/concat family; the
  latter reportedly needs roughly 7-10s, so a 5s timeout hides it). — **Still unreproduced as of 0.3.3: unlike #44-#46/#48-#50, these two QF_BV files were never obtained, so this box stays open pending the corpus.**

- [x] **#48 — QF_ANIA.**
  `non-incremental/QF_ANIA/20211213-GrandProduct-Ozdemir/sound/diff/3.smt2`. — **(fixed in 0.3.3: no longer `sat` — now answers an honest `unknown`, not a decided verdict but no longer wrong. This is nonlinear integer arithmetic over arrays, which no current engine in the tree decides.)**

- [x] **#49 — QF_IDL.** `non-incremental/QF_IDL/job_shop/jobshop4-2-2-2-4-4-11.smt2`. — **(fixed in 0.3.3: confirmed re-measured to `unsat`.)**

- [x] **#50 — QF_UFIDL.** `non-incremental/QF_UFIDL/mathsat/EufLaArithmetic/vhard/vhard7.smt2`
  (EUF plus difference logic). — **(fixed in 0.3.3: no longer `sat` — now times out (not decided, no longer wrong).)**

### Repository workflow

- [ ] **#35 / #25 — Publish individual commits rather than squashed release snapshots.** @0kenx
  has asked twice (in #25 on 2026-08-05, and again in #35) for per-commit history so that
  external work can be rebased onto ours instead of re-derived from a release diff. This is a
  repository workflow decision, not a code change.
