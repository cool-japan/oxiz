# OxiZ TODO

Last Updated: 2026-09-19

---

## Historical Milestone (v0.2.0): Initial 88-Benchmark Parity Suite

**Date Achieved**: February 5, 2026
**Release Status**: Published (Feb 6, 2026)

> Original v0.2.0 announcement, retained verbatim for historical record: "OxiZ has achieved **100% correctness parity with Z3** across all 88 benchmark tests spanning 8 core SMT-LIB logics. This validates OxiZ as a **production-ready Pure Rust SMT solver**."

**Superseded by the v0.2.4 honest re-audit.** The comparator that produced the table below counted an `Unknown` answer as a match (`bench/z3_parity/src/comparator.rs` — see the now-fixed `[x]` finding under "Production-Readiness Audit Findings" below), so "100%" was reachable by declining to answer rather than by matching Z3's verdict. The comparator used from 0.2.4 onward never counts `Unknown` as a match. The current, honestly-measured status lives in "Current Statistics" below, with the tracked per-environment snapshot `bench/z3_parity/results.<os>-<arch>.json` as the authoritative source — only `results.macos-aarch64.json` is currently in the tree, with `results.linux-x86_64.json` to join it once a Linux environment's run is committed. The methodology's agreement rule (every tracked snapshot must agree on the verdict of every benchmark — `oxiz_result`, `z3_result`, `match_status` — with only the timings expected to differ between machines) is enforced by `bench/z3_parity/tests/cross_env_verdict_agreement.rs` on every `cargo test`, but with a single snapshot in the tree that check is currently vacuous, not yet exercised across environments; the un-suffixed `results.json` is git-ignored local scratch output and is not evidence. As of this release, **170/170 Correct** on the extended 19-logic suite (the three quantified logics that were still below 100% at v0.3.0 — `UFLIA`/`UFLRA`/`AUFLIA` — are now all at 100%; the suite grew 168→170 in 0.3.3 with two new symbolic-`RoundingMode` QF_FP benchmarks, and no verdict moved on any of the 168 pre-existing benchmarks) and **88/88 Correct** on this original 8-logic/88-benchmark quickstart core, both under the honest comparator that never counts `Unknown` as a match. This is a claim about the differential parity suite, not a blanket claim of 100% Z3 compatibility.

### Z3 Parity Results (as originally reported, v0.2.0 — see supersession note above; table condensed 2026-09-15)

88/88 across the eight core logics as the v0.2.0 comparator counted them: QF_LIA 16, QF_LRA 16,
QF_NIA 1, QF_S 10, QF_BV 15, QF_FP 10, QF_DT 10, QF_A 10. The honest re-measured status is
"Current Statistics" below.

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
- **2026-09-14 (0.3.4 working tree, cargo-formal intake — see that section)**: **10,465 workspace tests** (nextest, release, `--workspace`; 10,465 passed, 13 skipped, 0 failures) after the U-Z10/U-Z11/U-Z12/U-Z13/U-Z14/U-Z17, `oxiz-sat` `Lit` and `#P2b-19`–`#P2b-22` push/pop fixes, up from 10,437 / 13 / 0 on the same gate on 2026-09-09 and from 9,953 at the 0.3.2 measurement above. The other figures in this section are still the 0.3.2 release-time numbers and were not re-measured.

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

## Completed: April 4, 2026 (condensed 2026-09-15)

Nine performance optimizations (arena allocator, clause pool, SIMD polynomial operations,
`TermKindHasher`, FP bit-blasting cache, model-generation cache, parallel theory checking,
lock-free structures, lazy evaluation), the `EasySolver` API with better error messages and
resource limits, four debugging facilities (state visualization, trace generation, conflict
explanations, model minimization), five documentation guides (performance tuning, theory, Z3
migration, pitfalls, case studies) and two file re-splits (`solve_eqs.rs` 1942 → 1553,
`rational.rs` 1940 → 1388 + 553 tests). Stats delta: tests 6,122 → 6,155, Rust LoC
392,274 → 393,292, clippy warnings 0, largest file 1,892 lines. The per-item list is the
"March 31, 2026 — Performance, UX, Debugging, Docs" entry under Recent Achievements below.

---

## Post-Parity Priorities (v0.3.0 and Beyond)

### High Priority: Performance Optimization (27/28 Complete - JIT specialization deferred to v0.4.0) (condensed 2026-09-15)

**Goal**: Achieve performance parity with Z3 (currently ~1.5-2x slower)

All 27 shipped items are recorded here in one block; the per-item detail was the
2026-04-19/24 planning and bench notes, and the code they describe is in the tree.
Allocation and layout: arena AST nodes, clause pooling (5 size buckets), `TermKindHasher`
interning, SIMD-friendly chunk-of-4 polynomial ops, FP bit-blasting cache, lazy model
evaluation, rayon-gated parallel theory checking, lock-free parallel structures.
Profiling: `ProfilingCategory` extended with the ten named hot paths (SAT propagation,
theory `check()`, e-graph merge, simplex pivot, BV propagation, string automata, array
extensionality, proof generation, parser, cache miss), instrumented at their call sites,
with `bench/profile/`, `scripts/flamegraph.sh --category` and `docs/PROFILING_REPORT.md`.
EUF pass (2026-04-24): production benchmarks plus a regression baseline, the fingerprint
pre-filter activated, cross-crate `#[inline]`, `get_function_props` hoisted (intern_leaf
-24%, intern_app -16%, merge_congruence -10%, merge_injective -22%), a reusable
canonicalize buffer and flat `SigUpdateEntry` (-6.7%/-6.7%/-8.2%), an incremental
`sig_table`/`fingerprint_table` undo trail replacing the O(|nodes|) rebuild on `pop`, and
the `ENode` layout reorder with the `NO_FUNC = u32::MAX` sentinel (`<= 56 B`, pinned by
`test_enode_size_regression`). Performance regression testing: CI tracking, automated
comparison against Z3, and the dashboard.

- [~] JIT-style specialization for hot theory operations
  - **Scope-box (2026-04-24):** Pure-Rust EUF data-layout + allocation-reduction +
    incremental-backtrack pass. Items 1-5 completed; parent umbrella (JIT/codegen layer)
    deferred to v0.4.0.

**Target**: Within 1.2x of Z3 performance by v0.3.0

### High Priority: Extended Theory Coverage

**Goal**: Support additional SMT-LIB logics beyond the core 8

- [x] Quantified logics (5 items)
  - [x] UFLIA - Uninterpreted Functions + Linear Integer Arithmetic [x] UFLRA - Uninterpreted Functions + Linear Real Arithmetic [x] AUFLIA - Arrays + UF + LIA [x] AUFLIRA - Arrays + UF + LIA + LRA [x] Improve quantifier instantiation heuristics

- [x] Combined theories (3 items)
  - [x] QF_AUFBV - Arrays + UF + BV (validation needed) [x] QF_ALIA - Arrays + LIA [x] QF_ABV - Arrays + BV

- [x] Non-linear arithmetic (2 items)
  - [x] Extend QF_NIA coverage (more benchmarks) [x] QF_NIRA - Non-linear Integer/Real Arithmetic

### Medium Priority: Advanced Features (Complete; per-item plans condensed 2026-09-15)

All three groups are delivered; the goal/design/files/tests/risk plan blocks written for them in
April 2026 are dropped here — the code and its tests are the record.

- [x] Enhanced preprocessing (5 items): bounded-model-checking tactics
  (`oxiz-spacer::tactics::BmcUnrollTactic` with its integration test), more aggressive
  simplification (new Boolean / arithmetic / bit-vector / ITE rewrite rules in
  `oxiz-core::simplification::AggressiveSimplifier`), context-dependent rewriting (dead-branch ITE
  elimination in `oxiz-core/src/tactic/ctx_simplify.rs`), symmetry breaking
  (`oxiz-sat::tactics::SymmetryBreakTactic`), and cube-generation improvements (the VSIDS-depth
  awareness of `CubeGenerator::depth_limit_for_cube`, validated end to end).
- [x] Better quantifier handling (4 items): pattern-based instantiation improvements
  (`PatternCoverScorer`), conflict-driven instantiation (`conflict_score` in
  `conflict_driven.rs`), quantifier-elimination enhancements (virtual substitution,
  Loos–Weispfenning, `oxiz-core/src/qe/virtual_substitution.rs`) and MBQI performance tuning
  (`MBQIBudget::per_quantifier`).
- [x] Proof system enhancements (3 items): optimized proof generation (arena-allocated
  `ProofStep`s in `recorder.rs`), proof minimization, and structured Nelson–Oppen theory-combination
  certificates (`oxiz-proof/src/theory_combination.rs`, with `oxiz-proof/tests/
  theory_combination_proof.rs`).

### Medium Priority: User Experience (Complete)

- [x] Documentation improvements (5 items)
  - [x] Performance tuning guide (docs/PERFORMANCE_TUNING.md) [x] Theory-specific guides (docs/THEORY_GUIDE.md) [x] Common pitfalls and solutions (docs/PITFALLS.md) [x] Migration guide from Z3 (docs/MIGRATION_Z3.md) [x] Case studies and examples (docs/CASE_STUDIES.md)

- [x] API improvements (3 items)
  - [x] EasySolver convenience API (builder pattern) [x] Better error messages (hints, did_you_mean, context_snippet) [x] Timeout and resource limit APIs (ResourceLimits, ResourceMonitor)

- [x] Debugging support (4 items)
  - [x] Solver state visualization (SolverStateSnapshot, DOT graph) [x] Trace generation (TraceEvent, JSON/text output) [x] Better conflict explanations (ConflictExplainer, UnsatExplanation) [x] Model minimization (linear and binary search strategies)

### Low Priority: Ecosystem Integration

- [x] Language bindings — the 2 sanctioned cross-language surfaces (Python + WASM/JS-TS) are both delivered; C/Java FFI bindings were explicitly dropped per the Pure-Rust no-FFI policy
  - [x] Improve Python bindings (oxiz-py enhancements) (planned 2026-04-19) **Goal:** Bring `oxiz-py` to 0.2.1 quality bar: full theory test coverage, README kept in sync with workspace version, parity matrix doc. **Design:** PyO3 surface (1583 LoC, 7 modules, 721-line stub) is mature. Add 5 pytest files for theories implied by stubs but not yet tested. Sync README version strings (`pyproject.toml` needs no sync — it uses `dynamic = ["version"]` via `[tool.maturin]`, reading the version from `Cargo.toml`'s `version.workspace = true` at build time, a more permanent solution than static syncing). Add `PARITY.md` table mapping z3 API → oxiz wrapper → status. **Files:** `oxiz-py/tests/test_quantifiers.py` (new), `oxiz-py/tests/test_arrays.py` (new), `oxiz-py/tests/test_fp.py` (new), `oxiz-py/tests/test_strings.py` (new), `oxiz-py/tests/test_unsat_cores.py` (new), `oxiz-py/PARITY.md` (new), `oxiz-py/pyproject.toml` (no change needed — version is dynamic via `[tool.maturin]`), `oxiz-py/README.md` (version + test-count update); minimal `src/*.rs` patches only if a wrapper is missing. **Tests:** Each pytest file has ≥3 assert cases. Run `cargo build -p oxiz-py --release` (always); `maturin develop + pytest` if toolchain available, else skip with explicit note. **Risk:** maturin unavailable. Mitigation: .py and .md files land regardless; test run is skipped. **Scope cap:** ≤700 LoC net-new. ≤3 new PyO3 wrappers × ≤50 LoC each if needed. [x] JavaScript/TypeScript bindings (via WASM) — **(fixed: js_api is fully wired to oxiz_solver with .d.ts, TS examples, and framework wrappers (React/Vue/Svelte/Deno))**

- [ ] Tool integration (3 items)
  - [x] SMT-COMP 2026 participation — entry package complete; submit when portal opens (~May 2026) [ ] Integration with symbolic execution tools [ ] Integration with verification frameworks

---

## Critical Priority (100% Complete)

### Spacer (PDR) Engine - KEY DIFFERENTIATOR
- [x] Implement Property Directed Reachability for Horn Clauses (CHC)
  - [x] CHC representation (predicates, rules, queries) [x] Frame management (F_0..F_N sequence) [x] POB (Proof Obligation) management [x] Reachability utilities (reach facts, counterexamples, generalization) [x] PDR core algorithm with propagation and blocking
- [x] Loop invariant inference
  - [x] Houdini algorithm for candidate elimination [x] Template-based inference (linear, octagon) [x] SMT-based verification integration
- [x] Software verification pipeline
  - [x] Full CHC solving with invariant synthesis

### Optimization (MaxSMT / OMT)
- [x] MaxSMT core implementation (Fu-Malik with core extraction)
- [x] Core-guided algorithms (OLL with totalizer, MSU3, WMax stratified)
- [x] Totalizer encoding for cardinality constraints
- [x] Optimization Modulo Theories (OMT) - binary/linear/geometric search
- [x] Linear Programming (LP) solver integration
  - [x] Revised simplex method [x] Branch-and-bound for MIP [x] Integer/Binary variable support
- [x] Mixed Integer Programming (MIP) support

### E-Graph Integration
- [x] Tailor e-graph for incremental SMT updates
  - [x] Incremental merge operations [x] Backtrackable union-find [x] Worklist-based congruence closure
- [x] Optimize congruence closure for theory propagation
  - [x] Theory propagator hooks [x] Analysis data per e-class
- [x] Custom e-graph implementation
  - [x] EGraph with EClassId, ENode, EClass abstractions [x] Explanation generation for merges

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
  - [x] CubeGenerator, ParallelCubeSolver, CubeAndConquer [x] 22 tests passing
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
  - [x] McMillan's algorithm (left-biased interpolants) [x] Pudlak's algorithm (symmetric interpolation) [x] Huang's algorithm (right-biased interpolants) [x] Theory-specific interpolants (LIA, EUF, Arrays) [x] Sequence and tree interpolation

### Advanced Features
- [x] Minimal Unsat Cores with parallel reduction
- [x] Craig Interpolation for model checking
- [x] XOR/Gaussian elimination solver
- [x] Quantifier Elimination (QE) enhancements
  - [x] Term graph analysis [x] QE Lite for fast approximation [x] Model-based interpolation (MBI)
- [x] Model subsystem
  - [x] Model evaluator with caching [x] Model completion [x] Prime implicant extraction [x] Value factories

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

### v0.1.3 - COMPLETE (Feb 5, 2026) and v0.2.0 - COMPLETE (Feb 6 - Mar 31, 2026) (condensed 2026-09-15)

v0.1.3 reached the 8-logic parity core with every theory solver validated; v0.2.0 reached 168/168
parity tests and shipped performance phase 1, the `EasySolver` API, the four debugging facilities
and five documentation guides — 6,155 tests (16 skipped, 0 failures), 393,292 Rust lines (312,495
code), 931 files, 0 clippy warnings. The per-item record is "Completed: April 4, 2026" and the
Recent Achievements entries above.

### v0.3.0 (Target: June 2026)
**Focus: Performance Parity and SMT-COMP**
- [~] Performance parity with Z3 (within 1.2x) (planned 2026-04-19)
  <!-- umbrella stays [~] until EP-6e (empirical geomean check) lands; children EP-6a..d may already be [x] -->
  - [x] EP-6a: Extended `Z3ComparisonReport` with `geomean_ratio`, `p50_ratio`, `p95_ratio`, `ratio_count` fields (`#[serde(default)]`); `within_target` recomputed from geomean ≤ 1.2 (not strict per-benchmark); 5 unit tests in `z3_compare.rs` (planned 2026-04-19) [x] EP-6b: `bench/z3_parity` gains `--export-history <dir>` mode writing versioned `history/<YYYY-MM-DD>_<sha>.json` snapshots with per-logic `RatioSummary` breakdown; 6 tests in `bench/z3_parity/tests/history_export.rs` (planned 2026-04-19) [x] EP-6c: `bench/regression/baseline.json` refreshed from v0.2.1 current-branch measurements (was v0.1.3 from Jan 2026, 3 months stale) (planned 2026-04-19) [x] EP-6d: `.github/workflows/perf-regression.yml` extended with `geomean-gate` step — soft-gate (passes when no Z3 data, exits non-zero when `geomean_ratio > 1.2`) (planned 2026-04-19) [ ] EP-6e: Empirical verification — confirm geomean ≤ 1.2 across QF_* logics with Z3 installed (deferred: requires Z3-equipped machine; run next /ultra pass with Z3 available)
- [x] Quantified logic support (UFLIA, UFLRA, AUFLIA)
- [x] Combined theory validation (QF_AUFBV, QF_ALIA, QF_ABV)
- [x] Enhanced preprocessing tactics (planned 2026-04-19)
- [x] Performance regression CI pipeline
- [x] SMT-COMP 2026 entry preparation (completed 2026-05-05)
  - [x] `Track` enum (5 variants: SingleQuery, Incremental, UnsatCore, ModelValidation, ProofExhibition) [x] `submission` module wired into `oxiz-smtcomp/src/lib.rs` with full public API [x] `default_oxiz_2026()` fixed: `bin/smtcomp2026` binary, version from `CARGO_PKG_VERSION` [x] Per-track `starexec_run_<track>` scripts in submission package [x] `smtcomp2026` binary extended with `--track` flag (single|incremental|unsat-core|model|proof) [x] `scripts/package_smtcomp.sh` — assembles complete StarExec ZIP [x] End-to-end submission tests in `oxiz-smtcomp/tests/submission_e2e.rs`

### v1.0.0 (Target: Q4 2026)
**Focus: Production Release**
- [ ] Full Z3 API compatibility
- [ ] Performance at or better than Z3
- [ ] Comprehensive documentation
- [ ] Stable API guarantees
- [ ] Industry adoption ready

---

## Recent Achievements

### 2026-06-01 → 2026-06-09 - v0.2.2 and v0.2.3 Releases (condensed 2026-09-16)

- **v0.2.3 (June 9)**: `DratWriter<W>`/`LratWriter<W>` generic over `W: Write + Send` (breaking rename from
  `DratProof`/`LratProof`); real resultants (Sylvester/Bareiss), leading-coefficient extraction and degree≥3 root
  isolation (`oxiz-nlsat`); sound Nelson-Oppen equality propagation, simplex `optimize_linexpr`, correct push/pop
  tableau snapshots (`oxiz-theories`); solver-backed `check_sat` and MaxSMT selector encoding (`oxiz-opt`); real BMC
  and sound k-induction (`oxiz-spacer`); `Context::eval_in_model`.
- **v0.2.2 (June 1)**: recursive BV term encoding in `BvSolver`; the Z3 API compatibility layer (`TacticRegistry`
  with 19 named tactics, `FuncInterp`/`FuncEntry`, `Z3SortKind`/`Z3Sort`, `substitute`, `Z3Pattern`); real LBD from
  the finalized 1-UIP clause; `BranchingHeuristic::on_conflict_var` feeding `MLEnhancedVSIDS`; LRU caches in the
  simplifier, EUF explanations and the theory-lemma combiner; Linux `VmHWM` peak memory; Big-M phase-1 simplex;
  `#![allow(dead_code)]` removed from 40+ modules. 6,735 tests, ~419,576 code lines across ~1,012 files.

### May 5–18, 2026 - v0.2.2 passes 2–6 (condensed 2026-09-19; the v0.2.2 entry above carries the outcome)

- Plus: the `MLBranchingHeuristic` adapter behind `SolverConfig::external_branching`; `oxiz-proof`'s
  `transform.rs`/`compression.rs` deleted (−1,167 lines over a non-existent `ProofRule`); the SMT-COMP 2026 entry
  package; `z3_compat_ext{,2,3}.rs`; LIA `feasibility_pump`/`probe_variables`/`manage_cuts`. 6,629 → 6,834 tests.

### April 4 → April 25, 2026 - Statistics snapshots (v0.2.1; condensed 2026-09-19)

- 931 → 1,182 Rust files, 312,495 → 408,320 code lines, 6,155 → 6,415 tests, 16 → 17 crates, 0 stubs; five EUF
  allocation improvements, the Set-theory CDCL(T) interface, the Sylvester discriminant (degree ≥ 4), Hong projection.

### Feb 5 → March 31, 2026 - parity, milestone and polish (condensed 2026-09-19)

- **March 31**: nine performance optimizations (arena allocator, clause pool, SIMD polynomial ops, caches, parallel
  theory checking, lock-free structures, lazy evaluation); the `EasySolver` API, better errors, resource limits,
  state visualization, traces, conflict explanations, model minimization; 6,155 tests. **March 23 (v0.3.0)**:
  168/168 Z3 parity tests, 5,993 tests, every file under 2,000 lines. **February 5**: 88/88 benchmarks, 8 logics.

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
   - Integration with symbolic-execution tools and verification frameworks (re-scope once a specific target is chosen) SMT-COMP 2026 submission once the portal opens

---

**Status**: Production Ready
**Current Version**: v0.3.4 (2026-08-26)
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
- [x] **Test infra: nextest has no `slow-timeout`, so a runaway test can only be stopped by the OOM killer** — nextest's default reports SLOW at 60s and never terminates, so an unbounded allocation ran until the machine died. — **(fixed: `.config/nextest.toml` sets `slow-timeout = { period = "60s", terminate-after = 3 }` on `profile.default` and `terminate-after = 2` on `profile.ci`. Amended 2026-09-19 under `#P2b-45` / decision (16): `bv_wide_soundness::wide_sub_of_add_two_vars_is_identity_above_64_bits` has its own override at `terminate-after = 10`, because it bit-blasts three wide two-variable soundness goals in one body and took 151.5 s inside a workspace run on this machine — under the shared 180 s ceiling its colour was a property of what else was running.)**
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

## Stubs to implement (2026-06-12 and 2026-06-22 /cooljapan-stub-check sweeps — all closed; condensed 2026-09-15)

Both sweeps found the same two items plus one more, and all three are fixed: `optimize()` routes
every feasibility subquery through a fresh theory-complete `Solver` (`x = y ∧ x ≠ y` is now
`Unsat`); `assert_is_normal`'s encoding was sound all along and the real defect was an unsound
double-solve retry in `FpSolver::check()` (`restore_to_trail_size` left residue), removed; and the
datatype case-analysis QE path has a real constructor case split in both `qe/datatype/plugin.rs`
and `qe/datatype/case_analysis.rs`.

---

## Production-Readiness Audit Findings (added 2026-07-16, ultracode audit)

**Method**: 19 scoped deep-audit agents (per-crate + cross-cutting: SMT-LIB 2.6 compliance, panic audit, Z3 gap vs upstream Z3, test-quality gap, release/packaging) followed by adversarial verification agents (90 verdicts collected before the run was stopped early by request; items below marked *unverified* did not get a verification pass — verify before fixing). **Build baseline (2026-07-16)**: `cargo check --workspace --all-features` clean; `cargo clippy --all-targets --all-features` 0 warnings; `cargo nextest run --workspace --all-features` 6826/6826 passed (16 skipped) — all tests passed *despite* the findings below, i.e. the suite did not exercise these paths (see the P2 test-gap items).
**Counts (after location-dedupe)**: P0 confirmed-critical 20 | P1 confirmed-major 30 | P2 unverified-critical 42 | P3 unverified-major 131 | P4 minor/downgraded 105

**Re-verification, three passes (2026-07-18 release-polish, 2026-07-21 v0.3.0 hardening, 2026-07-31 0.3.1 sweep; condensed 2026-09-15)**: every P0-P4 item below was re-read against the tree of the day and marked `[x]` only when the bug pattern was confirmed fixed, made honest (a documented no-op or an honest `Unknown` instead of a silent wrong answer), or shown not-a-bug by inspection. Running total after the third pass: **20/20 P0**, **30/30 P1**, **42/42 P2**, **129/131 P3**, **102/105 P4**, **7/7 Policy/Release Chores**. The 5 items still `[ ]` (2 P3, 3 P4) are Z3 `recfun` end-to-end support, the default-off `property-tests` feature, `oxiz-core`'s decorative secondary BV/FP/datatype theory submodule (two entries, no internal callers), and `mk_bv_concat`'s release-build width default. None is on the default solve path; the grouped record is the "Remaining (post-0.3.0 hardening)" section below.

### P0 — Confirmed Critical (soundness: wrong sat/unsat/model; fix first)

- [x] `oxiz-math/src/polynomial/extended_ops.rs:1069` — Sturm sequence built from pseudo-remainders without sign normalization yields wrong root counts *(scope: math; effort: small)*
  - pseudo_remainder scales by lc(b)^k which can be negative, breaking the Sturm sign invariant. Concretely p=-x^2+1 gives chain [-x^2+1, -2x, 2] so count_roots_in_interval(-2,2) returns 0 instead of 2. Propagates to isolate_roots, realclosure::AlgebraicNumber::new (assert panics), and CAD/nlsat root reasoning: wrong sat/unsat. **Fix**: Use exact rational remainder, or multiply pseudo-remainder by sign(lc_b)^k so the scale factor is always positive (Z3 uses signed pseudo-remainder).
- [x] `oxiz-math/src/grobner/buchberger.rs:993` — NraSolver::check_sat returns Sat without ever solving non-constant linear inequalities *(scope: math; effort: medium)*
  - Inequalities that reduce to non-constant polynomials of total_degree<=1 skip both the constant check and the has_complex_inequality Unknown path, so check_sat returns Sat. Asserting x>0 and x<0 (no equalities) returns Sat for an unsatisfiable system: a wrong answer from a public solver API. **Fix**: Route remaining linear inequalities through the simplex/LP solver; return Unknown for any inequality not fully decided instead of Sat.
- [x] `oxiz-sat/src/solver/conflict.rs:83` — Conflict analysis assumes reason clause lits[0] is the propagated literal; binary-graph propagation violates this, dropping antecedent literals *(scope: sat; effort: small)*
  - analyze() skips reason-clause position 0 (`start = 1`), assuming the propagated literal sits there. Watch-based propagation maintains that invariant, but binary-implication-graph propagation (propagate.rs:28 `assign_propagation(implied_lit, clause_id)`) never reorders the stored (sorted) clause. When the implied literal is lits[1] (~50% of original/hyper-binary clauses), the false antecedent lits[0] is silently omitted from the learned clause, producing over-strong clauses that can flip SAT instances to UNSAT. analyze_theory_conflict (line 453 `clause.lits[1..]`) has the same flaw. **Fix**: For binary-graph propagations, swap the implied literal to lits[0] before recording the reason, or resolve reason clauses by value (skip lit == current_lit) instead of by position.
- [x] `oxiz-sat/src/clause.rs:412` — Clause slot reuse via free_list breaks lazy watcher cleanup, letting stale watchers drive bogus unit propagations *(scope: sat; effort: medium)*
  - remove() pushes the ClauseId to free_list; add() immediately reuses the slot for a new clause. Stale watchers (cleaned only lazily via the `deleted` flag; WatchLists::remove_clause is dead_code) now reference a live, different clause. propagate() never verifies the watched literal is in the clause: it assumes the falsified literal is at lits[1] and may propagate lits[0] as "unit" while lits[1] is true/undef, and its swaps corrupt the real watchers' positions — unsound propagations and wrong answers on long runs with clause deletion. **Fix**: Do not recycle ClauseIds while stale watchers may exist: scrub watch lists on remove (use remove_clause), or defer slot reuse until a full watch-list garbage collection pass.
- [x] `oxiz-sat/src/solver/mod.rs:860` — solve_with_assumptions after a prior solve() treats leftover model decisions as fixed, returning false UNSAT *(scope: sat; effort: small)*
  - solve() returns Sat leaving the full trail (decisions at levels >0). solve_with_assumptions never backtracks to root first; assumption_level_start is captured at the dirty level and an assumption that merely disagrees with the previous arbitrary model hits `value.is_false()` and immediately returns (Unsat, core). Example: (a∨b); solve() picks ¬a,b; solve_with_assumptions([a]) reports UNSAT though a∧(a∨b) is SAT. This breaks the standard MaxSAT/incremental usage pattern. The extracted core also reads stale `seen` flags. **Fix**: Call backtrack_with_phase_saving(0) at the top of solve_with_assumptions before capturing assumption_level_start, and only report UNSAT when the assumption is false at level 0.
- [x] `oxiz-nlsat/src/solver/decide.rs:500` — Irrational roots silently dropped: solver returns wrong UNSAT for e.g. x^2 > 2 *(scope: nlsat; effort: large)* — **(fixed: SturmSequence::isolate_roots is now wired into the feasible-region path (pick_arith_value -> compute_arith_regions -> univariate_regions); x^2>2 returns Sat with a witness, x^2=2 returns Unknown, never a wrong Unsat)**
  - find_univariate_roots only finds RATIONAL roots; quadratic with non-square discriminant returns Vec::new() ('Irrational roots - cannot represent exactly'). compute_feasible_region then treats the polynomial as sign-constant, so for asserted 'x^2-2>0' the feasible set is EMPTY and solve() returns Unsat at level 0 (mod.rs:653). oxiz-theories/src/nlsat.rs:369 trusts Unsat for univariate atoms, so the final answer is wrong. **Fix**: Use SturmSequence root isolation (algebraic numbers / isolating intervals) instead of rational-only roots in find_univariate_roots, or return IntervalSet::reals() when roots may be missing and rely on validation.
- [x] `oxiz-nlsat/src/solver/mod.rs:655` — Infinite loop: empty feasible region at level>0 backtracks without learning, re-makes identical decision *(scope: nlsat; effort: large)* — **(fixed: pick_arith_value now returns a 4-variant ArithDecision (not None); the solve loop learns a lemma (ProvedEmpty) or terminates (Unknown/Unsat) in every branch, so the no-lemma backtrack spin is gone)**
  - When pick_arith_value returns None at level>0, solve() calls backtrack(level-1) with no lemma, no activity bump, no phase flip. decide() then re-picks the same variable with the same saved phase, reproducing the identical state forever. Example that hangs: (x>1) AND (x<-1 OR x<5) — trivially SAT but loops indefinitely. No conflict is counted so restarts never fire. **Fix**: Learn a clause negating the decisions/atoms whose interval intersection is empty (NLSAT semantic-conflict lemma), or at minimum flip the saved phase of the last decision before re-deciding.
- [x] `oxiz-nlsat/src/nia.rs:363` — NIA branch-and-bound adds both branch constraints permanently to one shared solver *(scope: nlsat; effort: large)* — **(fixed: create_branch is gone; branch_and_bound snapshots the base problem and rebuilds a fresh path-scoped NlsatSolver per node (rebuild_solver); push_branch only records bounds)**
  - create_branch adds 'x<=floor' and 'x>=ceil' as permanent unit clauses to the SAME NlsatSolver; popping a BranchNode never retracts them, so after pushing both branches the solver holds contradictory constraints and every node solves the same over-constrained problem. branch_and_bound then exhausts the stack and returns Unsat (nia.rs:276) for satisfiable integer problems. NiaSolver is the QF_NIA path in oxiz-theories. **Fix**: Use push/pop scopes or assumption literals per branch node so constraints are retracted on backtrack; never treat search-space exhaustion under leaked constraints as Unsat.
- [x] `oxiz-nlsat/src/solver/mod.rs:505` — solve() never resets trail/arithmetic state, breaking incremental re-solve *(scope: nlsat; effort: medium)*
  - After a Sat answer, the trail, decision levels, and arithmetic values remain assigned. NiaSolver re-invokes solve() after add_clause: the new unit literal is assigned at a stale non-zero level and theory_propagate evaluates it against the stale model, producing spurious conflicts; analyze_conflict resolves Unit/Theory-justified literals away with no reason clause and can return an empty learnt clause, which solve() reports as Unsat (mod.rs:548-549). **Fix**: Backtrack to level 0 and clear arithmetic assignments at solve() entry; give Unit/Theory-justified literals proper reasons in analyze_conflict instead of silently dropping them.
- [x] `oxiz-nlsat/src/cad.rs:518` — Sturm sequence built from sign-unnormalized pseudo-remainders gives wrong root counts *(scope: nlsat; effort: medium)*
  - pseudo_remainder scales by lc(divisor) on every reduction step; when the leading coefficient is negative an odd number of scalings flips the remainder's sign, so the chain is not a Sturm chain. Example: p = 4-x^2 yields chain (-x^2+4, -x, +4) and count_roots() = 0 despite roots +/-2. All root-atom evaluation (evaluate_root_atom) and CAD lifting depend on isolate_roots, so answers involving negative-leading-coefficient polynomials are wrong. **Fix**: Track the sign of lc(b)^k applied during pseudo-division and multiply the remainder by it (or normalize lc(b) positive before division) so the chain satisfies Sturm's sign conditions.
- [x] `oxiz-nlsat/src/portfolio.rs:261` — PortfolioSolver solves empty solvers: returns Sat for every input *(scope: nlsat; effort: large)*
  - run_parallel_solvers creates fresh NlsatSolver::new() instances and never copies the base problem ('simplified - no actual problem to solve yet'). The empty problem is trivially Sat, so PortfolioSolver::solve() always answers Sat with an empty model, including for unsatisfiable inputs. config.timeout and the diverse configs are also ignored. Public API re-exported from lib.rs. **Fix**: Clone base_solver's clauses/atoms into each worker via create_configured_solver, apply per-worker configs, honor timeout, and extract real models/cores; otherwise remove the API until implemented.
- [x] `oxiz-core/src/smtlib/parser/terms.rs:455` — Parser silently turns (div a b) and (mod a b) into subtraction *(scope: core-rest; effort: small)*
  - This parser IS the production path: oxiz-cli -> oxiz_solver::Context::execute_script -> parse_script. Any script using integer div/mod gets a semantically different formula, so check-sat can answer wrong on plausible LIA inputs. TermManager already has mk_div/mk_mod (ast/manager/builder.rs:314,320) but the parser ignores them. **Fix**: Route "div" to self.manager.mk_div(lhs, rhs) and "mod" to mk_mod; add regression tests like (assert (= (div 7 2) 3)).
- [x] `oxiz-core/src/smtlib/parser/terms.rs:929` — Real division "/", abs, to_real, to_int, divisible parsed as Bool-sorted uninterpreted functions *(scope: core-rest; effort: medium)*
  - The operator match has no case for "/" or other core Int/Real ops, so they fall to the default arm and become mk_apply with Bool default sort. Arithmetic theory then ignores these constraints entirely — QF_LRA scripts with division get wrong sat/unsat answers on the production parse path. **Fix**: Add explicit cases for "/", "abs", "to_real", "to_int", "divisible"; reject genuinely unknown undeclared operators with a ParseError instead of Bool-sorted apply.
- [x] `oxiz-core/src/ast/manager/query.rs:445` — TermManager::substitute silently skips Apply, BV, String, FP, Xor, Distinct, Div/Mod, quantifier and Let terms *(scope: core-ast; effort: medium)*
  - substitute_cached handles only ~15 term kinds; everything else hits 'Some(_) => id' with comment 'For complex terms, just return as-is for now'. Tactics solve_eqs, ackermann, propagate, ctx_simplify and quantifier instantiation (tactic/quantifier.rs:533) rely on it: substituting x->3 in f(x) or any bitvector/string assertion returns the term unchanged, so solved equations are dropped while occurrences remain — wrong sat/unsat and wrong models. **Fix**: Handle all TermKind variants generically via get_children plus a rebuild function (as rewrite_children does), and descend into quantifier bodies with bound-variable shadowing.
- [x] `oxiz-core/src/simplification/mod.rs:299` — Boolean absorption in AND/OR drops all other conjuncts/disjuncts *(scope: core-ast; effort: small)*
  - try_boolean_absorption_in_and returns just 'candidate' and simplify_and returns it as the whole result. And(a, Or(a,b), c) simplifies to 'a', silently dropping c — an UNSAT formula (c=false) becomes SAT. The OR variant (line 316) similarly turns Or(a, And(a,b), c) into 'a', dropping disjunct c and turning SAT into UNSAT. Reachable via AggressiveSimplifier with aggressive=true. **Fix**: Absorption must remove only the absorbed Or/And argument and keep the remaining args: rebuild mk_and(args minus the absorbed term), not return candidate alone.
- [x] `oxiz-core/src/simplification/mod.rs:343` — try_factor_or_of_ands discards every disjunct outside the matched pair *(scope: core-ast; effort: small)*
  - For Or(And(x,a), And(x,b), c, ...), the factoring rule returns And(x, Or(a,b)) and drops c and all other disjuncts, strengthening the formula — a satisfiable input can become UNSAT. Fires in aggressive simplification whenever any two AND disjuncts share a conjunct. **Fix**: Include the untouched disjuncts: build Or(And(common, Or(left_rest,right_rest)), remaining_args...).
- [x] `oxiz-core/src/rewrite/bv.rs:417` — BvShl/BvLshr rewrite returns Unchanged(lhs): x << y silently becomes x *(scope: core-ast; effort: small)*
  - rewrite_bvshl (line 417) and rewrite_bvlshr (line 462) end with RewriteResult::Unchanged(lhs) when args are non-constant. Through CombinedRewriter's result.term() (combined.rs:514) the entire shift expression is replaced by its left operand, changing formula semantics for any symbolic shift. **Fix**: Return Unchanged(manager.mk_bv_shl(lhs, rhs)) / mk_bv_lshr(lhs, rhs); both builders exist.
- [x] `oxiz-core/src/tactic/quantifier.rs:968` — DER forall rule is logically inverted: rewrites ∀x.(x=t ∨ ψ) to ψ[t/x], which is unsound *(scope: core-tactic; effort: medium)*
  - Correct DER eliminates a DISEQUALITY disjunct: ∀x.(x≠t ∨ ψ) ≡ ψ[t/x]. The code eliminates the positive equality instead, and also rewrites the x≠t→ψ implication (≡ x=t ∨ ψ) to ψ[t/x]. Goal {∀x.(x=5 ∨ P(x)), ¬P(6)} is UNSAT but becomes {P(5), ¬P(6)} = SAT. ∀x.(x=t) also rewrites to true. **Fix**: For Forall, match Not(Eq(x,t)) disjuncts (and Eq antecedents of Implies) instead of positive equalities; keep the exists/And path as-is.
- [x] `oxiz-core/src/tactic/quantifier.rs:689` — SkolemizationTactic reuses Skolem names across assertions and ignores polarity *(scope: core-tactic; effort: medium)*
  - skolemize() (ast/normal_forms.rs:758) resets counter=0 per call, so per-assertion calls give distinct existentials the SAME sk_0 variable: {∃x.P(x), ∃x.¬P(x)} (SAT) becomes {P(sk_0), ¬P(sk_0)} (UNSAT). skolemize also recurses through Not/Implies without flipping polarity, so ¬(∃x.P(x)) becomes ¬P(sk_0) (UNSAT→SAT), and Skolem function args are built with hardcoded bool_sort (normal_forms.rs:855). **Fix**: Thread one global fresh-name counter through the goal, track polarity (skolemize Exists only at positive polarity, Forall at negative), and use real universal-var sorts for Skolem function arguments.
- [x] `oxiz-core/src/tactic/quantifier.rs:624` — QuantifierInstantiationTactic instantiates Forall terms found at any polarity as asserted facts *(scope: core-tactic; effort: medium)*
  - collect_quantifiers (line 646) gathers every Forall subterm, including ones under Not, Or, or Implies antecedents, then pushes φ(t) as a new top-level assertion. For goal ¬(∀x.P(x)) ∧ ¬P(c) with trigger matching c, the added P(c) flips SAT to UNSAT. **Fix**: Only instantiate quantifiers that occur as positive-polarity top-level assertions (or track polarity during collection and skip negative/mixed occurrences).

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

### Audit Coverage Notes (scope summaries) (condensed 2026-09-16)

- Nineteen scoped agents (bindings, core-ast, core-rest, core-tactic, frontends, math, nlsat, opt-proof,
  panic-audit, release-audit, sat, smtlib-compliance, solver-core, solver-rest, spacer, test-gap, theories-arith,
  theories-rest, z3-gap); their per-item findings are the P0–P4 lists below. The pattern they agreed on: the
  CDCL(T) core, SAT engine, term manager and arithmetic/BV/EUF theories were real, while the layers around them —
  tactics, QE, string/FP/array pre-checks, MaxSAT/OMT, Spacer, bindings, most options — were stubs or facades.

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

## cargo-formal intake (added 2026-09-09, from cargo-formal Phase 2 / 2b)

[cargo-formal](https://github.com/cool-japan/cargo-formal) is a Cargo subcommand for formal
verification of Rust — it lowers reachable MIR into its own IR, generates bounded verification
conditions, and discharges them through **OxiZ as its SMT backend**. Its Phase 2 probes and its
`formal-conformance` differential suite exercised OxiZ from outside, at the `(check-sat)` /
`(get-value)` / `(get-proof)` boundary: the seventeen findings `U-Z1` … `U-Z17` below, plus the
two `oxiz-sat` `Lit` bugs its own self-hosting harness turned up; the `#P2b-*` items were raised
while fixing them. Fixtures live in that repo at `crates/formal-conformance/fixtures/{c,u}*.smt2`;
each `u*.expected` carries an `upstream: U-ZNN` line and fails on purpose once OxiZ agrees.

**Every `[x]` below was fixed in the 0.3.4 working tree on 2026-09-08/09, except
`#P2b-19`–`#P2b-22`, fixed on 2026-09-14**, and the whole set was gated together on the integrated
tree: `cargo nextest run --release --workspace` **10,465 passed / 13 skipped / 0 failures**
(10,437 / 13 / 0 was this same gate on 2026-09-09), `cargo clippy --release --workspace
--all-targets -- -D warnings` clean, `cargo deny check bans` ok, every touched file under 2,000
lines, the 22 conformance fixtures **21 agree / 1 disagree** (`u06`, U-Z16; 23/24 on 2026-09-15 with `u11`/`u12`), and the differential
fuzz that measured U-Z10 **0 wrong `sat`, 0 wrong `unsat`, 0 `unknown`** on its final re-run —
2026-09-14, independent per-block seeds, **24,054 verdicts at width 8, 17,713 truly-sat / 6,341
truly-unsat**, against crates.io 0.3.3's 846 wrong `sat` (this tree: 1 + 12 + 21 before
`#P2b-19`/`#P2b-20`). It supersedes *as the 0.3.3 baseline* the earlier "12,000 trials, 11,297
truly-sat / 703 truly-unsat, 34/703" figures, whose block seeds overlapped; per-family table
(A/B/C; B 6,300 trials at widths 65/96/128, C 1,600) in CHANGELOG 0.3.4. cargo-formal keeps its
`oxiz = "=0.3.3"` pin and Rule-W width restriction **until that release ships**.

Line numbers below were re-verified on 2026-09-09, after the two file splits this work forced
(`theory_manager.rs` → `theory_manager/bv_bridge.rs`, `bv/solver.rs` →
`bv/solver/{budget,combination}.rs`); where a line would rot quickly the anchor is a name.

### Fixed in the 0.3.4 working tree

- [x] **U-Z10 — Boolean structure over bit-vector atoms answered `sat` for unsatisfiable formulas,
  at a measured 4.8–9.2 % rate** (`(= a #x0f)` with `(not (and (bvule a #x0f) (bvule a #x10)))`;
  2/1,500 → 34/12,000 under re-seeding). `BvSolver::pop` (`oxiz-theories/src/bv/solver.rs:1728`)
  deleted the embedded SAT clauses but left `term_to_bv` / `ult_cache` / `bool_node` populated, so
  the idempotence guard (`oxiz-solver/src/solver/theory_bv_encode.rs:164`) never rebuilt the
  circuit — **not** the model gate the report guessed at. Fixtures `u01`, `u08`, `u09`; evidence
  `p2-oxiz-capabilities.md` §2.1, `p2-notes-s0b-oxiz.md` F1. — **(fixed in 0.3.4 by three undo
  journals, `term_to_bv_journal`/`ult_cache_journal`/`bool_node_journal` (`bv/solver.rs:134-138`,
  filled at `:270`/`:380`/`:727`, snapshotted at `:1716-1723`, truncated at `:1728`). Tests:
  `oxiz-solver/tests/bv_scope_rollback.rs` (19), `oxiz-theories/tests/bv_wide_scope_and_model.rs`
  (7); of the 32 for U-Z10 + U-Z11, **23 fail pre-fix**, the 9 passing on both are controls.)**
- [x] **U-Z10 (backstop) — the model-verification gate could not read a single bit-vector value,
  so it approved every QF_BV model unconditionally.** `EvalVal` was `Bool | Num`,
  `parse_value_term` answered `Undetermined` for a `BitVecConst`, every `Bv*` kind fell in
  `open_in_model`'s `_ =>` arm; five fixtures answered `sat` with a model falsifying their own
  assertions. — **(fixed in 0.3.4 by `EvalVal::Bv { value: BigInt, width: u32 }`
  (`oxiz-solver/src/solver/mod.rs:546`), `solver/model_eval_bv.rs` folding through
  `oxiz_core::ast::bv_fold` — 20 primitive `Bv*` arms plus `BitVecConst` and `Distinct`, the
  desugared forms pinned by two named tests. Tests: 17 unit tests vs a `u128` reference at widths
  1/7/8/64/65/128/200 and `oxiz-solver/tests/model_gate_bv.rs` (8; its 47-script table is the
  false-`unknown` guard). Backstops U-Z11: `u02`/`u10` moved wrong `sat` → `unknown` alone.)**
- [x] **U-Z10 (sibling family) — `Constraint::Gt` / `Ge` over bit-vector operands asserted nothing
  at all.** `(= a #x0f) ∧ (> a #x0f)` answered `sat` while the `<` control answered `unsat`: both
  inner `match constraint` blocks of the BV comparison arm handled only `Lt`/`Le` and ended in a
  silent `_ => {}`. Evidence cargo-formal `p2b/w0/I0-a.md` §4.1. — **(fixed in 0.3.4 at both
  layers: `Gt(a,b) => assert_ult(b,a)` / `Ge(a,b) => assert_ule(b,a)` and their negatives in
  `oxiz-solver/src/solver/theory_manager.rs`, both matches ending in a spelled-out `Eq | Diseq |
  BoolApp => false`, plus U-Z14's parse-time rejection of `(> bv bv)`. Tests: seven rows in
  `oxiz-solver/tests/bv_scope_rollback.rs`, two pinning the parser rejection.)**
- [x] **U-Z10 (fallout) — the model evaluator folded the term *tree*, not the DAG.** Terms are
  hash-consed, so a chain of `n` shared doublings costs `2^n` visits while `ENCODE_DEPTH_LIMIT`
  bounds depth, not node count — invisible until the bit-vector arms above landed: 90 ms at depth
  20, 1,376 ms at 24, 22,809 ms at 28. — **(fixed in 0.3.4 by a per-call `FxHashMap<TermId,
  EvalOutcome>` memo in `Solver::eval_in_model_outcome`
  (`oxiz-solver/src/solver/model_eval.rs:1076`/`:1096`/`:1123`), a `frame_terms` stack beside
  `frames`, `depth` not in the key. Test
  `model_gate_bv.rs::a_shared_bit_vector_dag_is_not_folded_as_a_tree`, depth 60, 0.024 s.)**
- [x] **U-Z11 — nested arithmetic over *free* bit-vectors wider than 64 bits answered both `sat`
  and `unsat` wrongly.** `encode_add_const` (`oxiz-theories/src/bv/solver.rs:1202`) read bit `i` of
  a `u64` constant as `(constant >> i) & 1` over the full width: a debug panic for `i >= 64`, bit
  `i % 64` in release, so every `bvsub`/`bvneg` circuit above 64 bits added `2^64 + 2^128 + …`.
  Fixtures `u02`, `u10`; seven shapes at widths 65/96/128: **21 wrong verdicts before**, 0 after.
  — **(fixed in 0.3.4 by a total `const_bit_of` (`bv/solver.rs:47`), a saturating `width_mask`
  (`bv/propagator.rs:34`; `full(65)` was `[0, 1]`, `full(128)` `[0, 0]`) and `get_model`'s `u64`
  key (which merged `1` and `2^64`) becoming the full `BigUint`. Tests:
  `oxiz-solver/tests/bv_wide_soundness.rs` §5 (8), 3 unit tests in `propagator.rs`.)**
- [x] **U-Z12 — `:timeout`, `:max-conflicts` and `:max-decisions` were not polled inside a
  bit-blasted solve; one 64-bit multiplier VC ran past 600 s under `:timeout 20000`.** Polled only
  *before* the work while `BvSolver::check` ran `self.sat.solve()` unbudgeted. Evidence
  `p2-oxiz-capabilities.md` §5.1, cargo-formal `p2b/w0/I0-b.md` §1. — **(fixed in 0.3.4:
  `set_deadline` / `set_max_decisions` polled in `should_stop_search` (`oxiz-sat/src/solver/mod.rs`,
  clock read once per `DEADLINE_POLL_INTERVAL = 256`); `BvSolver::set_budget`
  (`oxiz-theories/src/bv/solver/budget.rs:69`) applied before both `sat.solve()` calls, a *total*
  allowance surviving `reset`, 1/4 reserved for `Unsat` re-verification; one deadline per check
  (`check_core.rs:188`), budgets armed at `:231-237`; `bv_run_check` spelling out `Propagate(_)`
  and `Unknown | Err(_)`. Measured: multiplier VC `unsat` 2,808 ms unbudgeted; `:timeout 100`
  **`unknown` in 105 ms** vs 0.3.3's 17,052 ms; `:max-conflicts 50` 30 ms vs 13,079 ms;
  `:max-decisions 50` does **not** cut it. Tests: `oxiz-sat/tests/budget_deadline.rs` (10),
  `oxiz-theories/tests/bv_budget.rs` (5), `oxiz-solver/tests/budget_bitblast.rs` (8, one
  `#[ignore]`d 12.24 s control). Residual leaks: `#P2b-6`…`#P2b-9`.)**
- [x] **U-Z13 — `(get-value)` and `(get-model)` disagreed about a bit-vector value's radix, and an
  unassigned constant changed radix on its own** (`#x05` vs `#b00000101`; unconstrained, both
  `#b00000000`): two hand-rolled `format!("#b{:0>width$}", …)` calls in `model_fmt.rs` ignored the
  width. Evidence `p2-oxiz-capabilities.md` §4.1. — **(fixed in 0.3.4 by one rule in one place —
  `#x` when the width is a multiple of 4, `#b` otherwise — `format_bitvec_literal`
  (`oxiz-core/src/smtlib/printer/mod.rs:60`), with `Context::format_value` and `default_value`
  (`oxiz-solver/src/context/model_fmt.rs`) delegating to it; `printer/pretty.rs`'s `(fp …)`
  triples stay `#b`. Tests: `oxiz-solver/tests/get_value_radix.rs` (4 × widths 8/12/13/64/65).)**
- [x] **U-Z14 — `=`, `distinct`, `ite`, the connectives and `<`/`<=`/`>`/`>=` had no sort check,
  so contradictory scripts answered `sat`.** `(= a8 b16)`, `(not a8)` and `(> a8 #x0f)` interned
  silently as free Booleans; only the `bvadd` family was width-checked. Fixtures `u03`, `u04`,
  `u05`; cargo-formal `p2b/w0/I0-a.md` §4.1(b). — **(fixed in 0.3.4: `sorts_are_compatible`,
  `check_same_sorts`, `check_bool_operands`, `is_untyped_placeholder`, `check_arith_operands` in
  `oxiz-core/src/smtlib/parser/build.rs`, wired into `build_unary`/`build_ternary`/`build_variadic`;
  messages and the two relaxations (`Int`/`Real` still mix; `parse_term` outside script mode) in
  CHANGELOG 0.3.4; **no existing test changed**. Tests: `oxiz-core/tests/parser_sort_checks.rs` (9).)**
- [x] **U-Z17 — the SMT-LIB 2.7 bit-vector overflow predicates did not exist in the parser**
  (`bvuaddo`, `bvsaddo`, `bvusubo`, `bvssubo`, `bvumulo`, `bvsmulo`, `bvnego`; fixture `u07`
  errored where the answer is `unsat`). — **(fixed in 0.3.4 as desugarings into existing term
  kinds — `build_bv_overflow_binary` (`oxiz-core/src/smtlib/parser/build.rs:367`) and the `bvnego`
  arm of `build_unary`; definitions in CHANGELOG 0.3.4 "Added". Tests:
  `oxiz-solver/tests/overflow_predicates.rs` (13) — all 256 ordered width-4 pairs for all seven
  predicates and both polarities, 205 width-8 pairs against a reference cross-checked over all
  65,536, plus width 64.)**
- [x] **U-Z2 — the same seven overflow predicates, tracked twice.** A blueprint backlog item
  predating U-Z17; checked 2026-09-09, the two lists are identical and all seven parse and solve.
  — **(fixed in 0.3.4 by the U-Z17 change.)**

- [x] **#P2b-15/#P2b-16 — `oxiz-sat` literal arithmetic wrapped silently at the index extremes.**
  `Lit::pos(var)` computed `var.0 << 1`, so `Lit::pos(Var(2^31))` was **variable 0** with no
  diagnostic; and `from_dimacs(i32::MIN)`'s `to_dimacs` returned `i32::MIN` for *both* polarities
  in release (overflow panic in debug). Both found by cargo-formal's `oxiz-sat-lit` self-hosting
  harness (blueprint §14, previously unreported upstream). — **(fixed in 0.3.4: `Var::MAX_INDEX =
  (1 << 31) - 2` (`oxiz-sat/src/literal.rs:27`) enforced by `debug_assert!` in `Var::new`,
  `Lit::pos` and `Lit::neg`, `const fn`-legal with no release cost; `to_dimacs` (`:110`) delegates
  to the new non-panicking `try_to_dimacs` (`:127`, `checked_add` + `i32::try_from`) and otherwise
  panics through an explicit `match` — no `unwrap`, no `expect`. Tests: index-extreme round trips,
  `try_to_dimacs_is_none_exactly_where_to_dimacs_panics`, and five `#[should_panic]` guards.)**
- [x] **#P2b-17 — `#[must_use]` on the six `BvSolver::assert_*` methods, and 74 call sites that
  discarded the answer.** All six of `assert_eq`/`assert_neq`/`assert_ult`/`assert_ule`/`assert_slt`/
  `assert_sle` in `oxiz-theories/src/bv/solver.rs` return `bool` meaning "I actually asserted
  something", and a discarded `false` read as "no theory objection" about an atom the circuit was
  never told about. — **(fixed in 0.3.4: the attribute on all six, all 74 discarding sites across 10
  files rewritten as `assert!(<call>)`, every one passing; `bv_run_check`
  (`solver/theory_manager/bv_bridge.rs:84`) takes an `asserted: bool` and sets `bv_atom_unmodelled`
  (`theory_manager.rs:239`), folded into `resource_exhausted()` to turn a final `Sat` into
  `Unknown` — `#P2b-12`.)**
- [x] **#P2b-18 — two comments describing the model gate as bit-vector-blind became false.**
  `check_array/eval_bv.rs`'s module doc and `check_array/tests.rs:262` both said the gate's
  `EvalVal` "has no bit-vector variant at all", the load-bearing rationale for that file. —
  **(fixed in 0.3.4 by rewriting, not deleting: the gate reads bit-vector values but inspects a
  *finished* candidate model, while this evaluator runs *during* the search on the partial
  assignment, where it can find the conflict that reaches the correct `Unsat`.)**
- [x] **#P2b-19/#P2b-20 (2026-09-14) — two `push`/`pop` defects, one a false `unsat`.** (19) A
  learned clause outlived the scope that entailed it: `learn_clause` (`oxiz-sat/src/solver/learn.rs`,
  all three arms) pushed each id to `learned_clause_ids` but not to `assertion_clause_ids.last_mut()`,
  which `pop` deletes, so `(assert A) (push 1) (assert B) (check-sat) (pop 1) (assert C) (check-sat)`
  answered `unsat` for a `{A, C}` the same tree answered `sat` for spelled flat; reachable only once
  the U-Z10 journals re-encode a popped circuit, so 0.3.3 answers all thirteen witnesses correctly.
  (20) `pop` ended with an unconditional `self.trivially_unsat = false`, a latch over the whole
  clause database with no record of its level, so `(assert (distinct b b)) (push 1) (pop 1)
  (check-sat)` answered `sat`; sort-independent, not a 0.3.4 regression, and missed by the model
  gate too (`#P2b-22`). — **(fixed in 0.3.4 by `Solver::register_learned_at_assertion_level`
  (`learn.rs:129`) from all three arms, with `pop`'s per-clause `retain` folded into one `FxHashSet`
  pass that skips an already-deleted id so no duplicate `drat_delete`/`lrat_delete` is emitted
  (`oxiz-sat/tests/pop_proof_deletion_regressions.rs`: 554 ids, 0 duplicates); and by a parallel
  `assertion_trivially_unsat: Vec<bool>` snapshot stack (`oxiz-sat/src/solver/mod.rs:386`). Tests:
  `oxiz-solver/tests/bv_scope_rollback_pushpop.rs` — 4 reproducers, 13 witnesses, the four
  `minimal_*` cases, witness `wrong_005`, a 200-trial differential; **all 18 fail pre-fix**.)**

### Still open

- [ ] **U-Z1 — SMT-level proof generation is not wired through the `Context` path.**
  `produce-proofs` → `Proof` reconstruction → Alethe output, with the `ProofStep` construction
  points inside the solver. The visible symptom is U-Z16 below; this is the work behind it.
  Evidence: cargo-formal blueprint §14.1 item 1.
- [x] **#P2b-40 (2026-09-16) — a function whose *return* sort is uninterpreted prints as a constant, so its model
  falsifies its own assertion.** `(declare-fun g (U) U)` with `(assert (distinct (g p) (g q)))` answered `sat` and
  published `(define-fun g ((x!0 U)) U @uc_U_0)` beside `p = @uc_U_0`, `q = @uc_U_1` — one constant function, so the
  printed `g` made `(g p) = (g q)`. `build_class_values` minted `@uc_S_n` only for *declared constants*, so an
  `Apply` result class got no value and the interpretation collapsed to `default_value(ret_sort)`. Pre-existing,
  byte-identical on `c4b04b7`. — **(fixed 2026-09-18 under `#P2b-41` / decision (11): `build_class_values` walks the
  remaining EUF classes of each uninterpreted sort after the declared constants, numbered from the declared count
  upward so `p` and `q` keep `@uc_U_0`/`@uc_U_1`. Guarded by
  `round4_pass2_recheck_pins::an_uninterpreted_return_sort_prints_a_distinct_witness_per_class`, mutation-verified.
  Amended 2026-09-19 (`#P2b-45`): those witnesses are no longer *spellable* — the parser refuses a user symbol
  beginning with `@` or `.`, which SMT-LIB 2.6 §3.1 reserves for solver use.)**
- [ ] **U-Z3 — HORN has no script-path dispatch, and the Spacer parser has no BV/Array sorts.**
  Blueprint §14.1 item 3.
- [ ] **U-Z4 — QF_FP has no decision procedure (bit-blasting).** Blueprint §14.1 item 4.
- [ ] **U-Z5 — the `oxiz` CLI does not set its exit code from the verdict and has no `--seed`.**
  Blueprint §14.1 item 5.
- [ ] **U-Z6 — `produce-models` is not handled honestly, `ResourceLimits::with_timeout` is not
  wired, and `max_memory_mb` is Linux-only.** Blueprint §14.1 item 6. Partly adjacent to U-Z12:
  the *option* path is now bounded, the `ResourceLimits` path is not — see `#P2b-8`.
- [ ] **U-Z7 — public enums and config structs are not `#[non_exhaustive]`,** so every 0.3.x
  adds a breaking change for downstream `match` and struct-literal code. Blueprint §14.1 item 7;
  the 0.3.3 CHANGELOG's own "Breaking changes" section is the evidence.
- [ ] **U-Z8 — `REFINEMENT_TIME_CEILING_MS` makes a verdict depend on wall-clock time**; make it
  configurable and deterministic. Blueprint §14.1 item 8.
- [ ] **U-Z9 — LRAT for a bit-blasted problem cannot be obtained through `Context`.** The
  certificate exists at the `oxiz-sat` layer; there is no route from the SMT API to it, which is
  why cargo-formal's only machine-checkable evidence for a QF_BV `unsat` today is a re-run.
  Blueprint §14.1 item 9.
- [ ] **U-Z15 — `(get-info :all-statistics)` returns non-zero but meaningless numbers.**
  `:decisions` / `:restarts` / `:learned-clauses` are always `0` and `:conflicts` reads like `2`
  for a 2.8 s solve. Two mechanisms are known: `Statistics::decisions`
  (`oxiz-solver/src/solver/types.rs`) is never incremented anywhere in `oxiz-solver`, and on a
  QF_BV goal the outer counters stay at `0` because the work happens in the embedded solver —
  visible only through `Solver::bv_conflicts_spent()`. Evidence: `p2-oxiz-capabilities.md` §5.4,
  cargo-formal `p2b/w0/I0-b.md` §1.2, `p2b/w2/W2-a.md` §9.
- [ ] **U-Z16 — `(set-option :produce-proofs true)` is accepted and `(get-proof)` then always
  errors.** The verdict itself is correct; only the certificate is missing. This is **the one
  remaining disagreement of the 22 conformance fixtures**, measured on the fixed tree:
  `u06_proof_not_enabled` `expected=unsat`, `actual=error`, `(error "Proof generation not enabled.
  Set :produce-proofs to true")`; that fixture carries `upstream: U-Z16`, and the fix is U-Z1.
- [ ] **#P2b-1 — `BvSolver::assert_ule` memoises nothing, which the U-Z10 rollback makes more
  expensive.** `oxiz-theories/src/bv/solver.rs:411` allocates a fresh comparison variable and
  re-runs `encode_ult_result` on *every* call, unlike `assert_ult` (`:370`), which memoises in
  `ult_cache`. After the U-Z10 fix a backjump rebuilds circuits above the popped level, so
  repeated `bvule` assertions on the same operand pair recur more often and each adds a fresh
  `O(width)` circuit. Fix: give `assert_ule` the same memo (key `ComparisonKey { a: b, b: a }`,
  the pair it actually encodes) and journal it identically in `ult_cache_journal`. Performance
  only — evidence: cargo-formal Phase 2b, `p2b/w0/I0-a.md` §2.2.
- [ ] **#P2b-2 — keep BV *definitional* clauses at the SAT solver's base level and scope only
  the *assertion* units.** The U-Z10 fix retracts circuit nodes on `pop()`
  (`oxiz-theories/src/bv/solver.rs:1728`), so every circuit above a popped level is rebuilt with
  fresh SAT variables and the old ids leak (memory only). The standard shape keeps definitional
  clauses permanent at assertion level 0 and scopes only the unit assertions; that needs an
  `oxiz-sat` API for "add this clause at level 0 regardless of the open push depth" (`push` /
  `pop` bracket `assertion_clause_ids`) plus a split of `BvSolver`'s encoders into definitional
  and assertional halves. Much larger change; recorded — evidence: `p2b/w0/I0-a.md` §2.2.
- [ ] **#P2b-3 — an in-place polarity flip in a bit-vector problem may fabricate `unsat`
  (unverified, and *not* refuted).** `oxiz-solver/src/solver/theory_manager.rs:1420`: the
  rebuild-on-flip path is guarded `&& self.bv_terms.is_empty()`, so in a bit-vector problem a
  SAT-core in-place polarity flip falls through and the *old* polarity's assertion stays live in
  the circuit as a unit clause; a spurious conflict from it would be handed back as genuine. **A
  focused hunt found no witness**: 0 flip events in the guarded arm across 20,000 differential
  fuzz trials (40,000 solves), ~450 probe cases, 14 hand-built chains and 8 runs of the one LIA
  family known to trigger the flip. Not confirmed that the flip fires at all on 0.3.4 — the non-BV
  arm was never instrumented; instrument it first — evidence: cargo-formal `p2b/w0/I0-a.md` §4.2,
  `p2b/w1/W1-a.md` §5. **Update 2026-09-14**: the false-`unsat` family this was raised to explain
  is root-caused to `#P2b-19`, so the flip explains nothing measured.
- [x] **#P2b-4 (closed 2026-09-14) — clauses injected during `solve()` escape `pop`.** Raised as
  "clauses injected *outside* the learned list escape both `forget_learned_since` and `pop`";
  confirmed with the polarity inverted — the escapees were **in** the learned list: `#P2b-19`.
  — **(closed in 0.3.4 by the `#P2b-19` fix, after enumerating every `ClauseDatabase::add*` site
  in `oxiz-sat/src` reachable from a live `solve()`: `add_clause.rs`, `propagate.rs`, `probe.rs`,
  `mod.rs` and now `learn.rs`'s three arms all register in `assertion_clause_ids`; the `bve.rs` /
  `els.rs` / `preprocessing_core.rs` / `asymmetric_branching.rs` sites are gated off while an
  assertion level is open and off by default; `check_subsumption` deletes rather than adds. The one
  live site in **neither** list became `#P2b-21`.)**
- [x] **#P2b-21 (2026-09-14) — `add_theory_reason_clause` is registered in neither the learned
  list nor the assertion level (argued sound, untested).** `oxiz-sat/src/solver/learn.rs:292`
  installed a theory propagation's justifying clause via `clauses.add_learned` alone, so it
  survived `forget_learned_since` and `pop` — believed sound as a theory tautology, but only as
  strong as the theory's own reasons. — **(fixed in 0.3.4 by the cheaper remedy: it now calls
  `register_learned_at_assertion_level`, still deliberately **not** in `learned_clause_ids`. Tests:
  `oxiz-sat/tests/theory_reason_clause_scope_regression.rs` — **all three fail pre-fix**: after the
  `pop`, `num_learned_clauses()` is 1 not 0, and the satisfiable `(¬p) ∧ (q)` comes back `Unsat`.)**
- [x] **#P2b-22 (2026-09-14) — the model gate answers `Undetermined` for an assertion whose
  variables the encoder folded away, instead of refuting it structurally.** `mk_eq(b, b)` folds to
  `true`, so `(assert (distinct b b))` never reaches the bit-blaster, `b` has no model value, and
  `model_refutes_assertions` approved a wrong `sat` — why `#P2b-20` had no second line of defence.
  — **(fixed in 0.3.4 by a structural pre-pass in `Solver::open_in_model`
  (`oxiz-solver/src/solver/model_eval.rs`): `Eq(x, x)` is `true`, `Distinct(…)` naming one operand
  twice is `false` for any sort (`has_repeated_operand`). Measured with the `#P2b-20` fix reverted:
  `(assert (distinct x x)) (push 1) (pop 1) (check-sat)` `sat` with the pre-pass off, `unknown` with
  it on, for `(_ BitVec 8)`, `Int` and `Bool`. Tests: five unit tests in
  `oxiz-solver/src/solver/model_eval/tests.rs` (three fail pre-fix).)**
- [ ] **#P2b-23 (2026-09-14) — incremental LRAT cannot produce a checkable proof across a
  `push`/`pop`.** Two independent mechanisms, each deliberate alone, leave no script shape where a
  `pop` emits deletion records *and* the trace concludes with a checkable empty clause. (1)
  `Solver::lrat_emit_empty_from` (`oxiz-sat/src/solver/lrat_trace.rs`) does `self.lrat.take()` the
  instant it concludes an `Unsat`, so a trace whose *first* verdict is `Unsat` records nothing any
  later `pop` does. (2) `crate::proof::LratWriter` hands out original and derived clause ids from
  one monotone `next_id` (`reserve_original_id` and `add_clause` share it), but
  `oxiz_proof::lrat_check::check_lrat_proof` numbers the original formula `1..=n` in the order it
  is given, so original clauses added *after* learning began get ids the checker cannot reproduce
  and their hints dangle. Measured on the fixed tree by
  `oxiz-sat/tests/pop_proof_deletion_regressions.rs`: `LratCheckReport { verified: false,
  additions_checked: 187, deletions_applied: 554, failure: Some("line 742: addition of clause 577
  failed to verify: hint 575 does not reference a currently active clause (never added, or deleted
  before this point)") }` — id 575 is the first original clause added after the `pop`. **The `pop`
  path is not the cause**; the test pins the rejection so a future fix fails loudly (cf. `U-Z9`).
- [x] **#P2b-24 (2026-09-15) — an `ite` selector outside the bit-blaster's fragment turned into
  free bits (a wrong `sat`), and a live selector pinned after the last check was never re-checked.**
  `BvSolver::bv_ite` could not encode a selector the Boolean encoder did not cover (`xor`, `=>`,
  Bool `ite`, Bool `=`, `distinct` over Bool/BV, an opaque `Apply` or `select`), and answered with
  `new_bv(root, w)` — unconstrained bits for a `bvsub`/`ite` whose semantics are known. A second
  atom sharing the sub-term blasted it for real, the two disagreed, and `debug_verify_bv_circuits`
  panicked; crates.io 0.3.3 answers `sat`. A sibling gap: a Boolean pinned into a live selector
  *after* the last check never triggered one. — **(fixed in 0.3.4: `encode_bool_node`
  and `bit_blast_cond_operands` cover the whole Boolean fragment
  (`oxiz-theories/src/bv/solver/bool_node.rs`); the encoder abstracts an opaque leaf *at the leaf*
  and a remaining failure sets `bv_atom_unmodelled`; `BvSolver::bv_ite` is `#[must_use] -> bool`;
  `assert_bool_value` reports a pin on a live node and `final_check` runs one deferred
  `bv_check_after_pin`. Measured with `bv_ite_selfcheck_fuzz.rs`: 10 `unknown` in the 100-trial
  bounded slice before, 0 after; campaign figures in CHANGELOG 0.3.4. `c13` is cargo-formal's
  `u11_ite_selector_free_bits` (`upstream: U-Z18`).)**
- [x] **#P2b-25 (2026-09-15) — a theory conflict explanation omitted the outer Booleans pinned into
  the circuit: a false proof.** `on_assignment` mirrors every outer assignment into the bit-blaster
  through `BvSolver::assert_bool_value` (a unit clause on the selector's boolean node); the embedded
  SAT solver resolves against it like against an `assert_eq`, but `collect_conflict_terms` returned
  only `assertion_guard_terms`, so an `Unsat` resting on a pin came back as if it rested on the
  constraints alone and the CDCL(T) core learned a clause the theory never derived. Found by the
  same fuzz as a wrong `unsat` at width 63 (`FALSE_PROOF_MIN` in `bv_ite_selfcheck_regressions.rs`;
  witness `v0 = 4, v1 = 0x20cb882730550eec, v2 = 1, v3 = 7, p0 = true`); the un-minimised script
  reproduces on crates.io 0.3.3 (`unsat` in 3.8 s) and is cargo-formal's
  `u12_pinned_selector_conflict_explanation` (`c14` here, `upstream: U-Z19`; its comment lists the
  Bool-sorted definitions as `t5`, `t12`, `t25`, `t26` — `t10` is `(_ BitVec 32)`). Confirmed by
  measurement first: blaming the pinned atoms alone turns the verdict to `sat`. An LRAT check of
  the Boolean skeleton cannot catch this — the theory lemma is an axiom to it. — **(fixed in 0.3.4:
  `BvSolver::pinned_terms` (`bool_node.rs`), recorded by `pin_bool_var`, snapshotted by `push`,
  truncated by `pop`, cleared by `reset`, named by every explanation. Tests:
  `oxiz-theories/tests/bv_selector_fragment_and_pins.rs` and the `p2b25_*` script tests.)**
- [x] **#P2b-27 (2026-09-15) — a Boolean that occurs only as an `ite` selector was published with
  the sort default, and the model gate could not see it: the root cause of the pre-fix campaign's
  wrong `sat`.** Such a Boolean has no outer clause, the SAT core never assigns it, `build_model`
  recorded nothing, `(get-value)`/`(get-model)` printed `false`, and every assertion above it
  evaluated `Undetermined` to `model_refutes_assertions`, which skips what it cannot evaluate.
  `(= x (ite p #x01 #x02)) ∧ (= x #x01)` answered `sat` with `p = false` on 0.3.3, on the
  `#P2b-24`/`#P2b-25` tree, and in 19 distinct trials of the close-out's fresh-seed fuzz run
  (seeds 1000–1035, 280 trials). — **(fixed in 0.3.4: `build_model` publishes
  `BvSolver::bool_value` for every Bool variable with a circuit node the core never assigned, and
  `check_core` answers `unknown` instead of `sat` when an assertion is `Undetermined` because a
  Bool variable has no model entry (`Solver::model_leaves_a_boolean_undetermined`). Tests:
  `p2b27_*` in `bv_ite_selfcheck_regressions.rs`, `bv_ite_adversarial_probe.rs`, three
  `model_eval` unit tests.)**
- [x] **#P2b-28 (2026-09-15) — a wide comparison against the top of the `i64` range panicked in a
  debug build and lost its verdict in release.** `(assert (bvult #x7fffffffffffffff v))` alone:
  `bvult`/`bvule` were *also* parsed into the linear arithmetic solver as a bounded-integer
  relaxation held in `Rational64`; `assert_lt` rewrites `x < k` into `x ≤ k − 1` and `assert_le`
  negates the constant, so `−i64::MAX − 1` overflowed — `attempt to negate with overflow` in debug
  (11 distinct trials of the fresh-seed fuzz run, all width 64; `bvugt`/`bvuge` the `assert_gt`
  sibling), `unknown` for a satisfiable one-liner in release, on 0.3.3 and this tree alike.
  — **(fixed in 0.3.4 by retiring the mirror: the `BvUlt`/`BvUle` arms of `encode.rs` no longer
  call `parse_arith_comparison`, `track_theory_vars` no longer interns bit-vector variables into
  the `ArithSolver`, and `build_model` reads bit-vector values from the circuit only — the circuit
  decided every comparison exactly already, so the copy could only add `i64` hazards. Measured
  before/after on the QF_BV suites: with the mirror restored on this same tree the bit-vector suites (189 tests across 12 binaries) differ only in the three `p2b28_*` tests, which fail there by the debug panic and pass with it retired; the other 186 pass both ways, and the 15 `bench/z3_parity/benchmarks/qf_bv` scripts answer within microseconds both ways (3.3 ms → 0.6 ms on the first, noise on the rest), so retiring the mirror lost no verdict and no measurable time; the release-calibrated budget test `bv_wide_soundness::wide_sub_of_add_two_vars_is_identity_above_64_bits` took > 60 s under the load of three concurrent campaigns in one run and hit nextest's 180 s cap in the other, in both directions of the toggle — the load, not the mirror. Tests: `p2b28_*` in
  `bv_ite_selfcheck_regressions.rs` (all six comparisons, both operand orders, against `#x7fff…`,
  `#x8000…`, `#xffff…` at width 64 and their width-63 counterparts, each witness re-evaluated),
  `bv_ite_adversarial_probe.rs`; fixture `c16_i64_max_bound_wide_comparison` written for
  cargo-formal beside `c13`/`c14`.)**
- [x] **#P2b-29 (2026-09-15) — congruence never reached an opaque leaf under a bit-vector operation, and a
  circuit-entailed equality never reached congruence.** `(= a b) ∧ (distinct (bvadd (f a) #x01) (bvadd (f b) #x01))`
  answered `sat` on 0.3.3, HEAD and the `#P2b-24` tree: the encoder gave `(f a)` and `(f b)` two unrelated free
  bit-vectors and nothing carried EUF's `f(a) = f(b)` into the circuit — likewise under `bvnot`, inside two `ite`
  selectors, and with array `select`s under `(= i j)`. The reverse direction was open too: `(= (bvadd x #x01) (bvadd
  y #x01)) ∧ (distinct (g x) (g y))` answered `sat` because nothing told EUF the circuit forces `x = y`. — **(fixed
  in 0.3.4
  by a bidirectional exchange in `final_check` (`TheoryManager::combine_bv_with_euf`): opaque leaves are journalled by
  the bit-blaster, interned into congruence closure, and two leaves EUF holds equal get their bit-equality asserted
  under EUF's explanation; the other way, the circuit's model is read as a partition of the application arguments, EUF
  is asked in a scratch scope whether it accepts it, and a refusal becomes a *lemma* entailed by the atoms EUF named,
  asserted into the circuit (`BvSolver::assert_any`) and re-checked until EUF accepts a partition or the circuit
  refutes the lemmas; every crossing carries an explanation in `DerivedReasons`. Two consequences fixed with it: the
  `resync_theory_state` backstop is gated on the bit-blaster holding no circuit, and `build_model` publishes the
  circuit's value for a bit-vector variable occurring only as an application argument. Measured by
  `oxiz-solver/tests/bv_euf_combination.rs` against an exhaustive oracle over variable values *and* function tables:
  320 bounded scripts at widths 1–2 and 150 at widths 3/4/8, 0 wrong either way, 0 bad cores; the long campaign
  1,600 scripts with 6 `unknown` and 0 wrong answers. Tests: `p2b29_*` in `bv_ite_selfcheck_regressions.rs`,
  `bv_ite_adversarial_probe.rs`, `oxiz-theories/tests/bv_selector_fragment_and_pins.rs`; fixture
  `c15_congruence_under_bv_operation`. The close-out recheck root-caused the campaign's `unknown`s to a missing
  per-pair memo in `assert_any`, fixed by `BvSolver::eq_cache`. **Still open:** (1) the loop enumerates partitions of ~15 argument terms one
  lemma at a time (`MAX_LEMMAS = 512`); the remedy is to make the argument equalities atoms of the *outer* search
  (delayed theory combination / Ackermann-style splitting with CDCL learning); (2) the pigeonhole shape `(distinct (g
  x0) … (g x8))` at width 3 is `unsat` in 62–71 s in the outer CDCL(T) search; (3) 16 width-32 variables chained by
  `bvadd`/`bvsub`/`bvxor` take 147 s pure QF_BV and the 32-variable chain > 300 s — the bit-blaster's miter hardness
  (`#P2b-14`), not the exchange.)**
- [x] **#P2b-32 (2026-09-15) — array read-over-write was never instantiated for a `select` nested under a bit-vector
  or arithmetic operator: a wrong `sat`, pre-existing on 0.3.3.** `(distinct (bvadd (select (store arr i #x05) i)
  #x01) #x06)` answered `sat` on 0.3.3, HEAD and the `#P2b-24`–`#P2b-29` tree — likewise under `bvnot`, `concat` and
  `bvult` and in the `QF_AUFLIA` twins — while the same read as a *direct* atom operand was decided.
  `collect_array_structure` descended through a hand-written child list with `_ => Vec::new()`. **Scope**: this is
  the instantiator's *walk*; a read occurring only as an application's ARGUMENT is `#P2b-33`. — **(fixed in 0.3.4: the walk delegates to `term_walk::collect_structural_children` for every non-binder
  kind (`ground_children`); the model gate reads a `select` over a `store` as read-over-write under
  `SelectSemantics::ReadOverWrite`, while the instantiator keeps the published-leaf reading — one that evaluated a
  read by the axiom itself would find every instance satisfied and assert nothing. All 44 `f3arr` scripts answer as
  the theory says.
  Tests: `p2b32_*` in `bv_ite_selfcheck_regressions.rs`, `bv_ite_adversarial_probe.rs`, four `model_eval` and four
  `array_axioms` unit tests; fixture `c17_nested_select_read_over_write`.)**
- [x] **#P2b-33 (2026-09-15) — the array-axiom refinement loop never ran at all for a formula whose only `select` is
  an application's ARGUMENT: a wrong `sat`, pre-existing on 0.3.3.** `(distinct (f (select (store arr i v) i)) (f v))`
  answered `sat` on 0.3.3, on the `#P2b-32` tree and on every tree before it, in QF_AUF, QF_AUFBV and QF_AUFLIA alike
  — likewise the RoW-2 sibling, a binary `g`, a nested `(f (f sel))` and the read under `bvadd` *inside* the
  application — while the same read as a direct atom operand was decided. `Solver::has_array_ops` guards
  `instantiate_array_axioms`, and it was
  raised only by `track_theory_vars` and the encoder's `Select`/`Store` arm, and `track_theory_vars` deliberately
  does not descend into an application's arguments, so the flag stayed `false` and the read stayed a free leaf that
  the model gate vouched for. Found by the close-out review of `#P2b-32`. — **(fixed in 0.3.4: the guard is
  computed by the instantiator's own exhaustive walk (`array_axioms::Solver::mark_array_ops`, from `Solver::encode`)
  with the same `ground_children` binder exclusion, so guard and consumer agree on what "mentions an array" means;
  and EUF interns `store` as a ternary application of `TheoryManager::STORE_FUNC_ID` so congruence joins the
  purified and unpurified spellings. Measured: 45 array benchmarks keep every verdict at 1,250 ms; all 106 scripts
  of the review's battery answer as the theory says. Tests: seven `p2b33_*` in `array_axiom_instantiation.rs`, the
  inverted probe in `bv_ite_adversarial_probe.rs`, and `array_uf_combination.rs` — reverting `mark_array_ops` gives
  32 wrong `sat`. Fixture `c18_read_over_write_under_application` (`upstream: U-Z23`).)**
  — **(amended by `#P2b-37`: `mark_array_ops` fires on any array-*sorted* term, not only on a `select`/`store`
  application, and the two callers of `encode_depth(.., 0)` that bypassed `Solver::encode` —
  `int_case_split::assert_value_disjunction` and `encode::finite_map_ite` — now go through it.)**
- [x] **#P2b-34 (2026-09-15) — published models for array reads were wrong while the verdict was right, and
  `(get-model)` printed no function interpretations: pre-existing on 0.3.3.** `(= (bvadd (select arr i) #x01) #x06)`
  was answered `sat`, then printed `i = #x00` beside `arr = ((as const …) #x00)` — a model in which the assertion
  reads `0 + 1 = 6`. An array *always* printed the constant array of its sort default, contradicting its own `select`
  entries, and `(get-model)` omitted every declared function: `build_model`'s circuit-publication loop filtered
  `BvSolver::circuit_terms()` to `TermKind::Var`, so an opaque `Select`/`Apply` leaf the circuit had valued never
  reached the model. Every verdict in the family was correct. — **(fixed in 0.3.4: the circuit filter admits every
  opaque *leaf*; `model_builder::opaque_leaves` publishes each `select` the ground assertions and the asserted array
  lemmas mention, with the read's *index*; `context::model_fmt::array_model` renders an array as the `store` chain of
  its published reads over the constant default and prints a total `define-fun` for every declared *uninterpreted*
  function. Tests: six `p2b34_*` in `model_output_and_options.rs`.)** — **(amended in 0.3.4 by
  `#P2b-37`: the 0.3.4 rendering was a *regression* in one place and incomplete in three. The regression:
  `(= (f a) b)` with `(distinct a b)` printed `b = @uc_U_1` beside `f = @uc_U_0`, because `get_func_interp_raw` had
  its own class walk and no `@uc_S_n` synthesis. Fixed by one canonical class → value map per query
  (`context/model_fmt/class_values.rs`) that the constants, the `define-fun` interpretations, the array chains and
  `(get-value)` all read, keyed by the *rendered* argument tuple and guarded by a `debug_assert!`; plus three
  completions — an array rendered per EUF class, one described only from the outside inheriting the class of a
  `store` it is the base of, and two reads at model-equal indices forced to agree. Measured: the 217-script
  corpus-wide model check 4 → **0**, campaign A 234 → 12 of 6,000, campaign B 255 → 85. Tests: eleven in
  `model_one_reading.rs`.)** — **(amended 2026-09-19 by `#P2b-45`: the array renderer also names a position of an
  uninterpreted index sort, evaluates an `(as const)` default, and publishes the shortest faithful chain.)**
- [x] **#P2b-35 (2026-09-15) — the `(get-value)` residue left by `#P2b-26`'s half-fix.** Six shapes still echoed
  their body on 0.3.3 and on the `#P2b-26` tree: an `Int`-indexed read-over-write never folded while the bit-vector
  twin did; a strict comparison at its boundary and `distinct` over two assigned integers echoed although `(<= x 4)`
  answered `true`; a term mixing a defaulted constant with an assigned one printed half-substituted; real division
  printed as `div`; and an application absent from the assertions echoed. — **(fixed in 0.3.4: under
  `LeafSource::Model` the printed model *is* the model, so `combine_eq`, `cmp_strict`, `Op::Distinct` and the
  read-over-write index comparison fold exactly there while the gate (`LeafSource::Tableau`) keeps every softening;
  `(get-value)` evaluates against the model *completed* with `(get-model)`'s sort defaults; a `Real`-sorted `Div`
  folds and prints as `/`. Tests: six `p2b35_*` in `model_output_and_options.rs`.)** — **(amended by `#P2b-37`: the
  "answer `Undetermined` rather than guess an else-value" decision is retired, because `(get-model)` prints each
  declared function as a *total* function whose innermost `ite` branch is that else-value.
  `Context::applied_interp_value` evaluates the argument tuple exactly as the interpretation printer does. Tests:
  three `get_value_*` in `model_one_reading.rs`.)**
- [x] **#P2b-36 (2026-09-15) — a read of the SMT-LIB array constant `((as const (Array D R)) d)` was an opaque free
  leaf: a wrong `sat`, pre-existing on 0.3.3.** `(= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00) #x00)
  #x05)` answered `sat` on 0.3.3 and on every tree before this one, as did seven more shapes of the review's ten.
  The array constant has no term kind of its own: the parser turns the qualified identifier into an ordinary
  uninterpreted `Apply`, so no array axiom related the read to the default and the model gate — reading the same
  free leaf — vouched for it. — **(fixed in 0.3.4:
  `array_axioms::build_const_array_reads` instantiates the missing axiom — an array constant reads back its default at
  every index — as one unconditional ground equality per collected read, composing with the two families around it (a
  miss down a `store` chain is reduced by RoW-2 to a read *of* the constant; `arr = ((as const …) d)` is carried
  across by select congruence). `Solver::store_chain` gives the model evaluator the same reading. Tests: five
  `p2b36_*` in `array_axiom_instantiation.rs`, two in `model_output_and_options.rs`, the inverted probe in
  `bv_ite_adversarial_probe.rs`, and an `ArrTerm::Const` in `array_uf_combination.rs` — reverting the axiom gives 12
  wrong `sat` of 72. Fixture `c19_array_constant_read_default` (`upstream: U-Z24`).)** — **(amended by `#P2b-37`:
  the parser interns an array constant under `oxiz_core::smtlib::CONST_ARRAY_FUNC`, whose backslash no SMT-LIB
  symbol form can produce, so the shadowing bookkeeping — `const_array_symbol_shadowed`, `shadow_const_array_symbol`
  and the `Context::push_fun_decl` hook — is deleted and both printers render the application back from the node's
  own sort. The **open remainder** is closed by `#P2b-37`; the *indirect* half by `#P2b-41`.)**
- [x] **#P2b-37 (2026-09-16) — array extensionality: `distinct` over arrays recorded no pair, a store's own index was
  never read, and two arrays shared with the uninterpreted fragment were never compared. Four wrong-`sat` families,
  pre-existing on 0.3.3 and on the 0.3.4 base.** `(distinct arr brr)` with every index of a two-element index sort
  pinned equal answered `sat` (`r3/ext/e10`) — the shape needs neither an array constant nor a `store`, so the
  store-only and constant-only repairs before it left it standing. `collect_array_structure` recorded an array-sorted
  (dis)equality from `TermKind::Eq` alone. Three more holes rode with it: a store's own-index read is never a
  *collected* read unless the script spells it, so `(= (store ((as const A) #b1) i #b0) ((as const A) #b1))` had no
  read at all; the pair's own witness index never reached select congruence; and `(distinct (f arr) (f brr))` with
  every index pinned equal contains no array atom to fire on. Found by the round-3 adversarial recheck. — **(fixed in
  0.3.4, `oxiz-solver/src/solver/array_axioms.rs`: (a) every unordered pair of an `n`-ary `distinct` over array-sorted
  operands is an extensionality pair — `distinct` is pairwise, so five arrays over `(Array (_ BitVec 1) (_ BitVec 1))`
  are unsat by cardinality and four are sat; (b) `register_store_own_index_reads` registers `select(store(b,i,v), i)`
  as a read of every collected store and `i` as one of its read indices; (c) select congruence is instantiated at each
  pair's own witness index; (d) the ext rule for shared array terms (de Moura & Bjørner, FMCAD 2009) instantiates the
  witness lemma for a pair of *foreign* array terms — an uninterpreted application's argument, a value stored into
  another array, an array-sorted `ite` branch — whose EUF classes differ and which the candidate model does not
  already separate, one pair per round, only when the three syntactic families added nothing; (e) arrays of arrays
  fall out of the fixpoint. Three rules beyond that list were needed and are recorded as such: `mark_array_ops` fires
  on any array-*sorted* term; a cardinality-guarded **off-chain Skolem index** is minted per pair whose two store
  chains are laid over different base arrays, constrained different from every write on either chain and admitted only
  where the index sort provably has more elements than the chains have writes (`index_sort_lower_bound`, which answers
  *nothing* for an uninterpreted, datatype, floating-point or parametric sort, where minting one would make a
  satisfiable formula `unsat`); and the array constant is excluded from `purify_numeric_uf_args`. Model side: see the
  `#P2b-34` amendment. **Measured** against the round-3 recheck's evaluator, corrected first (its `ArrayVal.__eq__`
  compared `(default, overrides)` structurally, mis-scoring 17 of 6,000): campaign A 15 → **0** wrong `sat`, campaign
  B 4 → **0**, the 103-script battery 99/103 → **103/103**, 217-benchmark sweep verdict-identical. Tests: 25 in
  `array_extensionality_reserved.rs` plus `array_ext_shapes_bounded`. **Mutation counts** — re-measured under
  `#P2b-39`: reverting (a) → 61 wrong `sat` and 129 falsifying models; (d) → 1 wrong `sat`, 4 falsifying models, 92
  panics; the off-chain phase → 15 wrong `sat`. **Corrected 2026-09-18 by `#P2b-41`:** rule (b) *does* have a mutation
  witness (`round4_pass2_recheck_pins::the_store_own_index_read_is_what_decides_this_script`); (c), the witness-index
  congruence, remains genuinely unwitnessed and is labelled an absence of coverage rather than shown-redundant —
  **still true after `#P2b-45`, which did not mutate it either**. Fixtures `c20_array_extensionality_distinct`
  (`upstream: U-Z25`), `c21_store_equals_constant_array` (`upstream: U-Z26`) and `c22_array_constant_indirection`
  (`upstream: U-Z27`), all `sat` on crates.io 0.3.3 and on the 0.3.4 base and `unsat` here; carried in cargo-formal as
  `u18`/`u19`/`u20`, where the live set scores **31/32 agree** with `u06` (`U-Z16`, proof generation) the only
  disagreement.)** **Open residue:** 1 published model in 480 of the bounded campaign still falsifies its own script;
  the 2,350-script campaign of `#P2b-45` is at **0**.
- [x] **#P2b-39 (2026-09-16) — the round-4 adversarial recheck: a spellable Skolem name (a wrong `unsat` in six
  lines), a non-termination regression on `n`-ary array `distinct`, two published models that falsify their own
  script, and the `(get-value)` residue.** All six findings were against the `#P2b-37` tree; the first is the worst
  class of defect this project tracks. **(1) Wrong `unsat`.** The off-chain Skolem index was named
  `!oxiz!off!{lo}!{hi}`, and SMT-LIB 2.6 §3.1 admits `!` in a *simple* symbol, so a script that declared the name
  took the solver's Skolem constant as its own and inherited the `d != i_k` constraints the rule asserts *about that
  symbol*: six lines answered `unsat` where 0.3.3 and the base answer `sat`. **(2) Non-termination.** Four generated
  scripts the base answers in 4–23 ms ran past 60 s: every pair of an `n`-ary array `distinct` minted an off-chain
  index eagerly, each a fresh bit-vector argument term feeding the BV↔EUF partition exchange. **(3) Foreign arrays
  printed identically.** `(distinct (fa brr) (fa arr))` — five lines, no `select`, no `store`, no array equality —
  answered `sat` printing both arrays as the same constant, a model falsifying its own assertion:
  `instantiate_array_axioms` returned before phase 4 whenever the three syntactic families had nothing to collect.
  **(4)** `(select arr (bvadd k #b10))` published its entry at the value of `k`. **(5)** `select` and datatype
  selectors still echoed in `(get-value)` and a `define-fun` key came back as the inlined body. **(6)** `n`
  pairwise-distinct arrays over a sort with fewer than `n` elements answered `unknown`. —
  **(fixed in 0.3.4: (1) both prefixes moved to `oxiz_core::smtlib` and now carry a backslash,
  unspellable in both SMT-LIB symbol forms, and `reject_reserved_symbol` refuses them at the parser. (2) the
  off-chain family is one pair per refinement round, deferred behind the two cheaper phases. (3) phase 4 runs
  whenever the foreign set is non-empty, not only when the syntactic families collected something. (4) the entry is
  published at the index term's *model value*. (5) `(get-value)` folds selectors and array reads and keeps a
  `define-fun` key's source spelling. (6) the pigeonhole rule refutes `n` arrays over a sort with fewer than `n`
  elements. Tests: `round4_recheck_regressions.rs`, the recheck's own pins inverted into guards.)**
- [x] **#P2b-41 (2026-09-18) — the round-4 recheck, pass 2: the reserved-name fix was by name and not by class (wrong
  `unsat` from eight lines again), an array constant one indirection away was not refuted (wrong `sat`, 14.4 % of a
  generated corpus), the refinement budget was wall clock and decided verdicts, `(get-value)` answered a *wrong* value
  through an `ite`, and two model-quality families were systematic.** **(1)** `#P2b-39` reserved the two array
  prefixes only; `$encode-numarg!{id}`, `$encode-bool-arg!{id}`, `$encode-ite-elim!{id}`, `$lookup-result!{r}-{o}`,
  `dt.size!{id}`, `sk!{n}`, `skf!{n}` and `_no_purify_{n}` were still spellable, and the first two assert `proxy =
  arg`, so eight lines turned a satisfiable script into `unsat`. **(2)** `(= a ((as const A) #b1))` with `(= a ((as
  const A) #b0))` answered `sat` — each pair is instantiated at its *own* witness index; 52 of 361 oracle-decided
  scripts of the shape were wrong, on this tree and on the base alike. **(3) Decision (9):** the re-solve budget was
  `max(120 s, 20× spent)` of wall clock, so the same binary answered one script `sat` at 77.5 s alone and `unknown` at
  the 120 s floor with ten copies in flight. **(4)** `(get-value ((select (ite p a b) #b0)))` answered the *else*
  branch beside a `(get-model)` printing `p = true`. **(5)** an array-sorted datatype selector still echoed, and
  `(get-model)` printed a quoted symbol without its bars. **(6) Decision (12):** 153 of 300 datatype scripts and 89 of
  400 array-constant scripts published a model contradicting their own assertions. **(7) Decision (11):** `#P2b-40`. —
  **(fixed in 0.3.4: (1) the class moved behind `oxiz_core::smtlib::reserved_name(tag, suffix)`, the single
  constructor of the `\oxiz.` prefix, with `is_reserved_tag` as its read-side twin — the MBQI candidate filters were
  matching `starts_with("sk")` and would have gone silently dead — and `reject_reserved_symbol` collapsed to one
  `starts_with(RESERVED_PREFIX)` so a *new* mint site is covered the moment it calls the helper. Audited and reported
  honestly: `t{i}`, `x{level}`, `!filler!{i}`, `q{i}` and `p{i}` are all `cfg(test)`. (2)
  `build_const_array_witness_congruence` reads a constant's default at the extensionality witness of *every* array
  pair of its sort, so the two defaults meet at one index. The one-line `c1 = c2 ⇒ d1 = d2` was tried first and
  rejected: it introduces a new array-sorted equality atom, the BV↔EUF partition exchange reached its 512-round
  budget, and a script that answered `sat` in 28 ms answered `unknown` in 86 ms. (3)
  `ARRAY_REFINEMENT_RESOLVE_CONFLICTS = 50_000`, a ceiling on `SolverStats::conflicts` counted from the first array
  lemma; an explicit `:timeout` is the only wall clock left. (4) `resolve_array_branch` folds the `ite` condition
  through the model and the sort-default fallback is gated on the renderer describing the array at all. (5)
  `array_query_value` routes an array-sorted query term through the renderer `(get-model)` prints from, and one
  `oxiz_core::smtlib::format_symbol` serves both printers. (6) `datatype_class_value` builds the constructor value
  from the values the model gives the selector applications the script spells: 153/300 → 0/300, the base still 153.
  (7) `build_class_values` mints `@uc_S_n` for the remaining EUF classes of each uninterpreted sort, numbered from the
  declared count upward so `p`/`q` keep `@uc_U_0`/`@uc_U_1`. **Decision (10)** is *not* met and is reported open under
  `#P2b-38` (b). **Measured 2026-09-18** over 4,000 scored scripts: 0 wrong `sat` (from 52), 0 wrong `unsat`, 0
  panics, 0 falsifying models (from 89+153); 217-benchmark sweep 0 verdict differences. **Mutation counts:** eleven
  mutations, ten red. Tests: `round4_pass2_recheck_pins.rs`, its 20 pins inverted into guards.)**
- [x] **#P2b-45 (2026-09-19) — the round-4 recheck, pass 3: a `select` through an array-sorted `ite` was a free
  bit-vector (wrong `sat` from seven lines), the pass-2 const-array fix made a 0.5 ms `sat` never answer, three
  families of published model falsified their own script, two wall-clock gates decided verdicts, and the `@uc_S_n`
  witnesses were spellable.** **(1) Wrong `sat`, blocker.** `needs_ite_elimination` excluded `SortKind::Array`, so an
  array-sorted `ite` was never named by a fresh variable with its two defining implications, and `array_axioms`' walk
  only noted the branches as *foreign*: nothing related `select(ite(c,a,b),i)` to `select(a,i)`/`select(b,i)`. Ten of
  fifteen shapes and 48 of 114 falsifying models; identical on `c4b04b7` and crates.io 0.3.3, and missed by four
  passes because the in-tree generator had no array-sorted `ite`. **(2) Non-termination.** the eager const-array
  witness rule was cubic and `rc3/slow/m5.smt2` (six declarations, three assertions) ran 400 s with no answer; the
  conflict ceiling never fired because the loop accrued no conflicts. **(3) Machine-dependent verdicts.**
  `case_split_affordable` and `blocking_affordable` still read `Instant::elapsed()` against a 120 s ceiling with no
  user `:timeout`; a 1 ms mutation flips 7 of 217 `bench/` scripts `sat` → `unknown`. **(4)** three families of
  falsifying model: an uninterpreted index sort printed as one constant (50/50), array-sorted `ite` (48), `(as const)`
  with a variable default (16); **(5)** `@uc_U_0` is spellable, so a script could declare the model's own witness. —
  **(fixed in 0.3.4: (1) `Array` dropped from `needs_ite_elimination`, so the encoder names the `ite`; and
  `families::build_array_ite_reads` closes it from the theory side too — the array walk sees the *un-eliminated* term,
  so `select(store(ite(c,x,y),j,v),i)` reduced by read-over-write to a read of a term no rule knew, and one `store`
  was enough for the wrong `sat` to survive the encoder half. Array-sorted `ite` is now in the in-tree generator
  (`array_uf_combination/ext_shapes.rs`). **Mutation coverage, corrected 2026-09-19 (`#P2b-46`): the "71 of 480
  scripts red" figure recorded here does not reproduce and is withdrawn.** With `SortKind::Array` put back into
  `needs_ite_elimination` on an isolated copy of the final tree, the three recheck-pin files and
  `array_uf_combination` give **46 run / 46 passed / 0 failed** and 915 ground array-`ite` scripts score identically
  to the unmutated tree. Only the *theory* half is witnessed: stubbing `build_array_ite_reads` to `return` gives 1
  wrong `sat` (`ite_ite_under_store.smt2`) and exactly one red test,
  `round4_pass3_recheck_pins::a_read_through_an_array_ite_under_a_store_is_refuted`. The encoder half is recorded as
  **covered only in combination**, not as dead code: every ground quantifier-free script reaches the array-`ite`
  through `ArrayStructure::array_ites`, the path the theory half owns, which says nothing about a term the collector
  never sees (an MBQI instance, for one) — an absence of coverage under decision (7), not a proof of redundancy. (2)
  `build_const_array_witness_cell`, one (constant, pair) cell per refinement round and only for pairs the assignment
  has not decided, plus `ARRAY_REFINEMENT_LEMMA_BUDGET = 10_000` in `Statistics::array_lemma_instances` so a loop that
  only builds is bounded: m5 answers `sat` in 4 ms. (3) Both gates read `SatStats::propagations` against
  `REFINEMENT_WORK_CEILING_PROPAGATIONS = 100_000_000`; `int_case_split::REFINEMENT_TIME_CEILING_MS` is gone and
  `:timeout` is the only clock. (4) the array printer names a position of an uninterpreted index sort with the
  `@uc_S_n` witnesses, renders an `(as const)` class value from the *evaluated* default, publishes the `ite`'s
  defining variable, merges a `store` member's background chain after the reads instead of folding it into the base
  (it published the same index twice), and collapses a chain covering a finite index domain onto its majority value —
  a model a consumer cannot re-check is not much better than a wrong one. (5) the parser refuses a symbol beginning
  with `@` or `.`, which SMT-LIB 2.6 §3.1 reserves for solver use, and the tactic layer's `!ack_`, `!bb_`, `__card_*`,
  `__tot_*` and `{name}_bv` mints moved into `oxiz_core::smtlib::reserved_name` — they are asserted side conditions in
  the subgoals a caller gets back, and `x_bv` beside `x` was a collision a user could build by accident. **Measured
  2026-09-19, re-run after the last edit:** 2,350 generated scripts scored against a from-scratch total-table oracle
  with model replay — 0 wrong `sat` (from 1), 0 wrong `unsat`, **0 falsifying models (from 114)**, 0 timeouts (from
  1), 0 panics, 14 `unknown` on decided; the 15-shape array-`ite` battery 0/15 wrong (from 10/15). 217-benchmark sweep
  against `c4b04b7`, best of 3, both probes back to back: 0 verdict differences, 0 response differences, 2,437 ms vs
  2,440 ms = 0.999×. Determinism: 260 decided scripts, 0 verdict differences across a second release run, a
  six-way-loaded run and a debug build. **Mutation counts:** ten mutations, nine red — the `Array` exclusion (2
  guards; its generator-failure count is withdrawn, above), the `@`/`.` parser rule (1), the covered-domain collapse
  (3), the background merge (1),
  the enumerated-const-read gate (1), the budget re-base (1), the propagation ceiling (2), the enumeration limit (5),
  the ackermann mint (1); the eager const-witness family alone no longer reproduces its own non-termination, reported
  as an attribution gap rather than claimed. Tests: `round4_pass3_recheck_pins.rs` (every pass-3 pin inverted, plus
  the load-invariance and `:max-conflicts` guards decisions (18)/(20) ask for) and
  `oxiz-core/tests/round4_reserved_mints.rs`.)**
- [ ] **#P2b-38 — the array refinement loop is incomplete on `n`-ary and nested shapes, measured.** Campaign B (6,000
  random scripts, widths 1–64, `Int` and bit-vector) answers `unknown` on 6 formulas its exhaustive oracle decided,
  and campaign A on 166 of 5,024; the bounded extensionality campaign on 6 of 470, its long form on 57 of 5,898. All
  are budget exits, never wrong answers. Two shapes are named: an equality between two *concrete* arrays of arrays
  takes 12 s on the 0.3.4 base and longer here (`(= row0 (select matrix 0))` with both sides substituted as
  `define-fun`s), and a store chain whose writes cover the whole index domain needs the index case split the lemma
  loop does not make. The generative direction — a witness read of an array-of-arrays range is itself an array term,
  so each round can mint another pair — is bounded by `MAX_ARRAY_EXT_WITNESSES` (512), which reports through
  `Solver::array_axioms_incomplete` and so downgrades the `Sat` it would otherwise license to `Unknown`. **Three
  strands added by the round-4 recheck (`#P2b-39`).** (a) **Closed 2026-09-18 under `#P2b-45`.** *Satisfiable* array
  cardinality above the cliff was undecided: nine pairwise-distinct arrays over `(Array (_ BitVec 2) (_ BitVec 1))`
  (sixteen exist) answered `unknown` where eight answered `sat`, because the BV↔EUF partition-lemma exchange reached
  its 512-round budget (`#P2b-29`) before the nine Skolem witnesses were separated. Decision (10)'s enumeration lever
  removed the witnesses for such a sort. Release: every `n` from 2 to 16 answers `sat` (9 in 5.8 ms, 16 in 5.9 s) and
  17 answers `unsat` in 0.5 ms, where the base answers a wrong `sat` at 17. **Scope, corrected 2026-09-19 (`#P2b-46`):
  that ladder is `(Array (_ BitVec 2) (_ BitVec 1))` and nothing else** — one index bit wider the family is decided but
  costs seconds from n = 11, and at width 4 it answered `unknown` from n = 11 until `#P2b-46`. So the `n` = 2..16
  ladder stays an `#[ignore]`d cost pin scoped to that sort, and the non-`#[ignore]`d guard
  `round4_recheck_regressions::satisfiable_array_cardinality_at_the_cliff_is_decided` carries a cheap width-3 case
  beside the width-2 cliff, with width 4 in `::array_cardinality_above_the_enumeration_limit_is_decided`. (b)
  `n`-ary `distinct` over store chains costs more than the base. **Amended 2026-09-19 (`#P2b-46`); the `#P2b-45`
  figures are superseded where they conflict.** `ARRAY_INDEX_ENUMERATION_LIMIT = 8` decides a pair over a small index
  sort by enumeration instead of a Skolem witness, and phases 3a/3b/3c skip such a pair entirely; the four in-tree
  scripts of `array_distinct_timing.rs` are 0.32–1.30 ms against the base's 0.07–0.35 ms and `s30028_31` 3.29 ms
  against 0.07 ms. **Corrected cost model** (the `#P2b-45` sentence "above `ARRAY_INDEX_ENUMERATION_LIMIT` the Skolem
  path returns at its original cost" was **false** — it answered `unknown` where the base answers `sat` — and is
  withdrawn): the enumerated branch emits `C(n,2)` lemmas and `C(n,2)·|D|` bit-vector equality atoms **in one
  refinement round**, not `|D|` per pair, and the cost is not the lemmas — it is that the outer search then runs **one
  complete embedded `BvSolver::check` per bit-vector atom propagation**. Attributed by profiling and counting, release:
  twelve pairwise-distinct arrays over `(Array (_ BitVec 3) (_ BitVec 1))` = 528 atoms, **75,740 embedded checks**,
  22 s, in *one* round with 66 lemma instances and 1,444 conflicts, `combine_bv_with_euf` running exactly **once**;
  87 % of the samples are inside the embedded solver's `O(num_vars)` pre-search lucky phase, against a variable table
  that grows with the *search* (36,910 variables against 1,218 live original clauses — see (f)). **Still open, and not
  claimed:** decision (10) asked for a small constant factor and this is not that; the residue is unbounded in ratio
  (width 3, n = 20: `unknown` in 346.6 s against the base's 0.4 ms). What `#P2b-46` changed is that it *terminates*:
  `Statistics::bv_embedded_checks` and `BV_EMBEDDED_CHECK_CEILING = 250_000` end such a check with `Unknown` after a
  bounded, machine-independent amount of work, where nothing but a wall-clock `:timeout` did before. Cost-pinned by
  `round4_pass4_recheck_pins::the_index_width_three_cardinality_ladder_terminates` and `::the_store_term_cliff_terminates`.
  (c) **Closed 2026-09-18**: the budget was wall-clock, so on a frozen clock (`wasm32-unknown-unknown`, `no_std`) it
  never fired and, in the other direction, the same release binary answered one script `sat` at 77.5 s run alone and
  `unknown` at the 120 s floor with ten copies in flight. It is now `ARRAY_REFINEMENT_RESOLVE_CONFLICTS = 50_000`
  beside `ARRAY_REFINEMENT_LEMMA_BUDGET = 10_000` (`#P2b-45`), so a refinement loop that only *builds* is bounded too;
  an explicit `:timeout` is the only wall-clock limit left. (d) **Re-measured 2026-09-19 (`#P2b-45`):** 18 of the
  2,350-script campaign answer `unknown`, 14 on inputs the oracle decides (the pass-2 figure was 159 of 1,200 and is
  superseded, not re-run). Instrumented rather than guessed: **17 of the 18 reproduce and all 17** exit at
  `Context::check_sat_core`'s array honesty gate (`Solver::array_atoms_need_theory`), not at any budget — the
  pre-`#P2b-37` syntactic approximation, which vetoes a `Sat` the lazy refinement has already saturated for every
  positive store=store equality its crude `eval_read` scan cannot separate. **Reworded 2026-09-19 (`#P2b-46`): the
  "unsound-shaped in the other direction" claim previously recorded here is withdrawn as unsubstantiated.** Read at the
  source, the separation is conservative in the safe direction: `eval_read` returns `Some` only where the read is
  *forced* (it advances past a store link only when `are_different_values` holds, true only for two same-width
  `BitVecConst`/`IntConst`/`RealConst` literals with different values) and a match needs `terms_equal_simple`, so a
  separation at index `k` means both sides are forced to *different constants* there, which entails `x != y`. No
  counterexample exists in 3,470 scored scripts and none could be constructed. What **is** measured stands: the gate is
  incomplete on the `sat` side and order-dependent — two ground four-link chains denoting the same array differ only in
  link order and one answers `sat`, the other `unknown` (`rf5/replay_dedup.smt2` 6.5 ms, `rf5/replay_now.smt2` 0.77 ms),
  identically on `c4b04b7` and 0.3.3, so it is a pre-existing precision residue and not a regression. Deliberately
  **not** touched here: the gate decides `bench/` verdicts and the 0-verdict-differences sweep is binding. The strongest
  lead this round leaves open; it wants its own campaign.
- [x] **#P2b-46 (2026-09-19) — the round-4 recheck, pass 5: `n`-ary `distinct` over arrays lost its verdict above the
  enumeration limit, and below it nothing deterministic bounded the run.** The recheck's eleven findings, at the root.
  **(1) Verdict regression closed.** Eleven pairwise-distinct arrays over `(Array (_ BitVec 4) (_ BitVec 1))` — 65,536
  inhabitants, so trivially `sat` — answered `unknown`, where `c4b04b7` answers `sat` in 0.26 ms. Attributed by
  instrumentation to `bv_bridge`'s partition-lemma loop giving up at `MAX_LEMMAS = 512`: the Skolem cascade mints
  `C(11,2) = 55` fresh witness indices in one round, every one a `select` argument and so an exchange candidate, and
  512 lemmas cannot rule out the partitions of 55 candidates. The bound is 8,192; n = 11 answers `sat` in 0.91 s, 13 in
  3.03 s, 15 in 38.1 s, and index width 5 / n = 15 in 4.62 s. Guarded — not `#[ignore]`d, it is a verdict — by
  `round4_recheck_regressions::array_cardinality_above_the_enumeration_limit_is_decided`, with a kill ceiling of its
  own in `.config/nextest.toml` because it costs 113.9 s in the test profile. Mutation: back to 512, that guard is the
  one red test. **(2) The unbounded run is bounded, deterministically.** `Statistics::bv_embedded_checks` counts
  complete checks of the embedded bit-blasted solver and `theory_manager::BV_EMBEDDED_CHECK_CEILING = 250_000` ends the
  check with `Unknown` when they run out — the third currency beside `ARRAY_REFINEMENT_RESOLVE_CONFLICTS` (a loop that
  *searches*) and `ARRAY_REFINEMENT_LEMMA_BUDGET` (one that only *builds*), neither of which can see **one** round
  whose re-solve is enormous. `(distinct a0 … a19)` at index width 3 answered nothing in 900.03 s and now answers
  `unknown` after exactly 250,002 checks (346.6 s release); ten `store` terms over one base at width 3, which did not
  answer in 40 s, answer `unknown` after the same 250,002 checks (124.6 s). Calibration: `bench/` peaks at **207**
  checks per script (1,200x headroom), the width-2 cardinality ladder the cost pin requires at 36,281 (6.9x), the most
  expensive script that still decides at 76,860 (3.3x). Mutation: at a ceiling of 1, three `round4_pass2_recheck_pins`
  guards go red. **(3)** `:named` labels go through `Parser::reject_reserved_symbol` (decision (19)'s last door); other
  attribute values deliberately do not. **(4) Decision (16) swept:** `array_distinct_timing.rs` lost its four
  `(set-option :timeout 120000)` and its 45 s wall-clock `BUDGET` (the deterministic budget decides all four in 0.01
  s), and `array_uf_combination/ext_shapes.rs`'s generator lost `(set-option :timeout 1000)` — that clock did not only
  put a verdict behind the machine, it put the **tally** there, because `score` replays a model on the `sat` branch
  only, so `array_ext_shapes_bounded`'s `bad_model` bound moved with how many scripts got past the clock and could go
  red on a *faster* machine. **(5)** The two `include_str!` guards in `oxiz-core/tests/round4_reserved_mints.rs` are
  behavioural now: they drive `nla2bv` on a fully bounded integer goal and the bit-blaster on a bit-vector goal and
  require `RESERVED_PREFIX` on every `Var` the subgoals introduced that the input did not have; restoring either old
  spelling turns the matching guard red. **(6)** `array_uf_campaign` has the `.config/nextest.toml` override its
  sibling had. **(7)-(11)** are record corrections, each measured rather than inferred: the `71 of 480` mutation figure
  for the decision-(14) encoder half, the "unsound-shaped in the other direction" reading of
  `store_extensionality_conflict`, R6's scope, `#P2b-38` strand (b)'s cost model and its false "returns at its original
  cost" sentence, and the prohibited `c13`-`c21` agreement figure in `CHANGELOG.md`. **(e) Deviation, declared with its
  measurements.** The recheck asked for the enumerated extensionality family to be made lazy and model-guided, one pair
  per refinement round, as decision (15) did for the array-constant witness cell. It was implemented and measured, and
  it is **worse**, so it did not ship. It converges in `O(n)` rounds as intended (seven pairwise-distinct arrays at
  width 3: 11 rounds, 11 lemma instances, against `C(7,2) = 21` pairs) but a round is a whole re-solve, and a
  half-built family lets the search satisfy each lemma through its `a = b` arm and be refuted by the `distinct` in the
  theory, once per pair per re-solve: eight arrays at width 3 burned all 50,000 conflicts of
  `ARRAY_REFINEMENT_RESOLVE_CONFLICTS` in 61.9 s, where eager answers n = 9 in 28.6 ms. Ladder, release, 20 s cap —
  width 2, n = 7/9/11/13: eager 2.7 ms / 6.8 ms / 17.5 ms / 5.8 s against lazy 4.8 ms / >20 s / >20 s / >20 s; width 3,
  n = 5/7/9/11: eager 2.0 ms / 7.0 ms / 28.6 ms / 4.7 s against lazy 1.1 ms / 7.5 ms / >20 s / >20 s. Guarding the
  lemma with the `distinct` atom instead of a fresh `(= a b)` was tried beside it and is byte-identical in conflicts
  and propagations, so it did not ship either (decision (7)). `build_extensionality_and_congruence`'s doc carries the
  table. **(f) Left open, with the measurement, for whoever takes decision (10) next.** The embedded bit-blasted solver
  leaks SAT variables across `push`/`pop`: `assert_neq` mints one fresh variable per bit on every call, the theory
  manager calls it once per **trail assignment** of the atom rather than once per atom, and `sat.pop()` deletes the
  clauses but not the variables — 36,910 variables against 1,218 live original clauses and 3,096 literals on the
  twelve-array script. That is what makes each of the 75,740 embedded checks cost ~290 us, because the pre-search lucky
  phase is `O(num_vars)` per scan (87 % of the samples). An idempotence memo on `(pair, polarity)` was written and
  measured: it changes nothing, because `pop` runs on every backtrack and takes the memo with it. The fix is variable
  reclamation in `Solver::pop` or gate definitions installed at the root scope — both `oxiz-sat`/`oxiz-theories`
  changes with a blast radius this pass did not take on.

- [ ] **#P2b-26 — `(get-unsat-core)` re-solves every candidate subset from scratch; `(get-value …)`
  of any non-variable term printed its body.** cargo-formal's named form is ≥ 3.8× the plain script
  because `Solver::minimize_unsat_core` builds a fresh `Solver` per core member (c13: 1.8× here, 4
  names); 0.3.4 makes those solvers inherit the budgets and a *shrinking* `timeout_ms` (they started
  unbounded) — assumption-literal cores are the real fix and stay open. The `(get-value)` half is
  **fixed in 0.3.4**: `Model::eval` folds only the Boolean connectives and integer arithmetic, so a
  bit-vector operator, a comparison, an application and a `select` all echoed their body on 0.3.3
  and here — `((bvadd v #x01) (bvadd v #x01))`, `((f b) (f b))` with `f b = f a = #x07` in the
  model, `(select (store arr #x00 #x05) #x00)` unevaluated, and a `define-fun` name (its body, since
  the parser inlines it). `Context::format_get_value` now folds through `Solver::model_value_of`
  (the model gate's structural evaluator, read-over-write included, with every leaf read from the
  published model — `LeafSource::Model`; the gate keeps the tableau reading, and the first routing
  printed the tableau's stale `0` for a nonlinear goal whose model says `-2`) and, for an
  application the evaluator cannot fold, the value carried by a member of its congruence class
  (`Solver::euf_class_value`), printing the term only when neither answers. Test:
  `p2b26_get_value_folds_bit_vector_terms_comparisons_and_congruent_applications` in
  `bv_ite_selfcheck_regressions.rs`. Found with `#P2b-24`; the non-variable cases by the close-out
  recheck, and the six shapes that still echoed after this half-fix are closed as `#P2b-35`. **What
  is still open under this number is the unsat-core half only.**
- [ ] **#P2b-30 — a Bool-sorted uninterpreted application or array `select` used as an `ite`
  selector answers `unknown`.** `(assert (P a)) (assert (= x (ite (P a) #x01 #x02))) (assert (= x
  #x01))`: `encode_bool_node` has no arm for `Apply`/`Select`, so `bit_blast_cond_operands` fails
  and the atom is unmodelled — `unknown` since `#P2b-24`, where 0.3.3 answered `sat` over free bits
  (its unsat twin is `unknown` on both). Sound, and a precision regression against 0.3.3 on that
  shape. Fix: a fresh SAT variable for the Bool leaf, pinned by the outer atom's assignment through
  `assert_bool_value`, plus Bool-leaf equality sharing (`(P a) = (P b)` from `a = b`) so the leaf is
  not a free node either. Found by the `#P2b-24` close-out review.
- [ ] **#P2b-31 — a Bool-sorted uninterpreted-function argument has no two-element domain in
  EUF.** `(declare-fun P (Bool) Bool) (assert (P p)) (assert (not (P true))) (assert (not (P
  false)))` answers `sat` on 0.3.3 and here: `p` is an EUF leaf merged with neither `true` nor
  `false`, so congruence never fires. Pre-existing, outside cargo-formal's fragment; found while
  probing the `#P2b-27` gate. Fix: case-split Bool-sorted EUF leaves (`p = true ∨ p = false`) or
  intern them as their SAT literal.
- [ ] **#P2b-5 — the congruence half of the quantified model gate still ignores bit-vector-sorted
  applications.** `ArgKey::from_outcome` (`oxiz-solver/src/solver/model_eval.rs:72`) maps the new
  `EvalVal::Bv` to `None`, preserving pre-0.3.4 behaviour, so
  `quantified_model_refutes_ground_assertions` cannot catch a "not a function" witness over
  bit-vectors. Keying on the value would strengthen it; it is a separate change and wants its own
  measurement on the quantified suites — evidence: cargo-formal Phase 2b, `p2b/w1/W1-b.md` §6.4.
- [ ] **#P2b-6 — `:timeout` and all three budgets restart on every arithmetic-refinement round.**
  `Solver::check_with_arith_refinement` (`oxiz-solver/src/solver/mod.rs:786`) can call
  `check_core` up to 5 times per user `(check-sat)`, and the deadline plus the arming block live
  *inside* `check_core` (`oxiz-solver/src/solver/check_core.rs:188`, `:231-237`), so a goal that
  takes `R` rounds gets `R × timeout_ms` and `R × N` conflicts. Narrow path — only when a round
  ends `Sat` **and** `arith_defs_incomplete`, i.e. never on QF_BV — and outside the scope of the
  U-Z12 fix. Fix: hoist the deadline and the arming into `check_with_arith_refinement` and pass
  them down — evidence: cargo-formal Phase 2b, `p2b/w2/W2-a.md` §8.
- [ ] **#P2b-7 — `Solver::check_sat_only` arms nothing and can inherit an already-exceeded
  ceiling.** `oxiz-solver/src/solver/mod.rs:1114` runs `self.sat.solve()` (`:1140`) on the outer engine
  without going through `check_core`, so it never arms the budgets — and because the new outer
  ceilings are *relative* to a cumulative counter that no reset clears, a `check_sat_only` after
  a budget-exhausted `check()` sees a ceiling already exceeded and returns `Unknown` with no
  search at all. Sound (`mod.rs:1147` maps `SatResult::Unknown` to `SolverResult::Unknown`) but
  silent, and it is public API. Fix: arm from `self.config` there too, or clear the ceiling on
  the way out of `check_core` — evidence: cargo-formal Phase 2b, `p2b/w2/W2-a.md` §8.
- [ ] **#P2b-8 — `check_with_limits` reports `Ok(Unknown)`, not `Err(ResourceExhausted)`, when
  the new budgets fire.** `oxiz-solver/src/solver/mod.rs:1272` populates its `ResourceMonitor`
  from `self.statistics.conflicts` / `.decisions` (`:1297-1298`), which count theory conflicts
  and — for decisions — nothing at all, so the post-check `monitor.check()` (`:1303`) cannot see
  an outer-Boolean or bit-blasting exhaustion. Public through `oxiz/src/easy.rs`
  (`ResourceLimits`). Fix candidate: feed the monitor from `Solver::stats()` and
  `Solver::bv_conflicts_spent()` deltas taken around the check — evidence: cargo-formal Phase 2b,
  `p2b/w2/W2-a.md` §8.
- [ ] **#P2b-9 — the deadline does not bound *encoding*.** Bit-blasting a 128-bit multiplier
  happens before any `solve()` and polls nothing (`oxiz-solver/src/solver/theory_bv_encode.rs`,
  and `bv_mul` and friends in `oxiz-theories/src/bv/solver.rs`), which is why `:timeout 100`
  answers at ~105 ms of *solver* time while the check as a whole is bounded only once the
  circuit exists. `Solver::propagate_step_limit` (`oxiz-sat/src/solver/mod.rs:577`) is the
  existing second lever on the search side; the encoder has none — evidence: cargo-formal
  Phase 2b, `p2b/w2/W2-a.md` §8.
- [ ] **#P2b-10 — the theory-conflict budget stays cumulative across checks while the two new
  budgets are per check.** `Statistics::conflicts` is only cleared by
  `Solver::reset_statistics`, so after the U-Z12 work `:max-conflicts N` means three budgets of
  `N` that differ in *period*. Not changed — it is the pre-existing behaviour of the theory
  budget — but it is worth either documenting or unifying — evidence: cargo-formal Phase 2b,
  `p2b/w2/W2-a.md` §8.
- [x] **#P2b-11 — `BvSolver::notify_equality` is unreachable from the script path, so
  bit-vectors never took part in Nelson–Oppen equality exchange.**
  `oxiz-theories/src/bv/solver/combination.rs:25`: `TheoryManager` forwarded equality
  notifications only to the arithmetic solver, and `oxiz-theories/src/combination.rs` holds no
  `BvSolver` — evidence: cargo-formal `p2b/w0/I0-b.md` §1.3, `p2b/w2/W2-a.md` §8. — **(closed in
  0.3.4 by `#P2b-29`: bit-vectors now take part through `TheoryManager::combine_bv_with_euf`, which
  asserts shared equalities and lemmas with `assert_eq`/`assert_any` directly; `notify_equality`
  itself is still reachable only through direct API use.)**
- [x] **#P2b-12 — the `bv_atom_unmodelled` guard is untested.** `bv_run_check`
  (`oxiz-solver/src/solver/theory_manager/bv_bridge.rs`) downgrades a final `Sat` to `Unknown` when
  an `assert_*` reports that it modelled nothing; with the `Gt`/`Ge` arms in place no input set it
  across the 4,700-test suite, the 12,000-trial fuzz or the ~450-case probe battery, and the only
  evidence it worked was a deliberately broken tree — evidence: cargo-formal `p2b/w1/W1-a.md` §3.2
  (O5), `p2b/w1/gate1-fix.md` §2, §11.2. — **(closed in 0.3.4 by `#P2b-24`, which routes every
  encoder failure through the flag: `p2b24_selector_outside_the_fragment_is_unknown_not_sat` in
  `oxiz-solver/tests/bv_ite_selfcheck_regressions.rs` pins an `ite` whose selector is an `Int`
  comparison answering not-`sat`, and `#P2b-30` records the precision that costs.)**
- [ ] **#P2b-13 — `oxiz-sat --no-default-features` does not compile, and
  `--target wasm32-unknown-unknown` fails in `getrandom`.** 18 errors (`std` paths,
  `rustc_hash::FxHashMap`, `crate::proof`) for the first; the wasm target needs `getrandom`'s
  `wasm_js` feature and never reaches `oxiz-sat` code at all. Both **pre-existing** on the
  0.3.4 base commit and unrelated to any finding here — captured byte-identically before and
  after the Phase 2b changes, so the new ungated `Option<oxiz_time::Instant>` budget API adds no
  `no_std` burden — evidence: cargo-formal Phase 2b, `p2b/w0/I0-d.md` and `p2b/w2/W2-a.md` §5, §8.
- [ ] **#P2b-14 — no structural hashing / miter detection in the bit-blaster.** A miter of two
  *identical* circuits — cargo-formal's harness comparing `oxiarc`'s `xxhash32(bytes)` against
  `xxhash32_with_seed(bytes, 0)`, which are the same function — was not decided in 120,000 ms.
  Structural hashing (or explicit miter detection) would make an equivalence of two syntactically
  equal circuits trivial rather than a full search. Orthogonal to U-Z10 and to cargo-formal's
  version pin — evidence: cargo-formal Phase 2b, `p2b/w3/E3.md` §8 item 3, measured while
  writing `oxiarc/formal`.
