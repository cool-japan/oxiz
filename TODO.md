# OxiZ TODO

Last Updated: 2026-04-04

Reference: Z3 codebase at `../z3/` for algorithms and implementation strategies.

---

## Major Milestone Achieved: 100% Z3 Parity (v0.2.0)

**Date Achieved**: February 5, 2026
**Release Status**: Published (Feb 6, 2026)

OxiZ has achieved **100% correctness parity with Z3** across all 88 benchmark tests spanning 8 core SMT-LIB logics. This validates OxiZ as a **production-ready Pure Rust SMT solver**.

### Z3 Parity Results

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
| Post-Parity: Performance | 9 | 19 | 32% |
| Post-Parity: UX | 3 | 0 | 100% |
| Post-Parity: Debugging | 4 | 0 | 100% |
| Post-Parity: Docs | 5 | 0 | 100% |
| Post-Parity: Theories | 0 | 10 | 0% |
| Post-Parity: Advanced | 0 | 12 | 0% |
| Post-Parity: Ecosystem | 0 | 7 | 0% |
| **Total** | **87** | **48** | **64%** |

---

## Current Statistics (v0.2.0 - April 4, 2026)

- **Rust Lines of Code**: 393,292 total (312,495 code lines)
- **Rust Files**: 931
- **Unit Tests**: 6,155 passing (16 skipped, 0 failures)
- **Z3 Parity**: **100.0% (88/88)**
- **Perfect Logics**: **8/8 tested**
- **Workspace Crates**: 16 (15 Rust + 1 TypeScript)
- **todo!/unimplemented! macros**: 0 (all 15 Rust crates)
- **Clippy Warnings**: 0
- **Largest File**: 1,892 lines (all files under 2,000 lines)

---

## Beyond Z3: Key Differentiators

OxiZ is not just a Z3 port - it surpasses Z3 in critical areas:

1. **Machine-Checkable Proofs** (oxiz-proof) - DRAT, Alethe, LFSC + Coq/Lean/Isabelle exports
2. **Spacer/PDR** (oxiz-spacer) - Missing in CVC5, Yices, and most Z3 clones!
3. **WASM-First** (oxiz-wasm) - Target <2MB vs Z3's ~20MB
4. **Native Parallelism** - Rayon portfolio solving, work-stealing
5. **Memory Safety** - Pure Rust, no FFI, guaranteed safety
6. **Craig Interpolation** - McMillan, Pudlak, Huang algorithms with theory support
7. **100% Z3 Parity Validated** - Proven correctness across all core logics
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

### High Priority: Performance Optimization (Partial - 9/28 Complete)

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
  - [ ] JIT-style specialization for hot theory operations
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

- [ ] Language bindings (4 items)
  - [x] Improve Python bindings (oxiz-py enhancements) (planned 2026-04-19)
  - **Goal:** Bring `oxiz-py` to 0.2.1 quality bar: full theory test coverage, README + pyproject.toml synced to workspace version, parity matrix doc.
  - **Design:** PyO3 surface (1583 LoC, 7 modules, 721-line stub) is mature. Add 5 pytest files for theories implied by stubs but not yet tested. Sync version strings. Add `PARITY.md` table mapping z3 API → oxiz wrapper → status.
  - **Files:** `oxiz-py/tests/test_quantifiers.py` (new), `oxiz-py/tests/test_arrays.py` (new), `oxiz-py/tests/test_fp.py` (new), `oxiz-py/tests/test_strings.py` (new), `oxiz-py/tests/test_unsat_cores.py` (new), `oxiz-py/PARITY.md` (new), `oxiz-py/pyproject.toml` (version → 0.2.1), `oxiz-py/README.md` (version + test-count update); minimal `src/*.rs` patches only if a wrapper is missing.
  - **Tests:** Each pytest file has ≥3 assert cases. Run `cargo build -p oxiz-py --release` (always); `maturin develop + pytest` if toolchain available, else skip with explicit note.
  - **Risk:** maturin unavailable. Mitigation: .py and .md files land regardless; test run is skipped.
  - **Scope cap:** ≤700 LoC net-new. ≤3 new PyO3 wrappers × ≤50 LoC each if needed.
  - [ ] JavaScript/TypeScript bindings (via WASM)

- [ ] Tool integration (3 items)
  - [ ] SMT-COMP 2026 participation
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
- [ ] SMT-COMP 2026 entry preparation

### v1.0.0 (Target: Q4 2026)
**Focus: Production Release**
- [ ] Full Z3 API compatibility
- [ ] Performance at or better than Z3
- [ ] Comprehensive documentation
- [ ] Stable API guarantees
- [ ] Industry adoption ready

---

## Recent Achievements

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

1. **Performance Profiling and Optimization** (v0.3.0)
   - Profile remaining hot paths (SAT propagation, theory check, e-graph)
   - Reduce allocations further (in-place updates, allocation-free theory paths)
   - Incremental computation caching
   - Memory layout optimization guided by profiling

2. **Performance Regression Infrastructure** (v0.3.0)
   - CI/CD integration for performance tracking
   - Automated benchmark comparison vs Z3
   - Performance dashboard

3. **Extended Theory Coverage** (v0.3.0)
   - Implement quantified logic support (UFLIA, UFLRA, AUFLIA, AUFLIRA)
   - Validate combined theories (QF_AUFBV, QF_ALIA, QF_ABV)
   - Extend QF_NIA coverage and add QF_NIRA

4. **SMT-COMP 2026 Preparation** (v0.3.0)
   - Benchmark suite alignment with SMT-COMP categories
   - Competition binary builds and packaging
   - Performance tuning on competition benchmarks

5. **Ecosystem Growth**
   - Improve Python bindings
   - JavaScript/TypeScript bindings via WASM
   - Integration with verification frameworks

---

**Status**: Production Ready
**Current Version**: v0.2.0
**Tests**: 6,155 passing (16 skipped) | **LoC**: 393,292 total / 312,495 code | **Files**: 931 | **Clippy**: 0 warnings
**Next Milestone**: v0.3.0 - Performance Parity + SMT-COMP (Target: June 2026)
**Long-term Goal**: v1.0.0 - Industry-Ready SMT Solver (Target: Q4 2026)

---

## Proposed follow-ups

- **JIT-style specialization** (root TODO.md:158) — defer to v0.4.0 (oversized: requires IR + codegen layer).
- **JS/TS bindings via WASM** (root TODO.md:233) — defer until `oxiz-wasm` npm publish is authorized.
- **SMT-COMP 2026 participation** (root TODO.md:238) — gated on SMT-COMP submission portal (opens ~May 2026).
- **Symbolic execution tool integration** (root TODO.md:239) — vague; re-scope after user selects target (KLEE/angr/S2E).
- **Verification framework integration** (root TODO.md:240) — vague; re-scope after user selects target (Frama-C/CBMC/SeaHorn).
