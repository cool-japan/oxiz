# Profiling Report

| Category | Instrumented Location | Proposed Fix |
|---|---|---|
| SatPropagation | `oxiz-sat/src/solver/propagate.rs::Solver::propagate` | Reorder watch lists to keep satisfied blockers hot and shrink clause-database misses. |
| TheoryCheck | `oxiz-solver/src/combination/coordinator.rs::TheoryCoordinator::check_sat` | Memoize per-round theory results and skip unchanged theory dispatch. |
| EGraphMerge | `oxiz-core/src/ast/egraph.rs::EGraph::merge` | Switch to smaller node storage and rank-aware merge bookkeeping. |
| SimplexPivot | `oxiz-theories/src/arithmetic/simplex.rs::Simplex::pivot` | Replace full-row substitution with sparse dirty-row updates. |
| BvPropagation | `oxiz-theories/src/bv/propagator.rs::WordLevelPropagator::propagate` | Avoid cloning the constraint list on every propagation pass. |
| StringAutomata | `oxiz-theories/src/string/automata.rs::ConstraintAutomaton::accepts` | Cache DFA states for repeated prefix/suffix constrained checks. |
| ArrayExtensionality | `oxiz-theories/src/array/solver.rs::ArraySolver::check` | Index pending lemmas by array root and select index to cut quadratic scans. |
| ProofGeneration | `oxiz-proof/src/incremental.rs::ProofRecorder::record_step` | Batch arena-backed proof step storage to reduce per-step allocation. |
| Parser | `oxiz-core/src/smtlib/parser/commands.rs::Parser::parse_command` | Use a token buffer for command dispatch and reduce repeated keyword parsing. |
| CacheMiss | `oxiz-core/src/rewrite/combined.rs::CombinedRewriter::lookup_cache` | Add a small front-side cache for recent terms before the full map lookup. |

## SatPropagation

- Code path covered: Boolean unit propagation and watch processing.
- Instrumented location: `oxiz-sat/src/solver/propagate.rs::Solver::propagate`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Reorder watch lists and specialize short clauses.

## TheoryCheck

- Code path covered: theory-combination entry and theory check dispatch.
- Instrumented location: `oxiz-solver/src/combination/coordinator.rs::TheoryCoordinator::check_sat`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Cache unchanged theory results across combination rounds.

## EGraphMerge

- Code path covered: union-find based e-class merge.
- Instrumented location: `oxiz-core/src/ast/egraph.rs::EGraph::merge`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Reduce merge-time parent scanning and compact node storage.

## SimplexPivot

- Code path covered: tableau pivot and assignment refresh.
- Instrumented location: `oxiz-theories/src/arithmetic/simplex.rs::Simplex::pivot`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Incremental row updates instead of whole-tableau substitution.

## BvPropagation

- Code path covered: word-level bit-vector propagation fixpoint loop.
- Instrumented location: `oxiz-theories/src/bv/propagator.rs::WordLevelPropagator::propagate`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Remove repeated constraint cloning and reuse propagation buffers.

## StringAutomata

- Code path covered: automata-backed string acceptance checks.
- Instrumented location: `oxiz-theories/src/string/automata.rs::ConstraintAutomaton::accepts`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Precompute reusable transition summaries for repeated checks.

## ArrayExtensionality

- Code path covered: delayed read-over-write and extensionality lemma checking.
- Instrumented location: `oxiz-theories/src/array/solver.rs::ArraySolver::check`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Index selects/stores by representative array and index class.

## ProofGeneration

- Code path covered: proof step recording for SAT/SMT derivations.
- Instrumented location: `oxiz-proof/src/incremental.rs::ProofRecorder::record_step`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Use arena-backed proof step storage and cheaper batching.

## Parser

- Code path covered: top-level SMT-LIB command parsing.
- Instrumented location: `oxiz-core/src/smtlib/parser/commands.rs::Parser::parse_command`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Reduce token peeking and reuse parsed command buffers.

## CacheMiss

- Code path covered: combined rewriter cache misses.
- Instrumented location: `oxiz-core/src/rewrite/combined.rs::CombinedRewriter::lookup_cache`.
- Estimated share-of-total: TBD - run `bench-profile`.
- Proposed fix: Add a tiny hot cache and reduce unique-term churn in rewrite passes.
