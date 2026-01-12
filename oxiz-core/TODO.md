# oxiz-core TODO

Last Updated: 2026-01-06

Reference: Z3's `ast/`, `util/`, and `tactic/` directories at `../z3/src/`

## Progress: 100% Complete ✓

## Dependencies
- None (foundation crate)

## Provides (enables other crates)
- **oxiz-sat**: Term representation, literal mapping
- **oxiz-theories**: Sort system, AST traversal, congruence closure
- **oxiz-solver**: SMT-LIB2 parser, tactic framework
- **oxiz-proof**: Proof nodes, proof rules
- **All crates**: Error handling, source locations

---

## AST

- [x] Add term substitution
- [x] Implement term rewriting rules
- [x] Add DAG traversal utilities
- [x] Implement term pattern matching
- [x] Add term size/depth computation
- [x] Support for quantifier instantiation patterns

## Sorts

- [x] Add sort inference for terms
- [x] Implement parametric sorts
- [x] Add sort aliases
- [x] Support for recursive datatypes

## SMT-LIB2 Parser

- [x] Complete `define-fun` parsing
- [x] Add `define-sort` support
- [x] Implement `declare-datatype`
- [x] Parse annotations and attributes
- [x] Add `set-info` handling
- [x] Support for quoted symbols
- [x] Implement `reset-assertions`

## SMT-LIB2 Printer

- [x] Pretty printing with configurable indentation
- [x] Add S-expression configuration options (PrettyConfig)
- [x] Model output in SMT-LIB2 format
- [x] Proof output formatting

## Tactics

- [x] Implement `split` tactic (case splitting)
- [x] Add `propagate-values` tactic
- [x] Implement `ctx-simplify` (contextual simplification)
- [x] Add `elim-uncnstr` (eliminate unconstrained)
- [x] Implement `solve-eqs` (Gaussian elimination)
- [x] Add tactic combinators (parallel, timeout)
- [x] Implement `bit-blast` tactic for BV
- [x] Implement `ackermannize` tactic for function elimination

## Error Handling

- [x] Add source location tracking
- [x] Add error recovery in parser (infrastructure complete)
- [x] Improve error messages (comprehensive error types with source location)

## Model Validation

- [x] Term evaluation under models
- [x] Assertion validation
- [x] Full model validation for correctness checking

## Proof System

- [x] Basic proof representation (ProofNode, ProofRule)
- [x] Proof tree construction
- [x] Proof structural validation (cycle detection, premise checking)
- [x] Proof tree analysis (height, leaves)
- [x] Proof simplification and minimization
  - [x] Prune unreachable nodes
  - [x] Simplify redundant rewrites
  - [x] Minimize unused assumptions
  - [x] Proof statistics
- [x] Proof transformation (e.g., resolution to natural deduction)
- [x] Proof certificate generation (LFSC, Alethe, DRAT, Coq, Isabelle)

## Advanced Features

- [x] Interpolation support (Craig interpolants)
- [x] Incremental solving infrastructure (Context with push/pop)
- [x] Theory combination via Nelson-Oppen
- [x] E-graph based equality reasoning
- [x] Congruence closure for equality reasoning
  - [x] Union-find data structure
  - [x] Basic congruence detection
  - [x] Equivalence class management
  - [x] Full congruence propagation (advanced)
    - [x] Backtrackable union-find with undo trail
    - [x] Explanation tracking for proof generation
    - [x] Worklist-based propagation
    - [x] Disequality reasoning and conflict detection
- [x] More efficient hash-consing with GC (mark-and-sweep, statistics, aggressive mode)

## Performance Optimizations

- [x] Caching for term evaluation (CachedEvaluator)
- [x] Parallel tactic execution (ParallelTactic combinator with thread-based execution)
- [x] Memory pool for term allocation
  - [x] Chunked allocation strategy (4096 terms per chunk)
  - [x] Memory usage statistics and fragmentation tracking
  - [x] Cache-friendly block allocation
  - [x] Configurable capacity for pre-allocation
- [x] Optimized substitution with structure sharing
  - [x] SubstitutionBuilder with persistent cache
  - [x] Composition of substitutions
  - [x] Batch application of substitutions
  - [x] Structure sharing in substitute method

## Additional Theories

- [x] Arrays theory support (select/store axioms, read-over-write, extensionality)
- [x] String theory support (concatenation, length, substring, contains, prefix/suffix, indexof, replace, conversions)
- [x] Floating-point theory support (IEEE 754 operations, rounding modes, conversions, predicates)
- [x] Algebraic datatypes theory (constructors, testers, selectors for recursive datatypes)

## Enhancements (All Complete!)

- [x] Custom tactics via Rhai scripting
  - **Completed:** 2026-01-05
  - Enables user-defined simplification strategies
  - Pure Rust implementation using Rhai engine
  - Supports custom goal transformations and solve results
- [x] SIMD-accelerated term comparison
  - **Completed:** 2026-01-05
  - Portable SIMD using `wide` crate for cross-platform acceleration
  - 2-4x performance improvement for large term arrays (>4 elements)
  - Functions: simd_compare_termids, simd_hash_termids, simd_all_equal, simd_contains
  - Comprehensive test suite with 20 tests covering edge cases
- [x] Code quality improvements
  - **Completed:** 2026-01-06
  - Fixed all 41 clippy warnings (NO warnings policy achieved)
  - Added comprehensive documentation for BitVectorAxiom enum fields
  - Added comprehensive documentation for FloatingPointAxiom enum fields
  - Refactored identical blocks in bitvector theory for cleaner code
  - All 410 tests passing with zero warnings
- [x] Cost-based extraction for E-graphs
  - **Completed:** 2026-01-06
  - Enables extraction of optimal term representations from equivalence classes
  - Two extraction algorithms: `extract_best` (single e-class) and `extract_all` (global extraction)
  - User-defined cost functions for flexible optimization strategies
  - Comprehensive test suite with 6 tests covering various scenarios
  - Pure Rust implementation with efficient fixed-point iteration
  - All 415 tests passing with zero warnings
- [x] Theory lemma caching
  - **Completed:** 2026-01-06
  - Implements efficient caching for theory lemmas to avoid redundant propagation
  - LRU eviction policy with configurable cache size
  - Multiple indexing strategies (by conclusion, hypothesis, theory)
  - Lemma subsumption detection and minimization
  - Cache statistics tracking (hits, misses, hit rate)
  - Comprehensive test suite with 17 tests covering all cache operations
  - Pure Rust implementation with FxHashMap for performance
  - All 428 tests passing with zero warnings
- [x] Parallel term operations with Rayon
  - **Completed:** 2026-01-06
  - Leverages Rayon for data parallelism on multi-core systems
  - Automatic parallelization threshold (100 terms) for optimal performance
  - Core operations: parallel_reduce, parallel_find, parallel_all_distinct
  - Collection operations: parallel_filter, parallel_map, parallel_partition
  - Boolean operations: parallel_any, parallel_all, parallel_count
  - ParallelStats for tracking parallelization metrics
  - Comprehensive test suite with 14 tests
  - Pure Rust implementation with zero unsafe code
  - All 442 tests passing with zero warnings

---

## Completed

- [x] Hash-consed term representation
- [x] Basic sort system
- [x] SMT-LIB2 lexer
- [x] SMT-LIB2 parser (basic commands)
- [x] SMT-LIB2 printer
- [x] Tactic framework skeleton
- [x] SimplifyTactic implementation
- [x] Boolean simplifications (And, Or, Not, Implies, Ite)
- [x] Proof representation (proof.rs)
- [x] Model validation utilities (validation.rs)
- [x] Full congruence propagation with explanation tracking
- [x] Memory pool for term allocation
- [x] Optimized substitution with structure sharing
- [x] All theory AST representations (Arrays, Strings, FP, Datatypes)
- [x] Custom tactics via Rhai scripting (user-defined simplification strategies)
- [x] SIMD-accelerated term comparison (portable SIMD with wide crate)
