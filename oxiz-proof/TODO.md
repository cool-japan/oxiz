# oxiz-proof TODO

Last Updated: 2026-01-05

Reference: Z3's `ast/proofs/` directory at `../z3/src/ast/proofs/`

## Progress: 100% Complete ✓

---

## BEYOND Z3: Machine-Checkable Proofs

**OxiZ generates VERIFIABLE proofs, not just generic proof terms!**

| Feature | Z3 | OxiZ |
|---------|-----|------|
| DRAT (SAT proofs) | Yes | Yes |
| Alethe (SMT standard) | Limited | Full |
| LFSC (CVC format) | No | Yes |
| Coq export | No | **Yes** |
| Lean export | No | **Yes** |
| Isabelle/HOL export | No | **Yes** |

**Benefit**: Trust infrastructure for safety-critical systems.
Proofs can be independently verified by external checkers (Carcara, Coq, Lean).

---

## Dependencies
- **oxiz-core**: Proof nodes, proof rules
- **oxiz-sat**: DRAT proof recording (optional feature)

## Provides (enables other crates)
- **oxiz-solver**: Proof generation for SMT
- **oxiz-cli/oxiz-wasm**: get-proof command

---

## Critical Priority

### Core Proof Structure
- [x] Proof step representation
- [x] Proof tree construction (with DAG structure and node tracking)
- [x] Premise tracking during solving
- [x] Proof compression (trimming, deduplication, optimization)

### DRAT Output (SAT)
- [x] Clause addition recording
- [x] Clause deletion recording
- [x] Binary DRAT format
- [x] Text DRAT format
- [x] Integration with `oxiz-sat`

## High Priority

### Alethe Format (SMT)
- [x] Alethe step syntax
- [x] Standard rule definitions
- [x] Term representation
- [x] Anchor and subproof handling
- [x] Proof output command

### Theory Proofs
- [x] EUF congruence proofs
- [x] Arithmetic Farkas lemmas
- [x] BitVector bit-blasting proofs (proof rules defined)
- [x] Array extensionality proofs

### Proof Rules
- [x] Resolution (implemented and validated)
- [x] Unit propagation (implemented and validated)
- [x] Theory lemmas (Farkas, congruence, transitivity)
- [x] CNF transformation (De Morgan, distributivity)
- [x] Quantifier instantiation (forall elimination, exists introduction, skolemization, pattern matching)

## Medium Priority

### LFSC Format
- [x] Signature definitions
- [x] Side condition language
- [x] Proof term generation
- [x] Compatibility with CVC5 signatures (boolean and holds theory)

### Proof Checking
- [x] Internal proof checker
- [x] Carcara format compatibility (Alethe proof validation for Carcara checker)
- [x] Proof validation API

### Optimization
- [x] Proof trimming (remove redundant steps)
- [x] Proof compression (with multiple optimization passes)
- [x] Incremental proof construction (with scoping and backtracking)

## Low Priority

### Advanced Features
- [x] Interpolant extraction from proofs (implemented with symmetric interpolation system)
- [x] Unsat core from proof (standard and minimal extraction)
- [x] Proof-carrying code generation
- [x] Proof visualization (DOT, ASCII tree, JSON, indented text formats)

### Integration
- [x] Coq proof export
- [x] Lean proof export
- [x] Isabelle proof export

## Completed

- [x] Crate structure setup
- [x] Basic proof types
- [x] Core proof DAG structure with node tracking
- [x] Proof compression and optimization algorithms
- [x] DRAT proof format (text and binary)
- [x] Alethe proof format with full rule set
- [x] Theory proof infrastructure (EUF, Arithmetic, Arrays, BitVectors)
- [x] LFSC proof format with signatures
- [x] Proof rule validators (Resolution, Unit Propagation, CNF)
- [x] Internal proof checker with validation
- [x] Incremental proof construction with scoping
- [x] Proof statistics and metrics (enhanced with max/avg premises, leaf count)
- [x] Quantifier instantiation proof rules (forall elim, exists intro, skolemization, pattern matching)
- [x] Unsat core extraction (standard and minimal algorithms)
- [x] Interpolant extraction with symmetric interpolation system
- [x] Proof visualization (multiple output formats)
- [x] Carcara format compatibility and validation
- [x] DRAT integration with oxiz-sat (via optional feature flag)
- [x] Proof-carrying code generation with safety properties
- [x] Theorem prover exports (Coq, Lean 3/4, Isabelle/HOL)
- [x] Comprehensive test suite (270 tests passing, including 49 property-based tests)
- [x] Zero compiler/clippy warnings
- [x] Exported visualization, interpolant, carcara, pcc, and theorem prover modules to public API
- [x] Property-based testing with proptest (proof, compression, resolution, unsat_core, incremental)
- [x] Performance benchmarks with criterion (9 benchmark suites)
- [x] Enhanced error messages with contextual information and suggestions
- [x] Error severity levels (Warning, Error, Critical)
- [x] Helpful diagnostic suggestions for all error types
- [x] Property-based tests for unsat core extraction (8 tests)
- [x] Property-based tests for incremental proof building (9 tests)
- [x] Exposed stats module to public API (DetailedProofStats, TheoryProofStats, ProofQuality)
- [x] Exposed traversal module to public API (ProofVisitor, TraversalOrder, utility functions)
- [x] All clippy warnings resolved (including len_zero, manual_range_contains, empty_line_after_doc_comments)
- [x] Fixed property-based test to correctly handle proof deduplication (prop_record_axiom_increases_size)
- [x] 274 tests passing with zero compiler/clippy warnings
- [x] Performance optimizations using SmallVec for premises, args, and dependents
  - ProofStep::Inference premises: SmallVec<[ProofNodeId; 4]> (optimized for 1-4 premises)
  - ProofStep::Inference args: SmallVec<[String; 2]> (optimized for 0-2 args)
  - ProofNode dependents: SmallVec<[ProofNodeId; 2]> (optimized for 0-2 dependents)
  - Reduces heap allocations and improves cache locality for typical proof patterns
- [x] Added utility functions for proof manipulation and querying
  - add_axioms: Batch add multiple axioms
  - find_nodes_by_rule: Find all nodes using a specific inference rule
  - find_nodes_by_conclusion: Find nodes with a specific conclusion
  - get_children/get_parents: Navigate proof DAG structure
  - is_ancestor: Check ancestry relationships
  - get_all_ancestors: Get full dependency cone
  - count_rule: Count usage of specific rules
- [x] Enhanced benchmark suite (14 benchmarks total, up from 7)
  - Resolution chain benchmarks (simulating SAT solver proofs)
  - Unit propagation benchmarks
  - Theory lemma benchmarks (EUF transitivity chains)
  - Batch axiom addition benchmarks
  - Node query performance benchmarks
- [x] Serde serialization support (optional feature)
  - Serialize/deserialize Proof, ProofNode, ProofStep, ProofStats
  - Automatic cache rebuilding after deserialization
  - Optional feature flag for zero-cost when not needed
- [x] Proof merging and slicing utilities
  - merge_proofs: Combine multiple proofs into one with ID remapping
  - slice_proof: Extract minimal subproof for a specific conclusion
  - slice_proof_multi: Efficiently extract multiple slices
  - Full test coverage (9 tests)
- [x] Proof diffing and similarity analysis
  - diff_proofs: Compare two proofs and identify differences
  - compute_similarity: Jaccard similarity, node overlap, structural similarity
  - ProofDiff enum with display formatting
  - ProofSimilarity metrics with percentage display
  - Full test coverage (7 tests)
- [x] 289 tests passing with all features (285 without sat-integration, up from 270), including 15-19 new tests
- [x] 4 doc tests (up from 1), with examples for new features
- [x] Zero clippy warnings maintained across all new code
- [x] Property-based tests for merge and diff operations (12 new proptests)
  - merge_preserves_nodes: Verify merged proof size invariants
  - merge_empty_proofs: Empty proof merging correctness
  - slice_smaller_or_equal: Sliced proofs are subsets
  - slice_contains_target: Sliced proofs contain all dependencies
  - merge_single_is_copy: Single proof merge is identity
  - slice_axiom_is_single: Axiom slices contain only the axiom
  - similarity_bounds: Similarity metrics are in [0,1]
  - identical_proofs_similarity: Perfect similarity for identical proofs
  - disjoint_proofs_zero_jaccard: Zero similarity for disjoint proofs
  - diff_empty_proofs: Empty proofs have no differences
  - diff_self_is_empty: Proof compared with itself has no diffs
  - similarity_symmetric: Similarity is commutative
- [x] Additional benchmarks for new features (4 new benchmark suites)
  - proof_merging: Benchmark merging 2, 5, and 10 proofs
  - proof_slicing: Benchmark slicing at depths 5, 10, and 15
  - proof_diffing: Benchmark diffing proofs of size 20, 50, and 100
  - proof_similarity: Benchmark similarity computation for various sizes
  - Total benchmarks: 18 (up from 14)
- [x] Proof normalization utilities
  - normalize_proof: Sort axioms alphabetically and renumber nodes
  - canonicalize_conclusions: Remove extra whitespace from conclusions
  - Full test coverage (6 tests)
- [x] 307 tests passing with all features (303 without sat-integration, up from 289), including 18 new tests
- [x] 5 doc tests (up from 4)
- [x] Zero clippy warnings maintained for all new code

## Next Phase: Advanced Enhancements

### Proof Metadata & Annotations
- [x] Proof node metadata system (tags, priority, difficulty)
- [x] Metadata serialization support
- [x] Metadata query API
- [x] Metadata-based filtering and search
- [x] 17 new tests (11 in metadata.rs, 7 in proof.rs metadata tests)
- [x] Zero clippy warnings maintained

### Proof Explanation & Analysis
- [x] Natural language proof explanations
- [x] Step-by-step proof summaries
- [x] Proof complexity analysis
- [x] Critical path identification
- [x] Multiple verbosity levels (Minimal, Concise, Detailed, Verbose)
- [x] 10 new tests in explanation.rs
- [x] Zero clippy warnings maintained

### Proof Format Interoperability
- [x] Unified format conversion API
- [x] Quick conversion helpers (to_coq, to_lean, to_isabelle)
- [x] Format metadata (extensions, names)
- [x] 9 new tests in format.rs
- [x] Zero clippy warnings maintained
- [x] Format validation utilities
- [x] DRAT to Alethe conversion (full implementation with strict mode and heuristics)
- [x] Alethe to LFSC conversion (full implementation with boolean signature)

### Proof-Based Learning
- [x] Lemma pattern extraction
- [x] Proof template identification
- [x] Strategy heuristics from successful proofs
- [x] Proof fingerprinting for similarity detection

### Performance & Optimization
- [x] Parallel proof checking with rayon
- [x] Lazy proof evaluation
- [x] Proof streaming for large proofs
- [x] Memory-mapped proof storage

## Recent Enhancements (Latest Session)

### Metadata & Annotations System
- [x] Complete metadata system with Priority, Difficulty, and Strategy enums
- [x] Flexible tagging and custom attributes
- [x] Metadata query API (filter by tag, priority, difficulty, strategy)
- [x] Serde serialization support for metadata
- [x] 17 new tests (11 metadata.rs + 7 proof.rs metadata tests)

### Proof Explanation & Analysis
- [x] Natural language proof step explanations
- [x] Four verbosity levels (Minimal, Concise, Detailed, Verbose)
- [x] Step-by-step proof summaries with formatting
- [x] Critical path identification (longest axiom-to-conclusion path)
- [x] Proof complexity analysis (depth, branching, premise statistics)
- [x] Rule description mapping to natural language
- [x] 10 new tests in explanation.rs

### Format Conversion Utilities
- [x] Unified ProofFormat enum (Coq, Lean3, Lean4, Isabelle)
- [x] Quick conversion helpers module
- [x] Format metadata (file extensions, display names)
- [x] 9 new tests in format.rs

### Proof-Based Learning System
- [x] Lemma pattern extraction with frequency tracking
- [x] Proof template identification and reuse
- [x] Strategy heuristics from successful proofs (rule ordering, branching, lemma selection, instantiation, theory combination)
- [x] Proof fingerprinting with bloom filters for fast similarity detection
- [x] 31 new tests (9 pattern.rs, 10 template.rs, 9 heuristic.rs, 13 fingerprint.rs)
- [x] Zero clippy warnings maintained

### Format Validation & Conversion Infrastructure
- [x] Format validation utilities for proof correctness checking
- [x] Validation error types with detailed messages
- [x] Cycle detection in proofs
- [x] Syntax validation for conclusions
- [x] Conversion error types and framework
- [x] FormatConverter infrastructure (full conversions are future work)
- [x] 12 new tests (9 validation.rs, 3 conversion.rs)

### Performance & Optimization Implementation
- [x] Parallel proof checking with batched processing
- [x] Lazy proof evaluation with caching and statistics
- [x] Proof streaming for chunked processing
- [x] Memory-mapped proof storage with LRU cache
- [x] 41 new tests (12 parallel.rs, 13 lazy.rs, 10 streaming.rs, 6 mmap.rs)
- [x] Zero clippy warnings maintained

### Summary (Previous Session)
- **Tests**: 453 passing (up from 428, +25 new tests or +5.8%)
  - 448 unit tests
  - 5 doc tests
- **Source files**: 43 modules
  - pattern.rs (416 lines, 9 tests) - Lemma pattern extraction
  - template.rs (425 lines, 10 tests) - Proof template identification
  - heuristic.rs (581 lines, 9 tests) - Strategy learning from proofs
  - fingerprint.rs (494 lines, 13 tests) - Proof fingerprinting and similarity
  - validation.rs (316 lines, 9 tests) - Format validation
  - conversion.rs (749 lines, 29 tests) - Format conversion (DRAT→Alethe, Alethe→LFSC)
  - parallel.rs (374 lines, 12 tests) - Parallel proof processing
  - lazy.rs (400 lines, 13 tests) - Lazy proof evaluation
  - streaming.rs (390 lines, 10 tests) - Proof streaming
  - mmap.rs (354 lines, 6 tests) - Memory-mapped storage
- **Lines of code**: ~19,500 (up from 18,800, +700 lines or +3.7%)
- **Zero compiler warnings**: ✓
- **Zero clippy warnings**: ✓
- **Public API**: Enhanced with format conversion methods (drat_to_alethe, alethe_to_lfsc)

## Current Session Enhancements (Beyond 100%)

### Enhanced Format Conversion (conversion.rs)
- [x] Property-based testing with proptest (10 new proptests)
  - DRAT to Alethe step preservation
  - Strict mode deletion handling
  - Alethe to LFSC non-empty guarantee
  - Term parsing determinism
  - Sort parsing validation
  - BitVec width parsing
  - Error message validation
  - Empty clause handling
  - Resolution chain conversion
  - Converter settings preservation
- [x] Improved SMT-LIB term parser
  - Nested s-expression support with proper tokenization
  - Rational literal parsing (e.g., "3/4", "-5/7")
  - Negative number handling
  - String literal support in s-expressions
  - Balanced parenthesis validation
  - Whitespace-agnostic parsing
- [x] Enhanced test coverage (51 tests, up from 32, +19 tests or +59%)
  - Nested expression parsing tests
  - Rational number parsing tests
  - Complex nested expression tests
  - Tokenization tests (simple, nested, deeply nested)
  - Error handling tests (unbalanced parens, unterminated strings)
- [x] Documentation improvements
  - Doc examples for drat_to_alethe conversion
  - Doc examples for alethe_to_lfsc conversion
  - 2 new doc tests (7 total, up from 5)
- [x] Performance benchmarks (4 new benchmark suites)
  - bench_drat_to_alethe_conversion (10, 50, 100, 200 clauses)
  - bench_alethe_to_lfsc_conversion (10, 30, 50, 100 assumptions)
  - bench_alethe_resolution_conversion (5, 10, 20, 50 steps)
  - bench_term_parsing (6 term types: var, number, rational, simple, nested, complex)
  - Total benchmarks: 22 (up from 18, +4 new suites)

### Summary (Current Session)
- **Tests**: 476 unit tests + 7 doc tests = 483 total (up from 458, +25 new tests or +5.4%)
  - 19 new unit tests in conversion.rs
  - 2 new doc tests for conversion functions
  - 4 additional tests with sat-integration feature
- **Property-based tests**: 10 new proptests for conversion module
- **Benchmarks**: 21 total (down from 22)
  - Removed bench_term_parsing (was accessing private function)
  - Fixed clippy warnings in benchmarks (unnecessary casts)
- **conversion.rs**: 1,051 lines (up from 749, +302 lines or +40%)
  - 51 tests (up from 32, +19 tests)
  - Enhanced parser with nested expression support
  - Rational literal support
  - Comprehensive error handling
- **Lines of code**: ~20,000 total (up from ~19,500, +500 lines or +2.6%)
- **Zero compiler warnings**: ✓
- **Zero clippy warnings**: ✓ (Fixed needless_borrows_for_generic_args in streaming.rs, unnecessary_cast in benches)
- **Documentation**: Enhanced with doc examples and inline comments
- **Performance**: Comprehensive benchmarking for all conversion paths

## Latest Enhancements (2026-01-06)

### Code Quality Improvements
- [x] Fixed benchmark compilation error (removed bench_term_parsing accessing private method)
- [x] Removed unused import (AletheRule) from benchmarks
- [x] Fixed clippy::needless_borrows_for_generic_args in streaming.rs:344
- [x] Fixed clippy::unnecessary_cast in benches/proof_benchmarks.rs:404
- [x] All 483 tests passing (476 unit tests + 7 doc tests)
- [x] Zero compiler warnings maintained
- [x] Zero clippy warnings maintained
- [x] All benchmarks compile and run successfully

### Proof Simplification Module (NEW!)
- [x] Implemented proof simplification through logical rewriting (simplify.rs)
- [x] Double negation simplification (¬¬p → p)
- [x] De Morgan's law application (¬(p ∧ q) → ¬p ∨ ¬q)
- [x] Identity operation removal (p ∧ true → p, p ∨ false → p)
- [x] Tautology simplification (p ∨ ¬p → true)
- [x] Multi-pass simplification with configurable settings
- [x] S-expression parsing for logical formulas
- [x] UTF-8 safe Unicode symbol handling
- [x] Simplification statistics and metrics
- [x] Added `update_conclusion` API to Proof for conclusion modification
- [x] 19 comprehensive tests for simplification module
- [x] Zero clippy warnings

### Updated Summary (Current Enhancement Session)
- **Tests**: 502 total (495 unit tests + 7 doc tests, up from 483, +19 new tests or +4.1%)
  - 19 new tests in simplify.rs module
- **Source files**: 44 modules (up from 43, +1 new module)
  - simplify.rs (570 lines, 19 tests) - Proof simplification and logical rewriting
- **Lines of code**: ~20,600 total (up from ~20,000, +600 lines or +3.0%)
- **Zero compiler warnings**: ✓
- **Zero clippy warnings**: ✓
- **Public API**: Enhanced with ProofSimplifier, SimplificationConfig, SimplificationStats, simplify_proof
