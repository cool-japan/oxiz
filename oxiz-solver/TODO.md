# oxiz-solver TODO

Last Updated: 2026-01-05

Reference: Z3's `smt/` directory at `../z3/src/smt/`

## Progress: 100% Complete

## Dependencies
- **oxiz-core**: AST, SMT-LIB2 parser, tactic framework
- **oxiz-sat**: CDCL SAT solver
- **oxiz-theories**: Theory solvers (EUF, LRA, LIA, BV, Arrays, Strings, FP)

## Provides (enables other crates)
- **oxiz-spacer**: SMT solving for PDR queries
- **oxiz-opt**: OMT solving for optimization
- **oxiz-wasm/oxiz-cli**: Main solver API

---

## CDCL(T) Integration

- [x] Basic CDCL(T) structure with theory callbacks
- [x] Encoding happens at assertion time
- [x] SAT solver integration with theory hooks
- [x] Theory propagation during BCP (EUF equalities/disequalities)
- [x] Theory conflict clause generation (EUF conflicts with proper explanation)
- [x] Lazy theory checking (batch mode)
- [x] Eager theory checking (incremental)
- [x] Theory decision procedures (infrastructure added)
- [x] Theory-aware branching (infrastructure and enable/disable API)

## Theory Combination

- [x] Nelson-Oppen combination framework (infrastructure and equality sharing)
- [x] Equality sharing between theories (EqualityNotification and propagation)
- [x] Disequality propagation (integrated with EUF)
- [x] Model-based theory combination
- [x] TheoryCombination trait moved to oxiz-theories (proper architecture)
- [x] ArithSolver implements TheoryCombination trait
- [x] Arithmetic theory notified during equality propagation
- [x] Comprehensive test coverage for theory combination

## Boolean Encoding

- [x] Complete Tseitin encoding for all term kinds
- [x] Encoding for arithmetic predicates
- [x] Encoding for bit-vector operations
- [x] Encoding for Xor
- [x] Encoding for Distinct
- [x] Encoding for Let bindings
- [x] Encoding for quantifiers (basic)
- [x] Polarity-aware encoding
- [x] Clause sharing for common subterms (via term_to_var mapping)

## Model Generation

- [x] Basic model generation from SAT assignments
- [x] Model evaluation for boolean terms
- [x] Model evaluation with constant folding
- [x] Complete model for all theories (arithmetic, bitvector values)
- [x] Model minimization
- [x] Pretty printing models

## Unsatisfiable Core

- [x] Unsat core extraction (basic)
- [x] Core minimization (greedy deletion algorithm)
- [x] Assumption tracking (check-sat-assuming support)

## Proof Generation

- [x] Resolution proof for SAT (infrastructure with ProofStep and Proof)
- [x] Theory lemma proofs (TheoryLemma proof step type)
- [x] Combined proof output (Proof formatting and get-proof command)

## Context Management

- [x] Basic push/pop support with state restoration
- [x] Assertion stack management
- [x] Declaration scope tracking
- [x] Track trivial unsat (False assertions) across push/pop
- [x] Efficient trail-based undo for all components
- [x] Incremental declaration removal

## SMT-LIB2 Commands

- [x] `get-model` formatting
- [x] `get-assertions` (basic)
- [x] `get-unsat-core` (basic)
- [x] `echo` command
- [x] `exit` command
- [x] `get-value` evaluation with model evaluation
- [x] `check-sat-assuming` with assumption support
- [x] `get-proof` (proof generation infrastructure with formatting)
- [x] Full `get-unsat-core` with minimization (greedy deletion)

## Optimization

- [x] Minimize/maximize objectives (basic implementation with model evaluation)
- [x] Lexicographic optimization (priority-based objective handling)
- [x] Pareto optimization (multi-objective with Pareto front computation)
- [x] Iterative optimization (linear search for integer objectives)

## Preprocessing and Simplification

- [x] Formula simplification module (simplify.rs)
- [x] Constant propagation (not(true) => false, and(x, true) => x, etc.)
- [x] Boolean simplification (and/or/implies/ite reductions)
- [x] Trivial equality elimination (x = x => true)
- [x] Contradiction detection (true = false => false)
- [x] Integration with solver (simplify config option)
- [x] Simplification at assertion time
- [x] Statistics tracking for simplification

## Solver Enhancements

- [x] Extended SolverConfig with simplify flag
- [x] Resource limit configuration (max_conflicts, max_decisions)
- [x] Encoding for all TermKind variants (string and floating-point operations)
- [x] Complete pattern matching for string theory (StringLit, StrConcat, StrLen, etc.)
- [x] Complete pattern matching for floating-point theory (FpLit, FpAdd, FpMul, etc.)
- [x] Simplified term handling before encoding

## Preprocessing and Simplification (Enhanced)

- [x] Advanced preprocessing techniques
  - [x] Unit propagation at preprocessing level
  - [x] Pure literal detection and analysis
  - [x] Nested operation flattening (and/or)
  - [x] Duplicate literal elimination
  - [x] Tautology detection (or(x, not(x)) => true)
  - [x] Contradiction detection (and(x, not(x)) => false)
  - [x] Constant propagation
  - [x] Boolean simplification
  - [x] Trivial equality elimination

## Statistics and Monitoring

- [x] Resource limit enforcement (conflict/decision limits)
- [x] Detailed statistics tracking (decisions, conflicts, propagations)
- [x] Statistics API via Context (`get_statistics()`)
- [x] SMT-LIB2 `get-info :all-statistics` command support
- [x] Statistics tracking in theory callbacks

## Advanced SAT Solver Features (Integrated)

- [x] Clause minimization (learned clause reduction) - Exposed via `enable_clause_minimization`
- [x] Learned clause subsumption - Exposed via `enable_clause_subsumption`
- [x] Restart strategies configuration - Exposed via `restart_strategy` (Luby, Geometric, Glucose, LocalLbd)
- [x] Variable elimination during preprocessing - Exposed via `enable_variable_elimination`
- [x] Blocked clause elimination - Exposed via `enable_blocked_clause_elimination`
- [x] Symmetry breaking predicates - Exposed via `enable_symmetry_breaking`
- [x] Inprocessing (periodic preprocessing) - Exposed via `enable_inprocessing` and `inprocessing_interval`

Note: These features are implemented in `oxiz-sat` and exposed through `SolverConfig` in oxiz-solver.
The actual implementation leverages:
- RecursiveMinimizer for clause minimization
- SubsumptionChecker for learned clause subsumption
- Preprocessor for variable elimination and blocked clause elimination
- SymmetryBreaker for symmetry breaking predicates
- RestartStrategy enum for different restart policies

## Configuration Presets and Builder API

- [x] Preset configurations for common use cases:
  - [x] `SolverConfig::fast()` - Minimal preprocessing for quick results
  - [x] `SolverConfig::balanced()` - Default, good balance (used by `default()`)
  - [x] `SolverConfig::thorough()` - Aggressive preprocessing for hard problems
  - [x] `SolverConfig::minimal()` - Bare minimum for debugging
- [x] Builder-style API for configuration:
  - [x] `with_proof()` - Enable proof generation
  - [x] `with_timeout(ms)` - Set timeout
  - [x] `with_max_conflicts(n)` - Set conflict limit
  - [x] `with_max_decisions(n)` - Set decision limit
  - [x] `with_parallel(threads)` - Enable parallel solving
  - [x] `with_restart_strategy(strategy)` - Set restart strategy
  - [x] `with_theory_mode(mode)` - Set theory mode
- [x] Comprehensive test coverage for all presets and builder methods

## Performance and Testing

- [x] Comprehensive benchmarks for solver operations
  - [x] Simple SAT/UNSAT benchmarks
  - [x] Push/pop benchmarks
  - [x] Arithmetic constraint benchmarks
  - [x] Complex encoding benchmarks
  - [x] Solver configuration benchmarks
  - [x] Context API benchmarks
  - [x] Model generation benchmarks
  - [x] Simplification benchmarks
- [x] Utility methods for common patterns
  - [x] `assert_many()` - assert multiple terms at once
  - [x] `num_assertions()` - get assertion count
  - [x] `num_variables()` - get variable count
  - [x] `has_assertions()` - check if solver has assertions
  - [x] `context_level()` - get push/pop depth

## Examples and Documentation

- [x] Comprehensive example programs:
  - [x] `simple_sat.rs` - Boolean satisfiability examples
  - [x] `arithmetic.rs` - Linear arithmetic (LIA/LRA) examples
  - [x] `optimization.rs` - SMT optimization (OMT) examples
  - [x] `incremental.rs` - Incremental solving with push/pop
  - [x] `smt_lib_script.rs` - SMT-LIB2 script execution
- [x] All examples compile without warnings
- [x] All examples tested and verified
- [x] Comprehensive doc tests (7 tests):
  - [x] Library-level examples (3 tests)
  - [x] `Context` type examples (2 tests)
  - [x] `Optimizer` type examples (2 tests)
- [x] All doc tests pass and are auto-tested

## Completed

- [x] Basic CDCL(T) structure
- [x] SAT solver integration
- [x] Theory solver hooks
- [x] Push/pop support with state management
- [x] Context with SMT-LIB2 execution
- [x] Basic check-sat with theory integration
- [x] set-logic handling
- [x] declare-const/declare-fun
- [x] assert command
- [x] Complete Tseitin encoding for all boolean operators
- [x] Encoding for theory atoms (arithmetic, bitvector, arrays, UF)
- [x] Special handling for True/False constants
- [x] Basic unsat core generation
- [x] Named assertions for unsat core tracking
- [x] Model generation from SAT assignments
- [x] Model evaluation with constant folding
- [x] get-value command with proper evaluation
- [x] Recursive term evaluation in models
- [x] Boolean simplification in model evaluation
- [x] Theory propagation for EUF (equality and disequality constraints)
- [x] Constraint storage (var-to-constraint mapping)
- [x] Theory conflict detection during BCP
- [x] Theory conflict clause generation from EUF explanations
- [x] Conflict clause conversion (terms to literals)
- [x] Model pretty printing in SMT-LIB2 format
- [x] Model minimization API
- [x] Objective evaluation in optimization
- [x] Theory decision procedures infrastructure (TheoryDecision struct and get_decision_hints)
- [x] Theory-aware branching infrastructure (enable/disable API)
- [x] Unsat core minimization (greedy deletion algorithm)
- [x] Assumption-based solving (check_with_assumptions method)
- [x] Lexicographic optimization (multiple objectives with priorities)
- [x] Pareto optimization (multi-objective Pareto front computation)
- [x] Iterative optimization for integer objectives
- [x] Nelson-Oppen equality sharing infrastructure (TheoryCombination trait)
- [x] Equality notifications between theories (EqualityNotification)
- [x] Equality propagation mechanism (propagate_equalities method)
- [x] Proof generation infrastructure (Proof, ProofStep types)
- [x] Resolution proof tracking (ProofStep::Resolution)
- [x] Theory lemma proofs (ProofStep::TheoryLemma)
- [x] Proof formatting and output (Proof::format method)
- [x] get-proof command support (integrated with Context)
- [x] Resource limit enforcement (max_conflicts, max_decisions in SolverConfig)
- [x] Detailed solver statistics (Statistics struct with all counters)
- [x] Statistics tracking in theory callbacks (propagations, conflicts, theory operations)
- [x] Statistics API via Context (get_statistics method)
- [x] SMT-LIB2 get-info :all-statistics support
- [x] Enhanced preprocessing pipeline:
  - [x] Nested operation flattening
  - [x] Duplicate literal elimination
  - [x] Tautology and contradiction detection
  - [x] Unit propagation at preprocessing level
  - [x] Pure literal detection and analysis
