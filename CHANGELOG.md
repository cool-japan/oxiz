# Changelog

All notable changes to OxiZ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-12

### Added
- **New meta-crate `oxiz`**: Unified API with feature flags for modular usage
  - `default = ["solver"]`: Core SMT solving functionality
  - `nlsat`: Nonlinear real arithmetic solver
  - `optimization`: MaxSMT and optimization features
  - `spacer`: CHC solver for program verification
  - `proof`: Proof generation and checking
  - `standard`: All common features except SPACER
  - `full`: All features enabled
- Workspace-level lints configuration for consistent code quality

### Changed
- Removed redundant `rust-version` from workspace (Edition 2024 already requires Rust 1.85+)
- Updated README with meta-crate usage examples
- Updated crates.io badges to point to `oxiz` meta-crate
- Updated CDN documentation with 0.1.1 URLs

### Documentation
- Added comprehensive API documentation to meta-crate
- Created oxiz/README.md with feature flag guide
- Updated installation instructions across all documentation

### Fixed

#### MBQI (Model-Based Quantifier Instantiation)
- **Added missing comparison handlers**: Implemented `Gt`, `Ge`, and `Le` evaluation in `evaluate_under_model()`. Previously only `Lt` was supported, causing quantifier instantiation failures.

#### CDCL(T) Theory Propagation
- **Fixed simplex constraint handling**: `add_le()` now properly substitutes basic variables before adding constraints to the tableau, matching the behavior of `add_strict_lt()`. This resolves contradictory constraint satisfaction issues.
- **Fixed incremental solving**: `ArithSolver::push()` and `pop()` now correctly call `simplex.push()` and `simplex.pop()`, enabling proper backtracking in incremental solving scenarios.
- **Fixed theory-SAT synchronization**: Added `on_new_level()` callback to `TheoryCallback` trait. The SAT solver now notifies theory solvers when entering new decision levels, allowing proper theory state management and preventing stale state bugs.

#### Bitvector Theory
- **Basic bitvector support**: Integrated bitvector comparisons (`BvUlt`, `BvUle`, `BvSlt`, `BvSle`) by treating them as bounded integer comparisons. This enables arithmetic reasoning over bitvectors for common use cases.
- **BitVecConst handling**: Added support for bitvector constants in arithmetic constraint parsing, treating them as integer values.

### Changed
- **Code quality**: Eliminated all compiler warnings, achieving clippy clean status with `-D warnings` flag.
- **Test coverage**: All 84 solver tests passing (100% success rate).

### Compatibility
- **Breaking changes**: None. This is a backwards-compatible bug-fix release.
- **Verified with**: Legalis formal verification framework (467/467 tests passing).

## [0.1.0] - 2026-01-12

### Initial Release

OxiZ 0.1.0 marks the first public release of a Pure Rust SMT solver achieving ~90%+ feature parity with Z3.

### Added

#### Core Infrastructure
- Complete SMT-LIB2 parser and printer
- Term management with hash consing
- Sort system with parametric types
- Incremental solving with push/pop
- Model generation and evaluation

#### SAT Solver (`oxiz-sat`)
- CDCL (Conflict-Driven Clause Learning) with two-watched literals
- Multiple branching heuristics: VSIDS, LRB, VMTF, CHB
- Clause learning with recursive minimization
- Preprocessing: BCE, BVE, variable elimination, subsumption
- DRAT proof generation
- Local search integration
- Lookahead solving
- AllSAT enumeration
- Parallel portfolio solver

#### Theory Solvers (`oxiz-theories`)
- **EUF**: Congruence closure with explanation generation
- **LRA**: Simplex with Bland's rule, dual simplex
- **LIA**: Branch-and-bound, Gomory cuts, branch-and-cut
- **BitVectors**: Bit-blasting, word-level propagation
- **Arrays**: Theory of arrays with extensionality
- **Strings**: Automata-based solver, regex support
- **Floating-Point**: IEEE 754 semantics via bit-precise encoding
- **Datatypes**: Algebraic data types with constructors/selectors/testers
- **Pseudo-Boolean**: Cardinality and weighted PB constraints
- **Special Relations**: Partial/total orders, transitive closure
- **Difference Logic**: Graph-based DL solver
- **UTVPI**: Unit Two Variable Per Inequality solver

#### Nonlinear Arithmetic (`oxiz-nlsat`)
- NLSAT algorithm for nonlinear real arithmetic
- Cylindrical Algebraic Decomposition (CAD)
- Algebraic number representation
- Polynomial operations over exact rationals

#### Quantifier Handling
- E-matching with multi-pattern triggers
- MBQI (Model-Based Quantifier Instantiation)
- Skolemization
- DER (Destructive Equality Resolution)
- Model-Based Projection (MBP)
- Quantifier instantiation tactics

#### Tactics System (`oxiz-core`)
- 25+ tactics including:
  - Simplify, PropagateValues, BitBlast, Ackermannize
  - Fourier-Motzkin elimination
  - NNF, Tseitin CNF conversion
  - PB2BV, NLA2BV, LIA2Card
  - Context-solver simplification
  - Solve equations, eliminate unconstrained
- Tactic combinators: Then, OrElse, Repeat, Parallel, Timeout, Cond, When, FailIf
- Probe system with 11 built-in probes
- Scriptable tactic language

#### Optimization (`oxiz-opt`)
- MaxSAT solving: Fu-Malik, RC2, stratified
- Large Neighborhood Search (LNS)
- OMT (Optimization Modulo Theories)
- Lexicographic and Pareto optimization
- Weighted soft constraints

#### Model Checking (`oxiz-spacer`)
- CHC (Constrained Horn Clauses) solving
- PDR/IC3 with lemma generalization
- BMC (Bounded Model Checking)
- Distributed solving support
- Loop invariant inference

#### Proof Generation (`oxiz-proof`)
- DRAT proofs for SAT
- Alethe proof format
- LFSC proof format
- Carcara proof checker integration
- Export to Coq, Lean, Isabelle
- Craig interpolation (McMillan, Pudlak, Huang algorithms)

#### Mathematical Library (`oxiz-math`)
- Arbitrary-precision rationals
- Polynomial arithmetic
- Matrix operations with QR decomposition
- Grobner basis computation
- Real algebraic number arithmetic
- Linear programming (revised simplex)
- Sturm sequences for root isolation

#### WebAssembly (`oxiz-wasm`)
- Full WASM bindings for browser use
- Async solving API
- String utilities and object pools

#### Command-Line Interface (`oxiz-cli`)
- SMT-LIB2 file solving
- Interactive REPL mode
- Proof output
- Verbose/debug modes
- Portfolio solving

### Technical Details

- **Pure Rust**: Zero C/C++ dependencies
- **Lines of Code**: ~173,500 Rust LOC
- **Test Coverage**: 3,670 tests
- **Edition**: Rust 2024 (requires Rust 1.85+)

### Performance

- Competitive with established solvers on standard benchmarks
- SIMD-accelerated term comparison
- Efficient hash consing with string interning
- Parallel solving capabilities

### Known Limitations

- QF_NIA (nonlinear integer) support is partial
- Some advanced Z3 features not yet implemented:
  - Full Datalog engine
  - Complete Unicode character theory
  - Python bindings

## [Unreleased]

### Planned
- Enhanced parallel portfolio strategies
- Additional proof formats
- Performance optimizations
- Extended string theory support
