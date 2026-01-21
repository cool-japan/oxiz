# oxiz-cli TODO

Last Updated: 2026-01-17

## Progress: ~100% Complete

## Recent Enhancements (2026-01-17)

### TPTP Format Support - COMPLETED
**Support for more input formats**
- Created tptp.rs module for TPTP format parsing
- Enables interoperability with theorem prover community
- CLI support for .tptp and .p file extensions

### SMT-LIB2 v2.6 Extended Syntax - COMPLETED
**Enhanced SMT-LIB2 language features**
- Match expressions for pattern matching
- Parametric datatypes support
- Full v2.6 specification compliance

### Distributed Solving - COMPLETED
**Support for distributed problem solving**
- Created distributed.rs module for cluster-based solving
- Parallel problem distribution across nodes
- Load balancing and result aggregation

### Interpolant Generation - COMPLETED
**Craig interpolant support for analysis**
- Created interpolate.rs module for interpolant computation
- Useful for abstraction refinement and verification
- Integration with proof generation system

### Web Dashboard - COMPLETED
**Web-based monitoring interface**
- Created dashboard.rs module for HTTP-based dashboard
- Real-time solver status and statistics
- Visual progress tracking and history

### Python Bindings - COMPLETED
**PyO3-based Python integration**
- Created oxiz-py crate with full Python API
- Native Python types and error handling
- pip-installable package

### REST API Server - COMPLETED
**HTTP API for solver access**
- Created server.rs module with REST endpoints
- JSON-based request/response format
- Async solving with job management

### VS Code Extension - COMPLETED
**IDE integration for VS Code**
- Created oxiz-vscode/ extension package
- Syntax highlighting for SMT-LIB2
- Inline diagnostics and hover information
- Integrated solver execution

## Recent Enhancements (2026-01-05 - Latest)

### Parallel Portfolio Solving - COMPLETED
**Implementation of concurrent strategy execution**
- Created new portfolio.rs module with 5 default strategies
- Strategies: CDCL-Aggressive, CDCL-Stable, DPLL-Lookahead, LocalSearch, Simplify-Heavy
- Parallel execution with first-result-wins approach
- Configurable timeout per strategy
- Automatic strategy configuration (restarts, branching, preprocessing)
- Full test coverage with strategy verification
- CLI flags: --portfolio-mode, --portfolio-timeout

### Proof Checking and Verification - COMPLETED
**Resolution-based proof verification system**
- Created proof_checker.rs module with full proof validation
- Supports multiple inference rules: resolution, factoring, subsumption, tautology
- Validates proof tree structure and correctness
- UNSAT core extraction from proofs
- Simple proof format parser for easy testing
- Comprehensive test suite with valid/invalid proof cases
- CLI flags: --verify-proof, --proof-file

### Checkpointing System - COMPLETED
**Save and resume long-running solver tasks**
- Created checkpoint.rs module with full checkpoint management
- Saves solver state: learned clauses, assignments, decisions, conflicts
- Periodic checkpointing with configurable intervals
- Automatic cleanup of old checkpoints
- Progress tracking and metadata
- Resume capability from latest or specific checkpoint
- CLI flags: --checkpoint, --checkpoint-dir, --checkpoint-interval, --resume, --resume-from

### Code Organization - COMPLETED
Successfully refactored main.rs from 3,320 lines into smaller, focused modules:
- **main.rs**: 650 lines (core types, main function, orchestration)
- **format.rs**: ~1,100 lines (output formatting, statistics, colorization)
- **analysis.rs**: ~374 lines (complexity analysis, problem classification, auto-tuning)
- **interactive.rs**: ~280 lines (REPL, syntax highlighting)
- **processor.rs**: ~800 lines (file processing, parallel execution, caching)

Benefits:
- Improved code maintainability and readability
- Better separation of concerns
- Easier to test individual components
- Reduced complexity of main.rs by ~80%
- All tests passing with zero warnings

### Performance Optimization - COMPLETED
**LRU Cache Eviction Policy**
- Implemented Least Recently Used (LRU) eviction for in-memory cache
- Tracks access timestamps for all cache entries
- Automatically evicts least recently used entries when cache is full
- Maintains optimal cache hit rates for repeated queries
- Added comprehensive test coverage for LRU behavior

### User Experience - COMPLETED
**Progress Estimation with ETA**
- Enhanced progress bars with estimated time of arrival (ETA)
- Shows average time per file during processing
- Displays completion summary with total time and average
- Works for both sequential and parallel processing modes
- Format: `[elapsed] [progress bar] [files] [ETA] [current file (~Xs/file)]`

## Dependencies
- **oxiz-core**: SMT-LIB2 parser, AST
- **oxiz-solver**: Main solver API
- **oxiz-opt**: Optimization commands (optional)

## Provides
- SMT-LIB2 compliant command-line solver
- Interactive REPL with syntax highlighting
- LSP server for IDE integration
- Multiple output formats (SMT-LIB2, JSON, YAML)

---

## Command-Line Interface

- [x] Add proper argument parsing (clap)
- [x] Implement timeout option
- [x] Add verbosity levels (quiet, normal, verbose, debug, trace)
- [x] Add quiet mode
- [x] Implement output format options (SMT-LIB2, JSON, YAML)
- [x] Add color output for terminals (with --no-color flag)
- [x] Add progress indicators

## Interactive Mode

- [x] Add readline/rustyline for better line editing
- [x] Add command history (saved to ~/.oxiz_history)
- [x] Add tab completion for commands
- [x] Add syntax highlighting (keywords, operators, strings, comments, numbers)
- [x] Add multi-line input support (with automatic parenthesis balancing via validator)
- [x] Add help command

## File Handling

- [x] Support reading from stdin (when no input file provided)
- [x] Add recursive directory processing (--recursive/-R flag)
- [x] Add glob pattern support (*.smt2, etc.)
- [x] Implement parallel file processing (--parallel flag)

## Output Formatting

- [x] Structured output (JSON, YAML via --format flag)
- [x] Model pretty printing
- [x] Proof output formatting
- [x] Statistics output (--stats, --time, --memory flags)

## Diagnostics

- [x] Add timing information (--time flag)
- [x] Add memory usage statistics (--memory flag)
- [x] Add solver statistics (decisions, conflicts, etc.)
- [x] Add profiling mode (--profile flag with per-file timing, min/max/avg stats, detailed profiling data)

## Integration

- [x] Add LSP server mode for IDE integration (--lsp flag with document sync, diagnostics, hover, completion)
- [x] Add watch mode for file changes (--watch/-w flag)
- [x] Add pipe-friendly mode (via --quiet and --format flags)
- [x] Add SMT-COMP compatible output (--smtcomp flag)

## Testing

- [x] Add integration tests (13 tests covering CLI features, file I/O, formats, flags)
- [x] Add benchmark suite (6 benchmark tests)
- [x] Test with SMT-LIB benchmark suite (framework for testing with external benchmarks)

## Additional Enhancements

- [x] Configuration file support (.oxizrc in home directory or config.yaml in config dir)
- [x] Enhanced help command with comprehensive examples and documentation
- [x] Version info command with build details (version, authors, license, target, etc.)

## Recent Enhancements (2026-01-04)

### Cache Integration
- [x] Full cache integration with solver (check cache before solving, store results after)
- [x] Sequential mode: Cache automatically used for all SMT-LIB2 problems
- [x] Parallel mode: Cache disabled to avoid synchronization complexity
- [x] Removed TODO comment about cache integration

### Additional Features
- [x] Syntax validation mode (--validate-only flag)
  - Validates parenthesis matching, string literals, and basic structure
  - Fast syntax checking without solving
  - Provides line/column information for errors
- [x] Statistics export (--export-stats flag)
  - Export to CSV or JSON format based on file extension
  - Includes all solver metrics: time, memory, decisions, conflicts, etc.
  - Useful for performance tracking and analysis
- [x] SMT-LIB2 formatting (--format-smtlib flag)
  - Pretty-print and reformat SMT-LIB2 files
  - Configurable indentation width (--indent-width)
  - Proper handling of strings, comments, and nested expressions
  - Useful for code formatting and readability
- [x] Solver configuration presets (--preset flag)
  - Fast: Optimized for speed with minimal checking
  - Balanced: Good trade-off between speed and completeness
  - Thorough: Maximize completeness with proof generation
  - Minimal: Fastest with minimal processing

## Recent Enhancements (2026-01-03)

- [x] Shell completion script generation (--completions flag for bash, zsh, fish, powershell)
- [x] Enhanced LSP server with document symbol provider
- [x] LSP semantic analysis (symbol extraction for declare-const, declare-fun, define-fun)
- [x] LSP document outline support for IDE integration
- [x] DIMACS CNF format support (--dimacs, --dimacs-output, --input-format flags, auto-detect .cnf files)
- [x] Resource limit options (--memory-limit, --conflict-limit, --decision-limit)
- [x] Model minimization flag (--minimize-model)
- [x] Proof validation mode (--validate-proof)
- [x] Preprocessing and simplification (--simplify flag)
- [x] Solver strategy configuration (--strategy flag: cdcl, dpll, portfolio, local-search)
- [x] Model enumeration (--enumerate-models, --max-models flags)
- [x] Optimization objectives (--optimize flag for maximize/minimize)
- [x] Result caching system (--cache, --cache-dir flags)
- [x] Benchmark tracking and comparison (--benchmark-file flag)
- [x] Theory-specific optimizations (--theory-opt flag)
- [x] Enhanced error reporting (--enhanced-errors flag)

## Advanced Features

### DIMACS Support
- Full CNF format parser with validation
- Automatic format detection for .cnf files
- Conversion between DIMACS and SMT-LIB2
- DIMACS-compliant output format
- SMT-COMP compatible DIMACS mode

### Resource Management
- Memory limits (MB)
- Conflict limits for SAT solver
- Decision limits for search bound
- Configurable via CLI or config file

### Solver Configuration
- Multiple solving strategies (CDCL, DPLL, portfolio, local-search)
- Preprocessing and simplification options
- Model minimization for smallest satisfying assignments
- Proof generation and validation

### Model Enumeration
- Find all satisfying assignments
- Configurable maximum number of models
- Efficient enumeration with blocking clauses

### Optimization
- Maximize/minimize objectives
- MaxSMT and Pseudo-Boolean optimization
- Incremental optimization support

### Performance Optimization
- Result caching for repeated queries (fully integrated with sequential solving)
- Hash-based cache with disk persistence
- Automatic cache lookup before solving and storage after solving
- Cache disabled in parallel mode to avoid synchronization overhead
- Benchmark tracking and historical comparison
- Theory-specific optimizations (e.g., LIA fastpath, BV bitblasting)
- Enhanced error reporting with actionable suggestions

## Recent Enhancements (2026-01-04)

### Query Analysis and Intelligent Solving
- [x] Query complexity analysis (--analyze flag)
  - Analyzes SMT-LIB2 problems without solving
  - Reports declarations, assertions, nesting depth, quantifiers
  - Detects theories used (Arithmetic, BitVectors, Arrays)
  - Estimates problem difficulty and recommends solver strategy
  - Shows operator usage statistics
  - Outputs in SMT-LIB2, JSON, or YAML format

- [x] Problem classification (--classify flag)
  - Automatically classifies problem by logic (QF_LIA, QF_BV, etc.)
  - Determines complexity class (NP-complete, NP-hard, Undecidable)
  - Identifies primary theory and quantifier usage
  - Provides solver recommendations based on problem characteristics
  - Suggests timeout values based on estimated difficulty
  - Recommends theory-specific optimizations

- [x] Automatic solver tuning (--auto-tune flag)
  - Analyzes problem and applies optimal solver configuration
  - Automatically selects solver strategy (CDCL, portfolio, etc.)
  - Enables appropriate simplifications for complex problems
  - Sets timeout based on estimated difficulty
  - Applies theory-specific optimizations automatically
  - Enables proof generation for small, simple problems

- [x] Comprehensive usage examples (--examples flag)
  - Shows practical examples for all major features
  - Organized by category (basic usage, query analysis, performance, etc.)
  - Includes combined examples demonstrating feature combinations
  - Covers all CLI flags and common use cases
  - Helpful for new users learning the tool

## Recent Enhancements (2026-01-06 - Part 2)

### QDIMACS Support - COMPLETED
**Quantified Boolean Formula (QBF) format support**
- Extended DIMACS module to support QDIMACS format
- Added Quantifier enum (Universal/Existential) and QuantifierBlock structure
- Full parsing and writing of QDIMACS files with quantifier prefixes
- Automatic conversion to SMT-LIB2 with proper quantifier nesting
- Auto-detection of .qdimacs and .qcnf file extensions
- CLI flag: --input-format qdimacs
- Includes 3 comprehensive tests for parsing, writing, and conversion
- Enables solving quantified boolean formulas through SMT-LIB2 backend

### Approximate Model Counting - COMPLETED
**Statistical estimation of satisfying assignments**
- Created model_counter.rs module with counting algorithms
- Supports both exact and approximate counting methods
- Approximate counting using sampling-based estimation
- Configurable sample count for accuracy/performance trade-off
- Provides confidence intervals for estimates (95% default)
- Heuristic-based estimation considering problem structure
- CLI flags: --count-models, --count-method (exact/approximate), --count-samples, --count-export
- Includes 6 tests covering estimation, configuration, and formatting
- Useful for understanding solution space size and problem complexity

### CI/CD Integration Helpers - COMPLETED
**Machine-readable output for continuous integration pipelines**
- Created cicd.rs module with standardized reporting format
- JSON-based report format with version, status, and detailed results
- Automatic CI platform detection (GitHub Actions, GitLab CI, Jenkins, CircleCI, Travis, Buildkite)
- Summary statistics: total, SAT/UNSAT/UNKNOWN/errors counts, timing
- Platform-specific annotations for errors and warnings
- Exit codes: 0 (success), 1 (failure), 2 (error)
- CLI flags: --cicd, --cicd-report, --cicd-strict
- Includes 6 tests for report generation and formatting
- Designed for integration with automated testing and quality gates

## Recent Enhancements (2026-01-06 - Part 1)

### Dependency Analysis - COMPLETED
**Track dependencies between assertions for better problem understanding**
- Created dependency.rs module with full dependency graph analysis
- Extracts symbols from assertions and builds dependency relationships
- Finds hub symbols (highly connected) and isolated assertions
- Provides statistics on symbol usage and assertion complexity
- CLI flags: --dependencies, --dependencies-detailed, --dependencies-export
- Helps users understand the structure and relationships in their SMT problems

### Diagnostic Mode - COMPLETED
**Comprehensive problem debugging and issue detection**
- Created diagnostic.rs module with multi-level issue detection
- Checks for syntax errors (unbalanced parentheses, empty assertions)
- Detects symbol issues (undeclared, unused, duplicate declarations)
- Identifies type mismatches and common errors
- Warns about performance concerns (deep nesting, large problems)
- Detects complexity issues (non-linear arithmetic, quantifiers)
- Suggests best practices (set-logic, incremental mode, etc.)
- CLI flags: --diagnostic, --diagnostic-export
- Provides actionable suggestions for fixing issues

### Tutorial Mode - COMPLETED
**Interactive guided tutorial for learning OxiZ and SMT-LIB2**
- Created tutorial.rs module with comprehensive interactive tutorials
- Covers introduction to SMT solving and basic usage
- Explains different theories (LIA, BV, Arrays, Boolean)
- Demonstrates advanced features (optimization, cores, portfolio)
- Shows CLI options and best practices
- Interactive sections with examples and explanations
- CLI flag: --tutorial [section]
- Sections: intro, basic-usage, theories, advanced, cli-options, all
- Helps new users get started quickly

## Potential Future Enhancements

### Code Organization
- [x] Refactor main.rs (3320 lines → 650 lines) into smaller modules
  - [x] Extract formatting functions to src/format.rs (~1,100 lines)
  - [x] Extract analysis functions to src/analysis.rs (~374 lines)
  - [x] Extract interactive mode to src/interactive.rs (~280 lines)
  - [x] Extract file processing to src/processor.rs (~800 lines)

### Performance
- [x] Add incremental solving support with push/pop (--incremental flag for CLI support)
- [x] Implement parallel portfolio solving with different strategies (Completed 2026-01-05)
- [x] Add warm-start capability for related queries (via caching system)
- [x] Optimize cache with LRU eviction policy (Completed 2026-01-05)

### Analysis & Debugging
- [x] Add UNSAT core extraction and minimization (--unsat-core, --minimize-core flags)
- [x] Implement proof tree visualization (--proof-dot flag for DOT/GraphViz output)
- [x] Add model validation (--validate-model flag)
- [x] Implement proof checking and verification (Completed 2026-01-05)
- [x] Add dependency tracking between assertions (Completed 2026-01-06)
- [x] Add interpolant generation support (Completed 2026-01-17 - oxiz-cli/src/interpolate.rs)

### Advanced Features
- [x] Support for more input formats (TPTP) (Completed 2026-01-17 - oxiz-cli/src/tptp.rs)
- [x] Support for QDIMACS format (Completed 2026-01-06)
- [x] Add SMT-LIB2 v2.6 features (extended syntax) (Completed 2026-01-17 - Match expressions and parametric datatypes)
- [x] Implement model counting (approximate/exact) (Completed 2026-01-06)
- [x] Add constraint learning and sharing (Completed 2026-01-17 - oxiz-cli/src/learning.rs)
- [x] Support for distributed solving (Completed 2026-01-17 - oxiz-cli/src/distributed.rs)

### User Experience
- [x] Add progress estimation with ETA (Completed 2026-01-05)
- [x] Implement checkpointing for long-running tasks (Completed 2026-01-05)
- [x] Add diagnostic mode for problem debugging (Completed 2026-01-06)
- [x] Create tutorial mode with guided examples (Completed 2026-01-06)
- [x] Add web-based dashboard for monitoring (Completed 2026-01-17 - oxiz-cli/src/dashboard.rs)

### Integration
- [x] Python bindings (PyO3) (Completed 2026-01-17 - oxiz-py crate)
- [x] JavaScript/WASM bindings (Completed 2026-01-17 - oxiz-cli/src/wasm_bindings.rs, oxiz-cli/examples/)
- [x] REST API server mode (Completed 2026-01-17 - oxiz-cli/src/server.rs)
- [x] Integration with VS Code extension (Completed 2026-01-17 - oxiz-vscode/)
- [x] CI/CD pipeline integration helpers (Completed 2026-01-06)

## Completed

- [x] Basic file input
- [x] Basic output printing
- [x] SMT-LIB2 script execution
- [x] Interactive mode (basic)
