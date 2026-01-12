# SCIRS2 Policy for oxiz-cli

## Policy Statement

This crate (`oxiz-cli`) is a **command-line interface** for the OxiZ SMT solver and does **NOT** require SciRS2 dependencies.

## Rationale

### Nature of the Project
- **oxiz-cli** implements a command-line interface and REPL for SMT solving
- Primary focus: user interaction, file I/O, output formatting, terminal UI
- Domain: developer tools, command-line utilities

### Dependency Analysis

The project uses:
- `clap`: CLI argument parsing
- `rustyline`: Interactive line editing with history
- `owo-colors`: Terminal color output
- `serde`, `serde_json`, `serde_yaml`: Data serialization for structured output
- `indicatif`: Progress bars for long operations
- `notify`: File system watching
- `walkdir`, `globset`: File system traversal and pattern matching
- `rayon`: Parallel file processing
- `sysinfo`: System memory statistics
- `dirs`: Platform-specific directory paths

**NO** usage of:
- `rand` or `rand_distr` (no random sampling needed)
- `ndarray` (no multi-dimensional array computations)
- **NO** statistical or scientific computing requirements

### When SciRS2 Would Be Required

SciRS2 (and projects like NumRS2, ToRSh, etc.) would be required if this project:
- Performed statistical analysis of solver performance
- Implemented machine learning-based heuristics
- Required numerical optimization for parameter tuning
- Performed scientific simulations or data analysis
- Needed multi-dimensional array operations for benchmarking

### Current Status

✅ **COMPLIANT**: This project does not need to use SciRS2 as it does not perform scientific/statistical computing.

The project is a pure **CLI utility** focusing on:
- User interaction and command parsing
- File I/O and format conversion
- Terminal output formatting and coloring
- Progress tracking and statistics display
- Interactive REPL with syntax highlighting

All dependencies are appropriate for a command-line interface tool.

## Review Date

Last reviewed: 2025-12-28

---

**Note**: If future features require statistical analysis, machine learning, or numerical optimization (e.g., adaptive timeout prediction, performance modeling), this policy should be updated to use SciRS2-Core instead of rand/ndarray.
