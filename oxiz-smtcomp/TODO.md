# oxiz-smtcomp TODO

## Completed
- [x] Basic directory structure and Cargo.toml
- [x] Benchmark file discovery (loader.rs)
- [x] Benchmark runner with timeout (benchmark.rs)
- [x] SMT-COMP format reporting (reporter.rs)
- [x] Statistical analysis (statistics.rs)
- [x] Public API and module exports (lib.rs)
- [x] Parallel benchmark execution (parallel.rs - rayon integration)
- [x] Memory limit enforcement (memory.rs)
- [x] Model verification for SAT results (model_verify.rs)
- [x] StarExec format compatibility (starexec.rs)
- [x] Cactus plot generation (plotting.rs - SVG output)
- [x] HTML report generation (html_report.rs)
- [x] Incremental result saving/resumption (resumption.rs)
- [x] Benchmark filtering by expected status (filtering.rs)
- [x] Virtual best solver calculation (virtual_best.rs)
- [x] Integration with CI/CD for regression testing (ci_integration.rs)
- [x] Benchmark subset selection (sampling.rs - representative sampling)
- [x] Performance regression detection (regression.rs)
- [x] Web dashboard for result visualization (dashboard.rs - HTML/JS)
- [x] Integration testing with real benchmark suites (tests/integration.rs)

## Future Enhancements

### Performance
- [ ] GPU acceleration for parallel solving
- [ ] Distributed execution across multiple machines
- [ ] Caching of parsing results

### Analysis
- [ ] Machine learning-based difficulty prediction
- [ ] Automatic logic detection from formulas
- [ ] Benchmark classification by structure

### Visualization
- [ ] Interactive web-based result explorer
- [ ] Real-time progress monitoring WebSocket API
- [ ] PDF report generation

### Integration
- [ ] SMT-LIB 3.0 support when available
- [ ] Integration with other benchmark suites (SV-COMP, etc.)
- [ ] Docker container generation for reproducible runs

## Module Summary

| Module | Description | Status |
|--------|-------------|--------|
| benchmark.rs | Core runner with timeout | Complete |
| loader.rs | Benchmark discovery | Complete |
| reporter.rs | Result reporting (JSON/CSV/Text) | Complete |
| statistics.rs | Statistical analysis | Complete |
| parallel.rs | Parallel execution (rayon) | Complete |
| memory.rs | Memory limit enforcement | Complete |
| model_verify.rs | Model verification | Complete |
| starexec.rs | StarExec compatibility | Complete |
| plotting.rs | SVG plot generation | Complete |
| html_report.rs | HTML reports | Complete |
| resumption.rs | Incremental saving | Complete |
| filtering.rs | Benchmark filtering | Complete |
| virtual_best.rs | VBS calculation | Complete |
| ci_integration.rs | CI/CD support | Complete |
| sampling.rs | Representative sampling | Complete |
| regression.rs | Regression detection | Complete |
| dashboard.rs | Web dashboard | Complete |

## Dependencies
- `rayon` for parallel execution
- `rand` for sampling
- `serde` for serialization
- `thiserror` for error handling
- `tracing` for logging
