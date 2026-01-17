# oxiz-smtcomp TODO

## Completed
- [x] Basic directory structure and Cargo.toml
- [x] Benchmark file discovery (loader.rs)
- [x] Benchmark runner with timeout (benchmark.rs)
- [x] SMT-COMP format reporting (reporter.rs)
- [x] Statistical analysis (statistics.rs)
- [x] Public API and module exports (lib.rs)

## In Progress
- [ ] Integration testing with real benchmark suites

## Future Work

### High Priority
- [ ] Parallel benchmark execution (rayon integration)
- [ ] Memory limit enforcement
- [ ] Model verification for SAT results
- [ ] StarExec format compatibility

### Medium Priority
- [ ] Cactus plot generation (SVG/PNG output)
- [ ] HTML report generation
- [ ] Incremental result saving/resumption
- [ ] Benchmark filtering by expected status
- [ ] Virtual best solver calculation

### Low Priority
- [ ] Integration with CI/CD for regression testing
- [ ] Benchmark subset selection (representative sampling)
- [ ] Performance regression detection
- [ ] Web dashboard for result visualization

## Known Limitations
- Currently single-threaded execution only
- Memory limits not enforced (only tracked if available)
- No incremental solving benchmark support yet

## Dependencies to Consider
- `indicatif` for progress bars during long runs
- `rayon` for parallel execution
- `plotters` for chart generation
