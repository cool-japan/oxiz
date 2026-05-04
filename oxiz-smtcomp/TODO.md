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
- [x] Caching of parsing results (planned 2026-04-15, completed 2026-04-15)
  - **Goal:** Add a parse-result cache in the loader/benchmark pipeline so repeated loads of the same SMT-LIB file skip parsing. Correctness preserved via mtime + size keys.
  - **Design:** Introduce `ParseCache` wrapping `FxHashMap<(PathBuf, SystemTime, u64), Arc<ParsedBenchmark>>` behind `Mutex`. Expose via `Loader::with_cache(cache)`. On `load_benchmark`, check cache; on miss, parse and insert. Bounded LRU (N=1024 default, configurable). Subagent to verify the exact return type of `loader.rs::load_benchmark` and adapt.
  - **Files:** `oxiz-smtcomp/src/loader.rs`, `oxiz-smtcomp/src/lib.rs` (re-export).
  - **Tests:** `test_parse_cache_hit`, `test_parse_cache_miss_on_file_mutation`, `test_parse_cache_lru_eviction`. Use `std::env::temp_dir()` for fixtures.
  - **Risk:** mtime-granularity false positives on fast re-writes; mitigation: include file size as secondary key. Verify actual `loader.rs` API before committing to the constructor shape.

### Analysis
- [x] Machine learning-based difficulty prediction (planned 2026-05-04, completed 2026-05-05)
  - **Goal:** Pure-Rust ML difficulty predictor that, given a parsed benchmark's `BenchmarkMeta` (logic + `StructuralFeatures` + file size), predicts (a) expected runtime in seconds and (b) a difficulty class from `{Trivial, Easy, Medium, Hard, VeryHard}`. Three independent model implementations (linear regression, k-NN, regression tree), all train from a `Dataset` of historical `SingleResult` rows, all serialize to JSON, all expose the same `DifficultyModel` trait. The `ParallelRunner` gains a new `run_from_meta_with_predictor` method that performs **Longest Processing Time first (LPT) scheduling** — sorting tasks by predicted runtime descending before dispatch — to reduce makespan on heterogeneous batches. Existing `run_from_meta` / `run_all` paths are unchanged.
  - **Files:** `oxiz-smtcomp/src/predictor/{mod,features,dataset,models,linear,knn,tree,class,persistence,report}.rs` (new); `oxiz-smtcomp/src/lib.rs`, `oxiz-smtcomp/src/parallel.rs` (modified); 6 new integration test files.
  - **Tests:** `predictor_features`, `predictor_linear`, `predictor_knn`, `predictor_tree`, `predictor_persistence`, `predictor_scheduler` test suites; all 0 warnings, all pass.
- [x] Automatic logic detection from formulas (planned 2026-04-15, completed 2026-04-15)
  - **Goal:** When a benchmark has no `(set-logic)` header, walk the parsed asserted terms, identify theory features (LIA, LRA, BV, UF, Arrays, Strings, FP, DT, NIA/NRA, quantifiers), and emit a matching SMT-LIB logic string.
  - **Design:** New `oxiz-smtcomp/src/logic_detector.rs` with `LogicDetector` visitor collecting `TheoryBits`. Bits→logic table (e.g., `UF | LIA → "UFLIA"`, `ARR | BV → "QF_ABV"`), fallback `"ALL"` if unrepresentable. Call path integrated into `loader.rs` when logic unknown. Subagent to verify what AST type is available in the loader path (term AST vs. raw text) and pick the correct layer.
  - **Files:** `oxiz-smtcomp/src/logic_detector.rs` (new), `oxiz-smtcomp/src/lib.rs`, `oxiz-smtcomp/src/loader.rs`.
  - **Tests:** `test_detect_qf_lia`, `test_detect_uflia`, `test_detect_qf_aufbv`, `test_fallback_all`. In-memory benchmark fixtures; no file I/O.
  - **Risk:** Detection layer — if the loader only sees raw source and defers parsing to oxiz-core, detection must sit in the solver layer or run a lightweight parse. Verify before coding; mark `oversized` and re-plan if detection requires cross-crate refactoring.
- [x] Benchmark classification by structure

### Visualization
- [x] Interactive web-based result explorer (planned 2026-04-19)
  - **Goal:** Extend `oxiz-smtcomp::dashboard::DashboardGenerator` so the generated HTML page has client-side interactivity: row search/filter input, per-column sort, click-to-filter on the cactus + per-logic bar charts, per-row detail expander, and optional WebSocket live updates. All existing static cards/charts remain — changes are purely additive.
  - **Design:** All interactivity is client-side JS embedded as `r#"..."#` raw-string constants inside the existing `format!`-based HTML generator — no new crate deps. Existing data is already inlined as JSON in a `<script>` block. Add: (1) `<input id="oxiz-search">` above recent-results table with JS substring filter on name/logic; (2) `data-sort-key` headers with toggle-asc/desc JS sort; (3) Chart.js click handlers on cactus/bar segments to filter table by logic+status; (4) click-to-expand row detail showing full `ResultData` JSON; (5) `DashboardConfig.ws_url: Option<String>` — if `Some(url)`, JS opens `WebSocket(url)` and on each `message` re-renders table rows. `ws_url: None` keeps existing behaviour exactly.
  - **Files:** `oxiz-smtcomp/src/dashboard.rs` (extend `DashboardConfig`; embed JS raw strings; no new crate dep). New `oxiz-smtcomp/tests/dashboard_interactive.rs`.
  - **Tests:** (a) `test_dashboard_renders_search_input` — rendered HTML contains `<input id="oxiz-search"`; (b) `test_dashboard_renders_sort_handlers` — contains `data-sort-key=`; (c) `test_dashboard_includes_ws_when_configured` — `ws_url: Some("ws://localhost:8080")` produces `new WebSocket("ws://localhost:8080")` in output; (d) `test_dashboard_omits_ws_when_unconfigured` — default `ws_url: None` does not contain `WebSocket`; (e) `test_dashboard_html_well_formed` — `<html>` and `</html>` are present and `<script>` count matches expected.
  - **Risk:** Rust `format!` escapes colliding with embedded JS curly braces. Mitigation: embed JS as `const JS_BLOCK: &str = r#"..."#` constants and concatenate them rather than inlining inside `format!`. Test (e) catches drift.
  - **Scope cap:** ≤500 LoC net-new across `dashboard.rs` + new test file. No JS bundler, no WebAssembly client, no Tailwind/Bootstrap CDN.
- [x] Real-time progress monitoring WebSocket API
- [x] PDF report generation (planned 2026-04-15, completed 2026-04-15)
  - **Goal:** Emit a PDF summary (per-logic stats + per-benchmark table + totals) analogous to `html_report.rs`. Pure Rust. Feature-gated as `pdf-report` so default builds stay lean.
  - **Design:** New `oxiz-smtcomp/src/pdf_report.rs` mirroring the `HtmlReport` API (`PdfReport::from_runs(...)` → `write(path)`). Use a latest Pure Rust PDF crate; subagent picks (`printpdf` or `pdf-writer`) after checking crates.io latest versions. Simple tabular layout, default font only.
  - **Files:** `oxiz-smtcomp/src/pdf_report.rs` (new), `oxiz-smtcomp/Cargo.toml` (feature + dep via workspace), root `Cargo.toml` (workspace dep entry), `oxiz-smtcomp/src/lib.rs` (feature-gated re-export).
  - **Tests:** `test_pdf_report_minimal`, `test_pdf_report_multi_logic` — write PDF under `std::env::temp_dir()` and assert `%PDF-` magic + plausible file size. Feature-gated via `#[cfg(feature = "pdf-report")]`.
  - **Risk:** Root `Cargo.toml` is already dirty from the v0.2.1 bump — adding workspace deps is additive and won't conflict, but notifies the user. Scope creep from custom layouts; mitigation: plain table only.

### Integration
- [ ] SMT-LIB 3.0 support when available
- [x] Integration with other benchmark suites (SV-COMP, etc.)
- [x] Docker container generation for reproducible runs (planned 2026-04-15, completed 2026-04-15)
  - **Goal:** Provide a reproducible Docker image for `smtcomp2026` runs. Builder stage compiles `--release --bin smtcomp2026`; runtime stage is a slim image with only the binary and a mount point for benchmark dirs.
  - **Design:** Multi-stage Dockerfile at `oxiz-smtcomp/Dockerfile`. `.dockerignore` excludes target/, .git/. Entrypoint script at `oxiz-smtcomp/docker/entrypoint.sh` forwards args to the binary. Rust toolchain version pinned via build arg (default: current stable).
  - **Files:** `oxiz-smtcomp/Dockerfile` (new), `oxiz-smtcomp/.dockerignore` (new), `oxiz-smtcomp/docker/entrypoint.sh` (new).
  - **Tests:** No cargo tests — Dockerfile isn't Rust code. Acceptance = the Dockerfile syntax lints cleanly (`docker buildx build --check .` if available); otherwise subagent notes the manual-verification caveat in its report.
  - **Risk:** No automated build in CI here; caveat noted. Docker isn't invoked inside the /ultra run — only authored. User runs `docker build` themselves to verify.

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

## Current Status (v0.2.0)

| Metric | Value |
|--------|-------|
| Version | 0.2.0 |
| Status | Alpha |
| Tests | 104 passing |
| Rust LoC | 9,744 (20 files) |
| Public API items | 370 |
| `todo!`/`unimplemented!` | 0 |

*Last updated: 2026-03-28*

## Dependencies
- `rayon` for parallel execution
- `serde` for serialization
- `thiserror` for error handling
- `tracing` for logging
