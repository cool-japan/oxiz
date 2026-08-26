# Z3 Parity Test Methodology

## Overview

This test suite validates OxiZ's correctness by comparing its results against Z3, the de facto standard SMT solver. The goal is to ensure OxiZ produces the same satisfiability decisions as Z3 on a representative set of benchmarks.

## Test Structure

### Benchmark Categories

- **QF_LIA** (16 benchmarks): Quantifier-Free Linear Integer Arithmetic
- **QF_LRA** (16 benchmarks): Quantifier-Free Linear Real Arithmetic
- **QF_BV** (15 benchmarks): Quantifier-Free Bit-Vectors
- **QF_S** (10 benchmarks): Quantifier-Free Strings
- **QF_FP** (12 benchmarks): Quantifier-Free Floating Point
- **QF_DT** (10 benchmarks): Quantifier-Free Datatypes
- **QF_A** (10 benchmarks): Quantifier-Free Arrays
- **QF_ABV** (5 benchmarks): Quantifier-Free Arrays + Bit-Vectors
- **QF_ALIA** (5 benchmarks): Quantifier-Free Arrays + Linear Integer Arithmetic
- **QF_AUFBV** (5 benchmarks): Quantifier-Free Arrays + Uninterpreted Functions + Bit-Vectors
- **QF_AUFLIA** (5 benchmarks): Quantifier-Free Arrays + Uninterpreted Functions + LIA
- **QF_NIA** (1 benchmark): Quantifier-Free Nonlinear Integer Arithmetic
- **QF_NIRA** (5 benchmarks): Quantifier-Free Nonlinear Mixed Integer/Real Arithmetic
- **QF_UFLIA** (5 benchmarks): Quantifier-Free Uninterpreted Functions + LIA
- **QF_UFLRA** (5 benchmarks): Quantifier-Free Uninterpreted Functions + LRA
- **AUFLIA** (10 benchmarks): Quantified Arrays + Uninterpreted Functions + LIA
- **AUFLIRA** (5 benchmarks): Quantified Arrays + Uninterpreted Functions + Mixed Arithmetic
- **UFLIA** (20 benchmarks): Quantified Uninterpreted Functions + LIA
- **UFLRA** (10 benchmarks): Quantified Uninterpreted Functions + LRA

**Total:** 170 benchmarks across 19 logic families

### Benchmark Selection Criteria

Each benchmark was selected to:

1. **Cover diverse constraint patterns** - Not just simple satisfiability checks
2. **Mix of SAT and UNSAT** - The current corpus is 117 SAT and 53 UNSAT as decided by `z3`; every case is decisive on both sides, so none of them is `Inconclusive`
3. **Varied complexity** - Easy, medium, and hard instances
4. **Execute within timeout** - Both solvers should complete within 60 seconds
5. **Test edge cases** - Boundary conditions, special values, tricky patterns

### Benchmark Sources

Benchmarks are derived from:

- SMT-LIB benchmark repository
- Z3 test suite
- Manually crafted instances targeting specific features
- Real-world verification problems (simplified)

## Running Tests

### Prerequisites

1. **Install Z3**:
   ```bash
   # macOS
   brew install z3

   # Linux
   sudo apt-get install z3

   # Or download from: https://github.com/Z3Prover/z3/releases
   ```

2. **Build OxiZ**:
   ```bash
   cd "$(git rev-parse --show-toplevel)"
   cargo build --release
   ```

### Execute Parity Tests

```bash
cd bench/z3_parity
cargo run --release
```

This will:
1. Discover all `.smt2` files in `benchmarks/`
2. Run each benchmark on both Z3 and OxiZ (in parallel)
3. Compare results
4. Generate a summary report
5. Write the detailed results twice, with identical content: to `results.json` (git-ignored
   **scratch** copy of the run you just performed, not the project's recorded evidence) and to
   this machine's tracked snapshot `results.<os>-<arch>.json`. See "Result Files" immediately
   below for which one is evidence and which one is not.

### Result Files: One Tracked Snapshot per Environment

Two files, two different jobs:

| File | Tracked? | Role |
|------|----------|------|
| `results.json` | **No** — git-ignored | Scratch output of the most recent local run *on this machine*. Every run overwrites it. Never cite it as evidence. |
| `results.<os>-<arch>.json` | **Yes** — committed | The recorded parity result **for one environment**. Currently only `results.macos-aarch64.json` is in the tree; a `results.linux-x86_64.json` would join it once a Linux environment's run is committed. This is what `README.md`, `TODO.md` and `docs/smtcomp2026_participation.md` point at when they say "authoritative". |

A run writes both files itself, so publishing a result is just a matter of committing the right
one: stage `results.<os>-<arch>.json` for the environment you actually ran on, never `results.json`
and never another environment's snapshot. A machine may only speak for itself.

#### File schema (`schema_version` 1)

```json
{
  "schema_version": 1,
  "metadata": {
    "oxiz_version": "0.3.1",
    "z3_version": "4.15.4",
    "os": "macos",
    "arch": "aarch64",
    "generated_at": "2026-07-31T11:14:13+07:00",
    "benchmark_count": 168,
    "provenance": "optional; present only on migrated files"
  },
  "results": [
    {
      "benchmark": "array_extensionality.smt2",
      "logic": "AUFLIA",
      "oxiz_result": "Sat",
      "z3_result": "Sat",
      "oxiz_time": { "secs": 0, "nanos": 1340381 },
      "z3_time": { "secs": 0, "nanos": 17799137 },
      "match_status": "Correct"
    }
  ]
}
```

`results` holds the `ParityResult` objects the runner already produced — the field set is
unchanged from the pre-schema single-file format. `metadata.benchmark_count` must equal
`results.len()`, and `metadata.os` / `metadata.arch` must match the file's own name, so a snapshot
cannot be filed under another environment's identity. `metadata.z3_version` is the version probed
from the binary at run time, and is `null` when no `z3` could be probed — never a guessed or
transcribed value on a generated file. `metadata.provenance` is absent on generated files; see
below.

#### The agreement rule

> Every tracked `results.<os>-<arch>.json` must agree on the VERDICT of every benchmark
> (`oxiz_result`, `z3_result`, `match_status`). Timings (`oxiz_time`, `z3_time`) are
> machine-dependent and are expected to differ.

This is what makes a per-logic table in `README.md` a statement about OxiZ rather than about one
laptop. It was verified by hand when the layout was introduced — 168 benchmarks × 3 verdict fields
across the macOS record and a fresh Linux run, zero mismatches, with `oxiz_time` and `z3_time` the
only fields that differed anywhere in the file — and it is now a standing check:
`tests/cross_env_verdict_agreement.rs` discovers every tracked snapshot, validates the envelope,
and fails with the offending benchmark, field and both values whenever two snapshots disagree. It
reads committed JSON only: no `z3` binary, no solving, so it runs in any environment.

#### Why per-environment files

The suite used to write a single tracked `results.json` that every document cited as *the*
authoritative result, while it actually held "whatever machine ran last". On 2026-07-31 a Linux
run overwrote macOS-recorded numbers and nothing in the file signalled it — one platform could
silently overwrite another platform's recorded evidence, and a genuine cross-platform divergence
would have been indistinguishable from a routine re-run. Splitting the record per environment
makes both the overwrite impossible and the divergence visible.

#### `provenance` on migrated files

A snapshot produced by an actual run carries no `provenance` field. `results.macos-aarch64.json`
does: it was migrated from the single tracked `results.json` as recorded at commit `540b7d0`.
Metadata that had to be **reconstructed** during that migration (the environment, the tool
versions, the timestamp) is an attribution, not a measurement — the per-benchmark verdicts and
timings are exactly as recorded, but a reconstructed metadata field must not be read as if it had
been measured at run time.

#### The z3 version is part of the evidence

`metadata.z3_version` is not decoration. The recorded baseline is **z3 4.15.4**; Ubuntu's `apt`
currently ships **4.13.3**, which is a different solver for evidence purposes. If two snapshots
disagree on a verdict and were produced against different z3 versions, the disagreement is
*unattributable*: nothing in it can be pinned on OxiZ or on z3 until both sides are re-measured
against the same z3 binary. Prefer an upstream release matching the baseline over the distro
package, and always record the `z3 --version` you actually ran.

### Interpreting Results

#### Match Status

The classification is implemented by `compare_results` in `src/comparator.rs`, which uses an
**honest comparator**: an `UNKNOWN` answer from either solver can never produce a `Correct`
verdict, so a solver cannot inflate its parity score by declining to answer.

- **Correct**: Both solvers returned the *same decisive* answer (SAT/SAT or UNSAT/UNSAT)
- **Wrong**: Disagreement on SAT/UNSAT (critical bug!)
- **Inconclusive**: Either or both solvers answered UNKNOWN — including UNKNOWN/UNKNOWN. No parity claim can be made, because the decisive answer (if any) was never cross-checked
- **Timeout**: One or both solvers exceeded the 60s timeout
- **Error**: Parse error or execution failure

**Note:** `MatchStatus::is_decisive()` counts only `Correct` and `Wrong` towards the parity
percentage. A unit test in `src/comparator.rs` asserts that no combination involving `UNKNOWN`
can ever yield `Correct`.

#### Pass Criteria

For the **v0.3.1 release**, every logic family in the suite had to be **100% Correct with 0 Wrong,
0 Inconclusive, 0 Timeout and 0 Error** — the result recorded at the time was **168/168 Correct**
across all 19 logic families against a real `z3` 4.15.4 binary. That pass criterion applies to
every tracked `results.<os>-<arch>.json` snapshot; currently only `results.macos-aarch64.json` is
in the tree. The cross-environment agreement rule — every tracked snapshot must agree on every
benchmark's verdict, with only the recorded timings differing between them — is checked by
`tests/cross_env_verdict_agreement.rs`, but with a single snapshot present it is currently vacuous
rather than exercised across environments.

Two rules are non-negotiable regardless of the headline number:

- **0 Wrong** is a hard blocker — any decisive disagreement is a soundness bug.
- Any `Inconclusive`, `Timeout` or `Error` case must be reported as such and never re-labelled
  `Correct`, so the reported figure always reflects cross-checked decisive answers only.

This is a claim about *this suite*, not a blanket "100% Z3 compatibility" statement.

## Limitations

### What This Tests

✅ **Correctness of satisfiability decisions**
✅ **Logic-specific feature coverage**
✅ **Regression prevention**
✅ **Relative performance (execution time)**

### What This Does NOT Test

❌ **Model quality** - We don't validate model values, only SAT/UNSAT
❌ **Proof generation** - OxiZ's proof output is not cross-checked here (see `oxiz-proof`)
❌ **Incremental solving** - Benchmarks are one-shot
❌ **Performance limits** - All benchmarks finish in < 60s

## Differential Testing

In addition to the curated `.smt2` corpus above, this crate ships a
**generator-based differential-testing harness** that produces small,
random, well-typed SMT-LIB2 scripts and checks that OxiZ's sat/unsat
verdict agrees with Z3's. Unlike the curated corpus, this harness is not
limited to a fixed, hand-written set of benchmarks: every run explores a
fresh (but fully reproducible) slice of each logic's formula space.

### Components

- **Generator** (`src/generator.rs`): a dependency-free, seeded PRNG
  (SplitMix64) drives recursive term/formula builders for `QF_LIA`,
  `QF_LRA`, `QF_BV`, and `QF_UF`. `generate_script(logic, seed)` is a pure
  function of its two arguments — no wall-clock time or OS entropy is ever
  consulted, so any failing case is trivially reproducible from its
  `(logic, seed)` pair alone. Arithmetic terms are kept linear (constant
  \* subterm only, never variable \* variable) so `QF_LIA`/`QF_LRA` scripts
  never drift into `QF_NIA`/`QF_NRA`.
- **Runner** (`src/difftest.rs`): `run_case`/`run_cases` generate a script,
  write it to a scratch file, execute it through both `oxiz_runner::run_oxiz`
  (library call, in-process with a timeout) and `z3_runner::run_z3` (the
  `z3` binary discovered on `PATH`), and reuse `comparator::compare_results`
  — the same comparison logic the curated benchmark suite uses — to
  classify the outcome. `Unknown` on either side is `Inconclusive` (skipped,
  no parity claim), a definite `Sat`/`Unsat` disagreement is `Wrong` (a real
  bug), and `summarize()` buckets a batch of outcomes accordingly.
- **Repro capture**: any `Wrong` outcome has its exact reproducing script
  written under `std::env::temp_dir()/oxiz_difftest_repro/<logic>_<seed>_<ts>.smt2`
  via `difftest::save_repro`, and the path is included in the panic
  message.

### Running it

Two `cargo test` entry points under `tests/`:

1. **Always-on smoke test** (`tests/difftest_smoke.rs`) — a fixed seed,
   ~25 generated cases per logic (100 total). Runs as part of a plain
   `cargo test` in this crate with **no extra flags**. Like the existing
   `z3_runner` tests, it self-skips (prints a message, does not fail) when
   no `z3` binary is found on `PATH`, so CI without Z3 is never affected.
   Whenever Z3 *is* present, this is a real regression check: an OxiZ
   change that flips a sat/unsat verdict on any of the 100 fixed cases
   fails the test.
2. **Full sweep, opt-in** (`tests/difftest_full.rs`) — gated behind an
   environment variable so it never runs by accident:

   ```bash
   OXIZ_DIFFTEST=1 cargo test --test difftest_full -- --nocapture
   ```

   Tunable via:
   - `OXIZ_DIFFTEST_CASES` (default `200`) — cases generated per logic.
   - `OXIZ_DIFFTEST_SEED` (default `42`) — base PRNG seed (each logic
     derives its own seed from this so the four sweeps don't replay
     correlated streams).

   Also self-skips when no `z3` binary is present, or when
   `OXIZ_DIFFTEST` is unset/not `1`.

### Scope and honesty notes

- This harness targets **sat/unsat verdict parity only** (same contract as
  the curated benchmark suite above) — it does not validate model values or
  proofs.
- A `Wrong` verdict is the only thing that fails a differential test;
  `Unknown`/`Timeout`/solver `Error` on either side is reported but treated
  as inconclusive, never as a pass *or* a hard failure, so the harness can
  never be gamed by a solver that just gives up.
- There is intentionally no new CI workflow wired to this harness (Z3 is an
  external binary dependency this project does not control, and the
  project's CI-workflow policy restricts which `.github/workflows/*.yml`
  files may be added). Run it manually, or in any environment that happens
  to already have `z3` on `PATH`.

## Adding New Benchmarks

To add a new benchmark:

1. Create a `.smt2` file in the appropriate logic directory
2. Include a comment at the top documenting:
   - What feature/pattern it tests
   - Expected result (sat/unsat/unknown)
   - Source (if adapted from elsewhere)

Example:
```smt2
; Test: Branch and bound with negative coefficients
; Expected: unsat
; Source: Adapted from SMT-LIB QF_LIA/20230215-Barrett

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (+ (* -2 x) (* 3 y)) 7))
(assert (>= x 0))
(assert (<= x 5))
(assert (>= y 0))
(assert (<= y 2))

(check-sat)
```

3. Run the parity suite to validate
4. Commit the benchmark together with **your own environment's** `results.<os>-<arch>.json`
   (see "Result Files" above). Do **not** commit `results.json` — it is git-ignored scratch — and
   do **not** overwrite another environment's snapshot with your run's numbers: adding a benchmark
   means every other tracked snapshot is now missing it, which
   `tests/cross_env_verdict_agreement.rs` will report by name. Ask a maintainer on the other
   platform to re-run and commit theirs, and note in the pull request which environments are still
   pending, rather than making one machine speak for all of them.

## Troubleshooting

### "Z3 not found"

Ensure Z3 is installed and in your PATH:
```bash
which z3  # Should print /usr/local/bin/z3 or similar
z3 --version  # Should print version info
```

### All OxiZ tests showing "Error"

Check that OxiZ builds successfully:
```bash
cd ../../oxiz
cargo test
```

### Timeout issues

If benchmarks are timing out, increase the timeout in `z3_runner.rs` and `oxiz_runner.rs`:
```rust
const Z3_TIMEOUT_SECS: u64 = 120;  // Increase from 60
```

## Maintenance

This test suite should be run:

- **Before every release** - Blocking requirement
- **Weekly (CI)** - Automated regression testing
- **After major changes** - Especially to core theories
- **When adding new features** - Ensure no regressions

## Future Enhancements

- [ ] Model validation (check SAT models satisfy constraints)
- [ ] UNSAT core comparison
- [ ] Incremental solving tests (push/pop)
- [ ] Performance benchmarking (not just correctness)
- [ ] Fuzz-generated benchmarks
- [x] Quantified formula tests (`AUFLIA`, `AUFLIRA`, `UFLIA`, `UFLRA` are part of the suite)
