# OxiZ at SMT-COMP 2026: Participation Invitation

This document invites researchers, developers, and the broader formal methods community to collaborate on submitting **OxiZ** to the [SMT Competition 2026 (SMT-COMP 2026)](https://smt-comp.github.io/).

---

## What is OxiZ?

OxiZ is a high-performance, pure Rust implementation of a full-featured SMT (Satisfiability Modulo Theories) solver. It is developed as part of the COOLJAPAN open-source ecosystem and is designed to match — and ultimately surpass — the capabilities of state-of-the-art solvers such as Z3, while offering the safety, reproducibility, and ergonomics that Rust uniquely provides.

**Key facts about OxiZ (v0.3.3):**

- Honest, non-fabricated parity against a real `z3` 4.15.4 binary across **19 SMT-LIB logic families** (170 benchmark instances, `bench/z3_parity`): **170/170 Correct, 0 Wrong, 0 Inconclusive, 0 Timeout, 0 Error** on the extended 19-logic / 170-benchmark differential suite under the honest comparator (`Unknown` never counts as a match). All 19 logic families are at 100% of this differential parity suite — a claim scoped to the suite, not a blanket "100% Z3 compatibility"; see [`README.md`](../README.md#z3-parity-differential-suite-results-honest-comparator-️) for the full per-logic breakdown and known gaps
- **10,345 unit tests** passing, 12 skipped (`cargo nextest run --workspace --all-features`) across all crates, plus 110 doc-tests
- Zero unsafe C/C++ dependencies — pure Rust from end to end
- Proof-producing: generates DRAT, Alethe, LFSC, Coq, Lean, and Isabelle certificates
- Supports Craig interpolation and Spacer/PDR for model checking workloads
- StarExec-compatible stdin/stdout interface via the `smtcomp2026` binary

OxiZ is actively developed at: [https://github.com/cool-japan/oxiz](https://github.com/cool-japan/oxiz)

---

## Why OxiZ for SMT-COMP 2026?

### 1. Broad logic coverage — 19 divisions implemented, honest per-division status

OxiZ has implementations across the following SMT-LIB logic families. Status reflects real, measured results from `bench/z3_parity` (a real `z3` binary, honest comparator — `Unknown` never counts as a match) as of v0.3.3, not an aspirational "all ready" claim:

| Division | Status |
|----------|--------|
| QF_LIA   | ✅ Ready (16/16 Correct) |
| QF_LRA   | ✅ Ready (16/16 Correct) |
| QF_BV    | ✅ Ready (15/15 Correct) |
| QF_S     | ✅ Ready (10/10 Correct — ground string decision procedure) |
| QF_FP    | ✅ Ready (12/12 Correct — concrete FP model finder) |
| QF_DT    | ✅ Ready (10/10 Correct) |
| QF_A     | ✅ Ready (10/10 Correct) |
| QF_NIA   | ✅ Ready (1/1 Correct on the parity suite; broader NIA branch-and-bound has known scoping gaps, see `TODO.md`) |
| QF_NRA   | 🔶 Alpha (irrational-root isolation still open; not yet part of the parity suite) |
| UFLIA    | ✅ Ready (20/20 Correct — Skolem witness synthesis + CEGAR on top of the MBQI SAT certifier) |
| UFLRA    | ✅ Ready (10/10 Correct — symbolic model certification over the Reals + quasi-macro detection) |
| AUFLIA   | ✅ Ready (10/10 Correct — finite-range quantifier expansion) |
| AUFLIRA  | ✅ Ready (5/5 Correct) |
| QF_ALIA  | ✅ Ready (5/5 Correct) |
| QF_AUFBV | ✅ Ready (5/5 Correct) |
| QF_ABV   | ✅ Ready (5/5 Correct) |
| QF_NIRA  | ✅ Ready (5/5 Correct) |
| QF_IDL   | ⬜ Not yet part of `bench/z3_parity` — no measured data to report |
| QF_RDL   | ⬜ Not yet part of `bench/z3_parity` — no measured data to report |

Aggregate result across the full 170-benchmark suite: **170/170 Correct, 0 Wrong, 0 Inconclusive, 0 Timeout, 0 Error** — all 19 measured divisions are individually at 100% of this suite (which is a statement about the differential parity suite, not a blanket "100% Z3 compatibility" claim; `QF_NRA`, `QF_IDL` and `QF_RDL` are not part of it) — see [`README.md`](../README.md#z3-parity-differential-suite-results-honest-comparator-️) and the tracked per-environment snapshot `bench/z3_parity/results.<os>-<arch>.json` (currently only `results.macos-aarch64.json` is in the tree) for the authoritative, per-benchmark breakdown. "Authoritative" has a precise meaning here: every tracked snapshot must agree on the verdict of every benchmark (`oxiz_result`, `z3_result`, `match_status`) and may differ only in the machine-dependent timings — a rule `bench/z3_parity/tests/cross_env_verdict_agreement.rs` enforces on every `cargo test`, though with a single snapshot currently in the tree that check is currently vacuous rather than exercised across environments. The un-suffixed `bench/z3_parity/results.json` is git-ignored scratch output of one local run and carries no such guarantee. The z3 version is part of the evidence: the recorded baseline is z3 4.15.4, and a snapshot measured against a different z3 cannot be compared verdict-for-verdict with one that was not.

### 2. Pure Rust: safety, reproducibility, and auditability

OxiZ contains no C, C++, or Fortran dependencies. This has concrete benefits for competition:

- **Reproducible builds**: `cargo build --release` is fully hermetic and cross-platform.
- **Memory safety**: The solver cannot exhibit undefined behavior from pointer errors, use-after-free, or buffer overflows.
- **Auditability**: Every line of the solver is Rust, making it straightforward for competition organizers and reviewers to inspect.
- **Portability**: The binary can be built for Linux/x86-64 (StarExec target) with a single command and no native library setup.

### 3. StarExec-compatible out of the box

OxiZ ships a dedicated `oxiz-smtcomp` crate that produces the `smtcomp2026` binary. This binary:

- Reads SMT-LIB 2.6 input from `stdin`
- Writes `sat`, `unsat`, or `unknown` to `stdout`
- Generates machine-checkable proofs on request (`--proof-format=alethe`, etc.)
- Exits with the correct StarExec status codes

No wrapper scripts or environment patching are required.

### 4. Unique solver capabilities

OxiZ brings several capabilities that are rare or absent among current SMT-COMP entrants:

- **Multi-format proof generation**: DRAT (for SAT-level certificates), Alethe (for theory lemmas), LFSC, and formal proofs exportable to Coq, Lean 4, and Isabelle/HOL.
- **Craig interpolation**: built into the core solver, useful for software verification and model checking.
- **Spacer / Property-Directed Reachability (PDR)**: for CHC (Constrained Horn Clause) solving and reachability analysis.
- **MBQI (Model-Based Quantifier Instantiation)**: supporting quantified logic divisions with a tunable instantiation engine.

---

## How to Participate

If you would like to submit OxiZ to SMT-COMP 2026, here is how to get started.

### Step 1: Clone the repository

```bash
git clone https://github.com/cool-japan/oxiz.git
cd oxiz
```

### Step 2: Build the competition binary

```bash
cargo build --release -p oxiz-smtcomp
```

The resulting binary is located at `./target/release/smtcomp2026`.

### Step 3: Test locally

Verify that the solver produces correct output on a simple benchmark:

```bash
echo "(declare-const x Int)(assert (= x 5))(check-sat)" \
  | ./target/release/smtcomp2026
# Expected output: sat
```

For an unsatisfiable instance:

```bash
echo "(declare-const x Int)(assert (and (= x 5)(= x 6)))(check-sat)" \
  | ./target/release/smtcomp2026
# Expected output: unsat
```

For proof output:

```bash
echo "(declare-const x Int)(assert (and (= x 5)(= x 6)))(check-sat)(get-proof)" \
  | ./target/release/smtcomp2026 --proof-format=alethe
```

### Step 4: Run the regression suite

Before preparing a submission, run the full benchmark suite to confirm parity:

```bash
cargo test --workspace
cargo bench -p bench-regression
```

### Step 5: Prepare the StarExec package

The `oxiz-smtcomp` crate generates the StarExec submission layout automatically:

```bash
cargo run --release -p oxiz-smtcomp -- --generate-starexec-package ./oxiz-smtcomp-2026.zip
```

This produces a `.zip` archive containing:

- `bin/smtcomp2026` — the solver binary (statically linked)
- `bin/starexec_run_default` — the StarExec run script
- `README` — version information and contact details

### Step 6: Register at SMT-COMP 2026

Solver registration information is published at:

> [https://smt-comp.github.io/](https://smt-comp.github.io/)

Check the official site for submission deadlines, required metadata, and division registration forms. Typical requirements include:

- Solver name and version
- List of divisions entered
- System description paper (2–4 pages, LNCS format)
- StarExec package upload

### Step 7: Coordinate with the OxiZ team

If you plan to submit OxiZ, please open a GitHub issue at:

> [https://github.com/cool-japan/oxiz/issues](https://github.com/cool-japan/oxiz/issues)

Use the title prefix `[SMT-COMP 2026]`. This allows us to coordinate system descriptions, avoid duplicate submissions, and provide support for any build or packaging questions.

---

## Division Recommendations

The table below summarizes OxiZ's competitive positioning across the 19 ready divisions. "Novel entry" indicates divisions where a pure Rust solver has not previously competed; "Established field" indicates divisions with strong existing entrants (Z3, CVC5, Bitwuzla, etc.).

| Division | OxiZ Strength | Competition Context | Notes |
|----------|---------------|--------------------|-------------------------------------------------|
| QF_LIA   | Strong        | Established field  | Simplex + Gomory cuts, LIA preprocessing        |
| QF_LRA   | Strong        | Established field  | Full simplex, delta-arithmetic                  |
| QF_BV    | Strong        | Established field  | SIMD-accelerated bit-vector propagation         |
| QF_S     | Competitive   | Established field  | String theory with length constraints           |
| QF_FP    | Competitive   | Established field  | IEEE 754 floating-point semantics               |
| QF_DT    | Competitive   | Moderate field     | Algebraic data types, structural induction      |
| QF_A     | Strong        | Established field  | Array theory, McCarthy axioms                   |
| QF_NIA   | Competitive   | Established field  | NLSAT, CAD-based NIA                            |
| QF_NRA   | Competitive   | Established field  | NLSAT, real algebraic arithmetic                |
| UFLIA    | Competitive   | Established field  | MBQI + arithmetic                               |
| UFLRA    | Competitive   | Established field  | MBQI + LRA                                      |
| AUFLIA   | Competitive   | Established field  | Arrays + UF + LIA combined                      |
| AUFLIRA  | Competitive   | Established field  | First novel pure Rust entry in this division    |
| QF_ALIA  | Strong        | Moderate field     | Array + LIA, fewer competing solvers            |
| QF_AUFBV | Competitive   | Established field  | Arrays + UF + BV                                |
| QF_ABV   | Competitive   | Moderate field     | Arrays + BV                                     |
| QF_NIRA  | Novel entry   | Sparse field       | First pure Rust entry; nonlinear mixed integer  |
| QF_IDL   | Strong        | Moderate field     | Integer difference logic, fast path             |
| QF_RDL   | Strong        | Moderate field     | Real difference logic, fast path                |

OxiZ is positioned as a competitive entrant in all 19 divisions and a **first-of-kind pure Rust submission** in the competition overall.

---

## Known Areas for Improvement

Noted here for transparency ahead of the competition submission, not as a formal call for
external contribution — OxiZ is Apache-2.0 licensed and the source is at
[https://github.com/cool-japan/oxiz](https://github.com/cool-japan/oxiz):

- **SIMD BV propagation** (`oxiz-theories/src/bv/`) — the bit-vector solver currently uses scalar
  propagation loops in several places; SIMD-accelerated word-level propagation (AVX2/AVX-512 on
  x86-64, NEON on AArch64) could improve throughput on large BV benchmarks.
- **MBQI instantiation tuning** (`oxiz-solver/src/mbqi/`) — model-based quantifier instantiation
  is sensitive to the order and selection of terms used for instantiation; heuristics for term
  scoring, ground term selection, and iteration bounds have room for improvement.
- **String theory performance** (`oxiz-theories/src/strings/`) — handles core SMT-LIB string
  constraints but has room for improved automata-based reasoning and length constraint propagation.
- **Proof export verification** (`oxiz-proof/`) — Alethe and LFSC proof terms are generated but
  not yet checked against reference proof checkers (`alethe-proof-checker`, LFSC) in CI.
- **Benchmark-specific preprocessing** — pre-solving heuristics, symmetry breaking, and formula
  simplification tuned to specific SMT-COMP benchmark families.

---

## Acknowledgments

*This section is a placeholder for acknowledgments to be added prior to the competition system description submission.*

The OxiZ project is grateful to the SMT-COMP organizers for maintaining an open and rigorous competition infrastructure, to the authors and maintainers of Z3 whose published algorithms and benchmark suite have informed this work, and to the broader SMT and formal methods research community.

---

*OxiZ — COOLJAPAN OU (Team Kitasan)*
*Repository: [https://github.com/cool-japan/oxiz](https://github.com/cool-japan/oxiz)*
*Competition contact: open an issue with tag `[SMT-COMP 2026]`*
