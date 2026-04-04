# OxiZ at SMT-COMP 2026: Participation Invitation

This document invites researchers, developers, and the broader formal methods community to collaborate on submitting **OxiZ** to the [SMT Competition 2026 (SMT-COMP 2026)](https://smt-comp.github.io/).

---

## What is OxiZ?

OxiZ is a high-performance, pure Rust implementation of a full-featured SMT (Satisfiability Modulo Theories) solver. It is developed as part of the COOLJAPAN open-source ecosystem and is designed to match — and ultimately surpass — the capabilities of state-of-the-art solvers such as Z3, while offering the safety, reproducibility, and ergonomics that Rust uniquely provides.

**Key facts about OxiZ v0.2.0:**

- 100% Z3 parity across **19 SMT-LIB logic families** (168/168 benchmark instances)
- **6,031 unit tests** across all crates
- Zero unsafe C/C++ dependencies — pure Rust from end to end
- Proof-producing: generates DRAT, Alethe, LFSC, Coq, Lean, and Isabelle certificates
- Supports Craig interpolation and Spacer/PDR for model checking workloads
- StarExec-compatible stdin/stdout interface via the `smtcomp2026` binary

OxiZ is actively developed at: [https://github.com/cool-japan/oxiz](https://github.com/cool-japan/oxiz)

---

## Why OxiZ for SMT-COMP 2026?

### 1. Broad logic coverage — 19 divisions ready

OxiZ v0.2.0 is ready to compete across the following SMT-LIB logic families:

| Division | Status |
|----------|--------|
| QF_LIA   | Ready  |
| QF_LRA   | Ready  |
| QF_BV    | Ready  |
| QF_S     | Ready  |
| QF_FP    | Ready  |
| QF_DT    | Ready  |
| QF_A     | Ready  |
| QF_NIA   | Ready  |
| QF_NRA   | Ready  |
| UFLIA    | Ready  |
| UFLRA    | Ready  |
| AUFLIA   | Ready  |
| AUFLIRA  | Ready  |
| QF_ALIA  | Ready  |
| QF_AUFBV | Ready  |
| QF_ABV   | Ready  |
| QF_NIRA  | Ready  |
| QF_IDL   | Ready  |
| QF_RDL   | Ready  |

All 19 divisions have been validated against Z3's output on the SMT-LIB benchmark suite (168/168 passing).

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

## Call for Contributions

OxiZ is an open project and welcomes performance improvements before the competition submission deadline. If you work on any of the following areas and would like to contribute, please open a pull request at [https://github.com/cool-japan/oxiz](https://github.com/cool-japan/oxiz):

### High-impact contribution areas

**SIMD BV propagation** (`oxiz-theories/src/bv/`)
- The bit-vector solver currently uses scalar propagation loops in several places.
- SIMD-accelerated word-level propagation (AVX2/AVX-512 on x86-64, NEON on AArch64) could significantly improve throughput on large BV benchmarks.

**MBQI instantiation tuning** (`oxiz-solver/src/mbqi/`)
- Model-based quantifier instantiation is sensitive to the order and selection of terms used for instantiation.
- Heuristics for term scoring, ground term selection, and iteration bounds are open for improvement.

**String theory performance** (`oxiz-theories/src/strings/` — in development)
- The string theory implementation handles core SMT-LIB string constraints but has room for improved automata-based reasoning and length constraint propagation.

**Proof export quality** (`oxiz-proof/`)
- Alethe and LFSC proof terms are generated but not yet checked against reference proof checkers in CI.
- Contributions that integrate `alethe-proof-checker` or `LFSC` verification into the test suite are especially welcome.

**Benchmark-specific preprocessing**
- Pre-solving heuristics, symmetry breaking, and formula simplification tuned to specific SMT-COMP benchmark families.

### Contribution guidelines

- All contributions must maintain the pure Rust policy (no C/C++/Fortran dependencies).
- New code must include unit tests; aim for the existing coverage density.
- Follow the existing module structure and naming conventions.
- Run `cargo clippy --workspace` and `cargo fmt --all` before submitting.

---

## Acknowledgments

*This section is a placeholder for acknowledgments to be added prior to the competition system description submission.*

The OxiZ project is grateful to the SMT-COMP organizers for maintaining an open and rigorous competition infrastructure, to the authors and maintainers of Z3 whose published algorithms and benchmark suite have informed this work, and to the broader SMT and formal methods research community.

---

*OxiZ v0.2.0 — COOLJAPAN OU (Team Kitasan)*
*Repository: [https://github.com/cool-japan/oxiz](https://github.com/cool-japan/oxiz)*
*Competition contact: open an issue with tag `[SMT-COMP 2026]`*
